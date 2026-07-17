"""UAGF Training Pipeline.

Trains the Uncertainty-Guided Adaptive Gated Fusion module and a meta-
classifier (XGBoost with Optuna hyperparameter optimization) on extracted
backbone features.

Usage:
    python train.py --config config/isic2018.yaml
"""

import argparse
import json
import os
import time

import numpy as np
import optuna
import torch
import torch.optim as optim
import xgboost as xgb
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import AdaptiveFusionModule, LabelSmoothingCE, ProxyClassifier
from models.temperature import fit_branch_temperatures
from utils.data_loader import load_features, prepare_loader


def parse_args():
    parser = argparse.ArgumentParser(description="UAGF Training Pipeline")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    return parser.parse_args()


def train_fusion_module(cfg, Xc_tr, Xs_tr, Xe_tr, y_tr, Xc_va, Xs_va, Xe_va, y_va, device):
    """Train the Adaptive Fusion Module (Step 1: uncalibrated, Step 2: calibrate)."""
    tcfg = cfg["training"]
    gcfg = cfg["gating"]
    num_classes = cfg["num_classes"]

    c_dim, s_dim, e_dim = Xc_tr.shape[1], Xs_tr.shape[1], Xe_tr.shape[1]
    gating_temp = cfg["temperature"]["gating"]

    # Initialize model
    model = AdaptiveFusionModule(
        c_dim, s_dim, e_dim,
        num_classes=num_classes,
        temperature=gating_temp,
        branch_temps=(1.0, 1.0, 1.0),
        se_reduction=gcfg["se_reduction"],
    ).to(device)

    fused_dim = c_dim + s_dim + e_dim
    proxy_clf = ProxyClassifier(fused_dim, num_classes).to(device)
    criterion = LabelSmoothingCE(smoothing=tcfg["label_smoothing"])

    optimizer = optim.Adam(model.parameters(), lr=tcfg["learning_rate"])
    proxy_optimizer = optim.Adam(proxy_clf.parameters(), lr=tcfg["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    train_loader = prepare_loader(
        Xc_tr, Xs_tr, Xe_tr, y_tr,
        batch_size=tcfg["batch_size"], shuffle=True,
    )
    val_loader = prepare_loader(
        Xc_va, Xs_va, Xe_va, y_va,
        batch_size=tcfg["batch_size"], shuffle=False,
    )

    # Step 1: Training with T_k = 1.0
    print("=" * 60)
    print("STEP 1: Training Adaptive Fusion Module (T_k = 1.0)...")
    print("=" * 60)

    best_val_loss = float("inf")
    best_model_wts = None
    wait = 0

    for epoch in range(tcfg["num_epochs"]):
        model.train()
        proxy_clf.train()
        total_loss = 0

        for bc, bs, be, bl in train_loader:
            bc, bs, be, bl = bc.to(device), bs.to(device), be.to(device), bl.to(device)
            optimizer.zero_grad()
            proxy_optimizer.zero_grad()
            fused, _ = model(bc, bs, be)
            output = proxy_clf(fused)
            loss = criterion(output, bl)
            loss.backward()
            optimizer.step()
            proxy_optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        proxy_clf.eval()
        val_loss = 0
        with torch.no_grad():
            for bc, bs, be, bl in val_loader:
                bc, bs, be, bl = bc.to(device), bs.to(device), be.to(device), bl.to(device)
                fused, _ = model(bc, bs, be)
                output = proxy_clf(fused)
                val_loss += criterion(output, bl).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss - tcfg["min_delta"]:
            best_val_loss = val_loss
            wait = 0
            best_model_wts = model.state_dict()
        else:
            wait += 1
            if wait >= tcfg["patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # Step 2: Post-hoc temperature scaling
    print("\n" + "=" * 60)
    print("STEP 2: Fitting per-branch temperature scaling...")
    print("=" * 60)

    T_c, T_s, T_e = fit_branch_temperatures(
        model, val_loader, device,
        bounds=tuple(cfg["temperature"]["bounds"]),
    )
    print(f"  T_convnext     = {T_c:.4f}")
    print(f"  T_swin         = {T_s:.4f}")
    print(f"  T_efficientnet = {T_e:.4f}")

    # Step 3: Set calibrated temperatures
    model.set_branch_temps(T_c, T_s, T_e)

    return model, (T_c, T_s, T_e)


def extract_fused_features(model, Xc, Xs, Xe, device):
    """Extract fused features and gating weights from trained model."""
    model.eval()
    with torch.no_grad():
        c = torch.FloatTensor(Xc).to(device)
        s = torch.FloatTensor(Xs).to(device)
        e = torch.FloatTensor(Xe).to(device)
        fused, weights = model(c, s, e)
        return fused.cpu().numpy(), weights.cpu().numpy()


def train_xgboost(cfg, X_fused_tv, y_tv, X_fused_te, y_te, device_str="cuda"):
    """Train XGBoost meta-classifier with Optuna hyperparameter optimization."""
    xcfg = cfg["xgboost"]
    num_classes = cfg["num_classes"]
    nthread = os.cpu_count() or 8

    def objective(trial):
        scores = []
        for seed in xcfg["optuna_seeds"]:
            params = {
                "objective": "multi:softprob",
                "num_class": num_classes,
                "eval_metric": "mlogloss",
                "tree_method": "hist",
                "device": device_str,
                "seed": seed,
                "nthread": nthread,
                "eta": trial.suggest_float("eta", 0.003, 0.08, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 16),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "lambda": trial.suggest_float("lambda", 1e-3, 20.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-3, 20.0, log=True),
                "gamma": trial.suggest_float("gamma", 0, 5.0),
                "max_leaves": trial.suggest_int("max_leaves", 0, 128),
                "grow_policy": trial.suggest_categorical(
                    "grow_policy", ["depthwise", "lossguide"]
                ),
            }
            cv_scores = []
            skf = StratifiedKFold(n_splits=xcfg["n_folds"], shuffle=True, random_state=seed)
            for tr_idx, va_idx in skf.split(X_fused_tv, y_tv):
                dtrain = xgb.DMatrix(X_fused_tv[tr_idx], label=y_tv[tr_idx])
                dval = xgb.DMatrix(X_fused_tv[va_idx], label=y_tv[va_idx])
                bst = xgb.train(
                    params, dtrain,
                    num_boost_round=xcfg["num_boost_round"],
                    evals=[(dval, "val")],
                    early_stopping_rounds=xcfg["early_stopping_rounds"],
                    verbose_eval=False,
                )
                preds = np.argmax(bst.predict(dval), axis=1)
                cv_scores.append(f1_score(y_tv[va_idx], preds, average="weighted"))
            scores.append(np.mean(cv_scores))
        return np.mean(scores)

    print("\n" + "=" * 60)
    print(f"STEP 4: Optuna XGBoost optimization ({xcfg['n_trials']} trials)...")
    print("=" * 60)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=xcfg["n_trials"])
    print(f"Best CV F1: {study.best_value:.4f}")

    # Train final model
    best_params = study.best_params.copy()
    best_params.update({
        "objective": "multi:softprob",
        "num_class": num_classes,
        "tree_method": "hist",
        "device": device_str,
        "eval_metric": "mlogloss",
    })

    dtrain_full = xgb.DMatrix(X_fused_tv, label=y_tv)
    best_model = xgb.train(
        best_params, dtrain_full,
        num_boost_round=xcfg["num_boost_round"],
    )
    print("Final XGBoost model trained.")

    return best_model, study, best_params


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Dataset: {cfg['dataset']}")

    # Ensure output dirs exist
    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["figures_dir"], exist_ok=True)

    # Load features
    print("\nLoading features...")
    _, y_tr, Xc_tr, Xs_tr, Xe_tr = load_features(cfg, "train")
    _, y_va, Xc_va, Xs_va, Xe_va = load_features(cfg, "val")
    _, y_te, Xc_te, Xs_te, Xe_te = load_features(cfg, "test")
    print(f"Train: {Xc_tr.shape[0]} | Val: {Xc_va.shape[0]} | Test: {Xc_te.shape[0]}")

    # Train fusion module
    model, temps = train_fusion_module(
        cfg, Xc_tr, Xs_tr, Xe_tr, y_tr, Xc_va, Xs_va, Xe_va, y_va, device
    )

    # Extract fused features
    print("\nExtracting fused features with calibrated temperatures...")
    X_fused_tr, W_tr = extract_fused_features(model, Xc_tr, Xs_tr, Xe_tr, device)
    X_fused_va, W_va = extract_fused_features(model, Xc_va, Xs_va, Xe_va, device)
    X_fused_te, W_te = extract_fused_features(model, Xc_te, Xs_te, Xe_te, device)
    print(f"Fused feature shape: {X_fused_tr.shape}")

    # Merge train + val for XGBoost
    X_fused_tv = np.concatenate([X_fused_tr, X_fused_va], axis=0)
    y_tv = np.concatenate([y_tr, y_va], axis=0)

    # Train XGBoost
    best_model, study, best_params = train_xgboost(
        cfg, X_fused_tv, y_tv, X_fused_te, y_te, device_str
    )

    # Evaluate
    from utils.metrics import compute_overall_metrics, compute_per_class_metrics

    dtest = xgb.DMatrix(X_fused_te)
    y_prob = best_model.predict(dtest)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = compute_overall_metrics(y_te, y_pred, y_prob, cfg["num_classes"])
    print("\n" + "=" * 40)
    print("FINAL EVALUATION METRICS")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"  {k:<25}: {v:.6f}")

    per_class = compute_per_class_metrics(y_te, y_pred, cfg["class_names"])
    print(f"\nPer-class metrics:")
    print(per_class.to_string(index=False))

    # Save results
    out_dir = cfg["output"]["results_dir"]
    per_class.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)

    temp_info = {
        "T_convnext": temps[0], "T_swin": temps[1],
        "T_efficientnet": temps[2], "T_gating": cfg["temperature"]["gating"],
    }
    with open(os.path.join(out_dir, "calibrated_temperatures.json"), "w") as f:
        json.dump(temp_info, f, indent=2)

    with open(os.path.join(out_dir, "best_xgb_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    # Save fusion module
    torch.save(model.state_dict(), os.path.join(out_dir, "fusion_module.pt"))
    best_model.save_model(os.path.join(out_dir, "xgboost_model.json"))

    print(f"\nAll results saved to: {out_dir}")
    print("Training complete.")


if __name__ == "__main__":
    main()
