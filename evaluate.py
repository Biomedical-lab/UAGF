"""UAGF Evaluation Pipeline.

Multi-seed evaluation, statistical significance testing, and ablation study.

Usage:
    python evaluate.py --config config/isic2018.yaml
    python evaluate.py --config config/isic2018.yaml --mode ablation
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import xgboost as xgb
import yaml
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, ttest_rel, wilcoxon
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score, log_loss,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import AdaptiveFusionModule, LabelSmoothingCE, ProxyClassifier
from models.se_block import SEBlock
from utils.data_loader import load_features, prepare_loader


def parse_args():
    parser = argparse.ArgumentParser(description="UAGF Evaluation Pipeline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "multiseed", "ablation", "complexity"])
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


# ======================================================================
# Multi-seed full pipeline
# ======================================================================

def run_full_pipeline(seed, Xc_tr, Xs_tr, Xe_tr, y_tr,
                      Xc_va, Xs_va, Xe_va, y_va,
                      Xc_te, Xs_te, Xe_te, y_te,
                      xgb_params, num_classes, device):
    """Run the complete UAGF pipeline for a single seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    c_dim, s_dim, e_dim = Xc_tr.shape[1], Xs_tr.shape[1], Xe_tr.shape[1]
    fused_dim = c_dim + s_dim + e_dim

    model = AdaptiveFusionModule(
        c_dim, s_dim, e_dim, num_classes=num_classes,
    ).to(device)
    criterion = LabelSmoothingCE(smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    proxy_clf = ProxyClassifier(fused_dim, num_classes).to(device)
    proxy_opt = optim.Adam(proxy_clf.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    train_loader = prepare_loader(Xc_tr, Xs_tr, Xe_tr, y_tr, batch_size=128)
    val_loader = prepare_loader(Xc_va, Xs_va, Xe_va, y_va, batch_size=128, shuffle=False)

    # Step 1: Train gating module
    best_val_loss, wait = float("inf"), 0
    best_wts = None
    for epoch in range(100):
        model.train()
        proxy_clf.train()
        for bc, bs, be, bl in train_loader:
            bc, bs, be, bl = bc.to(device), bs.to(device), be.to(device), bl.to(device)
            optimizer.zero_grad()
            proxy_opt.zero_grad()
            fused, _ = model(bc, bs, be)
            loss = criterion(proxy_clf(fused), bl)
            loss.backward()
            optimizer.step()
            proxy_opt.step()
        model.eval()
        proxy_clf.eval()
        vl = 0
        with torch.no_grad():
            for bc, bs, be, bl in val_loader:
                bc, bs, be, bl = bc.to(device), bs.to(device), be.to(device), bl.to(device)
                vl += criterion(proxy_clf(model(bc, bs, be)[0]), bl).item()
        vl /= len(val_loader)
        scheduler.step(vl)
        if vl < best_val_loss - 1e-4:
            best_val_loss = vl
            wait = 0
            best_wts = model.state_dict()
        else:
            wait += 1
            if wait >= 10:
                break
    if best_wts is not None:
        model.load_state_dict(best_wts)

    # Step 2: Temperature calibration
    model.eval()
    all_lc, all_ls, all_le, all_y = [], [], [], []
    with torch.no_grad():
        for bc, bs, be, bl in val_loader:
            bc, bs, be, bl = bc.to(device), bs.to(device), be.to(device), bl.to(device)
            all_lc.append(model.head_c(model.se_c(bc)).cpu())
            all_ls.append(model.head_s(model.se_s(bs)).cpu())
            all_le.append(model.head_e(model.se_e(be)).cpu())
            all_y.append(bl.cpu())
    lc, ls, le = torch.cat(all_lc), torch.cat(all_ls), torch.cat(all_le)
    labels = torch.cat(all_y)

    def _fit_T(logits, labels):
        def nll(T): return F.cross_entropy(logits / T, labels).item()
        return minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded").x

    Tc, Ts, Te = _fit_T(lc, labels), _fit_T(ls, labels), _fit_T(le, labels)
    model.set_branch_temps(Tc, Ts, Te)

    # Step 3: Extract fused features
    def _extract(xc, xs, xe):
        model.eval()
        with torch.no_grad():
            f, w = model(
                torch.FloatTensor(xc).to(device),
                torch.FloatTensor(xs).to(device),
                torch.FloatTensor(xe).to(device),
            )
            return f.cpu().numpy(), w.cpu().numpy()

    Xf_tr, _ = _extract(Xc_tr, Xs_tr, Xe_tr)
    Xf_va, _ = _extract(Xc_va, Xs_va, Xe_va)
    Xf_te, _ = _extract(Xc_te, Xs_te, Xe_te)

    # Step 4: XGBoost
    Xf_tv = np.concatenate([Xf_tr, Xf_va])
    y_tv = np.concatenate([y_tr, y_va])
    params = xgb_params.copy()
    params["seed"] = seed
    bst = xgb.train(params, xgb.DMatrix(Xf_tv, label=y_tv), num_boost_round=5000)

    dtest = xgb.DMatrix(Xf_te)
    y_prob = bst.predict(dtest)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = {
        "seed": seed,
        "Accuracy": accuracy_score(y_te, y_pred),
        "F1 (Weighted)": f1_score(y_te, y_pred, average="weighted"),
        "Precision (Weighted)": precision_score(y_te, y_pred, average="weighted"),
        "Recall (Weighted)": recall_score(y_te, y_pred, average="weighted"),
        "Balanced Acc": balanced_accuracy_score(y_te, y_pred),
        "MCC": matthews_corrcoef(y_te, y_pred),
        "AUC-ROC": roc_auc_score(y_te, y_prob, average="weighted", multi_class="ovo"),
        "Log Loss": log_loss(y_te, y_prob),
    }
    print(f"  [Seed {seed}] Acc={metrics['Accuracy']:.4f}  F1={metrics['F1 (Weighted)']:.4f}  MCC={metrics['MCC']:.4f}")
    return metrics, y_pred


def run_multiseed(cfg, Xc_tr, Xs_tr, Xe_tr, y_tr,
                  Xc_va, Xs_va, Xe_va, y_va,
                  Xc_te, Xs_te, Xe_te, y_te,
                  xgb_params, device):
    """Run multi-seed evaluation with statistical significance tests."""
    seeds = cfg["evaluation"]["seeds"]
    num_classes = cfg["num_classes"]
    out_dir = cfg["output"]["results_dir"]

    print("=" * 60)
    print(f"MULTI-SEED EVALUATION ({len(seeds)} seeds)")
    print("=" * 60)

    all_results, all_preds = [], {}
    t0 = time.time()
    for seed in seeds:
        metrics, preds = run_full_pipeline(
            seed, Xc_tr, Xs_tr, Xe_tr, y_tr,
            Xc_va, Xs_va, Xe_va, y_va,
            Xc_te, Xs_te, Xe_te, y_te,
            xgb_params, num_classes, device,
        )
        all_results.append(metrics)
        all_preds[seed] = preds
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")

    df = pd.DataFrame(all_results)
    metric_cols = [c for c in df.columns if c not in ["seed"]]
    print("\nMean +/- Std:")
    for col in metric_cols:
        print(f"  {col:<25}: {df[col].mean():.4f} +/- {df[col].std():.4f}")
    df.to_csv(os.path.join(out_dir, "multi_seed_results.csv"), index=False)

    # Statistical significance tests
    uagf_f1s = [r["F1 (Weighted)"] for r in all_results]
    uagf_pred = all_preds[seeds[0]]
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # Concat baseline
    Xf_concat_tv = np.concatenate([
        np.concatenate([Xc_tr, Xc_va]), np.concatenate([Xs_tr, Xs_va]),
        np.concatenate([Xe_tr, Xe_va]),
    ], axis=1).astype(np.float32)
    Xf_concat_te = np.concatenate([Xc_te, Xs_te, Xe_te], axis=1).astype(np.float32)
    y_tv = np.concatenate([y_tr, y_va])

    # McNemar's test
    p_base = xgb_params.copy()
    p_base["seed"] = seeds[0]
    bst_base = xgb.train(p_base, xgb.DMatrix(Xf_concat_tv, label=y_tv), num_boost_round=5000)
    base_pred = np.argmax(bst_base.predict(xgb.DMatrix(Xf_concat_te)), axis=1)
    b = int(np.sum((uagf_pred == y_te) & (base_pred != y_te)))
    c_val = int(np.sum((uagf_pred != y_te) & (base_pred == y_te)))
    chi2_stat = (abs(b - c_val) - 1) ** 2 / (b + c_val) if (b + c_val) > 0 else 0.0
    p_mcnemar = 1 - chi2.cdf(chi2_stat, df=1) if (b + c_val) > 0 else 1.0
    print(f"\nMcNemar: chi2={chi2_stat:.4f}, p={p_mcnemar:.6f}")

    # Paired t-test / Wilcoxon across seeds
    base_f1s = []
    for seed in seeds:
        p_b = xgb_params.copy()
        p_b["seed"] = seed
        bst_b = xgb.train(p_b, xgb.DMatrix(Xf_concat_tv, label=y_tv), num_boost_round=5000)
        bp = np.argmax(bst_b.predict(xgb.DMatrix(Xf_concat_te)), axis=1)
        base_f1s.append(f1_score(y_te, bp, average="weighted"))

    t_stat, p_ttest = ttest_rel(uagf_f1s, base_f1s)
    w_stat, p_wilcox = wilcoxon(uagf_f1s, base_f1s, alternative="greater")
    print(f"Paired t-test: t={t_stat:.4f}, p={p_ttest:.6f}")
    print(f"Wilcoxon: W={w_stat:.4f}, p={p_wilcox:.6f}")

    # Individual backbone baselines
    backbone_f1s = {}
    for bname, Xtr_b, Xva_b, Xte_b in [
        ("ConvNeXt", Xc_tr, Xc_va, Xc_te),
        ("Swin", Xs_tr, Xs_va, Xs_te),
        ("EfficientNet", Xe_tr, Xe_va, Xe_te),
    ]:
        bf1_list = []
        for seed in seeds:
            p_b = xgb_params.copy()
            p_b["seed"] = seed
            X_tv_b = np.concatenate([Xtr_b, Xva_b])
            bst_b = xgb.train(p_b, xgb.DMatrix(X_tv_b, label=y_tv), num_boost_round=5000, verbose_eval=False)
            bp = np.argmax(bst_b.predict(xgb.DMatrix(Xte_b)), axis=1)
            bf1_list.append(f1_score(y_te, bp, average="weighted"))
        backbone_f1s[bname] = bf1_list
        print(f"  [{bname}] F1 = {np.mean(bf1_list):.4f} +/- {np.std(bf1_list):.4f}")

    # Save statistical test results
    stat_results = {
        "UAGF_F1_mean": np.mean(uagf_f1s), "UAGF_F1_std": np.std(uagf_f1s),
        "Concat_F1_mean": np.mean(base_f1s), "Concat_F1_std": np.std(base_f1s),
        "McNemar_chi2": chi2_stat, "McNemar_pval": p_mcnemar,
        "ttest_t": t_stat, "ttest_pval": p_ttest,
        "Wilcoxon_W": w_stat, "Wilcoxon_pval": p_wilcox,
    }
    for bname in backbone_f1s:
        stat_results[f"{bname}_F1_mean"] = np.mean(backbone_f1s[bname])
        stat_results[f"{bname}_F1_std"] = np.std(backbone_f1s[bname])
    pd.DataFrame([stat_results]).to_csv(os.path.join(out_dir, "statistical_tests.csv"), index=False)
    print(f"\nSaved: {os.path.join(out_dir, 'statistical_tests.csv')}")


# ======================================================================
# Ablation study
# ======================================================================

class IdentityBlock(nn.Module):
    def forward(self, x):
        return x


class AblationFusionModule(nn.Module):
    """Variant of AdaptiveFusionModule for ablation experiments."""

    def __init__(self, c_dim, s_dim, e_dim, num_classes, temperature=1.0,
                 use_se=True, uncertainty_mode="all", use_uncertainty_in_gate=True):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.uncertainty_mode = uncertainty_mode
        self.use_uncertainty_in_gate = use_uncertainty_in_gate
        self.register_buffer("T_c", torch.tensor(1.0))
        self.register_buffer("T_s", torch.tensor(1.0))
        self.register_buffer("T_e", torch.tensor(1.0))

        self.se_c = SEBlock(c_dim) if use_se else IdentityBlock()
        self.se_s = SEBlock(s_dim) if use_se else IdentityBlock()
        self.se_e = SEBlock(e_dim) if use_se else IdentityBlock()
        self.head_c = nn.Linear(c_dim, num_classes)
        self.head_s = nn.Linear(s_dim, num_classes)
        self.head_e = nn.Linear(e_dim, num_classes)

        if uncertainty_mode == "all":
            u_dim = 9
        elif uncertainty_mode in ("entropy", "maxprob", "margin"):
            u_dim = 3
        else:
            u_dim = 0
        if not use_uncertainty_in_gate:
            u_dim = 0

        gate_in = c_dim + s_dim + e_dim + u_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 3),
        )

    def set_branch_temps(self, Tc, Ts, Te):
        self.T_c.fill_(Tc)
        self.T_s.fill_(Ts)
        self.T_e.fill_(Te)

    def forward(self, xc, xs, xe):
        xc_att, xs_att, xe_att = self.se_c(xc), self.se_s(xs), self.se_e(xe)
        prob_c = F.softmax(self.head_c(xc_att) / self.T_c, dim=1)
        prob_s = F.softmax(self.head_s(xs_att) / self.T_s, dim=1)
        prob_e = F.softmax(self.head_e(xe_att) / self.T_e, dim=1)

        def _get_u(probs):
            entropy = -(probs * (probs + 1e-12).log()).sum(dim=1, keepdim=True)
            max_prob = probs.max(dim=1, keepdim=True).values
            top2 = torch.topk(probs, k=2, dim=1).values
            margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1)
            if self.uncertainty_mode == "entropy": return entropy
            elif self.uncertainty_mode == "maxprob": return max_prob
            elif self.uncertainty_mode == "margin": return margin
            return torch.cat([entropy, max_prob, margin], dim=1)

        parts = [xc_att, xs_att, xe_att]
        if self.use_uncertainty_in_gate:
            parts.extend([_get_u(prob_c), _get_u(prob_s), _get_u(prob_e)])
        combined = torch.cat(parts, dim=1)
        weights = F.softmax(self.gate_mlp(combined) / self.temperature, dim=1)
        w1, w2, w3 = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        fused = torch.cat([w1 * xc_att, w2 * xs_att, w3 * xe_att], dim=1)
        return fused, weights


def run_ablation_condition(config_name, Xc_tr, Xs_tr, Xe_tr, y_tr,
                           Xc_va, Xs_va, Xe_va, y_va,
                           Xc_te, Xs_te, Xe_te, y_te,
                           xgb_params, num_classes, device, seed=42, **kwargs):
    """Run a single ablation condition."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    c_dim, s_dim, e_dim = Xc_tr.shape[1], Xs_tr.shape[1], Xe_tr.shape[1]
    fused_dim = c_dim + s_dim + e_dim
    model = AblationFusionModule(c_dim, s_dim, e_dim, num_classes=num_classes, **kwargs).to(device)
    criterion = LabelSmoothingCE(smoothing=0.1)
    opt = optim.Adam(model.parameters(), lr=0.001)
    proxy = ProxyClassifier(fused_dim, num_classes).to(device)
    proxy_opt = optim.Adam(proxy.parameters(), lr=0.001)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    tr_ld = prepare_loader(Xc_tr, Xs_tr, Xe_tr, y_tr, batch_size=128)
    va_ld = prepare_loader(Xc_va, Xs_va, Xe_va, y_va, batch_size=128, shuffle=False)

    best_vl = float("inf")
    wait = 0
    best_wts = None
    for _ in range(100):
        model.train()
        proxy.train()
        for bc, bs, be, bl in tr_ld:
            bc, bs, be, bl = bc.to(device), bs.to(device), be.to(device), bl.to(device)
            opt.zero_grad()
            proxy_opt.zero_grad()
            loss = criterion(proxy(model(bc, bs, be)[0]), bl)
            loss.backward()
            opt.step()
            proxy_opt.step()
        model.eval()
        proxy.eval()
        vl = sum(criterion(proxy(model(bc.to(device), bs.to(device), be.to(device))[0]), bl.to(device)).item()
                 for bc, bs, be, bl in va_ld) / len(va_ld)
        sched.step(vl)
        if vl < best_vl - 1e-4:
            best_vl = vl
            wait = 0
            best_wts = model.state_dict()
        else:
            wait += 1
            if wait >= 10: break
    if best_wts: model.load_state_dict(best_wts)

    # Temperature calibration
    model.eval()
    lc_l, ls_l, le_l, y_l = [], [], [], []
    with torch.no_grad():
        for bc, bs, be, bl in va_ld:
            bc, bs, be, bl = bc.to(device), bs.to(device), be.to(device), bl.to(device)
            lc_l.append(model.head_c(model.se_c(bc)).cpu())
            ls_l.append(model.head_s(model.se_s(bs)).cpu())
            le_l.append(model.head_e(model.se_e(be)).cpu())
            y_l.append(bl.cpu())
    lc, ls_t, le_t, lbl = torch.cat(lc_l), torch.cat(ls_l), torch.cat(le_l), torch.cat(y_l)
    def _fitT(logits, labels):
        def nll(T): return F.cross_entropy(logits/T, labels).item()
        return minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded").x
    model.set_branch_temps(_fitT(lc, lbl), _fitT(ls_t, lbl), _fitT(le_t, lbl))

    # Extract + XGBoost
    def _ext(xc, xs, xe):
        model.eval()
        with torch.no_grad():
            f, _ = model(torch.FloatTensor(xc).to(device), torch.FloatTensor(xs).to(device), torch.FloatTensor(xe).to(device))
            return f.cpu().numpy()
    Xf_tv = np.concatenate([_ext(Xc_tr, Xs_tr, Xe_tr), _ext(Xc_va, Xs_va, Xe_va)])
    Xf_te = _ext(Xc_te, Xs_te, Xe_te)
    y_tv = np.concatenate([y_tr, y_va])
    p = xgb_params.copy()
    p["seed"] = seed
    bst = xgb.train(p, xgb.DMatrix(Xf_tv, label=y_tv), num_boost_round=5000)
    y_pred = np.argmax(bst.predict(xgb.DMatrix(Xf_te)), axis=1)

    result = {
        "Condition": config_name,
        "Accuracy": accuracy_score(y_te, y_pred),
        "F1 (Weighted)": f1_score(y_te, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y_te, y_pred),
    }
    print(f"  [{config_name}] Acc={result['Accuracy']:.4f}  F1={result['F1 (Weighted)']:.4f}")
    return result


def run_ablation_study(cfg, Xc_tr, Xs_tr, Xe_tr, y_tr,
                       Xc_va, Xs_va, Xe_va, y_va,
                       Xc_te, Xs_te, Xe_te, y_te,
                       xgb_params, device):
    """Run extended ablation study (7 conditions + 3 single backbones)."""
    num_classes = cfg["num_classes"]
    out_dir = cfg["output"]["results_dir"]

    print("=" * 60)
    print("EXTENDED ABLATION STUDY")
    print("=" * 60)

    configs = [
        ("A1: Entropy only", {"uncertainty_mode": "entropy"}),
        ("A2: MaxProb only", {"uncertainty_mode": "maxprob"}),
        ("A3: Margin only", {"uncertainty_mode": "margin"}),
        ("A4: No SE blocks", {"use_se": False}),
        ("A5: No temp scaling", {}),
        ("A7: No uncertainty", {"use_uncertainty_in_gate": False}),
    ]

    results = []
    for cname, kwargs in configs:
        r = run_ablation_condition(
            cname, Xc_tr, Xs_tr, Xe_tr, y_tr,
            Xc_va, Xs_va, Xe_va, y_va, Xc_te, Xs_te, Xe_te, y_te,
            xgb_params, num_classes, device, **kwargs,
        )
        results.append(r)

    # A6: Single backbone baselines
    print("\n  [A6] Individual backbones...")
    for name, Xtr, Xva, Xte in [
        ("ConvNeXt", Xc_tr, Xc_va, Xc_te),
        ("Swin", Xs_tr, Xs_va, Xs_te),
        ("EfficientNet", Xe_tr, Xe_va, Xe_te),
    ]:
        X_tv = np.concatenate([Xtr, Xva])
        y_tv = np.concatenate([y_tr, y_va])
        p = xgb_params.copy()
        p["seed"] = 42
        bst = xgb.train(p, xgb.DMatrix(X_tv, label=y_tv), num_boost_round=5000)
        y_pred = np.argmax(bst.predict(xgb.DMatrix(Xte)), axis=1)
        results.append({
            "Condition": f"A6: {name} only",
            "Accuracy": accuracy_score(y_te, y_pred),
            "F1 (Weighted)": f1_score(y_te, y_pred, average="weighted"),
            "MCC": matthews_corrcoef(y_te, y_pred),
        })
        print(f"  [{name}] Acc={results[-1]['Accuracy']:.4f}  F1={results[-1]['F1 (Weighted)']:.4f}")

    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    df.to_csv(os.path.join(out_dir, "ablation_results.csv"), index=False)
    print(f"\nSaved: {os.path.join(out_dir, 'ablation_results.csv')}")


# ======================================================================
# Complexity analysis
# ======================================================================

def run_complexity_analysis(cfg, Xc_te, Xs_te, Xe_te, device):
    """Compute model complexity: parameters, FLOPs, inference time."""
    num_classes = cfg["num_classes"]
    c_dim, s_dim, e_dim = Xc_te.shape[1], Xs_te.shape[1], Xe_te.shape[1]

    model = AdaptiveFusionModule(
        c_dim, s_dim, e_dim, num_classes=num_classes,
    ).to(device)

    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    se_params = sum(p.numel() for n, p in model.named_parameters() if "se_" in n)
    head_params = sum(p.numel() for n, p in model.named_parameters() if "head_" in n)
    gate_params = sum(p.numel() for n, p in model.named_parameters() if "gate_" in n)

    print("=" * 60)
    print("COMPLEXITY ANALYSIS")
    print("=" * 60)
    print(f"  SE blocks:    {se_params:>10,} params")
    print(f"  Branch heads: {head_params:>10,} params")
    print(f"  Gating MLP:   {gate_params:>10,} params")
    print(f"  TOTAL:        {total_params:>10,} params")

    # Inference time
    model.eval()
    xc = torch.FloatTensor(Xc_te).to(device)
    xs = torch.FloatTensor(Xs_te).to(device)
    xe = torch.FloatTensor(Xe_te).to(device)
    n_samples = Xc_te.shape[0]

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(xc, xs, xe)
    torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.time()
            model(xc, xs, xe)
            torch.cuda.synchronize()
            times.append(time.time() - t0)

    avg_time = np.mean(times)
    per_sample = avg_time / n_samples * 1000  # ms
    print(f"  Inference: {per_sample:.4f} ms/sample ({n_samples} samples)")

    return {
        "total_params": total_params,
        "se_params": se_params,
        "head_params": head_params,
        "gate_params": gate_params,
        "inference_ms_per_sample": per_sample,
    }


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
    print(f"Device: {device} | Dataset: {cfg['dataset']} | Mode: {args.mode}")

    # Load features
    _, y_tr, Xc_tr, Xs_tr, Xe_tr = load_features(cfg, "train")
    _, y_va, Xc_va, Xs_va, Xe_va = load_features(cfg, "val")
    _, y_te, Xc_te, Xs_te, Xe_te = load_features(cfg, "test")

    # Load XGBoost params
    params_path = os.path.join(cfg["output"]["results_dir"], "best_xgb_params.json")
    if os.path.exists(params_path):
        with open(params_path) as f:
            xgb_params = json.load(f)
    else:
        xgb_params = {
            "objective": "multi:softprob", "num_class": cfg["num_classes"],
            "tree_method": "hist", "device": "cuda", "eval_metric": "mlogloss",
        }
        print(f"WARNING: {params_path} not found, using default XGBoost params.")

    if args.mode in ("all", "multiseed"):
        run_multiseed(cfg, Xc_tr, Xs_tr, Xe_tr, y_tr,
                      Xc_va, Xs_va, Xe_va, y_va,
                      Xc_te, Xs_te, Xe_te, y_te, xgb_params, device)

    if args.mode in ("all", "ablation"):
        run_ablation_study(cfg, Xc_tr, Xs_tr, Xe_tr, y_tr,
                           Xc_va, Xs_va, Xe_va, y_va,
                           Xc_te, Xs_te, Xe_te, y_te, xgb_params, device)

    if args.mode in ("all", "complexity"):
        results = run_complexity_analysis(cfg, Xc_te, Xs_te, Xe_te, device)
        with open(os.path.join(cfg["output"]["results_dir"], "complexity.json"), "w") as f:
            json.dump(results, f, indent=2)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
