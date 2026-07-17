"""UAGF Visualization Pipeline.

Generate SHAP, t-SNE, confusion matrix, ROC, and alpha distribution plots.

Usage:
    python visualize.py --config config/isic2018.yaml --plot all
    python visualize.py --config config/isic2018.yaml --plot confusion_matrix
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import yaml
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize

from utils.data_loader import load_features


def parse_args():
    parser = argparse.ArgumentParser(description="UAGF Visualization")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--plot", type=str, default="all",
                        choices=["all", "confusion_matrix", "roc", "tsne",
                                 "shap", "alpha", "optuna"])
    parser.add_argument("--model", type=str, default=None,
                        help="Path to saved XGBoost model")
    return parser.parse_args()


def plot_confusion_matrix(y_true, y_pred, class_names, out_dir):
    """Plot and save normalized confusion matrix."""
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(np.float64)
    row_sum = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sum, where=(row_sum != 0))

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
        spine.set_alpha(0.08)
    plt.tight_layout()

    path = os.path.join(out_dir, "confusion_matrix.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_roc_curves(y_true, y_prob, class_names, out_dir):
    """Plot per-class ROC curves."""
    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for i, (cls, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{cls} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    path = os.path.join(out_dir, "roc_curves.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_tsne(X_fused, X_individual, y, class_names, out_dir):
    """Plot t-SNE of fused vs individual backbone features."""
    datasets = {"Fused": X_fused}
    datasets.update(X_individual)

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), dpi=300)
    if len(datasets) == 1:
        axes = [axes]

    for ax, (name, X) in zip(axes, datasets.items()):
        n_samples = min(2000, X.shape[0])
        idx = np.random.choice(X.shape[0], n_samples, replace=False)
        X_sub, y_sub = X[idx], y[idx]

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        emb = tsne.fit_transform(X_sub)

        for c, cls in enumerate(class_names):
            mask = y_sub == c
            ax.scatter(emb[mask, 0], emb[mask, 1], s=5, alpha=0.6, label=cls)

        ax.set_title(name, fontsize=11)
        ax.legend(fontsize=6, markerscale=2)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    path = os.path.join(out_dir, "tsne_comparison.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_alpha_distribution(W, backbone_names, out_dir):
    """Plot gating weight distributions."""
    symbols = [r"$\alpha$", r"$\beta$", r"$\gamma$"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for i, (ax, name, sym, color) in enumerate(zip(axes, backbone_names, symbols, colors)):
        vals = W[:, i]
        ax.hist(vals, bins=40, color=color, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_xlabel(f"{sym} ({name} weight)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"({chr(97+i)}) {name}", fontsize=11)
        ax.grid(True, alpha=0.18)
        ax.axvline(np.mean(vals), color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean = {np.mean(vals):.3f}")
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "alpha_distribution.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_shap(best_model, X_fused_te, feature_names, class_names, out_dir):
    """Plot SHAP beeswarm and bar plots."""
    import shap

    X_sample = X_fused_te[:10000] if X_fused_te.shape[0] > 10000 else X_fused_te
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_sample)

    # Bar plot
    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.title("")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_bar.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

    # Beeswarm plot
    plt.figure(figsize=(9, 7))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[0], X_sample, feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_beeswarm.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

    # Per-class beeswarm grid
    C = len(class_names)
    ncols = min(4, C)
    nrows = (C + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for c, cname in enumerate(class_names):
        ax = axes_flat[c]
        if isinstance(shap_values, list):
            sv_c = np.asarray(shap_values[c])
        elif shap_values.ndim == 3:
            sv_c = shap_values[:, :, c]
        else:
            sv_c = shap_values
        expl = shap.Explanation(values=sv_c[:, :len(feature_names)],
                                data=X_sample[:, :len(feature_names)],
                                feature_names=feature_names)
        plt.sca(ax)
        shap.plots.beeswarm(expl, max_display=15, show=False, ax=ax, plot_size=None)
        ax.set_title(cname, fontsize=9)
        ax.set_xlabel("")

    for i in range(C, len(axes_flat)):
        axes_flat[i].axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "shap_per_class_grid.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved SHAP plots to: {out_dir}")


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["output"]["figures_dir"]
    os.makedirs(out_dir, exist_ok=True)
    class_names = cfg["class_names"]
    print(f"Dataset: {cfg['dataset']} | Output: {out_dir}")

    # Load test features
    _, y_te, Xc_te, Xs_te, Xe_te = load_features(cfg, "test")

    # Load XGBoost model
    model_path = args.model or os.path.join(cfg["output"]["results_dir"], "xgboost_model.json")
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}. Run train.py first.")
        return
    best_model = xgb.Booster()
    best_model.load_model(model_path)

    # Load fused features (reconstruct from model or load from saved)
    import torch
    from models import AdaptiveFusionModule
    fusion_path = os.path.join(cfg["output"]["results_dir"], "fusion_module.pt")
    c_dim, s_dim, e_dim = Xc_te.shape[1], Xs_te.shape[1], Xe_te.shape[1]

    if os.path.exists(fusion_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fusion_model = AdaptiveFusionModule(
            c_dim, s_dim, e_dim, num_classes=cfg["num_classes"],
        ).to(device)
        fusion_model.load_state_dict(torch.load(fusion_path, map_location=device))
        fusion_model.eval()
        with torch.no_grad():
            xc = torch.FloatTensor(Xc_te).to(device)
            xs = torch.FloatTensor(Xs_te).to(device)
            xe = torch.FloatTensor(Xe_te).to(device)
            X_fused_te, W_te = fusion_model(xc, xs, xe)
            X_fused_te = X_fused_te.cpu().numpy()
            W_te = W_te.cpu().numpy()
    else:
        print(f"WARNING: {fusion_path} not found. Using raw concatenation.")
        X_fused_te = np.concatenate([Xc_te, Xs_te, Xe_te], axis=1)
        W_te = None

    # Predictions
    dtest = xgb.DMatrix(X_fused_te)
    y_prob = best_model.predict(dtest)
    y_pred = np.argmax(y_prob, axis=1)

    # Feature names
    feature_names = (
        [f"conv_{i}" for i in range(c_dim)] +
        [f"swin_{i}" for i in range(s_dim)] +
        [f"eff_{i}" for i in range(e_dim)]
    )

    # Generate plots
    if args.plot in ("all", "confusion_matrix"):
        plot_confusion_matrix(y_te, y_pred, class_names, out_dir)

    if args.plot in ("all", "roc"):
        plot_roc_curves(y_te, y_prob, class_names, out_dir)

    if args.plot in ("all", "tsne"):
        plot_tsne(
            X_fused_te,
            {"ConvNeXt": Xc_te, "Swin": Xs_te, "EfficientNet": Xe_te},
            y_te, class_names, out_dir,
        )

    if args.plot in ("all", "alpha") and W_te is not None:
        plot_alpha_distribution(
            W_te, ["ConvNeXt", "Swin", "EfficientNet-B3"], out_dir,
        )

    if args.plot in ("all", "shap"):
        plot_shap(best_model, X_fused_te, feature_names, class_names, out_dir)

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()
