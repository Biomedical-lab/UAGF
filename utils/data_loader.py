"""Data loading and preprocessing utilities for multi-backbone features."""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def _load_csv(feature_dir: str, filename: str) -> pd.DataFrame:
    """Load a CSV feature file."""
    path = os.path.join(feature_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found: {path}")
    return pd.read_csv(path)


def load_features(cfg: dict, split: str) -> tuple:
    """Load and align features from three backbones for a given split.

    Args:
        cfg: Configuration dictionary (from YAML).
        split: One of 'train', 'val', 'test'.

    Returns:
        Tuple (ids, y, Xc, Xs, Xe) where:
          - ids: Sample identifiers (N,).
          - y: Labels (N,) int64.
          - Xc: ConvNeXt features (N, c_dim) float32.
          - Xs: Swin features (N, s_dim) float32.
          - Xe: EfficientNet features (N, e_dim) float32.
    """
    fdir = cfg["feature_dir"]
    ffiles = cfg["feature_files"]
    id_col = cfg["id_col"]
    label_col = cfg["label_col"]
    prefixes = cfg["feature_prefixes"]

    conv = _load_csv(fdir, ffiles["convnext"][split])
    swin = _load_csv(fdir, ffiles["swin"][split])
    eff = _load_csv(fdir, ffiles["efficientnet"][split])

    # Sort by ID for alignment
    for df in [conv, swin, eff]:
        if id_col in df.columns:
            df.sort_values(id_col, inplace=True)
            df.reset_index(drop=True, inplace=True)

    # Validate ID alignment
    if id_col in conv.columns and id_col in swin.columns:
        if not np.array_equal(conv[id_col].values, swin[id_col].values):
            raise RuntimeError(f"ID mismatch: convnext vs swin ({split})")
    if id_col in conv.columns and id_col in eff.columns:
        if not np.array_equal(conv[id_col].values, eff[id_col].values):
            raise RuntimeError(f"ID mismatch: convnext vs efficientnet ({split})")

    # Validate label alignment
    if label_col not in conv.columns:
        raise RuntimeError(f"Missing '{label_col}' column in ConvNeXt {split}")
    for name, df in [("swin", swin), ("efficientnet", eff)]:
        if label_col in df.columns:
            if not np.array_equal(conv[label_col].values, df[label_col].values):
                raise RuntimeError(f"Label mismatch: convnext vs {name} ({split})")

    y = conv[label_col].to_numpy().astype(np.int64)
    ids = conv[id_col].to_numpy() if id_col in conv.columns else np.arange(len(conv))

    # Extract feature columns
    conv_cols = [c for c in conv.columns if c.startswith(prefixes["convnext"])]
    swin_cols = [c for c in swin.columns if c.startswith(prefixes["swin"])]
    eff_cols = [c for c in eff.columns if c.startswith(prefixes["efficientnet"])]

    if len(conv_cols) == 0 or len(swin_cols) == 0 or len(eff_cols) == 0:
        raise RuntimeError(f"Missing feature columns in split={split}")

    Xc = conv[conv_cols].to_numpy(dtype=np.float32)
    Xs = swin[swin_cols].to_numpy(dtype=np.float32)
    Xe = eff[eff_cols].to_numpy(dtype=np.float32)

    return ids, y, Xc, Xs, Xe


def fuse_scaled(Xc: np.ndarray, Xs: np.ndarray, Xe: np.ndarray) -> np.ndarray:
    """Concatenate backbone features into a single fused vector."""
    return np.concatenate([Xc, Xs, Xe], axis=1).astype(np.float32)


def compute_sample_weight(
    y: np.ndarray, num_classes: int, mode: str = "inv"
) -> np.ndarray:
    """Compute inverse-frequency sample weights for class imbalance.

    Args:
        y: Label array.
        num_classes: Total number of classes.
        mode: 'inv' for inverse frequency, 'sqrt' for inverse sqrt.
    """
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    if mode == "sqrt":
        w_cls = 1.0 / np.sqrt(counts + 1e-9)
    else:
        w_cls = 1.0 / (counts + 1e-9)
    w_cls = w_cls * (num_classes / w_cls.sum())
    return w_cls[y].astype(np.float32)


def prepare_loader(
    xc: np.ndarray,
    xs: np.ndarray,
    xe: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader from numpy arrays."""
    dataset = TensorDataset(
        torch.FloatTensor(xc),
        torch.FloatTensor(xs),
        torch.FloatTensor(xe),
        torch.LongTensor(y),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=2,
    )
