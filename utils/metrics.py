"""Evaluation metrics for skin lesion classification."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    log_loss,
    confusion_matrix,
)


def compute_overall_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, num_classes: int
) -> dict:
    """Compute overall evaluation metrics.

    Returns a dictionary with Accuracy, Precision, Recall, F1-Score,
    Specificity, Balanced Accuracy, MCC, AUC-ROC, and Log Loss.
    """
    cm = confusion_matrix(y_true, y_pred)
    specs = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (Weighted)": precision_score(y_true, y_pred, average="weighted"),
        "Recall (Weighted)": recall_score(y_true, y_pred, average="weighted"),
        "F1-Score": f1_score(y_true, y_pred, average="weighted"),
        "Specificity (Macro)": np.mean(specs),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUC-ROC (OVO)": roc_auc_score(
            y_true, y_prob, average="weighted", multi_class="ovo"
        ),
        "Log Loss": log_loss(y_true, y_prob),
    }
    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list
) -> pd.DataFrame:
    """Compute per-class Accuracy, Precision, Recall, Specificity, F1-Score.

    Returns a DataFrame with one row per class.
    """
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    total = cm.sum()
    rows = []

    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - (tp + fp + fn)

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        specificity = tn / (tn + fp + 1e-12)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        support = cm[i, :].sum()

        rows.append({
            "Class": cls,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Specificity": specificity,
            "F1-Score": f1,
            "Support": int(support),
        })

    return pd.DataFrame(rows)
