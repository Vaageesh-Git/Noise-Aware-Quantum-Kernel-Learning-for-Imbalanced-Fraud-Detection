"""
Evaluation Metrics Module for Fraud Detection.
Computes comprehensive fraud-class-focused evaluation metrics:
  - Precision, Recall, F1-Score for positive (fraud) class
  - Precision-Recall AUC (PR-AUC / Average Precision) - Main metric for imbalanced data
  - ROC-AUC - Secondary ranking metric
  - Confusion matrix (TN, FP, FN, TP) and Accuracy
"""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    """
    Calculate classification metrics focused on fraud detection.
    When y_prob is not provided, PR-AUC and ROC-AUC are approximated from y_pred.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    # Confusion matrix extraction
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Continuous probabilities are required for ranking metrics
    if y_prob is not None:
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = float(y_true.mean())
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = 0.50
    else:
        pr_auc = float(average_precision_score(y_true, y_pred))
        roc_auc = float(roc_auc_score(y_true, y_pred))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
