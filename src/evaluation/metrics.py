import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==================================================
# Compute all evaluation metrics
# ==================================================
def compute_metrics(labels, predictions, probabilities=None):
    """
    Compute classification metrics for GNN evaluation
    """

    metrics = {}

    metrics["accuracy"] = accuracy_score(labels, predictions)
    metrics["precision"] = precision_score(labels, predictions, zero_division=0)
    metrics["recall"] = recall_score(labels, predictions, zero_division=0)
    metrics["f1"] = f1_score(labels, predictions, zero_division=0)

    # --------------------------
    # Confusion matrix (safe)
    # --------------------------
    cm = confusion_matrix(labels, predictions)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    metrics["tn"] = tn
    metrics["fp"] = fp
    metrics["fn"] = fn
    metrics["tp"] = tp

    # --------------------------
    # AUC (safe check)
    # --------------------------
    if probabilities is not None:
        try:
            metrics["auc"] = roc_auc_score(labels, probabilities)
        except ValueError:
            metrics["auc"] = np.nan

    return metrics


# ==================================================
# Pretty print results
# ==================================================
def print_metrics(metrics):
    """Pretty print evaluation results"""

    print("\n" + "=" * 50)
    print("📊 Evaluation Metrics")
    print("=" * 50)

    auc = metrics.get("auc", np.nan)

    if auc is not None and not np.isnan(auc):
        print(f"ROC-AUC:     {auc:.4f}")
    else:
        print("ROC-AUC:     N/A")

    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1-Score:    {metrics['f1']:.4f}")

    print("-" * 30)
    print(f"True Pos:    {metrics['tp']}")
    print(f"True Neg:    {metrics['tn']}")
    print(f"False Pos:   {metrics['fp']}")
    print(f"False Neg:   {metrics['fn']}")

    print("=" * 50)
