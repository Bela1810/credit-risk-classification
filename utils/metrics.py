# ─────────────────────────────────────────────
# utils/metrics.py — evaluation helpers
# ─────────────────────────────────────────────
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)


def clf_metrics(y_true, y_pred, y_prob) -> dict:
    """Return the four core classification metrics as a dict."""
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred),              4),
        "F1 Score":  round(f1_score(y_true, y_pred, zero_division=0),   4),
        "ROC-AUC":   round(roc_auc_score(y_true, y_prob),               4),
        "Avg Prec.": round(average_precision_score(y_true, y_prob),     4),
    }


def evaluate_all(models: dict, X_test, y_test, threshold: float) -> dict:
    """
    Run predict_proba + threshold on every model.
    Returns a dict keyed by model name with y_prob, y_pred, metrics.
    """
    results = {}
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        results[name] = {
            "y_prob":  y_prob,
            "y_pred":  y_pred,
            "metrics": clf_metrics(y_test, y_pred, y_prob),
        }
    return results
