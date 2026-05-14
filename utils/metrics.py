# ─────────────────────────────────────────────
# utils/metrics.py — evaluation helpers
# ─────────────────────────────────────────────
import numpy as np
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)


def clf_metrics(y_true, y_pred, y_prob) -> dict:
    """Return the core classification metrics as a dict.

    Recall is computed on the positive class (Default = 1), which is the
    metric the project optimizes for in the training notebooks.
    """
    return {
        "Exactitud": round(accuracy_score(y_true, y_pred),            4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_true, y_prob),             4),
        "Prec. Prom.": round(average_precision_score(y_true, y_prob), 4),
    }


@st.cache_data(show_spinner="Calculando predicciones…")
def compute_probs(_models: dict, X_test, _model_keys: tuple) -> dict:
    """
    Run predict_proba once per model on X_test.
    `_models` is unhashable (sklearn pipelines) so it's prefixed with `_`
    to skip Streamlit's hashing; `_model_keys` carries the cache key.
    """
    return {
        name: _models[name].predict_proba(X_test)[:, 1]
        for name in _model_keys
    }


def evaluate_all(probs: dict, y_test, threshold: float) -> dict:
    """
    Apply the decision threshold to pre-computed probabilities and return
    per-model y_prob / y_pred / metrics. Cheap: no model inference here.
    """
    results = {}
    for name, y_prob in probs.items():
        y_pred = (y_prob >= threshold).astype(int)
        results[name] = {
            "y_prob":  y_prob,
            "y_pred":  y_pred,
            "metrics": clf_metrics(y_test, y_pred, y_prob),
        }
    return results


def get_pipeline_input_features(model) -> list:
    """
    For a sklearn Pipeline ending in a classifier and starting with a
    ColumnTransformer (named 'preprocessing' or first step), return the
    flat list of original column names the pipeline actually consumes.
    Falls back to None if the structure can't be introspected.
    """
    if not hasattr(model, "named_steps"):
        return None
    pre = model.named_steps.get("preprocessing", model[0])
    if not hasattr(pre, "transformers_"):
        return None

    cols: list = []
    for name, _trans, sel in pre.transformers_:
        if name == "remainder":
            continue
        if isinstance(sel, str):
            cols.append(sel)
        else:
            cols.extend(list(sel))
    seen, unique = set(), []
    for c in cols:
        if isinstance(c, str) and c not in seen:
            unique.append(c)
            seen.add(c)
    return unique or None
