"""Scores, decision thresholds and metrics for the fraud class.

Models are compared by PR-AUC (average precision). F1, precision and recall are reported at a
threshold tuned on the validation split. Accuracy is deliberately absent: ~95% of rows are real.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

DECIMALS = 6  # rounding keeps committed metrics stable across platforms


def fraud_scores(model: Any, X: Any) -> np.ndarray:
    """Higher = more likely fraud. Probabilities where available, otherwise margins (LinearSVC)."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    return np.asarray(model.decision_function(X))


def pr_auc(y: Any, scores: np.ndarray) -> float:
    return float(average_precision_score(y, scores))


def tune_threshold(y: Any, scores: np.ndarray) -> float:
    """The score threshold that maximizes F1 (predict fraud when score >= threshold)."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    # The last precision/recall pair has no threshold.
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(thresholds[int(np.argmax(f1))])


def classification_metrics(y: Any, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    rounded = {
        "pr_auc": pr_auc(y, scores),
        "roc_auc": float(roc_auc_score(y, scores)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return {k: round(v, DECIMALS) for k, v in rounded.items()} | {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def pr_curve(y: Any, scores: np.ndarray) -> pd.DataFrame:
    precision, recall, _ = precision_recall_curve(y, scores)
    curve = pd.DataFrame({"recall": recall, "precision": precision}).round(4)
    return curve.drop_duplicates(ignore_index=True)
