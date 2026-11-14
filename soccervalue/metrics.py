"""Scores for a probabilistic success model. Lower Brier is better."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def binary_scores(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities).astype(float)
    scores = {
        "n": int(y_true.size),
        "base_rate": float(y_true.mean()) if y_true.size else None,
        "brier": float(brier_score_loss(y_true, probabilities)) if y_true.size else None,
    }
    if y_true.size and len(np.unique(y_true)) == 2:
        scores["auroc"] = float(roc_auc_score(y_true, probabilities))
    else:
        scores["auroc"] = None
    return scores


def calibration_table(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> list[dict]:
    y_true = np.asarray(y_true).astype(float)
    probabilities = np.asarray(probabilities).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        if high == 1.0:
            mask = (probabilities >= low) & (probabilities <= high)
        else:
            mask = (probabilities >= low) & (probabilities < high)
        count = int(mask.sum())
        rows.append(
            {
                "bin_low": float(low),
                "bin_high": float(high),
                "n": count,
                "mean_predicted": float(probabilities[mask].mean()) if count else None,
                "fraction_success": float(y_true[mask].mean()) if count else None,
            }
        )
    return rows
