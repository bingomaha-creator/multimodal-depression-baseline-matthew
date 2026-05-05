from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    return {
        "acc": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def find_best_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    min_threshold: float = 0.1,
    max_threshold: float = 0.9,
    num_thresholds: int = 81,
) -> tuple[float, Dict[str, float]]:
    thresholds = np.linspace(min_threshold, max_threshold, num_thresholds)
    best_threshold = 0.5
    best_metrics: Dict[str, float] = {}
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        metrics = classification_metrics(labels, predictions)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics
