from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    return {
        "acc": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def detailed_classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, object]:
    """同时计算正类 binary 指标、weighted 指标和混淆矩阵。

    保留 ``acc``，并给两种平均方式添加明确前缀，避免汇报时把 binary F1 误写成
    LMVD 论文使用的 weighted F1。混淆矩阵固定标签顺序为 ``[0, 1]``。
    """

    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    return {
        "acc": float(accuracy_score(labels, predictions)),
        "binary_precision": float(precision_score(labels, predictions, average="binary", zero_division=0)),
        "binary_recall": float(recall_score(labels, predictions, average="binary", zero_division=0)),
        "binary_f1": float(f1_score(labels, predictions, average="binary", zero_division=0)),
        "weighted_precision": float(precision_score(labels, predictions, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(labels, predictions, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).astype(int).tolist(),
    }


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    errors = predictions - targets

    target_mean = float(np.mean(targets)) if targets.size else 0.0
    prediction_mean = float(np.mean(predictions)) if predictions.size else 0.0
    target_var = float(np.mean((targets - target_mean) ** 2)) if targets.size else 0.0
    prediction_var = float(np.mean((predictions - prediction_mean) ** 2)) if predictions.size else 0.0
    covariance = (
        float(np.mean((targets - target_mean) * (predictions - prediction_mean)))
        if targets.size
        else 0.0
    )
    ccc_denominator = target_var + prediction_var + (target_mean - prediction_mean) ** 2

    return {
        "mae": float(np.mean(np.abs(errors))) if errors.size else 0.0,
        "rmse": float(np.sqrt(np.mean(errors**2))) if errors.size else 0.0,
        "ccc": float((2.0 * covariance) / ccc_denominator) if ccc_denominator > 0.0 else 0.0,
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
