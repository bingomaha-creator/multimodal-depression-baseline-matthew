from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils.metrics import classification_metrics
from src.utils.seed import set_seed


MODEL_CHOICES = ("logistic_regression", "svm", "random_forest", "knn")
MODALITY_CHOICES = ("video", "audio", "both")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LMVD sklearn baselines with stratified CV.")
    parser.add_argument("--config", required=True, help="Path to LMVD YAML config.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_CHOICES), choices=MODEL_CHOICES)
    parser.add_argument("--modality", choices=MODALITY_CHOICES, default=None, help="Run one modality only.")
    parser.add_argument("--fold-limit", type=int, default=None, help="Optional fold limit for smoke tests.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument("--output-dir", default=None, help="Override ml_training.output_dir.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.output_dir is not None:
        config["ml_training"]["output_dir"] = args.output_dir
    return config


def load_cache(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        with open(path, "rb") as file:
            return pickle.load(file)


def as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def feature_matrix(items: list[Dict[str, Any]], modality: str) -> np.ndarray:
    features = []
    for item in items:
        parts = []
        if modality in {"video", "both"}:
            parts.append(as_numpy(item["video_embedding"]))
        if modality in {"audio", "both"}:
            parts.append(as_numpy(item["audio_embedding"]))
        features.append(np.concatenate(parts, axis=-1))
    return np.stack(features).astype(np.float32)


def labels_from_items(items: list[Dict[str, Any]]) -> np.ndarray:
    return np.array([int(item["label"]) for item in items], dtype=int)


def participant_ids_from_items(items: list[Dict[str, Any]]) -> list[str]:
    return [str(item["participant_id"]) for item in items]


def validate_cv_labels(labels: np.ndarray, n_splits: int) -> None:
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique.tolist(), counts.tolist()))
    print("Label counts:", label_counts)
    if len(label_counts) < 2:
        raise ValueError(f"Cross-validation requires both classes, got label counts: {label_counts}")
    smallest_class = int(counts.min())
    if smallest_class < n_splits:
        raise ValueError(
            f"n_splits={n_splits} is larger than the smallest class count={smallest_class}. "
            f"Label counts: {label_counts}"
        )


def build_estimator(model_name: str, seed: int) -> Any:
    if model_name == "logistic_regression":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if model_name == "svm":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=seed)),
            ]
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "knn":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ]
        )
    raise ValueError(f"Unsupported model: {model_name}")


def positive_probabilities(estimator: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(features)[:, 1]
    scores = estimator.decision_function(features)
    return 1.0 / (1.0 + np.exp(-scores))


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def summarize_cv(fold_metrics: list[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"folds": fold_metrics}
    for key in ("acc", "precision", "recall", "f1"):
        values = np.array([fold["metrics_at_0_5"][key] for fold in fold_metrics], dtype=float)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=0))
    return summary


def run_one_setting(
    model_name: str,
    modality: str,
    items: list[Dict[str, Any]],
    cfg: Dict[str, Any],
    fold_limit: int | None,
) -> Dict[str, Any]:
    seed = int(cfg["seed"])
    train_cfg = cfg["ml_training"]
    n_splits = int(train_cfg["n_splits"])
    output_dir = Path(train_cfg["output_dir"]) / model_name / modality
    metrics_dir = output_dir / "metrics"
    predictions_dir = output_dir / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    features = feature_matrix(items, modality)
    labels = labels_from_items(items)
    participant_ids = np.array(participant_ids_from_items(items))
    validate_cv_labels(labels, n_splits)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics = []
    for fold_idx, (train_indices, valid_indices) in enumerate(splitter.split(features, labels), start=1):
        if fold_limit is not None and fold_idx > fold_limit:
            break

        estimator = build_estimator(model_name, seed=seed + fold_idx)
        estimator.fit(features[train_indices], labels[train_indices])
        probabilities = positive_probabilities(estimator, features[valid_indices])
        predictions = (probabilities >= 0.5).astype(int)
        metrics = classification_metrics(labels[valid_indices], predictions)
        metrics["threshold"] = 0.5
        metrics["pred_pos_rate"] = float(np.mean(predictions))

        rows = pd.DataFrame(
            {
                "participant_id": participant_ids[valid_indices],
                "label": labels[valid_indices],
                "pred_label": predictions,
                "prob_depressed": probabilities,
                "fold": fold_idx,
            }
        )
        rows.to_csv(predictions_dir / f"fold_{fold_idx}_valid_predictions_at_0_5.csv", index=False)
        save_json(metrics_dir / f"fold_{fold_idx}_valid_metrics_at_0_5.json", metrics)
        print(
            f"model={model_name} modality={modality} fold={fold_idx} "
            f"acc={metrics['acc']:.4f} precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
        )
        fold_metrics.append({"fold": fold_idx, "metrics_at_0_5": metrics})

    summary = summarize_cv(fold_metrics)
    save_json(metrics_dir / "cv_summary.json", summary)
    print(json.dumps({"model": model_name, "modality": modality, **summary}, indent=2, ensure_ascii=False))
    return summary


def selected_modalities(arg_modality: str | None) -> Iterable[str]:
    if arg_modality is not None:
        return [arg_modality]
    return MODALITY_CHOICES


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    set_seed(int(config["seed"]))
    cache = load_cache(config["data"]["feature_cache_path"])
    items = cache["items"]
    print(f"Training LMVD sklearn baselines samples={len(items)} models={args.models}")

    all_summaries = {}
    for model_name in args.models:
        for modality in selected_modalities(args.modality):
            all_summaries[f"{model_name}/{modality}"] = run_one_setting(
                model_name=model_name,
                modality=modality,
                items=items,
                cfg=config,
                fold_limit=args.fold_limit,
            )

    output_dir = Path(config["ml_training"]["output_dir"])
    save_json(output_dir / "metrics" / "all_cv_summaries.json", all_summaries)


if __name__ == "__main__":
    main()
