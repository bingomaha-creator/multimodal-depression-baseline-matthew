from __future__ import annotations

"""在 LMVD 定长特征缓存上训练传统机器学习基线。

支持 Logistic Regression、RBF-SVM、Random Forest 和 KNN，并分别评估视频、音频、
双模态三种输入。默认使用同一个分层 K 折划分；传入 ``--split-file`` 时改用固定
train/valid/test。需要缩放的模型通过 sklearn
Pipeline 将 StandardScaler 和分类器绑定，因此 scaler 只会在每折训练集上拟合，
不会读取该折验证集的均值或标准差。

当前 ``classification_metrics`` 的 precision/recall/F1 是以标签 1 为正类的 binary
口径，不是 LMVD 论文表格中的 weighted 口径；论文对比时需要根据逐样本预测另算。
"""

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

from src.utils.lmvd_split import load_fixed_split_indices
from src.utils.metrics import classification_metrics, detailed_classification_metrics, find_best_threshold
from src.utils.seed import set_seed


MODEL_CHOICES = ("logistic_regression", "svm", "random_forest", "knn")
MODALITY_CHOICES = ("video", "audio", "both")


def parse_args() -> argparse.Namespace:
    """解析模型、模态和实验输出覆盖参数。"""

    parser = argparse.ArgumentParser(description="Train LMVD sklearn baselines with stratified CV.")
    parser.add_argument("--config", required=True, help="Path to LMVD YAML config.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_CHOICES), choices=MODEL_CHOICES)
    parser.add_argument("--modality", choices=MODALITY_CHOICES, default=None, help="Run one modality only.")
    parser.add_argument("--fold-limit", type=int, default=None, help="Optional fold limit for smoke tests.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument("--output-dir", default=None, help="Override ml_training.output_dir.")
    parser.add_argument(
        "--split-file",
        default=None,
        help="Optional index,label,fold CSV. When provided, run fixed train/valid/test instead of CV.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    """读取 YAML 实验配置。"""

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """用命令行参数覆盖少量常用配置；其余配置保持 YAML 中的值。"""

    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.output_dir is not None:
        config["ml_training"]["output_dir"] = args.output_dir
    return config


def load_cache(path: str) -> Dict[str, Any]:
    """读取 torch 或 pickle 格式的 LMVD 缓存。

    本地无 torch 环境可用缓存脚本的 ``--pickle-only`` 生成 pickle；服务器通常使用
    torch.save 格式。先尝试 torch.load，失败后再按 pickle 读取。
    """

    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        with open(path, "rb") as file:
            return pickle.load(file)


def as_numpy(value: Any) -> np.ndarray:
    """统一把缓存中的 Tensor 或数组转换成 CPU NumPy 数组。"""

    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def feature_matrix(items: list[Dict[str, Any]], modality: str) -> np.ndarray:
    """根据模态组装 sklearn 的二维输入矩阵 ``[样本数, 特征维]``。

    ``both`` 是在最后一维拼接视频和音频的定长表示，不是在这里做时间对齐。
    """

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
    """按缓存顺序提取二分类标签。"""

    return np.array([int(item["label"]) for item in items], dtype=int)


def participant_ids_from_items(items: list[Dict[str, Any]]) -> list[str]:
    """按缓存顺序提取 ID，写预测 CSV 时用于回溯样本。"""

    return [str(item["participant_id"]) for item in items]


def validate_cv_labels(labels: np.ndarray, n_splits: int) -> None:
    """在划分前确认两类样本数量足够支持指定折数。"""

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
    """创建一个新的 sklearn 分类器。

    LR、SVM、KNN 对特征尺度敏感，所以先 StandardScaler；RF 基于树分裂，不需要缩放。
    Pipeline 的 fit 只接收当前训练折，因而不会产生跨折标准化泄漏。
    """

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
        # RBF 核允许非线性决策边界。probability=True 会额外拟合概率校准，便于统一保存
        # prob_depressed；C 与 gamma 等仍是默认基线设置，没有在验证折上搜索参数。
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
    """返回标签 1 的概率；没有 predict_proba 时用 sigmoid 映射 decision score。"""

    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(features)[:, 1]
    scores = estimator.decision_function(features)
    return 1.0 / (1.0 + np.exp(-scores))


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """创建父目录并以可读格式保存指标 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def summarize_cv(fold_metrics: list[Dict[str, Any]]) -> Dict[str, Any]:
    """计算各折 binary 指标的宏观均值和总体标准差。

    这里是“先算每折指标，再对折取平均”，与把所有折预测合并后计算一次指标略有差异。
    """

    summary: Dict[str, Any] = {"folds": fold_metrics}
    for key in ("acc", "precision", "recall", "f1"):
        values = np.array([fold["metrics_at_0_5"][key] for fold in fold_metrics], dtype=float)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=0))
    return summary


def evaluate_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[Dict[str, object], np.ndarray]:
    """按指定阈值生成预测，并同时计算 binary 与 weighted 指标。"""

    predictions = (probabilities >= threshold).astype(int)
    metrics = detailed_classification_metrics(labels, predictions)
    metrics["threshold"] = float(threshold)
    metrics["pred_pos_rate"] = float(np.mean(predictions))
    return metrics, predictions


def fixed_prediction_rows(
    participant_ids: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    split_name: str,
) -> pd.DataFrame:
    """构造固定划分模式的逐样本预测表。"""

    return pd.DataFrame(
        {
            "participant_id": participant_ids,
            "label": labels,
            "pred_label": predictions,
            "prob_depressed": probabilities,
            "split": split_name,
        }
    )


def run_one_fixed_setting(
    model_name: str,
    modality: str,
    items: list[Dict[str, Any]],
    split_indices: Dict[str, np.ndarray],
    seed: int,
    output_root: Path,
) -> Dict[str, Any]:
    """使用公开的固定 train/valid/test 划分训练并评估一个模型组合。"""

    output_dir = output_root / model_name / modality
    metrics_dir = output_dir / "metrics"
    predictions_dir = output_dir / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    features = feature_matrix(items, modality)
    labels = labels_from_items(items)
    participant_ids = np.asarray(participant_ids_from_items(items))
    train_indices = split_indices["train"]
    valid_indices = split_indices["valid"]
    test_indices = split_indices["test"]

    train_classes = set(labels[train_indices].tolist())
    if train_classes != {0, 1}:
        raise ValueError(f"Fixed LMVD train split must contain labels 0 and 1, got {sorted(train_classes)}")

    # Pipeline 中的 StandardScaler 和分类器都只在 train 上 fit。
    estimator = build_estimator(model_name, seed=seed)
    estimator.fit(features[train_indices], labels[train_indices])

    valid_probabilities = positive_probabilities(estimator, features[valid_indices])
    test_probabilities = positive_probabilities(estimator, features[test_indices])
    best_threshold, _ = find_best_threshold(labels[valid_indices], valid_probabilities)

    valid_at_0_5, valid_predictions_at_0_5 = evaluate_probabilities(
        labels[valid_indices], valid_probabilities, threshold=0.5
    )
    valid_with_threshold, valid_predictions_with_threshold = evaluate_probabilities(
        labels[valid_indices], valid_probabilities, threshold=best_threshold
    )
    test_at_0_5, test_predictions_at_0_5 = evaluate_probabilities(
        labels[test_indices], test_probabilities, threshold=0.5
    )
    test_with_threshold, test_predictions_with_threshold = evaluate_probabilities(
        labels[test_indices], test_probabilities, threshold=best_threshold
    )

    fixed_prediction_rows(
        participant_ids[valid_indices],
        labels[valid_indices],
        valid_probabilities,
        valid_predictions_at_0_5,
        "valid",
    ).to_csv(predictions_dir / "valid_predictions_at_0_5.csv", index=False)
    fixed_prediction_rows(
        participant_ids[valid_indices],
        labels[valid_indices],
        valid_probabilities,
        valid_predictions_with_threshold,
        "valid",
    ).to_csv(predictions_dir / "valid_predictions_with_threshold.csv", index=False)
    fixed_prediction_rows(
        participant_ids[test_indices],
        labels[test_indices],
        test_probabilities,
        test_predictions_at_0_5,
        "test",
    ).to_csv(predictions_dir / "test_predictions_at_0_5.csv", index=False)
    fixed_prediction_rows(
        participant_ids[test_indices],
        labels[test_indices],
        test_probabilities,
        test_predictions_with_threshold,
        "test",
    ).to_csv(predictions_dir / "test_predictions_with_threshold.csv", index=False)

    summary = {
        "mode": "fixed_split",
        "model": model_name,
        "modality": modality,
        "split_counts": {fold: int(len(indices)) for fold, indices in split_indices.items()},
        "selected_threshold": float(best_threshold),
        "valid_at_0_5": valid_at_0_5,
        "valid_with_threshold": valid_with_threshold,
        "test_at_0_5": test_at_0_5,
        "test_with_threshold": test_with_threshold,
    }
    save_json(metrics_dir / "fixed_split_summary.json", summary)
    print(
        f"model={model_name} modality={modality} fixed-test "
        f"acc={test_at_0_5['acc']:.4f} binary_f1={test_at_0_5['binary_f1']:.4f} "
        f"weighted_f1={test_at_0_5['weighted_f1']:.4f} threshold=0.5"
    )
    return summary


def run_one_setting(
    model_name: str,
    modality: str,
    items: list[Dict[str, Any]],
    cfg: Dict[str, Any],
    fold_limit: int | None,
) -> Dict[str, Any]:
    """运行一个“模型 x 模态”组合的完整分层交叉验证。"""

    seed = int(cfg["seed"])
    train_cfg = cfg["ml_training"]
    n_splits = int(train_cfg["n_splits"])
    output_dir = Path(train_cfg["output_dir"]) / model_name / modality
    metrics_dir = output_dir / "metrics"
    predictions_dir = output_dir / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # 三个数组严格保持同一缓存顺序；KFold 返回的索引可同时切片特征、标签和 ID。
    features = feature_matrix(items, modality)
    labels = labels_from_items(items)
    participant_ids = np.array(participant_ids_from_items(items))
    validate_cv_labels(labels, n_splits)

    # 分层保证每折的正负样本比例接近全数据；固定 seed 使划分可复现。
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics = []
    for fold_idx, (train_indices, valid_indices) in enumerate(splitter.split(features, labels), start=1):
        if fold_limit is not None and fold_idx > fold_limit:
            break

        # 每折创建全新 estimator，避免上一折已经拟合的状态进入下一折。
        estimator = build_estimator(model_name, seed=seed + fold_idx)
        estimator.fit(features[train_indices], labels[train_indices])
        probabilities = positive_probabilities(estimator, features[valid_indices])
        # 传统基线统一使用固定 0.5 阈值，没有利用验证标签调阈值。
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
    """指定模态时只运行一个，否则依次运行 video/audio/both。"""

    if arg_modality is not None:
        return [arg_modality]
    return MODALITY_CHOICES


def main() -> None:
    """命令行入口：读取一次缓存，再遍历所有模型和模态组合。"""

    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    set_seed(int(config["seed"]))
    cache = load_cache(config["data"]["feature_cache_path"])
    items = cache["items"]
    print(f"Training LMVD sklearn baselines samples={len(items)} models={args.models}")

    if args.split_file is not None:
        split_indices = load_fixed_split_indices(args.split_file, items)
        output_root = Path(args.output_dir or "runs/LMVD/fixed_split/ml")
        print(
            f"Using fixed split {args.split_file}: "
            + ", ".join(f"{fold}={len(indices)}" for fold, indices in split_indices.items())
        )
        fixed_summaries = {}
        for model_name in args.models:
            for modality in selected_modalities(args.modality):
                fixed_summaries[f"{model_name}/{modality}"] = run_one_fixed_setting(
                    model_name=model_name,
                    modality=modality,
                    items=items,
                    split_indices=split_indices,
                    seed=int(config["seed"]),
                    output_root=output_root,
                )
        save_json(output_root / "metrics" / "all_fixed_split_summaries.json", fixed_summaries)
        return

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
