from __future__ import annotations

"""在 LMVD 定长特征缓存上训练 MLP，支持分层 K 折或固定数据划分。

每折都会重新建立模型、优化器和损失函数。训练集用于更新参数，验证折用于选择该折
表现最好的 epoch；随后保存固定 0.5 阈值和验证折最优阈值两套结果。

十折模式口径提示：验证折同时承担“选择最佳 epoch”和“汇报该折性能”的角色，
``best_threshold`` 也在同一验证折上搜索，因此二者不是严格独立测试集估计，可能偏乐观。
与论文做严谨比较时，应使用固定 epoch/阈值，或增加内层验证集形成 nested CV。
固定 0.5 的指标没有阈值搜索偏差，但仍受最佳 epoch 选择影响。

传入 ``--split-file`` 后改用 train/valid/test：valid 负责选择 epoch 和阈值，test 只在
训练完成后评估，因而适合与采用固定 8:1:1 划分的 DepMamba/CAF-Mamba 做对照。

当前 ``classification_metrics`` 的 precision/recall/F1 是标签 1 的 binary 口径，
并非 LMVD 论文使用的 weighted 口径。
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.lmvd_feature_dataset import LMVDFeatureDataset, collate_lmvd_features
from src.models.lmvd_feature_baseline import LMVDFeatureBaseline
from src.utils.lmvd_split import load_fixed_split_indices
from src.utils.metrics import classification_metrics, detailed_classification_metrics, find_best_threshold
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """解析配置路径和常用实验覆盖参数。"""

    parser = argparse.ArgumentParser(description="Train LMVD feature MLP baseline with stratified CV.")
    parser.add_argument("--config", required=True, help="Path to LMVD YAML config.")
    parser.add_argument("--max-train-steps", type=int, default=None, help="Optional train step limit for smoke tests.")
    parser.add_argument("--fold-limit", type=int, default=None, help="Optional fold limit for smoke tests.")
    parser.add_argument("--modality", choices=["video", "audio", "both"], default=None, help="Override model modality.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument("--output-dir", default=None, help="Override training.output_dir.")
    parser.add_argument(
        "--split-file",
        default=None,
        help="Optional index,label,fold CSV. When provided, run fixed train/valid/test instead of CV.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    """读取 YAML 配置并返回普通字典。"""

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """用命令行 seed/output-dir 覆盖 YAML，便于多次实验写入不同目录。"""

    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.output_dir is not None:
        config["training"]["output_dir"] = args.output_dir
    return config


def resolve_device(config_device: str) -> torch.device:
    """解析训练设备；auto 优先 CUDA，不可用时回退 CPU。"""

    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """只移动 Tensor，participant_id 等 Python 对象原样保留。"""

    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """创建父目录并以 UTF-8、缩进格式保存 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def validate_cv_labels(labels: np.ndarray, n_splits: int) -> None:
    """确认数据包含两个类别，且最小类别样本数不少于折数。"""

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


def compute_class_weights(labels: np.ndarray, train_indices: np.ndarray, device: torch.device) -> torch.Tensor:
    """仅根据当前训练折计算 CrossEntropyLoss 的类别权重。

    权重公式是 ``总数 / (类别数 * 该类数量)``。验证折标签不会参与权重计算，
    因此这里没有跨折标签泄漏。LMVD 类别接近平衡时，两个权重通常都接近 1。
    """

    train_labels = labels[train_indices].astype(int).tolist()
    num_neg = train_labels.count(0)
    num_pos = train_labels.count(1)
    total = num_neg + num_pos
    weights = torch.tensor(
        [
            total / (2 * max(num_neg, 1)),
            total / (2 * max(num_pos, 1)),
        ],
        dtype=torch.float,
        device=device,
    )
    print(
        f"class_weights: neg={weights[0].item():.4f}, pos={weights[1].item():.4f} "
        f"(train counts: neg={num_neg}, pos={num_pos})"
    )
    return weights


def build_model(metadata: Dict[str, Any], model_cfg: Dict[str, Any], modality: str) -> LMVDFeatureBaseline:
    """使用缓存记录的真实特征维度创建 MLP，避免在配置中手填维度。"""

    return LMVDFeatureBaseline(
        video_dim=int(metadata["video_dim"]),
        audio_dim=int(metadata["audio_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
        modality=modality,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    max_train_steps: Optional[int] = None,
) -> float:
    """训练一个 epoch，返回 batch loss 的平均值。

    ``max_train_steps`` 只用于快速冒烟测试；启用后每个 epoch 都只运行指定批次数，
    不应把这种结果当作完整训练结果。
    """

    model.train()
    running_loss = 0.0
    step_count = 0
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)

    for batch in progress:
        batch = move_batch_to_device(batch, device)
        # set_to_none=True 比把梯度清零更省一次写操作；下一次 backward 会重新创建梯度。
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            video_embeddings=batch["video_embeddings"],
            audio_embeddings=batch["audio_embeddings"],
        )
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        step_count += 1
        progress.set_postfix(loss=running_loss / step_count)
        if max_train_steps is not None and step_count >= max_train_steps:
            break

    return running_loss / max(step_count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[Dict[str, float], pd.DataFrame]:
    """在指定阈值下评估模型，同时返回指标和逐样本预测表。

    softmax 第 1 列被解释为“抑郁/标签 1”的概率。逐样本表是后续重新计算 weighted
    指标、绘制混淆矩阵和追踪错误样本的依据。
    """

    model.eval()
    losses = []
    labels = []
    predictions = []
    probabilities = []
    rows = []

    for batch in tqdm(loader, desc="evaluate", leave=False):
        batch = move_batch_to_device(batch, device)
        logits = model(
            video_embeddings=batch["video_embeddings"],
            audio_embeddings=batch["audio_embeddings"],
        )
        loss = criterion(logits, batch["labels"])
        # 模型输出两个未归一化 logits；softmax 后两列分别对应标签 0 和标签 1。
        probs = torch.softmax(logits, dim=-1)
        preds = (probs[:, 1] >= threshold).long()

        batch_labels = batch["labels"].detach().cpu().numpy().tolist()
        batch_preds = preds.detach().cpu().numpy().tolist()
        batch_probs = probs[:, 1].detach().cpu().numpy().tolist()
        losses.append(float(loss.item()))
        labels.extend(batch_labels)
        predictions.extend(batch_preds)
        probabilities.extend(batch_probs)

        for participant_id, label, pred, prob in zip(
            batch["participant_id"],
            batch_labels,
            batch_preds,
            batch_probs,
        ):
            rows.append(
                {
                    "participant_id": participant_id,
                    "label": label,
                    "pred_label": pred,
                    "prob_depressed": prob,
                }
            )

    metrics = classification_metrics(np.array(labels), np.array(predictions))
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["threshold"] = float(threshold)
    metrics["prob_min"] = float(np.min(probabilities)) if probabilities else 0.0
    metrics["prob_max"] = float(np.max(probabilities)) if probabilities else 0.0
    metrics["prob_mean"] = float(np.mean(probabilities)) if probabilities else 0.0
    metrics["pred_pos_rate"] = float(np.mean(predictions)) if predictions else 0.0
    return metrics, pd.DataFrame(rows)


def summarize_cv(fold_metrics: list[Dict[str, Any]]) -> Dict[str, Any]:
    """对固定 0.5 阈值下的每折指标计算均值和总体标准差。

    这是“每折先计算，再对各折取平均”的 CV 汇总，不等于把所有折预测合并后计算一次。
    """

    summary: Dict[str, Any] = {"folds": fold_metrics}
    for key in ("acc", "precision", "recall", "f1", "loss"):
        values = np.array([fold["metrics_at_0_5"][key] for fold in fold_metrics], dtype=float)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=0))
    return summary


def add_detailed_metrics(
    metrics: Dict[str, float],
    predictions: pd.DataFrame,
) -> Dict[str, object]:
    """在评估运行信息上补充 binary、weighted 指标和混淆矩阵。"""

    detailed = detailed_classification_metrics(
        predictions["label"].to_numpy(),
        predictions["pred_label"].to_numpy(),
    )
    for key in ("loss", "threshold", "prob_min", "prob_max", "prob_mean", "pred_pos_rate"):
        detailed[key] = float(metrics[key])
    return detailed


def run_fixed_split(
    config: Dict[str, Any],
    full_dataset: LMVDFeatureDataset,
    labels: np.ndarray,
    split_indices: Dict[str, np.ndarray],
    modality: str,
    device: torch.device,
    output_root: Path,
    max_train_steps: Optional[int],
) -> Dict[str, Any]:
    """在公开的固定 train/valid/test 划分上训练并最终评估 MLP。"""

    data_cfg = config["data"]
    train_cfg = config["training"]
    model_cfg = config["model"]
    output_dir = output_root / modality
    checkpoint_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    predictions_dir = output_dir / "predictions"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    train_indices = split_indices["train"]
    valid_indices = split_indices["valid"]
    test_indices = split_indices["test"]
    train_dataset = LMVDFeatureDataset(data_cfg["feature_cache_path"], train_indices)
    valid_dataset = LMVDFeatureDataset(data_cfg["feature_cache_path"], valid_indices)
    test_dataset = LMVDFeatureDataset(data_cfg["feature_cache_path"], test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg["num_workers"]),
        collate_fn=collate_lmvd_features,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        collate_fn=collate_lmvd_features,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        collate_fn=collate_lmvd_features,
    )

    model = build_model(full_dataset.metadata, model_cfg, modality).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    if bool(train_cfg.get("use_class_weights", True)):
        criterion = nn.CrossEntropyLoss(weight=compute_class_weights(labels, train_indices, device))
    else:
        criterion = nn.CrossEntropyLoss()

    best_metric = -1.0
    best_metrics: Dict[str, float] = {}
    best_path = checkpoint_dir / "best.pt"
    monitor_metric = str(train_cfg["monitor_metric"])
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            max_train_steps=max_train_steps,
        )
        valid_metrics, _ = evaluate(model, valid_loader, criterion, device, threshold=0.5)
        valid_metrics["train_loss"] = train_loss
        print(
            f"fixed epoch={epoch} train_loss={train_loss:.4f} "
            f"valid_acc={valid_metrics['acc']:.4f} valid_f1={valid_metrics['f1']:.4f}"
        )
        current_metric = float(valid_metrics[monitor_metric])
        if current_metric > best_metric:
            best_metric = current_metric
            best_metrics = valid_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "mode": "fixed_split",
                    "modality": modality,
                    "split_indices": {fold: indices.tolist() for fold, indices in split_indices.items()},
                    "valid_metrics_at_0_5": valid_metrics,
                    "metadata": full_dataset.metadata,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    valid_raw_0_5, valid_predictions_0_5 = evaluate(model, valid_loader, criterion, device, threshold=0.5)
    best_threshold, _ = find_best_threshold(
        valid_predictions_0_5["label"].to_numpy(),
        valid_predictions_0_5["prob_depressed"].to_numpy(),
    )
    valid_raw_tuned, valid_predictions_tuned = evaluate(
        model, valid_loader, criterion, device, threshold=best_threshold
    )
    test_raw_0_5, test_predictions_0_5 = evaluate(model, test_loader, criterion, device, threshold=0.5)
    test_raw_tuned, test_predictions_tuned = evaluate(
        model, test_loader, criterion, device, threshold=best_threshold
    )

    valid_at_0_5 = add_detailed_metrics(valid_raw_0_5, valid_predictions_0_5)
    valid_with_threshold = add_detailed_metrics(valid_raw_tuned, valid_predictions_tuned)
    test_at_0_5 = add_detailed_metrics(test_raw_0_5, test_predictions_0_5)
    test_with_threshold = add_detailed_metrics(test_raw_tuned, test_predictions_tuned)

    valid_predictions_0_5.assign(split="valid").to_csv(
        predictions_dir / "valid_predictions_at_0_5.csv", index=False
    )
    valid_predictions_tuned.assign(split="valid").to_csv(
        predictions_dir / "valid_predictions_with_threshold.csv", index=False
    )
    test_predictions_0_5.assign(split="test").to_csv(
        predictions_dir / "test_predictions_at_0_5.csv", index=False
    )
    test_predictions_tuned.assign(split="test").to_csv(
        predictions_dir / "test_predictions_with_threshold.csv", index=False
    )

    summary = {
        "mode": "fixed_split",
        "modality": modality,
        "split_counts": {fold: int(len(indices)) for fold, indices in split_indices.items()},
        "selected_threshold": float(best_threshold),
        "best_epoch_valid_metrics": best_metrics,
        "valid_at_0_5": valid_at_0_5,
        "valid_with_threshold": valid_with_threshold,
        "test_at_0_5": test_at_0_5,
        "test_with_threshold": test_with_threshold,
    }
    save_json(metrics_dir / "fixed_split_summary.json", summary)

    checkpoint["selected_threshold"] = float(best_threshold)
    checkpoint["fixed_split_summary"] = summary
    torch.save(checkpoint, best_path)
    print(
        f"fixed-test modality={modality} acc={test_at_0_5['acc']:.4f} "
        f"binary_f1={test_at_0_5['binary_f1']:.4f} "
        f"weighted_f1={test_at_0_5['weighted_f1']:.4f} threshold=0.5"
    )
    return summary


def main() -> None:
    """组织配置读取、数据划分、逐折训练、checkpoint 和指标落盘。"""

    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    set_seed(int(config["seed"]))

    data_cfg = config["data"]
    train_cfg = config["training"]
    model_cfg = config["model"]
    modality = args.modality or str(model_cfg.get("modality", "both"))
    device = resolve_device(str(train_cfg["device"]))
    # full_dataset 在这里主要提供完整 items 和 metadata；每折会再按索引创建子数据集。
    full_dataset = LMVDFeatureDataset(data_cfg["feature_cache_path"])
    labels = np.array([int(item["label"]) for item in full_dataset.items], dtype=int)
    participants = [str(item["participant_id"]) for item in full_dataset.items]
    max_train_steps = args.max_train_steps
    if max_train_steps is None:
        max_train_steps = train_cfg.get("max_train_steps")

    if args.split_file is not None:
        split_indices = load_fixed_split_indices(args.split_file, full_dataset.items)
        output_root = Path(args.output_dir or "runs/LMVD/fixed_split/mlp")
        print(
            f"Training LMVD MLP fixed split modality={modality} "
            + ", ".join(f"{fold}={len(indices)}" for fold, indices in split_indices.items())
        )
        run_fixed_split(
            config=config,
            full_dataset=full_dataset,
            labels=labels,
            split_indices=split_indices,
            modality=modality,
            device=device,
            output_root=output_root,
            max_train_steps=max_train_steps,
        )
        return

    output_dir = Path(train_cfg["output_dir"]) / modality
    output_dir.mkdir(parents=True, exist_ok=True)
    n_splits = int(train_cfg["n_splits"])
    print(f"Training LMVD MLP modality={modality} samples={len(full_dataset)}")
    validate_cv_labels(labels, n_splits)

    # 分层划分使每折正负比例接近整体；shuffle + 固定 seed 使具体 ID 划分可复现。
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(config["seed"]))
    fold_metrics = []
    for fold_idx, (train_indices, valid_indices) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
        if args.fold_limit is not None and fold_idx > args.fold_limit:
            break

        fold_dir = output_dir / f"fold_{fold_idx}"
        checkpoint_dir = fold_dir / "checkpoints"
        metrics_dir = fold_dir / "metrics"
        predictions_dir = fold_dir / "predictions"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"fold={fold_idx} train={len(train_indices)} valid={len(valid_indices)} "
            f"first_valid_participant={participants[valid_indices[0]]}"
        )
        # 两个 Dataset 只保留对应索引；valid_loader 不打乱，方便预测文件稳定复现。
        train_dataset = LMVDFeatureDataset(data_cfg["feature_cache_path"], train_indices)
        valid_dataset = LMVDFeatureDataset(data_cfg["feature_cache_path"], valid_indices)
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=True,
            num_workers=int(data_cfg["num_workers"]),
            collate_fn=collate_lmvd_features,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(data_cfg["num_workers"]),
            collate_fn=collate_lmvd_features,
        )

        # 每折从头初始化模型和优化器，不能复用上一折参数。
        model = build_model(full_dataset.metadata, model_cfg, modality).to(device)
        optimizer = AdamW(
            model.parameters(),
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        if bool(train_cfg.get("use_class_weights", True)):
            criterion = nn.CrossEntropyLoss(weight=compute_class_weights(labels, train_indices, device))
        else:
            criterion = nn.CrossEntropyLoss()

        # 当前实现用该折验证指标选择最佳 epoch。它适合模型开发，但同一折最终指标会
        # 带有模型选择偏差；严格泛化评估应增加独立测试层或 nested CV。
        best_metric = -1.0
        best_metrics: Dict[str, float] = {}
        best_path = checkpoint_dir / "best.pt"
        for epoch in range(1, int(train_cfg["epochs"]) + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                epoch,
                max_train_steps=max_train_steps,
            )
            valid_metrics, _ = evaluate(model, valid_loader, criterion, device, threshold=0.5)
            valid_metrics["train_loss"] = train_loss
            print(
                f"fold={fold_idx} epoch={epoch} train_loss={train_loss:.4f} "
                f"valid_acc={valid_metrics['acc']:.4f} valid_precision={valid_metrics['precision']:.4f} "
                f"valid_recall={valid_metrics['recall']:.4f} valid_f1={valid_metrics['f1']:.4f} "
                f"pred_pos_rate={valid_metrics['pred_pos_rate']:.4f}"
            )

            monitor_metric = str(train_cfg["monitor_metric"])
            current_metric = float(valid_metrics[monitor_metric])
            if current_metric > best_metric:
                best_metric = current_metric
                best_metrics = valid_metrics
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "fold": fold_idx,
                        "modality": modality,
                        "valid_indices": valid_indices.tolist(),
                        "metrics_at_0_5": valid_metrics,
                        "metadata": full_dataset.metadata,
                    },
                    best_path,
                )

        # 恢复该折最佳 epoch，而不是直接使用最后一个 epoch 的参数。
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        metrics_at_0_5, predictions_at_0_5 = evaluate(model, valid_loader, criterion, device, threshold=0.5)
        labels_valid = predictions_at_0_5["label"].to_numpy()
        probs_valid = predictions_at_0_5["prob_depressed"].to_numpy()
        # 这一阈值直接在当前验证折标签上搜索，metrics_with_threshold 只能作为分析结果，
        # 不能视为独立测试性能。对外主结果优先使用固定 0.5，或在内层验证集调阈值。
        best_threshold, _ = find_best_threshold(labels_valid, probs_valid)
        metrics_with_threshold, predictions_with_threshold = evaluate(
            model,
            valid_loader,
            criterion,
            device,
            threshold=best_threshold,
        )
        metrics_with_threshold["best_threshold"] = best_threshold

        predictions_at_0_5.to_csv(predictions_dir / "valid_predictions_at_0_5.csv", index=False)
        predictions_with_threshold.to_csv(predictions_dir / "valid_predictions_with_threshold.csv", index=False)
        save_json(metrics_dir / "valid_metrics_at_0_5.json", metrics_at_0_5)
        save_json(metrics_dir / "valid_metrics_with_threshold.json", metrics_with_threshold)
        save_json(metrics_dir / "best_epoch_metrics.json", best_metrics)

        # 把最终评估信息回写 checkpoint，便于之后只加载一个文件即可完成推理和追溯。
        checkpoint["best_threshold"] = best_threshold
        checkpoint["metrics_at_0_5"] = metrics_at_0_5
        checkpoint["metrics_with_threshold"] = metrics_with_threshold
        torch.save(checkpoint, best_path)

        fold_metrics.append(
            {
                "fold": fold_idx,
                "metrics_at_0_5": metrics_at_0_5,
                "metrics_with_threshold": metrics_with_threshold,
            }
        )

    summary = summarize_cv(fold_metrics)
    save_json(output_dir / "metrics" / "cv_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
