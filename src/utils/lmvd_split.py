from __future__ import annotations

"""读取并校验 LMVD 固定 train/valid/test 划分。"""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


EXPECTED_FOLDS = ("train", "valid", "test")
REQUIRED_COLUMNS = {"index", "label", "fold"}


def canonical_participant_id(value: Any) -> str:
    """规范化 ID，使 ``001`` 和缓存中的 ``1`` 能匹配。

    LMVD 的 ID 本质上是数字编号，但不同文件可能保留不同数量的前导零。非数字 ID
    原样保留，以便错误信息仍然可读。
    """

    participant_id = str(value).strip()
    if not participant_id:
        raise ValueError("LMVD participant ID cannot be empty")
    return str(int(participant_id)) if participant_id.isdigit() else participant_id


def _cache_id_map(items: List[Dict[str, Any]]) -> Dict[str, int]:
    id_to_index: Dict[str, int] = {}
    for index, item in enumerate(items):
        participant_id = canonical_participant_id(item["participant_id"])
        if participant_id in id_to_index:
            raise ValueError(f"Duplicate participant ID in feature cache: {participant_id}")
        id_to_index[participant_id] = index
    return id_to_index


def load_fixed_split_indices(
    split_path: str | Path,
    items: List[Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    """把 split CSV 中的参与者 ID 映射为特征缓存索引。

    CSV 必须包含 ``index,label,fold``，且与缓存恰好覆盖同一批样本。函数同时检查
    重复 ID、未知 fold、缺失样本以及标签不一致，防止静默使用错误划分。
    """

    path = Path(split_path)
    if not path.exists():
        raise FileNotFoundError(f"LMVD split file not found: {path}")

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"LMVD split file is missing columns: {sorted(missing_columns)}")

    frame = frame.loc[:, ["index", "label", "fold"]].copy()
    frame["canonical_id"] = frame["index"].map(canonical_participant_id)
    frame["fold"] = frame["fold"].str.strip().str.lower()

    duplicate_ids = sorted(frame.loc[frame["canonical_id"].duplicated(keep=False), "canonical_id"].unique())
    if duplicate_ids:
        raise ValueError(f"Duplicate participant IDs in LMVD split file: {duplicate_ids[:5]}")

    unknown_folds = sorted(set(frame["fold"]) - set(EXPECTED_FOLDS))
    if unknown_folds:
        raise ValueError(f"Unknown LMVD folds {unknown_folds}; expected {list(EXPECTED_FOLDS)}")

    cache_map = _cache_id_map(items)
    split_ids = set(frame["canonical_id"])
    cache_ids = set(cache_map)
    missing_ids = sorted(cache_ids - split_ids, key=lambda value: int(value) if value.isdigit() else value)
    extra_ids = sorted(split_ids - cache_ids, key=lambda value: int(value) if value.isdigit() else value)
    if missing_ids or extra_ids:
        # 错误信息使用文件中的原始 ID，而不是去掉前导零后的匹配键。
        missing_display = [str(items[cache_map[value]]["participant_id"]) for value in missing_ids]
        split_display = dict(zip(frame["canonical_id"], frame["index"]))
        extra_display = [str(split_display[value]) for value in extra_ids]
        raise ValueError(
            "LMVD split/cache ID mismatch: "
            f"missing from split={missing_display[:5]}, unknown in split={extra_display[:5]}"
        )

    split_indices: Dict[str, list[int]] = {fold: [] for fold in EXPECTED_FOLDS}
    for row in frame.itertuples(index=False):
        cache_index = cache_map[row.canonical_id]
        split_label = int(float(str(row.label).strip()))
        cache_label = int(items[cache_index]["label"])
        if split_label != cache_label:
            raise ValueError(
                f"Label mismatch for participant {row.index}: split={split_label}, cache={cache_label}"
            )
        split_indices[row.fold].append(cache_index)

    empty_folds = [fold for fold, indices in split_indices.items() if not indices]
    if empty_folds:
        raise ValueError(f"LMVD split contains empty folds: {empty_folds}")

    return {
        fold: np.asarray(indices, dtype=int)
        for fold, indices in split_indices.items()
    }
