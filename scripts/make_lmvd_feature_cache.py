from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


VIDEO_METADATA_COLUMNS = {"frame", "face_id", "timestamp"}


@dataclass(frozen=True)
class LMVDSample:
    participant_id: str
    video_path: Path
    audio_path: Path
    label_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an LMVD feature cache from released CSV/NPY features.")
    parser.add_argument("--dataset-root", default="/24zbma/data/LMVD", help="Root containing LMVD feature folders.")
    parser.add_argument("--video-dir", default="Video_feature", help="Video feature subdirectory.")
    parser.add_argument("--audio-dir", default="Audio_feature", help="Audio feature subdirectory.")
    parser.add_argument("--label-dir", default="label", help="Label subdirectory.")
    parser.add_argument("--output", default="data/lmvd_features.pt", help="Output cache path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests.")
    parser.add_argument(
        "--pickle-only",
        action="store_true",
        help="Write a pickle cache instead of torch.save. Useful on local machines without torch.",
    )
    return parser.parse_args()


def sort_key(participant_id: str) -> tuple[int, Any]:
    return (0, int(participant_id)) if participant_id.isdigit() else (1, participant_id)


def collect_video_files(video_root: Path) -> Dict[str, Path]:
    if not video_root.exists():
        return {}
    return {
        path.stem: path
        for path in video_root.rglob("*.csv")
        if not path.name.endswith("_Depression.csv")
    }


def collect_audio_files(audio_root: Path) -> Dict[str, Path]:
    if not audio_root.exists():
        return {}
    return {path.stem: path for path in audio_root.rglob("*.npy")}


def collect_label_files(label_root: Path) -> Dict[str, Path]:
    if not label_root.exists():
        return {}
    return {
        path.name[: -len("_Depression.csv")]: path
        for path in label_root.rglob("*_Depression.csv")
    }


def format_examples(values: List[str], limit: int = 5) -> str:
    if not values:
        return "[]"
    return str(values[:limit])


def format_discovery_error(
    dataset_root: str | Path,
    video_dir: str,
    audio_dir: str,
    label_dir: str,
) -> str:
    root = Path(dataset_root)
    video_root = root / video_dir
    audio_root = root / audio_dir
    label_root = root / label_dir
    video_files = collect_video_files(video_root)
    audio_files = collect_audio_files(audio_root)
    label_files = collect_label_files(label_root)
    complete_ids = set(video_files) & set(audio_files) & set(label_files)
    video_only = sorted(set(video_files) - set(audio_files) - set(label_files), key=sort_key)
    audio_only = sorted(set(audio_files) - set(video_files) - set(label_files), key=sort_key)
    label_only = sorted(set(label_files) - set(video_files) - set(audio_files), key=sort_key)

    return (
        f"No complete LMVD samples found under {dataset_root}. "
        f"Expected folders: video={video_root} exists={video_root.exists()}, "
        f"audio={audio_root} exists={audio_root.exists()}, "
        f"label={label_root} exists={label_root.exists()}. "
        f"Found video csv files={len(video_files)}, audio npy files={len(audio_files)}, "
        f"label csv files={len(label_files)}, complete id overlap={len(complete_ids)}. "
        f"Example video ids={format_examples(sorted(video_files, key=sort_key))}; "
        f"audio ids={format_examples(sorted(audio_files, key=sort_key))}; "
        f"label ids={format_examples(sorted(label_files, key=sort_key))}. "
        f"Unmatched examples: video_only={format_examples(video_only)}, "
        f"audio_only={format_examples(audio_only)}, label_only={format_examples(label_only)}."
    )


def discover_lmvd_samples(
    dataset_root: str | Path,
    video_dir: str = "Video_feature",
    audio_dir: str = "Audio_feature",
    label_dir: str = "label",
) -> List[LMVDSample]:
    root = Path(dataset_root)
    video_root = root / video_dir
    audio_root = root / audio_dir
    label_root = root / label_dir

    video_files = collect_video_files(video_root)
    audio_files = collect_audio_files(audio_root)
    label_files = collect_label_files(label_root)

    participant_ids = sorted(
        set(video_files) & set(audio_files) & set(label_files),
        key=sort_key,
    )
    return [
        LMVDSample(
            participant_id=participant_id,
            video_path=video_files[participant_id],
            audio_path=audio_files[participant_id],
            label_path=label_files[participant_id],
        )
        for participant_id in participant_ids
    ]


def read_lmvd_label(label_path: str | Path) -> int:
    columns = pd.read_csv(label_path, nrows=0).columns.tolist()
    if not columns:
        raise ValueError(f"Label file has no header: {label_path}")
    return int(float(str(columns[0]).strip()))


def summarize_temporal_features(features: np.ndarray, source: str = "features") -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.size == 0 or array.shape[0] == 0:
        raise ValueError(f"Empty temporal feature array: {source}")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)

    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([array.mean(axis=0), array.std(axis=0)], axis=0).astype(np.float32)


def load_video_embedding(video_path: str | Path) -> np.ndarray:
    frame = pd.read_csv(video_path)
    frame.columns = [str(column).strip() for column in frame.columns]
    numeric = frame.select_dtypes(include="number")
    feature_columns = [
        column
        for column in numeric.columns
        if column.strip().lower() not in VIDEO_METADATA_COLUMNS
    ]
    if not feature_columns:
        raise ValueError(f"No usable numeric video feature columns found: {video_path}")
    return summarize_temporal_features(numeric[feature_columns].to_numpy(dtype=np.float32), source=str(video_path))


def load_audio_embedding(audio_path: str | Path) -> np.ndarray:
    return summarize_temporal_features(np.load(audio_path, allow_pickle=False), source=str(audio_path))


def tensorize_cache(cache: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    items = []
    for item in cache["items"]:
        items.append(
            {
                **item,
                "video_embedding": torch.tensor(item["video_embedding"], dtype=torch.float),
                "audio_embedding": torch.tensor(item["audio_embedding"], dtype=torch.float),
            }
        )
    return {"metadata": cache["metadata"], "items": items}


def build_feature_cache(
    dataset_root: str | Path,
    output_path: str | Path,
    video_dir: str = "Video_feature",
    audio_dir: str = "Audio_feature",
    label_dir: str = "label",
    limit: int | None = None,
    save_torch: bool = True,
) -> Dict[str, Any]:
    samples = discover_lmvd_samples(dataset_root, video_dir=video_dir, audio_dir=audio_dir, label_dir=label_dir)
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError(format_discovery_error(dataset_root, video_dir, audio_dir, label_dir))

    items = []
    for sample in samples:
        items.append(
            {
                "participant_id": sample.participant_id,
                "video_path": str(sample.video_path),
                "audio_path": str(sample.audio_path),
                "label_path": str(sample.label_path),
                "video_embedding": load_video_embedding(sample.video_path),
                "audio_embedding": load_audio_embedding(sample.audio_path),
                "label": read_lmvd_label(sample.label_path),
            }
        )

    metadata = {
        "dataset": "LMVD",
        "dataset_root": str(Path(dataset_root)),
        "video_dim": int(items[0]["video_embedding"].shape[0]),
        "audio_dim": int(items[0]["audio_embedding"].shape[0]),
        "num_items": len(items),
        "pooling": "mean_std",
    }
    cache = {"metadata": metadata, "items": items}

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if save_torch:
        import torch

        torch.save(tensorize_cache(cache), output)
    else:
        with output.open("wb") as file:
            pickle.dump(cache, file)

    print(f"Wrote {output} with {len(items)} LMVD samples")
    print("Label counts:", pd.Series([item["label"] for item in items]).value_counts().sort_index().to_dict())
    print(f"video_dim={metadata['video_dim']} audio_dim={metadata['audio_dim']}")
    return cache


def main() -> None:
    args = parse_args()
    build_feature_cache(
        dataset_root=args.dataset_root,
        output_path=args.output,
        video_dir=args.video_dir,
        audio_dir=args.audio_dir,
        label_dir=args.label_dir,
        limit=args.limit,
        save_torch=not args.pickle_only,
    )


if __name__ == "__main__":
    main()
