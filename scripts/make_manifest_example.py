from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a manifest CSV for E-DAIC-style folders.")
    parser.add_argument(
        "--dataset-root",
        default="/home/rui/24zbma/data/E_DAIC/output",
        help="Root containing train/dev/test folders.",
    )
    parser.add_argument(
        "--labels-csv",
        default="/home/rui/24zbma/data/E_DAIC/detailed_lables.csv",
        help="CSV containing E-DAIC labels.",
    )
    parser.add_argument("--output", default="data/edaic_manifest.csv", help="Output CSV path.")
    parser.add_argument("--id-column", default="Participant", help="Subject id column in labels CSV.")
    parser.add_argument("--phq-column", default="Depression_severity", help="Depression severity column in labels CSV.")
    parser.add_argument("--label-column", default="Depression_label", help="Binary depression label column in labels CSV.")
    parser.add_argument("--split-column", default="split", help="Optional split column in labels CSV.")
    parser.add_argument("--participant-column", default="Participant", help="Optional numeric participant id column.")
    parser.add_argument("--audio-suffix", default="_AUDIO.wav", help="Audio filename suffix inside each subject folder.")
    parser.add_argument(
        "--transcript-suffix",
        default="_Transcript.csv",
        help="Transcript filename suffix inside each subject folder.",
    )
    return parser.parse_args()


def example_rows() -> list[dict]:
    return [
        {
            "participant_id": "300",
            "audio_path": "/path/to/audio/300.wav",
            "transcript_path": "/path/to/transcripts/300.txt",
            "phq_score": 4,
            "label": 0,
            "split": "train",
        },
        {
            "participant_id": "301",
            "audio_path": "/path/to/audio/301.wav",
            "transcript_path": "/path/to/transcripts/301.txt",
            "phq_score": 13,
            "label": 1,
            "split": "dev",
        },
        {
            "participant_id": "302",
            "audio_path": "/path/to/audio/302.wav",
            "transcript_path": "/path/to/transcripts/302.txt",
            "phq_score": 9,
            "label": 0,
            "split": "test",
        },
    ]


def normalize_split(value: str) -> str:
    value = value.strip().lower()
    if value in {"val", "valid", "validation"}:
        return "dev"
    return value


def load_labels(
    labels_csv: str,
    id_column: str,
    phq_column: str,
    label_column: str,
    split_column: str,
    participant_column: str,
) -> Dict[str, dict]:
    labels_df = pd.read_csv(labels_csv)
    missing = [
        column
        for column in (id_column, phq_column, label_column)
        if column not in labels_df.columns
    ]
    if missing:
        raise ValueError(f"Labels CSV is missing required columns: {missing}")

    has_split = split_column in labels_df.columns
    has_participant = participant_column in labels_df.columns
    return {
        str(row[id_column]): {
            "phq_score": float(row[phq_column]),
            "label": int(row[label_column]),
            "split": normalize_split(str(row[split_column])) if has_split else None,
            "participant_id": str(row[participant_column]) if has_participant else str(row[id_column]),
        }
        for _, row in labels_df.iterrows()
    }


def find_label_info(labels: Dict[str, dict], folder_id: str) -> tuple[str, dict]:
    if folder_id in labels:
        return folder_id, labels[folder_id]

    if folder_id.endswith("_P"):
        participant_id = folder_id[:-2]
        if participant_id in labels:
            return participant_id, labels[participant_id]

    raise KeyError(f"No label found for subject folder '{folder_id}'")


def find_file(subject_dir: Path, expected_name: str, suffix: str) -> Optional[Path]:
    expected_path = subject_dir / expected_name
    if expected_path.exists():
        return expected_path

    matches = sorted(subject_dir.glob(f"*{suffix}"))
    if matches:
        return matches[0]
    return None


def build_manifest(args: argparse.Namespace) -> list[dict]:
    dataset_root = Path(args.dataset_root)
    labels = load_labels(
        args.labels_csv,
        args.id_column,
        args.phq_column,
        args.label_column,
        args.split_column,
        args.participant_column,
    )
    rows = []
    split_dirs = {
        "train": "train",
        "dev": "dev",
        "val": "dev",
        "valid": "dev",
        "validation": "dev",
        "test": "test",
    }

    for folder_name, split in split_dirs.items():
        split_dir = dataset_root / folder_name
        if not split_dir.exists():
            continue

        for subject_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            folder_id = subject_dir.name
            audio_path = find_file(subject_dir, f"{folder_id}{args.audio_suffix}", args.audio_suffix)
            transcript_path = find_file(
                subject_dir,
                f"{folder_id}{args.transcript_suffix}",
                args.transcript_suffix,
            )
            if audio_path is None:
                raise FileNotFoundError(f"No audio file ending with {args.audio_suffix} in {subject_dir}")
            if transcript_path is None:
                raise FileNotFoundError(f"No transcript file ending with {args.transcript_suffix} in {subject_dir}")

            label_key, label_info = find_label_info(labels, folder_id)
            label_split = label_info["split"]
            if label_split is not None and label_split != split:
                raise ValueError(
                    f"Split mismatch for {folder_id} (label id '{label_key}'): folder is '{split}', "
                    f"but labels CSV says '{label_split}'"
                )

            rows.append(
                {
                    "participant_id": label_info["participant_id"],
                    "folder_name": folder_id,
                    "audio_path": str(audio_path.resolve()),
                    "transcript_path": str(transcript_path.resolve()),
                    "phq_score": label_info["phq_score"],
                    "label": label_info["label"],
                    "split": split,
                }
            )
    if not rows:
        raise ValueError(f"No subject folders found under {dataset_root}")
    return rows


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.dataset_root and args.labels_csv:
        rows = build_manifest(args)
    else:
        rows = example_rows()

    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
