from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


LABEL_MAP = {
    "MDD": 1,
    "HC": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a MODMA manifest with 18 audio/text segments per subject.")
    parser.add_argument("--transcript-json",
                        default='/home/rui/24zbma/data/audio_lanzhou_2015(1)/audio_lanzhou_2015/modma_final_text.json',
                        help="JSON file containing all subject transcripts.")
    parser.add_argument("--label-xlsx",
                        default='/home/rui/24zbma/data/audio_lanzhou_2015(1)/audio_lanzhou_2015/subjects_information_audio_lanzhou_2015.xlsx',
                        help="Excel file containing subject labels.")
    parser.add_argument("--audio-root", default='/home/rui/24zbma/data/audio_lanzhou_2015(1)/audio_lanzhou_2015',
                        help="Root directory containing subject audio folders.")
    parser.add_argument("--output", default="data/modma_manifest.csv", help="Output manifest CSV path.")
    parser.add_argument(
        "--processed-text-dir",
        default="data/modma_processed_texts",
        help="Directory for generated merged transcript txt files.",
    )
    parser.add_argument("--subject-column", default="subject id", help="Subject id column in the label xlsx.")
    parser.add_argument("--label-column", default="type", help="Label column in the label xlsx.")
    parser.add_argument("--subject-id-width", type=int, default=8, help="Zero-pad numeric subject ids to this width.")
    parser.add_argument("--num-segments", type=int, default=18, help="Number of required audio/text segments.")
    parser.add_argument("--split", default="all", help="Split value written to the manifest.")
    return parser.parse_args()


def normalize_subject_id(value: Any, width: int = 8) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    value = str(value).strip()
    if value.isdigit() and width > 0:
        return value.zfill(width)
    return value


def normalize_audio_index(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    value = str(value).strip()
    if value.isdigit():
        return value.zfill(2)
    return value


def load_labels(
    label_xlsx: str,
    subject_column: str,
    label_column: str,
    subject_id_width: int,
) -> Dict[str, Dict[str, Any]]:
    labels_df = pd.read_excel(label_xlsx, dtype={subject_column: str})
    missing = [column for column in (subject_column, label_column) if column not in labels_df.columns]
    if missing:
        raise ValueError(f"Label xlsx is missing required columns: {missing}")

    labels: Dict[str, Dict[str, Any]] = {}
    for _, row in labels_df.iterrows():
        subject_id = normalize_subject_id(row[subject_column], subject_id_width)
        label_name = str(row[label_column]).strip().upper()
        if not subject_id:
            continue
        if label_name not in LABEL_MAP:
            raise ValueError(f"Unsupported label '{label_name}' for subject '{subject_id}'")
        labels[subject_id] = {
            "label": LABEL_MAP[label_name],
            "label_name": label_name,
        }
    return labels


def load_transcripts(transcript_json: str, subject_id_width: int) -> Dict[str, Dict[str, str]]:
    with open(transcript_json, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("Transcript JSON must be a list of subject objects.")

    transcripts: Dict[str, Dict[str, str]] = {}
    for subject in payload:
        subject_id = normalize_subject_id(subject.get("subject_id"), subject_id_width)
        audio_data = subject.get("audio_data", [])
        if not subject_id:
            continue
        if not isinstance(audio_data, list):
            raise ValueError(f"audio_data must be a list for subject '{subject_id}'")

        segments: Dict[str, str] = {}
        for item in audio_data:
            audio_index = normalize_audio_index(item.get("audio_index"))
            content = str(item.get("content", "")).strip()
            if audio_index:
                segments[audio_index] = content
        transcripts[subject_id] = segments
    return transcripts


def required_indices(num_segments: int) -> List[str]:
    return [str(index).zfill(2) for index in range(1, num_segments + 1)]


def collect_subject(
    subject_id: str,
    label_info: Dict[str, Any],
    transcript_segments: Dict[str, str],
    audio_root: Path,
    processed_text_dir: Path,
    num_segments: int,
    split: str,
) -> Tuple[Dict[str, Any] | None, str | None]:
    indices = required_indices(num_segments)
    missing_text = [index for index in indices if not transcript_segments.get(index, "").strip()]
    if missing_text:
        return None, f"missing transcript segments: {missing_text}"

    audio_paths = []
    missing_audio = []
    for index in indices:
        audio_path = audio_root / subject_id / f"{index}.wav"
        if audio_path.exists():
            audio_paths.append(str(audio_path.resolve()))
        else:
            missing_audio.append(index)
    if missing_audio:
        return None, f"missing audio segments: {missing_audio}"

    processed_text_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = processed_text_dir / f"{subject_id}.txt"
    merged_text = "\n".join(transcript_segments[index].strip() for index in indices)
    transcript_path.write_text(merged_text, encoding="utf-8")

    return (
        {
            "participant_id": subject_id,
            "label": label_info["label"],
            "label_name": label_info["label_name"],
            "transcript_path": str(transcript_path.resolve()),
            "audio_paths": json.dumps(audio_paths, ensure_ascii=False),
            "num_segments": num_segments,
            "split": split,
        },
        None,
    )


def build_manifest(args: argparse.Namespace) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    labels = load_labels(args.label_xlsx, args.subject_column, args.label_column, args.subject_id_width)
    transcripts = load_transcripts(args.transcript_json, args.subject_id_width)
    audio_root = Path(args.audio_root)
    processed_text_dir = Path(args.processed_text_dir)

    rows = []
    skipped: List[Tuple[str, str]] = []
    for subject_id, label_info in sorted(labels.items()):
        if subject_id not in transcripts:
            skipped.append((subject_id, "missing transcript subject"))
            continue

        row, reason = collect_subject(
            subject_id=subject_id,
            label_info=label_info,
            transcript_segments=transcripts[subject_id],
            audio_root=audio_root,
            processed_text_dir=processed_text_dir,
            num_segments=int(args.num_segments),
            split=str(args.split),
        )
        if reason is not None:
            skipped.append((subject_id, reason))
            continue
        rows.append(row)

    if not rows:
        raise ValueError("No complete MODMA subjects were found.")
    return pd.DataFrame(rows), skipped


def print_summary(manifest: pd.DataFrame, skipped: List[Tuple[str, str]], total_labels: int) -> None:
    print(f"Total labeled subjects: {total_labels}")
    print(f"Written subjects: {len(manifest)}")
    print(f"Skipped subjects: {len(skipped)}")
    print("Label counts:")
    print(manifest["label_name"].value_counts())

    if skipped:
        print("Skipped details:")
        for subject_id, reason in skipped:
            print(f"- {subject_id}: {reason}")


def main() -> None:
    args = parse_args()
    manifest, skipped = build_manifest(args)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)

    total_labels = len(load_labels(args.label_xlsx, args.subject_column, args.label_column, args.subject_id_width))
    print(f"Wrote {output}")
    print_summary(manifest, skipped, total_labels)


if __name__ == "__main__":
    main()
