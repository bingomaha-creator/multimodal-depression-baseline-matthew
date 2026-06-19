from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


MODEL_ORDER = {"mlp": 0, "bigru": 1}
MODALITY_ORDER = {"audio": 0, "visual": 1, "both": 2}
METRICS = ("acc", "precision", "recall", "f1")


def collect_runs(
    root: str | Path,
    expected_seeds: Sequence[int],
    require_complete: bool = True,
) -> pd.DataFrame:
    root = Path(root)
    rows = []
    missing = []
    for model in MODEL_ORDER:
        for modality in MODALITY_ORDER:
            for seed in expected_seeds:
                path = root / model / modality / f"seed_{seed}" / "metrics" / "test_metrics_at_0_5.json"
                if not path.is_file():
                    if require_complete:
                        missing.append(str(path))
                    continue
                with path.open("r", encoding="utf-8") as file:
                    metrics = json.load(file)
                absent_metrics = [metric for metric in METRICS if metric not in metrics]
                if absent_metrics:
                    raise ValueError(f"Missing metrics {absent_metrics} in {path}")
                rows.append(
                    {
                        "model": model,
                        "modality": modality,
                        "seed": int(seed),
                        **{metric: float(metrics[metric]) for metric in METRICS},
                    }
                )
    if missing:
        examples = "\n".join(missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} expected D-Vlog runs. Examples:\n{examples}")
    if not rows:
        raise ValueError(f"No D-Vlog test metric files found under {root}")
    return pd.DataFrame(rows)


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, modality), group in runs.groupby(["model", "modality"], sort=False):
        row = {
            "model": model,
            "modality": modality,
            "num_seeds": int(len(group)),
        }
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["model", "modality"],
        key=lambda values: values.map(MODEL_ORDER if values.name == "model" else MODALITY_ORDER),
    ).reset_index(drop=True)


def format_markdown(summary: pd.DataFrame) -> str:
    columns = ["model", "modality", "num_seeds"] + [
        name for metric in METRICS for name in (f"{metric}_mean", f"{metric}_std")
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in summary[columns].itertuples(index=False, name=None):
        values = []
        for value in row:
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join(["# D-Vlog Baseline Summary", "", header, separator, *body, ""])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize D-Vlog baseline test metrics.")
    parser.add_argument("--runs-root", default="runs/D-vlog")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2025, 3407])
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.runs_root)
    runs = collect_runs(root, args.seeds, require_complete=not args.allow_incomplete)
    summary = summarize_runs(runs)
    output_dir = root / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dvlog_baselines_summary.csv"
    markdown_path = output_dir / "dvlog_baselines_summary.md"
    summary.to_csv(csv_path, index=False)
    markdown_path.write_text(format_markdown(summary), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote {csv_path} and {markdown_path}")


if __name__ == "__main__":
    main()
