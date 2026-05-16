from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize E-DAIC feature run metrics across output dirs.")
    parser.add_argument("run_dirs", nargs="+", help="Run directories, e.g. outputs_seed42/both.")
    parser.add_argument("--metric-file", default="metrics/test_metrics_at_0_5.json")
    return parser.parse_args()


def load_metric(run_dir: str, metric_file: str) -> Dict[str, Any]:
    path = Path(run_dir) / metric_file
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    args = parse_args()
    metrics = [load_metric(run_dir, args.metric_file) for run_dir in args.run_dirs]
    keys = ["acc", "precision", "recall", "f1", "loss", "pred_pos_rate", "prob_mean"]
    summary: Dict[str, Any] = {"runs": args.run_dirs, "metric_file": args.metric_file, "num_runs": len(metrics)}
    for key in keys:
        values: List[float] = [float(metric[key]) for metric in metrics if key in metric]
        summary[f"{key}_mean"] = mean(values) if values else None
        summary[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
