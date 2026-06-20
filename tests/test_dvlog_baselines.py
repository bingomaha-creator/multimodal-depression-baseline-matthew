from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.summarize_dvlog_runs import collect_runs, summarize_runs
from src.datasets.dvlog_dataset import (
    DVlogDataset,
    FeatureNormalizer,
    collate_dvlog_pooled,
    collate_dvlog_temporal,
    discover_dvlog_samples,
    load_feature_pair,
    summarize_sequence,
    validate_dvlog_samples,
)
from src.models.dvlog_baselines import DVlogBiGRU, DVlogMLP
from src.train_dvlog import compute_class_weights, is_better_checkpoint, split_samples


def write_dataset(root: Path, rows: list[dict], length: int = 4) -> None:
    pd.DataFrame(rows).to_csv(root / "labels.csv", index=False)
    for row in rows:
        sample_id = str(row["index"])
        sample_dir = root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        base = float(int(sample_id) + 1)
        audio = np.full((length, 25), base, dtype=np.float32)
        visual = np.full((length, 136), base, dtype=np.float32)
        np.save(sample_dir / f"{sample_id}_acoustic.npy", audio)
        np.save(sample_dir / f"{sample_id}_visual.npy", visual)


def basic_rows() -> list[dict]:
    return [
        {"index": 2, "label": "depression", "duration": 4.1, "gender": "f", "fold": "test"},
        {"index": 0, "label": "depression", "duration": 4.0, "gender": "f", "fold": "train"},
        {"index": 1, "label": "normal", "duration": 4.0, "gender": "m", "fold": "valid"},
    ]


def test_discovery_sorts_numeric_ids_and_maps_labels(tmp_path: Path) -> None:
    write_dataset(tmp_path, basic_rows())

    samples = discover_dvlog_samples(tmp_path)

    assert [sample.sample_id for sample in samples] == ["0", "1", "2"]
    assert [sample.label for sample in samples] == [1, 0, 1]
    assert samples[0].acoustic_path == tmp_path / "0" / "0_acoustic.npy"
    assert samples[0].visual_path == tmp_path / "0" / "0_visual.npy"


def test_validation_reports_split_and_class_counts(tmp_path: Path) -> None:
    write_dataset(tmp_path, basic_rows())

    summary = validate_dvlog_samples(discover_dvlog_samples(tmp_path))

    assert summary["split_counts"] == {"train": 1, "valid": 1, "test": 1}
    assert summary["label_counts"] == {0: 1, 1: 2}


def test_load_pair_rejects_length_mismatch(tmp_path: Path) -> None:
    write_dataset(tmp_path, basic_rows())
    np.save(tmp_path / "0" / "0_visual.npy", np.ones((3, 136), dtype=np.float32))
    sample = discover_dvlog_samples(tmp_path)[0]

    with pytest.raises(ValueError, match="length mismatch.*sample 0"):
        load_feature_pair(sample)


def test_load_pair_rejects_non_finite_values(tmp_path: Path) -> None:
    write_dataset(tmp_path, basic_rows())
    audio = np.ones((4, 25), dtype=np.float32)
    audio[0, 0] = np.nan
    np.save(tmp_path / "0" / "0_acoustic.npy", audio)
    sample = discover_dvlog_samples(tmp_path)[0]

    with pytest.raises(ValueError, match="non-finite.*sample 0"):
        load_feature_pair(sample)


def test_visual_zero_rows_are_masked(tmp_path: Path) -> None:
    write_dataset(tmp_path, basic_rows())
    visual = np.ones((4, 136), dtype=np.float32)
    visual[1] = 0.0
    np.save(tmp_path / "0" / "0_visual.npy", visual)

    _, _, mask = load_feature_pair(discover_dvlog_samples(tmp_path)[0])

    assert mask.tolist() == [True, False, True, True]


def test_normalizer_uses_only_passed_training_samples(tmp_path: Path) -> None:
    write_dataset(tmp_path, basic_rows())
    samples = discover_dvlog_samples(tmp_path)
    train = [sample for sample in samples if sample.split == "train"]

    normalizer = FeatureNormalizer.fit(train)

    assert np.allclose(normalizer.audio_mean, 1.0)
    assert np.allclose(normalizer.visual_mean, 1.0)


def test_normalizer_restores_missing_visual_rows_to_zero(tmp_path: Path) -> None:
    write_dataset(tmp_path, basic_rows())
    visual = np.full((4, 136), 2.0, dtype=np.float32)
    visual[1] = 0.0
    np.save(tmp_path / "0" / "0_visual.npy", visual)
    sample = discover_dvlog_samples(tmp_path)[0]
    _, raw_visual, mask = load_feature_pair(sample)
    normalizer = FeatureNormalizer.fit([sample])

    transformed = normalizer.transform_visual(raw_visual, mask)

    assert np.all(transformed[1] == 0.0)


def test_summarize_sequence_ignores_masked_rows() -> None:
    values = np.array([[1.0, 2.0], [100.0, 100.0], [3.0, 4.0]], dtype=np.float32)
    mask = np.array([True, False, True])

    pooled = summarize_sequence(values, mask)

    assert np.allclose(pooled, [2.0, 3.0, 1.0, 1.0])


def test_collators_produce_expected_shapes(tmp_path: Path) -> None:
    rows = basic_rows()[:2]
    write_dataset(tmp_path, rows, length=3)
    samples = discover_dvlog_samples(tmp_path)
    normalizer = FeatureNormalizer.fit([samples[0]])
    pooled = DVlogDataset(samples, normalizer, representation="pooled")
    temporal = DVlogDataset(samples, normalizer, representation="temporal")

    pooled_batch = collate_dvlog_pooled([pooled[0], pooled[1]])
    temporal_batch = collate_dvlog_temporal([temporal[0], temporal[1]])

    assert pooled_batch["audio_embeddings"].shape == (2, 50)
    assert pooled_batch["visual_embeddings"].shape == (2, 272)
    assert temporal_batch["audio"].shape == (2, 3, 25)
    assert temporal_batch["visual"].shape == (2, 3, 136)
    assert temporal_batch["visual_mask"].dtype == torch.bool


@pytest.mark.parametrize("modality", ["audio", "visual", "both"])
def test_mlp_forward_shape(modality: str) -> None:
    model = DVlogMLP(modality=modality)
    logits = model(torch.randn(2, 50), torch.randn(2, 272))
    assert logits.shape == (2, 2)


@pytest.mark.parametrize("modality", ["audio", "visual", "both"])
def test_bigru_forward_shape_and_padding_invariance(modality: str) -> None:
    torch.manual_seed(7)
    model = DVlogBiGRU(modality=modality, projection_dim=8, hidden_dim=6, dropout=0.0).eval()
    audio = torch.randn(2, 4, 25)
    visual = torch.randn(2, 4, 136)
    lengths = torch.tensor([4, 2])
    visual_mask = torch.arange(4).unsqueeze(0) < lengths.unsqueeze(1)
    padded_audio = torch.cat([audio, torch.zeros(2, 3, 25)], dim=1)
    padded_visual = torch.cat([visual, torch.zeros(2, 3, 136)], dim=1)
    padded_mask = torch.cat([visual_mask, torch.zeros(2, 3, dtype=torch.bool)], dim=1)

    with torch.no_grad():
        logits = model(audio, visual, lengths, visual_mask)
        padded_logits = model(padded_audio, padded_visual, lengths, padded_mask)

    assert logits.shape == (2, 2)
    assert torch.allclose(logits, padded_logits, atol=1e-6)


def test_split_samples_uses_official_folds(tmp_path: Path) -> None:
    write_dataset(tmp_path, basic_rows())
    splits = split_samples(discover_dvlog_samples(tmp_path))
    assert {name: [sample.sample_id for sample in values] for name, values in splits.items()} == {
        "train": ["0"],
        "valid": ["1"],
        "test": ["2"],
    }


def test_class_weights_balance_training_labels() -> None:
    weights = compute_class_weights([0, 0, 0, 1], torch.device("cpu"))
    assert torch.allclose(weights, torch.tensor([2.0 / 3.0, 2.0]))


def test_checkpoint_tie_prefers_lower_loss() -> None:
    assert is_better_checkpoint(0.7, 0.4, best_f1=0.7, best_loss=0.5)
    assert not is_better_checkpoint(0.7, 0.6, best_f1=0.7, best_loss=0.5)


def test_summary_aggregates_population_statistics(tmp_path: Path) -> None:
    for seed, f1 in [(42, 0.6), (2025, 0.7), (3407, 0.8)]:
        path = tmp_path / "mlp" / "audio" / f"seed_{seed}" / "metrics"
        path.mkdir(parents=True)
        (path / "test_metrics_at_0_5.json").write_text(
            json.dumps({"acc": f1, "precision": f1, "recall": f1, "f1": f1}),
            encoding="utf-8",
        )

    runs = collect_runs(tmp_path, expected_seeds=[42, 2025, 3407], require_complete=False)
    summary = summarize_runs(runs)

    assert len(summary) == 1
    assert summary.iloc[0]["f1_mean"] == pytest.approx(0.7)
    assert summary.iloc[0]["f1_std"] == pytest.approx(np.std([0.6, 0.7, 0.8], ddof=0))


def test_validate_data_cli_with_local_sample(tmp_path: Path) -> None:
    write_dataset(tmp_path, basic_rows())
    config = tmp_path / "config.yaml"
    config.write_text(
        "data:\n"
        f"  dataset_root: {tmp_path}\n"
        "  num_workers: 0\n"
        "training:\n"
        "  output_dir: runs/D-vlog\n"
        "  device: cpu\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "src.train_dvlog", "--config", str(config), "--validate-data"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '"train": 1' in result.stdout
