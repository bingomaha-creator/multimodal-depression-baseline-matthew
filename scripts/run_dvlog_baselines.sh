#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/baseline_dvlog.yaml}"
DEVICE="${2:-auto}"
SEEDS=(42 2025 3407)
MODELS=(mlp bigru)
MODALITIES=(audio visual both)

for model in "${MODELS[@]}"; do
  for modality in "${MODALITIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "Running model=${model} modality=${modality} seed=${seed}"
      python -m src.train_dvlog \
        --config "${CONFIG}" \
        --model "${model}" \
        --modality "${modality}" \
        --seed "${seed}" \
        --device "${DEVICE}"
    done
  done
done

python scripts/summarize_dvlog_runs.py \
  --runs-root runs/D-vlog \
  --seeds "${SEEDS[@]}"
