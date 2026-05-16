#!/usr/bin/env bash
set -euo pipefail

# This script runs the E-DAIC baseline experiment checklist end to end.
# It is intended for the server environment after the feature cache paths in
# the YAML configs are valid. If you want to run commands one by one instead,
# see scripts/edaic_manual_commands.txt.

# Override these from the shell if needed, for example:
# PLAIN_CONFIG=configs/other.yaml bash scripts/run_edaic_baseline_experiments.sh
PLAIN_CONFIG="${PLAIN_CONFIG:-configs/baseline_edaic_features.yaml}"
CHUNK_CONFIG="${CHUNK_CONFIG:-configs/baseline_edaic_text_chunk_features.yaml}"
PLAIN_CACHE="${PLAIN_CACHE:-data/edaic_features_320000.pt}"
CHUNK_CACHE="${CHUNK_CACHE:-data/edaic_text_chunk_features_320000.pt}"

echo "== E-DAIC baseline experiment runner =="
echo "plain config: ${PLAIN_CONFIG}"
echo "chunk config: ${CHUNK_CONFIG}"

# Plain feature cache:
# - Uses the original plain feature pipeline.
# - Text is the first 512-token RoBERTa representation.
# - Audio is wav2vec2 from max_audio_length=320000.
if [[ ! -f "${PLAIN_CACHE}" ]]; then
  echo "Missing ${PLAIN_CACHE}; extracting plain 320000 features first."
  python -m src.extract_edaic_features --config "${PLAIN_CONFIG}"
else
  echo "Found ${PLAIN_CACHE}; skip plain feature extraction."
fi

echo
echo "== Plain 320000 baseline: text/audio/both =="
# Basic modality ablation for the current main baseline.
python -m src.train_edaic_features \
  --config "${PLAIN_CONFIG}" \
  --modality text \
  --output-dir outputs_edaic_features_320000_baseline

python -m src.train_edaic_features \
  --config "${PLAIN_CONFIG}" \
  --modality audio \
  --output-dir outputs_edaic_features_320000_baseline

python -m src.train_edaic_features \
  --config "${PLAIN_CONFIG}" \
  --modality both \
  --output-dir outputs_edaic_features_320000_baseline

echo
echo "== Plain 320000 both: seed stability =="
# Same feature cache and model setting, different seeds.
# Use this to estimate whether the baseline is stable or mostly luck.
for seed in 42 3407 2025; do
  python -m src.train_edaic_features \
    --config "${PLAIN_CONFIG}" \
    --modality both \
    --seed "${seed}" \
    --output-dir "outputs_edaic_features_320000_seed${seed}"
done

python scripts/summarize_edaic_feature_runs.py \
  outputs_edaic_features_320000_seed42/both \
  outputs_edaic_features_320000_seed3407/both \
  outputs_edaic_features_320000_seed2025/both

echo
echo "== Plain 320000 both: class-weight ablation =="
# Class weights can improve recall but may over-predict the positive class.
# This pair checks that trade-off without changing the feature cache.
python -m src.train_edaic_features \
  --config "${PLAIN_CONFIG}" \
  --modality both \
  --use-class-weights true \
  --output-dir outputs_edaic_features_320000_class_weights_true

python -m src.train_edaic_features \
  --config "${PLAIN_CONFIG}" \
  --modality both \
  --use-class-weights false \
  --output-dir outputs_edaic_features_320000_class_weights_false

echo
echo "== Plain 320000 both: small hyperparameter checks =="
# Small dev-side checks for the lightweight MLP head.
# Keep this small to avoid turning the baseline into heavy tuning.
for lr in 5.0e-5 1.0e-4 3.0e-4; do
  python -m src.train_edaic_features \
    --config "${PLAIN_CONFIG}" \
    --modality both \
    --learning-rate "${lr}" \
    --dropout 0.1 \
    --hidden-dim 256 \
    --output-dir "outputs_edaic_features_320000_lr${lr}_dropout0.1_hidden256"
done

for dropout in 0.1 0.3; do
  python -m src.train_edaic_features \
    --config "${PLAIN_CONFIG}" \
    --modality both \
    --learning-rate 1.0e-4 \
    --dropout "${dropout}" \
    --hidden-dim 256 \
    --output-dir "outputs_edaic_features_320000_lr1e-4_dropout${dropout}_hidden256"
done

for hidden_dim in 128 256; do
  python -m src.train_edaic_features \
    --config "${PLAIN_CONFIG}" \
    --modality both \
    --learning-rate 1.0e-4 \
    --dropout 0.1 \
    --hidden-dim "${hidden_dim}" \
    --output-dir "outputs_edaic_features_320000_lr1e-4_dropout0.1_hidden${hidden_dim}"
done

echo
echo "== Text chunk feature extraction =="
# Text chunk feature cache:
# - Reads the full transcript.
# - Splits it into up to max_text_chunks chunks of 512 tokens.
# - Mean-pools chunk-level RoBERTa embeddings.
# - Keeps the same 320000 audio feature setting.
if [[ ! -f "${CHUNK_CACHE}" ]]; then
  python -m src.extract_edaic_text_chunk_features --config "${CHUNK_CONFIG}"
else
  echo "Found ${CHUNK_CACHE}; skip text chunk feature extraction."
fi

echo
echo "== Text chunk baseline: text/both =="
# Check whether full-transcript chunk text improves over first-512-token text.
python -m src.train_edaic_features \
  --config "${CHUNK_CONFIG}" \
  --modality text \
  --output-dir outputs_edaic_text_chunk_features_320000

python -m src.train_edaic_features \
  --config "${CHUNK_CONFIG}" \
  --modality both \
  --output-dir outputs_edaic_text_chunk_features_320000

echo
echo "Done. Main outputs:"
echo "- outputs_edaic_features_320000_baseline"
echo "- outputs_edaic_features_320000_seed*"
echo "- outputs_edaic_features_320000_class_weights_*"
echo "- outputs_edaic_features_320000_lr*"
echo "- outputs_edaic_text_chunk_features_320000"
