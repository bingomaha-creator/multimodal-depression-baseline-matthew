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
TEXT_AUDIO_CHUNK_CONFIG="${TEXT_AUDIO_CHUNK_CONFIG:-configs/baseline_edaic_text_audio_chunk_features.yaml}"
PLAIN_CACHE="${PLAIN_CACHE:-data/edaic_features_320000.pt}"
CHUNK_CACHE="${CHUNK_CACHE:-data/edaic_text_chunk_features_320000.pt}"
TEXT_AUDIO_CHUNK_CACHE="${TEXT_AUDIO_CHUNK_CACHE:-data/edaic_text_audio_chunk_features_20s12.pt}"

echo "== E-DAIC baseline experiment runner =="
echo "plain config: ${PLAIN_CONFIG}"
echo "chunk config: ${CHUNK_CONFIG}"
echo "text/audio chunk config: ${TEXT_AUDIO_CHUNK_CONFIG}"

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
  --output-dir runs/edaic/classification/plain_320000/main

python -m src.train_edaic_features \
  --config "${PLAIN_CONFIG}" \
  --modality audio \
  --output-dir runs/edaic/classification/plain_320000/main

python -m src.train_edaic_features \
  --config "${PLAIN_CONFIG}" \
  --modality both \
  --output-dir runs/edaic/classification/plain_320000/main

echo
echo "== Plain 320000 both: seed stability =="
# Same feature cache and model setting, different seeds.
# Use this to estimate whether the baseline is stable or mostly luck.
for seed in 42 3407 2025; do
  python -m src.train_edaic_features \
    --config "${PLAIN_CONFIG}" \
    --modality both \
    --seed "${seed}" \
    --output-dir "runs/edaic/classification/plain_320000/seed_${seed}"
done

python scripts/summarize_edaic_feature_runs.py \
  runs/edaic/classification/plain_320000/seed_42/both \
  runs/edaic/classification/plain_320000/seed_3407/both \
  runs/edaic/classification/plain_320000/seed_2025/both

echo
echo "== Plain 320000 both: class-weight ablation =="
# Class weights can improve recall but may over-predict the positive class.
# This pair checks that trade-off without changing the feature cache.
python -m src.train_edaic_features \
  --config "${PLAIN_CONFIG}" \
  --modality both \
  --use-class-weights true \
  --output-dir runs/edaic/classification/plain_320000/class_weight_true

python -m src.train_edaic_features \
  --config "${PLAIN_CONFIG}" \
  --modality both \
  --use-class-weights false \
  --output-dir runs/edaic/classification/plain_320000/class_weight_false

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
    --output-dir "runs/edaic/classification/plain_320000/lr_${lr}_dropout_0.1_hidden_256"
done

for dropout in 0.1 0.3; do
  python -m src.train_edaic_features \
    --config "${PLAIN_CONFIG}" \
    --modality both \
    --learning-rate 1.0e-4 \
    --dropout "${dropout}" \
    --hidden-dim 256 \
    --output-dir "runs/edaic/classification/plain_320000/lr_1e-4_dropout_${dropout}_hidden_256"
done

for hidden_dim in 128 256; do
  python -m src.train_edaic_features \
    --config "${PLAIN_CONFIG}" \
    --modality both \
    --learning-rate 1.0e-4 \
    --dropout 0.1 \
    --hidden-dim "${hidden_dim}" \
    --output-dir "runs/edaic/classification/plain_320000/lr_1e-4_dropout_0.1_hidden_${hidden_dim}"
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
  --output-dir runs/edaic/classification/text_chunk_320000/main

python -m src.train_edaic_features \
  --config "${CHUNK_CONFIG}" \
  --modality both \
  --output-dir runs/edaic/classification/text_chunk_320000/main

echo
echo "== Text chunk + uniform audio chunk feature extraction =="
# Text/audio chunk cache:
# - Text uses full-transcript 512-token chunks.
# - Audio uses evenly spaced 20-second chunks over the full waveform.
# - The saved cache is still compatible with src.train_edaic_features.
if [[ ! -f "${TEXT_AUDIO_CHUNK_CACHE}" ]]; then
  python -m src.extract_edaic_text_audio_chunk_features --config "${TEXT_AUDIO_CHUNK_CONFIG}"
else
  echo "Found ${TEXT_AUDIO_CHUNK_CACHE}; skip text/audio chunk feature extraction."
fi

echo
echo "== Text chunk + uniform audio chunk baseline: text/audio/both =="
python -m src.train_edaic_features \
  --config "${TEXT_AUDIO_CHUNK_CONFIG}" \
  --modality text \
  --output-dir runs/edaic/classification/text_audio_chunk_20s12/main

python -m src.train_edaic_features \
  --config "${TEXT_AUDIO_CHUNK_CONFIG}" \
  --modality audio \
  --output-dir runs/edaic/classification/text_audio_chunk_20s12/main

python -m src.train_edaic_features \
  --config "${TEXT_AUDIO_CHUNK_CONFIG}" \
  --modality both \
  --output-dir runs/edaic/classification/text_audio_chunk_20s12/main

echo
echo "== Text chunk + uniform audio chunk both: class-weight ablation =="
python -m src.train_edaic_features \
  --config "${TEXT_AUDIO_CHUNK_CONFIG}" \
  --modality both \
  --use-class-weights true \
  --output-dir runs/edaic/classification/text_audio_chunk_20s12/class_weight_true

python -m src.train_edaic_features \
  --config "${TEXT_AUDIO_CHUNK_CONFIG}" \
  --modality both \
  --use-class-weights false \
  --output-dir runs/edaic/classification/text_audio_chunk_20s12/class_weight_false

echo
echo "Done. Main outputs:"
echo "- runs/edaic/classification/plain_320000"
echo "- runs/edaic/classification/text_chunk_320000"
echo "- runs/edaic/classification/text_audio_chunk_20s12"
