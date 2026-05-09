# E-DAIC RoBERTa + wav2vec2 Baseline

This repository is a minimal PyTorch baseline for multimodal depression detection on E-DAIC/DAIC-style interview data. It follows the simplest structure from the paper: RoBERTa for transcripts, wav2vec2 for audio, feature concatenation, and an MLP classifier for PHQ-based binary depression detection.

The training command runs the full train, validation, and test flow. It saves the best checkpoint by validation F1 and automatically evaluates that checkpoint on the test split.

## Environment

Create and activate a Python environment on your server, then install dependencies:

```bash
pip install -r requirements.txt
```

If your server uses CUDA, install the PyTorch build that matches your CUDA version first, then run the command above for the remaining packages.

## Data Manifest

Prepare one CSV manifest with these columns:

```csv
participant_id,audio_path,transcript_path,phq_score,split
300,/path/to/audio/300.wav,/path/to/transcripts/300.txt,4,train
301,/path/to/audio/301.wav,/path/to/transcripts/301.txt,13,dev
302,/path/to/audio/302.wav,/path/to/transcripts/302.txt,9,test
```

Required columns:

- `participant_id`: subject/interview id.
- `audio_path`: path to the interview audio file.
- `transcript_path`: path to a text transcript file.
- `phq_score`: PHQ-8/PHQ-9 score.
- `split`: one of `train`, `dev`, or `test`.

Labels are created automatically:

- `1`: `phq_score >= 10`
- `0`: `phq_score < 10`

Paths can be absolute or relative to the directory where you run the command.

If your data is already split into E-DAIC-style folders like this:

```text
dataset_root/
  train/
    300_P/
      300_AUDIO.wav
      300_Transcript.csv
  dev/
    301_P/
      301_AUDIO.wav
      301_Transcript.csv
  test/
    302_P/
      302_AUDIO.wav
      302_Transcript.csv
```

and each transcript CSV contains a `Text` column:

```csv
Start_Time,End_Time,Text,Confidence
14.3,15.1,so I'm going to,0.934209526
```

create or reuse a labels CSV:

```csv
Participant_ID,Gender,PHQ_Binary,PHQ_Score,PCL-C (PTSD),PTSD Severity,split,folder_name
302,male,0,4,0,28.0,train,302_P
303,female,0,0,0,17.0,train,303_P
```

Then generate the manifest automatically:

```bash
python scripts/make_manifest_example.py \
  --dataset-root /path/to/E_DAIC/output \
  --labels-csv /path/to/E_DAIC/output/label_all.csv \
  --output data/edaic_manifest.csv
```

By default, the script reads `folder_name` as the subject folder id and `PHQ_Score` as the PHQ score. It accepts `train`, `dev`, `val`, `valid`, `validation`, and `test` split folders. `val`, `valid`, and `validation` are normalized to `dev` in the generated manifest. If `label_all.csv` also has a `split` column, the script checks that it matches the actual folder split.

## MODMA Manifest

For MODMA, one subject is one sample and the first 18 audio/text segments are required. Generate a MODMA manifest with:

```bash
python scripts/make_manifest_modma.py \
  --transcript-json /path/to/modma_transcripts.json \
  --label-xlsx /path/to/labels.xlsx \
  --audio-root /path/to/MODMA/audio \
  --subject-id-width 8 \
  --output data/modma_manifest.csv \
  --processed-text-dir data/modma_processed_texts
```

Expected inputs:

- Transcript JSON: one top-level list with `subject_id`, `audio_data`, `audio_index`, and `content`.
- Audio files: `{audio_root}/{subject_id}/01.wav` through `{audio_root}/{subject_id}/18.wav`.
- Label xlsx: columns `subject id` and `type`, where `MDD` maps to `1` and `HC` maps to `0`.
- Subject ids are zero-padded to 8 digits by default, so Excel values like `2030010` become `02030010`.

The generated manifest contains:

```csv
participant_id,label,label_name,transcript_path,audio_paths,num_segments,split
```

`audio_paths` is a JSON list string containing 18 absolute wav paths. Subjects missing any of the first 18 audio or transcript segments are skipped with a warning. The `split` value is written as `all`; five-fold training will create train/dev/test folds from this manifest later.

Check the generated manifest with:

```bash
python - <<'PY'
import json
import pandas as pd

df = pd.read_csv("data/modma_manifest.csv", dtype={"participant_id": str})
print(df.head())
print(df["label_name"].value_counts())
print(df["num_segments"].value_counts())
paths = json.loads(df.iloc[0]["audio_paths"])
print(len(paths), paths[:2], paths[-1])
PY
```

## Configure

Edit [configs/baseline_edaic.yaml](configs/baseline_edaic.yaml) and set:

```yaml
data:
  manifest_path: /path/to/manifest.csv
```

If RoBERTa and wav2vec2 are already downloaded on your server, point the config to the local model directories:

```yaml
model:
  text_model_name: /path/to/roberta-base
  audio_model_name: /path/to/wav2vec2-base
```

The local RoBERTa directory should contain files such as `config.json`, tokenizer files, and model weights. The local wav2vec2 directory should contain `config.json`, `preprocessor_config.json`, and model weights. The code uses HuggingFace `from_pretrained`, so local directories work the same way as online model names.

The default model freezes RoBERTa and wav2vec2, which is safer for small datasets and limited GPU memory. To fine-tune the backbones, set:

```yaml
model:
  freeze_backbones: false
```

## Train, Validate, and Test

Run the complete pipeline:

```bash
python -m src.train --config configs/baseline_edaic.yaml
```

The script will:

1. Load `train`, `dev`, and `test` splits.
2. Train for the configured number of epochs.
3. Evaluate on `dev` after each epoch with the fixed `0.5` threshold.
4. Save the best checkpoint by the fixed-threshold dev metric.
5. Load the best checkpoint at the end.
6. Search a best dev threshold once for analysis.
7. Evaluate `test` with both `0.5` and the dev-selected analysis threshold.
8. Save metrics and dev/test predictions.

Default outputs:

```text
outputs/checkpoints/best.pt
outputs/metrics/dev_best_metrics.json
outputs/metrics/dev_metrics_at_0_5.json
outputs/metrics/dev_metrics_best_threshold.json
outputs/metrics/test_metrics_at_0_5.json
outputs/metrics/test_metrics_with_dev_threshold.json
outputs/predictions/dev_predictions.csv
outputs/predictions/dev_predictions_with_best_threshold.csv
outputs/predictions/test_predictions.csv
outputs/predictions/test_predictions_with_dev_threshold.csv
```

For low-memory GPUs, start with:

```yaml
data:
  max_audio_length: 80000
  num_workers: 0

training:
  batch_size: 8
  device: cuda
```

When `model.freeze_backbones: true`, frozen RoBERTa and wav2vec2 forwards run under `torch.no_grad()` to reduce memory. If this fits, you can try increasing `max_audio_length` to `120000` or `160000`.

## Smoke Test

After the manifest is ready, run a very short training pass to verify model downloads, data paths, audio loading, and forward/backward passes:

```bash
python -m src.train --config configs/baseline_edaic.yaml --max-train-steps 2
```

For MODMA, first extract frozen features:

```bash
python -m src.extract_modma_features \
  --config configs/baseline_modma.yaml \
  --limit 2 \
  --output data/modma_features_smoke.pt
```

If the smoke test works, remove `--limit` and cache all features:

```bash
python -m src.extract_modma_features --config configs/baseline_modma.yaml
```

Then run one fold for two train steps:

```bash
python -m src.train_modma_cv \
  --config configs/baseline_modma.yaml \
  --fold-limit 1 \
  --max-train-steps 2
```

Before the full multimodal run, check each modality separately:

```bash
python -m src.train_modma_cv --config configs/baseline_modma.yaml --modality text --fold-limit 1
python -m src.train_modma_cv --config configs/baseline_modma.yaml --modality audio --fold-limit 1
python -m src.train_modma_cv --config configs/baseline_modma.yaml --modality both
```

## Optional Checkpoint Evaluation

The main training flow already tests the best checkpoint. The separate evaluator is only for later checkpoint inspection:

```bash
python -m src.evaluate \
  --config configs/baseline_edaic.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --split test
```

## Notes

- Audio is resampled to 16 kHz.
- Audio longer than `max_audio_length` is truncated.
- Audio shorter than `max_audio_length` is zero-padded.
- Transcript text is tokenized with `roberta-base` and truncated to `max_text_length`.
- The first RoBERTa token representation is used as the text embedding.
- wav2vec2 hidden states are mean-pooled with the audio attention mask.
- Transcript CSV files are supported; by default the dataset reads and concatenates the `Text` column. Change `data.transcript_text_column` in the config if your column name differs.
- Class weights are enabled by default with `training.use_class_weights: true`.
- Fixed threshold `0.5` is the main baseline evaluation. Dev threshold search runs only after checkpoint selection as an analysis view.
