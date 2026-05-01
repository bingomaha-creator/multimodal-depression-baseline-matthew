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

By default, the script reads `folder_name` as the subject folder id and `PHQ_Score` as the PHQ score. It accepts `train`, `dev`, `val`, `validation`, and `test` split folders. `val` and `validation` are normalized to `dev` in the generated manifest. If `label_all.csv` also has a `split` column, the script checks that it matches the actual folder split.

## Configure

Edit [configs/baseline_edaic.yaml](configs/baseline_edaic.yaml) and set:

```yaml
data:
  manifest_path: /path/to/manifest.csv
```

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
3. Evaluate on `dev` after each epoch.
4. Save the best checkpoint by `dev_f1`.
5. Load the best checkpoint at the end.
6. Evaluate on `test`.
7. Save metrics and test predictions.

Default outputs:

```text
outputs/checkpoints/best.pt
outputs/metrics/dev_best_metrics.json
outputs/metrics/test_metrics.json
outputs/predictions/test_predictions.csv
```

## Smoke Test

After the manifest is ready, run a very short training pass to verify model downloads, data paths, audio loading, and forward/backward passes:

```bash
python -m src.train --config configs/baseline_edaic.yaml --max-train-steps 2
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
