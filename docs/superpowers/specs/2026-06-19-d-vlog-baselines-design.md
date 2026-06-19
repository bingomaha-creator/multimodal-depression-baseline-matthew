# D-Vlog MLP and BiGRU Baselines

## Goal

Build a reproducible first-pass benchmark for the released D-Vlog acoustic and visual feature sequences. The benchmark must establish whether the released features are learnable with a simple global-summary model and whether preserving temporal order improves performance.

The benchmark will train two model families (MLP and BiGRU), each in three modality settings (audio, visual, and both), using the official train/valid/test split and three random seeds.

DepMamba reproduction is explicitly out of scope for this first pass. It can be added after these baselines validate the data and evaluation pipeline.

## Data Contract

The server dataset root is `/home/rui/24zbma/data/D-vlog`.

- `labels.csv` contains 961 rows with columns `index`, `label`, `duration`, `gender`, and `fold`.
- Sample IDs are the integers 0 through 960.
- Each sample has `{id}_acoustic.npy` with shape `(T, 25)` and `{id}_visual.npy` with shape `(T, 136)`.
- The two modalities must have the same `T` for a sample. A mismatch is a data error, not something to silently crop.
- Labels map `normal -> 0` and `depression -> 1`.
- Official splits contain 647 train, 102 valid, and 212 test samples.
- `duration` and `gender` are metadata only and must not be model inputs. Gender is retained for optional subgroup reporting.

The observed sample 0 has 823 time steps for a duration of 823.31 seconds, indicating approximately one feature vector per second. The visual dimension is consistent with 68 two-dimensional facial landmarks, and the acoustic dimension with the 25 OpenSMILE low-level descriptors documented for D-Vlog.

## Preprocessing

All preprocessing statistics are fitted on the training split only and reused unchanged for validation and test.

1. Load each array as `float32` and reject empty arrays, non-2D arrays, wrong feature dimensions, non-finite values, missing files, or cross-modal length mismatches.
2. Treat all-zero visual rows as missing detections. Fit visual normalization using valid visual rows only. After normalization, set missing visual rows back to zero and retain a visual-validity mask.
3. Fit independent per-feature mean and standard deviation for audio and visual data. Clamp near-zero standard deviations to one.
4. Never use `duration` as a feature. Sequence length remains necessary for padding masks and packed recurrent computation.
5. Pad sequences only within a batch. Preserve full released sequences; do not truncate or randomly crop in the primary baseline.

## Model 1: Statistical-Pooling MLP

For each selected modality, compute the mean and standard deviation across valid time steps. Visual pooling excludes all-zero missing-detection rows. Audio pooling uses every released row.

- Audio embedding: `25 mean + 25 std = 50` dimensions.
- Visual embedding: `136 mean + 136 std = 272` dimensions.
- Both embedding: concatenate audio and visual embeddings into 322 dimensions.

The classifier applies a normalization layer, a 256-unit linear layer, ReLU, dropout 0.2, and a two-class output layer. This model deliberately discards temporal order and serves as the sanity baseline.

## Model 2: Temporal BiGRU

Use an independent one-layer bidirectional GRU encoder for each selected modality.

- Project acoustic and visual inputs independently to a common 128-dimensional input space.
- Use hidden size 128 per direction.
- Use packed padded sequences so batch padding does not affect recurrent states.
- Apply masked mean and masked max pooling to the BiGRU outputs. Visual pooling excludes missing-detection frames; audio pooling excludes batch padding only.
- In the `both` setting, concatenate the pooled audio and visual representations.
- Classify with a 256-unit linear layer, ReLU, dropout 0.2, and a two-class output layer.

The modalities have separate encoders; fusion happens after temporal encoding. This keeps the baseline understandable and isolates temporal modeling from more advanced cross-modal fusion methods such as DepMamba.

## Training and Model Selection

Run all six settings:

| Model | Audio | Visual | Both |
| --- | --- | --- | --- |
| MLP | yes | yes | yes |
| BiGRU | yes | yes | yes |

Use seeds 42, 2025, and 3407 for every setting. Train with AdamW, learning rate `1e-4`, weight decay `1e-4`, batch size 16, and class-weighted cross-entropy computed from the training split. Train for at most 120 epochs with early stopping patience 15, selecting the checkpoint with the highest validation F1 at the fixed 0.5 probability threshold. Resolve exact ties by lower validation loss.

The fixed 0.5 threshold is the primary result. A validation-selected threshold may be saved as a clearly labeled secondary analysis, but it must not replace the fixed-threshold headline result.

## Evaluation and Outputs

Evaluate the selected checkpoint once on the test split for each seed. Report Accuracy, Precision, Recall, and F1 with `depression` as the positive class. Aggregate test metrics as mean and population standard deviation across the three seeds.

Store runs under `runs/D-vlog/{model}/{modality}/seed_{seed}/` with:

- `checkpoints/best.pt`
- `metrics/valid_metrics_at_0_5.json`
- `metrics/test_metrics_at_0_5.json`
- `predictions/valid_predictions_at_0_5.csv`
- `predictions/test_predictions_at_0_5.csv`

Write a combined summary CSV and Markdown table under `runs/D-vlog/metrics/`. Prediction rows contain sample ID, true label, predicted label, depression probability, gender, and duration. Gender and duration support audits only.

## Configuration and Commands

Add one YAML configuration with `/home/rui/24zbma/data/D-vlog` as the default dataset root and all hyperparameters above. Provide separate command-line entry points, or one entry point with a required model selector, supporting:

- a data-validation-only command;
- a smoke test limited to a few samples or optimizer steps;
- one model/modality/seed run;
- the complete six-setting, three-seed experiment suite.

## Error Handling

Fail early with sample IDs and paths when a feature file is missing, an array has the wrong shape, modality lengths differ, a split or label is unknown, or a sample has no valid visual frames. Print the discovered split and class counts before training. Refuse to evaluate test data unless a validation-selected checkpoint exists.

## Verification

Automated tests will cover:

- label mapping and official split counts;
- sample discovery and numeric ID ordering;
- train-only normalization and visual missing-frame masking;
- statistical-pooling dimensions for all modalities;
- padded BiGRU batching and invariance to extra batch padding;
- forward-pass shapes for all six model/modality combinations;
- metric serialization and summary aggregation;
- failures for missing files, wrong feature dimensions, non-finite values, and length mismatches.

Before a full server run, execute data validation and a short CPU/GPU smoke test. The full run is accepted when all 18 seed-level jobs finish, every run produces its checkpoint, metrics, and predictions, and the combined summary includes all six settings.
