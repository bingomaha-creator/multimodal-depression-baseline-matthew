# D-Vlog MLP and BiGRU Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify MLP and BiGRU depression-classification baselines for all audio, visual, and fused D-Vlog settings on the official split.

**Architecture:** A focused dataset module discovers released features, validates the official metadata, fits train-only normalizers, and supplies either pooled or padded temporal batches. A model module exposes statistical-pooling MLP and separate-encoder BiGRU classifiers behind one training entry point. Small orchestration and summary scripts run the 18 experiments and aggregate fixed-threshold test metrics.

**Tech Stack:** Python 3, NumPy, pandas, PyTorch, scikit-learn metrics, PyYAML, pytest, shell orchestration.

---

## File Map

- Create `src/datasets/dvlog_dataset.py`: discovery, validation, train-only normalization, sample loading, pooled and temporal collation.
- Create `src/models/dvlog_baselines.py`: statistical-pooling MLP and modality-aware BiGRU.
- Create `src/train_dvlog.py`: fixed-split training, early stopping, evaluation, checkpointing, predictions, and metrics.
- Create `configs/baseline_dvlog.yaml`: server paths and baseline hyperparameters.
- Create `scripts/run_dvlog_baselines.sh`: six settings times three seeds.
- Create `scripts/summarize_dvlog_runs.py`: aggregate seed-level test metrics to CSV and Markdown.
- Create `tests/test_dvlog_baselines.py`: unit and integration coverage for the complete local pipeline.
- Modify `README.md`: D-Vlog validation, smoke-test, single-run, and full-suite commands.

### Task 1: Dataset discovery, validation, and normalization

**Files:**
- Create: `src/datasets/dvlog_dataset.py`
- Test: `tests/test_dvlog_baselines.py`

- [ ] **Step 1: Write failing metadata and validation tests**

Create synthetic `labels.csv`, acoustic arrays, and visual arrays in `tmp_path`. Test label mapping, numeric ordering, split counts, missing-file errors, wrong dimensions, non-finite values, and cross-modal length mismatches. The central success case is:

```python
samples = discover_dvlog_samples(tmp_path)
assert [sample.sample_id for sample in samples] == ["0", "1", "2"]
assert [sample.label for sample in samples] == [1, 0, 1]
validate_dvlog_samples(samples)
```

- [ ] **Step 2: Run the focused test and confirm failure**

Run: `pytest tests/test_dvlog_baselines.py -k 'discovery or validation' -v`

Expected: collection or import failure because `src.datasets.dvlog_dataset` does not exist.

- [ ] **Step 3: Implement discovery and strict validation**

Define the following immutable record and implement `discover_dvlog_samples(dataset_root)`, `load_feature_pair(sample)`, and `validate_dvlog_samples(samples)` around it:

```python
@dataclass(frozen=True)
class DVlogSample:
    sample_id: str
    label: int
    split: str
    gender: str
    duration: float
    acoustic_path: Path
    visual_path: Path

LABEL_MAP = {"normal": 0, "depression": 1}
EXPECTED_DIMS = {"audio": 25, "visual": 136}
```

`load_feature_pair` returns float32 acoustic data, float32 visual data, and the visual-validity mask. It rejects missing files, empty or non-2D arrays, incorrect dimensions, non-finite values, unequal time lengths, and samples with no valid visual frame. `validate_dvlog_samples` rejects duplicate IDs and unknown labels/splits and returns split/class counts.

- [ ] **Step 4: Add and test streaming train-only normalizers**

Implement a numerically stable `FeatureNormalizer` with `fit(samples)`, `transform_audio(array)`, `transform_visual(array, valid_mask)`, `state_dict()`, and `from_state_dict()`. Fit only train samples, exclude all-zero visual rows, clamp standard deviations below `1e-6` to one, and restore invalid visual rows to zero after transformation.

Test with extreme validation/test values and assert that fitted means equal the train-only values rather than full-dataset values.

- [ ] **Step 5: Run dataset tests**

Run: `pytest tests/test_dvlog_baselines.py -k 'discovery or validation or normalizer' -v`

Expected: all selected tests pass.

- [ ] **Step 6: Commit the dataset foundation**

```bash
git add src/datasets/dvlog_dataset.py tests/test_dvlog_baselines.py
git commit -m "feat: add strict D-Vlog dataset loading"
```

### Task 2: Pooled and temporal batches

**Files:**
- Modify: `src/datasets/dvlog_dataset.py`
- Modify: `tests/test_dvlog_baselines.py`

- [ ] **Step 1: Write failing pooling and collation tests**

Test that `summarize_sequence(audio, mask)` returns `2 * D` values, ignores masked rows, and produces finite zero standard deviation for a single valid row. Test that temporal collation returns:

```python
assert batch["audio"].shape == (2, max_t, 25)
assert batch["visual"].shape == (2, max_t, 136)
assert batch["lengths"].tolist() == [short_t, long_t]
assert batch["visual_mask"].dtype == torch.bool
```

Also assert that metadata order is stable and padding is zero.

- [ ] **Step 2: Run the focused test and confirm failure**

Run: `pytest tests/test_dvlog_baselines.py -k 'pool or collate' -v`

Expected: failure because pooling and dataset classes are not implemented.

- [ ] **Step 3: Implement dataset modes and collators**

Add `DVlogDataset(samples, normalizer, representation)` where `representation` is `pooled` or `temporal`. Implement `summarize_sequence`, `collate_dvlog_pooled`, and `collate_dvlog_temporal`. Each item carries ID, label, gender, duration, normalized audio/visual arrays, lengths, and visual mask. Pooled mode emits 50-dimensional audio and 272-dimensional visual embeddings.

- [ ] **Step 4: Run batching tests**

Run: `pytest tests/test_dvlog_baselines.py -k 'pool or collate' -v`

Expected: all selected tests pass.

- [ ] **Step 5: Commit batching support**

```bash
git add src/datasets/dvlog_dataset.py tests/test_dvlog_baselines.py
git commit -m "feat: add D-Vlog pooled and temporal batches"
```

### Task 3: MLP and BiGRU models

**Files:**
- Create: `src/models/dvlog_baselines.py`
- Modify: `src/models/__init__.py`
- Modify: `tests/test_dvlog_baselines.py`

- [ ] **Step 1: Write failing model-shape tests**

Parameterize over `model_name in {mlp, bigru}` and `modality in {audio, visual, both}`. Build two-sample inputs and assert `logits.shape == (2, 2)`. For BiGRU, append extra zero batch padding without changing `lengths` and assert identical logits in evaluation mode.

- [ ] **Step 2: Run the model tests and confirm failure**

Run: `pytest tests/test_dvlog_baselines.py -k 'model or padding_invariant' -v`

Expected: import failure because `src.models.dvlog_baselines` does not exist.

- [ ] **Step 3: Implement the statistical-pooling MLP**

Implement `DVlogMLP(audio_dim=50, visual_dim=272, hidden_dim=256, dropout=0.2, modality="both")`. Validate modality, concatenate only selected embeddings, then apply `LayerNorm -> Linear -> ReLU -> Dropout -> Linear(2)`.

- [ ] **Step 4: Implement the modality-aware BiGRU**

Implement `TemporalEncoder(input_dim, projection_dim=128, hidden_dim=128, dropout=0.2)` with input projection, bidirectional one-layer GRU, masked mean and masked max pooling, and output dropout. Implement `DVlogBiGRU` with independent audio and visual encoders and the same 256-unit classification head. Use packed sequences for padding invariance and apply the visual mask during visual pooling.

- [ ] **Step 5: Run model tests**

Run: `pytest tests/test_dvlog_baselines.py -k 'model or padding_invariant' -v`

Expected: all selected tests pass.

- [ ] **Step 6: Commit models**

```bash
git add src/models/dvlog_baselines.py src/models/__init__.py tests/test_dvlog_baselines.py
git commit -m "feat: add D-Vlog MLP and BiGRU models"
```

### Task 4: Fixed-split training and evaluation

**Files:**
- Create: `src/train_dvlog.py`
- Create: `configs/baseline_dvlog.yaml`
- Modify: `tests/test_dvlog_baselines.py`

- [ ] **Step 1: Write failing training-helper tests**

Test `split_samples`, `compute_class_weights`, `is_better_checkpoint`, `evaluate`, and output paths. Ensure F1 ties prefer lower loss. Ensure prediction frames contain exactly `sample_id`, `label`, `pred_label`, `prob_depressed`, `gender`, and `duration`.

- [ ] **Step 2: Run helper tests and confirm failure**

Run: `pytest tests/test_dvlog_baselines.py -k 'split or class_weights or checkpoint or evaluate' -v`

Expected: import failure because `src.train_dvlog` does not exist.

- [ ] **Step 3: Implement CLI and fixed-split setup**

Support:

```text
python -m src.train_dvlog --config configs/baseline_dvlog.yaml --validate-data
python -m src.train_dvlog --config configs/baseline_dvlog.yaml --model mlp --modality both --seed 42
python -m src.train_dvlog --config configs/baseline_dvlog.yaml --model bigru --modality both --seed 42 --max-train-steps 2
```

Load config, apply CLI overrides, discover and validate all samples, split by official `fold`, fit normalizers on train only, select pooled or temporal data, and create deterministic loaders.

- [ ] **Step 4: Implement training, early stopping, and fixed-threshold evaluation**

Train with AdamW and optional train-derived class weights. Monitor validation F1 at 0.5; resolve ties with lower loss; stop after configured patience; reload `best.pt`; then write validation and test metrics/predictions. Save config, normalizer state, model state, epoch, and validation metrics in the checkpoint. Do not inspect test metrics during epoch selection.

- [ ] **Step 5: Add the server-default YAML**

Set dataset root `/home/rui/24zbma/data/D-vlog`, output root `runs/D-vlog`, model dimensions from the spec, batch size 16, 120 epochs, learning rate and weight decay `1e-4`, patience 15, class weighting enabled, device `auto`, workers 0, and seeds `[42, 2025, 3407]`.

- [ ] **Step 6: Run training-helper tests**

Run: `pytest tests/test_dvlog_baselines.py -k 'split or class_weights or checkpoint or evaluate' -v`

Expected: all selected tests pass.

- [ ] **Step 7: Commit training pipeline**

```bash
git add src/train_dvlog.py configs/baseline_dvlog.yaml tests/test_dvlog_baselines.py
git commit -m "feat: train D-Vlog baselines on official split"
```

### Task 5: Experiment runner and summary

**Files:**
- Create: `scripts/run_dvlog_baselines.sh`
- Create: `scripts/summarize_dvlog_runs.py`
- Modify: `tests/test_dvlog_baselines.py`

- [ ] **Step 1: Write failing summary tests**

Create fake seed metric JSON files and assert that aggregation yields six setting rows, three seeds per row, and population mean/std columns for `acc`, `precision`, `recall`, and `f1`. Test that a missing seed produces a clear error naming the absent run.

- [ ] **Step 2: Run summary tests and confirm failure**

Run: `pytest tests/test_dvlog_baselines.py -k 'summary' -v`

Expected: import failure because `scripts.summarize_dvlog_runs` does not exist.

- [ ] **Step 3: Implement summary generation**

Implement `collect_runs(root, expected_seeds)`, `summarize_runs(rows)`, and CLI output to `runs/D-vlog/metrics/dvlog_baselines_summary.csv` and `.md`. Use population standard deviation (`ddof=0`) and preserve model/modality order.

- [ ] **Step 4: Implement the full runner**

Loop over `mlp bigru`, `audio visual both`, and `42 2025 3407`; invoke `python -m src.train_dvlog` with explicit selectors; stop on the first failure; then run the summary script. Accept config and device as optional shell arguments.

- [ ] **Step 5: Run summary tests and shell syntax check**

Run: `pytest tests/test_dvlog_baselines.py -k 'summary' -v`

Run: `bash -n scripts/run_dvlog_baselines.sh`

Expected: tests pass and shell syntax exits zero.

- [ ] **Step 6: Commit orchestration**

```bash
git add scripts/run_dvlog_baselines.sh scripts/summarize_dvlog_runs.py tests/test_dvlog_baselines.py
git commit -m "feat: orchestrate and summarize D-Vlog runs"
```

### Task 6: Documentation and end-to-end verification

**Files:**
- Modify: `README.md`
- Modify: `tests/test_dvlog_baselines.py`

- [ ] **Step 1: Add an end-to-end synthetic smoke test**

Generate small official-split synthetic data with both labels in train/valid/test. Invoke `src.train_dvlog.main` or a subprocess for one MLP optimizer step and assert that checkpoint, validation metrics, test metrics, and both prediction files exist.

- [ ] **Step 2: Run the smoke test**

Run: `pytest tests/test_dvlog_baselines.py -k 'end_to_end' -v`

Expected: PASS in under one minute on CPU.

- [ ] **Step 3: Document server commands**

Add concise D-Vlog sections for data layout, data validation, MLP smoke test, BiGRU smoke test, one complete run, all 18 runs, summary locations, and the rule that 0.5-threshold test metrics are primary.

- [ ] **Step 4: Run the complete D-Vlog test file**

Run: `pytest tests/test_dvlog_baselines.py -v`

Expected: all tests pass.

- [ ] **Step 5: Run related regression tests**

Run: `pytest tests/test_lmvd_features.py tests/test_lmvd_fixed_split.py -v`

Expected: all tests pass; D-Vlog additions do not alter LMVD behavior.

- [ ] **Step 6: Run repository hygiene checks**

Run: `python -m compileall src scripts`

Run: `git diff --check`

Expected: compilation succeeds and no whitespace errors are reported.

- [ ] **Step 7: Commit documentation and final verification**

```bash
git add README.md tests/test_dvlog_baselines.py
git commit -m "docs: add D-Vlog baseline workflow"
```
