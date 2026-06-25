# Narrowband ML Training Log

This log records the first supervised laser-line ML detector run. The model is a
ragged point/set encoder, not a transformer. It is intended as a fast baseline
for learning whether an injected narrowband excess is present and, if present,
the approximate injected wavelength.

## 2026-06-24: cached line/no-line baseline

### Dataset build

Command:

```bash
ml/.venv/bin/python ml/datasets/build_ml_datasets.py \
  --dataset-name narrowband_line_cv_mega_v0 \
  --run-glob 'cv_june_g11_16_f500*_baseline' \
  --run-glob 'cv_june_g11_16_f500*_injected' \
  --quality-category good \
  --quality-category review \
  --max-targets-per-run 5000 \
  --workers 16 \
  --status-every 1
```

Output:

```text
/mnt/niroseti/spherex_cache/ml_datasets/narrowband_line_cv_mega_v0
```

Summary:

```text
selected runs:        81
successful runs:      81
target spectra:       206,168
spectral points:      24,442,726
injection truth rows: 3,297
quality good:         202,499
quality review:       3,669
dataset size:         3.3 GiB
```

Injection strength counts:

```text
0.5 sigma: 297
1.0 sigma: 635
2.0 sigma: 297
3.0 sigma: 630
5.0 sigma: 369
8.0 sigma: 700
12.0 sigma: 369
```

### Feature cache

The direct training prototype spent most of its time in single-process
pandas/groupby feature construction before reaching CUDA. The production path
for ML experiments now builds a tensor feature cache first.

Command used for this run:

```bash
ml/.venv/bin/python ml/narrowband_signal/build_feature_cache.py \
  --dataset-dir /mnt/niroseti/spherex_cache/ml_datasets/narrowband_line_cv_mega_v0 \
  --cache-name narrowband_line_cv_mega_v0_train_cache \
  --split train \
  --workers 32 \
  --max-points 384 \
  --status-every 1
```

Output:

```text
/mnt/niroseti/spherex_cache/ml_feature_caches/narrowband_line_cv_mega_v0_train_cache
```

Summary:

```text
run shards:       81
train examples:   143,923
positive examples: 1,628
cache size:       613 MiB
elapsed:          51.0 sec
```

Operational note: `32` workers is useful for dedicated batch windows, but it
can starve the desktop/viewer/NAS client. Use `24` workers by default on the
current 32-core development machine. Future larger systems may have around 192
CPU cores; keep worker counts explicit and leave OS/viewer/storage headroom.

### Training

Command:

```bash
ml/.venv/bin/python ml/narrowband_signal/train_line.py \
  --dataset-dir /mnt/niroseti/spherex_cache/ml_datasets/narrowband_line_cv_mega_v0 \
  --feature-cache-dir /mnt/niroseti/spherex_cache/ml_feature_caches/narrowband_line_cv_mega_v0_train_cache \
  --run-name narrowband_line_cv_mega_v0_train8_cached \
  --epochs 8 \
  --batch-size 512 \
  --max-points 384 \
  --hidden-dim 192 \
  --embedding-dim 128 \
  --device cuda
```

Output:

```text
/mnt/niroseti/spherex_cache/ml_runs/narrowband_line_cv_mega_v0_train8_cached
```

Important files:

```text
training_status.json
training_metrics.jsonl
model_card.json
checkpoints/best.pt
```

Final training status:

```text
examples:                       143,923
negative examples:              142,295
positive examples:              1,628
device:                         cuda
elapsed after cache load/train:  47.4 sec
latest train loss:              0.3928
injected detected fraction:     1.000
baseline false alarm fraction:  0.122
positive line MAE:              185.5 nm
```

### Interpretation

The cached path fixed the immediate performance problem: preprocessing reached
CUDA quickly, and the training loop completed in under a minute once the tensor
cache existed.

This is a useful baseline, not a finished detector:

- The model easily learns that injected spectra are different from baseline
  spectra.
- False alarms are still too high for science search.
- Wavelength localization is broad at roughly 185 nm mean absolute error.
- The current model is a small ragged point/set encoder. A later small
  transformer may be warranted if attention over local spectral neighborhoods
  improves false-positive rejection and wavelength localization.

Next steps:

1. Add validation/test feature caches and report held-out metrics.
2. Add an inference script that writes per-target ML candidate scores.
3. Compare ML scores against the deterministic GPU narrowband detector on the
   same injected and uninjected runs.
4. Add hard negatives from false positives and flagged/artifact-heavy spectra.
5. Consider a small spectral transformer once this set-encoder baseline is
   fully measured.
