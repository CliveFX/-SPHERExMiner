# LuxQuarry ML Workspace

This directory is for the two-model spectral ML effort described in
[`docs/ml_two_model_plan.md`](../docs/ml_two_model_plan.md).

The intent is to keep ML dataset building, model code, training jobs, and
training-status outputs separate from the current deterministic mining pipeline.
The miner remains the source of truth for photometry, injection, spectra
assembly, quality scoring, and candidate generation.

## Planned Layout

```text
ml/
  README.md
  datasets/
    build_ml_datasets.py
    schemas.md
  science_embedding/
    model.py
    train.py
    evaluate.py
  narrowband_signal/
    model.py
    train.py
    evaluate.py
  dashboards/
    status_schema.md
  configs/
    science_embedding_smoke.yaml
    narrowband_signal_smoke.yaml
```

## Environment

Use a separate ML virtual environment. Do not install PyTorch into the miner
`.venv`.

```bash
python3 -m venv ml/.venv
ml/.venv/bin/python -m pip install --upgrade pip
ml/.venv/bin/python -m pip install -r ml/requirements.txt
```

The current local smoke test used `torch==2.12.1+cu130` from PyPI and sees all
three local NVIDIA GPUs from `ml/.venv`.

## Model Tracks

Science embedding model:

- Ragged/set-based spectral encoder.
- Learns shape, absolute, and quality embeddings.
- Supports nearest-neighbor review, spectral family maps, and weird-object
  discovery.

Narrowband signal model:

- Uses injected spectra, baseline spectra, false positives, and matched-filter
  outputs.
- Produces signal probability, estimated wavelength, and review grade.
- Uses explicit injection provenance fields from future spectra outputs.
- Current baseline is a ragged point/set encoder trained from cached tensor
  shards. See
  [`docs/narrowband_ml_training_log.md`](../docs/narrowband_ml_training_log.md).

## Dashboard Contract

Training jobs should write small atomic JSON/JSONL files that the viewer can
serve without touching heavy parquet files:

```text
ml_runs/<run_name>/
  training_status.json
  training_metrics.jsonl
  dataset_summary.json
  model_card.json
  checkpoints/
```

The planned dashboards are:

- `/ml-training`
- `/ml-datasets`
- `/ml-embeddings`
- `/ml-narrowband`

## First Implementation Step

`ml/datasets/build_ml_datasets.py` now builds the first dataset shards. Example:

```bash
.venv/bin/python ml/datasets/build_ml_datasets.py \
  --dataset-name smoke_ml_v0 \
  --run-dir /mnt/niroseti/spherex_cache/runs/injrec_baseline_500_f80_g14_16_cpu \
  --run-dir /mnt/niroseti/spherex_cache/runs/injrec_injected_500_f80_g14_16_s8_gpu3 \
  --max-targets-per-run 50
```

Outputs land under:

```text
/mnt/niroseti/spherex_cache/ml_datasets/<dataset_name>/
```

## Smoke Training

Science embedding smoke run:

```bash
ml/.venv/bin/python ml/science_embedding/train.py \
  --dataset-dir /mnt/niroseti/spherex_cache/ml_datasets/smoke_ml_v0 \
  --run-name science_smoke_v0 \
  --epochs 2 \
  --batch-size 16 \
  --max-targets 64 \
  --max-points 128 \
  --device cuda
```

Narrowband smoke run:

```bash
ml/.venv/bin/python ml/narrowband_signal/train.py \
  --dataset-dir /mnt/niroseti/spherex_cache/ml_datasets/smoke_ml_v0 \
  --run-name narrowband_smoke_v0 \
  --epochs 2 \
  --batch-size 16 \
  --max-targets 100 \
  --max-points 128 \
  --device cuda
```

Current cached narrowband line/no-line run shape:

```bash
ml/.venv/bin/python ml/datasets/build_ml_datasets.py \
  --dataset-name narrowband_line_cv_mega_v0 \
  --run-glob 'cv_june_g11_16_f500*_baseline' \
  --run-glob 'cv_june_g11_16_f500*_injected' \
  --quality-category good \
  --quality-category review \
  --max-targets-per-run 5000 \
  --workers 16

ml/.venv/bin/python ml/narrowband_signal/build_feature_cache.py \
  --dataset-dir /mnt/niroseti/spherex_cache/ml_datasets/narrowband_line_cv_mega_v0 \
  --cache-name narrowband_line_cv_mega_v0_train_cache \
  --split train \
  --workers 24 \
  --max-points 384

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

Training outputs land under:

```text
/mnt/niroseti/spherex_cache/ml_runs/<run_name>/
```

The local viewer exposes training status at:

```text
http://192.168.1.224:8765/ml-training
```
