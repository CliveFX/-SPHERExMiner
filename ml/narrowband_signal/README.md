# Narrowband Signal Model

Planned model for injection-trained narrowband candidate ranking.

Inputs should include spectra points, injection truth, baseline negatives, hard
false positives, flags, uncertainties, aperture/PSF co-movement, and optional
science embeddings.

The first implementation should compare directly against the current GPU
matched-filter detector.

## Scaling note

Do not run serious narrowband training directly from the raw parquet point
tables. The first prototype proved that this spends most of its time in
single-process CPU dataframe grouping before CUDA training starts.

Use the feature-cache stage first:

```bash
ml/.venv/bin/python ml/narrowband_signal/build_feature_cache.py \
  --dataset-dir /mnt/niroseti/spherex_cache/ml_datasets/narrowband_line_cv_mega_v0 \
  --cache-name narrowband_line_cv_mega_v0_train_cache \
  --split train \
  --workers 24
```

Current development hardware has 32 physical CPU cores. Future target systems
may have around 192 CPU cores, so all large dataset preparation should expose
worker counts and shard work by run or by target block. On this 32-core system,
prefer about 24 feature-build workers for normal interactive use so the viewer,
OS, NAS client, and training process keep breathing room. Use 32 only for
dedicated batch windows. GPU training should read prepared tensor shards rather
than rebuilding ragged spectra from parquet each experiment.

## Architecture options

`train_line.py` supports two first-pass architectures:

```text
--architecture set
```

The default ragged point/set encoder. It is fast and uses mean/max pooling over
per-measurement features.

```text
--architecture transformer
```

A small spectral transformer with per-measurement tokens, a learned CLS token,
spectral position encoding, and padding masks for ragged spectra. Example:

```bash
ml/.venv/bin/python ml/narrowband_signal/train_line.py \
  --dataset-dir /mnt/niroseti/spherex_cache/ml_datasets/narrowband_line_cv_mega_v0 \
  --feature-cache-dir /mnt/niroseti/spherex_cache/ml_feature_caches/narrowband_line_cv_mega_v0_train_cache \
  --run-name narrowband_line_cv_mega_v0_transformer_train8 \
  --architecture transformer \
  --model-version narrowband_line_transformer_v0 \
  --epochs 8 \
  --batch-size 128 \
  --max-points 384 \
  --hidden-dim 192 \
  --embedding-dim 128 \
  --transformer-layers 3 \
  --transformer-heads 6 \
  --device cuda
```
