# Grid Science Embeddings

Grid survey runs can be exported into the same science embedding and UMAP
viewer flow as the older target-of-interest campaigns.

## One-Off Export

Use the existing metadata-aware science checkpoint to export embeddings for a
grid campaign prefix:

```bash
ml/.venv/bin/python tools/run_grid_science_embedding.py \
  --cache-root /mnt/niroseti/spherex_cache \
  --campaign-prefix dual_catalog_test \
  --embedding-name dual_catalog_test_science_v2 \
  --run-kind baseline \
  --workers 24 \
  --prep-workers 24 \
  --device cuda
```

Outputs:

```text
/mnt/niroseti/spherex_cache/ml_datasets/<embedding_name>/
  science_targets.parquet
  science_points.parquet
  dataset_summary.json

/mnt/niroseti/spherex_cache/ml_outputs/science_embeddings/<embedding_name>/
  embeddings.parquet
  umap_projection.parquet
  grid_embedding_status.json
```

The exported rows include:

- `campaign`
- `mag_bin`
- `grid_tile_id`
- `grid_nside`
- `grid_order`
- `grid_hpx`
- `grid_batch_index`

## Grid Dispatcher Integration

The grid dispatcher can run the same export automatically after an executed
dispatch:

```bash
grid_survey_v1/.venv/bin/python grid_survey_v1/tools/dispatch_healpix_mag_bins.py \
  --campaign-prefix my_grid_campaign \
  --nside 16 \
  --hpx 2007 \
  --catalog all \
  --mag-bin mid_g11_16:11:16:3000 \
  --pipeline injection \
  --execute \
  --science-embedding
```

The Grid Survey web UI exposes this as "Build science embedding + UMAP after
dispatch". It is default-off so ordinary grid mining behavior is unchanged.

By default, the embedding export uses `--run-kind baseline`. This is intentional:
the main science UMAP should organize real stellar spectra, not injected spectra.
