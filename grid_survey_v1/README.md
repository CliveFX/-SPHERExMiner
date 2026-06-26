# Grid Survey V1

HEALPix-front-end prototype for running the existing SPHEREx target pipeline as
a sky survey.

This workspace does **not** replace the current miner. It only changes target
selection:

```text
HEALPix cell id
  -> cell polygon
  -> local Gaia DuckDB query
  -> target YAML/parquet manifest
  -> existing per-target pipeline
```

The important design rule is that every selected Gaia source still runs through
the same target-of-interest pipeline: photometry, spectrum assembly, spectrum
quality, injection/recovery when requested, and narrowband scoring.

## First Tool

Build target manifests for one or more HEALPix cells:

```bash
grid_survey_v1/.venv/bin/python grid_survey_v1/tools/build_healpix_target_manifest.py \
  --nside 64 \
  --hpx 1234 \
  --g-min 11 \
  --g-max 16 \
  --max-sources 3000 \
  --batch-size 3000 \
  --output-root /mnt/niroseti/spherex_cache/grid_survey_v1/smoke
```

Outputs:

```text
<output-root>/survey_manifest.json
<output-root>/hpx_nside0064_nested_00001234/
  targets.yaml
  targets.parquet
  tile_summary.json
  batches/
    targets_batch_0000.yaml
```

The generated `targets.yaml` can be passed to the existing campaign runner:

```bash
grid_survey_v1/.venv/bin/python tools/run_visible_sky_injection_campaign.py \
  --targets <output-root>/hpx_nside0064_nested_00001234/targets.yaml \
  --campaign-prefix grid_v1_hpx0064_00001234_g11_16_f500 \
  --no-resolve-gaia-anchors \
  --gaia-g-min 11 \
  --gaia-g-max 16 \
  --max-gaia-sources 3000 \
  --limit-fields 500
```

For large tiles, prefer the files under `batches/`. A 100k-target YAML can be
50-90 MB and is awkward to parse, review, resume, and log. The default
`--batch-size 3000` keeps each campaign shard at the scale we have already been
running successfully.

## Notes

- HEALPix order is `nested`.
- Coordinates are ICRS.
- Gaia query uses the existing local Gaia lite DuckDB path.
- The tool writes manifests only; it does not start photometry.
- No large outputs should be committed to git.
