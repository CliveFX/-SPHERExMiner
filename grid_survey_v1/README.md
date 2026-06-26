# Grid Survey V1

HEALPix-front-end prototype for running the existing SPHEREx target pipeline as
a sky survey.

This workspace does **not** replace the current miner. It only changes target
selection:

```text
HEALPix cell id
  -> cell polygon
  -> local catalog DuckDB query
  -> target YAML/parquet manifest
  -> existing per-target pipeline
```

The important design rule is that every selected catalog source still runs
through the same target-of-interest pipeline: photometry, spectrum assembly,
spectrum quality, injection/recovery when requested, and narrowband scoring.

## First Tool

Build target manifests for one or more HEALPix cells:

```bash
grid_survey_v1/.venv/bin/python grid_survey_v1/tools/build_healpix_target_manifest.py \
  --nside 64 \
  --hpx 1234 \
  --g-min 11 \
  --g-max 16 \
  --max-sources 3000 \
  --catalog gaia \
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

The generated `targets.yaml` can be passed to the existing pipeline as fixed
targets. In grid mode, the HEALPix cell center is only used to select SPHEREx
frames; the catalog rows in the manifest are passed directly as the measured
target list.

```bash
grid_survey_v1/.venv/bin/python grid_survey_v1/tools/dispatch_healpix_mag_bins.py \
  --campaign-prefix grid_v1_hpx0064_00001234_g11_16_f500 \
  --nside 64 \
  --hpx 1234 \
  --catalog gaia \
  --mag-bin mid_g11_16:11:16:3000 \
  --limit-fields 500 \
  --pipeline baseline \
  --execute
```

For large tiles, prefer the files under `batches/`. A 100k-target YAML can be
50-90 MB and is awkward to parse, review, resume, and log. The default
`--batch-size 3000` keeps each campaign shard at the scale we have already been
running successfully.

## Dispatching Magnitude Bins

`tools/dispatch_healpix_mag_bins.py` is the grid-survey wrapper. It builds one
target manifest per HEALPix cell and magnitude bin, then either writes a dry
plan or runs one direct pipeline job per target batch.

Important: `--pipeline baseline` is direct grid mode. It runs one
`spherex-mine run-depth-test` per HEALPix/magnitude/batch with
`--fixed-targets-path`. It does **not** iterate each Gaia source as a separate
target-of-interest anchor. The legacy recursive target-of-interest behavior is
available only as `--pipeline anchor_ladder` for diagnostics.

Catalog modes:

```text
--catalog gaia    local Gaia lite query, Gaia G magnitude bins
--catalog 2mass   local 2MASS PSC query, 2MASS band magnitude bins
--catalog all     Gaia plus 2MASS rows in the same fixed-target manifest
```

For 2MASS grid runs, `g_min/g_max` in the `--mag-bin` string are interpreted as
the selected 2MASS band magnitude bounds. The default band is Ks:

```bash
grid_survey_v1/.venv/bin/python grid_survey_v1/tools/dispatch_healpix_mag_bins.py \
  --campaign-prefix grid_v1_2mass_k11_15 \
  --nside 64 \
  --hpx 1234 \
  --catalog 2mass \
  --twomass-band Ks \
  --twomass-quality ABC \
  --twomass-selection stratified \
  --mag-bin k11_15:11:15:3000 \
  --limit-fields 500 \
  --pipeline baseline \
  --execute
```

2MASS selection modes:

```text
--twomass-selection stratified  spread targets across the requested mag range
--twomass-selection brightest   take the bright edge first
--twomass-selection random      deterministic hash-random sample
```

Use `stratified` for survey planning. Otherwise a wide bin such as
`11:16:300` will mostly sample the bright edge near `Ks=11`.

Raw 2MASS PSC rows do not include proper motion or parallax, so catalog 2MASS
targets are static J2000/ICRS positions until the HPM enrichment pass exists.
Manual targets still use their configured proper-motion metadata.

Plan a three-bin grid run without starting photometry:

```bash
grid_survey_v1/.venv/bin/python grid_survey_v1/tools/dispatch_healpix_mag_bins.py \
  --campaign-prefix grid_v1_mag_sweep_smoke \
  --nside 64 \
  --hpx 1234 \
  --mag-bin lowmag_bright_g5_8:5:8:3000 \
  --mag-bin mid_g11_16:11:16:3000 \
  --mag-bin highmag_faint_g16_19:16:19:3000 \
  --limit-fields 500 \
  --max-field-workers 24 \
  --warp-devices cuda:0,cuda:1,cuda:2 \
  --pipeline baseline
```

Add `--execute` to run it. Without `--execute`, it only writes:

```text
/mnt/niroseti/spherex_cache/grid_survey_v1/dispatches/<campaign-prefix>/dispatch_plan.json
```

The same wrapper can run full injection/recovery campaigns:

```bash
grid_survey_v1/.venv/bin/python grid_survey_v1/tools/dispatch_healpix_mag_bins.py \
  --campaign-prefix grid_v1_mag_sweep_injected \
  --nside 64 \
  --hpx 1234 \
  --mag-bin mid_g11_16:11:16:3000 \
  --limit-fields 500 \
  --max-field-workers 24 \
  --warp-devices cuda:0,cuda:1,cuda:2 \
  --pipeline injection \
  --execute
```

Naming convention:

- `lowmag_bright_g5_8` means numerically low Gaia G magnitude and bright stars.
- `mid_g11_16` is the current safest science range.
- `highmag_faint_g16_19` means numerically high Gaia G magnitude and faint stars.

The wrapper deliberately keeps the miner output format unchanged. Each batch is
still a normal campaign/run under `spherex_cache`, so the existing spectra,
candidate, recovery, and status viewers can read the products.

## Web UI

The main viewer exposes a grid dispatcher at:

```text
http://192.168.1.224:8765/grid-survey
```

The UI supports:

- HEALPix `nside`, explicit cell lists, or start/count ranges.
- Multiple magnitude bins in `name:g_min:g_max:max_sources` form.
- Spectral depth, batch size, worker count, and GPU devices.
- Catalog selection: Gaia, 2MASS, or Gaia + 2MASS.
- 2MASS band, quality, selection mode, dataset, and HEALPix level controls.
- Direct baseline runs or direct injection/recovery runs.
- Pause/resume/stop files for batch-safe dispatch control.
- Pan/zoom/click on the sky map, with HEALPix cell borders colored by status.

`Plan` writes manifests and `dispatch_plan.json` only. `Start` currently
regenerates the plan from current UI values and then runs it, so reviewed plans
must be started without changing the controls.

For injection/recovery, the direct grid wrapper writes both raw truth-target
recovery products and the legacy paired-delta recovery products:

```text
runs/<prefix>_injected/narrowband_detector_truth/
runs/<prefix>_injected/classifier_paired_delta/
runs/<prefix>_injected/recovery_score_mixed_lasers/
grid_survey_v1/direct_injection/<prefix>/direct_injection_summary.json
```

Use `--injection-max-lines-per-target 1` or the UI `Max lines/target = 1`
control to prevent one target spectrum from receiving several injected laser
families.

## Notes

- HEALPix order is `nested`.
- Coordinates are ICRS.
- Gaia query uses the existing local Gaia lite DuckDB path.
- 2MASS query uses the processed local PSC lite Parquet cache.
- Without `--execute`, the dispatcher writes manifests only; it does not start
  photometry.
- No large outputs should be committed to git.
