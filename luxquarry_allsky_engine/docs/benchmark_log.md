# Benchmark Log

## 2026-06-28: Manifest Smoke

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky build-manifest \
  --input-root /mnt/niroseti/spherex_cache/raw/qr2/level2 \
  --out runs/manifest_smoke_v2/frame_manifest.parquet \
  --campaign-id manifest_smoke_v2 \
  --limit 10
```

Result:

```text
frame_count: 10
fits_total_bytes: 716,368,320
total_wall_sec: 0.260
discover_fits: 0.074 sec
read_headers: 0.176 sec
write_manifest: 0.007 sec
```

Notes:

- The FITS path layout includes a detector directory:
  `raw/qr2/level2/<planning_period>/<processing_version>/<detector>/<file>.fits`.
- The manifest parser now extracts planning period/version from the path and
  exposure/frame/detector from the filename.
- Header/WCS extraction is currently CPU/Astropy and already appears as a
  >5% stage, so it remains on the acceleration audit list. It is acceptable for
  now because Astropy is the correctness reference.

## 2026-06-28: Catalog Target Smoke

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky build-frame-targets \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --out runs/frame_targets_smoke_current/frame_targets.parquet \
  --catalog all \
  --max-sources-per-frame 500 \
  --limit-frames 10
```

Result:

```text
frame_count: 10
target_row_count: 5,000
unique_target_count: 1,256
total_wall_sec: 1.691
per_frame_wall_sec: 0.154-0.183
write_frame_targets: 0.006 sec
```

Notes:

- The first naive DuckDB version scanned the whole Gaia/2MASS parquet forest and
  was too slow. The current version computes candidate HPX tiles from each frame
  footprint and passes only those parquet files to DuckDB.
- Gaia uses HPX level 3 source partitions; 2MASS uses HPX level 5 coordinate
  partitions.
- `--catalog all` splits `--max-sources-per-frame` between Gaia and 2MASS.

## 2026-06-28: Vectorized WCS Projection Smoke

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky project-frame-targets \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --frame-targets runs/frame_targets_smoke_current/frame_targets.parquet \
  --out runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --limit-frames 10
```

Result:

```text
frame_count: 10
input_target_rows: 5,000
projected_target_rows: 5,000
in_frame_target_rows: 2,802
total_wall_sec: 0.260
per_frame_wall_sec: 0.018-0.025
write_projected_targets: 0.008 sec
```

Notes:

- Projection calls Astropy WCS once per frame with an array of target
  coordinates, not one target at a time.
- The RA/Dec catalog footprint query is intentionally a broad prefilter.
  The `in_frame` column is the photometry-ready mask.

## 2026-06-28: CPU Calibrated Aperture Smoke

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky run-cpu-aperture \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out runs/measurements_cpu_aperture_smoke10/measurements.parquet \
  --limit-frames 10
```

Result:

```text
frame_count: 10
input_projected_rows: 5,000
measurement_rows: 2,802
ok_measurement_rows: 2,766
total_wall_sec: 3.597
per_frame_wall_sec: 0.294-0.447
write_measurements: 0.006 sec
```

Notes:

- This is calibrated aperture photometry in uJy using cached SAPM files and
  `VARIANCE` propagation.
- It samples `CWAVE`/`CBAND` from the per-detector spectral WCS maps and stores
  calibration file provenance in every measurement row.
- A first implementation accidentally allocated full-frame masks per target and
  took 39.1 sec for two frames. Switching to local aperture cutouts reduced the
  same two-frame smoke to 0.77 sec.
- This CPU stage is a correctness/profiling baseline. The intended V2 hot path
  is the same frame-batched calculation on GPU.
