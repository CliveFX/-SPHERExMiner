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

## 2026-06-28: GPU Frame Aperture Smoke

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky run-gpu-aperture \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out runs/measurements_gpu_aperture_smoke10_warm/measurements.parquet \
  --limit-frames 10 \
  --device cuda:0
```

Result:

```text
frame_count: 10
input_projected_rows: 5,000
measurement_rows: 2,770
ok_measurement_rows: 2,766
total_wall_sec: 2.824
steady_frame_wall_sec: mostly 0.126-0.225
first_frame_wall_sec: 1.307
backend: Warp frame kernel + cuDF parquet
```

Correctness against CPU aperture baseline:

```text
matched_ok_measurements: 2,766
flux_median_abs_delta_uJy: 0.0022
flux_p95_abs_delta_uJy: 0.0253
flux_median_relative_delta_pct: 0.000070
flux_p95_relative_delta_pct: 0.000291
flux_max_relative_delta_pct: 0.603
cwave_p95_abs_delta_um: 2.3e-7
cband_p95_abs_delta_um: 1.9e-8
```

Notes:

- The GPU kernel receives raw `IMAGE`, `VARIANCE`, `FLAGS`, `SAPM`, `CWAVE`,
  `CBAND`, and target pixel arrays. It performs image calibration, variance
  scaling, wavelength sampling, aperture/background estimation, uncertainty,
  and flag summary in one launch per frame.
- Warp outputs are handed to CuPy/cuDF through DLPack for measurement columns.
  cuDF writes parquet shards.
- The current CLI still pays process/RAPIDS setup and first-touch calibration
  load costs. The production target is a persistent monolithic worker that keeps
  RAPIDS, kernels, and calibration maps resident across a large frame batch.
- CPU and GPU uncertainty can diverge on a few background edge cases because the
  CPU and Warp sigma-clipping implementations are not byte-for-byte identical.
  Flux and wavelength agreement are already tight enough for the next
  performance prototype.

## 2026-06-29: Persistent GPU Worker Smoke

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky run-persistent-gpu-worker \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/persistent_gpu_worker_smoke10 \
  --run-id persistent_smoke10 \
  --limit-frames 10 \
  --device cuda:0
```

Result:

```text
frame_count: 10
input_projected_rows: 5,000
measurement_rows: 2,770
ok_measurement_rows: 2,766
failed_frames: 0
calibration_upload_count: 3
total_wall_sec: 2.387
backend: persistent Warp frame kernel + cuDF shards
```

Stage profile from frame timings:

```text
kernel_wall_sec: ~0.030-0.035 sec/frame
fits_read_wall_sec: ~0.105-0.133 sec/frame
table_wall_sec: ~0.011 sec/frame after first frame
write_wall_sec: ~0.010 sec/frame after first frame
```

Correctness against CPU aperture baseline:

```text
matched_ok_measurements: 2,766
flux_median_abs_delta_uJy: 0.0022
flux_p95_abs_delta_uJy: 0.0253
flux_p95_relative_delta_pct: 0.000291
cwave_p95_abs_delta_um: 2.3e-7
cband_p95_abs_delta_um: 1.9e-8
```

Notes:

- Detector calibration maps are uploaded once per `(release, detector)` and
  kept resident for the worker lifetime.
- The worker writes one independent parquet shard per frame plus
  `run_summary.json` and atomic `run_status.json`.
- The measured bottleneck has shifted to FITS reading and table/shard overhead;
  the aperture kernel is no longer the dominant cost on this smoke set.

## 2026-06-29: Batched Shards and FITS Prefetch

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky run-persistent-gpu-worker \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/persistent_gpu_worker_smoke10_batch5_prefetch2 \
  --run-id persistent_smoke10_batch5_prefetch2 \
  --limit-frames 10 \
  --device cuda:0 \
  --shard-batch-frames 5 \
  --prefetch-frames 2
```

Result:

```text
frame_count: 10
input_projected_rows: 5,000
measurement_rows: 2,770
ok_measurement_rows: 2,766
failed_frames: 0
shards: 2
total_wall_sec: 1.156
```

Correctness:

```text
matched_ok_measurements_vs_cpu: 2,766
flux_p95_abs_delta_uJy_vs_cpu: 0.0253
```

Notes:

- Batching five frames per shard reduced output files from 10 to 2 on the smoke
  run.
- `--prefetch-frames 2` overlaps FITS reads with GPU/table work. The reported
  per-frame `fits_read_wall_sec` remains useful as I/O accounting, but much of
  it no longer blocks the hot loop.
- This is the current recommended local worker mode for longer frame queues.

## 2026-06-29: Three-GPU Dispatch Smoke

Commands:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky plan-gpu-dispatch \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/dispatch_smoke10 \
  --run-id dispatch_smoke10 \
  --plan-out runs/dispatch_smoke10/dispatch_plan.json \
  --devices cuda:0,cuda:1,cuda:2 \
  --limit-frames 10

runs/dispatch_smoke10/dispatch_plan.sh
```

Result:

```text
workers: 3
devices: cuda:0,cuda:1,cuda:2
frame split: 4 / 3 / 3
shards: 10
measurement_rows: 2,770
ok_measurement_rows: 2,766
failed_frames: 0
```

Notes:

- The dispatch contract is `frame_ordinal % worker_count == worker_index`.
- Each worker has its own output directory and status JSON.
- For tiny runs, launching three Python/RAPIDS processes costs more than it
  saves. This mode is for long queues where setup is amortized and GPUs stay
  busy.
