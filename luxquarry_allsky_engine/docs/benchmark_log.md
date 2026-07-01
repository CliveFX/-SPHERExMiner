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

## 2026-06-29: GPU PSF Candidate-Grid Smoke

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky run-dispatch-benchmark-sweep \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/dispatch_benchmark_psf_gpu_offsets_profiled_smoke10 \
  --run-id dispatch_psf_gpu_offsets_profiled_smoke10 \
  --devices cuda:0 \
  --workers-per-device 1 \
  --limit-frames 10 \
  --shard-batch-frames 5 \
  --prefetch-frames 0 \
  --worker-only \
  --enable-psf \
  --psf-kernel-build-mode gpu_spline \
  --status-snapshot-interval-sec 0
```

Result:

```text
frame_count: 10
measurement_rows: 2,770
psf_measurement_rows: 2,770
ok_psf_rows: 2,770
psf_candidate_count: 69,250
total_wall_sec: 4.836
worker_max_wall_sec: 2.013
measurements_per_sec_worker_payload: 1,375.8
psf_candidates_per_sec_device_submit_sync: 848,659.8
```

Profile rows:

```text
fits_read_wall_sec: 0.896
frame_upload_wall_sec: 0.240
aperture_kernel_wall_sec: 0.0277
psf_candidate_grid_wall_sec: 0.000619
psf_kernel_wall_sec: 0.449
psf_device_submit_sync_wall_sec: 0.0816
psf_spline_coeff_wall_sec: 0.141
psf_upload_wall_sec: 0.00761
psf_gather_wall_sec: 0.0568
write_wall_sec: 0.309
```

Notes:

- PSF candidate coordinates are no longer expanded into a full CPU-side
  target-by-grid array. The worker uploads target pixel arrays and the small
  subpixel offset bank; the Warp kernels compute candidate coordinates by
  indexing `(target_index, offset_index)`.
- A direct one-frame correctness check against the previous shared-frame PSF
  implementation matched key aperture and PSF columns exactly for the smoke
  frame: `psf_flux_uJy`, `psf_flux_unc_uJy`, `psf_x_fit`, `psf_y_fit`,
  `psf_fit_offset_pix`, `psf_grid_best_score`, and `aperture_flux_uJy`.
- On this small workload, the GPU-offset change is a modest end-to-end payload
  gain because FITS read, PSF spline coefficient setup, frame upload, PSF
  kernel/gather, and shard writes dominate. It does remove candidate-grid
  setup as a CPU bottleneck: `psf_candidate_grid_wall_sec` is about 0.6 ms
  total across 10 frames.

## 2026-06-29: Dense Galactic-Plane PSF Occupancy Smoke

The sparse smoke set had only about 277 selected targets per frame, which
underfed the PSF grid kernels. To test the GPU work shape under a more realistic
dense field, we built a full local QR2 frame manifest and selected the 20
available frames closest to the Galactic center. The local QR2 cache does not
hit the Galactic center directly; the nearest frames are about 15.5 degrees
away, around RA 269-270 deg and Dec -13.7 deg.

Manifest build:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky build-manifest \
  --input-root /mnt/niroseti/spherex_cache/raw/qr2/level2 \
  --out runs/manifest_qr2_local_all/frame_manifest.parquet \
  --campaign-id manifest_qr2_local_all
```

Result:

```text
frame_count: 8,035
fits_total_bytes: 575,589,729,600
discover_fits: 1.384 sec
read_headers: 258.133 sec
write_manifest: 0.029 sec
total_wall_sec: 259.578 sec
```

Notes:

- Header/WCS manifest generation is a setup-stage cost, not a hot worker-path
  cost. It should be prebuilt and reused for distributed runs.
- The nearest-20 Galactic-plane manifest was written to
  `runs/manifest_galactic_core_nearest20/frame_manifest.parquet`.

Catalog and projection:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky build-frame-targets \
  --manifest runs/manifest_galactic_core_nearest20/frame_manifest.parquet \
  --out runs/frame_targets_galactic_core_nearest20_allcat_g11_16_n20000/frame_targets.parquet \
  --catalog all \
  --gaia-g-min 11 \
  --gaia-g-max 16 \
  --twomass-mag-min 11 \
  --twomass-mag-max 16 \
  --max-sources-per-frame 20000

.venv/bin/luxquarry-allsky project-frame-targets \
  --manifest runs/manifest_galactic_core_nearest20/frame_manifest.parquet \
  --frame-targets runs/frame_targets_galactic_core_nearest20_allcat_g11_16_n20000/frame_targets.parquet \
  --out runs/projected_targets_galactic_core_nearest20_allcat_g11_16_n20000/frame_targets_projected.parquet
```

Result:

```text
frame_count: 20
frame_target_rows: 400,000
unique_target_count: 32,648
in_frame_projected_rows: 360,394
typical selected rows per frame: 17,000-18,000
catalog_query_wall_sec: 36.253
projection_wall_sec: 1.741
```

PSF worker-only benchmark:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky run-dispatch-benchmark-sweep \
  --manifest runs/manifest_galactic_core_nearest20/frame_manifest.parquet \
  --projected-targets runs/projected_targets_galactic_core_nearest20_allcat_g11_16_n20000/frame_targets_projected.parquet \
  --out-dir runs/dispatch_benchmark_psf_gc_dense_real_smoke10 \
  --run-id dispatch_psf_gc_dense_real_smoke10 \
  --devices cuda:0 \
  --workers-per-device 1 \
  --limit-frames 10 \
  --shard-batch-frames 5 \
  --prefetch-frames 0 \
  --worker-only \
  --enable-psf \
  --psf-kernel-build-mode gpu_spline \
  --status-snapshot-interval-sec 0
```

Result:

```text
measurement_rows: 178,705
ok_psf_rows: 176,860
psf_candidate_count: 4,467,625
worker_max_wall_sec: 3.621
measurements_per_sec_worker_payload: 49,349
psf_candidates_per_sec_device_submit_sync: 6,843,324
psf_candidate_grid_wall_sec: 0.000694
```

Sparse vs dense comparison:

```text
sparse10:    2,770 measurements,    69,250 PSF candidates,  1,376 measurements/sec payload, 0.85M PSF candidates/sec device
gc_dense2:  35,717 measurements,   892,925 PSF candidates,  7,926 measurements/sec payload, 6.78M PSF candidates/sec device
gc_dense5:  89,211 measurements, 2,230,275 PSF candidates, 37,712 measurements/sec payload, 6.88M PSF candidates/sec device
gc_dense10:178,705 measurements, 4,467,625 PSF candidates, 49,349 measurements/sec payload, 6.84M PSF candidates/sec device
```

Dense 10-frame profile:

```text
fits_read_wall_sec: 1.228
frame_upload_wall_sec: 0.153
aperture_kernel_wall_sec: 0.0310
psf_candidate_grid_wall_sec: 0.000694
psf_kernel_wall_sec: 1.040
psf_device_submit_sync_wall_sec: 0.653
psf_spline_coeff_wall_sec: 0.148
psf_upload_wall_sec: 0.00520
psf_gather_wall_sec: 0.0610
write_wall_sec: 0.690
```

Notes:

- The previous sparse PSF benchmark was strongly underfeeding the GPU. Dense
  real catalog fields moved device-side PSF candidate throughput from about
  0.85M candidates/sec to about 6.8M candidates/sec.
- Aperture work is effectively free compared with PSF and I/O at this density:
  178,705 aperture measurements took about 31 ms of kernel time.
- The next measured hot paths are FITS read, PSF kernel work, and parquet shard
  writes. Candidate-grid setup remains negligible.

## 2026-06-29: Dense PSF Multi-GPU Dispatch Smoke

After adding worker balance metrics to dispatch aggregation, the same dense
Galactic-plane-adjacent workload was run as an 18-frame balanced comparison on
one GPU and three GPUs.

Commands:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky run-dispatch-benchmark-sweep \
  --manifest runs/manifest_galactic_core_nearest20/frame_manifest.parquet \
  --projected-targets runs/projected_targets_galactic_core_nearest20_allcat_g11_16_n20000/frame_targets_projected.parquet \
  --out-dir runs/dispatch_benchmark_psf_gc_dense_real_1gpu18 \
  --run-id dispatch_psf_gc_dense_real_1gpu18 \
  --devices cuda:0 \
  --workers-per-device 1 \
  --limit-frames 18 \
  --shard-batch-frames 6 \
  --prefetch-frames 0 \
  --worker-only \
  --enable-psf \
  --psf-kernel-build-mode gpu_spline

.venv/bin/luxquarry-allsky run-dispatch-benchmark-sweep \
  --manifest runs/manifest_galactic_core_nearest20/frame_manifest.parquet \
  --projected-targets runs/projected_targets_galactic_core_nearest20_allcat_g11_16_n20000/frame_targets_projected.parquet \
  --out-dir runs/dispatch_benchmark_psf_gc_dense_real_3gpu18 \
  --run-id dispatch_psf_gc_dense_real_3gpu18 \
  --devices cuda:0,cuda:1,cuda:2 \
  --workers-per-device 1 \
  --limit-frames 18 \
  --shard-batch-frames 6 \
  --prefetch-frames 0 \
  --worker-only \
  --enable-psf \
  --psf-kernel-build-mode gpu_spline
```

Same-workload comparison:

```text
1gpu18: 320,768 measurements, payload 5.094 sec, total 8.634 sec, payload throughput 62,969 measurements/sec
3gpu18: 320,768 measurements, payload 3.351 sec, total 7.435 sec, payload throughput 95,711 measurements/sec

payload speedup: 1.52x
end-to-end speedup: 1.16x
3-GPU worker parallel efficiency: 95.7%
3-GPU worker wall skew ratio: 1.074
```

3-GPU worker split:

```text
cuda:0: 6 frames, 106,248 rows, 3.153 sec wall, 0.969 sec FITS read, 0.745 sec PSF
cuda:1: 6 frames, 107,214 rows, 3.122 sec wall, 1.005 sec FITS read, 0.737 sec PSF
cuda:2: 6 frames, 107,306 rows, 3.351 sec wall, 1.044 sec FITS read, 0.741 sec PSF
```

Notes:

- The frame modulo partitioning and materialized worker inputs balance cleanly
  across three GPUs on this workload.
- GPU math throughput remains stable: PSF device submit/sync is about
  6.4M-6.7M candidates/sec.
- Scaling is not linear because concurrent FITS reads and parquet shard writes
  become a larger fraction of wall time. The next allsky-engine optimization
  should target local object staging/NVMe cache, asynchronous write pressure,
  and larger worker batches before further PSF kernel tuning.

## 2026-06-29: Dense PSF Local Cache and Prefetch Smoke

The 18-frame, 3-GPU dense workload was rerun with critical-path worker metrics
and with local FITS staging/prefetch variants.

Commands used the same inputs as the previous section and varied only:

```text
uncached_p0:             --prefetch-frames 0
prefetch2_nocache:       --prefetch-frames 2
cache_prefetch2_cold:    --prefetch-frames 2 --local-cache-dir /tmp/luxquarry_gc_dense_cache_prefetch2
cache_prefetch2_warm:    repeat cache_prefetch2_cold with the same populated cache dir
```

Comparison:

```text
uncached_p0:          payload 2.915 sec, total 7.182 sec, 110,046 measurements/sec payload
prefetch2_nocache:    payload 2.791 sec, total 7.192 sec, 114,942 measurements/sec payload
cache_prefetch2_cold: payload 2.829 sec, total 7.195 sec, 113,372 measurements/sec payload
cache_prefetch2_warm: payload 2.177 sec, total 6.446 sec, 147,362 measurements/sec payload

warm-cache payload speedup vs uncached: 1.34x
warm-cache end-to-end speedup vs uncached: 1.11x
```

Critical-path max-worker timings:

```text
                 max_payload_wait  max_staging  max_fits_read  max_psf_kernel  max_write_wait  max_shard_write
uncached_p0              0.726        0.000         0.726          0.730           0.473           0.479
prefetch2_nocache        0.285        0.000         1.348          0.795           0.496           0.501
cache_prefetch2_cold     0.382        1.765         0.137          0.780           0.466           0.473
cache_prefetch2_warm     0.032        0.024         0.144          0.777           0.513           0.519
```

Notes:

- Prefetch reduces worker payload wait, but without a populated local cache it
  mainly moves NAS read work off the critical wait path rather than reducing
  total I/O work.
- Warm local cache plus prefetch is the best measured shape so far. It nearly
  removes payload wait and FITS-read critical-path cost.
- After warm cache/prefetch, the longest worker-path stages are PSF kernel work
  and async shard-write wait. That makes write combining/output strategy the
  next measured systems bottleneck.
- The new benchmark schema records both summed stage timings and critical-path
  max-worker timings. For multi-GPU runs, use max-worker timings to reason
  about wall time; summed timings represent total work across workers.

## 2026-06-29: Dense PSF No-Write Bound

To isolate parquet shard write overhead from photometry and FITS staging, the
warm local-cache/prefetch benchmark was rerun with:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky run-dispatch-benchmark-sweep \
  --manifest runs/manifest_galactic_core_nearest20/frame_manifest.parquet \
  --projected-targets runs/projected_targets_galactic_core_nearest20_allcat_g11_16_n20000/frame_targets_projected.parquet \
  --out-dir runs/dispatch_benchmark_psf_gc_dense_real_3gpu18_cache_prefetch2_discard_warm \
  --run-id dispatch_psf_gc_dense_real_3gpu18_cache_prefetch2_discard_warm \
  --devices cuda:0,cuda:1,cuda:2 \
  --workers-per-device 1 \
  --limit-frames 18 \
  --shard-batch-frames 6 \
  --prefetch-frames 2 \
  --local-cache-dir /tmp/luxquarry_gc_dense_cache_prefetch2 \
  --worker-only \
  --enable-psf \
  --psf-kernel-build-mode gpu_spline \
  --discard-measurement-shards
```

`--discard-measurement-shards` is a benchmark-only mode. It counts measurement
rows and keeps normal photometry execution, but drops frame results before
parquet shard flush. Do not use it for science runs or finalize runs.

Comparison:

```text
warm_write:   payload 2.177 sec, total 6.446 sec, 147,362 measurements/sec payload
warm_discard: payload 1.659 sec, total 5.931 sec, 193,371 measurements/sec payload

payload speedup without writes: 1.31x
end-to-end speedup without writes: 1.09x
critical-path write cost: ~0.518 sec
```

Critical-path max-worker timings:

```text
              max_psf_kernel  max_write_wait  max_shard_write  max_payload_wait
warm_write        0.777          0.513           0.519             0.032
warm_discard      0.761          0.000           0.000             0.028
```

Notes:

- The no-write run establishes a practical upper bound for the current worker
  shape on this workload: about 193k measurements/sec payload across 3 GPUs.
- Parquet shard output is the largest removable critical-path cost after warm
  local-cache/prefetch.
- Next production work should preserve durable output while reducing this cost:
  larger shard batches, fewer/wider shards, per-node writer processes, KvikIO
  evaluation on local NVMe, or writing compact column subsets for first-pass
  mining.

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
  --prefetch-frames 2 \
  --status-interval-frames 5
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

After target grouping, FP32 FITS payloads, and status throttling:

```text
total_wall_sec: 1.151
measurement_rows: 2,770
ok_measurement_rows: 2,766
matched_ok_measurements_vs_cpu: 2,766
flux_p95_abs_delta_uJy_vs_cpu: 0.0253
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
- `--status-interval-frames` reduces status JSON churn on large runs.
- The worker groups targets by frame once and reads FITS image/variance payloads
  as FP32 to avoid repeated target scans and float64 payload churn.
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

Collected run:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky collect-dispatch-run \
  --plan runs/dispatch_smoke10/dispatch_plan.json
```

Result:

```text
complete: true
complete_workers: 3
completed_frames: 10
frame_count: 10
measurement_rows: 2,770
ok_measurement_rows: 2,766
shard_count: 10
missing_shards: 0
worker_max_wall_sec: 1.381
collect_wall_sec: 0.008
```

The collector wrote:

```text
runs/dispatch_smoke10/aggregate_summary.json
runs/dispatch_smoke10/measurement_shard_manifest.parquet
```

Shard manifest validation:

```text
manifest_rows: 10
sum_rows: 2,770
sum_ok_rows: 2,766
sum_frame_count: 10
missing_shards: 0
```

## 2026-06-29: Local FITS Staging Smoke

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky run-persistent-gpu-worker \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/persistent_gpu_worker_smoke10_stage_hash_warm \
  --run-id persistent_smoke10_stage_hash_warm \
  --limit-frames 10 \
  --device cuda:0 \
  --shard-batch-frames 5 \
  --prefetch-frames 2 \
  --status-interval-frames 5 \
  --local-cache-dir /tmp/luxquarry_stage_smoke
```

First-touch staged run:

```text
total_wall_sec: 1.289
staged_bytes: 716.4 MB
staging_wall_sec_sum: 1.306
fits_read_wall_sec_sum: 0.160
measurement_rows: 2,770
ok_measurement_rows: 2,766
```

Warm local-cache run:

```text
total_wall_sec: 0.935
staged_bytes: 0
staging_wall_sec_sum: 0.033
fits_read_wall_sec_sum: 0.145
kernel_wall_sec_sum: 0.235
write_wall_sec_sum: 0.064
measurement_rows: 2,770
ok_measurement_rows: 2,766
```

Correctness against CPU aperture baseline on the warm run:

```text
matched_ok_measurements: 2,766
flux_median_abs_delta_uJy: 0.0022
flux_p95_abs_delta_uJy: 0.0253
flux_max_relative_delta_pct: 0.603
```

Notes:

- `--local-cache-dir` is additive; omitting it preserves the direct-FITS read
  path.
- Each measurement row stores the original `fits_path` and the actual
  `local_fits_path` used for the read.
- The first staged pass is a cache population pass. Warm cache behavior is the
  relevant model for repeated mining/scoring experiments and for nodes with
  local NVMe.

## 2026-06-29: Async Shard Write Smoke

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky run-persistent-gpu-worker \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/persistent_gpu_worker_smoke10_stage_async_final \
  --run-id persistent_smoke10_stage_async_final \
  --limit-frames 10 \
  --device cuda:0 \
  --shard-batch-frames 5 \
  --prefetch-frames 2 \
  --status-interval-frames 5 \
  --local-cache-dir /tmp/luxquarry_stage_smoke \
  --async-shard-writes
```

Result:

```text
total_wall_sec: 0.938
measurement_rows: 2,770
ok_measurement_rows: 2,766
queued_shard_writes_at_completion: 0
async_shard_write_wait_wall_sec: 0.013
shards: 2
```

Correctness against CPU aperture baseline:

```text
matched_ok_measurements: 2,766
flux_median_abs_delta_uJy: 0.0022
flux_p95_abs_delta_uJy: 0.0253
flux_max_relative_delta_pct: 0.603
```

Notes:

- On this tiny smoke, async writes are effectively tied with warm local staging
  alone (`0.938 sec` vs `0.935 sec`).
- The architectural win is that shard writes are no longer inline in the frame
  loop. The worker queues writes, continues processing, then waits before final
  summary emission.
- `plan-gpu-dispatch --async-shard-writes` now passes the flag through to every
  generated persistent worker.

## 2026-06-29: Batched Table Assembly Smoke

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky run-persistent-gpu-worker \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/persistent_gpu_worker_smoke10_stage_async_batchtable \
  --run-id persistent_smoke10_stage_async_batchtable \
  --limit-frames 10 \
  --device cuda:0 \
  --shard-batch-frames 5 \
  --prefetch-frames 2 \
  --status-interval-frames 5 \
  --local-cache-dir /tmp/luxquarry_stage_smoke \
  --async-shard-writes \
  --batch-table-assembly
```

Result:

```text
total_wall_sec: 0.876
measurement_rows: 2,770
ok_measurement_rows: 2,766
shards: 2
queued_shard_writes_at_completion: 0
async_shard_write_wait_wall_sec: 0.170
```

Correctness against the previous GPU async/staged output:

```text
same_columns: true
aperture_flux_uJy_max_delta: 0.0
aperture_flux_unc_uJy_max_delta: 0.0
cwave_um_max_delta: 0.0
cband_um_max_delta: 0.0
flags_summary_max_delta: 0.0
aperture_status_code_max_delta: 0.0
```

Correctness against CPU aperture baseline:

```text
matched_ok_measurements: 2,766
flux_p95_abs_delta_uJy: 0.0253
flux_max_relative_delta_pct: 0.603
```

Notes:

- Per-frame `table_wall_sec` after the first frame drops from roughly
  `0.011 sec` to mostly `0.002 sec` because cuDF construction moves to the
  shard writer.
- The first async shard write was slower internally (`0.395 sec`) because it now
  includes batched table assembly, but most of that work overlapped with later
  frame processing.
- `plan-gpu-dispatch --batch-table-assembly` passes this mode through to every
  generated persistent worker.

## 2026-06-29: Materialized Dispatch Input Smoke

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky plan-gpu-dispatch \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/dispatch_smoke10_materialized2 \
  --run-id dispatch_smoke10_materialized2 \
  --plan-out runs/dispatch_smoke10_materialized2/dispatch_plan.json \
  --devices cuda:0,cuda:1,cuda:2 \
  --shard-batch-frames 5 \
  --prefetch-frames 2 \
  --status-interval-frames 5 \
  --local-cache-dir /tmp/luxquarry_stage_smoke \
  --async-shard-writes \
  --batch-table-assembly \
  --materialize-worker-inputs \
  --limit-frames 10

runs/dispatch_smoke10_materialized2/dispatch_plan.sh

.venv/bin/luxquarry-allsky collect-dispatch-run \
  --plan runs/dispatch_smoke10_materialized2/dispatch_plan.json
```

Materialized inputs:

```text
source_manifest_rows: 10
source_projected_target_rows: 5,000
materialized_manifest_rows: 10
materialized_projected_target_rows: 5,000
worker frame slices: 4 / 3 / 3
worker projected-target slices: 2,000 / 1,500 / 1,500
```

Collected result:

```text
complete: true
complete_workers: 3
completed_frames: 10
measurement_rows: 2,770
ok_measurement_rows: 2,766
shard_count: 3
missing_shards: 0
worker_max_wall_sec: 0.811
worker_sum_wall_sec: 2.397
collect_wall_sec: 0.008
```

Comparison with the earlier non-materialized dispatch:

```text
old_rows: 2,770
new_rows: 2,770
old_ok_rows: 2,766
new_ok_rows: 2,766
aperture_flux_uJy_max_delta: 0.0
aperture_flux_unc_uJy_max_delta: 0.0
flags_summary_max_delta: 0.0
aperture_status_code_max_delta: 0.0
cwave_um_max_delta: 4.8e-7
cband_um_max_delta: 4.3e-8
```

Notes:

- Materialization is an input-startup optimization. It prevents every worker
  from reading the full projected-target parquet just to filter out frames owned
  by other workers.
- Generated worker commands point at `worker_inputs/wXXXX/*.parquet` and run
  with `--worker-index 0 --worker-count 1`; the logical worker index is still
  recorded in the dispatch plan for aggregation.
- This is the preferred dispatch shape for large local, multi-node, or EKS
  execution.

## 2026-06-29: RAPIDS Spectra Assembly Smoke

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky assemble-spectra \
  --shard-manifest runs/dispatch_smoke10_materialized2/measurement_shard_manifest.parquet \
  --out-dir runs/dispatch_smoke10_materialized2/spectra_fast \
  --run-id dispatch_smoke10_materialized2_fast \
  --device cuda:0
```

Result:

```text
backend: cudf_spectra_assembly
shard_count: 3
input_measurement_rows: 2,770
spectra_measurement_rows: 2,770
target_count: 720
read_shards_wall_sec: 0.151
sort_wall_sec: 0.041
write_spectra_wall_sec: 0.039
target_summary_wall_sec: 0.152
total_wall_sec: 1.248
```

Validation:

```text
spectra_rows: 2,770
target_summary_rows: 720
ok_rows: 2,766
spectra_sort_order: catalog,target_id,cwave_um,frame_group_id,image_id
summary_measurement_count_sum: 2,770
summary_ok_measurement_count_sum: 2,766
```

`--only-ok` result:

```text
spectra_measurement_rows: 2,766
target_count: 719
total_wall_sec: 1.242
```

Notes:

- The spectra product is still ragged. It does not resample or interpolate; it
  preserves one row per calibrated measurement.
- A first target-summary implementation used repeated groupby/merge passes and
  took `6.47 sec` on this tiny smoke. The current implementation uses one cuDF
  groupby aggregation and reduced target summary time to `0.15 sec`.
- The outputs are:

```text
<run_id>.spectra_measurements.parquet
<run_id>.target_summary.parquet
assemble_summary.json
```

## 2026-06-29: Kubernetes Job Manifest Smoke

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky write-k8s-jobs \
  --plan runs/dispatch_smoke10_materialized2/dispatch_plan.json \
  --out-dir runs/dispatch_smoke10_materialized2/k8s \
  --image luxquarry-allsky:local \
  --namespace luxquarry \
  --container-executable luxquarry-allsky \
  --working-dir /workspace/luxquarry_allsky_engine \
  --pvc-name luxquarry-data \
  --mount-path /workspace \
  --env LUXQUARRY_MODE=smoke
```

Result:

```text
backend: kubernetes_job_manifest_generator
run_id: dispatch_smoke10_materialized2
job_count: 3
namespace: luxquarry
image: luxquarry-allsky:local
materialize_worker_inputs: true
manifest: runs/dispatch_smoke10_materialized2/k8s/dispatch_smoke10_materialized2.worker-jobs.yaml
```

Structural validation:

```text
documents: 3
kind: Job
command: luxquarry-allsky
args include: run-persistent-gpu-worker
gpu request/limit: nvidia.com/gpu=1
workingDir: /workspace/luxquarry_allsky_engine
volume mount: /workspace
runtime worker args: --worker-index 0 --worker-count 1
```

Notes:

- The manifest generator is dependency-free. It writes JSON-shaped YAML
  documents, which Kubernetes accepts as YAML and local tests can parse with the
  Python standard library.
- The generated Jobs are a dispatch artifact for baseline measurement workers.
  A science campaign still requires spectra assembly, blind scoring,
  injected-run scoring, truth-target recovery, and candidate/false-positive
  review products.

## 2026-06-29: Campaign Contract Smoke

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky write-campaign-contract \
  --campaign-id dispatch_smoke10_materialized2_contract \
  --out runs/dispatch_smoke10_materialized2/campaign_contract.json \
  --baseline-plan runs/dispatch_smoke10_materialized2/dispatch_plan.json \
  --baseline-spectra-dir runs/dispatch_smoke10_materialized2/spectra_fast
```

Result:

```text
backend: luxquarry_campaign_contract
stage_count: 8
complete_stage_count: 2
missing_stage_count: 6
science_complete: false
baseline_run_id: dispatch_smoke10_materialized2
baseline_spectra_run_id: dispatch_smoke10_materialized2_fast
```

Complete stages:

```text
baseline_dispatch
baseline_spectra_assembly
```

Missing or blocked stages:

```text
baseline_blind_scoring
injected_dispatch
injected_spectra_assembly
injected_blind_scoring
truth_target_recovery
viewer_indexes
```

Notes:

- The contract reads `assemble_summary.json` so spectra run IDs can differ from
  dispatch run IDs without breaking artifact detection.
- This is only a status/contract layer. It does not implement injection or
  recovery yet, but it prevents the all-sky engine from treating baseline
  photometry as a complete science campaign.

## 2026-06-29: Dispatch Finalizer Smoke

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky finalize-dispatch-run \
  --plan runs/dispatch_smoke10_materialized2/dispatch_plan.json \
  --spectra-out-dir runs/dispatch_smoke10_materialized2/spectra_finalize_smoke \
  --spectra-run-id dispatch_smoke10_materialized2_finalize \
  --campaign-id dispatch_smoke10_materialized2_finalize \
  --campaign-contract-out runs/dispatch_smoke10_materialized2/campaign_contract_finalize.json \
  --device cuda:0
```

Result:

```text
backend: luxquarry_finalize_dispatch
dispatch_complete: true
measurement_rows: 2,770
spectra_measurement_rows: 2,770
target_count: 720
science_complete: false
total_wall_sec: 1.247
```

Sub-stage timings:

```text
collect_wall_sec: 0.0079
read_shards_wall_sec: 0.1510
sort_wall_sec: 0.0454
write_spectra_wall_sec: 0.0389
target_summary_wall_sec: 0.1465
spectra_total_wall_sec: 1.237
```

Complete campaign-contract stages:

```text
baseline_dispatch
baseline_spectra_assembly
```

Missing or blocked stages:

```text
baseline_blind_scoring
injected_dispatch
injected_spectra_assembly
injected_blind_scoring
truth_target_recovery
viewer_indexes
```

Kubernetes postprocess manifest smoke:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky write-k8s-postprocess-job \
  --plan runs/dispatch_smoke10_materialized2/dispatch_plan.json \
  --out-dir runs/dispatch_smoke10_materialized2/k8s_postprocess \
  --image luxquarry-allsky:local \
  --namespace luxquarry \
  --container-executable luxquarry-allsky \
  --working-dir /workspace/luxquarry_allsky_engine \
  --pvc-name luxquarry-data \
  --mount-path /workspace \
  --campaign-id dispatch_smoke10_materialized2_finalize
```

Validation:

```text
documents: 1
kind: Job
command: luxquarry-allsky
args include: finalize-dispatch-run
gpu request/limit: nvidia.com/gpu=1
workingDir: /workspace/luxquarry_allsky_engine
```

Notes:

- `finalize-dispatch-run` is the preferred local and Kubernetes post-worker
  handoff because it keeps collection, spectra assembly, and campaign-contract
  writing together.
- The command fails unless the dispatch aggregate reports complete worker and
  shard outputs. `--allow-incomplete` is diagnostic-only.
- The generated postprocess Job assumes paths are relative to the configured
  container working directory. With `/workspace/luxquarry_allsky_engine`, use
  `runs/...` paths.

## 2026-06-29: Local Dispatch Runner Smoke

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky run-local-dispatch \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/local_dispatch_smoke2 \
  --run-id local_dispatch_smoke2 \
  --devices cuda:0 \
  --limit-frames 2 \
  --shard-batch-frames 2 \
  --prefetch-frames 1 \
  --status-interval-frames 2 \
  --local-cache-dir /tmp/luxquarry_stage_smoke \
  --async-shard-writes \
  --batch-table-assembly \
  --finalize-device cuda:0
```

Result:

```text
backend: luxquarry_local_dispatch_runner
status: complete
devices: cuda:0
worker_count: 1
failed_worker_count: 0
materialize_worker_inputs: true
plan_wall_sec: 0.027
worker_wall_sec: 2.985
finalize_wall_sec: 7.011
total_wall_sec: 10.023
measurement_rows: 573
spectra_measurement_rows: 573
target_count: 310
```

Worker aggregate:

```text
completed_frames: 2
worker_summary_wall_sec: 0.429
measurement_rows: 573
ok_measurement_rows: 572
shard_count: 1
missing_shards: 0
```

Outputs:

```text
runs/local_dispatch_smoke2/dispatch_plan.json
runs/local_dispatch_smoke2/worker_inputs/w0000/*.parquet
runs/local_dispatch_smoke2/worker_logs/*.log
runs/local_dispatch_smoke2/aggregate_summary.json
runs/local_dispatch_smoke2/measurement_shard_manifest.parquet
runs/local_dispatch_smoke2/spectra/*.parquet
runs/local_dispatch_smoke2/campaign_contract.json
runs/local_dispatch_smoke2/local_dispatch_summary.json
```

Notes:

- The local runner launches the exact worker argv stored in the dispatch plan.
  That keeps local testing aligned with the shell and Kubernetes paths.
- Worker stdout/stderr are redirected to files, avoiding pipe backpressure and
  preserving logs for failed-worker review.
- The small smoke paid cold process and cuDF/groupby startup costs, especially
  in target-summary assembly. Longer frame queues should amortize this.

Resume command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky run-local-dispatch \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/local_dispatch_smoke2 \
  --run-id local_dispatch_smoke2 \
  --devices cuda:0 \
  --limit-frames 2 \
  --shard-batch-frames 2 \
  --prefetch-frames 1 \
  --status-interval-frames 2 \
  --local-cache-dir /tmp/luxquarry_stage_smoke \
  --async-shard-writes \
  --batch-table-assembly \
  --finalize-device cuda:0 \
  --resume
```

Resume result:

```text
resume: true
launched_worker_count: 0
skipped_worker_count: 1
worker_wall_sec: 0.0001
finalize_wall_sec: 1.202
total_wall_sec: 1.229
measurement_rows: 573
spectra_measurement_rows: 573
target_count: 310
```

Resume mode skips workers only when their `run_summary.json` has
`completed_utc` and zero failed frames. It still rebuilds the plan and reruns
finalization so aggregate summaries, spectra, and campaign contracts are fresh.

## 2026-06-29: Dispatch Status Snapshot Smoke

Commands:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky dispatch-status \
  --plan runs/local_dispatch_smoke2/dispatch_plan.json

.venv/bin/luxquarry-allsky dispatch-status \
  --plan runs/dispatch_smoke10_materialized2/dispatch_plan.json \
  --out runs/dispatch_smoke10_materialized2/dispatch_status_custom.json
```

Local one-worker result:

```text
complete: true
worker_count: 1
complete_workers: 1
completed_frames: 2
frame_count: 2
measurement_rows: 573
ok_measurement_rows: 572
snapshot_wall_sec: 0.0003
```

Materialized three-worker result:

```text
complete: true
worker_count: 3
complete_workers: 3
completed_frames: 10
frame_count: 10
measurement_rows: 2,770
ok_measurement_rows: 2,766
snapshot_wall_sec: 0.0006
```

Notes:

- The snapshot reads per-worker `run_status.json` first and falls back to
  `run_summary.json`.
- It writes atomically through a temporary file.
- This is the intended low-overhead dashboard/control-plane status source. It
  does not read measurement shards or parquet data.

## 2026-06-29: Local Runner Integrated Status Smoke

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky run-local-dispatch \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/local_dispatch_status_smoke2 \
  --run-id local_dispatch_status_smoke2 \
  --devices cuda:0 \
  --limit-frames 2 \
  --shard-batch-frames 2 \
  --prefetch-frames 1 \
  --status-interval-frames 1 \
  --status-snapshot-interval-sec 0.1 \
  --local-cache-dir /tmp/luxquarry_stage_smoke \
  --async-shard-writes \
  --batch-table-assembly \
  --finalize-device cuda:0
```

Result:

```text
status: complete
worker_count: 1
launched_worker_count: 1
status_snapshot_interval_sec: 0.1
worker_wall_sec: 3.064
finalize_wall_sec: 1.158
total_wall_sec: 4.249
measurement_rows: 573
target_count: 310
```

Automatically written snapshot:

```text
path: runs/local_dispatch_status_smoke2/dispatch_status.json
complete: true
worker_count: 1
completed_frames: 2
frame_count: 2
measurement_rows: 573
snapshot_wall_sec: 0.0002
```

Notes:

- The local runner now refreshes `dispatch_status.json` while worker processes
  run and writes a final snapshot after they exit.
- `--status-snapshot-interval-sec 0` disables periodic refresh and keeps the
  final snapshot only.

## 2026-06-29: Baseline Candidate Scoring Smoke

Standalone scorer command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky score-spectra-candidates \
  --spectra runs/local_dispatch_smoke2/spectra/local_dispatch_smoke2.spectra_measurements.parquet \
  --out-dir runs/local_dispatch_smoke2/candidates_score_smoke \
  --run-id local_dispatch_smoke2 \
  --device cuda:0 \
  --min-measurements 2 \
  --min-abs-zscore 0.5 \
  --max-candidates 20
```

Result:

```text
backend: cudf_simple_target_zscore_scorer
input_measurement_rows: 573
filtered_measurement_rows: 572
target_count: 310
candidate_count_before_cap: 524
candidate_count: 20
total_wall_sec: 1.328
read_spectra_wall_sec: 0.136
score_wall_sec: 0.266
write_candidates_wall_sec: 0.024
```

Finalizer scorer command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky finalize-dispatch-run \
  --plan runs/local_dispatch_smoke2/dispatch_plan.json \
  --spectra-out-dir runs/local_dispatch_smoke2/spectra_scored_finalize \
  --spectra-run-id local_dispatch_smoke2_scored_finalize \
  --campaign-id local_dispatch_smoke2_scored_finalize \
  --campaign-contract-out runs/local_dispatch_smoke2/campaign_contract_scored_finalize.json \
  --candidate-dir runs/local_dispatch_smoke2/candidates_scored_finalize \
  --score-baseline \
  --candidate-min-measurements 2 \
  --candidate-min-abs-zscore 0.5 \
  --candidate-max-rows 20 \
  --device cuda:0
```

Result:

```text
dispatch_complete: true
science_complete: false
measurement_rows: 573
spectra_measurement_rows: 573
target_count: 310
baseline_candidate_count: 20
baseline_scorer_wall_sec: 0.128
finalize_total_wall_sec: 1.340
complete campaign stages: 3 / 8
```

Notes:

- The scoring threshold was intentionally loose because this is a two-frame
  smoke run. Normal runs should use deeper spectra and stricter thresholds.
- The campaign contract now marks `baseline_blind_scoring` complete when
  `--score-baseline` is used.
- Injected dispatch, injected scoring, and truth-target recovery remain
  explicit missing/blocked stages. They were not silently folded into baseline
  scoring.

## 2026-06-29: Injected Scoring and Recovery Smoke

Standalone recovery smoke used a tiny synthetic injection manifest derived from
one emitted candidate. This only verifies the manifest/candidate join and output
contract; it is not a science recovery benchmark.

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky score-injection-recovery \
  --manifest runs/local_dispatch_smoke2/recovery_smoke_manifest.json \
  --candidates runs/local_dispatch_smoke2/candidates_scored_finalize/baseline_candidates.parquet \
  --out-dir runs/local_dispatch_smoke2/recovery_smoke \
  --min-score 0.5 \
  --wavelength-tolerance-nm 1.0
```

Result:

```text
injection_count: 1
recovered_count: 1
missed_count: 0
candidate_count_above_threshold: 20
false_positive_count: 19
total_wall_sec: 0.036
```

Full finalizer smoke reused the existing smoke spectra as a stand-in injected
spectra directory:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky finalize-dispatch-run \
  --plan runs/local_dispatch_smoke2/dispatch_plan.json \
  --spectra-out-dir runs/local_dispatch_smoke2/spectra_recovery_finalize \
  --spectra-run-id local_dispatch_smoke2_recovery_finalize \
  --campaign-id local_dispatch_smoke2_recovery_finalize \
  --campaign-contract-out runs/local_dispatch_smoke2/campaign_contract_recovery_finalize.json \
  --candidate-dir runs/local_dispatch_smoke2/candidates_recovery_finalize \
  --score-baseline \
  --score-injected \
  --injected-spectra-dir runs/local_dispatch_smoke2/spectra_scored_finalize \
  --injection-truth runs/local_dispatch_smoke2/recovery_smoke_manifest.json \
  --recover-injections \
  --candidate-min-measurements 2 \
  --candidate-min-abs-zscore 0.5 \
  --candidate-max-rows 20 \
  --recovery-min-score 0.5 \
  --recovery-wavelength-tolerance-nm 1.0 \
  --device cuda:0
```

Result:

```text
dispatch_complete: true
science_complete: false
complete campaign stages: 6 / 8
baseline_candidate_count: 20
injected_candidate_count: 20
injection_count: 1
recovered_count: 1
false_positive_count: 19
baseline_scorer_wall_sec: 0.124
injected_scorer_wall_sec: 0.045
recovery_wall_sec: 0.022
finalize_total_wall_sec: 1.424
```

Notes:

- The finalizer now writes baseline candidates, injected candidates, truth
  recovery, and false-positive summaries when the relevant flags and inputs are
  supplied.
- The campaign contract still marks the run incomplete unless injected dispatch
  and viewer indexes exist. In this smoke, injected spectra were supplied but
  no injected dispatch plan was supplied.

## 2026-06-29: Dispatch Benchmark Sweep Harness Smoke

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky run-dispatch-benchmark-sweep \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/dispatch_benchmark_sweep_smoke \
  --run-id dispatch_sweep_smoke \
  --devices cuda:0 \
  --workers-per-device 1 \
  --limit-frames 2 \
  --shard-batch-frames 1,2 \
  --prefetch-frames 0 \
  --repetitions 1 \
  --local-cache-dir /tmp/luxquarry_stage_smoke \
  --score-baseline \
  --candidate-min-measurements 2 \
  --candidate-min-abs-zscore 0.5 \
  --candidate-max-rows 20 \
  --status-snapshot-interval-sec 0
```

Outputs:

```text
runs/dispatch_benchmark_sweep_smoke/sweep_results.parquet
runs/dispatch_benchmark_sweep_smoke/sweep_results.json
runs/dispatch_benchmark_sweep_smoke/profile_summary.parquet
runs/dispatch_benchmark_sweep_smoke/profile_summary.json
runs/dispatch_benchmark_sweep_smoke/perf_summary.json
```

Trial summary:

```text
trial s1:
  measurement_rows: 573
  total_wall_sec: 4.583
  worker_wall_sec: 3.252
  worker_payload_wall_sec: 0.425
  worker_launch_overhead_sec: 2.828
  finalize_wall_sec: 1.305
  measurements_per_sec_total: 125.0
  measurements_per_sec_worker_payload: 1,349.1

trial s2:
  measurement_rows: 573
  total_wall_sec: 3.121
  worker_wall_sec: 3.016
  worker_payload_wall_sec: 0.439
  worker_launch_overhead_sec: 2.576
  finalize_wall_sec: 0.092
  measurements_per_sec_total: 183.6
  measurements_per_sec_worker_payload: 1,303.9
```

Profile rows above 5%:

```text
s1 worker_phase: 70.96%
s1 finalize_total: 28.47%
s1 assemble_spectra: 25.61%
s2 worker_phase: 96.64%
```

Notes:

- This is a harness smoke, not a saturation benchmark.
- The tiny two-frame workload is dominated by subprocess/worker launch and
  RAPIDS startup costs. The measured worker payload is roughly 0.43 sec, but
  the worker phase seen by the parent is roughly 3.0 sec.
- `shard_batch_frames=2` wins end-to-end on this tiny test because it avoids
  extra shard/finalize overhead.
- The next real benchmark should sweep at least 10, 50, and 100 frames on all
  local GPUs. That is the first point where setup overhead should stop hiding
  GPU and storage throughput.

## 2026-06-29: Worker-Only Dispatch Sweep Smoke

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky run-dispatch-benchmark-sweep \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/dispatch_benchmark_worker_only_smoke \
  --run-id dispatch_worker_only_smoke \
  --devices cuda:0 \
  --workers-per-device 1 \
  --limit-frames 2 \
  --shard-batch-frames 1,2 \
  --prefetch-frames 0 \
  --repetitions 1 \
  --local-cache-dir /tmp/luxquarry_stage_smoke \
  --status-snapshot-interval-sec 0 \
  --worker-only
```

Trial summary:

```text
trial s1:
  measurement_rows: 573
  total_wall_sec: 3.298
  worker_wall_sec: 3.269
  worker_payload_wall_sec: 0.401
  worker_launch_overhead_sec: 2.869
  collect_wall_sec: 0.002
  measurements_per_sec_total: 173.7
  measurements_per_sec_worker_payload: 1,429.7

trial s2:
  measurement_rows: 573
  total_wall_sec: 3.267
  worker_wall_sec: 3.252
  worker_payload_wall_sec: 0.438
  worker_launch_overhead_sec: 2.815
  collect_wall_sec: 0.002
  measurements_per_sec_total: 175.4
  measurements_per_sec_worker_payload: 1,309.5
```

Profile rows:

```text
s1 worker_phase: 99.12%
s2 worker_phase: 99.55%
```

Notes:

- Worker-only mode confirms spectra assembly/scoring are not the dominant
  cost for tiny jobs. The parent process still sees roughly 3 seconds because
  launching a fresh worker process dominates.
- The actual persistent-worker payload is roughly 0.40-0.42 sec for this
  two-frame workload.
- This supports two next steps:
  1. benchmark much larger frame batches, where launch cost is amortized;
  2. design the next version as a long-lived worker service or queue consumer,
     not one subprocess per small dispatch.

## 2026-06-29: Local Task Queue Service Smoke

Commands:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky write-task-queue \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/service_queue_smoke_v1/queue \
  --campaign-id service_queue_smoke_v1 \
  --frames-per-task 1 \
  --limit-frames 2

.venv/bin/luxquarry-allsky run-gpu-worker-service \
  --queue-dir runs/service_queue_smoke_v1/queue \
  --out-dir runs/service_queue_smoke_v1 \
  --run-id service_queue_smoke_v1 \
  --worker-id worker-cuda0 \
  --device cuda:0 \
  --local-cache-dir /tmp/luxquarry_stage_service_smoke \
  --max-tasks 2 \
  --async-shard-writes \
  --batch-table-assembly \
  --shard-batch-frames 1

.venv/bin/luxquarry-allsky collect-task-queue-run \
  --queue-dir runs/service_queue_smoke_v1/queue \
  --out runs/service_queue_smoke_v1/aggregate_summary.json

.venv/bin/luxquarry-allsky assemble-spectra \
  --shard-manifest runs/service_queue_smoke_v1/measurement_shard_manifest.parquet \
  --out-dir runs/service_queue_smoke_v1/spectra \
  --run-id service_queue_smoke_v1 \
  --device cuda:0
```

Result:

```text
queue_writer:
  frame_count: 2
  task_count: 2
  frames_per_task: 1
  write_wall_sec: 0.030

worker_service:
  tasks_completed: 2
  tasks_failed: 0
  frames_completed: 2
  measurement_rows: 573
  ok_measurement_rows: 572
  summary_total_wall_sec: 1.942
  shell_observed_wall_sec: 3.199
  task_wall_sum_sec: 0.682

task_000000:
  task_wall_sec: 0.502
  frame_compute_wall_sec: 0.200
  kernel_wall_sec: 0.033
  table_wall_sec: 0.060
  staging_wall_sec: 0.112
  write_wall_sec: 0.152

task_000001:
  task_wall_sec: 0.181
  frame_compute_wall_sec: 0.028
  kernel_wall_sec: 0.026
  table_wall_sec: 0.002
  staging_wall_sec: 0.106
  write_wall_sec: 0.022

collect:
  complete: true
  shard_count: 2
  collect_wall_sec: 0.023

assemble_spectra:
  input_measurement_rows: 573
  target_count: 310
  total_wall_sec: 1.249
```

Notes:

- The service queue produced the same downstream measurement shard manifest
  shape as dispatch, and `assemble-spectra` consumed it unchanged.
- The first task still pays warmup/calibration/table overhead. The second task
  shows the intended direction: the same worker process and resident GPU
  calibration cache continue without a new subprocess.
- This prototype still materializes per-task parquet inputs and calls the
  existing worker run method per task. It proves the service contract, not the
  final monolithic frame miner.

## 2026-06-29: Service Frame-Batch API Smoke

Change:

- `PersistentGpuFrameWorker.run()` now delegates to a reusable
  `process_frame_batch()` method.
- `run-gpu-worker-service` reads each task's parquet inputs and calls
  `process_frame_batch()` directly, so the service has a lower-level resident
  worker API to build on.

Command shape:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky write-task-queue \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/service_queue_smoke_v2/queue \
  --campaign-id service_queue_smoke_v2 \
  --frames-per-task 1 \
  --limit-frames 2

.venv/bin/luxquarry-allsky run-gpu-worker-service \
  --queue-dir runs/service_queue_smoke_v2/queue \
  --out-dir runs/service_queue_smoke_v2 \
  --run-id service_queue_smoke_v2 \
  --worker-id worker-cuda0 \
  --device cuda:0 \
  --local-cache-dir /tmp/luxquarry_stage_service_smoke \
  --max-tasks 2 \
  --async-shard-writes \
  --batch-table-assembly \
  --shard-batch-frames 1
```

Result:

```text
worker_service:
  tasks_completed: 2
  tasks_failed: 0
  frames_completed: 2
  measurement_rows: 573
  ok_measurement_rows: 572
  summary_total_wall_sec: 1.688
  shell_observed_wall_sec: 2.927
  task_wall_sum_sec: 0.466

task_000000:
  task_input_read_wall_sec: 0.016
  task_wall_sec: 0.389
  frame_compute_wall_sec: 0.190
  staging_wall_sec: 0.005
  kernel_wall_sec: 0.033
  table_wall_sec: 0.060
  write_wall_sec: 0.156

task_000001:
  task_input_read_wall_sec: 0.005
  task_wall_sec: 0.077
  frame_compute_wall_sec: 0.028
  staging_wall_sec: 0.002
  kernel_wall_sec: 0.026
  table_wall_sec: 0.002
  write_wall_sec: 0.023

collect:
  complete: true
  shard_count: 2
  collect_wall_sec: 0.009

assemble_spectra:
  input_measurement_rows: 573
  target_count: 310
  total_wall_sec: 1.258
```

Comparison to previous service smoke:

```text
service summary wall: 1.942 sec -> 1.688 sec
task wall sum:        0.682 sec -> 0.466 sec
shell observed wall:  3.199 sec -> 2.927 sec
measurement rows:    unchanged at 573
target spectra:      unchanged at 310
```

Notes:

- The warm second task is now roughly 77 ms wall time in this tiny smoke.
- FITS staging is near zero because the same local SSD cache path was warm.
- This still is not the final frame miner: per-task parquet slices remain, and
  the next target is a service-owned manifest/target index plus larger frame
  batches so the worker is not constantly reopening tiny parquet files.

## 2026-06-29: Lightweight Queue / Resident Source Table Smoke

Change:

- `write-task-queue --no-materialize-task-inputs` now writes task JSON with
  frame IDs only.
- `run-gpu-worker-service` loads the source frame manifest and projected target
  table once at startup for lightweight queues, then slices resident DataFrames
  per task.
- Existing materialized task queues remain supported for compatibility.

Command shape:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky write-task-queue \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/service_queue_smoke_v3/queue \
  --campaign-id service_queue_smoke_v3 \
  --frames-per-task 1 \
  --limit-frames 2 \
  --no-materialize-task-inputs

.venv/bin/luxquarry-allsky run-gpu-worker-service \
  --queue-dir runs/service_queue_smoke_v3/queue \
  --out-dir runs/service_queue_smoke_v3 \
  --run-id service_queue_smoke_v3 \
  --worker-id worker-cuda0 \
  --device cuda:0 \
  --local-cache-dir /tmp/luxquarry_stage_service_smoke \
  --max-tasks 2 \
  --async-shard-writes \
  --batch-table-assembly \
  --shard-batch-frames 1
```

Result:

```text
queue_writer:
  materialized_task_inputs: false
  frame_count: 2
  task_count: 2
  write_wall_sec: 0.015

worker_service:
  materialized_task_inputs: false
  resident_source_inputs: true
  source_input_load_wall_sec: 0.019
  tasks_completed: 2
  tasks_failed: 0
  frames_completed: 2
  measurement_rows: 573
  ok_measurement_rows: 572
  summary_total_wall_sec: 1.685
  shell_observed_wall_sec: 2.925
  task_wall_sum_sec: 0.442

task_000000:
  task_input_read_wall_sec: 0.000
  task_input_select_wall_sec: 0.001
  task_wall_sec: 0.371
  frame_compute_wall_sec: 0.194
  kernel_wall_sec: 0.031
  table_wall_sec: 0.064
  write_wall_sec: 0.154

task_000001:
  task_input_read_wall_sec: 0.000
  task_input_select_wall_sec: 0.001
  task_wall_sec: 0.071
  frame_compute_wall_sec: 0.028
  kernel_wall_sec: 0.026
  table_wall_sec: 0.002
  write_wall_sec: 0.021

collect:
  complete: true
  shard_count: 2
  collect_wall_sec: 0.010

assemble_spectra:
  input_measurement_rows: 573
  target_count: 310
  total_wall_sec: 1.242
```

Comparison:

```text
materialized v2 task wall sum: 0.466 sec
resident-table v3 task wall sum: 0.442 sec
materialized v2 queue write:    0.030 sec
resident-table v3 queue write:  0.015 sec
measurement rows:               unchanged at 573
target spectra:                 unchanged at 310
```

Notes:

- This removes per-task parquet materialization and per-task parquet reads from
  service mode.
- The tiny smoke only shows a small absolute gain because FITS/kernel/shard
  overhead dominate two single-frame tasks. The benefit should matter more for
  large queues because task creation becomes metadata-only and workers keep the
  source tables resident.
- The next target is larger frame batches and multi-worker service mode, then
  S3/object-cache input support.

## 2026-06-29: Shuffled Shard Assembly Validation Smoke

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky validate-assembly-order \
  --shard-manifest runs/service_queue_smoke_v3/measurement_shard_manifest.parquet \
  --out-dir runs/service_queue_smoke_v3/assembly_order_validation_v2 \
  --run-id service_queue_smoke_v3_order_check_v2 \
  --device cuda:0 \
  --repetitions 3
```

Result:

```text
passed: true
mismatch_count: 0
shard_count: 2
repetitions: 3
baseline_spectra_hash: b18a15d5b41dfd85c0e15940f6aa81e3b33f2776b579f9e16fb1e23c306281fd
baseline_target_summary_hash: 1a62c5839294a1569fd8c23e75b82b97210a3b4dfa306957799d5d86d9c909ca

all runs:
  input_measurement_rows: 573
  spectra_measurement_rows: 573
  target_count: 310
```

Notes:

- `validate-assembly-order` assembles the original shard manifest plus shuffled
  copies, then compares logical parquet hashes for spectra measurements and
  target summaries.
- This gives us a regression gate for the S3/EKS path, where worker completion
  and shard listing order cannot be assumed.

## 2026-06-29: Retry Duplicate Assembly Dedup Smoke

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky validate-assembly-retry-dedup \
  --shard-manifest runs/service_queue_smoke_v3/measurement_shard_manifest.parquet \
  --out-dir runs/service_queue_smoke_v3/retry_dedup_validation \
  --run-id service_queue_smoke_v3_retry_dedup \
  --device cuda:0
```

Result:

```text
passed: true
source_shard_count: 2
duplicate_shard_count: 2
duplicated_manifest_shard_count: 4

baseline input_measurement_rows: 573
baseline duplicate_measurement_rows_dropped: 0
baseline spectra_measurement_rows: 573
baseline target_count: 310

duplicated input_measurement_rows: 1,146
duplicated duplicate_measurement_rows_dropped: 573
duplicated spectra_measurement_rows: 573
duplicated target_count: 310

spectra_hash:
  b18a15d5b41dfd85c0e15940f6aa81e3b33f2776b579f9e16fb1e23c306281fd
target_summary_hash:
  1a62c5839294a1569fd8c23e75b82b97210a3b4dfa306957799d5d86d9c909ca
```

Notes:

- `assemble-spectra --drop-duplicate-measurements` drops retry duplicates using
  `catalog`, `target_id`, `frame_group_id`, `image_id`, and `detector`.
- This remains opt-in so historical local assemblies do not silently change.
- EKS/S3 postprocess should enable this flag because retrying task leases can
  legitimately re-emit a shard.

## 2026-06-29: Public S3 FITS Staging Smoke

Manifest rewrite:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky rewrite-manifest-paths \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --out runs/manifest_smoke_v2_s3/frame_manifest.parquet \
  --strip-prefix /mnt/niroseti/spherex_cache/raw \
  --uri-prefix s3://nasa-irsa-spherex
```

First rewritten path:

```text
s3://nasa-irsa-spherex/qr2/level2/2025W52_2A/l2b-v22-2026-006/5/level2_2025W52_2A_0305_3D5_spx_l2b-v22-2026-006.fits
```

S3 HEAD/content-length probe:

```text
content_length_bytes: 71,637,120
```

Worker smoke:

```bash
.venv/bin/luxquarry-allsky write-task-queue \
  --manifest runs/manifest_smoke_v2_s3/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/service_queue_smoke_s3_source/queue \
  --campaign-id service_queue_smoke_s3_source \
  --frames-per-task 1 \
  --limit-frames 1 \
  --no-materialize-task-inputs

.venv/bin/luxquarry-allsky run-gpu-worker-service \
  --queue-dir runs/service_queue_smoke_s3_source/queue \
  --out-dir runs/service_queue_smoke_s3_source \
  --run-id service_queue_smoke_s3_source \
  --worker-id worker-cuda0 \
  --device cuda:0 \
  --local-cache-dir /tmp/luxquarry_s3_stage_smoke \
  --max-tasks 1 \
  --async-shard-writes \
  --batch-table-assembly \
  --shard-batch-frames 1
```

Result:

```text
tasks_completed: 1
tasks_failed: 0
frames_completed: 1
measurement_rows: 280
ok_measurement_rows: 280
staged_bytes: 71,637,120
staging_wall_sec: 3.611
fits_read_wall_sec: 0.014
frame_compute_wall_sec: 0.196
kernel_wall_sec: 0.033
task_wall_sec: 3.982
service_wall_sec: 5.258
```

Assembly:

```bash
.venv/bin/luxquarry-allsky assemble-spectra \
  --shard-manifest runs/service_queue_smoke_s3_source/measurement_shard_manifest.parquet \
  --out-dir runs/service_queue_smoke_s3_source/spectra \
  --run-id service_queue_smoke_s3_source \
  --device cuda:0 \
  --drop-duplicate-measurements
```

Assembly result:

```text
input_measurement_rows: 280
spectra_measurement_rows: 280
target_count: 280
duplicate_measurement_rows_dropped: 0
total_wall_sec: 1.236
```

Notes:

- S3 paths require `--local-cache-dir`; the worker stages the object before
  Astropy reads it.
- The first implementation uses dependency-free anonymous HTTPS derived from
  `s3://bucket/key`.
- The next performance step is asynchronous S3 prefetch so downloads overlap
  with GPU work instead of blocking the frame loop.

## 2026-06-29: S3 Prefetch Width Smoke

Input:

```text
manifest: runs/manifest_smoke_v2_s3/frame_manifest.parquet
frames: 2
frame bytes: 71,637,120 each
task shape: one two-frame lightweight task
```

Prefetch width 1:

```bash
.venv/bin/luxquarry-allsky run-gpu-worker-service \
  --queue-dir runs/service_queue_smoke_s3_prefetch2/queue \
  --out-dir runs/service_queue_smoke_s3_prefetch2 \
  --run-id service_queue_smoke_s3_prefetch2 \
  --worker-id worker-cuda0 \
  --device cuda:0 \
  --local-cache-dir /tmp/luxquarry_s3_stage_prefetch_smoke \
  --max-tasks 1 \
  --async-shard-writes \
  --batch-table-assembly \
  --shard-batch-frames 1 \
  --prefetch-frames 1
```

Result:

```text
frames_completed: 2
measurement_rows: 573
service_wall_sec: 21.534
task_wall_sec: 20.289
frame0 staging_wall_sec: 15.682
frame1 staging_wall_sec: 4.521
```

Prefetch width 2:

```bash
.venv/bin/luxquarry-allsky run-gpu-worker-service \
  --queue-dir runs/service_queue_smoke_s3_prefetch2w/queue \
  --out-dir runs/service_queue_smoke_s3_prefetch2w \
  --run-id service_queue_smoke_s3_prefetch2w \
  --worker-id worker-cuda0 \
  --device cuda:0 \
  --local-cache-dir /tmp/luxquarry_s3_stage_prefetch2w_smoke \
  --max-tasks 1 \
  --async-shard-writes \
  --batch-table-assembly \
  --shard-batch-frames 1 \
  --prefetch-frames 2
```

Result:

```text
frames_completed: 2
measurement_rows: 573
service_wall_sec: 9.249
task_wall_sec: 7.978
frame0 staging_wall_sec: 7.581
frame1 staging_wall_sec: 3.842
assembly target_count: 310
```

Notes:

- With `--prefetch-frames 1`, the executor has one download worker. The first
  frame gates the task and the second frame can only overlap with the first
  frame's short compute/shard work.
- With `--prefetch-frames 2`, both frame downloads can run concurrently at task
  start. This reduced observed service wall from 21.5 sec to 9.25 sec in this
  small smoke.
- The reported per-frame `staging_wall_sec` is measured inside each prefetch
  thread. In concurrent mode, those durations overlap and should not be summed
  as wall-clock task time.

## 2026-06-29: Payload Wait Timing Smoke

Change:

- Worker frame timings now include `payload_prefetched` and
  `payload_wait_wall_sec`.
- `payload_wait_wall_sec` measures how long the main frame loop blocked waiting
  for `_read_frame_payload()` to finish.
- This is the practical prefetch tuning metric. High wait means the GPU/table
  path is starved on IO; near-zero wait means the payload was ready when needed.

Local cached two-frame smoke:

```text
run_id: service_queue_smoke_prefetch_metrics_local
frames_completed: 2
measurement_rows: 573
task_wall_sec: 0.410

frame fg_00000000:
  payload_prefetched: true
  payload_wait_wall_sec: 0.020
  staged_bytes: 0
  staging_wall_sec: 0.005
  frame_compute_wall_sec: 0.197

frame fg_00000001:
  payload_prefetched: true
  payload_wait_wall_sec: 0.000006
  staged_bytes: 0
  staging_wall_sec: 0.008
  frame_compute_wall_sec: 0.030
```

Cached S3 two-frame smoke:

```text
run_id: service_queue_smoke_prefetch_metrics_s3_cached
frames_completed: 2
measurement_rows: 573
task_wall_sec: 0.709
assembly target_count: 310

frame fg_00000000:
  payload_prefetched: true
  payload_wait_wall_sec: 0.317
  staged_bytes: 0
  staging_wall_sec: 0.304
  frame_compute_wall_sec: 0.201

frame fg_00000001:
  payload_prefetched: true
  payload_wait_wall_sec: 0.000006
  staged_bytes: 0
  staging_wall_sec: 0.291
  frame_compute_wall_sec: 0.074
```

Notes:

- The first payload in a task usually has nonzero wait because the frame loop
  cannot do useful photometry until at least one payload exists.
- Later payload waits should approach zero when prefetch width is sufficient.
- For S3-backed work, dashboard/status cards should report payload wait
  separately from staging duration because staging may overlap across prefetch
  threads.

## 2026-06-29: Task Queue Collection Timing Aggregates

Change:

- `collect-task-queue-run` now writes frame-level timing rows to
  `task_queue_frames.parquet`.
- `aggregate_summary.json` and `task_queue_tasks.parquet` include aggregate
  payload wait, staging, FITS read, kernel, and frame-compute timing fields.

Recollected cached S3 metric run:

```text
run_id: service_queue_smoke_prefetch_metrics_s3_cached
completed_frames: 2
measurement_rows: 573
frame_timing_rows: 2
payload_wait_wall_sec: 0.317
payload_wait_mean_wall_sec: 0.158
staging_wall_sec: 0.596
staged_bytes: 0
fits_read_wall_sec: 0.028
kernel_wall_sec: 0.061
frame_compute_wall_sec: 0.276
frame_table_path: runs/service_queue_smoke_prefetch_metrics_s3_cached/task_queue_frames.parquet
```

Recollected local metric run:

```text
run_id: service_queue_smoke_prefetch_metrics_local
completed_frames: 2
measurement_rows: 573
frame_timing_rows: 2
payload_wait_wall_sec: 0.020
payload_wait_mean_wall_sec: 0.010
staging_wall_sec: 0.013
staged_bytes: 0
fits_read_wall_sec: 0.030
kernel_wall_sec: 0.061
frame_compute_wall_sec: 0.227
frame_table_path: runs/service_queue_smoke_prefetch_metrics_local/task_queue_frames.parquet
```

Notes:

- These fields let the simple status/dashboard layer show IO starvation without
  parsing every nested worker `run_summary.json`.
- The frame table is the right source for per-frame plots and distribution
  views.

## 2026-06-29: Target-Partitioned Spectra Assembly Smoke

Change:

- Added `luxquarry-allsky assemble-spectra-partitions`.
- The command assigns each row to a stable cuDF hash bucket over
  `(catalog, target_id)` and writes one spectra parquet plus one target-summary
  parquet per bucket.
- Passing `--partition-index N` assembles only one reducer bucket, which is the
  horizontal postprocess shape for large campaigns.

All-partition smoke:

```text
input manifest: runs/service_queue_smoke_v3/measurement_shard_manifest.parquet
partition_count: 4
shard_count: 2
input_measurement_rows: 573
spectra_measurement_rows: 573
target_count: 310
read_shards_wall_sec: 0.139
bucket_wall_sec: 0.033
total_wall_sec: 7.291
```

Partition row counts:

```text
partition 0: 140 measurements, 78 targets
partition 1: 130 measurements, 69 targets
partition 2: 147 measurements, 79 targets
partition 3: 156 measurements, 84 targets
```

Single-partition smoke:

```text
partition_index: 2
spectra_measurement_rows: 147
target_count: 79
read_shards_wall_sec: 0.138
bucket_wall_sec: 0.029
total_wall_sec: 1.242
```

Correctness check:

```text
all-partitions part00002 spectra hash:
  a96b096e958e51422890020548e8c6bb61b51153b21f648c061c09f1e36fd1f6
single-partition part00002 spectra hash:
  a96b096e958e51422890020548e8c6bb61b51153b21f648c061c09f1e36fd1f6

all-partitions part00002 target-summary hash:
  b1d0fdc61f7cb3f61564745f84a36b347b50f3db4260094ee87e5dabfa7c52db
single-partition part00002 target-summary hash:
  b1d0fdc61f7cb3f61564745f84a36b347b50f3db4260094ee87e5dabfa7c52db
```

Empty-partition check:

```text
partition_count: 1024
partition_index: 0
spectra_measurement_rows: 0
target_count: 0
status: completed without error
```

Notes:

- The first all-partition target-summary groupby paid a cuDF warmup cost
  (`~6.1 sec`), while later partitions were `~0.012 sec`. Reducer jobs should be
  long-lived or large enough to amortize this.
- This is not the final all-sky reducer yet. It proves the deterministic
  target-bucket contract. The next optimization is avoiding repeated full-shard
  reads by using Dask-cuDF or writing pre-shuffled reducer inputs.

## 2026-06-29: Pre-Shuffled Measurement Reducer Inputs

Change:

- Added `luxquarry-allsky partition-measurement-shards`.
- The command reads measurement shards once with cuDF, hashes rows by
  `(catalog, target_id)`, and writes one measurement parquet plus one one-row
  shard manifest per non-empty target bucket.
- Reducers can now run `assemble-spectra` against a single partition manifest
  instead of all original measurement shards.
- `measurement_partition_summary.json` keeps stdout/status bounded with
  `partition_preview`; the full partition table is
  `<run_id>.measurement_partition_manifest.parquet`.

Four-partition smoke:

```text
input manifest: runs/service_queue_smoke_v3/measurement_shard_manifest.parquet
partition_count: 4
shard_count: 2
input_measurement_rows: 573
partitioned_measurement_rows: 573
target_count_sum_by_partition: 310
written_partitions: 4
read_shards_wall_sec: 0.139
bucket_wall_sec: 0.029
write_partitions_wall_sec: 0.063
total_wall_sec: 1.134
```

Partition row counts:

```text
partition 0: 140 measurements, 78 targets
partition 1: 130 measurements, 69 targets
partition 2: 147 measurements, 79 targets
partition 3: 156 measurements, 84 targets
```

Reducer smoke over only partition 2:

```text
input manifest:
  runs/service_queue_smoke_v3/measurement_partition_smoke/partition_manifests/service_queue_smoke_v3_measurement_partitions.part00002.measurement_shard_manifest.parquet
input_measurement_rows: 147
spectra_measurement_rows: 147
target_count: 79
shard_count: 1
total_wall_sec: 1.210
```

Correctness check:

```text
direct partitioned assembly part00002 spectra hash:
  a96b096e958e51422890020548e8c6bb61b51153b21f648c061c09f1e36fd1f6
pre-shuffled reducer part00002 spectra hash:
  a96b096e958e51422890020548e8c6bb61b51153b21f648c061c09f1e36fd1f6

direct partitioned assembly part00002 target-summary hash:
  b1d0fdc61f7cb3f61564745f84a36b347b50f3db4260094ee87e5dabfa7c52db
pre-shuffled reducer part00002 target-summary hash:
  b1d0fdc61f7cb3f61564745f84a36b347b50f3db4260094ee87e5dabfa7c52db
```

Sparse-bucket smoke:

```text
partition_count: 1024
written_partitions: 272
partitioned_measurement_rows: 573
target_count_sum_by_partition: 310
summary_partition_limit: 5
partition_preview_truncated: true
total_wall_sec: 5.509
```

Notes:

- Empty partitions are skipped by default to avoid creating thousands of tiny
  files. `--write-empty-partitions` is available for launchers that require a
  file for every bucket.
- This creates the desired horizontal reducer shape:
  photometry shards -> one GPU shuffle -> many independent spectra reducers.

## 2026-06-29: Reducer Fanout Plan and Kubernetes Jobs

Change:

- Added `luxquarry-allsky write-reducer-plan`.
- Added `luxquarry-allsky write-k8s-reducer-jobs`.
- Reducer plans consume the measurement partition manifest and emit one
  `assemble-spectra` task per non-empty target bucket.
- Kubernetes reducer jobs default to `--device cuda:0` inside one-GPU pods,
  even when the local reducer plan was built with multiple local device names.

Reducer plan smoke:

```text
input partition manifest:
  runs/service_queue_smoke_v3/measurement_partition_smoke/service_queue_smoke_v3_measurement_partitions.measurement_partition_manifest.parquet
max_partitions: 3
devices: cuda:0,cuda:1
reducer_count: 3
total_measurement_rows: 417

part00000: cuda:0, 140 measurements
part00001: cuda:1, 130 measurements
part00002: cuda:0, 147 measurements
```

Generated files:

```text
runs/service_queue_smoke_v3/reducer_plan_smoke/reducer_plan.json
runs/service_queue_smoke_v3/reducer_plan_smoke/reducer_plan.sh
```

Kubernetes reducer job smoke:

```text
image: luxquarry-allsky:test
namespace: luxquarry
job_count: 3
gpu request per job: 1
container command: luxquarry-allsky
container args: assemble-spectra ...
device override inside pod: cuda:0
```

Generated file:

```text
runs/service_queue_smoke_v3/reducer_plan_smoke/k8s_device_override/service_queue_smoke_v3_reducers.reducer-jobs.yaml
```

## 2026-06-29: Local Reducer Lifecycle and Collection

Change:

- Added `luxquarry-allsky run-reducer-plan`.
- Added `luxquarry-allsky collect-reducer-plan`.
- Local reducer runs support `--resume` and `--max-parallel`.
- Reducer collection writes `reducer_outputs.parquet`, which is the handoff for
  scorer fanout and viewer/ClickHouse indexing.

Lifecycle smoke:

```text
input partition manifest:
  runs/service_queue_smoke_v3/measurement_partition_smoke/service_queue_smoke_v3_measurement_partitions.measurement_partition_manifest.parquet
max_partitions: 2
devices: cuda:0
max_parallel: 1
reducer_count: 2
total planned measurement rows: 270
```

Run result:

```text
status: complete
launched_reducer_count: 2
failed_reducer_count: 0
total_wall_sec: 5.208

part00000 wall_sec: 2.604
part00001 wall_sec: 2.503
```

Collection result:

```text
complete: true
complete_reducer_count: 2
failed_reducer_count: 0
input_measurement_rows: 270
spectra_measurement_rows: 270
target_count: 147
collect_wall_sec: 0.009
reducer_manifest_path:
  runs/service_queue_smoke_v3/reducer_lifecycle_smoke/reducer_outputs.parquet
```

Collected reducer rows:

```text
part00000: 140 spectra measurements, 78 targets
part00001: 130 spectra measurements, 69 targets
```

Resume check:

```text
run-reducer-plan --resume
launched_reducer_count: 0
skipped_reducer_count: 2
failed_reducer_count: 0
status: complete
```

## 2026-06-29: Candidate Scorer Fanout Lifecycle

Change:

- Added `luxquarry-allsky write-candidate-fanout-plan`.
- Added `luxquarry-allsky run-candidate-fanout-plan`.
- Added `luxquarry-allsky collect-candidate-fanout-plan`.
- Candidate scorer fanout consumes `reducer_outputs.parquet` and emits one
  scorer task per reducer spectra partition.
- Candidate collection writes `candidate_scorer_outputs.parquet`, which is the
  handoff for candidate aggregation, viewer indexing, and later ClickHouse
  loading.

Plan smoke:

```text
input reducer outputs:
  runs/service_queue_smoke_v3/reducer_lifecycle_smoke/reducer_outputs.parquet
devices: cuda:0
max_parallel: 1
min_abs_zscore: 1.0
min_measurements: 2
include_flagged: true
scorer_count: 2
total planned spectra measurement rows: 270
```

Run result:

```text
status: complete
launched_scorer_count: 2
failed_scorer_count: 0
total_wall_sec: 5.408

part00000 wall_sec: 2.704
part00001 wall_sec: 2.604
```

Collection result:

```text
complete: true
complete_scorer_count: 2
failed_scorer_count: 0
input_measurement_rows: 270
filtered_measurement_rows: 270
target_count_by_partition: 147
candidate_count: 0
candidate_target_count_by_partition: 0
collect_wall_sec: 0.009
scorer_sum_wall_sec: 2.365
scorer_manifest_path:
  runs/service_queue_smoke_v3/candidate_fanout_smoke/candidate_scorer_outputs.parquet
```

Collected scorer rows:

```text
part00000: 140 input measurements, 0 candidates
part00001: 130 input measurements, 0 candidates
```

Resume check:

```text
run-candidate-fanout-plan --resume
launched_scorer_count: 0
skipped_scorer_count: 2
failed_scorer_count: 0
status: complete
```

Notes:

- Zero candidates is acceptable for this lifecycle smoke. The purpose here was
  fanout, resume, collection, and manifest handoff, not detection threshold
  tuning.
- The next cloud-scale step is to generate Kubernetes scorer Jobs from the same
  fanout plan, mirroring the reducer Job path.

## 2026-06-29: Candidate Scorer Kubernetes Jobs

Change:

- Added `luxquarry-allsky write-k8s-candidate-scorer-jobs`.
- The command consumes `candidate_fanout_plan.json` and emits one Kubernetes Job
  per candidate scorer partition.
- The generator rewrites `--device` to the pod-local device string, so a plan
  built with local multi-GPU names can still run as one-GPU pods using
  `--device cuda:0`.

Smoke command:

```bash
luxquarry-allsky write-k8s-candidate-scorer-jobs \
  --candidate-plan runs/service_queue_smoke_v3/candidate_fanout_smoke/candidate_fanout_plan.json \
  --out-dir runs/service_queue_smoke_v3/candidate_fanout_smoke/k8s \
  --image luxquarry-allsky:test \
  --namespace luxquarry \
  --container-executable luxquarry-allsky \
  --working-dir /workspace/luxquarry_allsky_engine \
  --pvc-name luxquarry-data \
  --mount-path /workspace \
  --device cuda:0
```

Result:

```text
job_count: 2
image: luxquarry-allsky:test
namespace: luxquarry
total_spectra_measurement_rows: 270
total_target_count_by_partition: 147
manifest:
  runs/service_queue_smoke_v3/candidate_fanout_smoke/k8s/service_queue_smoke_v3_candidate_fanout.candidate-scorer-jobs.yaml
```

Generated Jobs:

```text
service-queue-smoke-v3-candidate-fanout-part00000
service-queue-smoke-v3-candidate-fanout-part00001
```

Manifest inspection confirmed:

```text
container command: luxquarry-allsky
container args: score-spectra-candidates ...
device override inside pod: cuda:0
gpu request per job: 1
job-role label: candidate-scorer
pvc mount: /workspace
```

## 2026-06-29: Standalone Object Staging Benchmark

Change:

- Added `luxquarry-allsky benchmark-object-staging`.
- The command stages manifest `path` entries through the same
  `stage_input_file()` implementation used by persistent workers.
- It sweeps staging concurrency without running FITS reads, photometry kernels,
  spectra assembly, or parquet scoring.
- It writes aggregate JSON plus per-object parquet:

```text
object_staging_summary.json
object_staging_results.parquet
```

Local two-frame smoke:

```bash
luxquarry-allsky benchmark-object-staging \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --out-dir runs/object_staging_bench_local_smoke \
  --cache-dir /tmp/luxquarry_object_staging_local_smoke \
  --concurrency 1,2 \
  --limit 2 \
  --cache-mode per-concurrency
```

Result:

```text
input_rows: 2
bytes per sweep: 143,274,240

concurrency 1:
  wall_sec: 0.204
  transferred_mib_per_sec: 668.3
  cache_miss_count: 2

concurrency 2:
  wall_sec: 0.131
  transferred_mib_per_sec: 1044.8
  cache_miss_count: 2
```

S3 one-frame smoke:

```bash
luxquarry-allsky benchmark-object-staging \
  --manifest runs/manifest_smoke_v2_s3/frame_manifest.parquet \
  --out-dir runs/object_staging_bench_s3_smoke \
  --cache-dir /tmp/luxquarry_object_staging_s3_smoke \
  --concurrency 1 \
  --limit 1 \
  --cache-mode per-concurrency \
  --require-s3
```

Result:

```text
input_rows: 1
bytes_written: 71,637,120
wall_sec: 4.651
transferred_mib_per_sec: 14.7
cache_miss_count: 1
```

S3 two-frame concurrency smoke:

```bash
luxquarry-allsky benchmark-object-staging \
  --manifest runs/manifest_smoke_v2_s3/frame_manifest.parquet \
  --out-dir runs/object_staging_bench_s3_concurrency_smoke \
  --cache-dir /tmp/luxquarry_object_staging_s3_concurrency_smoke \
  --concurrency 1,2 \
  --limit 2 \
  --cache-mode per-concurrency \
  --require-s3
```

Result:

```text
input_rows: 2
bytes per sweep: 143,274,240

concurrency 1:
  wall_sec: 11.233
  transferred_mib_per_sec: 12.2
  p95_stage_wall_sec: 7.257

concurrency 2:
  wall_sec: 13.320
  transferred_mib_per_sec: 10.3
  p95_stage_wall_sec: 13.167
```

Notes:

- On this route, local staging is hundreds of MiB/sec and S3 staging is tens of
  MiB/sec, so S3/object-store staging is a credible bottleneck before GPU
  occupancy matters.
- Concurrency 2 was slower than concurrency 1 for this tiny S3 smoke. That does
  not prove concurrency is bad globally, but it proves we need to sweep staging
  independently on each target environment instead of assuming more downloads
  always help.
- For EKS/V100-class runs, this benchmark should run per instance family and
  placement before choosing worker prefetch width and task batch size.

## 2026-06-29: Persistent GPU PSF Grid Smoke

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky run-persistent-gpu-worker \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/persistent_gpu_worker_psf_smoke10 \
  --run-id persistent_psf_smoke10 \
  --device cuda:0 \
  --limit-frames 10 \
  --batch-table-assembly \
  --shard-batch-frames 5 \
  --enable-psf \
  --psf-kernel-build-mode gpu_spline
```

Result:

```text
frame_count: 10
measurement_rows: 2,770
ok_measurement_rows: 2,766
psf_measurement_rows: 2,770
ok_psf_rows: 2,770
failed_frames: 0
total_wall_sec: 2.325
measurements_per_sec: 1,191
```

PSF timing summary:

```text
psf_grid_offsets: 25
psf_candidate_count_total: ~69,250
psf_device_submit_sync_wall_sec_sum: 0.0787
psf_candidates_per_gpu_submit_sync_sec: ~880,000
psf_kernel_wall_sec_sum: 0.6348
psf_spline_coeff_wall_sec_sum: 0.1501
psf_upload_wall_sec_sum: 0.1658
fits_read_wall_sec_sum: 0.9533
```

Notes:

- The allsky worker now supports GPU local-grid PSF photometry via
  `--enable-psf`.
- Aperture geometry is explicit at both run level and row level:
  `aperture_radius_pix`, `annulus_inner_pix`, `annulus_outer_pix`, and
  `edge_margin_pix` are written to `run_summary.json` and every measurement
  row. `plan-gpu-dispatch` and `run-local-dispatch` forward these settings to
  generated workers.
- The PSF path is candidate-parallel: each target expands into a local grid of
  shifted PSF candidates, each candidate is fit independently on GPU, then a GPU
  reduction picks the best candidate per target.
- The default production/correctness mode is `--psf-kernel-build-mode
  gpu_spline`, which CPU-builds cubic spline coefficients from the FITS PSF cube
  and samples shifted PSFs on GPU.
- This smoke is intentionally tiny, with only ~250-300 selected targets per
  frame. It underfeeds high-end GPUs. The device math is not the dominant cost;
  FITS read, upload, spline setup, and shard work dominate the wall clock.
- For A100/H100/B300-class nodes, the next benchmark must use large per-frame
  source counts and/or multiple queued frame tasks per GPU so occupancy reflects
  the intended survey workload.

## 2026-06-29: PSF Dispatch Benchmark Harness Smoke

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky run-dispatch-benchmark-sweep \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/dispatch_benchmark_psf_worker_only_smoke10_writefix \
  --run-id dispatch_psf_worker_only_smoke10_writefix \
  --devices cuda:0 \
  --workers-per-device 1 \
  --limit-frames 10 \
  --shard-batch-frames 5 \
  --prefetch-frames 0 \
  --worker-only \
  --enable-psf \
  --psf-kernel-build-mode gpu_spline \
  --status-snapshot-interval-sec 0
```

Result:

```text
completed_frames: 10
measurement_rows: 2,770
psf_measurement_rows: 2,770
ok_psf_rows: 2,770
psf_candidate_count: 69,250
psf_candidates_per_sec_device_submit_sync: 879,484
measurements_per_sec_worker_payload: 1,251
worker_max_wall_sec: 2.214
total_wall_sec: 5.086
```

Profile rows above 5% total wall:

```text
worker_phase: 98.4%
fits_read: 17.5%
psf_kernel_total: 12.4%
write_measurement_shards: 6.0%
aperture_kernel: 5.5%
```

Notes:

- The dispatch benchmark sweep now accepts `--enable-psf` and PSF grid options.
- `collect-dispatch-run` aggregates PSF rows, candidate counts, and PSF timing
  fields from worker `frame_timings`.
- `sweep_results.parquet` and `perf_summary.json` now include PSF occupancy
  metrics, including `psf_candidates_per_sec_device_submit_sync`.
- `profile_summary.parquet` emits separate rows for FITS read, aperture kernel,
  PSF total, PSF device submit/sync, PSF spline setup, PSF upload, PSF gather,
  shard writes, and collect.

## 2026-06-29: Shared GPU Frame Buffer Smoke

Change:

```text
Before:
  aperture uploaded IMAGE/VARIANCE/FLAGS
  PSF uploaded IMAGE/VARIANCE/FLAGS again

After:
  worker uploads IMAGE/VARIANCE/FLAGS once as a resident frame object
  aperture and PSF both consume that frame object
```

Benchmark command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky run-dispatch-benchmark-sweep \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/dispatch_benchmark_psf_worker_only_shared_frame_smoke10 \
  --run-id dispatch_psf_worker_only_shared_frame_smoke10 \
  --devices cuda:0 \
  --workers-per-device 1 \
  --limit-frames 10 \
  --shard-batch-frames 5 \
  --prefetch-frames 0 \
  --worker-only \
  --enable-psf \
  --psf-kernel-build-mode gpu_spline \
  --status-snapshot-interval-sec 0
```

Before/after against the previous PSF benchmark smoke:

```text
worker_max_wall_sec: 2.214 -> 2.037  (-8.0%)
measurements_per_sec_worker_payload: 1,251 -> 1,360  (+8.7%)
psf_kernel_wall_sec: 0.632 -> 0.471  (-25.6%)
psf_upload_wall_sec: 0.169 -> 0.009  (-94.6%)
aperture_kernel_wall_sec: 0.281 -> 0.029  (-89.9%)
frame_upload_wall_sec: 0.248  (new shared upload metric)
psf_candidates_per_sec_device_submit_sync: ~879k -> ~882k
```

Notes:

- `frame_upload_wall_sec` now records the one shared frame upload.
- `aperture_kernel_wall_sec` now reflects mostly aperture device work and target
  array setup, not raw frame upload.
- `psf_upload_wall_sec` now reflects PSF-specific uploads only.
- The device math throughput is effectively unchanged; the win is less duplicate
  transfer/setup work around the kernels.

## 2026-06-29/30: Dense Galactic-Core PSF Throughput and Output Bounds

Workload:

```text
manifest: runs/manifest_galactic_core_nearest20/frame_manifest.parquet
projected targets: runs/projected_targets_galactic_core_nearest20_allcat_g11_16_n20000/frame_targets_projected.parquet
frames: 18
devices: cuda:0,cuda:1,cuda:2
workers/device: 1
prefetch: 2
shard_batch_frames: 6
photometry: aperture + GPU local-grid PSF
PSF kernel build: gpu_spline
```

Warm local-cache durable-output run:

```text
measurement_rows: 320,768
worker_max_wall_sec: 2.233
measurements_per_sec_worker_payload: 143,661
total_wall_sec: 6.700
write_wall_sec summed across workers: 1.442
max_worker_shard_write_wall_sec: 0.485
max_worker_async_shard_write_wait_wall_sec: 0.478
shard_total_bytes: 33,305,757
shard_bytes_per_measurement: 103.83
```

No-write bound on the same warm workload:

```text
measurement_rows: 320,768
worker_max_wall_sec: 1.659
measurements_per_sec_worker_payload: 193,371
total_wall_sec: 5.931
write_wall_sec: 0.000
```

Interpretation:

- The GPU math path is no longer the only visible bottleneck. With dense real
  fields, PSF candidate math reaches about 6.7M candidates/sec in the device
  submit/sync section.
- Durable output costs about 0.5 sec on the critical worker path for this
  18-frame test. That is large enough to matter for cloud GPU economics.
- `--discard-measurement-shards` is a benchmark-only upper bound. It must not be
  used for science output because spectra assembly and candidate review require
  durable per-measurement rows.

Compact output profile:

```text
--measurement-column-profile compact
```

The first compact 18-frame run was not materially faster:

```text
full profile payload:    2.233 sec, 143,661 measurements/sec, 103.83 bytes/row
compact profile payload: 2.240 sec, 143,225 measurements/sec, 103.70 bytes/row
```

So compact output is not a write-throughput fix by itself on this workload. It
is still useful as a schema-control mechanism.

A follow-up compact smoke was patched and verified to preserve per-point audit
provenance:

```text
rows: 35,717
columns: 53
kept: frame_group_id, image_id, fits_path, detector, release
kept: aperture_radius_pix, annulus_inner_pix, annulus_outer_pix, edge_margin_pix
kept: wavelength_source, wavelength_calibration_file, sapm_file
kept: cwave_um, cband_um, flags_summary
kept: aperture_flux_uJy, aperture_flux_unc_uJy
kept: psf_flux_uJy, psf_flux_unc_uJy, PSF fit diagnostics
dropped: local_fits_path
```

`local_fits_path` is intentionally omitted from compact output because it is an
ephemeral local-cache/staging path. The durable provenance key is `fits_path`,
plus `frame_group_id` and `image_id`.

Next performance move:

```text
keep workers alive
feed them frame-batch tasks
write wider/fewer shards
evaluate NVMe/KvikIO or a dedicated writer path
```

Process launch and output draining still consume enough wall time that small
runs are not representative of steady-state cloud throughput.

## 2026-06-29/30: Task-Queue Worker Service Smoke

Purpose:

```text
Verify the persistent worker-service path, not just one-shot dispatch.
```

Queue command:

```bash
luxquarry_allsky_engine/.venv/bin/luxquarry-allsky write-task-queue \
  --manifest luxquarry_allsky_engine/runs/manifest_galactic_core_nearest20/frame_manifest.parquet \
  --projected-targets luxquarry_allsky_engine/runs/projected_targets_galactic_core_nearest20_allcat_g11_16_n20000/frame_targets_projected.parquet \
  --out-dir luxquarry_allsky_engine/runs/task_queue_gc_dense_smoke_verify \
  --campaign-id task_queue_gc_dense_smoke_verify \
  --frames-per-task 2 \
  --limit-frames 4
```

Queue result:

```text
frame_count: 4
task_count: 2
projected_target_rows: 400,000 source rows
task projected rows: 40,000 per task
```

Worker command:

```bash
luxquarry_allsky_engine/.venv/bin/luxquarry-allsky run-gpu-worker-service \
  --queue-dir luxquarry_allsky_engine/runs/task_queue_gc_dense_smoke_verify \
  --out-dir luxquarry_allsky_engine/runs/task_queue_gc_dense_smoke_verify/worker_cuda0 \
  --run-id task_queue_gc_dense_smoke_verify \
  --worker-id worker-cuda0 \
  --device cuda:0 \
  --shard-batch-frames 2 \
  --prefetch-frames 2 \
  --local-cache-dir /tmp/luxquarry_gc_dense_cache_prefetch2 \
  --measurement-column-profile compact \
  --enable-psf \
  --psf-kernel-build-mode gpu_spline
```

Worker-service result:

```text
tasks_completed: 2
tasks_failed: 0
frames_completed: 4
measurement_rows: 71,319
ok_measurement_rows: 69,613
total_wall_sec: 2.99
column_profile: compact
column_count: 53
```

Collection and downstream checks:

```text
collect-task-queue-run:
  complete: true
  shard_count: 2
  missing_shards: 0
  measurement_rows: 71,319

assemble-spectra:
  input_measurement_rows: 71,319
  spectra_measurement_rows: 71,319
  target_count: 18,914
  total_wall_sec: 1.41

score-spectra-candidates:
  input_measurement_rows: 71,319
  filtered_measurement_rows: 69,613
  target_count: 18,770
  candidate_count: 0
  total_wall_sec: 1.27
```

This proves the current queue shape can claim tasks, run persistent GPU
photometry, collect out-of-order worker shards, assemble spectra, and score
candidates from compact output. It is still a local smoke, not a production EKS
run, but it validates the data-product contract we need for horizontal scaling.

## 2026-06-29/30: Resident Source-Input Queue Smoke

Purpose:

```text
Avoid per-task parquet input materialization and expose calibration cache reuse.
```

Queue command:

```bash
luxquarry_allsky_engine/.venv/bin/luxquarry-allsky write-task-queue \
  --manifest luxquarry_allsky_engine/runs/manifest_galactic_core_nearest20/frame_manifest.parquet \
  --projected-targets luxquarry_allsky_engine/runs/projected_targets_galactic_core_nearest20_allcat_g11_16_n20000/frame_targets_projected.parquet \
  --out-dir luxquarry_allsky_engine/runs/task_queue_gc_dense_resident_verify \
  --campaign-id task_queue_gc_dense_resident_verify \
  --frames-per-task 2 \
  --limit-frames 8 \
  --no-materialize-task-inputs
```

Worker command:

```bash
luxquarry_allsky_engine/.venv/bin/luxquarry-allsky run-gpu-worker-service \
  --queue-dir luxquarry_allsky_engine/runs/task_queue_gc_dense_resident_verify \
  --out-dir luxquarry_allsky_engine/runs/task_queue_gc_dense_resident_verify/worker_cuda0 \
  --run-id task_queue_gc_dense_resident_verify \
  --worker-id worker-cuda0 \
  --device cuda:0 \
  --shard-batch-frames 2 \
  --prefetch-frames 2 \
  --local-cache-dir /tmp/luxquarry_gc_dense_cache_prefetch2 \
  --measurement-column-profile compact \
  --enable-psf \
  --psf-kernel-build-mode gpu_spline
```

Result:

```text
materialized_task_inputs: false
resident_source_inputs: true
source_input_load_wall_sec: 0.157
task_count: 4
frames_completed: 8
tasks_completed: 4
tasks_failed: 0
measurement_rows: 142,906
ok_measurement_rows: 140,291
worker service total_wall_sec: 4.19
```

Collected queue summary:

```text
complete: true
shard_count: 4
missing_shards: 0
calibration_cache_hits: 2
calibration_cache_misses: 6
calibration_load_wall_sec: 0.571
max_worker_resident_calibration_count: 6
```

The first two tasks used four unique release/detector calibrations and missed
the cache. Later tasks reused detector 3 and detector 6 and produced cache hits.
That verifies that one worker service keeps calibration maps resident across
claimed tasks.

Downstream spectra assembly:

```text
input_measurement_rows: 142,906
spectra_measurement_rows: 142,906
target_count: 23,915
shard_count: 4
total_wall_sec: 1.46
```

This is the preferred local shape for the next benchmark phase: queue once,
load source tables once per worker service, keep calibration resident, process
claimed frame batches, collect shards, then reduce spectra.

## 2026-06-29/30: 3-GPU Resident Queue Scale Smoke

Purpose:

```text
Verify that multiple persistent GPU services can drain one queue and produce
collectable out-of-order shards.
```

Workload:

```text
frames: 18
targets: dense Galactic-core projected target table
photometry: aperture + GPU local-grid PSF
source inputs: resident per worker service
workers: cuda:0, cuda:1, cuda:2
measurement profile: compact
async shard writes: enabled
```

Variant A used six tasks of three frames each:

```text
queue_dir: runs/task_queue_gc_dense_resident_3gpu18_verify
frames_per_task: 3
task_count: 6
completed_frames: 18
complete_tasks: 6
failed_tasks: 0
measurement_rows: 320,768
ok_measurement_rows: 315,754
shard_count: 6
missing_shards: 0
worker_payload_max_wall_sec: 2.407
measurements_per_sec_worker_payload: 133,268
worker_parallel_efficiency: 0.968
calibration_cache_hits: 4
calibration_cache_misses: 14
calibration_load_wall_sec: 1.452
spectra_measurement_rows: 320,768
target_count: 29,500
spectra_total_wall_sec: 1.536
```

Variant B used three tasks of six frames each:

```text
queue_dir: runs/task_queue_gc_dense_resident_3gpu18_fpt6_verify
frames_per_task: 6
task_count: 3
completed_frames: 18
complete_tasks: 3
failed_tasks: 0
measurement_rows: 320,768
ok_measurement_rows: 315,754
shard_count: 3
missing_shards: 0
worker_payload_max_wall_sec: 2.421
measurements_per_sec_worker_payload: 132,509
worker_parallel_efficiency: 0.990
calibration_cache_hits: 2
calibration_cache_misses: 16
calibration_load_wall_sec: 1.973
spectra_measurement_rows: 320,768
target_count: 29,500
spectra_total_wall_sec: 1.559
```

Comparison against one-shot dispatch on the same 18-frame dense workload:

```text
dispatch compact warm:
  worker_payload_max_wall_sec: 2.240
  measurements_per_sec_worker_payload: 143,225
  worker_parallel_efficiency: 0.955

resident queue, 6 x 3-frame tasks:
  worker_payload_max_wall_sec: 2.407
  measurements_per_sec_worker_payload: 133,268
  worker_parallel_efficiency: 0.968

resident queue, 3 x 6-frame tasks:
  worker_payload_max_wall_sec: 2.421
  measurements_per_sec_worker_payload: 132,509
  worker_parallel_efficiency: 0.990

dispatch no-write warm bound:
  worker_payload_max_wall_sec: 1.659
  measurements_per_sec_worker_payload: 193,371
```

Interpretation:

- The queue system now proves horizontal local execution: multiple worker
  services claimed tasks concurrently and the collector assembled out-of-order
  shards without missing outputs.
- The queue path is currently about 7-8% slower than one-shot compact dispatch
  on this small 18-frame test.
- Larger six-frame tasks reduced shard count and improved worker balance, but
  did not improve throughput. Startup/finalization/source-table service costs
  and calibration loading are still visible at this small scale.
- The no-write bound remains far ahead. Output/table/write architecture is
  still the dominant next optimization target.
- For production-scale cloud runs, this queue shape is still directionally
  right because it decouples task claiming from workers and supports many nodes;
  it simply needs larger steady-state queues where setup costs are amortized.

## 2026-06-29/30: First-Class Local Queue Runner Smoke

The ad hoc Python subprocess launcher was promoted to a CLI command:

```text
run-local-task-queue
```

Smoke command:

```bash
luxquarry_allsky_engine/.venv/bin/luxquarry-allsky run-local-task-queue \
  --manifest luxquarry_allsky_engine/runs/manifest_galactic_core_nearest20/frame_manifest.parquet \
  --projected-targets luxquarry_allsky_engine/runs/projected_targets_galactic_core_nearest20_allcat_g11_16_n20000/frame_targets_projected.parquet \
  --out-dir luxquarry_allsky_engine/runs/local_task_queue_runner_smoke_2gpu4 \
  --run-id local_task_queue_runner_smoke_2gpu4 \
  --devices cuda:0,cuda:1 \
  --frames-per-task 2 \
  --limit-frames 4 \
  --shard-batch-frames 2 \
  --prefetch-frames 2 \
  --local-cache-dir /tmp/luxquarry_gc_dense_cache_prefetch2 \
  --measurement-column-profile compact \
  --async-shard-writes \
  --enable-psf \
  --psf-kernel-build-mode gpu_spline \
  --assemble-spectra \
  --score-baseline
```

Result:

```text
status: complete
worker_count: 2
failed_workers: 0
frames: 4
measurement_rows: 71,319
ok_measurement_rows: 69,613
shard_count: 2
missing_shards: 0
worker_payload_max_wall_sec: 1.027
measurements_per_sec_worker_payload: 69,456
worker_parallel_efficiency: 0.994
spectra_measurement_rows: 71,319
target_count: 18,914
baseline_candidate_count: 0
```

This is now the preferred local test harness for the next-gen allsky path. It
uses separate worker-service subprocesses per GPU, which keeps the local runner
aligned with the eventual Kubernetes/EKS execution model.

## 2026-06-29/30: Batch Table Assembly Queue Benchmark

Purpose:

```text
Move cuDF table construction out of the per-frame critical section and do it at
shard flush.
```

Command shape:

```bash
luxquarry_allsky_engine/.venv/bin/luxquarry-allsky run-local-task-queue \
  --manifest luxquarry_allsky_engine/runs/manifest_galactic_core_nearest20/frame_manifest.parquet \
  --projected-targets luxquarry_allsky_engine/runs/projected_targets_galactic_core_nearest20_allcat_g11_16_n20000/frame_targets_projected.parquet \
  --out-dir luxquarry_allsky_engine/runs/local_task_queue_gc_dense_3gpu18_batch_table_verify \
  --run-id local_task_queue_gc_dense_3gpu18_batch_table_verify \
  --devices cuda:0,cuda:1,cuda:2 \
  --frames-per-task 6 \
  --limit-frames 18 \
  --shard-batch-frames 6 \
  --prefetch-frames 2 \
  --local-cache-dir /tmp/luxquarry_gc_dense_cache_prefetch2 \
  --measurement-column-profile compact \
  --async-shard-writes \
  --batch-table-assembly \
  --enable-psf \
  --psf-kernel-build-mode gpu_spline \
  --assemble-spectra
```

Result:

```text
completed_frames: 18
measurement_rows: 320,768
ok_measurement_rows: 315,754
shard_count: 3
missing_shards: 0
worker_payload_max_wall_sec: 2.288
measurements_per_sec_worker_payload: 140,213
worker_parallel_efficiency: 0.954
spectra_measurement_rows: 320,768
spectra_total_wall_sec: 1.368
```

Comparison:

```text
queue resident, no batch table assembly:
  worker_payload_max_wall_sec: 2.421
  measurements_per_sec_worker_payload: 132,509
  summed frame table_wall_sec: 1.297
  summed frame_compute_wall_sec: 6.319

queue resident, batch table assembly:
  worker_payload_max_wall_sec: 2.288
  measurements_per_sec_worker_payload: 140,213
  summed frame table_wall_sec: 0.076
  summed frame_compute_wall_sec: 4.666

one-shot compact dispatch:
  worker_payload_max_wall_sec: 2.240
  measurements_per_sec_worker_payload: 143,225

no-write bound:
  worker_payload_max_wall_sec: 1.659
  measurements_per_sec_worker_payload: 193,371
```

Interpretation:

- Batch table assembly recovers most of the queue-mode throughput gap.
- It moves table construction into shard flush, so shard `write_wall_sec`
  includes both table assembly and parquet write. That is acceptable because the
  high-level target is critical-path payload time, not making one metric look
  smaller.
- `run-local-task-queue` now defaults to batch table assembly. The lower-level
  `run-gpu-worker-service` keeps its explicit flag so debugging remains
  straightforward.
- The durable-output gap remains: even after batch table assembly, the no-write
  bound is still much faster.

## 2026-06-29: Measurement Parquet Compression Check

After batch table assembly moved table construction out of the per-frame hot
path, the next question was whether parquet compression itself was a major
durable-output cost. The same dense 18-frame, 3-GPU resident queue workload was
run with compact measurement columns and no parquet compression.

Command:

```bash
luxquarry_allsky_engine/.venv/bin/luxquarry-allsky run-local-task-queue \
  --manifest luxquarry_allsky_engine/runs/manifest_galactic_core_nearest20/frame_manifest.parquet \
  --projected-targets luxquarry_allsky_engine/runs/projected_targets_galactic_core_nearest20_allcat_g11_16_n20000/frame_targets_projected.parquet \
  --out-dir luxquarry_allsky_engine/runs/local_task_queue_gc_dense_3gpu18_batch_table_nocompress_verify \
  --run-id local_task_queue_gc_dense_3gpu18_batch_table_nocompress_verify \
  --devices cuda:0,cuda:1,cuda:2 \
  --frames-per-task 6 \
  --limit-frames 18 \
  --shard-batch-frames 6 \
  --prefetch-frames 2 \
  --local-cache-dir /tmp/luxquarry_gc_dense_cache_prefetch2 \
  --measurement-column-profile compact \
  --measurement-parquet-compression none \
  --async-shard-writes \
  --enable-psf \
  --psf-kernel-build-mode gpu_spline \
  --assemble-spectra
```

Result:

```text
completed_frames: 18
measurement_rows: 320,768
ok_measurement_rows: 315,754
shard_count: 3
missing_shards: 0
worker_payload_max_wall_sec: 2.258
measurements_per_sec_worker_payload: 142,079
worker_parallel_efficiency: 0.946
spectra_measurement_rows: 320,768
spectra_target_count: 29,500
spectra_total_wall_sec: 1.349
shard_total_bytes: 34,997,955
shard_bytes_per_measurement: 109.1
summed_shard_write_wall_sec: 1.220
```

Comparison:

```text
compact + snappy:
  worker_payload_max_wall_sec: 2.288
  measurements_per_sec_worker_payload: 140,213
  summed_shard_write_wall_sec: 1.449

compact + no compression:
  worker_payload_max_wall_sec: 2.258
  measurements_per_sec_worker_payload: 142,079
  summed_shard_write_wall_sec: 1.220

no-write bound:
  worker_payload_max_wall_sec: 1.659
  measurements_per_sec_worker_payload: 193,371
```

Interpretation:

- Disabling parquet compression helped only modestly on this workload: about
  1.3% better payload throughput than compact snappy output.
- The durable-output gap remains large. The remaining cost is not just snappy;
  it is the combined table-to-parquet write path and filesystem durability.
- Keep `snappy` as the default for normal runs because it preserves portable,
  compressed durable artifacts. Use `--measurement-parquet-compression none`
  for targeted write-path benchmarks or if downstream storage bandwidth is
  known to be cheaper than compression CPU time.

## 2026-06-30: Local Task-Queue Bottleneck Report

One-off `run-local-task-queue` runs now write a stage-level performance report
alongside the run summary:

```text
task_queue_perf_report.json
task_queue_perf_profile.json
task_queue_perf_profile.parquet
```

The report can also be regenerated from an existing run directory:

```bash
luxquarry_allsky_engine/.venv/bin/luxquarry-allsky collect-task-queue-run \
  --queue-dir luxquarry_allsky_engine/runs/local_task_queue_gc_dense_3gpu18_batch_table_nocompress_verify \
  --out luxquarry_allsky_engine/runs/local_task_queue_gc_dense_3gpu18_batch_table_nocompress_verify/task_queue_collect_summary.json

luxquarry_allsky_engine/.venv/bin/luxquarry-allsky summarize-task-queue-perf \
  --run-dir luxquarry_allsky_engine/runs/local_task_queue_gc_dense_3gpu18_batch_table_nocompress_verify
```

The latest dense 18-frame, 3-GPU, compact, no-compression run produced:

```text
completed_frames: 18
measurement_rows: 320,768
worker_payload_max_wall_sec: 2.258
measurements_per_sec_worker_payload: 142,079
worker_parallel_efficiency: 0.946
shard_total_bytes: 34,997,955
shard_bytes_per_measurement: 109.1
```

Critical-path worker payload bottlenecks above the 5% threshold:

```text
write_measurement_shards: 0.415 sec, 18.4%
psf_device_submit_sync:   0.403 sec, 17.8%
upload_frame_to_gpu:      0.150 sec,  6.7%
read_fits:                0.130 sec,  5.7%
```

Interpretation:

- Aperture is not a material bottleneck on this workload.
- The next speed work should not be another scheduler abstraction. The current
  local queue already has good worker balance (`0.946` parallel efficiency).
- The top durable-output target is the shard writer/table-to-parquet path.
- The top GPU/science target is PSF device submit/sync and associated
  host/device synchronization. Keep PSF kernels resident and remove avoidable
  CPU/device round trips before changing science math.
- FITS read and frame upload are smaller but still above threshold; they are
  candidates for local NVMe/S3 staging, KvikIO experiments, and better overlap.

### PSF Fused-Candidate Experiment

An experimental fused PSF candidate kernel was tested locally but reverted
before commit. The idea was to remove the large temporary candidate kernel bank
(`n_candidates * MAX_PSF_PIXELS`) by computing each shifted PSF sample directly
inside the candidate fit kernel.

One-frame smoke result:

```text
old two-kernel PSF path:
  worker_payload_max_wall_sec: 0.857
  measurements_per_sec_worker_payload: 20,850
  psf_device_submit_sync: 0.064 sec

experimental fused candidate path:
  worker_payload_max_wall_sec: 1.923
  measurements_per_sec_worker_payload: 9,297
  psf_device_submit_sync: 1.254 sec
```

Numerical comparison on the same frame matched the existing PSF output columns
for the checked columns, but performance was much worse. The fused version
avoided the kernel-bank write/read, but recomputed shifted PSF samples multiple
times per candidate, which dominated the memory savings.

Decision:

- Do not replace the current PSF path with this fused recomputation approach.
- The next PSF optimization should preserve or cache shifted kernel samples
  rather than recomputing them for normalization, fitting, and chi-square.
- Better candidates are tiled/shared-memory kernel construction, smaller
  candidate-output surfaces, or a two-stage fit that avoids full chi-square for
  losing candidates.

### PSF Winner-Only Chi-Square Experiment

A second PSF experiment kept the current two-kernel candidate bank but skipped
candidate chi-square during the all-candidate fit in default SNR mode. After
the reducer selected the best candidate per target, a small extra kernel
computed chi-square only for the winners.

One-frame smoke:

```text
old path, warmed:
  worker_payload_max_wall_sec: 0.857
  measurements_per_sec_worker_payload: 20,850
  psf_device_submit_sync: 0.064 sec

winner-only chi-square, warmed:
  worker_payload_max_wall_sec: 0.766
  measurements_per_sec_worker_payload: 23,326
  psf_device_submit_sync: 0.074 sec
```

Dense 18-frame, 3-GPU run:

```text
old path:
  worker_payload_max_wall_sec: 2.258
  measurements_per_sec_worker_payload: 142,079
  psf_device_submit_sync critical path: 0.403 sec

winner-only chi-square:
  worker_payload_max_wall_sec: 2.414
  measurements_per_sec_worker_payload: 132,869
  psf_device_submit_sync critical path: 0.386 sec
```

Numerical comparison on the smoke frame matched the checked PSF output columns,
including flux, uncertainty, fit position, score, reduced chi-square, status,
sector, and kernel sum. The dense throughput regression means this was also
reverted before commit.

Decision:

- Do not add an extra winner-only chi-square kernel in this form.
- If revisiting this idea, fold winner chi-square into a later GPU-resident
  reducer/output path where it does not add another launch and synchronization
  point.

## 2026-06-30: Shard Write Sub-Stage Timing

The local task-queue profiler now breaks shard output into:

```text
write_measurement_shards        inclusive shard flush
shard_table_assembly            cuDF table construction / device-column concat
shard_column_profile            compact/full column projection
parquet_write                   cudf.to_parquet wall time
```

Dense 18-frame, 3-GPU, compact, no-compression verification:

```text
completed_frames: 18
measurement_rows: 320,768
worker_payload_max_wall_sec: 2.266
measurements_per_sec_worker_payload: 141,549
worker_parallel_efficiency: 0.957
```

Top critical-path stages:

```text
write_measurement_shards: 0.469 sec, 20.7%
psf_device_submit_sync:   0.410 sec, 18.1%
shard_table_assembly:     0.339 sec, 15.0%
read_fits:                0.138 sec,  6.1%
upload_frame_to_gpu:      0.136 sec,  6.0%
parquet_write:            0.130 sec,  5.7%
```

Shard writer breakdown by worker:

```text
worker-cuda-0: rows 107,259, write 0.398 sec, table 0.271 sec, parquet 0.126 sec
worker-cuda-1: rows 107,147, write 0.404 sec, table 0.275 sec, parquet 0.129 sec
worker-cuda-2: rows 106,362, write 0.469 sec, table 0.339 sec, parquet 0.130 sec
```

Interpretation:

- The bigger durable-output cost is not parquet write alone; it is cuDF shard
  table assembly, especially concatenating per-frame device columns into the
  shard table.
- `shard_column_profile` is negligible, so compact/full projection is not the
  problem.
- The next output-path optimization should focus on avoiding per-frame column
  concatenation at shard flush. Candidate directions: preallocate shard columns,
  append measurements into device-side buffers as frames complete, or write
  larger but less frequently assembled frame-group tables.

## 2026-06-30: Compact Profile-Aware Shard Assembly

Compact output previously assembled the full measurement table first, including
columns that compact mode later dropped, then projected down to
`COMPACT_MEASUREMENT_COLUMNS`. The worker now passes `measurement_column_profile`
into shard table assembly so compact mode avoids building unused columns before
cuDF construction.

The compact profile still preserves per-measurement science provenance:

```text
frame_group_id
image_id
catalog/source/target identifiers
sky and pixel coordinates
fits_path
detector/release
aperture settings
wavelength calibration source/file/collection
sapm_file
cwave_um/cband_um
aperture and PSF flux/status columns
```

Compact mode intentionally omits local staging-only paths and human-readable
status strings. The numeric status codes and source FITS/calibration provenance
remain.

One-frame smoke comparison:

```text
before:
  worker_payload_max_wall_sec: 0.763
  measurements_per_sec_worker_payload: 23,440
  shard_table_assembly_wall_sec: 0.152
  parquet_write_wall_sec: 0.043

after:
  worker_payload_max_wall_sec: 0.742
  measurements_per_sec_worker_payload: 24,100
  shard_table_assembly_wall_sec: 0.134
  parquet_write_wall_sec: 0.039
```

Dense 18-frame, 3-GPU comparison:

```text
before:
  worker_payload_max_wall_sec: 2.266
  measurements_per_sec_worker_payload: 141,549
  worker_parallel_efficiency: 0.957
  write_measurement_shards: 0.469 sec critical path
  shard_table_assembly: 0.339 sec critical path
  parquet_write: 0.130 sec critical path

after:
  worker_payload_max_wall_sec: 2.176
  measurements_per_sec_worker_payload: 147,391
  worker_parallel_efficiency: 0.987
  write_measurement_shards: 0.432 sec critical path
  shard_table_assembly: 0.310 sec critical path
  parquet_write: 0.122 sec critical path
```

Correctness check:

```text
same columns: true
same rows: true
same targets: true
checked science/provenance columns identical:
  aperture_flux_uJy
  cwave_um
  psf_flux_uJy
  psf_grid_best_score
  psf_status_code
  aperture_status_code
```

Interpretation:

- This is a real but small win: about 4.1% better payload throughput on the
  dense 3-GPU benchmark.
- It does not solve the output path; shard table assembly remains about 14% of
  critical-path payload wall time.
- The next output optimization should avoid concatenating per-frame device
  columns at shard flush, rather than merely trimming columns earlier.

## 2026-06-30: Shard Assembly Deep Timing

Shard assembly now reports deeper sub-stages:

```text
metadata_concat
device_column_concat
status_concat
metadata_to_cudf
column_attach
status_attach
```

Dense 18-frame, 3-GPU verification:

```text
worker_payload_max_wall_sec: 2.252
measurements_per_sec_worker_payload: 142,431
worker_parallel_efficiency: 0.953
```

Critical-path shard output stages:

```text
write_measurement_shards: 0.400 sec, 17.8%
shard_table_assembly:     0.279 sec, 12.4%
metadata_to_cudf:         0.219 sec,  9.7%
parquet_write:            0.123 sec,  5.5%
metadata_concat:          0.020 sec,  0.9%
column_attach:            0.009 sec,  0.4%
device_column_concat:     0.004 sec,  0.2%
status_attach:            0.004 sec,  0.2%
```

Interpretation:

- The earlier assumption that device-column concatenation was the main shard
  assembly problem was wrong for this workload.
- The dominant assembly sub-stage is converting metadata from pandas to cuDF,
  mostly string/provenance columns.
- Device-column concatenation is cheap here.
- The next output-path optimization should attack metadata conversion: keep
  metadata GPU-native, dictionary-code repeated provenance fields, or split
  row-level numeric measurements from sidecar provenance tables where that is
  scientifically acceptable.

### Scalar Metadata Experiment

An experiment attempted to reduce `metadata_to_cudf` by removing constant
compact provenance columns from the pandas metadata table, converting the
smaller table with `cudf.from_pandas`, then assigning constant columns as cuDF
scalar columns.

Correctness:

```text
same columns: true
same rows: true
same targets: true
checked provenance/science columns identical:
  fits_path
  wavelength_calibration_file
  sapm_file
  aperture_flux_uJy
  cwave_um
  psf_flux_uJy
  psf_status_code
```

Dense 18-frame, 3-GPU comparison:

```text
baseline deep timing:
  worker_payload_max_wall_sec: 2.252
  measurements_per_sec_worker_payload: 142,431
  write_measurement_shards: 0.400 sec critical path
  shard_table_assembly: 0.279 sec critical path
  metadata_to_cudf: 0.219 sec critical path
  column_attach: 0.009 sec critical path

scalar metadata experiment:
  worker_payload_max_wall_sec: 2.286
  measurements_per_sec_worker_payload: 140,339
  write_measurement_shards: 0.466 sec critical path
  shard_table_assembly: 0.342 sec critical path
  metadata_to_cudf: 0.197 sec critical path
  column_attach: 0.013 sec critical path
```

Decision:

- Reverted before commit. The experiment reduced `metadata_to_cudf`, but
  total shard assembly got worse.
- Assigning many scalar/string columns after cuDF construction is not a free
  win.
- The better metadata strategy is likely dictionary coding or a sidecar
  provenance table keyed by frame/calibration IDs, not per-column scalar
  assignment after `from_pandas`.

## 2026-06-30: Resident Task Input Indexing

The local task queue previously loaded resident source inputs once per worker
service, but each claimed task still selected its frame subset by scanning the
full projected-target table with pandas `.isin()`. That is acceptable for a
small number of large tasks, but it becomes setup overhead when we intentionally
use many small tasks for horizontal scheduling.

Patch:

- Build `source_manifest_by_frame` once per worker service.
- Build `source_target_indices_by_frame` once per worker service using
  `source_targets.groupby("frame_group_id").indices`.
- For each task, select frames by indexed `.loc[]` and target rows by
  positional `.take()` rather than rescanning the full source target table.
- Report `source_input_index_wall_sec`,
  `resident_source_frame_count`, and `resident_source_target_rows` in worker
  service and local-runner summaries.

Smoke command:

```bash
cd /home/clive/dev/NIROSETI_SPHEREx
luxquarry_allsky_engine/.venv/bin/luxquarry-allsky run-local-task-queue \
  --manifest luxquarry_allsky_engine/runs/manifest_galactic_core_nearest20/frame_manifest.parquet \
  --projected-targets luxquarry_allsky_engine/runs/projected_targets_galactic_core_nearest20_allcat_g11_16_n20000/frame_targets_projected.parquet \
  --out-dir luxquarry_allsky_engine/runs/local_task_queue_resident_index_smoke \
  --run-id local_task_queue_resident_index_smoke \
  --devices cuda:0 \
  --frames-per-task 1 \
  --limit-frames 6 \
  --enable-psf \
  --psf-kernel-build-mode gpu_bilinear \
  --measurement-column-profile compact \
  --measurement-parquet-compression none \
  --shard-batch-frames 6 \
  --prefetch-frames 1
```

Result:

```text
source table: 20 frames, 400,000 projected target rows
completed_frames: 6
measurement_rows: 107,259
failed_frames: 0
failed_tasks: 0
task_input_select_wall_sec: 0.0033-0.0041 sec per one-frame task
```

A one-frame summary smoke confirmed the runner emits the new resident-input
fields:

```text
resident_source_frame_count: 20
resident_source_target_rows: 400,000
source_input_load_wall_sec: 0.155
source_input_index_wall_sec: 0.064
```

Provenance check:

```text
compact shard rows: 17,875
compact shard columns: 53
present per measurement:
  frame_group_id
  image_id
  catalog
  target_id
  source_id
  ra_deg
  dec_deg
  x_pix
  y_pix
  fits_path
  detector
  wavelength_source
  wavelength_calibration_file
  sapm_file
  cwave_um
  cband_um
  aperture_flux_uJy
  psf_flux_uJy
```

Decision:

- Keep. This is a low-risk setup optimization for many-task scheduling.
- It does not change science math or measurement schema.
- It does not attack the current dense-run critical path; FITS reads, parquet
  writes, metadata conversion, and PSF gather remain larger targets.

## 2026-06-30: Compact Metadata Construction and Non-Batch Fix

Patch:

- Compact measurement mode now skips `local_fits_path` while building
  per-frame pandas metadata instead of adding it and trimming it later.
- The durable source `fits_path` is still emitted per measurement.
- The non-batch table path now passes `measurement_column_profile` into
  `_measurement_to_cudf()`.
- Fixed the non-batch path to unpack `_measurement_to_cudf()` correctly. Before
  this fix it treated the returned `(table, timings)` tuple as a table, which
  could yield a completed task with a failed frame and bogus row accounting.

Batch compact smoke:

```text
run: local_task_queue_compact_metadata_early_smoke
completed_frames: 1
measurement_rows: 17,875
failed_frames: 0
column_count: 53
```

Schema/provenance comparison against the previous compact reference:

```text
rows equal: true
columns equal: true
checked equal:
  frame_group_id
  image_id
  target_id
  fits_path
  wavelength_calibration_file
  sapm_file
  cwave_um
  cband_um
  aperture_flux_uJy
  psf_flux_uJy
  aperture_status_code
  psf_status_code
local_fits_path present in compact output: false
```

Non-batch compact verification:

```text
bad pre-fix smoke:
  run: local_task_queue_compact_nonbatch_smoke
  error: AttributeError: 'tuple' object has no attribute 'columns'
  failed_frames: 1
  measurement_rows: 2

fixed smoke:
  run: local_task_queue_compact_nonbatch_fix_smoke
  completed_frames: 1
  failed_frames: 0
  measurement_rows: 17,875
  column_count: 53
```

Dense 18-frame, 3-GPU verification:

```text
run: local_task_queue_compact_metadata_early_3gpu18_verify
completed_frames: 18
measurement_rows: 320,768
failed_frames: 0
worker_parallel_efficiency: 0.947
worker_payload_max_wall_sec: 2.913
measurements_per_sec_worker_payload: 110,101
```

Interpretation:

- This did not set a throughput record. The dense verification was dominated by
  FITS/calibration read variance and came in below the prior best retained
  compact-output run.
- Keep anyway because it fixes a real non-batch compact correctness bug and
  makes compact metadata construction more direct.
- The next meaningful performance work remains the same: reduce FITS read/cache
  overhead, reduce metadata-to-cuDF conversion, and reduce PSF gather/host-sync
  work.

## 2026-06-30: Worker In-Frame Target Prefilter

Patch:

- `process_frame_batch()` now filters the task target table to `in_frame`
  targets once during batch setup.
- The frame loop now reads an already in-frame group from `targets_by_frame`
  instead of applying `frame_targets_all["in_frame"].astype(bool)` for every
  frame.
- Worker summaries now include:
  - `input_projected_rows`
  - `in_frame_projected_rows`
  - `target_setup_wall_sec`

One-frame compact+PSF smoke:

```text
run: local_task_queue_inframe_prefilter_smoke1
input_projected_rows: 20,000
in_frame_projected_rows: 18,088
target_setup_wall_sec: 0.0080
completed_frames: 1
measurement_rows: 17,875
failed_frames: 0
```

Schema/provenance comparison against the prior compact reference:

```text
rows equal: true
columns equal: true
checked equal:
  frame_group_id
  image_id
  target_id
  fits_path
  cwave_um
  cband_um
  aperture_flux_uJy
  psf_flux_uJy
  aperture_status_code
  psf_status_code
```

Six one-frame task verification:

```text
run: local_task_queue_inframe_prefilter_6f_verify
completed_frames: 6
measurement_rows: 107,259
ok_measurement_rows: 104,665
failed_frames: 0
measurements_per_sec_worker_payload: 52,270
target_setup_wall_sec total: 0.0399
per-task target setup: 0.0063-0.0080 sec
```

Shard manifest comparison against the previous six-frame compact verification:

```text
shard_count: 6 -> 6
measurement_rows: 107,259 -> 107,259
ok_rows: 104,665 -> 104,665
```

Decision:

- Keep. This is a modest setup cleanup, not a critical-path breakthrough.
- The effect is largest for many small scheduler tasks because it removes
  repeated per-frame target masking and gives us explicit target setup timing.
- Larger performance work remains in FITS/cache I/O, output metadata
  conversion, and PSF host-sync/gather paths.

## 2026-06-30: Async Shard Writes Default for Local Task Queue

The high-level `run-local-task-queue` command now enables async shard writes by
default. It still accepts `--async-shard-writes`, but callers now use
`--sync-shard-writes` to force the old synchronous behavior. Low-level worker
commands remain explicit.

The benchmark shape intentionally uses one six-frame task with one shard per
frame. This gives the worker useful frame work to overlap with queued shard
writes; one-frame tasks cannot benefit much because each task immediately waits
at completion.

Synchronous baseline:

```text
run: local_task_queue_shard_async_sync_baseline_6f
frames_per_task: 6
shard_batch_frames: 1
completed_frames: 6
measurement_rows: 107,259
ok_measurement_rows: 104,665
failed_frames: 0
worker_payload_max_wall_sec: 1.759
measurements_per_sec_worker_payload: 60,980
write_measurement_shards critical path: 0.542 sec
```

Async enabled:

```text
run: local_task_queue_shard_async_enabled_6f
frames_per_task: 6
shard_batch_frames: 1
completed_frames: 6
measurement_rows: 107,259
ok_measurement_rows: 104,665
failed_frames: 0
worker_payload_max_wall_sec: 1.458
measurements_per_sec_worker_payload: 73,581
```

Manifest comparison:

```text
shard_count: 6 -> 6
measurement_rows: 107,259 -> 107,259
ok_rows: 104,665 -> 104,665
bytes: 16,953,504 -> 16,953,504
```

Default-path smoke:

```text
run: local_task_queue_async_default_report_smoke2
command: run-local-task-queue with no explicit --async-shard-writes
recorded worker command includes: --async-shard-writes
completed_frames: 2
measurement_rows: 35,717
ok_measurement_rows: 34,877
failed_frames: 0
async_shard_write_wait_wall_sec: 0.059
```

Sync opt-out smoke:

```text
run: local_task_queue_sync_flag_smoke1
command: run-local-task-queue --sync-shard-writes
recorded worker command omits: --async-shard-writes
completed_frames: 1
measurement_rows: 17,875
failed_frames: 0
async_shard_writes: false
```

Profiler fix:

- Async shard writer raw work is now reported under phase `async_writer`.
- The worker-payload report now shows the real hot-path costs:
  `submit_measurement_shard` and `async_shard_write_wait`.
- Example from `local_task_queue_async_default_report_smoke2`:

```text
write_measurement_shards: async_writer, 0.260 sec critical writer time
shard_table_assembly: async_writer, 0.178 sec critical writer time
metadata_to_cudf: async_writer, 0.146 sec critical writer time
parquet_write: async_writer, 0.082 sec critical writer time
async_shard_write_wait: worker_payload, 0.059 sec critical path
submit_measurement_shard: worker_payload, 0.001 sec critical path
```

Decision:

- Keep and promote for `run-local-task-queue`. In this overlap-friendly shape,
  async writes improved worker-payload throughput by about 21%.
- Do not claim raw parquet writing became faster. The win is overlap: the frame
  loop keeps moving while the writer thread builds/writes durable shards.
- For fully accurate bottleneck reading, use the updated performance profile and
  distinguish `async_writer` from `worker_payload`.

## 2026-06-30 / Async shard size check

Question:

Should the local task-queue default move from one-frame measurement shards to a
larger six-frame shard now that async writes are enabled by default?

Comparable runs:

```text
run: local_task_queue_shard_async_enabled_6f
frames_per_task: 6
shard_batch_frames: 1
completed_frames: 6
measurement_rows: 107,259
ok_measurement_rows: 104,665
shard_count: 6
shard_total_bytes: 16,953,504
worker_payload_max_wall_sec: 1.458
measurements_per_sec_worker_payload: 73,581
total_wall_sec: 4.379
```

```text
run: local_task_queue_shard_batch6_async_6f
frames_per_task: 6
shard_batch_frames: 6
completed_frames: 6
measurement_rows: 107,259
ok_measurement_rows: 104,665
shard_count: 1
shard_total_bytes: 11,584,621
worker_payload_max_wall_sec: 1.544
measurements_per_sec_worker_payload: 69,447
total_wall_sec: 4.492
```

Schema/provenance check:

```text
compact shard columns: 53
kept durable provenance:
  catalog
  target_id
  source_id
  ra_deg
  dec_deg
  frame_group_id
  image_id
  fits_path
  detector
  release
  wavelength_source
  wavelength_calibration_file
  wavelength_calibration_collection
  flags_summary
  cwave_um
  aperture_status_code
  psf_status_code
dropped in compact mode:
  local_fits_path
```

Decision:

- Keep the one-frame shard default for now.
- Six-frame shards reduce bytes and shard count, but this run lost enough async
  overlap that the worker payload was slower.
- Both shapes are runnable and row-equivalent. Larger shards remain useful for
  storage/viewer-pressure experiments, but they are not the default performance
  setting from this evidence.

## 2026-06-30 / Frame-boundary profile cleanup

Purpose:

Avoid mistaking overlapped prefetch work for critical-path GPU-worker time. The
frame worker can prefetch FITS payloads in a background thread. In that mode,
the critical path is `payload_wait`, not the full summed `fits_read_wall_sec`.

Run:

```text
run: local_task_queue_frame_boundary_profile_6f
frames_per_task: 6
devices: cuda:0
prefetch_frames: 1
shard_batch_frames: 1
measurement_column_profile: compact
measurement_parquet_compression: none
enable_psf: true
psf_kernel_build_mode: gpu_bilinear
completed_frames: 6
measurement_rows: 107,259
ok_measurement_rows: 104,665
failed_frames: 0
worker_payload_max_wall_sec: 1.451
measurements_per_sec_worker_payload: 73,900
```

New per-frame timing fields:

```text
coordinate_extract_wall_sec
edge_filter_wall_sec
target_row_select_wall_sec
metadata_build_wall_sec
```

Profiler correction:

- If `payload_prefetched` is true, `fits_read_wall_sec` and
  `staging_wall_sec` are reported as `prefetch_worker`, not
  `worker_payload`.
- `psf_gather` is labeled as `cupy_psf_best_candidate_gather`. It is device-side
  CuPy/DLPack candidate selection, not a CPU gather.

Corrected top worker-payload bottlenecks:

```text
upload_frame_to_gpu:       0.162 sec, 11.18%
async_shard_write_wait:    0.127 sec,  8.75%
psf_device_submit_sync:    0.096 sec,  6.62%
psf_gather:                0.095 sec,  6.52%
payload_wait:              0.091 sec,  6.30%
```

Important non-critical async-writer costs:

```text
metadata_to_cudf:          0.274 sec, async_writer phase
column_attach:             0.084 sec, async_writer phase
```

Decision:

- Do not chase FITS read as a direct worker critical-path problem while prefetch
  is active; it is mostly overlapped in this shape.
- The next meaningful architecture work is larger GPU work units: frame-stack
  processing, resident target arrays, and PSF candidate/gather fusion where
  possible.
- Keep table/metadata assembly visible because it can become a real bottleneck
  when async writer wait grows or output pressure increases.
