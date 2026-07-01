# Benchmark Plan

## First Benchmark

Run a local fixed workload:

```text
100 FITS frames
Gaia + 2MASS targets
aperture photometry only
500 GB local cache cap
```

## Required Metrics

Write `perf_summary.json` with:

```json
{
  "campaign_id": "string",
  "output_mode": "audit|survey",
  "catalog_selection": "gaia_g_8_14|twomass_all_usable|combined",
  "frame_count": 0,
  "target_count": 0,
  "gaia_target_count": 0,
  "twomass_target_count": 0,
  "deduplicated_target_count": 0,
  "measurement_count": 0,
  "spectra_count": 0,
  "candidate_count": 0,
  "retained_raw_measurement_count": 0,
  "total_wall_sec": 0.0,
  "stage_wall_sec": {
    "stage_fits": 0.0,
    "catalog_query": 0.0,
    "wcs_projection": 0.0,
    "gpu_photometry": 0.0,
    "write_measurements": 0.0,
    "assemble_spectra": 0.0,
    "score_candidates": 0.0
  },
  "throughput": {
    "frames_per_sec": 0.0,
    "measurements_per_sec": 0.0,
    "measurements_per_gpu_sec": 0.0,
    "spectra_per_sec": 0.0,
    "spectra_per_gpu_sec": 0.0,
    "parquet_rows_per_sec": 0.0
  },
  "io": {
    "fits_read_bytes": 0,
    "catalog_read_bytes": 0,
    "parquet_write_bytes": 0,
    "bytes_per_measurement": 0.0,
    "bytes_per_spectrum": 0.0,
    "local_cache_peak_bytes": 0
  },
  "gpu": {
    "device_count": 0,
    "kernel_wall_sec": 0.0,
    "estimated_occupancy": null
  },
  "economics": {
    "target_cloud_instance": "string",
    "estimated_cloud_gpu_hourly_cost": 0.0,
    "estimated_cloud_total_cost": 0.0,
    "estimated_cost_per_billion_measurements": 0.0,
    "estimated_cost_per_million_spectra": 0.0,
    "fits_under_5000_usd_for_gaia_g_8_14_plus_2mass": false
  }
}
```

The v2 economic gate is documented in `survey_output_contract.md`: `$5k` should
buy the accessible-sky survey for Gaia G ~= 8-14 plus all 2MASS point-source
catalog rows present in the processed local 2MASS PSC cache. Benchmarks that do
not estimate this cost are incomplete. Benchmarks built from capped
projected-target tables are useful performance samples, but they are not valid
all-2MASS cost estimates.

Run the survey planner after building a manifest/projected-target table:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky plan-survey-economics \
  --manifest runs/manifest_galactic_core_nearest20/frame_manifest.parquet \
  --projected-targets runs/projected_targets_galactic_core_nearest20_allcat_g11_16_n20000/frame_targets_projected.parquet \
  --out-dir runs/survey_plan_gc_nearest20_combined \
  --catalog-selection combined \
  --output-mode survey \
  --raw-retention-fraction 0.01 \
  --measurements-per-gpu-sec 73687 \
  --gpu-hourly-cost 6.88 \
  --gpu-count 8
```

For the actual Gaia G 8-14 plus all-2MASS economics gate, first build/project
targets with uncapped 2MASS input:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky build-frame-targets \
  --manifest runs/manifest_sample/frame_manifest.parquet \
  --out runs/frame_targets_sample_all2mass/frame_targets.parquet \
  --catalog all \
  --gaia-g-min 8 \
  --gaia-g-max 14 \
  --gaia-max-sources-per-frame 0 \
  --all-2mass

.venv/bin/luxquarry-allsky project-frame-targets \
  --manifest runs/manifest_sample/frame_manifest.parquet \
  --frame-targets runs/frame_targets_sample_all2mass/frame_targets.parquet \
  --out runs/projected_targets_sample_all2mass/frame_targets_projected.parquet
```

Use capped `--gaia-max-sources-per-frame` and
`--twomass-max-sources-per-frame` values for smoke tests. Do not treat capped
tables as all-2MASS economics estimates.

For any number that will be used as an all-2MASS cloud-cost claim, add:

```bash
--require-all-2mass-input
```

and confirm the resulting economics JSON contains:

```text
all_2mass_input_valid: true
all_2mass_input_status: proven
```

If it says `unknown` or `invalid`, the estimate can still be useful for
software smoke testing, but it is not evidence for the `$5k` target.

Run the economics estimator directly when materialized plan products are not
needed:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky estimate-survey-economics \
  --manifest runs/manifest_galactic_core_nearest20/frame_manifest.parquet \
  --projected-targets runs/projected_targets_galactic_core_nearest20_allcat_g11_16_n20000/frame_targets_projected.parquet \
  --out runs/survey_economics_gc_nearest20_combined/summary.json \
  --catalog-selection combined \
  --output-mode survey \
  --raw-retention-fraction 0.01 \
  --measurements-per-gpu-sec 73687 \
  --gpu-hourly-cost 6.88 \
  --gpu-count 8
```

When only representative cell plans exist, extrapolate them:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky extrapolate-survey-sample \
  --plan-summary runs/survey_plan_gc_nearest20_combined/survey_plan_summary.json \
  --out-dir runs/survey_sample_extrapolation_smoke \
  --target-cell-count 192
```

The repeatable local sweep command is:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky benchmark-object-staging \
  --manifest runs/manifest_smoke_v2_s3/frame_manifest.parquet \
  --out-dir runs/object_staging_bench_s3_concurrency_smoke \
  --cache-dir /tmp/luxquarry_object_staging_s3_concurrency_smoke \
  --concurrency 1,2,4,8 \
  --limit 10 \
  --cache-mode per-concurrency \
  --require-s3
```

Run the staging-only benchmark before the worker sweep on each target machine or
instance type. It isolates S3/object-store and local-cache throughput from FITS
read, GPU photometry, and parquet write costs.

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky run-dispatch-benchmark-sweep \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/dispatch_benchmark_sweep_smoke \
  --run-id dispatch_sweep_smoke \
  --devices cuda:0 \
  --workers-per-device 1 \
  --limit-frames 2,10,100 \
  --shard-batch-frames 1,2,5,10 \
  --prefetch-frames 0,2 \
  --local-cache-dir /tmp/luxquarry_stage_smoke \
  --score-baseline
```

Add `--worker-only` to skip spectra assembly, scoring, and campaign
finalization. Worker-only sweeps are the correct tool for isolating worker
process launch overhead from persistent GPU worker payload throughput.

For PSF occupancy work, run the same sweep with PSF enabled:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky run-dispatch-benchmark-sweep \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/dispatch_benchmark_psf_worker_only \
  --run-id dispatch_psf_worker_only \
  --devices cuda:0 \
  --workers-per-device 1 \
  --limit-frames 10,100 \
  --shard-batch-frames 5,10 \
  --prefetch-frames 0,2 \
  --worker-only \
  --enable-psf \
  --psf-kernel-build-mode gpu_spline
```

PSF-enabled sweep rows include:

```text
psf_measurement_rows
ok_psf_rows
psf_candidate_count
worker_min_wall_sec
worker_avg_wall_sec
worker_max_wall_sec
worker_parallel_efficiency
worker_wall_skew_ratio
staging_wall_sec
payload_wait_wall_sec
fits_read_wall_sec
frame_upload_wall_sec
psf_candidate_grid_wall_sec
psf_kernel_wall_sec
psf_device_submit_sync_wall_sec
psf_spline_coeff_wall_sec
psf_upload_wall_sec
psf_gather_wall_sec
table_wall_sec
shard_submit_wall_sec
async_shard_write_wait_wall_sec
write_wall_sec
max_worker_payload_wait_wall_sec
max_worker_staging_wall_sec
max_worker_fits_read_wall_sec
max_worker_psf_kernel_wall_sec
max_worker_async_shard_write_wait_wall_sec
max_worker_shard_write_wall_sec
psf_candidates_per_sec_device_submit_sync
```

`psf_candidates_per_sec_device_submit_sync` is the closest current proxy for
GPU PSF math throughput. Total and worker-payload candidate/sec include process
launch, FITS read, table assembly, and parquet overhead, so use them for cost
modeling rather than kernel occupancy.

For multi-GPU runs, the summed stage timings show total work across workers,
not critical wall time. Use the `max_worker_*` columns to identify what actually
limits elapsed worker payload. The first dense 3-GPU local-cache sweep showed
that warm local cache plus prefetch reduces max-worker payload wait from about
0.73 sec to about 0.03 sec on the 18-frame test. After that, PSF kernel work
and async shard-write wait are the measured critical-path costs.

To bound write overhead, run the same worker-only benchmark with
`--discard-measurement-shards`. This is not a science-output mode; it exists to
measure photometry throughput without parquet shard flush:

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

The first no-write bound saved about 0.52 sec of max-worker critical path and
raised payload throughput from about 147k to about 193k measurements/sec across
3 GPUs. Production work must recover as much of that gap as possible while
still writing durable parquet shards.

To test schema pressure separately from write/no-write pressure, use:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky run-dispatch-benchmark-sweep \
  --manifest runs/manifest_galactic_core_nearest20/frame_manifest.parquet \
  --projected-targets runs/projected_targets_galactic_core_nearest20_allcat_g11_16_n20000/frame_targets_projected.parquet \
  --out-dir runs/dispatch_benchmark_psf_gc_dense_compact \
  --run-id dispatch_psf_gc_dense_compact \
  --devices cuda:0,cuda:1,cuda:2 \
  --workers-per-device 1 \
  --limit-frames 18 \
  --shard-batch-frames 6 \
  --prefetch-frames 2 \
  --local-cache-dir /tmp/luxquarry_gc_dense_cache_prefetch2 \
  --worker-only \
  --enable-psf \
  --psf-kernel-build-mode gpu_spline \
  --measurement-column-profile compact
```

`compact` is not allowed to drop science audit provenance. It keeps per-point
`fits_path`, `frame_group_id`, `image_id`, detector, wavelength calibration,
SAPM calibration, aperture geometry, wavelength, flags, aperture flux, and PSF
flux/diagnostics. It drops ephemeral staging-only fields such as
`local_fits_path`.

The first compact/full comparison showed nearly identical bytes per row and
payload time on the dense 18-frame workload. Treat compact as schema control,
not as the current solution to output write pressure.

For persistent-worker validation, use the task-queue path instead of repeatedly
paying launch/setup cost:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky write-task-queue \
  --manifest runs/manifest_galactic_core_nearest20/frame_manifest.parquet \
  --projected-targets runs/projected_targets_galactic_core_nearest20_allcat_g11_16_n20000/frame_targets_projected.parquet \
  --out-dir runs/task_queue_gc_dense_smoke \
  --campaign-id task_queue_gc_dense_smoke \
  --frames-per-task 2 \
  --limit-frames 4

.venv/bin/luxquarry-allsky run-gpu-worker-service \
  --queue-dir runs/task_queue_gc_dense_smoke \
  --out-dir runs/task_queue_gc_dense_smoke/worker_cuda0 \
  --run-id task_queue_gc_dense_smoke \
  --worker-id worker-cuda0 \
  --device cuda:0 \
  --shard-batch-frames 2 \
  --prefetch-frames 2 \
  --local-cache-dir /tmp/luxquarry_gc_dense_cache_prefetch2 \
  --measurement-column-profile compact \
  --enable-psf \
  --psf-kernel-build-mode gpu_spline
```

The task-queue service is the local analogue of the eventual Kubernetes worker:
claim frame batches, keep GPU runtime initialized, write independent
measurement shards, and let a reducer assemble spectra afterward.

Use dense real fields when possible. The sparse 10-frame smoke set has only a
few hundred selected targets per frame and underfeeds the PSF kernels. A better
occupancy smoke is the Galactic-plane-adjacent set:

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
  --psf-kernel-build-mode gpu_spline
```

If a real field is still too sparse, create a benchmark-only target-density
ladder by multiplying a projected-target parquet:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky inflate-projected-targets \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out runs/projected_targets_smoke_current/frame_targets_projected_x10.parquet \
  --repeat-factor 10 \
  --jitter-pix 0.05
```

Inflated target parquets are synthetic benchmark inputs only. They preserve the
frame geometry and add `benchmark_inflated`, `benchmark_repeat_index`, and
`benchmark_parent_target_id` columns so they cannot be confused with a science
catalog.

`frame_upload_wall_sec` is the shared upload of FITS `IMAGE`, `VARIANCE`, and
`FLAGS` to GPU. Aperture and PSF both consume this resident frame object.
`psf_upload_wall_sec` should stay small; it covers PSF-specific buffers such as
the shifted-kernel source data and subpixel offset bank, not a second copy of
the raw frame. `psf_candidate_grid_wall_sec` should be near zero because the
PSF candidate grid is represented as target arrays plus a small offset table;
the full target-by-offset coordinates are generated inside the GPU kernels.

It writes:

```text
sweep_results.parquet
sweep_results.json
profile_summary.parquet
profile_summary.json
perf_summary.json
trials/<trial_run_id>/
```

`perf_summary.json` reports both `best_trial` by end-to-end measurements/sec
and `best_payload_trial` by worker payload measurements/sec. The split matters:
small jobs are dominated by worker launch/setup, while large jobs should expose
the actual GPU and I/O throughput.

## Function-Level Profiling

Every benchmark run should optionally emit function/stage profiling:

```text
profile_summary.parquet
profile_summary.json
```

The profiler must identify any function or stage consuming more than 5% of
total wall time. Those rows feed the acceleration audit.

Minimum fields:

```text
stage
function_or_script
wall_time_sec
wall_time_pct
cpu_time_sec
gpu_time_sec
io_wait_sec
call_count
rows_in
rows_out
bytes_in
bytes_out
backend
```

The goal is not just to make the code faster. The goal is to maintain a ranked
queue of measured bottlenecks and accelerate the highest-impact pieces first.

## Correctness Comparison

Compare against the current miner on overlapping targets.

Required checks:

- aperture flux absolute error
- aperture flux percent error
- uncertainty percent error
- flag agreement
- wavelength agreement
- target pixel coordinate agreement

The benchmark should fail if wavelength provenance differs unexpectedly.

## Benchmark Ladder

Run increasing workloads:

1. 10 frames
2. 100 frames
3. 1,000 frames
4. One HPX region
5. One night-scale campaign
6. All-sky dry run manifest

Each rung must report cost model estimates for EKS:

- expected pod hours
- expected S3 read/write
- expected output storage
- expected frames/sec/node
