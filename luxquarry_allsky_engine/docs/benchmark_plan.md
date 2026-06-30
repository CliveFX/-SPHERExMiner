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
  "frame_count": 0,
  "target_count": 0,
  "measurement_count": 0,
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
    "parquet_rows_per_sec": 0.0
  },
  "io": {
    "fits_read_bytes": 0,
    "catalog_read_bytes": 0,
    "parquet_write_bytes": 0,
    "local_cache_peak_bytes": 0
  },
  "gpu": {
    "device_count": 0,
    "kernel_wall_sec": 0.0,
    "estimated_occupancy": null
  }
}
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
