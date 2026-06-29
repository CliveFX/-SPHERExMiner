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
