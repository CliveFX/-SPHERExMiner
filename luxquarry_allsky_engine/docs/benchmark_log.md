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
