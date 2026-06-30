# Long-Lived GPU Worker Service Design

## Why This Exists

The local worker-only benchmark shows the current dispatch shape is too
expensive for small work packets:

```text
2-frame worker payload: 0.40-0.44 sec
parent-observed worker phase: 3.25 sec
launch/setup overhead: 2.8 sec
```

That is acceptable for early smoke tests, but it is the wrong shape for a
high-throughput all-sky miner. The next generation should keep each GPU worker
alive and feed it many frame batches through a queue. Startup should happen once
per worker lifetime, not once per small dispatch.

## Current Shape

Current local and Kubernetes dispatch:

```text
plan-gpu-dispatch
  -> materialized worker inputs
  -> one process/job per worker slice
  -> worker loads CUDA/RAPIDS/Warp
  -> worker processes frames
  -> worker exits
  -> finalize-dispatch-run
```

This already gives good correctness boundaries:

- independent worker outputs
- append-only measurement shards
- per-worker status JSON
- no live database in the hot loop
- postprocess split from photometry

But it still pays worker startup per dispatch slice.

## Target Shape

Long-lived service:

```text
controller
  -> writes frame-batch tasks
  -> starts N GPU workers

gpu worker process, one per GPU
  -> initialize CUDA/RAPIDS/Warp once
  -> initialize RMM memory pool once
  -> keep detector calibration maps resident on GPU
  -> poll/claim next frame-batch task
  -> stage FITS to local SSD
  -> run frame-first GPU photometry
  -> write measurement shard
  -> atomically mark task complete
  -> repeat until queue drained or shutdown requested

postprocess
  -> collect shard manifest
  -> assemble spectra with cuDF/Dask-cuDF
  -> score baseline/injected candidates
  -> recovery/index products
```

## Work Unit

The queue unit should be a frame batch, not a target and not a single tiny
frame unless retry pressure requires it.

Suggested fields:

```text
task_id
campaign_id
batch_index
frame_group_ids
frame_manifest_path_or_key
projected_targets_path_or_key
path_overrides_path_or_key
output_prefix
cache_policy
attempt
created_utc
lease_owner
lease_expires_utc
status
```

Initial local file-backed tasks can be JSON files. EKS/S3 tasks can be JSON
objects with lease markers, or later SQS/DynamoDB if file/object leasing becomes
the bottleneck.

## Worker Contract

Worker process startup:

1. Select CUDA device.
2. Initialize RAPIDS/cuDF/CuPy/Warp.
3. Configure RMM memory pool.
4. Allocate reusable GPU buffers.
5. Load detector/wavelength/calibration metadata cache.
6. Open status writer.

Per task:

1. Claim task with an atomic lease.
2. Stage FITS and task parquets to local SSD/NVMe.
3. Load frame manifest and projected targets.
4. Group rows by frame.
5. For each frame:
   - read image, variance, flags, wavelength fields
   - upload or reuse GPU buffers
   - run aperture kernel
   - later run PSF kernel
   - emit measurement rows with full provenance
6. Write shard to a temporary path.
7. Rename or commit shard atomically.
8. Write task summary.
9. Mark task complete.

A task must be idempotent. Rerunning the same task should overwrite only its own
temporary/output prefix or produce the same shard name.

## Queue and Lease Model

Minimum viable local model:

```text
queue/
  pending/<task_id>.json
  leased/<task_id>.<worker_id>.json
  complete/<task_id>.json
  failed/<task_id>.<attempt>.json
```

Claim:

```text
rename pending/<task>.json -> leased/<task>.<worker>.json
```

Complete:

```text
write output shard
write complete/<task>.json.tmp
rename complete/<task>.json.tmp -> complete/<task>.json
remove leased marker
```

Recovery:

- If a lease expires, a controller moves it back to pending with `attempt + 1`.
- If attempts exceed a configured max, move to failed.
- Workers never coordinate with each other directly.

For S3, object renames are not atomic. The EKS version should use either:

- SQS visibility timeouts for task leases; or
- DynamoDB conditional writes for `task_id` status; or
- Kubernetes indexed Jobs for coarse batches until the queue service is ready.

## Status Model

The dashboard should not query parquet data for live status.

Worker writes one small JSON heartbeat:

```text
status/workers/<worker_id>.json
```

Task summaries:

```text
status/tasks/complete/<task_id>.json
status/tasks/failed/<task_id>.json
```

Aggregate status is a periodic reducer:

```text
dispatch_status.json.tmp -> dispatch_status.json
```

Suggested metrics:

```text
pending_tasks
leased_tasks
complete_tasks
failed_tasks
active_workers
frames_done
measurements_written
rows_per_sec_total
rows_per_sec_payload
gpu_worker_payload_wall_sec
task_stage_wall_sec
cache_hit_fraction
fits_read_bytes
parquet_write_bytes
```

## GPU/RAPIDS Use

Keep the hot path GPU-first:

- CuPy/Warp for aperture and PSF kernels.
- RMM memory pool per worker process.
- cuDF for per-task measurement table assembly where possible.
- KvikIO/Parquet writes if benchmarks show a real gain.
- Dask-cuDF for postprocess over many shards.

Avoid GPU round trips for:

- calibration arrays
- wavelength maps
- PSF kernel banks
- target pixel arrays inside a frame batch

CPU remains acceptable for:

- Astropy WCS/projection until a verified accelerated replacement exists
- FITS header parsing
- small task/heartbeat JSON

Anything above 5% wall time in `profile_summary.parquet` needs an explicit
acceleration decision.

## Local Prototype

The local prototype now includes:

```text
write-task-queue
run-gpu-worker-service
collect-task-queue-run
```

Example:

```bash
luxquarry-allsky write-task-queue \
  --manifest runs/manifest.parquet \
  --projected-targets runs/projected_targets.parquet \
  --out-dir runs/service_smoke/queue \
  --frames-per-task 25

luxquarry-allsky run-gpu-worker-service \
  --queue-dir runs/service_smoke/queue \
  --out-dir runs/service_smoke \
  --device cuda:0 \
  --local-cache-dir /tmp/luxquarry_stage \
  --max-tasks 100

luxquarry-allsky collect-task-queue-run \
  --queue-dir runs/service_smoke/queue \
  --out runs/service_smoke/measurement_shard_manifest.parquet
```

The service smoke should compare against `run-dispatch-benchmark-sweep` on the
same frame count. The expected improvement is not faster kernel math; it is
removing repeated process/RAPIDS/CUDA startup from many small dispatches.

Current v1 behavior:

- `write-task-queue` splits a frame manifest into frame-batch JSON tasks and
  materializes per-task manifest/projected-target parquet slices.
- `run-gpu-worker-service` initializes RAPIDS/CuPy/Warp once, constructs one
  `PersistentGpuFrameWorker`, claims tasks with atomic file renames, and writes
  normal measurement shards.
- `collect-task-queue-run` writes a `measurement_shard_manifest.parquet` that
  is directly consumable by `assemble-spectra`.

Known v1 limitations:

- It is still a local filesystem queue, not an EKS queue.
- It still reads per-task parquet inputs instead of holding one global manifest
  index in process.
- It still calls the existing worker `.run()` method once per task, so the next
  optimization is a lower-level `process_frame_batch()` API that avoids
  rebuilding per-run summaries and task-local data structures.
- It has no lease expiry/retry controller yet; failed tasks are isolated, but
  stale leases are not automatically requeued.

## EKS Mapping

Near term:

- Use coarse materialized Kubernetes Jobs for large batches.
- Keep one pod per GPU.
- Use postprocess Job for spectra/scoring/recovery.

Next version:

- One Deployment or Job per GPU worker.
- Workers poll an external queue.
- Pods run until queue drained.
- Node-local NVMe is the hot cache.
- S3 is durable input/output.

Preferred queue choices:

```text
SQS + S3 task payloads
DynamoDB task table + S3 payloads
Kubernetes indexed Jobs for coarse batches
```

Do not use a shared filesystem as the live queue for EKS unless benchmarked.

## Acceptance Criteria

A service-mode prototype is useful only if it proves:

```text
same measurement schema as dispatch worker
same aperture outputs within tolerance
worker startup paid once per worker process
tasks retry cleanly
aggregate status updates without parquet reads
spectra assembly consumes service shards unchanged
injection/recovery contract remains intact
```

Minimum benchmark:

```text
10 frames
50 frames
100 frames
1 GPU and all local GPUs
dispatch subprocess mode vs service mode
```

Report:

```text
end-to-end measurements/sec
worker payload measurements/sec
launch/setup overhead
frames/sec/GPU
cache hit fraction
parquet write rows/sec
profile rows above 5%
```
