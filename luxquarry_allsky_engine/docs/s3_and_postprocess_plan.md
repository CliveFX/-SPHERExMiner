# S3 Input and Postprocess Performance Plan

## Why This Exists

The GPU worker service reduces local setup overhead, but all-sky mining still
depends on two other hot paths:

- loading SPHEREx FITS and calibration products from object storage without
  stalling GPUs;
- assembling spectra, campaign products, and candidate summaries from many
  shards without assuming shard order.

This document is the next-step plan after the local lightweight queue service.

## SPHEREx S3 Source

Primary references:

- IRSA SPHEREx mission page:
  `https://irsa.ipac.caltech.edu/Missions/spherex.html`
- IRSA cloud data access:
  `https://irsa.ipac.caltech.edu/cloud_access/`
- AWS Open Data Registry entry:
  `https://registry.opendata.aws/spherex-qr/`

The AWS Open Data entry lists the public bucket and region:

```text
bucket: nasa-irsa-spherex
region: us-east-1
QR2 spectral images prefix: s3://nasa-irsa-spherex/qr2/level2/
QR2 solid angle pixel map prefix: s3://nasa-irsa-spherex/qr2/solid_angle_pixel_map/
QR2 spectral WCS prefix: s3://nasa-irsa-spherex/qr2/spectral_wcs/
access mode: no-sign-request / anonymous
```

The QR2 spectral image FITS products include image, flags, variance, zodiacal
model, exposure-averaged PSF, and wavelength WCS extensions. Those products are
the correct object-storage input for the next miner.

## Current Local S3 Hook

Implemented commands and behavior:

```bash
luxquarry-allsky rewrite-manifest-paths \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --out runs/manifest_smoke_v2_s3/frame_manifest.parquet \
  --strip-prefix /mnt/niroseti/spherex_cache/raw \
  --uri-prefix s3://nasa-irsa-spherex
```

This rewrites local cache paths such as:

```text
/mnt/niroseti/spherex_cache/raw/qr2/level2/...
```

to public object URIs such as:

```text
s3://nasa-irsa-spherex/qr2/level2/...
```

`run-gpu-worker-service` can now process manifest rows whose `path` is an S3 URI
when `--local-cache-dir` is set. The worker stages the FITS object into the
local cache, then reads it with the existing Astropy FITS path. Local paths keep
the old copy/cache behavior.

Current implementation uses anonymous HTTPS derived from the S3 URI, not boto3
or s3fs. That keeps the first S3 path dependency-free; boto3/s3fs/fsspec remain
benchmark candidates for discovery and async prefetch.

The existing worker `--prefetch-frames` option also applies to S3 staging. A
two-frame smoke showed that width 2 can start both S3 downloads concurrently,
while width 1 serializes the first download and only partially overlaps the
second. For S3-backed runs, prefetch width should be sized to the intended
number of concurrent in-flight downloads per worker, bounded by local SSD,
network, and memory pressure.

Worker frame timings now include:

```text
payload_prefetched
payload_wait_wall_sec
staging_wall_sec
staged_bytes
```

Use `payload_wait_wall_sec` as the dashboard tuning metric. `staging_wall_sec`
is measured inside the prefetch thread and may overlap with other payloads;
`payload_wait_wall_sec` is the time the main photometry loop actually blocked
waiting for payload availability.

`collect-task-queue-run` now lifts those frame timings into:

```text
aggregate_summary.json:
  payload_wait_wall_sec
  payload_wait_mean_wall_sec
  staging_wall_sec
  staged_bytes
  fits_read_wall_sec
  kernel_wall_sec
  frame_compute_wall_sec
  frame_table_path

task_queue_tasks.parquet:
  per-task sums for the same metrics

task_queue_frames.parquet:
  one row per completed frame
```

Dashboards should read the aggregate JSON for cards and
`task_queue_frames.parquet` for per-frame plots.

## Input Strategy

Do not stream every photometry read directly from S3 into Astropy. That would
make GPU occupancy depend on S3 latency.

Preferred shape:

```text
controller
  -> builds task queue with S3 object keys, frame metadata, and frame_group_ids

worker startup
  -> initialize CUDA/RAPIDS/Warp once
  -> load local/resident source manifest and target tables
  -> start async object prefetcher

per frame batch
  -> prefetch FITS objects from S3 to node-local NVMe/SSD
  -> verify object size/checksum when available
  -> reuse cached local object if present
  -> process frame batch on GPU
  -> write local parquet shards
  -> copy or sync durable outputs to object storage
```

Implementation options to benchmark:

- `aws s3 cp --no-sign-request` for a simple process-level staging baseline.
- current dependency-free HTTPS staging as the baseline inside the worker.
- `boto3` anonymous client for a Python-managed async prefetcher.
- `s3fs` or `fsspec` for manifest discovery and lightweight metadata, not for
  hot FITS reads until benchmarked.
- KvikIO only after local NVMe staging is proven, because FITS decompression and
  Astropy file access may dominate before GPU direct storage matters.

## Assembly and Campaign Postprocess

Assembly must be shard-order-independent. Workers may finish out of order, retry
tasks, and emit shards from multiple GPUs/nodes.

Rules:

- The shard manifest is authoritative, not filesystem ordering.
- Each measurement row must keep frame, FITS, detector, image ID, wavelength
  calibration, source catalog, target ID, and status/flag provenance.
- Assembly sorts spectra by target ID and wavelength after reading all shards.
- Duplicate shard rows from retries must be removable by deterministic keys:
  `run_id`, `task_id`, `frame_group_id`, `target_id`, `image_id`, and detector.
- The postprocess should tolerate missing/failed tasks when explicitly requested,
  but the default finalization should fail closed.

Performance target:

```text
read many parquet shards with cuDF/Dask-cuDF
normalize schema once
drop duplicate retry rows by deterministic key
hash-partition by catalog + target_id
sort each partition by target_id and wavelength
write spectra_measurements.parquet and target_summary.parquet
score candidates from assembled spectra
write small JSON summaries for dashboards
```

Near-term implementation:

1. Add an assembly benchmark mode that shuffles the shard manifest order and
   verifies identical spectra output. Implemented as
   `luxquarry-allsky validate-assembly-order`.
2. Add duplicate retry-row handling with an explicit deterministic key.
   Implemented as `assemble-spectra --drop-duplicate-measurements` plus
   `luxquarry-allsky validate-assembly-retry-dedup`.
3. Add target-hash partitioned spectra assembly. Implemented as
   `luxquarry-allsky assemble-spectra-partitions`.
4. Add pre-shuffled reducer inputs so each reducer reads only its target bucket.
   Implemented as `luxquarry-allsky partition-measurement-shards`.
5. Add status cards for assembly/scoring throughput:
   rows/sec, shards/sec, bytes/sec, GPU read/compute/write wall time.

Current order-validation command:

```bash
luxquarry-allsky validate-assembly-order \
  --shard-manifest runs/service_queue_smoke_v3/measurement_shard_manifest.parquet \
  --out-dir runs/service_queue_smoke_v3/assembly_order_validation_v2 \
  --run-id service_queue_smoke_v3_order_check_v2 \
  --device cuda:0 \
  --repetitions 3
```

The command assembles the original shard manifest and N shuffled manifests, then
hashes the logical parquet output for spectra measurements and target summaries.
It fails if row counts or hashes differ.

Current retry-dedup validation command:

```bash
luxquarry-allsky validate-assembly-retry-dedup \
  --shard-manifest runs/service_queue_smoke_v3/measurement_shard_manifest.parquet \
  --out-dir runs/service_queue_smoke_v3/retry_dedup_validation \
  --run-id service_queue_smoke_v3_retry_dedup \
  --device cuda:0
```

This command duplicates shard-manifest entries to simulate a retry re-emitting
the same measurements, assembles with `--drop-duplicate-measurements`, and
compares logical spectra/summary hashes against a non-duplicated baseline.

Current partitioned spectra command:

```bash
luxquarry-allsky assemble-spectra-partitions \
  --shard-manifest runs/service_queue_smoke_v3/measurement_shard_manifest.parquet \
  --out-dir runs/service_queue_smoke_v3/partitioned_spectra_smoke \
  --run-id service_queue_smoke_v3_partitioned \
  --device cuda:0 \
  --partition-count 64 \
  --drop-duplicate-measurements
```

To run one reducer partition on a node:

```bash
luxquarry-allsky assemble-spectra-partitions \
  --shard-manifest runs/service_queue_smoke_v3/measurement_shard_manifest.parquet \
  --out-dir runs/service_queue_smoke_v3/partitioned_spectra_part00002 \
  --run-id service_queue_smoke_v3_partitioned \
  --device cuda:0 \
  --partition-count 64 \
  --partition-index 2 \
  --drop-duplicate-measurements
```

The partition key is a cuDF GPU hash over `(catalog, target_id)`. This keeps all
measurements for a target in the same reducer bucket, so spectra assembly can be
fanned out horizontally without relying on shard arrival order. This direct
command is useful for correctness checks, but it still reads the original shard
set before filtering a bucket. For large campaigns, use the pre-shuffled reducer
input command below.

Current pre-shuffled reducer input command:

```bash
luxquarry-allsky partition-measurement-shards \
  --shard-manifest runs/service_queue_smoke_v3/measurement_shard_manifest.parquet \
  --out-dir runs/service_queue_smoke_v3/measurement_partition_smoke \
  --run-id service_queue_smoke_v3_measurement_partitions \
  --device cuda:0 \
  --partition-count 64 \
  --drop-duplicate-measurements
```

This reads the original measurement shard manifest once with cuDF, hashes rows
by `(catalog, target_id)`, writes one measurement parquet per non-empty target
bucket, and writes:

```text
<run_id>.measurement_partition_manifest.parquet
partition_manifests/<run_id>.partNNNNN.measurement_shard_manifest.parquet
measurement_partition_summary.json
```

Each `partition_manifests/*.measurement_shard_manifest.parquet` is directly
consumable by `assemble-spectra`. That gives the EKS reducer shape:

```bash
luxquarry-allsky assemble-spectra \
  --shard-manifest partition_manifests/<run_id>.part00002.measurement_shard_manifest.parquet \
  --out-dir spectra_part00002 \
  --run-id <run_id>.part00002 \
  --device cuda:0 \
  --drop-duplicate-measurements
```

The partitioner skips empty buckets by default to avoid thousands of tiny empty
files. Use `--write-empty-partitions` only when a downstream launcher wants a
manifest file for every possible bucket. The JSON summary includes a bounded
`partition_preview`; the full partition table lives in parquet.

Current local reducer lifecycle:

```bash
luxquarry-allsky write-reducer-plan \
  --partition-manifest runs/service_queue_smoke_v3/measurement_partition_smoke/service_queue_smoke_v3_measurement_partitions.measurement_partition_manifest.parquet \
  --out-dir runs/service_queue_smoke_v3/reducer_lifecycle_smoke \
  --run-id service_queue_smoke_v3_reducer_lifecycle \
  --plan-out runs/service_queue_smoke_v3/reducer_lifecycle_smoke/reducer_plan.json \
  --executable .venv/bin/luxquarry-allsky \
  --devices cuda:0 \
  --max-partitions 2

luxquarry-allsky run-reducer-plan \
  --plan runs/service_queue_smoke_v3/reducer_lifecycle_smoke/reducer_plan.json \
  --max-parallel 1

luxquarry-allsky collect-reducer-plan \
  --plan runs/service_queue_smoke_v3/reducer_lifecycle_smoke/reducer_plan.json \
  --out runs/service_queue_smoke_v3/reducer_lifecycle_smoke/reducer_collect_summary.json
```

`run-reducer-plan --resume` skips reducers whose `assemble_summary.json`,
spectra parquet, and target-summary parquet already exist. `collect-reducer-plan`
writes:

```text
reducer_collect_summary.json
reducer_outputs.parquet
```

`reducer_outputs.parquet` is the handoff table for scorer fanout, viewer-index
loading, or ClickHouse ingestion.

Current local candidate scorer lifecycle:

```bash
luxquarry-allsky write-candidate-fanout-plan \
  --reducer-outputs runs/service_queue_smoke_v3/reducer_lifecycle_smoke/reducer_outputs.parquet \
  --out-dir runs/service_queue_smoke_v3/candidate_fanout_smoke \
  --run-id service_queue_smoke_v3_candidate_fanout \
  --plan-out runs/service_queue_smoke_v3/candidate_fanout_smoke/candidate_fanout_plan.json \
  --executable .venv/bin/luxquarry-allsky \
  --devices cuda:0

luxquarry-allsky run-candidate-fanout-plan \
  --plan runs/service_queue_smoke_v3/candidate_fanout_smoke/candidate_fanout_plan.json \
  --max-parallel 1

luxquarry-allsky collect-candidate-fanout-plan \
  --plan runs/service_queue_smoke_v3/candidate_fanout_smoke/candidate_fanout_plan.json \
  --out runs/service_queue_smoke_v3/candidate_fanout_smoke/candidate_fanout_collect_summary.json
```

This stage writes:

```text
candidate_fanout_collect_summary.json
candidate_scorer_outputs.parquet
```

`candidate_scorer_outputs.parquet` is the handoff to candidate aggregation,
viewer-index loading, or ClickHouse ingestion. It records each candidate parquet
path, candidate counts, target counts, and scorer timing.

For Kubernetes/EKS, replace the local runner with:

```bash
luxquarry-allsky write-k8s-candidate-scorer-jobs \
  --candidate-plan runs/service_queue_smoke_v3/candidate_fanout_smoke/candidate_fanout_plan.json \
  --out-dir runs/service_queue_smoke_v3/candidate_fanout_smoke/k8s \
  --image <ecr-image-uri> \
  --namespace luxquarry \
  --container-executable luxquarry-allsky \
  --working-dir /workspace/luxquarry_allsky_engine \
  --device cuda:0
```

That emits one candidate scorer Job per reducer spectra partition. The collector
command is unchanged after the Jobs finish.

The current measurement dedupe key is:

```text
catalog
target_id
frame_group_id
image_id
detector
```

## EKS Mapping

For AWS scale runs:

```text
S3 bucket/prefix
  -> source FITS and calibration products
  -> durable output shard prefixes

SQS or DynamoDB task table
  -> task leasing
  -> retry counters
  -> lease expiry

EKS worker pods
  -> one process per GPU
  -> node-local NVMe cache
  -> resident GPU worker service
  -> append-only parquet shards

postprocess job
  -> Dask-cuDF cluster or single multi-GPU node
  -> shard manifest -> assembled spectra -> scoring/recovery products
```

The local filesystem queue is only a correctness and performance prototype. The
cloud version should use SQS visibility timeouts or DynamoDB conditional writes,
not shared filesystem renames.

## Open Questions

- How much SPHEREx FITS data should be staged per GPU before photometry begins?
- Should we cache full FITS files, cutouts, or both?
- Is Astropy FITS reading the long-term reader, or do we need a lower-level FITS
  reader optimized for selected extensions?
- How large should the shard size be for Dask-cuDF assembly: by frame count,
  row count, or byte size?
- Should candidate scoring run incrementally per shard, or only after complete
  spectra assembly?
