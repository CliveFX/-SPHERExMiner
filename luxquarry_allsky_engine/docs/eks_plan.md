# EKS Plan

## Purpose

EKS is for scale mining after the local frame-first engine proves correctness
and performance. The goal is all-sky processing across tens of GPU nodes.

## Storage

Use S3 for durable storage:

```text
s3://<bucket>/luxquarry/<campaign_id>/manifest/
s3://<bucket>/luxquarry/<campaign_id>/measurements/
s3://<bucket>/luxquarry/<campaign_id>/spectra/
s3://<bucket>/luxquarry/<campaign_id>/candidates/
s3://<bucket>/luxquarry/<campaign_id>/status/
s3://<bucket>/luxquarry/<campaign_id>/logs/
```

Use instance NVMe for hot cache inside each pod.

Avoid shared filesystems in the hot loop until benchmarked.

## Work Unit

The first generated Kubernetes shape is one Job per materialized GPU worker.
Each Job receives a pre-sliced frame manifest and projected-target parquet. That
is coarser than one frame group per Job and avoids tiny pod startup overhead.
Later queue-fed workers can reduce the unit back to individual frame groups if
retry granularity matters more than startup cost.

Local worker-only benchmarking has now measured startup/setup overhead at about
2.8 sec on a two-frame dispatch, while the actual worker payload is about
0.40-0.44 sec. For EKS-scale mining, the target is therefore long-lived GPU
workers polling frame-batch tasks, not one tiny pod/job per small frame group.
See `worker_service_design.md` for the queue and lease contract.

Each job:

1. Stages FITS and reads pre-built target slices.
2. Runs frame-first GPU photometry.
3. Writes measurement shards.
4. Writes status JSON.
5. Exits cleanly.

## Pod Shape

Initial target:

```text
1 GPU per pod
8-16 CPU cores
64-128 GB RAM
local NVMe scratch
```

For multi-GPU nodes, prefer one pod per GPU until profiling proves a multi-GPU
pod is simpler/faster.

## Status Model

Status should be object-based and append-friendly:

```text
status/frame_group_id.json
logs/frame_group_id.log
```

Fields:

```text
pending
staging
running
writing
complete
failed
retry_count
```

## Post-Processing

After frame jobs complete:

1. Run Dask/RAPIDS spectra assembly over measurement shards.
2. Run quality scoring.
3. Run narrowband candidate scoring.
4. Run injected-frame or injected-measurement jobs for the same target set.
5. Run injected spectra assembly.
6. Run injected raw/quality candidate scoring.
7. Run truth-target recovery and false-positive accounting.
8. Build viewer indexes.

Baseline photometry alone is not a science-complete campaign. The EKS contract
must preserve these phases so cloud scale does not drop injection/recovery:

```text
baseline raw spectra
baseline candidate scan
injected raw spectra
injected candidate scan
truth-target recovery
candidate/false-positive review products
```

## Failure Model

Frame-group jobs must be idempotent:

- Write to a temporary S3 prefix.
- On success, write final `_SUCCESS` marker or status JSON.
- Failed jobs can be rerun without corrupting global products.

## Cost Model Inputs

The local benchmark must estimate:

- frames/sec/node
- measurements/sec/node
- output bytes/frame
- cache bytes/frame group
- S3 read/write bytes
- pod runtime
- retry rate

These numbers determine the 50-node all-sky plan.

## Current Local Handoff

The local prototype can now build a worker image and write Kubernetes Job YAML
from a dispatch plan:

```bash
cd luxquarry_allsky_engine
docker build -f container/Dockerfile -t luxquarry-allsky:local .

.venv/bin/luxquarry-allsky write-k8s-jobs \
  --plan runs/dispatch_smoke10_materialized2/dispatch_plan.json \
  --out-dir runs/dispatch_smoke10_materialized2/k8s \
  --image luxquarry-allsky:local \
  --namespace luxquarry \
  --container-executable luxquarry-allsky \
  --working-dir /workspace/luxquarry_allsky_engine \
  --pvc-name luxquarry-data \
  --mount-path /workspace

.venv/bin/luxquarry-allsky write-k8s-postprocess-job \
  --plan runs/dispatch_smoke10_materialized2/dispatch_plan.json \
  --out-dir runs/dispatch_smoke10_materialized2/k8s \
  --image luxquarry-allsky:local \
  --namespace luxquarry \
  --container-executable luxquarry-allsky \
  --working-dir /workspace/luxquarry_allsky_engine \
  --pvc-name luxquarry-data \
  --mount-path /workspace \
  --campaign-id dispatch_smoke10_materialized2_finalize
```

Validated on the 10-frame materialized smoke plan:

```text
jobs: 3
gpu request per job: 1
worker args: run-persistent-gpu-worker
materialized worker runtime args: --worker-index 0 --worker-count 1
postprocess jobs: 1
postprocess args: finalize-dispatch-run
```

The image contract is:

```text
entrypoint: luxquarry-allsky
working directory: /workspace/luxquarry_allsky_engine
input/output mount: /workspace
GPU request: one nvidia.com/gpu per Job by default
```

For EKS, push the same image to ECR and pass the ECR image URI to
`write-k8s-jobs --image`.

The postprocess Job should run after worker Jobs complete. It executes
`finalize-dispatch-run`, which collects worker outputs, assembles spectra with
cuDF, and writes the campaign contract. The command refuses incomplete dispatch
runs unless `--allow-incomplete` is passed.

## Reducer Fanout Handoff

For larger campaigns, do not use a single postprocess Job to assemble all
spectra. Use the two-stage reducer path:

```bash
.venv/bin/luxquarry-allsky partition-measurement-shards \
  --shard-manifest runs/<run_id>/measurement_shard_manifest.parquet \
  --out-dir runs/<run_id>/measurement_partitions \
  --run-id <run_id>_measurement_partitions \
  --device cuda:0 \
  --partition-count 1024 \
  --drop-duplicate-measurements

.venv/bin/luxquarry-allsky write-reducer-plan \
  --partition-manifest runs/<run_id>/measurement_partitions/<run_id>_measurement_partitions.measurement_partition_manifest.parquet \
  --out-dir runs/<run_id>/reducers \
  --run-id <run_id>_reducers \
  --plan-out runs/<run_id>/reducers/reducer_plan.json \
  --devices cuda:0

.venv/bin/luxquarry-allsky write-k8s-reducer-jobs \
  --reducer-plan runs/<run_id>/reducers/reducer_plan.json \
  --out-dir runs/<run_id>/reducers/k8s \
  --image <ecr-image-uri> \
  --namespace luxquarry \
  --container-executable luxquarry-allsky \
  --working-dir /workspace/luxquarry_allsky_engine \
  --device cuda:0 \
  --pvc-name luxquarry-data \
  --mount-path /workspace
```

`write-reducer-plan` is also useful locally because it writes
`reducer_plan.sh`. The plan distributes reducer tasks across the listed devices
round-robin. For Kubernetes one-GPU pods, `write-k8s-reducer-jobs --device
cuda:0` rewrites each reducer command to use `cuda:0` inside the container, even
if the local plan was built with `cuda:0,cuda:1,...`.

The all-sky postprocess sequence should become:

```text
worker Jobs
  -> collect measurement shard manifest
  -> GPU measurement partition shuffle
  -> reducer fanout Jobs
  -> scorer fanout / candidate aggregation
  -> viewer-index load
```

ClickHouse or another serving database belongs in the final viewer-index load,
not in the worker or reducer hot path.
