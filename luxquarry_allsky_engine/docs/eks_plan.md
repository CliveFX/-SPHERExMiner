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

One Kubernetes Job should process one frame group.

Each job:

1. Downloads/stages FITS and catalog tiles.
2. Runs frame-first GPU photometry.
3. Writes measurement shards to S3.
4. Writes status JSON to S3.
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
4. Run injection recovery if applicable.
5. Build viewer indexes.

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

