# LuxQuarry All-Sky Container

This image is the deployment unit for persistent GPU workers. It is intentionally
thin: start from a RAPIDS CUDA image, install the local package, and use
`luxquarry-allsky` as the entrypoint.

## Build

Build from the `luxquarry_allsky_engine` directory:

```bash
cd luxquarry_allsky_engine
docker build \
  -f container/Dockerfile \
  -t luxquarry-allsky:local \
  .
```

The default base is:

```text
rapidsai/base:26.06-cuda12-py3.12
```

Override it if the cluster standardizes on a different RAPIDS/CUDA image:

```bash
docker build \
  -f container/Dockerfile \
  --build-arg RAPIDS_IMAGE=rapidsai/base:26.06-cuda12-py3.12 \
  -t luxquarry-allsky:local \
  .
```

RAPIDS Docker tags use CUDA major-version tags such as `cuda12` and `cuda13`.
Pick the tag that matches the cluster driver/toolkit policy.

## Local Smoke

The entrypoint is `luxquarry-allsky`, so a quick GPU visibility check is:

```bash
docker run --rm --gpus all luxquarry-allsky:local env-probe
```

A mounted smoke worker looks like:

```bash
docker run --rm --gpus '"device=0"' \
  -v /home/clive/dev/NIROSETI_SPHEREx:/workspace \
  -v /mnt/niroseti:/mnt/niroseti \
  -w /workspace/luxquarry_allsky_engine \
  luxquarry-allsky:local \
  run-persistent-gpu-worker \
    --manifest runs/dispatch_smoke10_materialized2/worker_inputs/w0000/frame_manifest.parquet \
    --projected-targets runs/dispatch_smoke10_materialized2/worker_inputs/w0000/projected_targets.parquet \
    --out-dir runs/container_worker_smoke_w0000 \
    --run-id container_worker_smoke_w0000 \
    --device cuda:0 \
    --worker-index 0 \
    --worker-count 1 \
    --status-path runs/container_worker_smoke_w0000/run_status.json \
    --shard-batch-frames 5 \
    --prefetch-frames 2 \
    --status-interval-frames 5 \
    --local-cache-dir /tmp/luxquarry_stage \
    --async-shard-writes \
    --batch-table-assembly
```

## Kubernetes Contract

`write-k8s-jobs` generates Jobs that assume this image contract:

- command/entrypoint: `luxquarry-allsky`
- working directory: `/workspace/luxquarry_allsky_engine`
- input/output volume mounted at `/workspace`
- one GPU requested per worker pod by default
- worker args copied directly from the dispatch plan

After worker Jobs complete, generate the postprocess Job:

```bash
luxquarry-allsky write-k8s-postprocess-job \
  --plan runs/<run_id>/dispatch_plan.json \
  --out-dir runs/<run_id>/k8s \
  --image luxquarry-allsky:local \
  --namespace luxquarry \
  --container-executable luxquarry-allsky \
  --working-dir /workspace/luxquarry_allsky_engine \
  --pvc-name luxquarry-data \
  --mount-path /workspace \
  --campaign-id <run_id>_finalize
```

That Job runs `finalize-dispatch-run`, which collects worker summaries,
assembles spectra with cuDF, and writes the campaign contract.

For EKS, replace the local image tag with the registry image, for example:

```bash
luxquarry-allsky write-k8s-jobs \
  --plan runs/<run_id>/dispatch_plan.json \
  --out-dir runs/<run_id>/k8s \
  --image <account>.dkr.ecr.<region>.amazonaws.com/luxquarry-allsky:<tag> \
  --namespace luxquarry \
  --container-executable luxquarry-allsky \
  --working-dir /workspace/luxquarry_allsky_engine \
  --pvc-name luxquarry-data \
  --mount-path /workspace
```

The image does not solve durable cloud storage by itself. The current smoke
contract uses a mounted workspace/PVC. S3 staging and object-output paths remain
future work.
