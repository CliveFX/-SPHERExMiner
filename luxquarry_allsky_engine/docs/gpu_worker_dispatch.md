# GPU Worker Dispatch

LuxQuarry V2 is moving away from one-shot target/campaign commands and toward
persistent frame workers. A worker owns one GPU, initializes RAPIDS/Warp once,
keeps detector calibration maps resident on that GPU, and processes a disjoint
slice of a frame manifest.

## Worker Contract

Each worker receives the same inputs:

```text
frame_manifest.parquet
projected_targets.parquet
worker_index
worker_count
device
output_dir
```

Partitioning is deterministic:

```text
frame_ordinal % worker_count == worker_index
```

This means local processes, Kubernetes Jobs, or many nodes can consume the same
manifest without a live scheduler in the hot path. A failed worker can be
restarted with the same `worker_index` and `worker_count`.

## Hot Path

The persistent worker uses this frame-level GPU path:

```text
read FITS IMAGE/VARIANCE/FLAGS
load/reuse resident SAPM + CWAVE + CBAND maps on GPU
launch one Warp kernel per frame
emit device columns through DLPack to CuPy/cuDF
write independent cuDF parquet shard
atomically rewrite run_status.json
```

The kernel performs:

- image unit conversion to uJy using SAPM
- variance scaling
- CWAVE/CBAND bilinear sampling
- aperture flux
- annulus background
- uncertainty
- aperture flag summary

## Dispatch Plan

Generate a local multi-GPU dispatch plan:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky plan-gpu-dispatch \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/dispatch_smoke10 \
  --run-id dispatch_smoke10 \
  --plan-out runs/dispatch_smoke10/dispatch_plan.json \
  --devices cuda:0,cuda:1,cuda:2 \
  --shard-batch-frames 5 \
  --prefetch-frames 2 \
  --status-interval-frames 25 \
  --limit-frames 10
```

This writes:

```text
dispatch_plan.json
dispatch_plan.sh
```

The shell file launches one persistent worker per listed GPU. The JSON file is
the portable contract for an EKS Job generator.

## Current Benchmark

Smoke dataset:

```text
10 frames
5,000 projected rows
2,770 GPU measurement rows
2,766 ok measurements
```

Single persistent worker, one shard per frame:

```text
total wall: 2.39 sec
kernel: ~0.03 sec/frame
FITS read: ~0.11-0.13 sec/frame
table assembly: ~0.011 sec/frame after first frame
write shard: ~0.010 sec/frame after first frame
```

Single persistent worker, batched shards and FITS prefetch:

```text
command flags: --shard-batch-frames 5 --prefetch-frames 2
total wall: 1.16 sec
shards: 2
rows: 2,770
ok rows: 2,766
matched CPU ok rows: 2,766
flux p95 abs delta vs CPU: 0.025 uJy
```

The worker also supports `--status-interval-frames N` to avoid rewriting the
status JSON every frame on large runs. Use a small number for interactive local
smokes and a larger value for all-sky batch jobs.

Three-worker local dispatch over three GPUs:

```text
workers: 3
frame split: 4 / 3 / 3
shards: 10
rows: 2,770
ok rows: 2,766
failed frames: 0
```

For this tiny smoke test, process startup dominates. The point of the worker
contract is long frame queues where startup and calibration upload are amortized.

## Current Bottleneck

The aperture kernel is no longer the dominant stage for this smoke. FITS reads
are. Prefetching overlaps much of that read time with GPU/table work, but it
does not eliminate storage bandwidth pressure. The next performance work should
focus on:

1. local NVMe staging
2. reducing FITS extension read overhead
3. avoiding per-frame metadata table construction where possible
4. writing larger shard batches tuned to downstream spectra assembly
5. true async write queue so parquet flushes never block the frame loop
6. evaluating KvikIO/local NVMe paths for staged FITS and output shards

RAPIDS should remain the table/shard engine, while Warp/CUDA owns the aperture
kernel.

## Precision

The current hot path uses FP32 image/variance payloads and FP32 Warp kernels.
RTX 6000 Ada cards support FP64 functionally, but their FP64 throughput is far
below FP32. On the smoke benchmark, FP32 GPU photometry matches the CPU baseline
for 2,766 ok measurements with p95 aperture flux delta of about 0.025 uJy.

If we need numerical audits, add a small FP64 audit mode for selected frames and
targets. Do not make FP64 the default mining path unless it changes a science
decision.
