# LuxQuarry All-Sky Engine

Next-generation frame-first SPHEREx survey miner.

This workspace is intentionally separate from the current target/campaign miner.
The current miner remains the correctness/reference system. LuxQuarry All-Sky
Engine is for throughput: local multi-GPU benchmarking first, then EKS-scale
all-sky mining.

## Core Model

The engine is frame-first, not target-first.

```text
SPHEREx FITS frame/group
  -> compute sky footprint
  -> query local catalog tiles for targets in footprint
  -> run GPU aperture/PSF photometry for all targets in that frame
  -> append measurement parquet shards
  -> assemble spectra later by target_id
  -> score spectra/candidates
```

This avoids repeatedly loading the same FITS files for target-centered depth
runs. Each frame becomes an independent work unit and can be scheduled locally
or in Kubernetes.

## Design Goals

- Load each FITS frame once per work unit.
- Keep hot data on local NVMe, not NAS/S3/EFS.
- Use GPU kernels for photometry and narrowband scoring.
- Use RAPIDS/cuDF/Dask-cuDF for large measurement tables where it helps.
- Emit append-only measurement shards with complete provenance.
- Make every frame group retryable and independent.
- Preserve compatibility with current viewer/scoring concepts through derived
  spectra/candidate products.

## Local Prototype Target

Prototype on the local workstation before any cloud deployment.

- Up to 500 GB local SSD cache.
- 1-3 local GPUs.
- 100-frame benchmark first.
- Aperture photometry first.
- PSF photometry second.
- Virtual injection after aperture correctness is proven.

The first local benchmark must emit:

```text
measurement_shards/
spectra/
candidates/
perf_summary.json
correctness_summary.json
```

## Current CLI

The prototype CLI is installed inside the fresh local virtualenv:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky --help
```

Implemented stages:

```bash
# Probe Python/CUDA/RAPIDS availability.
.venv/bin/luxquarry-allsky env-probe --out runs/env_probe.json

# Build a FITS frame manifest with WCS footprints.
.venv/bin/luxquarry-allsky build-manifest \
  --input-root /mnt/niroseti/spherex_cache/raw/qr2/level2 \
  --out runs/manifest_smoke_v2/frame_manifest.parquet \
  --campaign-id manifest_smoke_v2 \
  --limit 10

# Query local Gaia/2MASS parquet tiles for targets near each frame footprint.
.venv/bin/luxquarry-allsky build-frame-targets \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --out runs/frame_targets_smoke_current/frame_targets.parquet \
  --catalog all \
  --max-sources-per-frame 500 \
  --limit-frames 10

# Project frame target coordinates to detector pixels with vectorized Astropy WCS.
.venv/bin/luxquarry-allsky project-frame-targets \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --frame-targets runs/frame_targets_smoke_current/frame_targets.parquet \
  --out runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --limit-frames 10

# Write calibrated uJy aperture measurement rows for in-frame targets.
.venv/bin/luxquarry-allsky run-cpu-aperture \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out runs/measurements_cpu_aperture_smoke10/measurements.parquet \
  --limit-frames 10

# Write calibrated aperture measurement rows with a frame-level GPU kernel
# and cuDF parquet output.
.venv/bin/luxquarry-allsky run-gpu-aperture \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out runs/measurements_gpu_aperture_smoke10_warm/measurements.parquet \
  --limit-frames 10 \
  --device cuda:0

# Run a persistent GPU worker. This keeps RAPIDS/Warp initialized and caches
# detector calibration maps on the GPU for the lifetime of the process.
.venv/bin/luxquarry-allsky run-persistent-gpu-worker \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/persistent_gpu_worker_smoke10 \
  --run-id persistent_smoke10 \
  --limit-frames 10 \
  --device cuda:0 \
  --shard-batch-frames 5 \
  --prefetch-frames 2 \
  --async-shard-writes \
  --local-cache-dir /tmp/luxquarry_stage_smoke

# Write a multi-GPU dispatch plan. The generated shell script launches one
# persistent worker per device; the JSON is the same contract an EKS job
# generator should use.
.venv/bin/luxquarry-allsky plan-gpu-dispatch \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/dispatch_smoke10 \
  --run-id dispatch_smoke10 \
  --plan-out runs/dispatch_smoke10/dispatch_plan.json \
  --devices cuda:0,cuda:1,cuda:2 \
  --shard-batch-frames 5 \
  --prefetch-frames 2 \
  --async-shard-writes \
  --local-cache-dir /tmp/luxquarry_stage_smoke \
  --limit-frames 10
```

The target selection stage is still a prefilter. Photometry should consume only
rows where `in_frame` is true after `project-frame-targets`.

The CPU aperture stage is a correctness/profiling baseline. The GPU aperture
stage pushes the hot science loop into one frame-level Warp kernel:

```text
IMAGE + VARIANCE + FLAGS + SAPM + CWAVE + CBAND + target pixels
  -> calibrated flux, uncertainty, wavelength, flags
  -> cuDF parquet measurement shard
```

The next production path is a persistent monolithic worker that keeps RAPIDS,
Warp kernels, and detector calibration maps resident across many frames instead
of paying setup costs per CLI invocation.

The first persistent worker exists now. It writes independent frame shards and
uses modulo frame partitioning:

```text
frame_ordinal % worker_count == worker_index
```

That makes local multi-GPU dispatch and future Kubernetes dispatch the same
basic model.

Workers can also stage FITS inputs onto local SSD/NVMe with
`--local-cache-dir`. On a 10-frame smoke, a warm staged cache reduced the
persistent worker from about 1.15 sec to about 0.94 sec while preserving the
same CPU-baseline correctness. First-touch staging is mainly a way to populate
the local cache; repeated passes and long worker queues are where it should pay.

`--async-shard-writes` queues cuDF parquet writes on a background thread and
waits for them before completion. On the same tiny smoke it is roughly tied with
the warm staged path, but it removes inline shard write blocking from the frame
loop and is the better shape for long queues.

## EKS Target

The cloud version should run one independent frame-group worker per GPU/pod.

- S3 for durable input/output.
- Instance NVMe for hot cache.
- No shared filesystem in the hot loop unless benchmarked.
- Kubernetes Jobs or queue-fed workers for frame groups.
- Post-processing Dask/RAPIDS jobs for spectra assembly and candidate scoring.

## Repository Layout

```text
luxquarry_allsky_engine/
  README.md
  docs/
    architecture.md
    implementation_plan.md
    benchmark_plan.md
    benchmark_log.md
    cuda_and_rapids_strategy.md
    gpu_worker_dispatch.md
    local_environment.md
    eks_plan.md
  src/
    luxquarry_allsky_engine/
      cli.py
      manifest.py
      catalog.py
      projection.py
  benchmarks/
    # fixed benchmark manifests and scripts
  k8s/
    # future Kubernetes manifests/Helm templates
```
