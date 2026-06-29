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
  --batch-table-assembly \
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
  --batch-table-assembly \
  --materialize-worker-inputs \
  --local-cache-dir /tmp/luxquarry_stage_smoke \
  --limit-frames 10

# After the generated shell finishes, collect worker summaries into one
# aggregate summary and one measurement shard manifest.
.venv/bin/luxquarry-allsky collect-dispatch-run \
  --plan runs/dispatch_smoke10/dispatch_plan.json

# Or write Kubernetes Job manifests from the same dispatch plan. This is the
# cloud/EKS handoff artifact: one pod per materialized GPU worker.
.venv/bin/luxquarry-allsky write-k8s-jobs \
  --plan runs/dispatch_smoke10/dispatch_plan.json \
  --out-dir runs/dispatch_smoke10/k8s \
  --image luxquarry-allsky:local \
  --namespace luxquarry \
  --container-executable luxquarry-allsky \
  --working-dir /workspace/luxquarry_allsky_engine \
  --pvc-name luxquarry-data \
  --mount-path /workspace

# Assemble target-ordered ragged spectra from the collected shard manifest.
.venv/bin/luxquarry-allsky assemble-spectra \
  --shard-manifest runs/dispatch_smoke10/measurement_shard_manifest.parquet \
  --out-dir runs/dispatch_smoke10/spectra \
  --run-id dispatch_smoke10 \
  --device cuda:0

# Write a campaign-level stage contract. This is the guardrail that keeps
# baseline, injected, scoring, recovery, and viewer-index products tied together.
.venv/bin/luxquarry-allsky write-campaign-contract \
  --campaign-id dispatch_smoke10_contract \
  --out runs/dispatch_smoke10/campaign_contract.json \
  --baseline-plan runs/dispatch_smoke10/dispatch_plan.json \
  --baseline-spectra-dir runs/dispatch_smoke10/spectra
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

`collect-dispatch-run` turns independent worker outputs back into one
run-level contract:

```text
aggregate_summary.json
measurement_shard_manifest.parquet
```

The aggregate summary reports missing/incomplete workers, failed frames, missing
shards, total rows, ok rows, and max worker wall time. The shard manifest is the
input list downstream spectra assembly should consume.

`write-k8s-jobs` converts that same dispatch plan into dependency-free
Kubernetes Job YAML. The current generator is intentionally simple: it emits one
Job per worker, requests `nvidia.com/gpu`, preserves the worker argument vector,
and optionally attaches a PVC plus environment variables. It is a deployment
artifact generator, not a different scheduler.

`assemble-spectra` reads the measurement shard manifest with cuDF, writes a
target/wavelength-sorted ragged spectra table, and writes one target summary
row per object:

```text
<run_id>.spectra_measurements.parquet
<run_id>.target_summary.parquet
assemble_summary.json
```

For large dispatches, add `--materialize-worker-inputs` to the plan command.
That writes per-worker input slices before launch:

```text
worker_inputs/w0000/frame_manifest.parquet
worker_inputs/w0000/projected_targets.parquet
...
```

Workers then run with `worker_index=0` and `worker_count=1` against their own
small manifest and projected-target parquet. This avoids every worker scanning
the full target table at startup.

Workers can also stage FITS inputs onto local SSD/NVMe with
`--local-cache-dir`. On a 10-frame smoke, a warm staged cache reduced the
persistent worker from about 1.15 sec to about 0.94 sec while preserving the
same CPU-baseline correctness. First-touch staging is mainly a way to populate
the local cache; repeated passes and long worker queues are where it should pay.

`--async-shard-writes` queues cuDF parquet writes on a background thread and
waits for them before completion. On the same tiny smoke it is roughly tied with
the warm staged path, but it removes inline shard write blocking from the frame
loop and is the better shape for long queues.

`--batch-table-assembly` keeps per-frame kernel outputs as CuPy device columns
and builds the cuDF measurement table once per shard batch. On the 10-frame
smoke this reduced wall time to about 0.88 sec with identical GPU output.

## EKS Target

The cloud version should run one independent frame-group worker per GPU/pod.

- S3 for durable input/output.
- Instance NVMe for hot cache.
- No shared filesystem in the hot loop unless benchmarked.
- Kubernetes Jobs or queue-fed workers for frame groups.
- Post-processing Dask/RAPIDS jobs for spectra assembly and candidate scoring.

## Campaign Completion Contract

A survey run is not complete when baseline aperture shards exist. The complete
campaign contract is:

```text
baseline measurement shards
baseline spectra assembly
baseline blind scoring and quality-gated scoring
injected-frame or injected-measurement variant using the same target set
injected spectra assembly
injected blind scoring and quality-gated scoring
truth-target recovery summary
candidate and false-positive review indexes
```

The current all-sky engine has the baseline GPU photometry, dispatch,
collection, and spectra assembly pieces. Injection, recovery, and the
narrowband candidate scorer must be promoted into this frame-first contract
before using the engine for science-grade all-sky mining.

`write-campaign-contract` records that status in a machine-readable JSON file.
It marks stages as `complete`, `missing`, or `blocked` based on expected
artifacts. On the current smoke run, baseline dispatch and baseline spectra are
complete, while injected dispatch/recovery are correctly blocked because no
injection truth table or injected plan has been supplied.

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
      campaign.py
      manifest.py
      catalog.py
      projection.py
      kubernetes.py
  benchmarks/
    # fixed benchmark manifests and scripts
  k8s/
    # future Kubernetes manifests/Helm templates
```
