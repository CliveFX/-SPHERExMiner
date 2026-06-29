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
    local_environment.md
    eks_plan.md
  src/
    # future Python package / CLI
  benchmarks/
    # fixed benchmark manifests and scripts
  k8s/
    # future Kubernetes manifests/Helm templates
```
