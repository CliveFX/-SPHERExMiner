# Implementation Plan

## Phase 0: Constraints

Do not rewrite the existing miner in place. The existing system is the reference
implementation for correctness comparisons.

The first implementation should be standalone and small enough to throw away if
the architecture needs revision.

## Phase 0.5: Function-Level Acceleration Audit

Before optimizing by instinct, audit the current pipeline function by function.
Every meaningful stage in the existing miner should be classified by:

- current wall-time percentage
- call count
- input/output size
- CPU serial, CPU vectorized, GPU kernel, GPU dataframe, or I/O bound
- whether it can be vectorized
- whether it can be moved to CUDA/Warp/CuPy/Numba/RAPIDS
- correctness risk if rewritten
- proposed replacement or reason to leave it alone

Acceleration threshold:

- If a function/stage is more than 5% of total pipeline wall time, it needs an
  explicit acceleration decision.
- The decision can be "do nothing" only if the measured risk/complexity is
  higher than the likely speedup.
- Anything above 10% should get a benchmark prototype unless it is clearly I/O
  bound or blocked by external storage.

The audit output should be a checked-in table:

```text
docs/current_pipeline_acceleration_audit.md
```

Required columns:

```text
stage
function_or_script
wall_time_pct
wall_time_sec
calls
current_backend
bottleneck_type
vectorizable
gpu_candidate
rapids_candidate
rewrite_candidate
correctness_reference
decision
next_action
```

Examples of likely audit targets:

- FITS discovery and staging
- FITS image/variance/flag reads
- WCS target projection
- catalog footprint queries
- aperture photometry
- PSF photometry
- spectra assembly
- spectrum quality scoring
- narrowband scoring
- injection planning
- FITS-level injection
- parquet writes
- viewer index generation

Astropy-specific rule:

- Astropy remains the correctness reference.
- Any accelerated WCS/coordinate/proper-motion replacement must report
  residuals against Astropy and fail closed if residuals exceed tolerance.

## Phase 1: Local Frame Benchmark

Goal: process a fixed small frame set with aperture photometry and compare the
result to the current miner.

Inputs:

- 10-100 SPHEREx FITS frames.
- Gaia/2MASS catalog tiles already available locally.
- Local SSD cache cap: 500 GB.

Outputs:

```text
runs/<benchmark_id>/
  manifest/
  local_cache/
  measurement_shards/
  spectra/
  candidates/
  perf_summary.json
  correctness_summary.json
```

Tasks:

1. Implement `build-manifest`. Done for local FITS discovery/header WCS.
2. Implement local cache manager.
3. Implement frame footprint extraction. Done via Astropy WCS footprint.
4. Implement catalog candidate query per frame. Done for Gaia/2MASS parquet
   with HPX-pruned DuckDB reads.
5. Implement CPU-vectorized pixel projection first. Done via array Astropy WCS
   with `in_frame` masking.
6. Implement GPU aperture photometry for all in-frame targets.
7. Write measurement shards.
8. Assemble spectra from shards.
9. Compare overlapping target spectra to current miner.

Exit criteria:

- Reproduces aperture measurements within agreed tolerance.
- Emits complete measurement provenance.
- Produces `perf_summary.json`.

## Phase 2: GPU Occupancy Pass

Goal: keep GPUs fed.

Tasks:

1. Batch multiple frames per GPU residency window.
2. Use one worker process per GPU.
3. Add prefetch/stage queue for local SSD.
4. Add async write queue for output shards.
5. Measure GPU occupancy and I/O wait.

Metrics:

- frames/sec/GPU
- measurements/sec/GPU
- GPU occupancy
- local SSD read/write GB/sec
- parquet rows/sec
- CPU time in WCS/projection

## Phase 3: PSF Photometry

Goal: add PSF photometry without breaking aperture results.

Tasks:

1. Port existing GPU PSF kernel-bank/grid approach.
2. Keep PSF kernels resident on GPU.
3. Emit aperture and PSF rows from the same frame pass.
4. Compare to current PSF tester and known brown dwarf examples.

## Phase 4: Virtual Injection

Goal: generate injection/recovery training data without copying FITS files.

Tasks:

1. Define injection truth table keyed by target/frame/wavelength.
2. Apply synthetic source inside frame worker.
3. Emit injection metadata in measurement rows.
4. Verify against physical FITS injection on a small case.

## Phase 5: Campaign Products

Goal: produce viewer-compatible products at scale.

Tasks:

1. Build spectra shards.
2. Build candidate indexes.
3. Build detector/wavelength systematics summaries.
4. Build injection recovery summaries.
5. Build ML training datasets.

## Phase 6: EKS Deployment

Goal: run all-sky frame groups across many GPU nodes.

Tasks:

1. Containerize worker.
2. Store manifests and outputs in S3.
3. Add Kubernetes Job templates.
4. Add retry/status model.
5. Add cost/performance reporting.
