# CUDA and RAPIDS Strategy

## Principle

Use GPU-native libraries as aggressively as practical, but do not force every
operation through a database or dataframe abstraction.

The split should be:

- RAPIDS for table-scale data movement, filtering, joins, groupby, sorting,
  aggregation, parquet I/O, and ML-style analytics.
- Custom CUDA/Warp kernels for SPHEREx-specific photometry, PSF fitting,
  virtual injection, and narrowband response scoring.

## Why

The current target-first miner proved the science workflow but underfeeds the
GPU. The next engine should keep data resident on the GPU longer and avoid
round-tripping through CPU/Pandas/PyArrow for every stage.

The frame-first engine should treat each frame group as a GPU dataflow problem:

```text
stage FITS/catalog locally
  -> upload frame/image/catalog arrays
  -> run photometry kernels
  -> keep rows in cuDF where useful
  -> filter/group/sort/write compact shards
```

## RAPIDS Components

### cuDF

Use for:

- measurement tables emitted by photometry kernels
- filtering invalid/fatal-flag rows
- joining target metadata onto measurements
- computing derived quality columns
- grouping measurements by target
- sorting spectra by wavelength/time/detector
- writing parquet shards

### Dask-cuDF

Use for:

- multi-GPU spectra assembly
- larger-than-memory campaign post-processing
- whole-campaign candidate summaries
- EKS-scale aggregation after frame jobs complete

### cuML

Use for:

- PCA/UMAP/embedding experiments
- spectra clustering
- quality classifiers
- narrowband ML experiments

The deterministic GPU narrowband scorer remains the primary science scanner
until ML models prove clear value.

### RMM

Use for:

- shared GPU memory pool
- reducing allocation overhead
- making cuDF and custom CUDA/Warp stages coexist more predictably

### KvikIO

Evaluate for:

- fast local NVMe parquet/Arrow I/O
- GPUDirect Storage where available

Do not assume KvikIO helps on NAS or S3. Benchmark it on local NVMe first.

### cuSpatial

Evaluate for:

- point-in-polygon or footprint filtering
- bounding candidate catalog rows by frame footprint

Sky geometry and WCS correctness are science-critical. Start with conservative
CPU/astropy validation, then promote only well-tested footprint operations.

## Custom GPU Kernels

Use custom CUDA/Warp for:

- aperture photometry
- PSF photometry
- subpixel PSF grid search
- detector-local texture/pixel sampling
- variance/flag propagation
- virtual fake-source injection
- narrowband response matched filtering

These operations encode SPHEREx-specific math and calibration choices. They
should be directly testable against the current miner.

For PSF photometry, the production work shape is not "one target equals one
kernel." It is:

```text
one frame + all in-frame targets
  -> expand targets to target x local_subpixel_grid candidates
  -> build shifted PSF kernels on GPU
  -> fit every candidate independently on GPU
  -> reduce to one best fit per target
```

That candidate-parallel shape is what can fill A100/H100/B300-class devices.
Small smoke tests with only a few hundred targets per frame are useful for
correctness, but they are not representative occupancy tests.

The 2026-06-29 dense Galactic-plane benchmark confirms this. Sparse smoke
frames delivered only about 0.85M PSF candidates/sec in the device submit/sync
section; dense real fields delivered about 6.8M candidates/sec. On one local
node, 18 dense frames split across three RTX 6000 Ada GPUs showed good worker
balance:

```text
worker_parallel_efficiency: 95.7%
worker_wall_skew_ratio: 1.074
payload speedup vs one GPU: 1.52x
end-to-end speedup vs one GPU: 1.16x
```

The frame-worker partitioning is therefore viable, but the measured limit is
now feeding and draining workers: FITS reads, local/object staging, and parquet
shard writes. Future performance work should treat NVMe/S3 staging and write
combining as GPU-utilization features, not as secondary plumbing.

A follow-up 18-frame, 3-GPU local-cache sweep measured the first I/O mitigation:

```text
uncached payload: 2.915 sec
warm local cache + prefetch=2 payload: 2.177 sec
payload speedup: 1.34x
end-to-end speedup: 1.11x
max payload wait: 0.726 sec -> 0.032 sec
```

This confirms that local object staging and prefetch are part of the GPU
utilization strategy. Once reads are hidden, async shard-write wait remains a
critical-path cost, so output shard batching/combining should be optimized
before spending effort on small PSF kernel micro-optimizations.

The first no-write bound used `--discard-measurement-shards` on the same
warm-cache workload:

```text
warm cache with parquet writes: 2.177 sec payload, 147k measurements/sec
warm cache with writes discarded: 1.659 sec payload, 193k measurements/sec
critical-path output cost: ~0.52 sec
```

This flag is benchmark-only. The production target is not to drop data; it is
to preserve durable parquet outputs while closing that 0.52 sec critical-path
gap through wider shards, write combining, KvikIO/NVMe evaluation, or compact
first-pass output schemas.

The first compact schema test did not close that gap:

```text
full warm payload:    2.233 sec, 143.7k measurements/sec, 103.83 bytes/row
compact warm payload: 2.240 sec, 143.2k measurements/sec, 103.70 bytes/row
```

Compact output still matters for schema discipline, but the current durable
write bottleneck is not solved by dropping a handful of repeated string
columns. Future output work should therefore prioritize:

- fewer and wider shard writes
- writer-process or writer-thread isolation
- local NVMe/KvikIO experiments
- reducer paths that tolerate out-of-order shards
- avoiding tiny-run process launch overhead by using task queues and
  long-lived workers

Per-measurement provenance remains mandatory. Even compact science output must
preserve enough columns to trace each spectral point back to the source FITS
object and calibration choices:

```text
fits_path
frame_group_id
image_id
detector
wavelength_calibration_file
sapm_file
aperture geometry
cwave_um/cband_um
flags_summary
aperture and PSF flux/status columns
```

Ephemeral local staging paths are not mandatory provenance and can be omitted
from compact output.

The first 3-GPU resident task-queue smoke validates the horizontal control
plane, but not a throughput win yet:

```text
one-shot compact dispatch:        ~143k measurements/sec payload
resident queue, 6 x 3-frame task: ~133k measurements/sec payload
resident queue, 3 x 6-frame task: ~133k measurements/sec payload
no-write bound:                   ~193k measurements/sec payload
```

This is useful evidence. The queue architecture is correct for scale-out
because workers can claim frame batches concurrently and reducers can assemble
out-of-order shards. The small-run throughput result says the next optimization
is not another scheduler abstraction; it is amortizing setup over longer-lived
workers and reducing output/table write pressure.

Batch table assembly is now the queue-runner baseline:

```text
resident queue without batch table assembly: ~132.5k measurements/sec
resident queue with batch table assembly:    ~140.2k measurements/sec
one-shot compact dispatch:                   ~143.2k measurements/sec
no-write bound:                              ~193.4k measurements/sec
```

This confirms that per-frame cuDF table construction was a material hot-path
cost. The next target is the combined shard flush path, where deferred table
assembly and parquet write now live together.

Parquet compression is not the dominant remaining output cost on the dense
18-frame/3-GPU smoke:

```text
compact + snappy:         ~140.2k measurements/sec payload
compact + no compression: ~142.1k measurements/sec payload
no-write bound:           ~193.4k measurements/sec payload
```

`snappy` stays the default because normal runs should preserve compact portable
artifacts. `--measurement-parquet-compression none` is useful for isolating
write-path cost, but it does not close the durable-output gap by itself.

## CPU Remains Allowed

Do not force fragile science transforms onto the GPU before correctness is
understood.

CPU is acceptable for:

- FITS header parsing
- astropy WCS reference calculations
- calibration metadata extraction
- manifest construction
- small control-plane decisions
- correctness comparison

The long-term goal is to move repeated, high-volume numeric operations to GPU,
not to eliminate CPU control flow.

## Acceleration Rule

Every current-pipeline function or stage above 5% of total wall time must be
reviewed for:

- vectorized CPU implementation
- CUDA/Warp/CuPy/Numba implementation
- RAPIDS/cuDF/Dask-cuDF implementation
- I/O architecture change

The decision should be evidence-based. A function can stay on CPU only if the
profile shows it is not material, or if the correctness risk outweighs the
expected gain.

Astropy is the reference, not automatically the production hot path. Repeated
array transforms such as WCS projection should first be made vectorized through
Astropy, then considered for compiled/GPU replacement if they remain a
meaningful bottleneck.

## Data Ownership

Parquet shards remain the durable source of truth.

RAPIDS/cuDF is the hot processing layer. A GPU database may be useful later for
interactive analytics, but it should not become the canonical storage layer
until the schema and query workload are proven.

## Local Benchmark Requirements

Every RAPIDS/CUDA decision must show up in a run artifact:

- benchmark sweeps: `perf_summary.json` and `profile_summary.parquet`
- local task queues: `task_queue_perf_report.json` and
  `task_queue_perf_profile.parquet`

Required metrics:

- frame staging time
- catalog query time
- CPU WCS/projection time
- GPU photometry kernel time
- PSF candidate count and candidates/sec
- PSF spline coefficient wall time
- PSF upload wall time
- cuDF table construction time
- cuDF filter/group/sort time
- parquet write time
- GPU memory peak
- local SSD read/write bytes
- measurements/sec/GPU

If a GPU library adds complexity without improving one of these measurements,
remove it from the hot path.

For local task-queue runs, optimize from critical-path worker timings rather
than summed timings. Summed timings show total work across all GPUs; critical
path timings show what actually controls wall clock.
