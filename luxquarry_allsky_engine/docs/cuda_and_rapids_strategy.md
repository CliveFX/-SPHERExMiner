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

Every RAPIDS/CUDA decision must show up in `perf_summary.json`.

Required metrics:

- frame staging time
- catalog query time
- CPU WCS/projection time
- GPU photometry kernel time
- cuDF table construction time
- cuDF filter/group/sort time
- parquet write time
- GPU memory peak
- local SSD read/write bytes
- measurements/sec/GPU

If a GPU library adds complexity without improving one of these measurements,
remove it from the hot path.
