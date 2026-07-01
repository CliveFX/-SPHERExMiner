# Architecture

## Current Reference System

The existing miner is target/campaign oriented:

```text
target list
  -> select frames per target/campaign
  -> run photometry
  -> assemble spectra
  -> score/recover/inject
```

That system is valuable because it established correctness, calibration choices,
viewer workflows, injection mechanics, and narrowband scoring. It is not the
right shape for whole-sky scale.

## Next-Generation System

LuxQuarry All-Sky Engine inverts the workflow:

```text
frame group
  -> frame footprint
  -> catalog targets inside footprint
  -> GPU photometry for all targets
  -> audit or survey output products
  -> spectra/candidate assembly as downstream GPU jobs
```

The key product is not a run folder full of target spectra. In audit mode the
key product is an append-only measurement table. In survey mode the key product
is a reduced spectra/candidate product set with raw measurement retention only
for injections, candidates, holdouts, debug samples, and explicitly requested
targets. See `survey_output_contract.md`.

The measured worker-only smoke shows small dispatches are dominated by worker
process startup, not GPU payload. The next performance shape is a long-lived
GPU worker service that polls frame-batch tasks and pays CUDA/RAPIDS startup
once per worker lifetime. See `worker_service_design.md`.

## Work Units

A work unit is a frame group.

Frame groups should be sized by:

- GPU memory.
- local SSD cache budget.
- FITS read throughput.
- catalog query overhead.
- output shard size.

Each work unit must be independent and retryable. No worker should need to
coordinate with another worker while processing frames.

## Economic Target

The v2 economic target is:

```text
$5k of cloud GPU compute should buy the accessible-sky survey for Gaia G ~= 8-14
plus all 2MASS point-source catalog rows present in the processed local 2MASS
PSC cache.
```

This target is intentionally narrower than all Gaia, but broader than a toy
campaign. It is the cost gate that should drive architecture decisions. The
planner must report Gaia source count, 2MASS source count, deduplicated target
count, frame count, estimated measurement count, estimated output bytes, and
estimated cloud cost before a large run. For `combined` survey economics, Gaia
is magnitude-filtered and 2MASS is not; any per-frame source cap must happen
before this estimate and must be called out as an incomplete all-2MASS plan.

## Measurement Schema

Every emitted measurement row must be self-describing enough to rebuild spectra
and audit candidates later.

Minimum columns:

```text
campaign_id
frame_group_id
measurement_id
target_id
catalog
source_id
hpx
ra_reference_deg
dec_reference_deg
reference_epoch_yr
pmra_masyr
pmdec_masyr
parallax_mas
ra_epoch_deg
dec_epoch_deg
image_id
fits_path_or_key
release
processing_version
detector
observation_id
obs_mid_time
cwave_um
cband_um
wavelength_source
wavelength_calibration_file
x_pix
y_pix
edge_distance_pix
aperture_flux_uJy
aperture_flux_unc_uJy
psf_flux_uJy
psf_flux_unc_uJy
photometry_backend
psf_photometry_backend
flags_summary
fatal_flag_present
input_file_checksum
calibration_version
```

## Stages

### 1. Manifest Builder

Builds a frame-group manifest from available SPHEREx products.

Output:

```text
frame_groups.parquet
```

Columns:

```text
frame_group_id
image_ids
fits_keys
release
detectors
estimated_bytes
status
```

### 2. Frame Worker

Processes one frame group.

Steps:

1. Stage FITS files and needed catalog tiles to local SSD.
2. Read FITS headers and image/variance/flag planes.
3. Compute frame footprint.
4. Query catalog targets in footprint.
5. Project targets to detector pixels.
6. Run GPU photometry.
7. Write measurement parquet shards.
8. Write status/performance JSON.

### 3. Spectra Assembler

Reads measurement shards and groups by `target_id`.

Outputs:

```text
target_spectra.parquet
target_summary.parquet
spectrum_quality.parquet
```

At all-sky scale this should run as:

```text
measurement shards
  -> GPU target-hash shuffle
  -> partitioned measurement buckets
  -> independent spectra reducers
```

The target-hash shuffle is intentionally separate from the frame workers. Frame
workers stay append-only and retryable; reducer jobs can be fanned out across
nodes and GPUs without asking every reducer to reread every original shard.

### 4. Candidate Scorer

Runs narrowband scoring and optional ML filters.

Outputs:

```text
narrowband_candidates.parquet
narrowband_line_scores.parquet
narrowband_detector_summary.json
candidate_index.parquet
```

### 5. Injection Mode

Two injection modes should exist:

- Physical FITS injection: correctness/reference, slower.
- Virtual injection: scale benchmark and ML data generation.

Virtual injection applies the synthetic source in the frame-worker measurement
path without rewriting FITS files.

## RAPIDS Use

Use RAPIDS where it fits naturally:

- cuDF: measurement tables, filtering, joins, grouping.
- Dask-cuDF: multi-GPU spectra assembly and large campaign post-processing.
- cuML: embeddings, PCA/UMAP/classifiers.
- RMM: GPU memory pool.
- KvikIO: GPU-friendly local parquet/Arrow I/O where practical.
- cuSpatial: candidate for footprint filtering if it proves clean.

Custom CUDA/Warp kernels remain appropriate for aperture/PSF photometry.

## Serving and Viewer Indexes

The compute path should not write directly to an interactive database. The
durable source of truth remains immutable parquet/JSON shards. A separate
serving/index stage can load selected products into ClickHouse or another
columnar serving store for dashboards and drilldown.

The scalable handoff into that serving layer is:

```text
measurement_shard_manifest.parquet
  -> measurement partition manifest
  -> reducer_outputs.parquet
  -> candidate_scorer_outputs.parquet
  -> viewer/ClickHouse index
```

`candidate_scorer_outputs.parquet` records candidate parquet paths, candidate
counts, target counts, and scorer timing by reducer partition. The viewer should
read from that index path, not crawl arbitrary run directories on each page
load.

Likely ClickHouse tables:

```text
campaign_runs
frame_status
measurement_partitions
target_summary
spectrum_quality
narrowband_candidates
candidate_line_scores
injection_truth
recovery_summary
```

This keeps viewer latency independent of worker throughput. It also lets us
rebuild the viewer index from parquet if the serving schema changes.

## GPU Target Class

The current hot path is not obviously H100/B300-only work. The core operations
are:

- FITS/object staging.
- WCS/projection and target selection.
- aperture/PSF CUDA kernels.
- cuDF filtering/hash/groupby/sort.
- parquet shard IO.

V100-class datacenter GPUs are credible for worker and reducer nodes, especially
when many nodes are available. Newer GPUs should help, but the architecture must
not rely on premium training accelerators to be economical. Benchmark decisions
should be driven by measurements per dollar, object-store bandwidth, local SSD
bandwidth, and parquet/groupby throughput, not peak tensor performance.
