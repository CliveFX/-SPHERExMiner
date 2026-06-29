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
  -> append measurement rows
  -> spectra assembly as a downstream groupby/sort job
```

The key product is not a run folder full of target spectra. The key product is
an append-only measurement table.

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

