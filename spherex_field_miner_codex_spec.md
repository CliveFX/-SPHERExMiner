# SPHEREx Field-Based Forced Photometry Miner
## Codex Build Specification v0.1

## 0. Executive Summary

Build a scalable, field-first SPHEREx data-mining system to search for narrowband spectral excesses consistent with possible interstellar laser beacons.

The pipeline must not be target-first. The fundamental work unit is one SPHEREx Level 2 spectral image / detector / processing version. For each field, the system selects all eligible Gaia/manual targets inside the footprint, performs forced photometry on every selected target, writes append-only measurement shards, and later assembles spectra and searches for narrowband residuals.

The core architecture is:

```text
SPHEREx field manifest
    -> distributed field-job queue
    -> worker claims one image/detector job
    -> worker caches/loads full parent MEF and calibration products
    -> worker queries local Gaia/manual target index inside field footprint
    -> worker applies target filters
    -> worker propagates target positions to observation epoch
    -> worker performs aperture + PSF forced photometry in batch
    -> worker writes measurement Parquet shard + QA artifacts
    -> worker registers shard/provenance
    -> later spectrum assembly and narrowband excess search
```

This is not a wrapper around SPExPI. SPExPI is a reference implementation and validation source, not the production architecture.

The first useful smoke test is field-based: process a whole SPHEREx parent field containing the SIMP brown dwarf target used by SPExPI, and verify that SIMP appears as just another manual target among many field-selected Gaia targets.

The final production system must run in containers and be deployable locally, on EC2/AWS Batch, and eventually on Kubernetes/EKS for full-archive parallel mining.

---

## 1. Scientific Purpose

The purpose of this project is to mine SPHEREx data for candidate narrowband spectral excesses that could be consistent with artificial interstellar laser beacons.

The first objective is not to prove a technosignature detection. The first objective is to build a robust, scalable, reproducible measurement engine that can:

1. Cache and process SPHEREx Level 2 fields.
2. Select plausible astrophysical targets from a local Gaia-derived spatial index.
3. Run forced photometry on every selected target in each field.
4. Store native per-field/per-detector/per-target measurements with full provenance.
5. Assemble native SPHEREx measurement streams into per-target spectra.
6. Search those spectra for statistically significant unresolved/narrowband excesses.
7. Generate signal-candidate packets for human review and follow-up.

SPHEREx has low-to-moderate spectral resolution, so an intrinsically narrow laser line will not appear as an infinitely narrow line. It will appear as excess energy in one or a small number of SPHEREx wavelength samples after convolution with the SPHEREx response. Therefore, the detection problem is:

```text
Find unresolved line-like excesses inconsistent with a smooth astrophysical SED,
inconsistent with instrument/data artifacts, and preferentially repeatable across
independent observations at the same astrophysical wavelength.
```

---

## 2. Authoritative References

The implementation must use the official SPHEREx documentation and the SPExPI source as references.

### 2.1 SPHEREx Explanatory Supplement

Required file:

```text
https://irsa.ipac.caltech.edu/data/SPHEREx/docs/SPHEREx_Expsupp_QR.pdf
```

Store locally:

```text
/spherex_cache/external/docs/SPHEREx_Expsupp_QR.pdf
```

Use this as the authoritative reference for:

```text
MEF HDU structure
IMAGE units
FLAGS interpretation
VARIANCE interpretation
ZODI interpretation
PSF extension/cube behavior
WCS-WAVE behavior
Spectral WCS calibration products
QR caveats and release notes
```

Do not infer HDU meanings from examples alone.

### 2.2 Official IRSA SPHEREx archive documentation

Useful pages:

```text
https://caltech-ipac.github.io/spherex-archive-documentation/spherex-data-products/
https://caltech-ipac.github.io/spherex-archive-documentation/spherex-data-access/
https://irsa.ipac.caltech.edu/data/SPHEREx/docs/overview_qr.html
https://caltech-ipac.github.io/irsa-tutorials/spherex-intro/
```

Important data-product assumptions to verify in code/docs:

```text
Level 2 Spectral Image files are multi-extension FITS files.
IMAGE contains calibrated spectral-image surface brightness, typically MJy/sr.
FLAGS contains per-pixel status/processing flags.
VARIANCE contains per-pixel variance estimates.
ZODI contains the zodiacal model and is not subtracted from IMAGE.
PSF contains exposure-averaged PSF information.
WCS-WAVE maps detector/spectral-image pixel coordinates to wavelength/bandwidth.
For science-grade wavelength/bandwidth, prefer dedicated Spectral WCS calibration products when available.
```

### 2.3 AWS Open Data / S3 access

SPHEREx QR2 products are available as public AWS open data.

Important source:

```text
https://registry.opendata.aws/spherex-qr/
```

The QR2 Level 2 S3 path listed by the AWS Open Data registry is:

```text
s3://nasa-irsa-spherex/qr2/level2/
```

Example anonymous listing command:

```bash
aws s3 ls --no-sign-request s3://nasa-irsa-spherex/qr2/level2/
```

### 2.4 SPExPI reference implementation

Required source repository:

```text
https://github.com/fkiwy/spexpi
```

Store locally:

```text
/spherex_cache/external/source/spexpi/
```

SPExPI should be treated as:

1. A reference implementation for point-source SPHEREx extraction.
2. A validation oracle for selected test cases.
3. A source of practical implementation details.
4. Not the production architecture for full-field laser-beacon mining.

---

## 3. Hardware, Storage, and Network Assumptions

### 3.1 Local compute

```text
32 CPU cores
256 GB RAM
3x NVIDIA RTX Ada 6000 GPUs, 48 GB VRAM each
```

### 3.2 Local storage and network

```text
49 TB free on NAS
10 GbE direct NAS link to main compute machine
10 GbE wider internet link
Observed wider-internet throughput around 3501.2 Mb/s
```

### 3.3 Design implications

The pipeline should be designed around heavy local caching.

Rules:

```text
Never redownload a SPHEREx file if a verified local copy exists.
Cache full parent MEFs when doing field-based processing.
Cache calibration products.
Cache manifests and remote query results.
Cache target-index shards.
Cache measurement Parquet shards.
Cache QA plots, logs, and candidate packets.
Make cache invalidation explicit and versioned.
```

The local system should behave like a serious working archive, not a temporary scratch directory.

---

## 4. Core Mental Model

SPHEREx fields are the conveyor belt.

Gaia is the local high-speed target selector.

Measurements are the durable product.

Signal candidates are downstream analysis products.

Correct architecture:

```text
For each SPHEREx field/detector:
    load/cache the full parent MEF
    determine the image footprint
    query local Gaia/manual target index for objects inside that footprint
    apply target filters
    propagate target positions to observation epoch
    convert RA/Dec to detector x/y
    perform forced photometry on all selected targets
    write source_measurement rows as an append-only Parquet shard
    register shard/provenance
```

Incorrect architecture:

```text
For each Gaia source:
    repeatedly query SPHEREx remotely
    download target cutouts one at a time
    build one source spectrum interactively
```

The target-first model is useful for debugging and comparison to SPExPI, but it is not the mining architecture.

---

## 5. Terminology

Use these terms consistently.

```text
field
    One SPHEREx Level 2 image/detector product.

image_id
    Stable identifier for one SPHEREx field/detector/processing-version product.

target
    An object to measure in a field.
    Usually Gaia-derived, but may be manually defined.

target_prior
    The local catalog/index of objects to consider before looking at SPHEREx fluxes.

measurement
    One extracted SPHEREx flux point for one target in one field/detector.

spectrum
    Many native measurements assembled for one target across wavelength, detector, and time.

signal_candidate
    A possible narrowband excess found after measurement/spectrum analysis.

candidate_target
    Do not use this term in code. Use target.

candidate
    Avoid this term unless it means signal_candidate.
```

---

## 6. Local Cache Layout

Use a durable, explicit cache layout.

```text
/spherex_cache/
  raw/
    qr2/
      level2/
        <planning_period>/
          <processing_version>/
            detector=<detector>/
              <original_file>.fits

  calibration/
    qr2/
      spectral_wcs/
      psf/
      sapm/
      flags/
      dark/
      readnoise/
      nonlinear/
      dichroic/
      gain_factors/
      other/

  manifests/
    image_manifest/
    s3_inventory/
    irsa_sia2_queries/
    irsa_tap_queries/
    local_inventory/

  gaia/
    raw_projection/
    target_index/
    crossmatch_vetoes/

  manual_targets/
    targets.yaml
    targets.parquet

  derived/
    measurements/
    spectra/
    signal_candidates/
    candidate_packets/
    qa/
    plots/
    logs/

  external/
    docs/
      SPHEREx_Expsupp_QR.pdf
    source/
      spexpi/
```

### 6.1 Cache metadata

Every cached file should have metadata:

```text
cache_id
source_url
s3_uri
local_path
file_size
checksum
checksum_method
download_timestamp
release
processing_version
planning_period
observation_id
detector
last_used_timestamp
cache_status
derived_products_depending_on_it
```

Do not rely on filenames alone.

### 6.2 Cache tiers

#### Tier 1: permanent cache

Always retain:

```text
SPHEREx Explanatory Supplement
SPExPI source checkout
pipeline configs
Gaia target-prior catalog/index
calibration products used by processed measurements
image manifests
measurement Parquet shards
candidate packets
reproducibility logs
```

#### Tier 2: sticky cache

Keep unless space pressure becomes severe:

```text
parent MEFs used by high-value candidates
parent MEFs used by validation targets
parent MEFs for benchmark fields
parent MEFs for repeated extraction runs
cutouts used for debugging
```

#### Tier 3: evictable cache

Can be deleted/regenerated:

```text
parent MEFs from exploratory blind sweeps
temporary intermediate files
failed partial downloads
duplicate older processing attempts
```

With 49 TB free, default mode should be cache-first, evict-later.

---

## 7. Local Gaia Target Index

The local Gaia copy is not merely a hand-picked list. It is a high-speed spatial database used by field workers.

Create two layers:

```text
gaia_raw_projection
    Big but stripped-down Gaia projection containing only columns needed for filtering,
    spatial indexing, and coordinate propagation.

gaia_target_index
    HEALPix-partitioned, filterable, field-queryable local index used by workers.
```

### 7.1 Minimum Gaia-derived columns

```text
source_id
ra
dec
ref_epoch
pmra
pmdec
parallax
parallax_error
phot_g_mean_mag
phot_bp_mean_mag
phot_rp_mean_mag
bp_rp
ruwe
duplicated_source
astrometric_params_solved
non_single_star_flag_if_available
variable_flag_if_available
healpix_6
healpix_8
healpix_10
rough_distance_pc
rough_absolute_mag
rough_color_class
target_filter_flags
priority_score
```

### 7.2 Later enrichments

```text
2MASS
WISE
SIMBAD
known exoplanet hosts
known ultracool dwarfs
known white dwarfs
Gaia quasar candidates / galaxy vetoes
manual include/exclude lists
```

### 7.3 Field query pattern

Workers need to answer this fast:

```sql
SELECT *
FROM gaia_target_index
WHERE healpix_10 IN (<field_covering_healpix_cells>)
  AND passes_filter_profile = true
  AND source position is inside SPHEREx field footprint;
```

The field footprint may require a coarse HEALPix prefilter followed by an exact polygon or WCS-boundary test.

---

## 8. Manual Targets

The pipeline must support objects without Gaia IDs.

Manual target schema:

```text
target_id
target_type
object_name
ra_deg
dec_deg
reference_epoch_yr
pmra_masyr
pmdec_masyr
parallax_mas
source_catalog
source_catalog_id
priority_score
notes
```

Manual targets enter through the same target-selection system as Gaia targets. The field worker should not special-case them.

### 8.1 Required SIMP manual target

Include the SPExPI example brown dwarf as a manual target:

```text
object_name: T2 SIMP J013656.5+093347.3
ra_deg: 24.2412498
dec_deg: 9.5630705
pmra_masyr: 1238.244
pmdec_masyr: -16.156
reference_epoch_yr: 2016.0
target_type: manual_coordinate
source_catalog: manual
source_catalog_id: simp0136
```

This object may not have a useful Gaia source ID. The pipeline must not require Gaia IDs.

---

## 9. Target Filters

Target selection happens per field.

The worker should select objects satisfying configurable filters:

```text
inside SPHEREx field footprint
inside usable detector region
passes magnitude limits
not obviously saturated
not uselessly faint
good enough astrometry
not duplicated/problematic unless allowed
not quasar/galaxy-like
not too crowded
not in manual exclusion list
```

Implement filter profiles, not hard-coded rules.

Recommended profiles:

```text
broad_debug
all_clean_stars
nearby_clean_fgkm
nearby_clean_m_dwarfs
ultracool_dwarfs
white_dwarfs
manual_targets
control_giants
```

Every measurement row must record which filter profile selected it.

### 9.1 Early default filter

For the first smoke test, use a permissive profile:

```text
filter_profile = broad_debug
```

Characteristics:

```text
include manual targets
include Gaia stars in footprint
basic magnitude cuts only
basic astrometric quality cuts
avoid aggressive astrophysical cuts
allow enough targets to test field processing
```

Do not tune science filters before the field pipeline works.

---

## 10. SPHEREx Image Manifest

Build a manifest containing one row per SPHEREx image/detector product.

Schema:

```text
image_id
release
processing_version
planning_period
observation_id
detector
s3_uri
irsa_url
local_path
file_size
checksum
obs_start_time
obs_mid_time
obs_end_time
footprint_polygon
healpix_cover
wavelength_range_min_um
wavelength_range_max_um
status
created_at
updated_at
```

The manifest builder should support:

```text
IRSA SIA2 discovery
IRSA TAP discovery
AWS S3 listing / inventory discovery
local cache scanning
```

For scale, one field job should correspond to one image/detector/processing-version/filter-profile combination.

---

## 11. Correct Field Photometry System

The field photometry engine is the heart of the project.

Given one cached SPHEREx MEF, it must:

```text
1. Open the MEF.
2. Read IMAGE.
3. Read VARIANCE.
4. Read FLAGS.
5. Read ZODI.
6. Read PSF information.
7. Read wavelength/bandwidth calibration.
8. Determine observation epoch/time.
9. Determine image footprint.
10. Query local Gaia/manual target index.
11. Propagate each target to observation epoch.
12. Convert propagated RA/Dec to detector x/y.
13. Reject targets too close to edge or bad regions.
14. Perform aperture photometry.
15. Perform PSF photometry where possible.
16. Estimate local background.
17. Use VARIANCE and empirical background scatter for uncertainty.
18. Store zodi model value separately.
19. Store flags and quality masks.
20. Write measurement rows.
```

Aperture photometry is required for simplicity and debugging.

PSF photometry is required for production-quality extraction, especially in crowded fields and for reliable narrowband residuals.

---

## 12. Coordinate Propagation and WCS

### 12.1 Proper motion

For every target, propagate position from reference epoch to observation epoch using proper motion.

Required fields:

```text
ra_reference_deg
dec_reference_deg
reference_epoch_yr
pmra_masyr
pmdec_masyr
parallax_mas optional
obs_mid_time
ra_epoch_deg
dec_epoch_deg
```

High-proper-motion objects must not be silently mishandled.

### 12.2 Sky-to-detector transform

For every propagated target:

```text
RA/Dec at observation epoch -> detector x/y pixel coordinate
```

Record:

```text
x_pix
y_pix
wcs_solution_id
inside_footprint
edge_distance_pix
```

Reject or flag targets near edges.

---

## 13. Wavelength Handling

Do not treat SPHEREx as fixed wavelength buckets during extraction.

Each measurement row must record native wavelength information:

```text
cwave_um
cband_um
wavelength_source
wavelength_calibration_file
wavelength_quality_flag
detector
x_pix
y_pix
observation_id
processing_version
```

Rebinning into spectra happens later.

### 13.1 Calibration priority

Use the best available wavelength/bandwidth calibration. Prefer dedicated Spectral WCS calibration products such as CWAVE/CBAND when available and recommended by SPHEREx docs. WCS-WAVE in the MEF may be used for initial smoke tests/debugging, but code must make the wavelength source explicit and store it in every measurement row.

---

## 14. Aperture Photometry Requirements

For each target:

```text
center aperture at propagated x/y
use configurable aperture radius
use configurable background annulus
mask flagged pixels
sigma-clip background annulus
sum IMAGE values in aperture
estimate background-subtracted flux
propagate uncertainty from VARIANCE
include empirical local background noise
record number of good/bad pixels
record aperture correction if applied
```

Important unit rule:

```text
IMAGE is surface brightness. The code must handle point-source flux conversion carefully.
Do not blindly sum MJy/sr pixels and call that flux without recording pixel solid angle,
calibration assumptions, aperture correction assumptions, and unit conversion.
```

Aperture output fields:

```text
aperture_flux
aperture_flux_unc
aperture_flux_unit
aperture_radius_pix
annulus_inner_pix
annulus_outer_pix
background
background_unc
n_aperture_pixels
n_good_aperture_pixels
n_bad_aperture_pixels
aperture_correction
aperture_status
```

---

## 15. PSF Photometry Requirements

Implement a production PSF/PRF fitting path.

For each target:

```text
extract postage stamp
load/interpolate appropriate SPHEREx PSF for detector/field/sector/wavelength
fit source amplitude plus local background
optionally fit small centroid offset
mask bad pixels using FLAGS
weight pixels using VARIANCE
return amplitude, uncertainty, chi-square, fit quality, and residual metrics
```

Minimum PSF model:

```text
flux_scale * PSF(x - x0, y - y0) + constant_background
```

Better model:

```text
flux_scale * PSF(x - x0, y - y0) + tilted_plane_background
```

Record:

```text
psf_model_id
psf_sector
psf_wavelength
psf_flux
psf_flux_unc
psf_background
psf_chi2
psf_dof
centroid_dx_pix
centroid_dy_pix
fit_status
fit_warning_flags
```

PSF fitting must be optional in the first smoke test but mandatory in the production photometry design.

---

## 16. FLAGS, VARIANCE, ZODI, and Background Handling

### 16.1 FLAGS

Use FLAGS to mask bad pixels and compute measurement quality.

Record:

```text
flags_summary
quality_mask
bad_pixel_fraction_aperture
bad_pixel_fraction_annulus
fatal_flag_present
```

The exact bit meanings must be taken from the Explanatory Supplement / current IRSA docs.

### 16.2 VARIANCE

Use VARIANCE for uncertainty propagation, but also compute empirical local background scatter.

Record both:

```text
variance_model_unc
empirical_background_unc
combined_flux_unc
```

### 16.3 ZODI

ZODI is not subtracted from IMAGE in QR Level 2 spectral images. Store zodi model values separately.

Do not destructively subtract zodi in the first production measurement table. If a zodi-subtracted value is computed, store it as a separate derived column with explicit provenance.

Record:

```text
zodi_model_at_target
zodi_aperture_mean
zodi_annulus_mean
zodi_handling_mode
```

---

## 17. Measurement Output

Write append-only Parquet shards.

Do not make every worker insert billions of rows into one live SQL table.

Path pattern:

```text
/spherex_cache/derived/measurements/
  release=qr2/
    processing_version=<version>/
      planning_period=<period>/
        detector=<detector>/
          image_id=<image_id>/
            filter_profile=<profile>/
              measurements.parquet
              qa.json
              extraction.log
```

### 17.1 Measurement row schema

```text
measurement_id
target_id
target_type
source_id
image_id
release
processing_version
planning_period
detector
observation_id
obs_mid_time
filter_profile
ra_reference_deg
dec_reference_deg
reference_epoch_yr
pmra_masyr
pmdec_masyr
ra_epoch_deg
dec_epoch_deg
x_pix
y_pix
edge_distance_pix
cwave_um
cband_um
wavelength_source
wavelength_calibration_file
image_value_raw
image_unit
aperture_flux
aperture_flux_unc
aperture_flux_unit
psf_flux
psf_flux_unc
psf_flux_unit
background
background_unc
variance_model_unc
empirical_background_unc
zodi_model_at_target
zodi_aperture_mean
zodi_annulus_mean
flags_summary
quality_mask
bad_pixel_fraction_aperture
bad_pixel_fraction_annulus
n_aperture_pixels
n_good_aperture_pixels
n_bad_aperture_pixels
aperture_radius_pix
annulus_inner_pix
annulus_outer_pix
aperture_correction
psf_model_id
psf_sector
psf_wavelength
psf_fit_status
psf_chi2
psf_dof
centroid_dx_pix
centroid_dy_pix
photometry_backend
input_file_path
calibration_file_path
pipeline_version
config_hash
created_at
```

---

## 18. QA Outputs

Each field job should produce QA artifacts.

Required smoke-test QA:

```text
qa.json
extraction.log
field_footprint.png
detector_targets.png
simp_postage_aperture.png if SIMP is in field
simp_postage_psf.png if PSF photometry enabled
flux_vs_wavelength_debug.png if multiple fields are processed
```

Recommended `qa.json` fields:

```text
image_id
job_id
detector
filter_profile
targets_considered
targets_inside_footprint
targets_after_filters
targets_measured
targets_rejected_edge
targets_rejected_flags
targets_failed_photometry
median_background
median_zodi
median_flux_unc
bad_pixel_fraction_image
bad_pixel_fraction_measured_apertures
runtime_seconds
download_seconds
fits_open_seconds
target_query_seconds
photometry_seconds
psf_fit_seconds
write_seconds
```

---

## 19. Job System for N-Node Scaling

Use a central lightweight job database for manifests and state.

Core tables:

```text
spherex_image_manifest
field_job
measurement_shard_registry
worker_heartbeat
```

### 19.1 Field job schema

```text
job_id
image_id
release
processing_version
planning_period
detector
filter_profile
pipeline_version
config_hash
status
claimed_by
claimed_at
started_at
finished_at
attempt_count
max_attempts
error_message
output_shard_path
created_at
updated_at
```

States:

```text
pending
claimed
running
done
failed
needs_retry
skipped
```

### 19.2 Idempotency rules

A field job is uniquely identified by:

```text
image_id
detector
release
processing_version
filter_profile
pipeline_version
config_hash
```

Before processing:

```text
if final shard exists
and registry says done
and schema validates
and config_hash matches:
    skip or mark done
else:
    process or reprocess
```

During processing:

```text
write measurements.tmp.parquet
validate file and schema
rename or copy atomically to measurements.parquet
register shard
mark job done
```

Never write directly to the final shard path.

### 19.3 Shard registry schema

```text
shard_id
job_id
image_id
release
processing_version
planning_period
detector
filter_profile
pipeline_version
config_hash
row_count
file_path
file_size
checksum
created_at
status
```

---

## 20. N-Node Scaling Model

Workers should scale over SPHEREx field/detector jobs.

Correct scale unit:

```text
one SPHEREx image/detector/processing-version/filter-profile job
```

This supports:

```text
1 local worker
32 local workers
100 AWS Batch jobs
1000 EKS pods
mixed CPU/GPU worker pools
```

Persistent shared state lives in:

```text
S3 or shared NAS for raw/cache/data products
S3 or shared NAS for measurement shards
Postgres/RDS or local DB for job/manifests/shard registry
container image version
config hash
```

---

## 21. GPU/Warp/CUDA Policy

Do not port the whole pipeline to CUDA.

Most of the pipeline is I/O, FITS handling, WCS, metadata, storage, and orchestration.

Design a pluggable photometry backend:

```text
cpu_numpy
numba_cpu
cupy_gpu
warp_gpu
```

Start with CPU correctness.

GPU is justified only after profiling shows photometry or PSF fitting dominates wall-clock time, especially in field-level batch extraction.

Do not let GPU become the only truth path. The CPU backend is the reference implementation.

Recommended interface:

```python
class PhotometryBackend:
    def aperture_batch(self, image, variance, flags, targets, config):
        ...

    def annulus_background_batch(self, image, variance, flags, targets, config):
        ...

    def psf_fit_batch(self, image, variance, flags, psf_model, targets, config):
        ...
```

---

## 22. Containerized Deployment Requirement

The final production version must run as a containerized distributed worker system deployable on AWS.

Supported deployment modes:

```text
local_single_node
local_multi_worker
aws_ec2_single_node
aws_batch
aws_eks_kubernetes
```

Local execution is the development/validation path. AWS/EKS is the scale-out path for whole-archive mining.

### 22.1 Container images

Minimum container images:

```text
spherex-mine-worker
    Runs field jobs: cache/read MEF, select targets, forced photometry, write measurement shard.

spherex-mine-cli
    Administrative CLI: build manifests, submit jobs, inspect status, retry failures.

spherex-mine-notebook
    Optional analysis/debug image with notebooks and plotting tools.

spherex-mine-api
    Optional later service for browsing targets, measurements, spectra, and candidates.
```

The first production image should be:

```text
spherex-mine-worker
```

### 22.2 Container build expectations

Use Python 3.12.

Likely dependencies:

```text
astropy
astroquery
pyvo
numpy
scipy
pandas or polars
pyarrow
duckdb
sqlalchemy
psycopg
boto3
s3fs
fsspec
healpy
matplotlib
tqdm
pydantic
typer
pytest
```

Optional later variants:

```text
CuPy CUDA image
NVIDIA Warp image
GPU-enabled PSF photometry worker
```

Do not make GPU dependencies mandatory for the CPU worker.

### 22.3 Runtime configuration

All runtime configuration must come from environment variables and mounted config files, not hard-coded paths.

Required configuration categories:

```text
SPHEREX_CACHE_ROOT
SPHEREX_RELEASE
SPHEREX_PROCESSING_VERSION
TARGET_INDEX_URI
MEASUREMENT_OUTPUT_URI
JOB_DB_URI
AWS_REGION
WORKER_ID
PHOTOMETRY_BACKEND
FILTER_PROFILE
MAX_FIELDS_PER_WORKER
LOG_LEVEL
```

Support local paths and S3 URIs.

Examples:

```bash
SPHEREX_CACHE_ROOT=/spherex_cache
MEASUREMENT_OUTPUT_URI=/spherex_cache/derived/measurements
```

or:

```bash
SPHEREX_CACHE_ROOT=s3://my-spherex-cache/raw
MEASUREMENT_OUTPUT_URI=s3://my-spherex-results/measurements
```

---

## 23. AWS Architecture

Recommended AWS production architecture:

```text
S3 bucket: raw/cache SPHEREx products
S3 bucket: measurement Parquet shards
S3 bucket: QA artifacts and logs
RDS Postgres or Aurora Postgres: job queue, manifests, shard registry
ECR: container images
EKS or AWS Batch: distributed workers
CloudWatch: logs and metrics
IAM roles for service accounts / task roles
Optional FSx for Lustre: high-throughput shared cache for repeated large campaigns
```

AWS Batch may be simpler for the first archive-scale run.

EKS is appropriate if we want:

```text
long-running worker pools
autoscaling
custom queues
mixed CPU/GPU node groups
dashboard/API services
interactive operations
```

Do not require Kubernetes for local development.

### 23.1 Kubernetes/EKS model

In EKS, workers should run as Kubernetes Jobs or as a scalable worker Deployment.

Preferred model:

```text
Central job table contains pending field jobs.

Each worker pod:
    claims one field job atomically
    downloads/caches required files
    runs field photometry
    writes output shard to S3
    registers shard
    marks job done
    claims next job or exits
```

Worker pods must be stateless except for ephemeral scratch storage.

Persistent state lives in:

```text
S3
RDS/Postgres
container image version
config hash
```

### 23.2 Data locality and cache modes on AWS

Supported cache modes:

```text
none
    Stream/read directly from S3 where practical.

ephemeral
    Download each MEF/calibration product to pod-local NVMe/scratch for the job.

shared
    Use FSx for Lustre or mounted high-throughput storage for repeated access.

persistent-s3
    Maintain our own S3 mirror/cache of SPHEREx products and calibration files.
```

For first AWS run, use ephemeral scratch plus S3 output.

For repeated full-archive campaigns, consider persistent S3 cache or FSx for Lustre.

---

## 24. Observability

Production workers must emit structured logs and metrics.

Minimum metrics:

```text
jobs_claimed
jobs_completed
jobs_failed
download_seconds
fits_open_seconds
target_query_seconds
photometry_seconds
psf_fit_seconds
write_seconds
targets_selected
targets_measured
targets_rejected
bad_pixel_fraction
measurement_rows_written
output_bytes_written
```

Every log line should include:

```text
worker_id
job_id
image_id
detector
release
processing_version
filter_profile
pipeline_version
config_hash
```

---

## 25. Smoke Test: SIMP Field-Based Scanner

The first smoke test must be field-based.

Goal:

```text
Process an entire SPHEREx field containing the SIMP brown dwarf and produce measurements
for all selected targets, including SIMP as a manual target.
```

This is not a target-by-target SPExPI extraction.

### 25.1 Smoke test procedure

```text
1. Add SIMP J013656.5+093347.3 as a manual target.

2. Query/build the SPHEREx image manifest for QR2 Level 2 spectral images whose footprint
   contains the SIMP propagated coordinate.

3. Select one parent MEF/detector image that contains SIMP.

4. Cache the full parent MEF locally.

5. Build a mini Gaia target index for the same field footprint:
   - Gaia stars inside the field polygon
   - basic quality cuts
   - reasonable magnitude limits
   - no aggressive science cuts yet

6. Combine:
   - Gaia targets inside the field
   - manual SIMP target
   - optional negative-control coordinate nearby with no known source

7. Run the normal field worker:
   - load full MEF
   - determine footprint
   - select field targets
   - propagate positions
   - convert to detector x/y
   - run aperture photometry
   - optionally run PSF photometry
   - write measurement shard

8. Confirm:
   - SIMP target was selected
   - SIMP landed at plausible x/y
   - SIMP was not rejected by edge/flag logic
   - measurement row exists for SIMP
   - Gaia field targets also produced rows
   - negative-control coordinate behaves as expected
   - QA plots are generated

9. Compare gross results against SPExPI only as a sanity check:
   - similar wavelength
   - similar flux scale within understood differences
   - no obvious coordinate propagation failure
```

### 25.2 Smoke test expected outputs

```text
runs/smoke_simp_field/
  config.yaml
  field_job.json
  cached_files.json
  target_selection.parquet
  measurements.parquet
  qa.json
  extraction.log
  plots/
    field_footprint.png
    detector_targets.png
    simp_postage_aperture.png
    simp_postage_psf.png
    flux_vs_wavelength_debug.png
```

### 25.3 Smoke test success criteria

```text
A single SPHEREx parent field is processed by the field-based scanner.
The output contains many Gaia-selected field targets.
The manual SIMP target is included and measured.
The measurement shard has valid x/y, wavelength, flags, background, zodi,
aperture flux, optional PSF flux, and provenance fields.
The run is reproducible from local cached files.
```

If no QR2 SPHEREx field currently contains SIMP, fall back to:

```text
1. another SPExPI example target with QR2 coverage, or
2. a known QR2 field with a manually selected bright isolated Gaia star.
```

But the preferred smoke test is the SIMP-containing field.

---

## 26. Codebase Layout

Recommended Python package:

```text
spherex_laser_miner/
  pyproject.toml
  Dockerfile
  README.md
  configs/
    local.yaml
    docker.yaml
    aws_batch.yaml
    eks.yaml
    smoke_simp.yaml

  spherex_laser_miner/
    __init__.py
    config.py
    logging.py
    cli.py

    cache/
      manager.py
      inventory.py
      s3.py
      local.py

    manifest/
      spherex_images.py
      irsa_sia.py
      irsa_tap.py
      s3_inventory.py
      local_scan.py

    catalog/
      gaia_projection.py
      gaia_index.py
      manual_targets.py
      filters.py
      proper_motion.py
      spatial.py

    fitsio/
      reader.py
      hdu.py
      flags.py
      zodi.py
      variance.py
      psf.py
      wavelength.py
      footprint.py

    photometry/
      aperture.py
      background.py
      psf_fit.py
      uncertainty.py
      backends/
        cpu_numpy.py
        numba_cpu.py
        cupy_gpu.py
        warp_gpu.py

    storage/
      parquet.py
      duckdb.py
      postgres.py
      schemas.py
      shard_registry.py

    jobs/
      queue.py
      worker.py
      heartbeat.py
      retry.py

    pipelines/
      build_manifest.py
      build_gaia_index.py
      run_field.py
      run_field_smoke_test.py
      submit_jobs.py
      assemble_spectra.py
      search_lines.py

    qa/
      plots.py
      reports.py
      metrics.py

    validation/
      spexpi.py
      compare.py

  tests/
    test_cache.py
    test_manifest.py
    test_gaia_index.py
    test_proper_motion.py
    test_wavelength.py
    test_aperture_photometry.py
    test_psf_fit.py
    test_field_worker_smoke.py

  notebooks/
    00_smoke_simp_review.ipynb
    01_field_measurement_review.ipynb
    02_spectrum_assembly_review.ipynb
```

---

## 27. CLI Commands

Required early CLI commands:

```bash
spherex-mine download-docs
spherex-mine build-manifest --release qr2
spherex-mine build-mini-gaia-index --around-target simp0136 --radius-deg 2
spherex-mine run-field --image-id <image_id> --filter-profile broad_debug
spherex-mine run-field-smoke-test --target simp0136 --release qr2
spherex-mine worker --mode claim-one
spherex-mine submit-jobs --release qr2 --filter-profile broad_debug --limit 10
spherex-mine inspect-job --job-id <job_id>
spherex-mine retry-failed --release qr2
```

Container acceptance command:

```bash
docker run --rm \
  -v /spherex_cache:/spherex_cache \
  -e SPHEREX_CACHE_ROOT=/spherex_cache \
  spherex-mine-worker:latest \
  spherex-mine run-field-smoke-test --target simp0136 --release qr2
```

---

## 28. Development Milestones

### Milestone 1: one-field smoke test

Deliver:

```text
field worker
mini Gaia index
manual target support
full-MEF cache
aperture photometry
measurement shard
QA plots
SIMP field smoke test
Docker container that can run the smoke test
```

### Milestone 2: correct PSF photometry

Deliver:

```text
PSF reader
PSF selection/interpolation logic
weighted PSF fit
fit diagnostics
comparison against aperture photometry
QA residual plots
```

### Milestone 3: manifest and N-node queue

Deliver:

```text
image manifest builder
job queue
worker claiming
atomic shard writes
worker heartbeat
retries
provenance registry
```

### Milestone 4: larger Gaia field processing

Deliver:

```text
process 100 fields
write measurement shards
query with DuckDB
assemble per-target native spectra
basic residual plots
```

### Milestone 5: AWS small batch

Deliver:

```text
ECR image
S3 output path
RDS/Postgres job table or equivalent
CloudWatch logs
10-field AWS Batch or EKS test
idempotent retry behavior
```

### Milestone 6: narrowband excess search

Deliver:

```text
native measurement spectrum assembler
smooth continuum model
positive residual search
artifact vetoes
signal_candidate table
candidate packets
```

### Milestone 7: full-archive mining

Deliver:

```text
manifest for all available QR2 image/detector products
job submission for full archive
autoscaled AWS workers
measurement shards in S3
spectrum assembly from shards
line-search candidate generation
```

---

## 29. Narrowband Excess Search: Later Stage Only

Do not build this first, but reserve architecture for it.

The signal-candidate database is downstream of measurement shards.

A signal candidate should eventually represent:

```text
source_id / target_id
line_wavelength_um
line_wavelength_uncertainty
line_snr
flux_excess
continuum_model_id
repeatability_score
artifact_veto_score
source_class
followup_priority
candidate_packet_path
review_status
```

Detection should compare:

```text
H0: smooth stellar SED + calibration residual + noise
H1: H0 + unresolved emission line convolved with SPHEREx response
```

But this is not Milestone 1.

---

## 30. Codex Immediate Instructions

Build the one-field SIMP smoke test inside the same containerized field-worker architecture intended for AWS/EKS.

Do not write a special SIMP extractor.

Do not build a SPExPI wrapper as the main pipeline.

Do not build the signal-candidate database yet.

The SIMP target should enter only through the same manual-target table used by the general field scanner.

The worker should not know or care that SIMP is special.

### Immediate task list

1. Create the repository skeleton.
2. Add Python package structure and CLI.
3. Add Dockerfile for CPU worker.
4. Add config system supporting local paths and S3 URIs.
5. Add cache manager.
6. Add downloader for SPHEREx Explanatory Supplement.
7. Add SPExPI clone helper, but do not make SPExPI the main path.
8. Add manual target loader containing SIMP.
9. Add manifest discovery sufficient to find a SPHEREx QR2 field containing SIMP.
10. Add mini Gaia field-index builder for the selected field.
11. Add field worker:
    - load MEF
    - read IMAGE/VARIANCE/FLAGS/ZODI/wavelength info
    - determine footprint
    - select Gaia/manual targets
    - propagate positions
    - compute x/y
    - run aperture photometry
    - write measurements.parquet
    - write qa.json and plots
12. Add optional stub interface for PSF photometry.
13. Add smoke-test CLI:

```bash
spherex-mine run-field-smoke-test --target simp0136 --release qr2
```

14. Make the smoke test runnable inside Docker:

```bash
docker run --rm \
  -v /spherex_cache:/spherex_cache \
  -e SPHEREX_CACHE_ROOT=/spherex_cache \
  spherex-mine-worker:latest \
  spherex-mine run-field-smoke-test --target simp0136 --release qr2
```

### Milestone 1 acceptance test

The smoke test passes when:

```text
one full SPHEREx parent field is processed
many Gaia-selected targets are measured
manual SIMP target is included and measured
measurements.parquet exists and validates against schema
qa.json exists
field/detector target plot exists
SIMP postage stamp plot exists
all output paths are under /spherex_cache or configured output URI
run is reproducible from cached files
Docker execution works without host-specific code paths
```

---

## 31. Non-Goals for the First Build

Do not do these first:

```text
Do not build a pure target-by-target extractor.
Do not make SPExPI the main pipeline.
Do not build the final signal-candidate database.
Do not port anything to CUDA/Warp yet.
Do not ingest full Gaia before proving the one-field smoke test.
Do not mine the whole archive before the measurement rows are obviously sane.
Do not hard-code the SIMP object in the field worker.
Do not assume fixed SPHEREx wavelength buckets during measurement extraction.
```

---

## 32. The Plot in One Sentence

Build a field-first, cache-heavy, containerized SPHEREx forced-photometry miner: each worker processes one cached SPHEREx field, selects all eligible local Gaia/manual targets inside the field, extracts aperture/PSF measurements with full wavelength/flag/zodi/provenance metadata, writes append-only Parquet shards, and later the analysis layer assembles spectra to search for narrowband laser-like excesses.
