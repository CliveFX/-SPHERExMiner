# Survey Output Contract

## Economic Target

The near-term LuxQuarry All-Sky Engine target is:

```text
$5k of cloud GPU compute buys a full accessible-sky SPHEREx survey for:

1. Gaia sources in the selected bright/mid magnitude range, currently G ~= 8-14.
2. All 2MASS point-source catalog rows present in the processed local 2MASS PSC
   cache.
```

This is intentionally not "all Gaia." The all-catalog target is useful as a
long-term stress case, but it is too broad for the first cost gate. Gaia G 8-14
is the right optical-bright target band because it is large enough to require a
real frame-first GPU engine while avoiding the faint/noisy and bright/saturated
tails that produce poor spectra and wasted scoring work. 2MASS is included as a
separate all-source infrared-selected target set because it finds red/cool
objects that Gaia cuts can miss.

For this economic gate, 2MASS is not magnitude-capped. The planner's
`combined` mode means:

```text
Gaia:   catalog == gaia_dr3 and G within gaia_mag_min..gaia_mag_max
2MASS:  catalog == 2mass_psc, no magnitude filter
```

The estimator can only count rows that exist in the input projected-target
Parquet. A true all-2MASS estimate therefore requires an upstream frame-target
projection that was not capped to a small per-frame 2MASS sample.

Use capped frame-target products only for smoke tests and local performance
sampling:

```bash
.venv/bin/luxquarry-allsky build-frame-targets \
  --manifest runs/manifest_sample/frame_manifest.parquet \
  --out runs/frame_targets_sample_capped/frame_targets.parquet \
  --catalog all \
  --gaia-g-min 8 \
  --gaia-g-max 14 \
  --twomass-no-mag-filter \
  --gaia-max-sources-per-frame 20000 \
  --twomass-max-sources-per-frame 20000
```

Use the explicit all-2MASS path only when the frame count and expected row count
are intentional:

```bash
.venv/bin/luxquarry-allsky build-frame-targets \
  --manifest runs/manifest_sample/frame_manifest.parquet \
  --out runs/frame_targets_sample_all2mass/frame_targets.parquet \
  --catalog all \
  --gaia-g-min 8 \
  --gaia-g-max 14 \
  --gaia-max-sources-per-frame 0 \
  --all-2mass
```

`--all-2mass` disables both the 2MASS `mag_primary` filter and the 2MASS SQL
`LIMIT`. The Gaia G range remains active.

For serious cloud estimates, require the planner to prove the target table came
from uncapped/no-filter 2MASS input:

```bash
.venv/bin/luxquarry-allsky plan-survey-economics \
  --manifest runs/manifest_sample/frame_manifest.parquet \
  --projected-targets runs/projected_targets_sample_all2mass/frame_targets_projected.parquet \
  --out-dir runs/survey_plan_sample_all2mass \
  --catalog-selection combined \
  --require-all-2mass-input
```

The economics JSON always includes:

```text
all_2mass_input_valid
all_2mass_input_status: proven | invalid | unknown | not_applicable
all_2mass_input_reason
all_2mass_input_metadata
```

Treat `unknown` and `invalid` as smoke/test estimates, not cloud cost evidence.

Before every serious cloud estimate, the planner must write the actual catalog
cardinality used for the run:

```text
catalog: gaia | 2mass | all
magnitude_system: G | J | H | K | mixed
magnitude_min
magnitude_max
catalog_selection: gaia_g_8_14 | twomass_all_usable | combined
sky_region
gaia_target_count
twomass_target_count
deduplicated_target_count
frame_count
estimated_measurement_count
estimated_raw_output_bytes
```

Do not extrapolate from folklore counts when the local catalog can answer the
question.

## Output Modes

The engine has two product modes.

### Audit Mode

Audit mode is for correctness, injection/recovery validation, candidate
drilldown, and algorithm development.

It writes durable per-measurement parquet shards:

```text
frame/task shard manifest
compact measurement shards
spectra measurements
target summary
spectrum quality
candidate tables
line-score tables
injection/recovery products when enabled
```

Every measurement row keeps enough provenance to audit a spectral point:

```text
catalog/source/target IDs
RA/Dec and pixel position
fits_path
frame_group_id
image_id
detector/release
wavelength source and calibration files
SAPM calibration file
aperture/PSF geometry
cwave/cband
flags/status
aperture and PSF flux/uncertainty/diagnostics
```

Audit mode is allowed to be expensive. It is not the default shape for a
multi-billion-row survey.

### Survey Mode

Survey mode is the economic production path.

It still runs frame-first photometry, but it does not persist every raw
measurement by default. It writes reduced science products first:

```text
run manifest
task/frame status
target spectra products
target quality products
candidate products
candidate line-score products
injection truth and recovery summaries when enabled
retained raw measurement subsets
```

Raw measurement retention is controlled by policy. The default policy keeps raw
rows only for:

```text
injected targets
recovered candidates
high-rank science candidates
quality-control holdouts
debug samples
explicitly requested targets
failed or suspicious frames when configured
```

Survey mode must still make drilldown possible. A reduced spectral point must
carry enough compact provenance to find the contributing FITS products and, when
raw rows were retained, the exact measurement rows.

## Required Economics Metrics

Every benchmark and campaign summary should carry these fields once the inputs
are known:

```text
target_count
frame_count
measurement_count
spectra_count
candidate_count
retained_raw_measurement_count
parquet_write_bytes
bytes_per_measurement
bytes_per_spectrum
measurements_per_sec
measurements_per_gpu_sec
spectra_per_sec
spectra_per_gpu_sec
estimated_cloud_gpu_hourly_cost
estimated_cloud_total_cost
estimated_cost_per_billion_measurements
estimated_cost_per_million_spectra
```

The optimizer should prefer changes that improve `estimated_cost_per_million_spectra`
or `estimated_cost_per_billion_measurements` for the Gaia G 8-14 plus 2MASS
target set. Local wall time is secondary unless it changes those economics.

## GPU-First Requirements

Survey mode should push hot work into CUDA/RAPIDS where it makes sense:

```text
frame/image upload once per frame or frame stack
target coordinate arrays resident on device where possible
aperture and PSF kernels over large target arrays
PSF candidate reduction/gather fused where practical
cuDF filtering, grouping, sorting, and reduced table writes
Dask-cuDF for multi-GPU reducers
async shard/status writers outside the photometry critical path
```

The first implementation may still write audit-style compact shards internally,
but the product contract is reduced-output-first. If the engine cannot run the
Gaia G 8-14 plus 2MASS survey economically while writing every raw row, survey
mode must reduce earlier rather than merely scale storage.

## Acceptance Gate

A v2 cloud-readiness benchmark should answer:

```text
How many Gaia G 8-14 targets and 2MASS PSC targets are in the selected accessible sky?
How many SPHEREx frame measurements are required?
What fraction of raw rows are retained in survey mode?
How much data is written?
How many GPU-hours are required on the chosen instance class?
Would the estimated compute bill fit under $5k?
```

If the answer is no, the next optimization should attack the largest contributor
to cost for this target, not a generic micro-bottleneck.

## Estimator Command

Use the planner before a large run when you want materialized plan products:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky plan-survey-economics \
  --manifest runs/manifest_galactic_core_nearest20/frame_manifest.parquet \
  --projected-targets runs/projected_targets_galactic_core_nearest20_allcat_g11_16_n20000/frame_targets_projected.parquet \
  --out-dir runs/survey_plan_gc_nearest20_combined \
  --catalog-selection combined \
  --output-mode survey \
  --raw-retention-fraction 0.01 \
  --measurements-per-gpu-sec 73687 \
  --gpu-hourly-cost 6.88 \
  --gpu-count 8
```

It writes:

```text
survey_plan_frames.parquet
survey_plan_targets.parquet
survey_plan_unique_targets.parquet
survey_plan_summary.json
survey_economics_summary.json
```

Use the estimator directly when the plan products already exist or are not
needed:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky estimate-survey-economics \
  --manifest runs/manifest_galactic_core_nearest20/frame_manifest.parquet \
  --projected-targets runs/projected_targets_galactic_core_nearest20_allcat_g11_16_n20000/frame_targets_projected.parquet \
  --out runs/survey_economics_gc_nearest20_combined/summary.json \
  --catalog-selection combined \
  --output-mode survey \
  --raw-retention-fraction 0.01 \
  --measurements-per-gpu-sec 73687 \
  --gpu-hourly-cost 6.88 \
  --gpu-count 8
```

The first dense 20-frame sample produced:

```text
frame_count: 20
planned_projected_target_rows: 400,000
target_count: 32,648
gaia_target_count: 15,790
twomass_target_count: 16,858
in_frame_target_count: 29,918
measurement_count: 360,394
retained_raw_measurement_count: 3,604
estimated_output_gib: 0.115
estimated_compute_cost_usd: 0.0093
estimated_cost_per_billion_measurements: 25.94
estimated_cost_per_million_spectra: 0.31
```

That sample is not a cloud estimate; it is a local frame subset. The full
estimate must use a full accessible-sky projected target table before the `$5k`
gate means anything.

## Sample Extrapolation

When a full accessible-sky projected target table is not available yet, run the
planner on representative cells and extrapolate:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky extrapolate-survey-sample \
  --plan-summary runs/survey_plan_gc_nearest20_combined/survey_plan_summary.json \
  --out-dir runs/survey_sample_extrapolation_smoke \
  --target-cell-count 192
```

The output is:

```text
survey_sample_cell_summaries.parquet
survey_sample_extrapolation.json
```

This command is deliberately caveated. It is useful for answering whether the
architecture is roughly 2x, 10x, or 100x away from the `$5k` target. It is not a
replacement for a real full-sky plan, and it is only as good as the sampled
cells. A useful sample must include dense Galactic-plane regions and sparse
high-latitude regions.
