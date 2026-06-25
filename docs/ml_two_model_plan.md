# Two-Model Spectral ML Plan

This plan adds machine-learning systems on top of the existing SPHEREx mining
pipeline without replacing the deterministic photometry, injection, and GPU
matched-filter stack. The goal is to build two separate models:

1. A stellar science embedding model for organizing and comparing spectra.
2. A narrowband signal model trained with injected spectra and hard negatives.

The separation is intentional. The science model should learn stellar structure
without treating every unusual spectrum as a laser candidate. The narrowband
model should specialize in response-shaped excesses, injection recovery, and
false-positive rejection.

## Current Inputs

Existing outputs that should feed the ML dataset builders:

- `spectra/target_spectra.parquet`
  - Per-measurement aperture and PSF fluxes.
  - Uncertainties from SPHEREx variance products.
  - Wavelength fields using the calibrated wavelength path.
  - Detector/camera/observation/image metadata.
  - Pixel coordinates, edge distance, and flags.
  - Future runs include `run_kind`, `run_injection_applied`,
    `point_injection_applied`, `injection_applied`, and
    `injection_manifest_path`.
- `spectra/target_summary.parquet`
  - One row per target with measurement counts and flux summaries.
  - Future runs include target-level injection counts/fractions.
- `spectra/spectrum_quality.parquet`
  - Deterministic quality labels: `good`, `review`, `bad`.
  - Smoothness, flag fraction, aperture/PSF agreement, SNR, and usable point
    counts.
- Injection manifests and recovery products
  - `injection_manifest.json`
  - `narrowband_detector_truth/narrowband_recovery.parquet`
  - `narrowband_detector_raw/narrowband_candidates.parquet`
  - paired recovery outputs where present.
- Gaia metadata already carried through target rows
  - G/BP/RP magnitudes, color, parallax, proper motion, RUWE, source id.

## Model 1: Stellar Science Embedding

Purpose:

- Identify similar stellar spectra.
- Learn spectral families independent of apparent magnitude.
- Find SIMP/UCS-like objects and other unusual continua.
- Support nearest-neighbor browsing and clustering.
- Learn distance/luminosity relationships where Gaia parallax is reliable.

This model is not a narrowband detector.

### Representation

Do not require regular wavelength gridding as the primary representation. Treat
each spectrum as a ragged set/sequence of measured points:

```text
target:
  [
    wavelength_nm,
    cband_nm,
    aperture_flux_uJy,
    aperture_unc_uJy,
    psf_flux_uJy,
    psf_unc_uJy,
    fatal_flag_present,
    decoded flag features,
    detector/camera/channel ids,
    x_pix,
    y_pix,
    edge_distance_pix,
    obs_mid_time
  ]
```

Gaia metadata is attached at the target level:

```text
phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp,
parallax_mas, pmra_masyr, pmdec_masyr, ruwe
```

The model should split the learned representation into at least:

- `shape_embedding`: normalized spectral morphology, as magnitude-invariant as
  possible.
- `absolute_embedding`: spectral shape plus flux/magnitude/parallax context.
- `quality_embedding`: data reliability/artifact representation.

### Architecture

First serious implementation:

```text
RaggedSpectrumEncoder
  per-point MLP or small attention block
  detector/camera/channel embeddings
  wavelength positional features
  attention or mean/max pooling
  target-level Gaia metadata encoder
  fusion block

Heads:
  shape_embedding
  absolute_embedding
  quality_embedding
  masked-point reconstruction head
  Gaia prediction head
  quality prediction head
```

PCA and gridded models may be used as baselines, but they should not be treated
as the canonical representation because the source spectra are naturally ragged.

### Self-Supervised Training

Training examples come from good/review spectra. Bad spectra may be held out or
used for quality/artifact tasks.

Positive pairs:

- Two random subsets of points from the same target.
- Aperture view and PSF view of the same target.
- Same Gaia source observed in different runs/visits, when available.

Augmentations:

- Drop random points.
- Drop or downweight flagged points.
- Add noise consistent with uncertainty columns.
- Slightly perturb flux normalization.
- Split by detector/visit when enough measurements exist.

Losses:

- Contrastive loss: same-target views close, different targets farther apart.
- Masked point reconstruction: predict held-out fluxes from observed points.
- Gaia prediction: predict BP/RP, G, parallax-derived absolute magnitude where
  reliable.
- Quality prediction: reproduce deterministic `spectrum_quality_category` and
  key quality metrics.

### Outputs

Write model outputs as appendable parquet shards:

```text
ml_outputs/science_embeddings/<model_version>/
  embeddings.parquet
  reconstruction_errors.parquet
  nearest_neighbors.parquet
  model_card.json
  training_metrics.jsonl
```

Minimum columns:

```text
run_name
run_kind
target_id
source_id
object_name
phot_g_mean_mag
bp_rp
parallax_mas
spectrum_quality_score
spectrum_quality_category
n_measurements
n_usable_measurements
shape_embedding_000..shape_embedding_N
absolute_embedding_000..absolute_embedding_N
quality_embedding_000..quality_embedding_N
reconstruction_error
nearest_cluster_id
```

## Model 2: Narrowband Signal Model

Purpose:

- Learn the observed shape of injected narrowband excesses in real SPHEREx
  spectra.
- Rank candidates from raw science runs.
- Improve false-positive rejection beyond hand-tuned matched filters.
- Preserve aperture/PSF co-movement and detector response behavior.

This model is allowed to specialize in injection recovery and line-like signals.

### Inputs

Use the same ragged per-point spectra as Model 1, plus injection/recovery truth:

```text
run_kind
run_injection_applied
point_injection_applied
injection_manifest_path
injection_id
line_family
injected_line_nm
line_width_nm
find_me_snr
line_flux_uJy
frames_written
gpu/recovery matched candidate fields
```

The science embedding from Model 1 may be used as contextual input after Model 1
is stable:

```text
normal stellar shape context + local spectral measurements -> narrowband score
```

### Training Sets

Positive examples:

- Injected spectra with successful FITS-level injection.
- Include weak, medium, and strong signals.
- Include many wavelengths, detectors, cameras, magnitudes, backgrounds, and
  stellar colors.

Negative examples:

- Matched baseline runs for injected targets.
- Uninjected science spectra.
- Real false positives from blind scans.
- Hard negatives near detector edges, high flag fractions, and known artifact
  shapes.

Splits must avoid leakage:

- Hold out entire stars.
- Hold out entire sky regions/campaign centers.
- Hold out wavelength intervals.
- Hold out injection strength ranges for interpolation tests.
- Never split different injected versions of the same target across train/test
  unless the split is explicitly marked as a leakage diagnostic.

### Outputs

```text
ml_outputs/narrowband/<model_version>/
  candidate_scores.parquet
  injection_recovery.parquet
  false_positive_review.parquet
  model_card.json
  training_metrics.jsonl
```

Minimum candidate columns:

```text
run_name
run_kind
target_id
source_id
candidate_line_nm
candidate_width_nm
ml_signal_probability
ml_review_grade
ml_rank_score
aperture_support_score
psf_support_score
aperture_psf_agreement
flag_penalty
nearest_flag_distance
matched_filter_score
injection_truth_id
is_injected_target
is_recovered_injection
```

Review grades should remain compatible with current candidate UI semantics:

- `S`: unambiguous, immediate outside-look candidate. Expected to be extremely
  rare.
- `A`: strong candidate.
- `B`: plausible candidate.
- `C`: weak/review-only candidate.
- `D`: artifact/low priority.

## Injection Data Enablement

Future runs now carry explicit injection provenance in spectra outputs. Dataset
builders should use those fields first and manifest inference second:

1. Read `target_spectra.parquet`.
2. Use `run_kind`, `run_injection_applied`, `point_injection_applied`, and
   `injection_manifest_path` when present.
3. If those columns are absent in old runs, fall back to:
   - run name suffixes,
   - `run_summary.json` `path_overrides_path`,
   - nearby `injection_manifest.json`,
   - recovery table joins.
4. Join manifests and recovery tables onto target spectra by `target_id` and
   `injection_id`.
5. Preserve one clean label namespace:

```text
label_source = injected_manifest | baseline_pair | blind_candidate | unknown
is_injected_run
is_injected_target
is_injected_measurement
is_recovered
```

### Required Dataset Builder

Create:

```text
tools/build_ml_datasets.py
```

Responsibilities:

- Scan run directories or campaign directories.
- Read spectra, quality, summaries, manifests, and candidate/recovery outputs.
- Emit separate science and narrowband dataset shards.
- Produce train/validation/test split IDs.
- Write a manifest with row counts, spectra counts, injected counts, quality
  distributions, wavelength distributions, and leakage checks.

Suggested outputs:

```text
/mnt/niroseti/spherex_cache/ml_datasets/<dataset_name>/
  science_targets.parquet
  science_points.parquet
  narrowband_targets.parquet
  narrowband_points.parquet
  injection_truth.parquet
  split_manifest.parquet
  dataset_summary.json
```

## Training Dashboards

Training should have live in-progress dashboards from the start. Do not wait
until models are mature.

### Dashboard Index

Add links from the existing `/dashboards` page:

- `/ml-training`
- `/ml-datasets`
- `/ml-embeddings`
- `/ml-narrowband`

### `/ml-training`

Purpose: monitor active and recent training jobs.

Data source:

```text
ml_runs/<run_name>/training_status.json
ml_runs/<run_name>/training_metrics.jsonl
ml_runs/<run_name>/checkpoints/
```

Cards:

- Active model name/version.
- Model type: `science_embedding` or `narrowband`.
- Epoch/step.
- Examples/sec.
- GPU utilization if captured.
- Train loss.
- Validation loss.
- Contrastive accuracy or retrieval recall.
- Injection recovery rate for narrowband model.
- False-positive rate on baseline validation.
- Best checkpoint path.
- Runtime and ETA.

Plots:

- Train/validation loss over time.
- Learning rate over time.
- Retrieval recall@K for science model.
- Recovery fraction by SNR for narrowband model.
- False positives per 1k targets for narrowband model.

### `/ml-datasets`

Purpose: inspect dataset composition before training.

Cards/tables:

- Total targets.
- Total spectral points.
- Good/review/bad spectra counts.
- Baseline/injected run counts.
- Injected target count.
- Injection count by line wavelength.
- Injection count by strength.
- Magnitude histogram.
- Detector/camera histogram.
- Train/validation/test split counts.
- Leakage warnings.

### `/ml-embeddings`

Purpose: inspect science model output.

Features:

- 2D projection of `shape_embedding` using PCA/UMAP for display only.
- Color by quality, magnitude, BP/RP, run kind, cluster, or reconstruction
  error.
- Click point -> spectra page.
- Search target id/source id/object name.
- Nearest-neighbor table for selected target.
- Known anchor panel for UCS/SIMP-like examples.

### `/ml-narrowband`

Purpose: inspect narrowband model output.

Features:

- Candidate table with ML score, grade, wavelength, target id, and links.
- Injection recovery table.
- Missed-injection table.
- False-positive table.
- Recovery by SNR and wavelength.
- Spectra page links.
- Overlay of model score window and deterministic matched-filter score where
  available.

## Implementation Phases

### Phase 0: Documentation and Provenance

- Maintain explicit injection provenance fields in future spectra/quality
  outputs.
- Keep all old-run readers backward compatible.
- Document dataset schemas before writing training code.

### Phase 1: Dataset Builder

- Implement `tools/build_ml_datasets.py`.
- Build a small dataset from current UCS/Arcturus/Castro Valley runs.
- Verify counts against existing recovery and quality dashboards.
- Add `/ml-datasets` static/live dashboard backed by `dataset_summary.json`.

### Phase 2: Science Model Prototype

- Implement ragged set encoder in PyTorch.
- Train on quality-good/review spectra.
- Write `science_embeddings.parquet`.
- Build nearest-neighbor smoke tests:
  - UCS/SIMP anchors should retrieve plausible red/ultracool-like neighbors.
  - Brightness changes should not dominate `shape_embedding`.
  - Bad spectra should show higher reconstruction error or quality anomaly.
- Add `/ml-embeddings` viewer.

### Phase 3: Narrowband Model Prototype

- Build narrowband dataset from baseline/injected paired runs. Initial dataset:
  `narrowband_line_cv_mega_v0`, with 206,168 target spectra and 3,297
  injection-truth rows.
- Train supervised ragged model with hard negatives. Initial cached baseline:
  `narrowband_line_cv_mega_v0_train8_cached`.
- Compare against current GPU matched-filter scorer:
  - recovery by SNR,
  - false positives per target,
  - missed injections,
  - candidate grade distribution.
- Add `/ml-narrowband` viewer.

The first cached line/no-line training run is documented in
[`narrowband_ml_training_log.md`](narrowband_ml_training_log.md). Its key lesson
is operational: do not train directly from raw parquet point tables for serious
experiments. Build parallel tensor feature caches first, then train from those
shards.

### Phase 4: Scale and Hardening

- Run larger injection campaigns stratified by magnitude, detector, wavelength,
  and strength.
- Add held-out sky/wavelength/source validation.
- Add checkpoint/resume and deterministic split manifests.
- Add training job status snapshots.
- Start testing on large campaign archives.

### Phase 5: Blind Object Discovery

The two-model ML stack should eventually support sources that were not selected
from Gaia or a manual catalog. The next-generation design is documented in
[`next_generation_blind_object_miner.md`](next_generation_blind_object_miner.md).

Core idea:

- start from SPHEREx frames, not a target catalog
- mask/subtract known Gaia/2MASS/WISE sources
- detect residual point-source candidates
- associate detections across frames and visits
- optionally shift-stack over proper-motion vectors, Tycho-Tracker style
- extract spectra for blind candidates using the same spectra schema
- use the science embedding and ultracool/reference anchors to rank brown-dwarf
  and other unusual-object candidates

## Open Questions

- How should source photon noise be modeled and written into injected variance?
- How should repeated visits be grouped so the science model can learn
  same-source cross-visit consistency?
- What is the first accepted representation for decoded flags: bit-vector,
  summary penalties, or both?
- Should the narrowband model consume deterministic matched-filter scores as
  input, or only use them for comparison?
- Which embedding distance should power nearest-neighbor review: cosine,
  Euclidean after normalization, or learned metric?
- How many S-tier candidates per sky area is operationally acceptable?
