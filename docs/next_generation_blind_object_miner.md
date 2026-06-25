# Next-Generation Blind Object Miner

The current LuxQuarry/SPHEREx miner is catalog-driven:

```text
Gaia or manual target list -> WCS target projection -> aperture/PSF photometry -> spectra -> scoring/viewers
```

That is the right architecture for known Gaia stars, SIMP/UCS anchors, injection
tests, and narrowband signal recovery. It is not enough for discovering objects
that are missing, faint, high-proper-motion, or poorly represented in Gaia.

This note captures a future pipeline family for image-space target discovery,
especially ultracool brown dwarfs and other red/faint/moving sources.

## Goal

Build an object detector that starts from SPHEREx frames rather than from a
catalog. It should find candidate point sources, build spectra for them, and
rank objects that look like ultracool dwarfs or other unusual astrophysical
sources.

This is a separate next-generation system, not a replacement for the current
catalog miner.

## Motivation

Known-catalog mining gives clean target coordinates but inherits catalog
selection limits. Brown dwarfs and related objects may be:

- too faint for Gaia in optical bands
- very red, with most useful flux in IR bands
- high proper-motion, so catalog positions can be stale or mismatched
- blended or missing from Gaia but visible in SPHEREx/WISE-like IR data
- discoverable only after coadding or stacking multiple frames

The current pipeline can classify and compare recovered spectra, but it does
not yet discover new source positions from the images themselves.

## Tycho-Tracker-Inspired Direction

Tycho Tracker searches for moving Solar System objects by shifting and stacking
images over a grid of possible motion vectors. A real moving object adds
coherently along the correct path while static sources and noise smear out.

For SPHEREx, the analogous idea is:

```text
SPHEREx frames
  -> mask/subtract known catalog sources
  -> search residual point sources
  -> optionally shift-stack over proper-motion vectors
  -> extract spectra
  -> compare to ultracool/reference embeddings
```

This is "inverse Tycho" in the sense that we are not only looking for fast
moving asteroid streaks; we are looking for faint red source tracks or static
IR sources that survive source masking and spectral consistency tests.

## Pipeline Sketch

### 1. Frame Ingestion

For a sky region:

- group SPHEREx frames by sky footprint, detector, visit, and wavelength band
- read image, variance, flags, WCS, calibrated wavelength metadata
- carry detector/camera/visit provenance for every detection

### 2. Static-Sky and Catalog Masking

Create a background/source model:

- project Gaia/2MASS/WISE/known bright-source masks into each frame
- mask saturated, bloomed, ghosted, persistent, edge, and high-flag pixels
- estimate local background and variance
- optionally subtract a PSF model for known bright catalog stars

Output:

```text
residual_frame = image - background - known_source_model
```

### 3. Residual Source Detection

Run point-source detection on residual frames:

- matched filter with SPHEREx PSF where available
- local SNR thresholding using variance products
- multi-band consistency checks
- reject detections near flags, edges, ghosts, and bright-star residuals

Candidate rows should include:

```text
candidate_id
frame_id / image_id / detector / camera
ra, dec
x_pix, y_pix
wavelength/cband
flux, flux_unc
local_background
flag_summary
source_mask_distance
psf_score
```

### 4. Cross-Frame Association

Associate detections into source hypotheses:

- static source hypothesis: same sky position across frames/bands
- proper-motion hypothesis: position evolves linearly over observation time
- reject one-frame artifacts unless independently repeated

This stage should produce:

```text
object_candidate_id
ra0, dec0
pmra_candidate, pmdec_candidate
detection_count
band_count
visit_count
fit_chi2 / association_score
```

### 5. Motion-Stack Search

For high-proper-motion or marginal objects:

- grid over proper-motion vectors
- shift residual frames into a candidate rest frame
- stack weighted by variance and PSF response
- score point-source likelihood in the stack

This can be GPU accelerated because the work is regular:

```text
frames tensor + variance tensor + WCS/time metadata + motion grid
  -> GPU shift/sample/accumulate
  -> likelihood cube over sky position and motion vector
```

### 6. Spectral Extraction

Once a candidate source position/path exists:

- run aperture and PSF photometry using the same calibrated SPHEREx metadata
- build a ragged spectrum with the same schema as catalog-mined targets
- include `target_type = blind_object_candidate`
- include discovery provenance and association/motion scores

The existing spectra viewer, quality scorer, embedding exporter, UMAP viewer,
and narrowband detector should work if the output schema is aligned.

### 7. Ultracool Candidate Scoring

Rank candidates using:

- similarity to known SIMP/UCS/L/T/Y dwarf spectra
- science embedding nearest neighbors
- red continuum / IR-heavy flux ratios
- broad molecular-absorption-like shape features
- Gaia non-match or faint/red Gaia match
- WISE/2MASS crossmatch where available
- proper-motion/parallax plausibility where measurable
- spectrum quality and artifact rejection

Outputs:

```text
blind_object_candidates.parquet
blind_object_spectra.parquet
ultracool_candidate_scores.parquet
motion_stack_diagnostics.parquet
```

## GPU Architecture

The long-term implementation should be frame-scale and GPU-first:

```text
CPU:
  select frame groups
  read FITS/image/variance/flags
  prepare WCS/time/source-mask metadata

GPU:
  background-normalized residual tensors
  PSF/matched-filter convolution
  source likelihood maps
  shift-stack over motion vectors
  candidate peak extraction

CPU/GPU:
  candidate association
  parquet shard writing
```

The current catalog miner underfeeds the GPUs because it is target/frame
oriented. Blind frame-scale detection should batch entire exposures and motion
grids to keep the GPUs occupied.

## Validation Plan

1. **Known-object recovery**
   - Verify that SIMP/UCS and known ultracool dwarfs are rediscovered when in
     frame.
   - Compare recovered spectra with catalog-driven photometry.

2. **Negative controls**
   - Run on regions dominated by normal Gaia stars and known artifacts.
   - Measure false residual-source rate after masks.

3. **Injection tests**
   - Inject synthetic red/cold point-source spectra at FITS level.
   - Vary magnitude, color, background, flags, detector, and motion.
   - Test static and proper-motion recovery.

4. **Cross-survey confirmation**
   - Crossmatch candidates against Gaia, 2MASS, WISE, CatWISE, and known brown
     dwarf catalogs.
   - Promote only candidates with reproducible spectra and plausible external
     constraints.

## Relationship to Existing Work

Existing system:

- catalog target selection
- aperture/PSF spectra
- FITS-level narrowband injection
- GPU photometry/scoring experiments
- science embeddings and UMAP browsing
- candidate/recovery dashboards

Future blind object miner:

- image-space source discovery
- static residual source detection
- motion-stack search
- blind object spectra
- ultracool/brown-dwarf candidate mining

The two systems should converge at the spectra schema. If blind candidates emit
the same target/point parquet fields as catalog targets, the viewers, quality
scorers, embeddings, and signal detectors can be reused.

## Open Questions

- Which SPHEREx bands/visits are best for first-pass ultracool discovery?
- How aggressive can bright-source subtraction be without creating false
  residual candidates?
- What motion-vector grid is needed for nearby brown dwarfs over SPHEREx time
  baselines?
- Should the first implementation use WISE/CatWISE priors, or be fully blind?
- Can PSF photometry and motion stacking share a common GPU kernel bank?
- How should candidate promotion combine spectrum similarity, motion evidence,
  non-detection in Gaia, and external survey crossmatches?
