# SPHEREx Fake-Signal Injection and Recovery Plan

## Goal

Build a repeatable injection/recovery benchmark for narrowband point-source excesses in SPHEREx Level 2 frames.

The benchmark should answer three questions:

1. If we inject a fake narrowband source into copied FITS frames, can the normal pipeline recover it?
2. How faint can the injected signal be before recovery fails?
3. At a chosen recovery threshold, how many false positives do we create among non-injected stars?

This is not a detection claim system. It is a calibration and validation system for the search.

## Current Starting Point

Working pieces already exist:

- GPU aperture photometry pipeline.
- Assembled target spectra in `spectra/target_spectra.parquet`.
- FITS-level fake line injector: `tools/inject_fake_line.py`.
- Blink comparator for original/injected FITS cutouts: `tools/blink_injection_viewer.py`.
- Known-good checkpoint tag: `injection-blink-working`.
- Working branch for this effort: `injection-recovery-benchmark`.

The current injector accepts an observed line strength in `uJy` or a `find-me` SNR multiplier. That is the right low-level interface for testing the pipeline. A physical beacon interface should be layered above it.

## High-Level Workflow

### 1. Select Holdout Targets

Choose a target set that is not used for tuning the classifier thresholds.

Start small:

- 5-10 stars.
- 1 wavelength, probably 1064 nm.
- 2-3 injected strengths.
- 20-50 frames.

Then scale:

- 100+ stars.
- multiple wavelengths.
- multiple magnitudes.
- multiple detectors.
- multiple sky densities and crowding regimes.
- edge and non-edge detector positions.

Target metadata to preserve:

- `target_id`
- Gaia `source_id`
- RA/Dec
- propagated RA/Dec at observation epoch
- `phot_g_mean_mag`
- color fields if available, such as `bp_rp`
- proper motion fields
- detector pixel position per frame
- crowding / nearest-neighbor estimate
- edge distance

### 2. Run a Baseline Pass

Run the normal pipeline with no injection.

Save:

- target list
- run config
- spectra parquet
- per-frame measurement rows
- flags
- uncertainties
- live/summary metrics

This baseline is the real noise/artifact distribution for those targets and frames.

### 3. Create an Injection Plan

The injection plan is a manifest of intended fake signals.

Each planned injection should specify:

- target identity
- line wavelength
- line width
- injected strength
- physical beacon parameters if used
- selected frames
- expected spectral response per frame
- output directory
- random seed if any randomized choices are used

The plan should be saved before any FITS files are written.

### 4. Inject Into Copied FITS

Never modify raw cached SPHEREx FITS.

For each injection:

1. Read target rows from the baseline run.
2. Identify every FITS frame associated with the target.
3. Compute the spectral response at the injected line wavelength.
4. Extract the native SPHEREx PSF from the FITS `PSF` extension at the target pixel.
5. Convert desired `uJy` per frame into FITS image units using SAPM.
6. Write copied FITS under an injection output directory.
7. Write `path_overrides.json`.
8. Write `injection_manifest.json`.

### 5. Run the Pipeline on Injected Products

Run the normal photometry pipeline using the same target list, but with FITS path overrides.

The pipeline should not use the injection truth during photometry or candidate scoring.

Inputs:

- baseline target list
- frame list
- path overrides
- normal photometry config

Outputs:

- injected run spectra
- candidate list
- performance metrics
- run summary

### 6. Score Recovery

Join the classifier output against the injection truth manifest.

Metrics:

- recovered / missed for each injected signal
- recovered wavelength error
- recovered amplitude error
- candidate score
- number of independent supporting measurements
- number of false positives on non-injected targets
- false positives per target-spectrum
- false positives per frame
- false positives per wavelength interval

The main product should be a detection-efficiency curve:

```text
injected strength -> recovery fraction at fixed false-positive budget
```

## Unit Model

There are two useful layers of units:

1. Observed injection units used by the pipeline.
2. Physical beacon units used by humans.

The pipeline should remain based on observed units. Physical units should convert into observed units before injection.

## Pipeline Injection Units

The current low-level injector should use:

- line center: `line_nm`
- line width: `line_width_nm`
- peak observed line flux density: `line_flux_uJy`
- optional convenience strength: `find_me_snr`

`line_flux_uJy` means the apparent point-source flux density at the top of the spectral response curve. For a frame with response `R`, the injected frame flux is:

```text
frame_flux_uJy = line_flux_uJy * R
```

That flux is distributed spatially using the native SPHEREx PSF:

```text
pixel_flux_uJy = frame_flux_uJy * psf_pixel_fraction
```

Then SAPM converts `uJy/pixel` into FITS image units.

## Physical Beacon Units

A higher-level physical model can produce `line_flux_uJy` from human-facing parameters.

Core variables:

- `transmitter_power_MW`
- `distance_pc`
- `beam_divergence_arcsec` or `beam_divergence_rad`
- `wavelength_nm`
- `line_bandwidth_Hz` or `line_width_nm`
- `pointing_offset_arcsec`
- `beam_profile`
- `emission_duty_cycle`
- `polarization_factor` if relevant
- `aperture_diameter_m` if modeling a diffraction-limited transmitter
- `transmitter_efficiency`
- `receiver/instrument spectral response`

Important distinction:

- `transmitter_power_MW` is emitted optical power.
- `line_flux_uJy` is observed flux density at Earth.

The conversion is not unique unless the beam geometry and bandwidth are defined.

## Isotropic Reference Case

For an isotropic emitter, bolometric flux at distance `d` is:

```text
F_W_m2 = P_W / (4 * pi * d_m^2)
```

For a narrow line with bandwidth `delta_nu_Hz`, approximate flux density is:

```text
F_nu_W_m2_Hz = F_W_m2 / delta_nu_Hz
F_nu_Jy = F_nu_W_m2_Hz / 1e-26
F_nu_uJy = F_nu_Jy * 1e6
```

This is useful as a sanity check, but it is usually the wrong model for an intentional beacon because intentional beacons are likely beamed.

## Beamed Reference Case

For a beamed transmitter, replace `4*pi` steradians with the beam solid angle `Omega_beam_sr`.

```text
F_W_m2 = P_W / (Omega_beam_sr * d_m^2)
```

For a Gaussian beam, a rough solid-angle approximation from FWHM divergence is:

```text
Omega_beam_sr ~= 1.133 * theta_fwhm_rad^2
```

Then:

```text
F_nu_uJy = (P_W / (Omega_beam_sr * d_m^2 * delta_nu_Hz)) / 1e-26 * 1e6
```

If there is pointing error, apply a beam attenuation factor:

```text
attenuation = exp(-0.5 * (theta_offset_rad / sigma_beam_rad)^2)
```

where:

```text
sigma_beam_rad = theta_fwhm_rad / 2.3548
```

Then:

```text
observed_line_flux_uJy = F_nu_uJy * attenuation * transmitter_efficiency * duty_cycle
```

## Bandwidth and Wavelength Conversion

Line width may be more intuitive in nm, but flux density uses bandwidth in Hz.

For small wavelength widths:

```text
delta_nu_Hz ~= c * delta_lambda_m / lambda_m^2
```

So a fixed `line_width_nm` corresponds to a different Hz bandwidth at different wavelengths.

This matters because a narrower line has higher flux density for the same total emitted power.

## Diffraction-Limited Beam Option

If we model a telescope-like transmitter aperture, beam divergence can be estimated from aperture diameter:

```text
theta_rad ~= 1.22 * lambda_m / aperture_diameter_m
```

Variables:

- `aperture_diameter_m`
- `wavelength_nm`
- beam quality factor, e.g. `beam_quality_m2`

Then:

```text
theta_effective_rad = beam_quality_m2 * 1.22 * lambda_m / aperture_diameter_m
```

This gives a physically motivated beam spread instead of asking for beam divergence directly.

## Recommended Unit Interface

The low-level FITS injector should keep accepting observed units:

```text
--line-nm
--line-width-nm
--line-flux-uJy
```

The benchmark planner should optionally accept physical units:

```text
--transmitter-power-mw
--distance-pc
--beam-divergence-arcsec
--aperture-diameter-m
--line-width-hz
--line-width-nm
--duty-cycle
--pointing-offset-arcsec
```

The planner converts those into:

```text
line_flux_uJy
```

and records both the physical parameters and converted observed units in the injection manifest.

## Injected Signal Ceiling

Before using physical beacon parameters, define an observed-unit ceiling in `line_flux_uJy`.

This should be based on the actual survey products, not only on a speculative transmitter model. The ceiling protects the benchmark from spending too much time on signals that are visually obvious, detector-dominating, or outside the regime where false-positive control matters.

Use three ceilings:

### 1. Visual Debug Ceiling

Purpose: verify that FITS injection, PSF placement, path overrides, and blink viewing work.

Recommended range:

```text
10,000-50,000 uJy
```

This is intentionally huge. It should make the injected source obvious in the blink comparator or difference image. It is not a realistic detection-threshold benchmark.

The first smoke injection used roughly this scale:

```text
line_flux_uJy ~= 44,500 uJy
```

### 2. Recovery Benchmark Ceiling

Purpose: test whether the pipeline and classifier can recover signals without creating unacceptable false positives.

Recommended initial range:

```text
300-5,000 uJy
```

Reasoning from current runs:

- typical unflagged aperture uncertainty: about `150-170 uJy`
- p95 unflagged uncertainty: about `270-290 uJy`
- p99 unflagged uncertainty: about `410-450 uJy`

So a useful first sweep is:

```text
1x, 2x, 3x, 5x, 10x, 20x local uncertainty
```

For a typical target, that maps roughly to:

```text
150, 300, 500, 850, 1,700, 3,400 uJy
```

The benchmark ceiling should initially sit around:

```text
5,000 uJy
```

If everything below that is trivially recovered with low false positives, raise the ceiling. If false positives are uncontrolled at lower levels, do not raise it.

### 3. Physical Plausibility Ceiling

Purpose: convert human beacon parameters into observed `uJy` and reject absurd combinations.

This ceiling should be defined after we implement the physical planner. It depends on:

- transmitter power
- distance
- beam spread
- bandwidth
- duty cycle
- pointing offset

The physical ceiling should not replace the observed-unit ceiling. It should be recorded alongside it.

## Recommended First Strength Sweep

For the first real injection/recovery test, use local-SNR-scaled injections instead of fixed physical power:

```text
find_me_snr = 2, 3, 5, 8, 12, 20
```

This makes the benchmark comparable across bright stars, faint stars, and noisy frames.

For entertainment and visual sanity checks, keep one obvious injection:

```text
find_me_snr = 100-300
```

but exclude those rows from threshold-setting statistics.

## Classifier Inputs

The classifier should not just look for one high point.

Useful features:

- matched-filter response score around the target wavelength
- local continuum residual
- local robust z-score
- number of neighboring channels with expected response shape
- repeated independent hits at consistent wavelength
- detector / edge / flag metadata
- background level
- crowding estimate
- whether the hit appears in multiple frames
- whether nearby stars also show a same-frame spike

## False-Positive Controls

Every injection run should contain many non-injected targets.

False-positive checks:

- same wavelength spike on many targets in same frame
- detector edge or bad flag association
- cosmic ray / artifact-like single-frame behavior
- background structure correlation
- wavelength-dependent detector artifacts
- PSF mismatch or centroid offset
- recurrence only in one detector/channel family

The benchmark should report recovery only alongside the false-positive rate.

## First Implementation Steps

1. Add `tools/make_injection_plan.py`.
2. Add `tools/run_injection_plan.py` or extend `tools/inject_fake_line.py` to consume a plan.
3. Add `tools/score_injection_recovery.py`.
4. Wire path overrides into the main run path if not already supported.
5. Run a tiny benchmark:

```text
10 stars
1 wavelength: 1064 nm
3 strengths: obvious, marginal, below threshold
50 frames max
```

6. Save:

```text
baseline run
injection plan
injection manifest
path overrides
injected run
scorer output
plots
```

## Implemented First-Pass Tools

The first implementation keeps benchmark orchestration in standalone tools and adds only a small path-override hook to the main pipeline.

Tools:

- `tools/make_injection_plan.py`
- `tools/run_injection_plan.py`
- `tools/classify_spectra_matched_filter.py`
- `tools/score_injection_recovery.py`

Pipeline hook:

- `spherex-mine run-benchmark --path-overrides path_overrides.json`
- `spherex-mine run-depth-test --path-overrides path_overrides.json`

Run mode rule:

- No `--path-overrides`: run against the normal cached SPHEREx FITS.
- With `--path-overrides`: run the same target/field configuration against copied injected FITS.

This should remain a permanent switch. Every benchmark run should be reproducible in paired form:

```bash
spherex-mine run-benchmark \
  --run-name baseline_example \
  ...

spherex-mine run-benchmark \
  --run-name injected_example \
  --path-overrides /path/to/injection_campaign/path_overrides.json \
  ...
```

Both runs should write `run_summary.json`. Injected runs should include:

```text
path_overrides_path
path_override_count
```

Important execution rule:

Do not inject multiple strengths for the same target/line into one campaign unless intentionally testing accumulation. A strength sweep should run one strength slice at a time:

```bash
.venv/bin/python tools/run_injection_plan.py \
  --plan /path/to/injection_plan.json \
  --strength-sigma 5
```

The runner refuses to accumulate duplicate target/line strength variants by default.

## End-to-End Roadmap

The near-term destination is a spectra-level classifier/recoverer that can be tested against fake-signal injections and then exercised on several 20k-star runs.

### Phase 0: Preserve the Known-Good Baseline

Status: done.

- Commit current injection and blink tools.
- Tag the checkpoint as `injection-blink-working`.
- Work on branch `injection-recovery-benchmark`.
- Keep raw SPHEREx FITS read-only by convention.

Exit criteria:

- Can inject one obvious fake line into copied FITS.
- Can blink original versus injected cutouts.
- Worktree has a stable checkpoint.

### Phase 1: Plan-Driven Injection

Goal: move from one-off CLI injection to repeatable injection campaigns.

Build:

- `tools/make_injection_plan.py`
- plan schema in JSON
- target selector from an existing baseline run
- line-family grid support
- strength sweep support

Plan dimensions:

```text
targets x line_families x wavelength_offsets x strengths
```

Example first grid:

```text
targets: 10
line family: 1064 nm
offsets: -20, -10, -5, 0, +5, +10, +20 nm
strengths: 2, 3, 5, 8, 12, 20 sigma
```

Exit criteria:

- A saved plan fully describes what will be injected before FITS files are copied.
- Every injection has a stable ID.
- The plan records both observed units and any physical beacon parameters.

### Phase 2: Batch FITS Injection

Goal: execute a whole injection plan into copied FITS products.

Build:

- `tools/run_injection_plan.py`, or equivalent plan mode in `tools/inject_fake_line.py`
- merged `path_overrides.json`
- aggregate `injection_manifest.json`
- per-injection frame manifests
- resume/skip support

Important behavior:

- Multiple injections into the same FITS copy must accumulate correctly.
- Output FITS should be grouped by injection campaign, not mixed into raw cache.
- The manifest must preserve ground truth for every injected target/wavelength/strength.

Exit criteria:

- Can inject a 10-star plan without modifying raw FITS.
- Can inspect several injected products in the blink viewer.
- Can feed the generated path overrides into a pipeline run.

### Phase 3: Pipeline Override Run

Goal: run normal photometry on injected FITS without teaching the photometry code about ground truth.

Build or verify:

- path override support in the main pipeline
- run metadata that records the override file
- output spectra parquet with source FITS path or original FITS ID preserved
- summary showing baseline versus injected run identity

Exit criteria:

- Same target/frame plan can be run baseline and injected.
- Injected run outputs spectra normally.
- Each spectral point can be traced to target, frame, wavelength, and FITS path.

### Phase 4: Spectra-Level Matched-Filter Classifier

Goal: score assembled spectra for narrowband excesses with the expected SPHEREx response shape.

This first classifier should operate on `spectra/target_spectra.parquet`, not directly on FITS.

Inputs:

- `target_id`
- `cwave_um`
- `cband_um`
- `aperture_flux_uJy`
- `aperture_flux_unc_uJy`
- flags
- detector
- image/frame ID
- x/y pixel
- edge distance
- optional background/crowding metadata

Core algorithm:

1. Sort each target spectrum by wavelength.
2. Mask fatal-flagged points by default.
3. Estimate a robust local continuum.
4. Compute residuals:

```text
residual_uJy = aperture_flux_uJy - local_continuum_uJy
```

5. For each candidate line wavelength, build the expected response template from nearby `cwave_um` and `cband_um`.
6. Compute a weighted matched-filter amplitude and score.
7. Record the best candidate peaks per target.

Initial output columns:

- `target_id`
- `candidate_line_nm`
- `line_family`
- `score`
- `matched_flux_uJy`
- `matched_flux_unc_uJy`
- `matched_snr`
- `n_supporting_points`
- `n_flagged_nearby`
- `best_frame_ids`
- `detectors`
- `local_continuum_uJy`
- `local_residual_rms_uJy`
- `candidate_status`

Exit criteria:

- Classifier finds the obvious debug injection.
- Classifier recovers moderate fake injections in a tiny run.
- Classifier output is a compact parquet/JSON table that the scorer can consume.

### Phase 5: Injection Recovery Scorer

Goal: compare classifier candidates against injection truth.

Build:

- `tools/score_injection_recovery.py`
- truth join between `injection_manifest.json` and classifier output
- wavelength tolerance handling
- line-family tolerance handling
- false-positive accounting on non-injected targets

Scorer outputs:

- per-injection recovered/missed table
- per-strength recovery fraction
- false positives per target-spectrum
- false positives per wavelength family
- wavelength error distribution
- score threshold curves
- recommended operating thresholds

Key distinction:

```text
recovery = did we find the injected truth?
false positive = did we flag something without injected truth?
```

Exit criteria:

- Can report recovery versus strength for a 10-star injection plan.
- Can report false positives among non-injected stars in the same run.
- Can produce a machine-readable summary for comparing runs.

### Phase 6: Small Calibration Runs

Goal: debug the full benchmark loop before spending time on large runs.

Run sequence:

1. Baseline run:

```text
10 injected-candidate stars + 100 non-injected control stars
50-100 fields
```

2. Injected run:

```text
same targets
same fields
same photometry settings
path overrides enabled
```

3. Classifier run.
4. Scorer run.

Exit criteria:

- Obvious injections recover near 100%.
- Low-strength injections fall off smoothly.
- False positives are measurable, not hand-waved.
- Any bad wavelength families or detector regions are visible.

### Phase 7: First 20k-Star Runs

Goal: measure false positives and recovery behavior at a scale large enough to be meaningful.

Run A: baseline 20k-star control

```text
targets: 20,000 Gaia/SPHEREx stars
injections: none
purpose: false-positive baseline
```

Run B: sparse injected 20k-star run

```text
targets: same 20,000 stars
injected targets: 100-500
line families: 1064 nm first
strengths: 3, 5, 8, 12, 20 sigma
purpose: recovery versus false positives
```

Run C: wavelength-family stress run

```text
targets: 20,000 stars
injected targets: 100-500
line families: several predicted laser lines
offsets around each family
purpose: identify artifact-prone wavelengths
```

Candidate line families to include after 1064 nm:

- 1064 nm
- 1550 nm
- 532 nm if usable in SPHEREx coverage/products
- other literature/predicted laser lines after review

Exit criteria:

- Classifier throughput is acceptable.
- False-positive rate is quantified at 20k-star scale.
- Injection recovery curves exist for at least one line family.
- We can decide whether to scale to full-survey mining.

### Phase 8: Threshold and Triage Viewer

Goal: make candidate review fast enough that false positives are manageable.

Build:

- spectra candidate viewer
- link from candidate to spectrum plot
- link from candidate to contributing FITS frames
- blink/difference view for injected validation cases
- per-candidate metadata card

Triage fields:

- target ID
- candidate wavelength
- score/SNR
- line family
- number of supporting points
- flags
- detectors/frames
- local continuum plot
- matched-filter template overlay

Exit criteria:

- Can review top candidates from a 20k-star run without hand-opening files.
- Can quickly distinguish injected truth, ordinary stellar spectra, and obvious artifacts.

## Minimal Classifier Definition

The first classifier should be intentionally simple and auditable.

For a target and candidate wavelength:

```text
y_i = aperture_flux_uJy_i
sigma_i = aperture_flux_unc_uJy_i
c_i = robust local continuum estimate
r_i = y_i - c_i
t_i = expected SPHEREx response at candidate wavelength
```

Weighted matched-filter amplitude:

```text
A = sum(t_i * r_i / sigma_i^2) / sum(t_i^2 / sigma_i^2)
```

Uncertainty:

```text
sigma_A = 1 / sqrt(sum(t_i^2 / sigma_i^2))
```

Score:

```text
matched_snr = A / sigma_A
```

This is not the final statistical model. It is the first recoverer because it is transparent, cheap, and directly tied to the injection template.

## 20k-Run Acceptance Criteria

Before running larger than 20k stars, require:

- recovery curves for injected strengths
- false-positive rate from a no-injection control run
- top-candidate triage output
- runtime and throughput summary
- known artifact notes
- reproducible run manifests

Do not scale to full survey until the no-injection 20k-star run has a tolerable false-positive rate or at least a clear path to reducing it.

## Open Questions

- Should the physical model default to beamed or isotropic? Beamed is more realistic for an intentional beacon, isotropic is safer as a reference unit.
- Should `line_flux_uJy` mean peak flux density or integrated line flux? Current tool uses peak apparent flux density.
- Should line width default to Hz or nm? Hz is physically cleaner for power conversion; nm is easier for SPHEREx response thinking.
- Should injected strength sweeps be linear in power, flux density, or SNR?
- How much recurrence is possible for a given target in current SPHEREx coverage?
- What false-positive budget is acceptable per million target-spectra?
