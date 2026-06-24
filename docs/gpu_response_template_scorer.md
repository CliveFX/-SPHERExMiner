# GPU Response-Template Scorer

The current implementation is:

```text
tools/warp_narrowband_detector.py
```

It is the primary raw blind candidate generator for new campaigns. It supersedes
the older `blind_classifier_*_warp` raw scanner for science candidate review.
The older paired-delta scanner remains useful as an optimistic injection sanity
check because it subtracts baseline from injected spectra.

## Inputs

The scorer reads:

```text
/mnt/niroseti/spherex_cache/runs/<run_name>/spectra/target_spectra.parquet
```

Required columns include:

```text
target_id
cwave_um
cband_um
aperture_flux_uJy
aperture_flux_unc_uJy
psf_flux_uJy
psf_flux_unc_uJy
```

Science runs should use wavelength source:

```text
spectral_wcs_CWAVE_CBAND
```

The scorer refuses approximate wavelength products unless
`--allow-approx-wavelengths` is explicitly passed.

## What It Computes

For each selected target and each trial wavelength:

1. Estimate local continuum from training cells outside the template guard.
2. Build a Gaussian response template from `cwave_um`, `cband_um`, and requested
   line width.
3. Compute aperture and PSF matched-filter amplitudes and SNRs independently.
4. Joint-rank candidates by the weaker of aperture and PSF evidence.
5. Apply a Gross-Vitells style look-elsewhere correction using the measured
   up-crossing count.
6. Emit sparse top-K candidates, not a dense score cube.

Current response model hash:

```text
gaussian_cwave_cband_v1
```

## Outputs

For a normal raw scan:

```text
narrowband_detector_raw/narrowband_candidates.parquet
narrowband_detector_raw/narrowband_line_scores.parquet
narrowband_detector_raw/narrowband_detector_summary.json
```

For focused injected truth recovery:

```text
narrowband_detector_truth/narrowband_candidates.parquet
narrowband_detector_truth/narrowband_recovery.parquet
narrowband_detector_truth/narrowband_line_scores.parquet
narrowband_detector_truth/narrowband_detector_summary.json
```

`narrowband_line_scores.parquet` is a compact debug table around retained
candidates. It is used by the blind candidate browser to show aperture and PSF
line-score curves. It is intentionally bounded by:

```text
--diagnostic-line-half-window-nm
--diagnostic-line-max-rows-per-candidate
```

Do not write the full target x wavelength score cube for normal runs.

## Manual Command

```bash
.venv/bin/python tools/warp_narrowband_detector.py \
  --run-dir /mnt/niroseti/spherex_cache/runs/<run_name> \
  --output-dir /mnt/niroseti/spherex_cache/runs/<run_name>/narrowband_detector_raw \
  --grid-step-nm 1.0 \
  --min-joint-rho 3.0 \
  --top-k-per-target 20 \
  --device cuda:0 \
  --quality-min-support 3 \
  --quality-max-flagged-points 3 \
  --quality-max-candidates-per-target 5 \
  --quality-max-aperture-psf-ratio 3.0 \
  --diagnostic-line-half-window-nm 80 \
  --diagnostic-line-max-rows-per-candidate 201
```

For truth-target injected recovery, add:

```bash
--manifest /mnt/niroseti/spherex_cache/injection_campaigns/<campaign_target>_mixed_lasers_s1_3_8/injection_manifest.json \
--target-ids-file /mnt/niroseti/spherex_cache/runs/<injected_run>/blind_raw_recovery_truth_target_ids.txt
```

## Review

Use:

```text
http://192.168.1.224:8765/blind-candidates?run=<run_name>&scope=raw
```

The plot at the bottom of the blind candidate browser uses
`narrowband_line_scores.parquet` when available. If the page says no line score
rows are saved, rerun the scorer with the diagnostic line-score flags above.

## Remaining Gaps

- Noise modeling for FITS injection still needs source variance/random source
  noise.
- Gross-Vitells `N0` should be calibrated per detector/band on larger baseline
  archives.
- Source-frame recurrence is not yet the hard promotion gate it should become.
- The target-centered scheduler still underfeeds the GPUs compared with a future
  frame-scale batch scheduler.
