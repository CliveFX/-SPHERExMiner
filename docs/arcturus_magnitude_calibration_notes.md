# Arcturus Magnitude Calibration Notes

Date: 2026-06-23

These notes summarize the Arcturus-field Gaia magnitude calibration runs used to decide which stars are worth targeting for SPHEREx aperture spectra.

## Runs

Bright-to-mid calibration:

- Run: `/mnt/niroseti/spherex_cache/runs/arcturus_mag_calibration_g5_16_n100_f220_cwave_gpu_w8`
- Target set: `/mnt/niroseti/spherex_cache/calibration_runs/arcturus_mag_calibration_targets.parquet`
- Stats:
  - `/mnt/niroseti/spherex_cache/runs/arcturus_mag_calibration_g5_16_n100_f220_cwave_gpu_w8/mag_calibration_stats/mag_bin_quality.csv`
  - `/mnt/niroseti/spherex_cache/runs/arcturus_mag_calibration_g5_16_n100_f220_cwave_gpu_w8/mag_calibration_stats/target_quality.csv`

Faint-end calibration:

- Run: `/mnt/niroseti/spherex_cache/runs/arcturus_faint_mag_calibration_g15_20_n120_f220_cwave_gpu_w4`
- Target set: `/mnt/niroseti/spherex_cache/calibration_runs/arcturus_faint_mag_calibration_targets.parquet`
- Stats:
  - `/mnt/niroseti/spherex_cache/runs/arcturus_faint_mag_calibration_g15_20_n120_f220_cwave_gpu_w4/mag_calibration_stats/mag_bin_quality.csv`
  - `/mnt/niroseti/spherex_cache/runs/arcturus_faint_mag_calibration_g15_20_n120_f220_cwave_gpu_w4/mag_calibration_stats/target_quality.csv`

Both runs used science wavelength provenance:

- `wavelength_source = spectral_wcs_CWAVE_CBAND`
- `wavelength_calibration_collection = cal-wcs-v4-2025-254`
- all six detectors represented in the assembled spectra

## Allowance Cut

The current default "allowed target" cut is:

- at least 40 usable nonfatal measurements
- fatal fraction <= 0.6
- median aperture SNR >= 2

This is a practical mining cut, not a final science-quality cut. It is meant to reject sources that are too bright, too faint, too masked, or too unstable for useful spectra.

## Result

Practical Arcturus-field target range:

- Best: `G = 12..18`
- Mixed: `G = 11` and `G = 19`
- Avoid: `G <= 10`, mostly too bright/flagged
- Avoid-ish: `G = 20`, mostly too faint under the current cut

Bright-to-mid run:

| Gaia G bin | Allowed / Total | Notes |
| --- | ---: | --- |
| 16 | 10 / 10 | clean |
| 15 | 10 / 10 | clean |
| 14 | 10 / 10 | clean |
| 13 | 10 / 10 | clean |
| 12 | 10 / 10 | clean |
| 11 | 7 / 10 | usable but mixed |
| 10 | 2 / 10 | mostly too flagged |
| 9 | 0 / 10 | too bright/flagged |
| 8 | 0 / 10 | too bright/flagged |
| 7 | 0 / 8 | too bright/flagged |
| 6 | 0 / 1 | unusable |
| 5 | 0 / 1 | unusable |

Faint-end run:

| Gaia G bin | Allowed / Total | Median SNR | Notes |
| --- | ---: | ---: | --- |
| 15 | 20 / 20 | 38.6 | clean overlap/control |
| 16 | 20 / 20 | 20.5 | clean overlap/control |
| 17 | 20 / 20 | 11.2 | clean |
| 18 | 19 / 20 | 5.9 | mostly clean |
| 19 | 12 / 20 | 3.1 | mixed faint edge |
| 20 | 5 / 20 | 1.0 | mostly below SNR cut |

## Operational Guidance

For Arcturus-region mining, start with:

```bash
--gaia-g-min 12 --gaia-g-max 18
```

If target count is too low, expand cautiously:

```bash
--gaia-g-min 11 --gaia-g-max 19
```

Do not use broad `G=5..17` style cuts for science runs in this region. That range mixes saturated/flagged bright stars with marginal faint stars and makes the spectra viewer look much worse than the actual usable range.

## Variance Field

The SPHEREx Level 2 FITS `VARIANCE` extension is already used by the pipeline.

In the FITS file, `VARIANCE` is the per-pixel variance estimate in `(MJy/sr)^2`. It is the built-in uncertainty/noise model from SPHEREx processing, including Poisson and electronic/read-noise contributions, then calibrated into the same surface-brightness unit system as the image.

Pipeline usage:

- `field_worker.py` loads `IMAGE`, `VARIANCE`, `FLAGS`, and calibration maps.
- `calibration.variance_to_ujy2()` converts the variance image from `(MJy/sr)^2` into `uJy^2` using the SAPM solid-angle map.
- aperture photometry sums per-pixel variance over the aperture to estimate `aperture_flux_unc_uJy`.
- Warp/GPU calibrated aperture photometry receives the same `var_ujy2` image and propagates it.
- injection "find-me" strengths, SNR values, and matched-filter scoring are downstream of these uncertainties.

## GPU Launch Note

Concurrent Warp/CUDA initialization can fail when many field worker threads enter kernel setup at the same time. A staggered launch option was added:

```bash
--field-launch-stagger-sec 0.5
```

This delays field-worker submission without disabling parallel processing after launch. Use `0.5` or `1.0` seconds for GPU runs when starting many workers.
