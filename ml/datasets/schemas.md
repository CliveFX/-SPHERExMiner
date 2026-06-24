# ML Dataset Schemas

This file defines the planned dataset outputs for the two-model ML stack.

## Science Dataset

Target table:

```text
science_targets.parquet
  dataset_name
  split_id
  run_name
  run_kind
  target_id
  source_id
  object_name
  spectrum_quality_score
  spectrum_quality_category
  n_measurements
  n_usable_measurements
  phot_g_mean_mag
  phot_bp_mean_mag
  phot_rp_mean_mag
  bp_rp
  parallax_mas
  pmra_masyr
  pmdec_masyr
  ruwe
```

Point table:

```text
science_points.parquet
  dataset_name
  split_id
  run_name
  target_id
  point_index
  cwave_um
  cband_um
  aperture_flux_uJy
  aperture_flux_unc_uJy
  psf_flux_uJy
  psf_flux_unc_uJy
  fatal_flag_present
  flags_summary
  detector
  wavelength_detector
  image_id
  observation_id
  obs_mid_time
  x_pix
  y_pix
  edge_distance_pix
```

## Narrowband Dataset

Target and point tables should include all science fields plus:

```text
run_injection_applied
point_injection_applied
injection_applied
injection_manifest_path
is_injected_target
is_injected_measurement
injection_id
line_family
injected_line_nm
line_width_nm
find_me_snr
line_flux_uJy
frames_written
is_recovered
matched_candidate_line_nm
matched_snr
matched_filter_score
```

## Split Rules

Splits must be assigned by source/target and campaign region, not by individual
point rows. This prevents train/test leakage from multiple injected copies of
the same target.
