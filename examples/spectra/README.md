# Example Spectra

Small spectra examples for documentation, UI development, and quick plotting.

These are not canonical science products. They are median-binned extracts from
local campaign runs with fatal-flagged measurements removed. The full source
products remain in `/mnt/niroseti/spherex_cache/runs/.../spectra/`.

Files:

- `ucs_0972_gpu_psf_sample.csv`: ultracool brown dwarf/manual target sample from
  a GPU aperture+PSF run.
- `arcturus_anchor_threepart_baseline_sample.csv`: safe Gaia anchor near
  Arcturus from the three-part campaign baseline run.
- `manifest.json`: source run paths, target IDs, and columns.

Columns:

- `target_id`
- `wavelength_bin_um`
- `cwave_um`
- `cband_um`
- `aperture_flux_uJy`
- `aperture_flux_unc_uJy`
- `psf_flux_uJy`
- `psf_flux_unc_uJy`
- `measurement_count`
