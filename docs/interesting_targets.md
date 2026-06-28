# Interesting Targets

Targets in this file are not claimed detections. They are objects worth
revisiting because the pipeline or manual inspection found unusual structure,
good diagnostic value, or a useful failure mode.

## 2MASS J20105039+3322233

- Pipeline target id: `twomass_psc_20105039+3322233`
- Position: RA `302.709983 deg`, Dec `+33.373142 deg`
- Galactic position: approximately `l=71.166 deg`, `b=-0.088 deg`
- Context: crowded Galactic-plane field.
- Run: `deep_grid_test_all_catalog_mid_g11_16_hpx_nside0016_nested_00000912_b0008_baseline`
- Candidate line: approximately `4502 nm`
- Candidate tier in the raw narrowband detector: `S_go_look`
- FITS inspector:
  - `http://192.168.1.224:8776/?run=deep_grid_test_all_catalog_mid_g11_16_hpx_nside0016_nested_00000912_b0008_baseline&target=twomass_psc_20105039%2B3322233&line_nm=4502.0`
  - line-driver frame index: `160`
  - line-driver image id: `level2_2025W19_1B_0217_1D6_spx_l2b-v20-2025-241`
  - raw peak-flux frame index: `79`

### Catalog Cross-Matches

All matches below are within about `0.15 arcsec` of the 2MASS position.

| Catalog | Identifier | Notes |
| --- | --- | --- |
| 2MASS PSC | `20105039+3322233` | `J=13.914`, `H=12.564`, `Ks=12.039`, `ph_qual=AAA`, `cc_flg=000` |
| AllWISE | `J201050.40+332223.2` | `W1=11.647`, `W2=11.668`, `W3=10.049`, `W4=7.804`; `ph_qual=AAAB`; `cc_flags=0hh0` |
| CatWISE2020 | `J201050.39+332223.2` | `W1=11.603`, `W2=11.664`; high W1/W2 SNR |
| Gaia DR3 | `2055231534351389056` | `G=18.777`, `BP=21.412`, `RP=17.288`, `RUWE=1.043` |

Gaia DR3 astrometry:

```text
pmra     = -3.20 +/- 0.16 mas/yr
pmdec    = -4.28 +/- 0.21 mas/yr
parallax =  0.323 +/- 0.176 mas
```

Gaia DR3 does not provide XP or RVS spectra for this source:

```text
has_xp_continuous = false
has_xp_sampled    = false
has_rvs           = false
```

### SPHEREx Candidate Notes

The broad recovered SPHEREx spectrum has good mechanical quality by our current
spectra-quality pass:

- measurements: `185`
- usable measurements: `173`
- wavelength coverage: approximately `0.750-4.998 um`
- aperture/PSF agreement: approximately `0.986`

The 4502 nm event is driven by a line-compatible frame rather than by the raw
maximum-flux frame:

```text
line-driver frame index: 160
image id: level2_2025W19_1B_0217_1D6_spx_l2b-v20-2025-241
cwave: 4.503862 um
detector: 6
x/y: 533.272, 1666.329
aperture flux: about 8074 uJy
PSF flux: about 7498 uJy
aperture SNR: about 23
response to 4502 nm: about 0.99
row fatal flag: false
flags_summary: 2097152
```

The `2097152` flag is the source-mask bit, not one of the fatal/problem bits
used by the miner. The standalone FITS inspector was updated so red overlay
dots mark only fatal/problem flag pixels, while source-mask pixels are counted
separately.

### Current Interpretation

This remains an interesting candidate, not a detection. Plausible explanations
include:

- real narrowband/transient emission at the target sky position
- cosmic ray or local detector transient not caught by the current fatal flags
- crowded-field or Galactic-plane background artifact
- persistence/ghost/scattered-light effect not obvious from the first cutout
  inspection
- ordinary astrophysical transient or background event

The reason this object is worth keeping is that a true narrowband emitter could
look like a small number of line-response-compatible SPHEREx measurements rather
than a smooth continuum feature. The key follow-up is recurrence: check whether
the same sky position produces a compatible excess at the same wavelength in
independent visits, detectors, or future data releases.

### Follow-Up Checklist

- Inspect all line-response support frames in the standalone FITS inspector.
- Compare nearby stars in the same line-driver FITS frame for similar excesses.
- Add a same-frame artifact scan around the line-driver exposure.
- Check recurrence at the same sky position and same wavelength response in
  later SPHEREx products.
- Query broader spectroscopic archives for external spectra or classifications.
- Keep this target in regression tests for the narrowband detector and the FITS
  frame inspector.

## 2MASS J17585426-1109157

- Pipeline target id: `twomass_psc_17585426-1109157`
- Position: RA `269.726109 deg`, Dec `-11.154362 deg`
- Galactic position: approximately `l=16.940 deg`, `b=+6.326 deg`
- Context: Scutum/Aquila grid run, above the Galactic plane.
- Run: `cv_grid_g11_16_allcat_f500_n20000_inj_night1_mid_g11_16_hpx_nside0016_nested_00001843_b0010_baseline`
- Candidate line: approximately `4525 nm`
- Candidate tier in the raw narrowband detector: `S_go_look`
- FITS inspector:
  - `http://192.168.1.224:8776/?run=cv_grid_g11_16_allcat_f500_n20000_inj_night1_mid_g11_16_hpx_nside0016_nested_00001843_b0010_baseline&target=twomass_psc_17585426-1109157&line_nm=4525`
  - line-driver frame index: `137`
  - line-driver image id: `level2_2026W13_1A_0290_1D6_spx_l2b-v24-2026-088`
  - raw peak-flux frame index: `51`

### SPHEREx Candidate Notes

This is a second S-tier baseline candidate near the same broad wavelength region
as `2MASS J20105039+3322233`, but in a different part of the sky.

```text
line-driver frame index: 137
image id: level2_2026W13_1A_0290_1D6_spx_l2b-v24-2026-088
cwave: 4.529461 um
detector: 6
x/y: 339.629, 1590.253
aperture flux: about 5191 uJy
PSF flux: about 3486 uJy
row fatal flag: false
flags_summary: 2097152
candidate rank score: about 16.35
aperture SNR: about 16.35
PSF SNR: about 19.14
support count: 5
flagged points near candidate: 0
```

### Current Interpretation

This target is interesting, but it is also part of a developing detector/wavelength
systematics pattern:

- candidate wavelength is around `4.5 um`
- line-driver frame is on detector `6`
- detector `6` visually appears snowy in the FITS inspector
- the previous interesting `4.5 um` candidate was also detector `6`
- row fatal flag is false, but `flags_summary=2097152` source-mask bookkeeping
  is present

This should be treated as a detector-6/wavelength-neighborhood candidate under
systematics review, not as a clean signal. It is useful because it may define a
quality penalty or recurrence requirement for detector `6` near `4.5 um`.

### Follow-Up Checklist

- Build a baseline candidate histogram by detector and wavelength.
- Check whether S/A/B candidates cluster near `4.4-4.7 um` on detector `6`.
- Compare same-frame neighboring stars in the line-driver FITS image.
- Require stronger recurrence before promoting detector-6 `4.5 um` candidates.
- Keep this as a regression target for the detector/wavelength systematics
  dashboard.
