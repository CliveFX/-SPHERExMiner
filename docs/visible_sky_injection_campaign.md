# Visible-Sky Injection/Recovery Campaign

This is the close-out workflow for the current target-centered SPHEREx miner.
It is intentionally a campaign wrapper around the validated prototype, not the
future frame-scale survey engine.

## Scope

The campaign runner starts from a list of bright sky centers visible from Castro
Valley during June. Before running the miner, it resolves each bright center to a
nearby safe Gaia source, by default `G=12..14`, and uses that Gaia source as the
actual manual target. This avoids centering the run on saturated stars such as
Markab, Vega, Antares, or Arcturus. The science sample is then selected from the
broader Gaia magnitude range.

Current operational defaults:

- Target centers: `configs/castro_valley_june_survey_targets.yaml`
- Actual run anchors: generated Gaia sources in
  `/mnt/niroseti/spherex_cache/campaigns/<campaign>/resolved_gaia_anchor_targets.yaml`
- Safe anchor search: nearby Gaia `G=12..14` within 1 degree of each bright center
- Magnitude cut: Gaia `G=11..16`
- Requested field depth: `500`
- Gaia safety cap per review run: `500` for overnight review; raise only for
  deliberate larger surveys
- Field workers: `24`
- Photometry: GPU aperture plus GPU PSF
- Injection lines: 808, 980, 1064, 1310, 1550, and 2000 nm
- Injection strengths: `1,3,8` find-me sigma for current overnight review
  campaigns; wider threshold campaigns may use `0.5,1,2,3,5,8,12`
- Injection flux cap: `50000 uJy`
- Injection density: `3` targets per line/strength cell
- Blind scan: GPU narrowband detector on raw baseline, raw injected, and
  truth-target raw injected recovery; paired-delta sanity products also written
- Blind grid: `1.0 nm` for current deep campaign
- Blind diagnostic rows: `narrowband_line_scores.parquet` with +/-80 nm around
  retained candidates

## Command

Smoke one target:

```bash
.venv/bin/python tools/run_visible_sky_injection_campaign.py \
  --only-target cvj_arcturus
```

Run the full target list with current defaults:

```bash
.venv/bin/python tools/run_visible_sky_injection_campaign.py
```

Run the current three-part campaign shape, reusing the already-resolved safe
Gaia anchors from the previous campaign:

```bash
tmux new-session -d -s spherex-threepart-v1 \
  'cd /home/clive/dev/NIROSETI_SPHEREx && .venv/bin/python tools/run_visible_sky_injection_campaign.py \
    --campaign-prefix cv_june_g11_16_f500_threepart_v1 \
    --targets /mnt/niroseti/spherex_cache/campaigns/cv_june_g11_16_f500_wideinj/resolved_gaia_anchor_targets.yaml \
    --no-resolve-gaia-anchors \
    --limit-fields 500 \
    --max-gaia-sources 6000 \
    --gaia-g-min 11 \
    --gaia-g-max 16 \
    --max-field-workers 24 \
    --warp-devices cuda:0,cuda:1,cuda:2 \
    --strengths-sigma 0.5,1,2,3,5,8,12 \
    --max-line-flux-uJy 50000 \
    --min-snr 1.5 \
    --blind-grid-step-nm 1.0 \
    --viewer-base-url http://192.168.1.224:8765'
```

Run the current overnight diagnostic campaign shape from bright sky centers
while resolving each one to a safe Gaia anchor:

```bash
tmux new-session -d -s spherex-overnight-diag \
  'cd /home/clive/dev/NIROSETI_SPHEREx && .venv/bin/python tools/run_visible_sky_injection_campaign.py \
    --campaign-prefix cv_june_g11_16_f500_diag_overnight_v1 \
    --targets configs/castro_valley_june_survey_targets.yaml \
    --resolve-gaia-anchors \
    --only-target cvj_regulus \
    --only-target cvj_denebola \
    --only-target cvj_porrima \
    --only-target cvj_spica \
    --only-target cvj_arcturus \
    --only-target cvj_izar \
    --only-target cvj_alphecca \
    --only-target cvj_unukalhai \
    --only-target cvj_antarest \
    --only-target cvj_rasalhague \
    --only-target cvj_vega \
    --only-target cvj_sheltan \
    --only-target cvj_tarazed \
    --only-target cvj_deneb \
    --only-target cvj_sadr \
    --only-target cvj_enif \
    --only-target cvj_scheat \
    --only-target cvj_markab \
    --only-target cvj_fomalhaut \
    --limit-fields 500 \
    --max-gaia-sources 500 \
    --gaia-g-min 11 \
    --gaia-g-max 16 \
    --max-field-workers 24 \
    --warp-devices cuda:0,cuda:1,cuda:2 \
    --strengths-sigma 1,3,8 \
    --max-line-flux-uJy 50000 \
    --min-snr 1.5 \
    --blind-scanner narrowband_gpu \
    --blind-grid-step-nm 1.0 \
    --blind-top-k-per-target 20 \
    --narrowband-min-joint-rho 3.0 \
    --narrowband-diagnostic-line-half-window-nm 80 \
    --narrowband-diagnostic-line-max-rows-per-candidate 201 \
    --viewer-base-url http://192.168.1.224:8765 \
    2>&1 | tee /mnt/niroseti/spherex_cache/campaigns/cv_june_g11_16_f500_diag_overnight_v1/campaign_stdout.log'
```

The command above intentionally skips `cvj_altair`, which previously produced
no measured parent fields in this prototype.

## Per-Target Workflow

For each target anchor, the runner performs:

1. Baseline depth run.
2. Mixed-laser injection plan generation from baseline spectra.
3. FITS-level injection into copied files.
4. Injected depth run using `path_overrides.json`.
5. Raw science blind scan on the baseline run.
6. Raw injected blind scan on the injected run.
7. Focused raw blind scan only on injected truth target IDs.
8. Paired baseline/injected matched-filter classification.
9. Recovery scoring against injection truth.
10. False-positive review manifest generation.

Outputs are resumable. If a stage output exists, the runner skips that stage
unless `--force` is supplied.

The bright center IDs, such as `cvj_markab`, are only used for Gaia-anchor
selection. After resolution, run names include the generated Gaia target ID.

## Output Layout

Main campaign root:

```text
/mnt/niroseti/spherex_cache/campaigns/<campaign>/
```

Important files:

```text
campaign_manifest.json
logs/<target>_baseline.log
logs/<target>_narrowband_baseline_raw.log
logs/<target>_make_plan.log
logs/<target>_inject.log
logs/<target>_injected.log
logs/<target>_narrowband_injected_raw.log
logs/<target>_narrowband_raw_recovery.log
logs/<target>_blind_paired_delta_aperture.log
logs/<target>_blind_paired_delta_psf.log
logs/<target>_blind_paired_delta_joint.log
logs/<target>_classify.log
logs/<target>_score.log
false_positive_reviews/<target>.json
```

Per-target run outputs live under:

```text
/mnt/niroseti/spherex_cache/runs/<campaign>_<target>_baseline/
/mnt/niroseti/spherex_cache/runs/<campaign>_<target>_injected/
```

Injection campaign products live under:

```text
/mnt/niroseti/spherex_cache/injection_campaigns/<campaign>_<target>_mixed_lasers/
```

## Reviewing Candidates And False Positives

Campaign status:

```text
http://192.168.1.224:8765/campaign-status?campaign=<campaign>
```

Science raw candidates:

```text
http://192.168.1.224:8765/candidate-summary?campaign=<campaign>&source=baseline
```

Injected raw QA candidates:

```text
http://192.168.1.224:8765/candidate-summary?campaign=<campaign>&source=injected
```

Paired-delta sanity candidates:

```text
http://192.168.1.224:8765/candidate-summary?campaign=<campaign>&source=paired
```

The recovery scorer writes:

```text
narrowband_detector_raw/narrowband_candidates.parquet
narrowband_detector_raw/narrowband_line_scores.parquet
narrowband_detector_raw/narrowband_detector_summary.json
narrowband_detector_truth/narrowband_candidates.parquet
narrowband_detector_truth/narrowband_recovery.parquet
narrowband_detector_truth/narrowband_line_scores.parquet
narrowband_detector_truth/narrowband_detector_summary.json
recovery_score_mixed_lasers/false_positive_candidates.parquet
recovery_score_mixed_lasers/recovery_summary.json
```

The existing web viewer visualizes these through the injection viewer:

```text
http://127.0.0.1:8765/injections?run=<injected_run_name>&status=candidate
```

Each per-target review JSON contains that URL plus the false-positive parquet
path. The injection viewer shows the injected truth, the target spectrum, and
scorer candidates for the selected target. Candidate rows that are above the
threshold but do not match injection truth are the false positives to inspect.

## What We Built

- Local Gaia parquet querying and Gaia target selection.
- SPHEREx Level 2 FITS field evaluation for manual anchors.
- Proper-motion propagation and vectorized WCS target projection.
- Calibrated aperture photometry and GPU PSF photometry.
- Spectra assembly into Parquet products.
- FITS-level fake narrowband injection using SPHEREx PSF placement.
- Raw blind baseline science scanning.
- Raw blind injected recovery scanning against truth targets.
- Paired baseline/injected matched-filter recovery sanity scoring.
- Web viewers for campaign status, spectra, candidates, and injection/recovery inspection.
- Lightweight JSON status snapshots, replacing SQLite live status.

## What We Learned

- The science path works: spectra can be recovered from SPHEREx frames.
- The current runner is target-centered and parallel across fields.
- GPU occupancy is poor because each field launches relatively small GPU jobs.
- Increasing workers from 24 to 72 did not materially improve throughput and
  increased memory use substantially.
- A `G=11..16` 20x20 degree Arcturus box contains roughly 121k Gaia stars.
- Current output storage for a 20x20 `G=11..16` run is likely on the order of
  15-25 GB per baseline or injected run, excluding shared raw FITS cache.

## What Remains

The next major codebase should be a frame-scale survey engine, not another
incremental extension of this target-centered prototype.

Needed next:

- Survey/chunk scheduler.
- GPU job scheduler with large prepared frame/target batches.
- Durable campaign/chunk manifest and resume model.
- Scalable output partitioning for later spectra rebuilds and candidate search.
- Viewer pagination and server-side filtering for large campaign outputs.
- Injection/recovery campaign handling at survey-chunk scale.
- Later: Kubernetes/Ray/Prefect-or-Dagster deployment model.
