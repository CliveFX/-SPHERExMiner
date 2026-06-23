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

Defaults:

- Target centers: `configs/castro_valley_june_survey_targets.yaml`
- Actual run anchors: generated Gaia sources in
  `/mnt/niroseti/spherex_cache/campaigns/cv_june_g11_16_f500/resolved_gaia_anchor_targets.yaml`
- Safe anchor search: nearby Gaia `G=12..14` within 1 degree of each bright center
- Magnitude cut: Gaia `G=11..16`
- Requested field depth: `500`
- Gaia safety cap per run: `6000`
- Field workers: `24`
- Photometry: GPU aperture plus GPU PSF
- Injection lines: 808, 980, 1064, 1310, 1550, and 2000 nm
- Injection strengths: `5,8,12` find-me sigma
- Injection density: `3` targets per line/strength cell

## Command

Smoke one target:

```bash
.venv/bin/python tools/run_visible_sky_injection_campaign.py \
  --only-target cvj_arcturus
```

Run the full 20-anchor list:

```bash
.venv/bin/python tools/run_visible_sky_injection_campaign.py
```

Run in a background tmux session:

```bash
tmux new-session -d -s cv-june-campaign \
  '.venv/bin/python tools/run_visible_sky_injection_campaign.py'
```

## Per-Target Workflow

For each target anchor, the runner performs:

1. Baseline depth run.
2. Mixed-laser injection plan generation from baseline spectra.
3. FITS-level injection into copied files.
4. Injected depth run using `path_overrides.json`.
5. Paired baseline/injected matched-filter classification.
6. Recovery scoring against injection truth.
7. False-positive review manifest generation.

Outputs are resumable. If a stage output exists, the runner skips that stage
unless `--force` is supplied.

The bright center IDs, such as `cvj_markab`, are only used for Gaia-anchor
selection. After resolution, run names include the generated Gaia target ID.

## Output Layout

Main campaign root:

```text
/mnt/niroseti/spherex_cache/campaigns/cv_june_g11_16_f500/
```

Important files:

```text
campaign_manifest.json
logs/<target>_baseline.log
logs/<target>_make_plan.log
logs/<target>_inject.log
logs/<target>_injected.log
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

## Reviewing False Positives

The recovery scorer writes:

```text
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
- Paired baseline/injected matched-filter recovery scoring.
- Web viewers for spectra and injection/recovery inspection.
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
