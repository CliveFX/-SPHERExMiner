# Blind Candidate Quality Scoring

The blind scanner intentionally produces a broad candidate set. The quality
scorer is a second triage layer that ranks or rejects obvious artifacts before
human review. It does not change the matched-filter math and it does not declare
a discovery.

## Files

- Config: `configs/blind_candidate_quality.yaml`
- Standalone scorer: `tools/score_blind_candidate_quality.py`
- Joint ranker integration: `tools/rank_blind_candidates.py`

`rank_blind_candidates.py` applies quality scoring by default and writes the
quality columns into `blind_joint_candidates.parquet`.

## Output Columns

The scorer adds:

- `quality_score`: weighted score for review ordering.
- `quality_pass`: true only when all configured hard cuts pass.
- `quality_category`: `high_confidence`, `review`, or `reject`.
- `reject_reasons`: comma-separated hard-cut failures.
- `quality_detector_count`: detector count parsed from `detectors`.
- `quality_frame_count`: frame count parsed from `best_frame_ids`.
- `quality_support_min`: minimum aperture/PSF support when both exist.
- `quality_flux_ratio_ok`: true when PSF/aperture flux ratio is in range.

## Default Hard Cuts

The defaults are conservative for raw uninjected science scans:

- Require aperture and PSF to both detect the candidate.
- Require `flagged_points_sum <= 1`.
- Require aperture and PSF support of at least 3 points each.
- Require at least 2 frames.
- Require `0.3 <= flux_ratio_psf_aperture <= 3.0`.
- Reject pathologically high aperture or PSF SNR above 1000.
- Reject broad clusters above 120 nm.

These defaults are meant to suppress detector artifacts, PSF normalization
failures, isolated bad pixels, and single-method false positives.

## Standalone Use

```bash
.venv/bin/python tools/score_blind_candidate_quality.py \
  --input /path/to/blind_joint_candidates.parquet \
  --output /path/to/blind_joint_candidates_scored.parquet \
  --config configs/blind_candidate_quality.yaml
```

## Integrated Use

```bash
.venv/bin/python tools/rank_blind_candidates.py \
  --aperture-dir /path/to/blind_classifier_aperture_warp \
  --psf-dir /path/to/blind_classifier_psf_warp \
  --output-dir /path/to/blind_classifier_joint_warp \
  --quality-config configs/blind_candidate_quality.yaml
```

To keep old behavior:

```bash
.venv/bin/python tools/rank_blind_candidates.py ... --no-quality-score
```

## Tuning

Edit `configs/blind_candidate_quality.yaml`.

Use hard cuts when a condition should reject a candidate regardless of score.
Use weights when a condition should change ordering without necessarily
rejecting the candidate.

Good first tuning knobs:

- `max_flagged_points_sum`
- `flux_ratio_psf_aperture_min`
- `flux_ratio_psf_aperture_max`
- `max_psf_peak_snr`
- `max_aperture_peak_snr`
- `min_aperture_support`
- `min_psf_support`
- `max_cluster_width_nm`

The viewer should default to showing `quality_pass=true` candidates once scored
joint files are available, with rejected candidates still accessible for
debugging.
