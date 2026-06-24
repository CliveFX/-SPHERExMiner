from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "blind_candidate_quality.yaml"


def _num(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _split_csv(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [part for part in str(value).split(",") if part]


def load_quality_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def score_joint_candidates(joint: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if joint.empty:
        out = joint.copy()
        out["quality_score"] = pd.Series(dtype=float)
        out["quality_pass"] = pd.Series(dtype=bool)
        out["quality_category"] = pd.Series(dtype=str)
        out["reject_reasons"] = pd.Series(dtype=str)
        return out
    thresholds = dict(config.get("thresholds") or {})
    weights = dict(config.get("weights") or {})
    categories = dict(config.get("categories") or {})
    rows = [_score_row(row, thresholds, weights, categories) for row in joint.to_dict(orient="records")]
    scored = joint.copy()
    score_df = pd.DataFrame(rows)
    for col in score_df.columns:
        scored[col] = score_df[col].values
    sort_cols = [col for col in ("quality_category_sort", "quality_score", "tier", "rank_score") if col in scored]
    if sort_cols:
        ascending = [True, False, True, False][: len(sort_cols)]
        scored = scored.sort_values(sort_cols, ascending=ascending, na_position="last", kind="mergesort")
    return scored


def _score_row(
    row: dict[str, Any],
    thresholds: dict[str, Any],
    weights: dict[str, Any],
    categories: dict[str, Any],
) -> dict[str, object]:
    reasons: list[str] = []
    score = float(weights.get("base", 0.0))
    has_ap = bool(row.get("has_aperture"))
    has_psf = bool(row.get("has_psf"))
    has_both = has_ap and has_psf
    ap_support = int(row.get("aperture_support") or 0)
    psf_support = int(row.get("psf_support") or 0)
    support = min(ap_support, psf_support) if has_both else max(ap_support, psf_support)
    flags = int(row.get("flagged_points_sum") or 0)
    detectors = _split_csv(row.get("detectors"))
    frames = _split_csv(row.get("best_frame_ids"))
    ratio = _num(row.get("flux_ratio_psf_aperture"))
    ap_snr = _num(row.get("aperture_peak_snr"))
    psf_snr = _num(row.get("psf_peak_snr"))
    peak_snr = max(ap_snr or 0.0, psf_snr or 0.0)
    width = _num(row.get("cluster_width_nm")) or 0.0

    if has_both:
        score += float(weights.get("both_methods_bonus", 0.0))
    elif thresholds.get("require_both_methods", False):
        reasons.append("missing_aperture_or_psf_match")
        score -= float(weights.get("missing_method_penalty", 0.0))

    score += support * float(weights.get("support_per_point", 0.0))
    score += min(len(detectors), 4) * float(weights.get("detector_per_detector", 0.0))
    score += min(len(frames), 8) * float(weights.get("frame_per_frame", 0.0))
    if peak_snr > 0:
        score += math.log10(max(peak_snr, 1.0)) * float(weights.get("snr_log_bonus", 0.0))

    max_flags = thresholds.get("max_flagged_points_sum")
    if max_flags is not None and flags > int(max_flags):
        reasons.append("too_many_flagged_points")
    score -= flags * float(weights.get("flagged_point_penalty", 0.0))

    min_ap_support = thresholds.get("min_aperture_support")
    if min_ap_support is not None and ap_support < int(min_ap_support):
        reasons.append("aperture_support_too_low")
    min_psf_support = thresholds.get("min_psf_support")
    if min_psf_support is not None and psf_support < int(min_psf_support):
        reasons.append("psf_support_too_low")

    if len(detectors) < int(thresholds.get("min_detector_count", 0) or 0):
        reasons.append("detector_count_too_low")
    if len(frames) < int(thresholds.get("min_frame_count", 0) or 0):
        reasons.append("frame_count_too_low")

    ratio_min = _num(thresholds.get("flux_ratio_psf_aperture_min"))
    ratio_max = _num(thresholds.get("flux_ratio_psf_aperture_max"))
    if ratio is None or ratio <= 0:
        reasons.append("flux_ratio_missing_or_invalid")
        score -= float(weights.get("flux_ratio_penalty_per_log10", 0.0)) * 2.0
    else:
        score -= abs(math.log10(ratio)) * float(weights.get("flux_ratio_penalty_per_log10", 0.0))
        if ratio_min is not None and ratio < ratio_min:
            reasons.append("psf_aperture_flux_ratio_too_low")
        if ratio_max is not None and ratio > ratio_max:
            reasons.append("psf_aperture_flux_ratio_too_high")

    max_ap_snr = _num(thresholds.get("max_aperture_peak_snr"))
    if max_ap_snr is not None and ap_snr is not None and ap_snr > max_ap_snr:
        reasons.append("aperture_snr_pathologically_high")
    max_psf_snr = _num(thresholds.get("max_psf_peak_snr"))
    if max_psf_snr is not None and psf_snr is not None and psf_snr > max_psf_snr:
        reasons.append("psf_snr_pathologically_high")

    max_width = _num(thresholds.get("max_cluster_width_nm"))
    if max_width is not None and width > max_width:
        reasons.append("cluster_too_wide")
    score -= max(width, 0.0) / 100.0 * float(weights.get("cluster_width_penalty_per_100nm", 0.0))

    if reasons:
        score -= len(set(reasons)) * float(weights.get("failed_threshold_penalty", 0.0))

    quality_pass = not reasons
    category = _category(score, quality_pass, categories)
    return {
        "quality_score": float(score),
        "quality_pass": bool(quality_pass),
        "quality_category": category,
        "quality_category_sort": {"high_confidence": 0, "review": 1, "reject": 2}.get(category, 9),
        "reject_reasons": ",".join(dict.fromkeys(reasons)),
        "quality_detector_count": len(detectors),
        "quality_frame_count": len(frames),
        "quality_support_min": support,
        "quality_flux_ratio_ok": bool(ratio is not None and ratio > 0 and (ratio_min is None or ratio >= ratio_min) and (ratio_max is None or ratio <= ratio_max)),
    }


def _category(score: float, quality_pass: bool, categories: dict[str, Any]) -> str:
    high = dict(categories.get("high_confidence") or {})
    if score >= float(high.get("min_quality_score", 70.0)) and (quality_pass or not high.get("require_quality_pass", True)):
        return "high_confidence"
    review = dict(categories.get("review") or {})
    if score >= float(review.get("min_quality_score", 35.0)) and (quality_pass or not review.get("require_quality_pass", False)):
        return "review"
    return "reject"


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply configurable quality scoring to blind joint candidates.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    config = load_quality_config(args.config)
    joint = pd.read_parquet(args.input)
    scored = score_joint_candidates(joint, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(args.output, index=False)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "config": str(args.config),
        "candidate_count": int(len(scored)),
        "quality_pass_count": int(scored["quality_pass"].fillna(False).astype(bool).sum()) if "quality_pass" in scored else 0,
        "quality_category_counts": scored["quality_category"].value_counts().sort_index().to_dict()
        if "quality_category" in scored
        else {},
        "reject_reason_counts": _reject_reason_counts(scored),
    }
    summary_path = args.summary or args.output.with_name("blind_quality_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def _reject_reason_counts(scored: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if "reject_reasons" not in scored:
        return counts
    for value in scored["reject_reasons"].fillna("").astype(str):
        for reason in value.split(","):
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    main()
