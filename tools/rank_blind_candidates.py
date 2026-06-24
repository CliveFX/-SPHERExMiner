from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.score_blind_candidate_quality import DEFAULT_CONFIG, load_quality_config, score_joint_candidates


JOINT_COLUMNS = [
    "joint_candidate_id",
    "target_id",
    "peak_line_nm",
    "line_min_nm",
    "line_max_nm",
    "cluster_width_nm",
    "has_aperture",
    "has_psf",
    "aperture_peak_snr",
    "psf_peak_snr",
    "aperture_peak_flux_uJy",
    "psf_peak_flux_uJy",
    "aperture_support",
    "psf_support",
    "flux_ratio_psf_aperture",
    "detectors",
    "best_frame_ids",
    "flagged_points_sum",
    "aperture_cluster_id",
    "psf_cluster_id",
    "tier",
    "rank_score",
]


def _read_clusters(path: Path, flux_kind: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty or "peak_snr" not in df.columns:
        return df
    df = df.copy()
    df["flux_kind"] = flux_kind
    return df


def _split_csv(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    return {part for part in str(value).split(",") if part}


def _nearby_rows(left: pd.Series, right_df: pd.DataFrame, tolerance_nm: float) -> pd.DataFrame:
    if right_df.empty:
        return right_df
    rows = right_df[right_df["target_id"].astype(str).eq(str(left["target_id"]))].copy()
    if rows.empty:
        return rows
    return rows[(pd.to_numeric(rows["peak_line_nm"], errors="coerce") - float(left["peak_line_nm"])).abs() <= tolerance_nm]


def _support_tier(ap_support: int, psf_support: int, has_both: bool) -> str:
    support = max(ap_support, psf_support)
    if support >= 3 and has_both:
        return "A"
    if support >= 3:
        return "B"
    if support >= 2 and has_both:
        return "B"
    if support >= 2:
        return "C"
    return "D"


def _rank_score(row: dict[str, Any]) -> float:
    ap_snr = _num(row.get("aperture_peak_snr"))
    psf_snr = _num(row.get("psf_peak_snr"))
    peak = max(ap_snr or 0.0, psf_snr or 0.0)
    support = max(int(row.get("aperture_support") or 0), int(row.get("psf_support") or 0))
    has_both = bool(row.get("has_aperture") and row.get("has_psf"))
    detector_count = len(_split_csv(row.get("detectors")))
    frame_count = len(_split_csv(row.get("best_frame_ids")))
    flags = int(row.get("flagged_points_sum") or 0)
    width = _num(row.get("cluster_width_nm")) or 0.0
    score = peak
    score += min(support, 5) * 2.5
    if has_both:
        score += 8.0
    score += min(detector_count, 3) * 1.5
    score += min(frame_count, 5) * 0.75
    if support <= 1:
        score -= 12.0
    score -= min(flags, 10) * 2.0
    if width > 250.0:
        score -= 5.0
    return float(score)


def _num(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _merge_pair(ap: pd.Series | None, psf: pd.Series | None, index: int) -> dict[str, Any]:
    base = ap if ap is not None else psf
    assert base is not None
    ap_support = int(ap.get("supporting_points_max") or 0) if ap is not None else 0
    psf_support = int(psf.get("supporting_points_max") or 0) if psf is not None else 0
    ap_snr = _num(ap.get("peak_snr")) if ap is not None else None
    psf_snr = _num(psf.get("peak_snr")) if psf is not None else None
    ap_flux = _num(ap.get("peak_flux_uJy")) if ap is not None else None
    psf_flux = _num(psf.get("peak_flux_uJy")) if psf is not None else None
    line_values = [value for value in [_num(ap.get("peak_line_nm")) if ap is not None else None, _num(psf.get("peak_line_nm")) if psf is not None else None] if value is not None]
    peak_line_nm = float(np.mean(line_values)) if line_values else float(base.get("peak_line_nm"))
    detectors = sorted(_split_csv(ap.get("detectors") if ap is not None else None) | _split_csv(psf.get("detectors") if psf is not None else None))
    frames = list(
        dict.fromkeys(
            list(_split_csv(ap.get("best_frame_ids") if ap is not None else None))
            + list(_split_csv(psf.get("best_frame_ids") if psf is not None else None))
        )
    )
    line_min_values = [
        value
        for value in [
            _num(ap.get("line_min_nm")) if ap is not None else None,
            _num(psf.get("line_min_nm")) if psf is not None else None,
            peak_line_nm,
        ]
        if value is not None
    ]
    line_max_values = [
        value
        for value in [
            _num(ap.get("line_max_nm")) if ap is not None else None,
            _num(psf.get("line_max_nm")) if psf is not None else None,
            peak_line_nm,
        ]
        if value is not None
    ]
    line_min = min(line_min_values)
    line_max = max(line_max_values)
    row: dict[str, Any] = {
        "joint_candidate_id": f"{base['target_id']}::joint::{index}::{peak_line_nm:g}",
        "target_id": str(base["target_id"]),
        "peak_line_nm": peak_line_nm,
        "line_min_nm": float(line_min),
        "line_max_nm": float(line_max),
        "cluster_width_nm": float(line_max - line_min),
        "has_aperture": ap is not None,
        "has_psf": psf is not None,
        "aperture_peak_snr": ap_snr,
        "psf_peak_snr": psf_snr,
        "aperture_peak_flux_uJy": ap_flux,
        "psf_peak_flux_uJy": psf_flux,
        "aperture_support": ap_support,
        "psf_support": psf_support,
        "flux_ratio_psf_aperture": float(psf_flux / ap_flux) if psf_flux is not None and ap_flux not in (None, 0.0) else np.nan,
        "detectors": ",".join(detectors),
        "best_frame_ids": ",".join(frames[:12]),
        "flagged_points_sum": int((ap.get("flagged_points_sum") if ap is not None else 0) or 0)
        + int((psf.get("flagged_points_sum") if psf is not None else 0) or 0),
        "aperture_cluster_id": ap.get("cluster_id") if ap is not None else None,
        "psf_cluster_id": psf.get("cluster_id") if psf is not None else None,
    }
    row["tier"] = _support_tier(ap_support, psf_support, bool(row["has_aperture"] and row["has_psf"]))
    row["rank_score"] = _rank_score(row)
    return row


def build_joint(aperture: pd.DataFrame, psf: pd.DataFrame, tolerance_nm: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    used_psf: set[int] = set()
    index = 0
    if aperture.empty or "peak_snr" not in aperture.columns:
        aperture = pd.DataFrame(columns=["peak_snr"])
    if psf.empty or "peak_snr" not in psf.columns:
        psf = pd.DataFrame(columns=["peak_snr"])
    for _, ap_row in aperture.sort_values("peak_snr", ascending=False, na_position="last").iterrows():
        matches = _nearby_rows(ap_row, psf, tolerance_nm)
        matches = matches[~matches.index.isin(used_psf)]
        if matches.empty:
            rows.append(_merge_pair(ap_row, None, index))
        else:
            psf_row = matches.sort_values("peak_snr", ascending=False, na_position="last").iloc[0]
            used_psf.add(int(psf_row.name))
            rows.append(_merge_pair(ap_row, psf_row, index))
        index += 1
    for _, psf_row in psf[~psf.index.isin(used_psf)].sort_values("peak_snr", ascending=False, na_position="last").iterrows():
        rows.append(_merge_pair(None, psf_row, index))
        index += 1
    if not rows:
        return pd.DataFrame(columns=JOINT_COLUMNS)
    return pd.DataFrame(rows).sort_values(["tier", "rank_score"], ascending=[True, False], na_position="last")


def main() -> None:
    parser = argparse.ArgumentParser(description="Join and rank aperture/PSF blind candidate clusters.")
    parser.add_argument("--aperture-dir", type=Path, required=True)
    parser.add_argument("--psf-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--match-tolerance-nm", type=float, default=10.0)
    parser.add_argument("--quality-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--no-quality-score", action="store_true")
    args = parser.parse_args()

    aperture = _read_clusters(args.aperture_dir / "blind_candidate_clusters.parquet", "aperture")
    psf = _read_clusters(args.psf_dir / "blind_candidate_clusters.parquet", "psf")
    joint = build_joint(aperture, psf, args.match_tolerance_nm)
    if not args.no_quality_score:
        joint = score_joint_candidates(joint, load_quality_config(args.quality_config))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "blind_joint_candidates.parquet"
    summary_path = args.output_dir / "blind_joint_summary.json"
    joint.to_parquet(out_path, index=False)
    summary = {
        "aperture_dir": str(args.aperture_dir),
        "psf_dir": str(args.psf_dir),
        "match_tolerance_nm": args.match_tolerance_nm,
        "aperture_cluster_count": int(len(aperture)),
        "psf_cluster_count": int(len(psf)),
        "joint_candidate_count": int(len(joint)),
        "tier_counts": joint["tier"].value_counts().sort_index().to_dict() if "tier" in joint else {},
        "quality_config": str(args.quality_config) if not args.no_quality_score else None,
        "quality_pass_count": int(joint["quality_pass"].fillna(False).astype(bool).sum()) if "quality_pass" in joint else None,
        "quality_category_counts": joint["quality_category"].value_counts().sort_index().to_dict()
        if "quality_category" in joint
        else {},
        "joint_candidates_path": str(out_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
