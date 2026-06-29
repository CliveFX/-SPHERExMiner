from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class InjectionRecoveryConfig:
    manifest_path: Path
    candidates_path: Path
    output_dir: Path
    min_score: float = 5.0
    wavelength_tolerance_nm: float = 10.0
    require_line_family: bool = False


def score_injection_recovery(config: InjectionRecoveryConfig) -> dict[str, Any]:
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    truth = _load_truth(config.manifest_path)
    if truth.empty:
        raise ValueError(f"No injections found in {config.manifest_path}")

    candidates = _load_candidates(config.candidates_path)
    recovery_rows: list[dict[str, Any]] = []
    matched_candidate_indexes: set[int] = set()
    for _, injection in truth.iterrows():
        match = _best_match(
            truth=injection,
            candidates=candidates,
            wavelength_tolerance_nm=config.wavelength_tolerance_nm,
            min_score=config.min_score,
            require_line_family=config.require_line_family,
        )
        recovered = match is not None
        row = injection.to_dict()
        row["recovered"] = bool(recovered)
        if recovered:
            matched_candidate_indexes.add(int(match.name))
            row.update(
                {
                    "recovered_line_nm": float(match["candidate_line_nm"]),
                    "wavelength_error_nm": float(match["candidate_line_nm"] - float(injection["injected_line_nm"])),
                    "recovered_score": float(match["candidate_score"]),
                    "recovered_flux_uJy": _maybe_float(match.get("candidate_flux_uJy")),
                    "candidate_rank": _maybe_int(match.get("candidate_rank")),
                    "candidate_status": "recovered",
                }
            )
        else:
            row.update(
                {
                    "recovered_line_nm": math.nan,
                    "wavelength_error_nm": math.nan,
                    "recovered_score": math.nan,
                    "recovered_flux_uJy": math.nan,
                    "candidate_rank": None,
                    "candidate_status": "missed",
                }
            )
        recovery_rows.append(row)

    recovery = pd.DataFrame(recovery_rows)
    above = candidates[pd.to_numeric(candidates.get("candidate_score"), errors="coerce") >= config.min_score].copy()
    false_positive = above[~above.index.isin(matched_candidate_indexes)].copy()

    by_strength = _group_recovery(recovery, "find_me_snr")
    by_line = _group_recovery(recovery, "line_family")

    recovery_path = config.output_dir / "injection_recovery.parquet"
    false_positive_path = config.output_dir / "false_positive_candidates.parquet"
    by_strength_path = config.output_dir / "recovery_by_strength.parquet"
    by_line_path = config.output_dir / "recovery_by_line.parquet"
    summary_path = config.output_dir / "truth_recovery_summary.json"
    false_summary_path = config.output_dir / "false_positive_summary.json"

    recovery.to_parquet(recovery_path, index=False)
    false_positive.to_parquet(false_positive_path, index=False)
    by_strength.to_parquet(by_strength_path, index=False)
    by_line.to_parquet(by_line_path, index=False)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "luxquarry_manifest_truth_recovery",
        "manifest_path": str(config.manifest_path),
        "candidates_path": str(config.candidates_path),
        "output_dir": str(config.output_dir),
        "min_score": config.min_score,
        "wavelength_tolerance_nm": config.wavelength_tolerance_nm,
        "require_line_family": config.require_line_family,
        "injection_count": int(len(recovery)),
        "recovered_count": int(recovery["recovered"].sum()),
        "missed_count": int((~recovery["recovered"]).sum()),
        "recovery_fraction": float(recovery["recovered"].mean()),
        "candidate_count_above_threshold": int(len(above)),
        "false_positive_count": int(len(false_positive)),
        "false_positives_per_injection": float(len(false_positive) / max(len(recovery), 1)),
        "recovery_path": str(recovery_path),
        "false_positive_path": str(false_positive_path),
        "recovery_by_strength_path": str(by_strength_path),
        "recovery_by_line_path": str(by_line_path),
        "summary_path": str(summary_path),
        "false_positive_summary_path": str(false_summary_path),
        "total_wall_sec": time.perf_counter() - started,
    }
    false_summary = {
        "created_utc": summary["created_utc"],
        "backend": summary["backend"],
        "candidates_path": str(config.candidates_path),
        "min_score": config.min_score,
        "candidate_count_above_threshold": int(len(above)),
        "matched_candidate_count": int(len(matched_candidate_indexes)),
        "false_positive_count": int(len(false_positive)),
        "false_positive_path": str(false_positive_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    false_summary_path.write_text(json.dumps(false_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _load_truth(manifest_path: Path) -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for injection in manifest.get("injections", []):
        frames = injection.get("frames") or []
        rows.append(
            {
                "manifest_path": str(manifest_path),
                "injection_id": injection.get("injection_id"),
                "target_id": str(injection.get("target_id")),
                "line_family": injection.get("line_family"),
                "nominal_line_nm": _maybe_float(injection.get("nominal_line_nm")),
                "injected_line_nm": _maybe_float(injection.get("injected_line_nm")),
                "offset_nm": _maybe_float(injection.get("offset_nm")),
                "line_width_nm": _maybe_float(injection.get("line_width_nm")),
                "strength_mode": injection.get("strength_mode"),
                "find_me_snr": _maybe_float(injection.get("find_me_snr")),
                "line_flux_uJy": _maybe_float(injection.get("line_flux_uJy")),
                "frames_written": _maybe_int(injection.get("frames_written")) or 0,
                "frames_skipped": _maybe_int(injection.get("frames_skipped")) or 0,
                "max_frame_flux_uJy": max([_maybe_float(frame.get("injected_flux_uJy")) or 0.0 for frame in frames] or [0.0]),
            }
        )
    return pd.DataFrame(rows)


def _load_candidates(candidates_path: Path) -> pd.DataFrame:
    if not candidates_path.exists():
        return _empty_candidate_frame()
    raw = pd.read_parquet(candidates_path)
    if raw.empty:
        return _empty_candidate_frame()
    out = raw.copy()
    if "candidate_line_nm" not in out:
        if "cwave_um" in out:
            out["candidate_line_nm"] = pd.to_numeric(out["cwave_um"], errors="coerce") * 1000.0
        elif "cwave_nm" in out:
            out["candidate_line_nm"] = pd.to_numeric(out["cwave_nm"], errors="coerce")
        else:
            out["candidate_line_nm"] = math.nan
    if "candidate_score" not in out:
        if "matched_snr" in out:
            out["candidate_score"] = pd.to_numeric(out["matched_snr"], errors="coerce")
        elif "abs_zscore" in out:
            out["candidate_score"] = pd.to_numeric(out["abs_zscore"], errors="coerce")
        elif "zscore" in out:
            out["candidate_score"] = pd.to_numeric(out["zscore"], errors="coerce").abs()
        else:
            out["candidate_score"] = math.nan
    if "candidate_flux_uJy" not in out:
        if "matched_flux_uJy" in out:
            out["candidate_flux_uJy"] = pd.to_numeric(out["matched_flux_uJy"], errors="coerce")
        elif "aperture_flux_uJy" in out:
            out["candidate_flux_uJy"] = pd.to_numeric(out["aperture_flux_uJy"], errors="coerce")
        else:
            out["candidate_flux_uJy"] = math.nan
    out["target_id"] = out["target_id"].astype(str)
    return out


def _empty_candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["target_id", "candidate_line_nm", "line_family", "candidate_score", "candidate_flux_uJy"])


def _best_match(
    truth: pd.Series,
    candidates: pd.DataFrame,
    wavelength_tolerance_nm: float,
    min_score: float,
    require_line_family: bool,
) -> pd.Series | None:
    injected_line_nm = _maybe_float(truth.get("injected_line_nm"))
    if injected_line_nm is None:
        return None
    rows = candidates[candidates["target_id"].astype(str).eq(str(truth["target_id"]))].copy()
    if rows.empty:
        return None
    if require_line_family and "line_family" in rows.columns:
        rows = rows[rows["line_family"].astype(str).eq(str(truth["line_family"]))]
    rows = rows[
        (pd.to_numeric(rows["candidate_line_nm"], errors="coerce") - injected_line_nm).abs()
        <= wavelength_tolerance_nm
    ]
    rows = rows[pd.to_numeric(rows["candidate_score"], errors="coerce") >= min_score]
    if rows.empty:
        return None
    return rows.sort_values("candidate_score", ascending=False).iloc[0]


def _group_recovery(recovery: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in recovery:
        return pd.DataFrame()
    out = (
        recovery.groupby(column, dropna=False)
        .agg(injections=("injection_id", "size"), recovered=("recovered", "sum"), median_recovered_score=("recovered_score", "median"))
        .reset_index()
    )
    out["recovery_fraction"] = out["recovered"] / out["injections"]
    return out


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
