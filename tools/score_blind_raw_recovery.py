from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def _read_truth(manifest_path: Path) -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for injection in manifest.get("injections", []):
        rows.append(
            {
                "injection_id": injection.get("injection_id"),
                "target_id": str(injection.get("target_id")),
                "line_family": injection.get("line_family"),
                "injected_line_nm": _num(injection.get("injected_line_nm")),
                "find_me_snr": _num(injection.get("find_me_snr")),
                "line_flux_uJy": _num(injection.get("line_flux_uJy")),
                "max_frame_flux_uJy": max(
                    [_num(frame.get("injected_flux_uJy")) or 0.0 for frame in injection.get("frames", [])] or [0.0]
                ),
                "frames_written": int(injection.get("frames_written") or 0),
                "frames_skipped": int(injection.get("frames_skipped") or 0),
            }
        )
    return pd.DataFrame(rows)


def _num(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _target_ids_in_spectra(run_dir: Path) -> set[str]:
    path = run_dir / "spectra" / "target_spectra.parquet"
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=["target_id"])
    return set(df["target_id"].dropna().astype(str))


def score_blind_raw_recovery(
    *,
    manifest_path: Path,
    injected_run_dir: Path,
    candidates_path: Path,
    output_dir: Path,
    wavelength_tolerance_nm: float,
) -> dict[str, Any]:
    truth = _read_truth(manifest_path)
    candidates = pd.read_parquet(candidates_path) if candidates_path.exists() else pd.DataFrame()
    present_ids = _target_ids_in_spectra(injected_run_dir)
    rows: list[dict[str, Any]] = []
    for truth_row in truth.to_dict(orient="records"):
        target_id = str(truth_row.get("target_id"))
        line_nm = _num(truth_row.get("injected_line_nm"))
        matched = pd.DataFrame()
        if line_nm is not None and not candidates.empty and "target_id" in candidates and "peak_line_nm" in candidates:
            same_target = candidates[candidates["target_id"].astype(str).eq(target_id)].copy()
            if not same_target.empty:
                delta = (pd.to_numeric(same_target["peak_line_nm"], errors="coerce") - line_nm).abs()
                matched = same_target[delta <= float(wavelength_tolerance_nm)].copy()
                if not matched.empty:
                    if "quality_pass" in matched:
                        matched = matched.sort_values(["quality_pass", "quality_score"], ascending=[False, False], na_position="last")
                    elif "rank_score" in matched:
                        matched = matched.sort_values("rank_score", ascending=False, na_position="last")
        out = dict(truth_row)
        out["target_present_in_spectra"] = target_id in present_ids
        if matched.empty:
            out.update(
                {
                    "blind_raw_match": False,
                    "blind_raw_quality_pass": False,
                    "blind_raw_quality_category": "no_blind_match",
                    "blind_raw_reject_reasons": "no_blind_match",
                    "blind_raw_peak_line_nm": None,
                    "blind_raw_quality_score": None,
                    "blind_raw_tier": None,
                    "blind_raw_aperture_peak_snr": None,
                    "blind_raw_psf_peak_snr": None,
                }
            )
        else:
            best = matched.iloc[0].to_dict()
            out.update(
                {
                    "blind_raw_match": True,
                    "blind_raw_quality_pass": bool(best.get("quality_pass", False)),
                    "blind_raw_quality_category": best.get("quality_category", "unscored"),
                    "blind_raw_reject_reasons": best.get("reject_reasons"),
                    "blind_raw_peak_line_nm": _num(best.get("peak_line_nm")),
                    "blind_raw_quality_score": _num(best.get("quality_score")),
                    "blind_raw_tier": best.get("tier"),
                    "blind_raw_aperture_peak_snr": _num(best.get("aperture_peak_snr")),
                    "blind_raw_psf_peak_snr": _num(best.get("psf_peak_snr")),
                }
            )
        rows.append(out)

    scored = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    recovery_path = output_dir / "blind_raw_recovery.parquet"
    summary_path = output_dir / "blind_raw_recovery_summary.json"
    scored.to_parquet(recovery_path, index=False)
    present = scored[scored["target_present_in_spectra"].fillna(False).astype(bool)] if not scored.empty else scored
    summary = {
        "scorer_mode": "blind_raw_injected_truth_targets",
        "manifest_path": str(manifest_path),
        "injected_run_dir": str(injected_run_dir),
        "candidates_path": str(candidates_path),
        "wavelength_tolerance_nm": float(wavelength_tolerance_nm),
        "truth_count": int(len(scored)),
        "target_present_truth_count": int(len(present)),
        "blind_raw_match_count": int(scored.get("blind_raw_match", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "blind_raw_quality_pass_count": int(
            scored.get("blind_raw_quality_pass", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
        ),
        "target_present_blind_raw_match_count": int(
            present.get("blind_raw_match", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
        ),
        "target_present_blind_raw_quality_pass_count": int(
            present.get("blind_raw_quality_pass", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
        ),
        "quality_category_counts": scored.get("blind_raw_quality_category", pd.Series(dtype=str)).value_counts().sort_index().to_dict(),
        "recovery_path": str(recovery_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Score raw injected blind candidates against injection truth.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--injected-run-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wavelength-tolerance-nm", type=float, default=10.0)
    args = parser.parse_args()
    print(
        json.dumps(
            score_blind_raw_recovery(
                manifest_path=args.manifest,
                injected_run_dir=args.injected_run_dir,
                candidates_path=args.candidates,
                output_dir=args.output_dir,
                wavelength_tolerance_nm=args.wavelength_tolerance_nm,
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
