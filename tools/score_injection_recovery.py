from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_truth(manifest_path: Path) -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for injection in manifest.get("injections", []):
        frames = injection.get("frames", [])
        rows.append(
            {
                "injection_id": injection.get("injection_id"),
                "target_id": injection.get("target_id"),
                "line_family": injection.get("line_family"),
                "nominal_line_nm": injection.get("nominal_line_nm"),
                "injected_line_nm": injection.get("injected_line_nm"),
                "offset_nm": injection.get("offset_nm"),
                "line_width_nm": injection.get("line_width_nm"),
                "strength_mode": injection.get("strength_mode"),
                "find_me_snr": injection.get("find_me_snr"),
                "line_flux_uJy": injection.get("line_flux_uJy"),
                "frames_written": injection.get("frames_written"),
                "frames_skipped": injection.get("frames_skipped"),
                "max_frame_flux_uJy": max(
                    [float(frame.get("injected_flux_uJy") or 0.0) for frame in frames] or [0.0]
                ),
            }
        )
    return pd.DataFrame(rows)


def _best_match(
    truth: pd.Series,
    candidates: pd.DataFrame,
    wavelength_tolerance_nm: float,
    min_snr: float,
    require_line_family: bool,
) -> pd.Series | None:
    rows = candidates[candidates["target_id"].astype(str).eq(str(truth["target_id"]))].copy()
    if rows.empty:
        return None
    if require_line_family and "line_family" in rows.columns:
        rows = rows[rows["line_family"].astype(str).eq(str(truth["line_family"]))]
    rows = rows[
        (pd.to_numeric(rows["candidate_line_nm"], errors="coerce") - float(truth["injected_line_nm"])).abs()
        <= wavelength_tolerance_nm
    ]
    rows = rows[pd.to_numeric(rows["matched_snr"], errors="coerce") >= min_snr]
    if rows.empty:
        return None
    return rows.sort_values("matched_snr", ascending=False).iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score matched-filter candidates against fake injection truth.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-snr", type=float, default=5.0)
    parser.add_argument("--wavelength-tolerance-nm", type=float, default=10.0)
    parser.add_argument("--require-line-family", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    truth = _load_truth(args.manifest)
    if truth.empty:
        raise SystemExit(f"No injections found in {args.manifest}")
    candidates = pd.read_parquet(args.candidates)
    if candidates.empty:
        candidates = pd.DataFrame(
            columns=["target_id", "candidate_line_nm", "line_family", "matched_snr", "matched_flux_uJy"]
        )

    recovery_rows: list[dict[str, Any]] = []
    matched_candidate_indexes: set[int] = set()
    for _, injection in truth.iterrows():
        match = _best_match(
            truth=injection,
            candidates=candidates,
            wavelength_tolerance_nm=args.wavelength_tolerance_nm,
            min_snr=args.min_snr,
            require_line_family=args.require_line_family,
        )
        recovered = match is not None
        row = injection.to_dict()
        row["recovered"] = bool(recovered)
        if recovered:
            matched_candidate_indexes.add(int(match.name))
            row.update(
                {
                    "recovered_line_nm": float(match["candidate_line_nm"]),
                    "wavelength_error_nm": float(match["candidate_line_nm"] - injection["injected_line_nm"]),
                    "recovered_snr": float(match["matched_snr"]),
                    "recovered_flux_uJy": float(match["matched_flux_uJy"]),
                    "candidate_status": match.get("candidate_status"),
                }
            )
        else:
            row.update(
                {
                    "recovered_line_nm": np.nan,
                    "wavelength_error_nm": np.nan,
                    "recovered_snr": np.nan,
                    "recovered_flux_uJy": np.nan,
                    "candidate_status": "missed",
                }
            )
        recovery_rows.append(row)

    recovery = pd.DataFrame(recovery_rows)
    above = candidates[pd.to_numeric(candidates.get("matched_snr"), errors="coerce") >= args.min_snr].copy()
    false_positive = above[~above.index.isin(matched_candidate_indexes)].copy()

    by_strength = pd.DataFrame()
    if "find_me_snr" in recovery.columns:
        by_strength = (
            recovery.groupby("find_me_snr", dropna=False)
            .agg(
                injections=("injection_id", "size"),
                recovered=("recovered", "sum"),
                median_recovered_snr=("recovered_snr", "median"),
            )
            .reset_index()
        )
        by_strength["recovery_fraction"] = by_strength["recovered"] / by_strength["injections"]

    by_line = (
        recovery.groupby("line_family", dropna=False)
        .agg(injections=("injection_id", "size"), recovered=("recovered", "sum"))
        .reset_index()
    )
    by_line["recovery_fraction"] = by_line["recovered"] / by_line["injections"]

    output_dir = args.output_dir or (args.manifest.parent / "recovery_score")
    output_dir.mkdir(parents=True, exist_ok=True)
    recovery_path = output_dir / "injection_recovery.parquet"
    false_positive_path = output_dir / "false_positive_candidates.parquet"
    by_strength_path = output_dir / "recovery_by_strength.parquet"
    by_line_path = output_dir / "recovery_by_line.parquet"
    summary_path = output_dir / "recovery_summary.json"
    recovery.to_parquet(recovery_path, index=False)
    false_positive.to_parquet(false_positive_path, index=False)
    by_strength.to_parquet(by_strength_path, index=False)
    by_line.to_parquet(by_line_path, index=False)

    summary = {
        "manifest": str(args.manifest),
        "candidates": str(args.candidates),
        "min_snr": args.min_snr,
        "wavelength_tolerance_nm": args.wavelength_tolerance_nm,
        "require_line_family": args.require_line_family,
        "injection_count": int(len(recovery)),
        "recovered_count": int(recovery["recovered"].sum()),
        "recovery_fraction": float(recovery["recovered"].mean()),
        "candidate_count_above_threshold": int(len(above)),
        "false_positive_count": int(len(false_positive)),
        "false_positives_per_injection": float(len(false_positive) / max(len(recovery), 1)),
        "recovery_path": str(recovery_path),
        "false_positive_path": str(false_positive_path),
        "recovery_by_strength_path": str(by_strength_path),
        "recovery_by_line_path": str(by_line_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
