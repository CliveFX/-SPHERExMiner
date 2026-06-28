from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.make_injection_plan import _parse_csv_floats, _parse_line_families, _safe_id
from tools.wavelength_guard import assert_science_wavelengths


DEFAULT_RUN_DIR = Path("/mnt/niroseti/spherex_cache/runs/injrec_baseline_10k_f80_g14_16_gpu3")
DEFAULT_OUTPUT_ROOT = Path("/mnt/niroseti/spherex_cache/injection_campaigns")


def _response(cwave_um: np.ndarray, cband_um: np.ndarray, line_um: float, line_width_um: float) -> np.ndarray:
    fwhm = np.sqrt(np.maximum(cband_um, 0.0) ** 2 + max(line_width_um, 0.0) ** 2)
    sigma = fwhm / 2.354820045
    out = np.zeros_like(cwave_um, dtype=float)
    good = np.isfinite(cwave_um) & np.isfinite(sigma) & (sigma > 0.0)
    out[good] = np.exp(-0.5 * ((cwave_um[good] - line_um) / sigma[good]) ** 2)
    return out


def _target_support(df: pd.DataFrame, line_nm: float, line_width_nm: float, min_response: float) -> pd.DataFrame:
    work = df.copy()
    cwave = pd.to_numeric(work["cwave_um"], errors="coerce").to_numpy(dtype=float)
    cband = pd.to_numeric(work["cband_um"], errors="coerce").to_numpy(dtype=float)
    unc = pd.to_numeric(work["aperture_flux_unc_uJy"], errors="coerce").to_numpy(dtype=float)
    response = _response(cwave, cband, line_nm / 1000.0, line_width_nm / 1000.0)
    work["_usable"] = np.isfinite(unc) & (unc > 0.0) & (response >= min_response)
    work["_max_response_weight"] = response
    grouped = work.groupby("target_id", dropna=False)
    summary = grouped.agg(
        usable_points=("_usable", "sum"),
        max_response=("_max_response_weight", "max"),
        n_measurements=("target_id", "size"),
        median_unc_uJy=("aperture_flux_unc_uJy", "median"),
        phot_g_mean_mag=("phot_g_mean_mag", "median") if "phot_g_mean_mag" in work.columns else ("target_id", "size"),
    ).reset_index()
    summary = summary[summary["usable_points"].gt(0)].copy()
    return summary.sort_values(["median_unc_uJy", "usable_points"], ascending=[True, False])


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a mixed laser-line injection plan without stacking strength variants.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--campaign-id", default="injrec_10k_f80_mixed_lasers")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--line-families",
        default="diode_808:808,diode_980:980,nd_yag_1064:1064,telecom_1310:1310,telecom_1550:1550,thulium_2000:2000",
    )
    parser.add_argument("--strengths-sigma", default="3,5,8,12,20")
    parser.add_argument("--targets-per-cell", type=int, default=20)
    parser.add_argument(
        "--max-lines-per-target",
        type=int,
        default=1,
        help="Maximum distinct line-family injections allowed on one target. Use 1 for one fake line per spectrum.",
    )
    parser.add_argument("--line-width-nm", type=float, default=1.0)
    parser.add_argument("--min-response", type=float, default=1e-3)
    parser.add_argument("--max-frames-per-injection", type=int)
    parser.add_argument("--min-measurements", type=int, default=20)
    parser.add_argument(
        "--max-line-flux-uJy",
        type=float,
        help=(
            "Reject target/line/strength choices whose estimated intrinsic line flux exceeds this cap. "
            "The estimate is strength_sigma * median_unc_uJy / max_response."
        ),
    )
    parser.add_argument(
        "--allow-approx-wavelengths",
        action="store_true",
        help="Allow old MEF WCS-WAVE spectra. Not valid for science-grade injection/recovery.",
    )
    parser.add_argument("--seed", type=int, default=20260622)
    args = parser.parse_args()

    spectra_path = args.run_dir / "spectra" / "target_spectra.parquet"
    if not spectra_path.exists():
        raise SystemExit(f"Missing spectra parquet: {spectra_path}")
    df = pd.read_parquet(spectra_path)
    try:
        assert_science_wavelengths(df, spectra_path, allow_approx=args.allow_approx_wavelengths)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if "fatal_flag_present" in df.columns:
        df = df[~df["fatal_flag_present"].fillna(False).astype(bool)].copy()
    df = df.groupby("target_id").filter(lambda group: len(group) >= args.min_measurements)

    line_families = _parse_line_families(args.line_families)
    strengths = _parse_csv_floats(args.strengths_sigma)
    rng = np.random.default_rng(args.seed)
    target_line_counts: dict[str, int] = {}
    targets_by_id: dict[str, dict[str, Any]] = {}
    injections: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for family in line_families:
        line_family = str(family["line_family"])
        nominal_line_nm = float(family["nominal_line_nm"])
        support = _target_support(df, nominal_line_nm, args.line_width_nm, args.min_response)
        if support.empty:
            print(f"no support for {line_family} {nominal_line_nm:g}nm", flush=True)
            continue
        # Prefer stable, lower-uncertainty targets but shuffle within a bounded support pool.
        pool = support.head(max(args.targets_per_cell * 12, args.targets_per_cell * 2, 100)).copy()
        for strength in strengths:
            skipped_cap = 0
            for _, row in pool.iterrows():
                target_id = str(row["target_id"])
                median_unc = _finite_float(row["median_unc_uJy"])
                max_response = _finite_float(row["max_response"])
                estimated_line_flux = None
                if median_unc is not None and max_response is not None and max_response > 0:
                    estimated_line_flux = float(strength) * median_unc / max_response
                if (
                    args.max_line_flux_uJy is not None
                    and estimated_line_flux is not None
                    and estimated_line_flux > args.max_line_flux_uJy
                ):
                    skipped_cap += 1
                    continue
                candidates.append(
                    {
                        "target_id": target_id,
                        "line_family": line_family,
                        "nominal_line_nm": nominal_line_nm,
                        "injected_line_nm": nominal_line_nm,
                        "offset_nm": 0.0,
                        "line_width_nm": float(args.line_width_nm),
                        "strength_mode": "find_me_snr",
                        "find_me_snr": float(strength),
                        "min_response": float(args.min_response),
                        "max_frames": args.max_frames_per_injection,
                        "estimated_line_flux_uJy": estimated_line_flux,
                        "line_flux_cap_uJy": args.max_line_flux_uJy,
                        "n_measurements": int(row["n_measurements"]),
                        "median_unc_uJy": median_unc,
                        "phot_g_mean_mag": _finite_float(row.get("phot_g_mean_mag")),
                    }
                )
            if skipped_cap:
                print(
                    f"warning: skipped {skipped_cap} candidates for {line_family} s={strength:g} by flux cap",
                    flush=True,
                )

    if candidates:
        order = rng.permutation(len(candidates))
        for index in order:
            if len(injections) >= args.targets_per_cell:
                break
            candidate = dict(candidates[int(index)])
            target_id = str(candidate["target_id"])
            if args.max_lines_per_target > 0 and target_line_counts.get(target_id, 0) >= args.max_lines_per_target:
                continue
            target_line_counts[target_id] = target_line_counts.get(target_id, 0) + 1
            targets_by_id[target_id] = {
                "target_id": target_id,
                "n_measurements": int(candidate.pop("n_measurements")),
                "median_unc_uJy": candidate.pop("median_unc_uJy"),
                "phot_g_mean_mag": candidate.pop("phot_g_mean_mag"),
            }
            candidate["injection_id"] = _safe_id(
                f"{target_id}_{candidate['line_family']}_{candidate['nominal_line_nm']:g}nm_s{candidate['find_me_snr']:g}"
            )
            injections.append(candidate)

    if len(injections) < args.targets_per_cell:
        print(
            f"warning: only picked {len(injections)}/{args.targets_per_cell} total injection targets",
            flush=True,
        )

    campaign_root = args.output_root / args.campaign_id
    output = args.output or (campaign_root / "injection_plan.json")
    plan = {
        "plan_version": 1,
        "plan_kind": "mixed_laser_line_strength_grid",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": args.campaign_id,
        "baseline_run_dir": str(args.run_dir.resolve()),
        "campaign_root": str(campaign_root.resolve()),
        "spectra_path": str(spectra_path.resolve()),
        "selection": {
            "line_families": args.line_families,
            "strengths_sigma": strengths,
            "targets_per_cell": args.targets_per_cell,
            "targets_per_cell_semantics": "total_injection_targets",
            "max_lines_per_target": args.max_lines_per_target,
            "line_width_nm": args.line_width_nm,
            "min_response": args.min_response,
            "min_measurements": args.min_measurements,
            "max_line_flux_uJy": args.max_line_flux_uJy,
            "seed": args.seed,
        },
        "line_families": line_families,
        "strengths_sigma": strengths,
        "targets": list(targets_by_id.values()),
        "injections": injections,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "plan_path": str(output),
                "campaign_id": args.campaign_id,
                "target_count": len(targets_by_id),
                "injection_count": len(injections),
                "campaign_root": str(campaign_root),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
