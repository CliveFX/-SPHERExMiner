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

from tools.wavelength_guard import assert_science_wavelengths  # noqa: E402


DEFAULT_RUN_DIR = Path("/mnt/niroseti/spherex_cache/runs/ucs0972_gpu_g14_16_n1000_f500")
DEFAULT_OUTPUT_ROOT = Path("/mnt/niroseti/spherex_cache/injection_campaigns")


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._+-" else "_" for ch in value).strip("_")


def _parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_line_families(value: str) -> list[dict[str, Any]]:
    families: list[dict[str, Any]] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" in raw:
            name, wave = raw.split(":", 1)
        else:
            wave = raw
            name = f"line_{float(wave):g}nm"
        nominal = float(wave)
        families.append(
            {
                "line_family": _safe_id(name),
                "nominal_line_nm": nominal,
            }
        )
    return families


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _choose_targets(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = df.copy()
    if "fatal_flag_present" in rows.columns and args.ignore_flagged:
        rows = rows[~rows["fatal_flag_present"].fillna(False).astype(bool)]
    if "phot_g_mean_mag" in rows.columns:
        if args.g_min is not None:
            rows = rows[pd.to_numeric(rows["phot_g_mean_mag"], errors="coerce") >= args.g_min]
        if args.g_max is not None:
            rows = rows[pd.to_numeric(rows["phot_g_mean_mag"], errors="coerce") <= args.g_max]
    if args.target_id:
        wanted = {item.strip() for item in args.target_id.split(",") if item.strip()}
        rows = rows[rows["target_id"].astype(str).isin(wanted)]
    if args.require_wavelength_nm and "cband_um" in rows.columns:
        line_um = float(args.require_wavelength_nm) / 1000.0
        line_width_um = float(args.line_width_nm) / 1000.0
        cwave = pd.to_numeric(rows["cwave_um"], errors="coerce").to_numpy(dtype=float)
        cband = pd.to_numeric(rows["cband_um"], errors="coerce").to_numpy(dtype=float)
        fwhm = np.sqrt(np.maximum(cband, 0.0) ** 2 + max(line_width_um, 0.0) ** 2)
        sigma = fwhm / 2.354820045
        response = np.zeros(len(rows), dtype=float)
        good_sigma = np.isfinite(cwave) & np.isfinite(sigma) & (sigma > 0.0)
        response[good_sigma] = np.exp(-0.5 * ((cwave[good_sigma] - line_um) / sigma[good_sigma]) ** 2)
        unc = pd.to_numeric(rows["aperture_flux_unc_uJy"], errors="coerce").to_numpy(dtype=float)
        rows["_required_line_usable"] = np.isfinite(unc) & (unc > 0.0) & (response >= float(args.min_response))

    grouped = rows.groupby("target_id", dropna=False)
    target_summary = grouped.agg(
        n_measurements=("target_id", "size"),
        wavelength_min_um=("cwave_um", "min"),
        wavelength_max_um=("cwave_um", "max"),
        median_unc_uJy=("aperture_flux_unc_uJy", "median"),
        median_flux_uJy=("aperture_flux_uJy", "median"),
        phot_g_mean_mag=("phot_g_mean_mag", "median") if "phot_g_mean_mag" in rows.columns else ("target_id", "size"),
        usable_required_line_points=("_required_line_usable", "sum")
        if "_required_line_usable" in rows.columns
        else ("target_id", "size"),
    ).reset_index()
    target_summary = target_summary[
        target_summary["n_measurements"].ge(args.min_measurements)
        & target_summary["median_unc_uJy"].replace([np.inf, -np.inf], np.nan).notna()
    ].copy()
    if args.require_wavelength_nm:
        wave_um = args.require_wavelength_nm / 1000.0
        target_summary = target_summary[
            target_summary["wavelength_min_um"].le(wave_um) & target_summary["wavelength_max_um"].ge(wave_um)
        ]
        if "usable_required_line_points" in target_summary.columns:
            target_summary = target_summary[target_summary["usable_required_line_points"].gt(0)]
    target_summary = target_summary.sort_values(["median_unc_uJy", "n_measurements"], ascending=[True, False])
    if args.max_targets:
        target_summary = target_summary.head(args.max_targets)
    return target_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a repeatable fake-line injection plan.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--campaign-id", help="Stable campaign id; default is generated from line grid.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, help="Plan JSON path.")
    parser.add_argument("--target-id", help="Comma-separated explicit target IDs.")
    parser.add_argument("--max-targets", type=int, default=10)
    parser.add_argument("--min-measurements", type=int, default=20)
    parser.add_argument("--g-min", type=float)
    parser.add_argument("--g-max", type=float)
    parser.add_argument("--ignore-flagged", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-wavelength-nm", type=float, default=1064.0)
    parser.add_argument(
        "--line-families",
        default="nd_yag_1064:1064",
        help="Comma list of name:nm entries, e.g. nd_yag_1064:1064,telecom_1550:1550.",
    )
    parser.add_argument("--offsets-nm", default="-20,-10,-5,0,5,10,20")
    parser.add_argument("--strengths-sigma", default="2,3,5,8,12,20")
    parser.add_argument("--line-width-nm", type=float, default=1.0)
    parser.add_argument("--min-response", type=float, default=1e-3)
    parser.add_argument("--max-frames-per-injection", type=int)
    parser.add_argument(
        "--allow-approx-wavelengths",
        action="store_true",
        help="Allow old MEF WCS-WAVE spectra. Not valid for science-grade injection/recovery.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    spectra_path = args.run_dir / "spectra" / "target_spectra.parquet"
    if not spectra_path.exists():
        raise SystemExit(f"Missing spectra parquet: {spectra_path}")
    df = pd.read_parquet(spectra_path)
    required = {"target_id", "cwave_um", "aperture_flux_uJy", "aperture_flux_unc_uJy", "input_file_path"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required spectra columns: {', '.join(missing)}")
    try:
        assert_science_wavelengths(df, spectra_path, allow_approx=args.allow_approx_wavelengths)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    targets = _choose_targets(df, args)
    if targets.empty:
        raise SystemExit("No targets matched plan selection")

    line_families = _parse_line_families(args.line_families)
    offsets_nm = _parse_csv_floats(args.offsets_nm)
    strengths_sigma = _parse_csv_floats(args.strengths_sigma)
    campaign_id = args.campaign_id or _safe_id(
        f"inj_{len(targets)}targets_{'_'.join(f['line_family'] for f in line_families)}"
    )
    campaign_root = args.output_root / campaign_id
    output = args.output or (campaign_root / "injection_plan.json")

    injections: list[dict[str, Any]] = []
    for _, target in targets.iterrows():
        target_id = str(target["target_id"])
        for family in line_families:
            nominal = float(family["nominal_line_nm"])
            for offset_nm in offsets_nm:
                injected_line_nm = nominal + float(offset_nm)
                for strength in strengths_sigma:
                    injection_id = _safe_id(
                        f"{target_id}_{family['line_family']}_{injected_line_nm:g}nm_s{strength:g}"
                    )
                    injections.append(
                        {
                            "injection_id": injection_id,
                            "target_id": target_id,
                            "line_family": family["line_family"],
                            "nominal_line_nm": nominal,
                            "injected_line_nm": injected_line_nm,
                            "offset_nm": float(offset_nm),
                            "line_width_nm": float(args.line_width_nm),
                            "strength_mode": "find_me_snr",
                            "find_me_snr": float(strength),
                            "min_response": float(args.min_response),
                            "max_frames": args.max_frames_per_injection,
                        }
                    )

    plan = {
        "plan_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": campaign_id,
        "baseline_run_dir": str(args.run_dir.resolve()),
        "campaign_root": str(campaign_root.resolve()),
        "spectra_path": str(spectra_path.resolve()),
        "selection": {
            "max_targets": args.max_targets,
            "min_measurements": args.min_measurements,
            "g_min": args.g_min,
            "g_max": args.g_max,
            "ignore_flagged": args.ignore_flagged,
            "require_wavelength_nm": args.require_wavelength_nm,
            "seed": args.seed,
        },
        "line_families": line_families,
        "offsets_nm": offsets_nm,
        "strengths_sigma": strengths_sigma,
        "targets": [
            {
                "target_id": str(row["target_id"]),
                "n_measurements": int(row["n_measurements"]),
                "wavelength_min_um": _finite_float(row["wavelength_min_um"]),
                "wavelength_max_um": _finite_float(row["wavelength_max_um"]),
                "median_unc_uJy": _finite_float(row["median_unc_uJy"]),
                "median_flux_uJy": _finite_float(row["median_flux_uJy"]),
                "phot_g_mean_mag": _finite_float(row.get("phot_g_mean_mag")),
                "usable_required_line_points": int(row.get("usable_required_line_points") or 0),
            }
            for _, row in targets.iterrows()
        ],
        "injections": injections,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "plan_path": str(output),
                "campaign_id": campaign_id,
                "target_count": len(plan["targets"]),
                "injection_count": len(injections),
                "campaign_root": str(campaign_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
