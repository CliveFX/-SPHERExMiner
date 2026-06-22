from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = Path("/mnt/niroseti/spherex_cache/runs/ucs0972_gpu_g14_16_n1000_f500")


def _parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _line_grid_from_plan(plan_path: Path) -> list[dict[str, Any]]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    seen: set[tuple[str, float, float]] = set()
    lines: list[dict[str, Any]] = []
    for injection in plan.get("injections", []):
        line_family = str(injection.get("line_family") or "custom")
        line_nm = float(injection["injected_line_nm"])
        line_width_nm = float(injection.get("line_width_nm", 1.0))
        key = (line_family, line_nm, line_width_nm)
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            {
                "line_family": line_family,
                "candidate_line_nm": line_nm,
                "line_width_nm": line_width_nm,
                "nominal_line_nm": injection.get("nominal_line_nm"),
                "offset_nm": injection.get("offset_nm"),
            }
        )
    return lines


def _line_grid_from_args(line_nm: str, line_width_nm: float) -> list[dict[str, Any]]:
    return [
        {
            "line_family": f"line_{wave:g}nm",
            "candidate_line_nm": wave,
            "line_width_nm": line_width_nm,
            "nominal_line_nm": wave,
            "offset_nm": 0.0,
        }
        for wave in _parse_csv_floats(line_nm)
    ]


def _response(cwave_um: np.ndarray, cband_um: np.ndarray, line_um: float, line_width_um: float) -> np.ndarray:
    fwhm = np.sqrt(np.maximum(cband_um, 0.0) ** 2 + max(line_width_um, 0.0) ** 2)
    sigma = fwhm / 2.354820045
    out = np.zeros_like(cwave_um, dtype=float)
    good = np.isfinite(cwave_um) & np.isfinite(sigma) & (sigma > 0.0)
    out[good] = np.exp(-0.5 * ((cwave_um[good] - line_um) / sigma[good]) ** 2)
    return out


def _continuum(
    wave_um: np.ndarray,
    flux: np.ndarray,
    template: np.ndarray,
    line_um: float,
    local_window_um: float,
    exclude_template_above: float,
) -> tuple[float, float, int]:
    local = np.isfinite(wave_um) & np.isfinite(flux) & (np.abs(wave_um - line_um) <= local_window_um)
    continuum_mask = local & (template <= exclude_template_above)
    values = flux[continuum_mask]
    if values.size < 5:
        values = flux[local & np.isfinite(flux)]
    if values.size < 5:
        values = flux[np.isfinite(flux)]
    if values.size == 0:
        return float("nan"), float("nan"), 0
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    robust_rms = 1.4826 * mad if mad > 0 else float(np.nanstd(values))
    return median, robust_rms, int(values.size)


def _score_one_target(
    target_id: str,
    rows: pd.DataFrame,
    lines: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = rows.sort_values("cwave_um").copy()
    if args.ignore_flagged and "fatal_flag_present" in rows.columns:
        rows = rows[~rows["fatal_flag_present"].fillna(False).astype(bool)]
    wave = pd.to_numeric(rows["cwave_um"], errors="coerce").to_numpy(dtype=float)
    cband = pd.to_numeric(rows["cband_um"], errors="coerce").to_numpy(dtype=float)
    flux = pd.to_numeric(rows["aperture_flux_uJy"], errors="coerce").to_numpy(dtype=float)
    unc = pd.to_numeric(rows["aperture_flux_unc_uJy"], errors="coerce").to_numpy(dtype=float)
    valid_base = np.isfinite(wave) & np.isfinite(cband) & np.isfinite(flux) & np.isfinite(unc) & (unc > 0.0)
    if np.count_nonzero(valid_base) < args.min_points:
        return []

    out: list[dict[str, Any]] = []
    for line in lines:
        line_nm = float(line["candidate_line_nm"])
        line_um = line_nm / 1000.0
        line_width_um = float(line.get("line_width_nm", args.line_width_nm)) / 1000.0
        template = _response(wave, cband, line_um, line_width_um)
        near = valid_base & (template >= args.min_template_response)
        if np.count_nonzero(near) < args.min_supporting_points:
            continue
        continuum, local_rms, continuum_points = _continuum(
            wave,
            flux,
            template,
            line_um,
            args.local_window_um,
            args.continuum_exclude_template_above,
        )
        if not math.isfinite(continuum):
            continue
        residual = flux - continuum
        weights = np.zeros_like(unc, dtype=float)
        weights[valid_base] = 1.0 / np.maximum(unc[valid_base], 1e-9) ** 2
        denom = float(np.sum(template[near] ** 2 * weights[near]))
        if denom <= 0.0 or not math.isfinite(denom):
            continue
        amp = float(np.sum(template[near] * residual[near] * weights[near]) / denom)
        amp_unc = float(1.0 / math.sqrt(denom))
        matched_snr = amp / amp_unc if amp_unc > 0.0 else float("nan")
        support_rows = rows.iloc[np.where(near)[0]].copy()
        best_rows = support_rows.assign(_template=template[near]).sort_values("_template", ascending=False).head(5)
        flagged_nearby = 0
        if "fatal_flag_present" in rows.columns:
            nearby = np.abs(wave - line_um) <= args.local_window_um
            flagged_nearby = int(np.count_nonzero(rows["fatal_flag_present"].fillna(False).astype(bool).to_numpy() & nearby))
        out.append(
            {
                "target_id": target_id,
                "candidate_line_nm": line_nm,
                "line_family": line.get("line_family"),
                "nominal_line_nm": line.get("nominal_line_nm"),
                "offset_nm": line.get("offset_nm"),
                "line_width_nm": float(line.get("line_width_nm", args.line_width_nm)),
                "matched_flux_uJy": amp,
                "matched_flux_unc_uJy": amp_unc,
                "matched_snr": float(matched_snr),
                "score": float(matched_snr),
                "n_supporting_points": int(np.count_nonzero(near)),
                "n_flagged_nearby": flagged_nearby,
                "local_continuum_uJy": continuum,
                "local_residual_rms_uJy": local_rms,
                "continuum_points": continuum_points,
                "wavelength_min_um": float(np.nanmin(wave[valid_base])),
                "wavelength_max_um": float(np.nanmax(wave[valid_base])),
                "best_frame_ids": ",".join(str(value) for value in best_rows.get("image_id", pd.Series(dtype=str)).tolist()),
                "detectors": ",".join(
                    sorted({str(int(value)) for value in support_rows.get("detector", pd.Series(dtype=float)).dropna().tolist()})
                ),
                "candidate_status": "candidate" if matched_snr >= args.min_snr else "below_threshold",
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify assembled SPHEREx spectra with a simple matched filter.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--spectra-path", type=Path)
    parser.add_argument("--plan", type=Path, help="Injection plan JSON; candidate line grid is derived from it.")
    parser.add_argument("--line-nm", default="1064", help="Comma-separated candidate wavelengths if no plan is supplied.")
    parser.add_argument("--line-width-nm", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ignore-flagged", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-points", type=int, default=8)
    parser.add_argument("--min-supporting-points", type=int, default=1)
    parser.add_argument("--min-template-response", type=float, default=1e-3)
    parser.add_argument("--local-window-um", type=float, default=0.18)
    parser.add_argument("--continuum-exclude-template-above", type=float, default=0.25)
    parser.add_argument("--min-snr", type=float, default=5.0)
    parser.add_argument("--max-targets", type=int)
    args = parser.parse_args()

    spectra_path = args.spectra_path or (args.run_dir / "spectra" / "target_spectra.parquet")
    if not spectra_path.exists():
        raise SystemExit(f"Missing spectra parquet: {spectra_path}")
    lines = _line_grid_from_plan(args.plan) if args.plan else _line_grid_from_args(args.line_nm, args.line_width_nm)
    if not lines:
        raise SystemExit("No candidate lines configured")

    df = pd.read_parquet(spectra_path)
    required = {"target_id", "cwave_um", "cband_um", "aperture_flux_uJy", "aperture_flux_unc_uJy"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required spectra columns: {', '.join(missing)}")
    target_ids = list(df["target_id"].dropna().astype(str).drop_duplicates())
    if args.max_targets:
        target_ids = target_ids[: int(args.max_targets)]

    results: list[dict[str, Any]] = []
    grouped = df[df["target_id"].astype(str).isin(target_ids)].groupby("target_id", sort=False)
    for index, (target_id, rows) in enumerate(grouped, start=1):
        if index % 250 == 0:
            print(f"classified {index}/{len(target_ids)} targets", flush=True)
        results.extend(_score_one_target(str(target_id), rows, lines, args))

    out_df = pd.DataFrame(results)
    output_dir = args.output_dir or (args.run_dir / "classifier")
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "matched_filter_scores.parquet"
    candidates_path = output_dir / "matched_filter_candidates.parquet"
    summary_path = output_dir / "matched_filter_summary.json"
    out_df.to_parquet(scores_path, index=False)
    if out_df.empty:
        candidates = out_df
    else:
        candidates = out_df[out_df["matched_snr"] >= args.min_snr].sort_values("matched_snr", ascending=False)
    candidates.to_parquet(candidates_path, index=False)
    summary = {
        "run_dir": str(args.run_dir),
        "spectra_path": str(spectra_path),
        "plan_path": str(args.plan) if args.plan else None,
        "line_count": len(lines),
        "target_count": len(target_ids),
        "score_rows": int(len(out_df)),
        "candidate_rows": int(len(candidates)),
        "min_snr": args.min_snr,
        "scores_path": str(scores_path),
        "candidates_path": str(candidates_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
