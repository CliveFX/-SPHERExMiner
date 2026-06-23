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

from tools.classify_spectra_matched_filter import _line_grid_from_args, _line_grid_from_plan, _response


BASELINE_RUN = Path("/mnt/niroseti/spherex_cache/runs/injrec_baseline_10k_f80_g14_16_gpu3")
INJECTED_RUN = Path("/mnt/niroseti/spherex_cache/runs/injrec_injected_10k_f80_mixed_lasers_gpu3")


def _continuum_delta(
    wave_um: np.ndarray,
    delta_flux: np.ndarray,
    template: np.ndarray,
    line_um: float,
    local_window_um: float,
    exclude_template_above: float,
) -> tuple[float, float, int]:
    local = np.isfinite(wave_um) & np.isfinite(delta_flux) & (np.abs(wave_um - line_um) <= local_window_um)
    continuum_mask = local & (template <= exclude_template_above)
    values = delta_flux[continuum_mask]
    if values.size < 3:
        values = delta_flux[local & np.isfinite(delta_flux)]
    if values.size < 3:
        values = delta_flux[np.isfinite(delta_flux)]
    if values.size == 0:
        return float("nan"), float("nan"), 0
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    robust_rms = 1.4826 * mad if mad > 0 else float(np.nanstd(values))
    return median, robust_rms, int(values.size)


def _load_pair(args: argparse.Namespace) -> pd.DataFrame:
    base_path = args.baseline_spectra or (args.baseline_run_dir / "spectra" / "target_spectra.parquet")
    inj_path = args.injected_spectra or (args.injected_run_dir / "spectra" / "target_spectra.parquet")
    if not base_path.exists():
        raise SystemExit(f"Missing baseline spectra parquet: {base_path}")
    if not inj_path.exists():
        raise SystemExit(f"Missing injected spectra parquet: {inj_path}")

    columns = [
        "target_id",
        "image_id",
        "cwave_um",
        "cband_um",
        "aperture_flux_uJy",
        "aperture_flux_unc_uJy",
        "fatal_flag_present",
        "detector",
        "observation_id",
        "input_file_path",
        "original_input_file_path",
        "path_override_applied",
    ]
    base = pd.read_parquet(base_path)
    inj = pd.read_parquet(inj_path)
    base = base[[col for col in columns if col in base.columns]].copy()
    inj = inj[[col for col in columns if col in inj.columns]].copy()
    required = {"target_id", "image_id", "cwave_um", "cband_um", "aperture_flux_uJy", "aperture_flux_unc_uJy"}
    missing_base = sorted(required - set(base.columns))
    missing_inj = sorted(required - set(inj.columns))
    if missing_base:
        raise SystemExit(f"Baseline spectra missing required columns: {', '.join(missing_base)}")
    if missing_inj:
        raise SystemExit(f"Injected spectra missing required columns: {', '.join(missing_inj)}")

    key = ["target_id", "image_id"]
    base = base.drop_duplicates(key, keep="first")
    inj = inj.drop_duplicates(key, keep="first")
    paired = inj.merge(base, on=key, how="inner", suffixes=("_inj", "_base"))
    if paired.empty:
        raise SystemExit("No paired rows after target_id/image_id join")

    paired["cwave_um"] = pd.to_numeric(paired["cwave_um_inj"], errors="coerce")
    paired["cband_um"] = pd.to_numeric(paired["cband_um_inj"], errors="coerce")
    paired["delta_flux_uJy"] = (
        pd.to_numeric(paired["aperture_flux_uJy_inj"], errors="coerce")
        - pd.to_numeric(paired["aperture_flux_uJy_base"], errors="coerce")
    )
    inj_unc = pd.to_numeric(paired["aperture_flux_unc_uJy_inj"], errors="coerce")
    base_unc = pd.to_numeric(paired["aperture_flux_unc_uJy_base"], errors="coerce")
    if args.uncertainty_mode == "injected":
        paired["delta_flux_unc_uJy"] = inj_unc
    elif args.uncertainty_mode == "baseline":
        paired["delta_flux_unc_uJy"] = base_unc
    else:
        paired["delta_flux_unc_uJy"] = np.sqrt(inj_unc**2 + base_unc**2)
    paired["fatal_flag_present"] = False
    for col in ("fatal_flag_present_inj", "fatal_flag_present_base"):
        if col in paired.columns:
            paired["fatal_flag_present"] |= paired[col].fillna(False).astype(bool)
    for col in ("detector", "observation_id", "input_file_path", "original_input_file_path", "path_override_applied"):
        inj_col = f"{col}_inj"
        base_col = f"{col}_base"
        if inj_col in paired.columns:
            paired[col] = paired[inj_col]
        elif base_col in paired.columns:
            paired[col] = paired[base_col]
    paired["paired_row_count"] = len(paired)
    return paired


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
    flux = pd.to_numeric(rows["delta_flux_uJy"], errors="coerce").to_numpy(dtype=float)
    unc = pd.to_numeric(rows["delta_flux_unc_uJy"], errors="coerce").to_numpy(dtype=float)
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
        continuum, local_rms, continuum_points = _continuum_delta(
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
                "score_mode": "paired_delta",
                "uncertainty_mode": args.uncertainty_mode,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify injected-minus-baseline paired spectra with a matched filter.")
    parser.add_argument("--baseline-run-dir", type=Path, default=BASELINE_RUN)
    parser.add_argument("--injected-run-dir", type=Path, default=INJECTED_RUN)
    parser.add_argument("--baseline-spectra", type=Path)
    parser.add_argument("--injected-spectra", type=Path)
    parser.add_argument("--plan", type=Path, help="Injection plan JSON; candidate line grid is derived from it.")
    parser.add_argument("--line-nm", default="1064", help="Comma-separated candidate wavelengths if no plan is supplied.")
    parser.add_argument("--line-width-nm", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ignore-flagged", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--uncertainty-mode", choices=["quadrature", "injected", "baseline"], default="quadrature")
    parser.add_argument("--min-points", type=int, default=8)
    parser.add_argument("--min-supporting-points", type=int, default=1)
    parser.add_argument("--min-template-response", type=float, default=1e-3)
    parser.add_argument("--local-window-um", type=float, default=0.18)
    parser.add_argument("--continuum-exclude-template-above", type=float, default=0.25)
    parser.add_argument("--min-snr", type=float, default=5.0)
    parser.add_argument("--max-targets", type=int)
    args = parser.parse_args()

    lines = _line_grid_from_plan(args.plan) if args.plan else _line_grid_from_args(args.line_nm, args.line_width_nm)
    if not lines:
        raise SystemExit("No candidate lines configured")
    paired = _load_pair(args)
    target_ids = list(paired["target_id"].dropna().astype(str).drop_duplicates())
    if args.max_targets:
        target_ids = target_ids[: int(args.max_targets)]

    results: list[dict[str, Any]] = []
    grouped = paired[paired["target_id"].astype(str).isin(target_ids)].groupby("target_id", sort=False)
    for index, (target_id, rows) in enumerate(grouped, start=1):
        if index % 250 == 0:
            print(f"classified paired delta {index}/{len(target_ids)} targets", flush=True)
        results.extend(_score_one_target(str(target_id), rows, lines, args))

    out_df = pd.DataFrame(results)
    output_dir = args.output_dir or (args.injected_run_dir / "classifier_paired_delta")
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "matched_filter_scores.parquet"
    candidates_path = output_dir / "matched_filter_candidates.parquet"
    paired_path = output_dir / "paired_delta_measurements.parquet"
    summary_path = output_dir / "matched_filter_summary.json"
    paired.to_parquet(paired_path, index=False)
    out_df.to_parquet(scores_path, index=False)
    if out_df.empty:
        candidates = out_df
    else:
        candidates = out_df[out_df["matched_snr"] >= args.min_snr].sort_values("matched_snr", ascending=False)
    candidates.to_parquet(candidates_path, index=False)
    summary = {
        "baseline_run_dir": str(args.baseline_run_dir),
        "injected_run_dir": str(args.injected_run_dir),
        "plan_path": str(args.plan) if args.plan else None,
        "line_count": len(lines),
        "target_count": len(target_ids),
        "paired_measurement_rows": int(len(paired)),
        "score_rows": int(len(out_df)),
        "candidate_rows": int(len(candidates)),
        "min_snr": args.min_snr,
        "ignore_flagged": args.ignore_flagged,
        "uncertainty_mode": args.uncertainty_mode,
        "paired_path": str(paired_path),
        "scores_path": str(scores_path),
        "candidates_path": str(candidates_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
