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

from tools.classify_paired_delta_matched_filter import _load_pair  # noqa: E402
from tools.classify_spectra_matched_filter import _response  # noqa: E402
from tools.wavelength_guard import assert_science_wavelengths  # noqa: E402


def _line_grid(min_nm: float, max_nm: float, step_nm: float, width_nm: float) -> list[dict[str, float | str]]:
    if step_nm <= 0.0:
        raise ValueError("--grid-step-nm must be positive")
    values = np.arange(min_nm, max_nm + step_nm * 0.5, step_nm, dtype=float)
    return [
        {
            "line_family": "blind",
            "candidate_line_nm": float(round(value, 6)),
            "line_width_nm": float(width_nm),
            "nominal_line_nm": float(round(value, 6)),
            "offset_nm": 0.0,
        }
        for value in values
    ]


def _flux_columns(flux_kind: str) -> tuple[str, str]:
    if flux_kind == "psf":
        return "psf_flux_uJy", "psf_flux_unc_uJy"
    return "aperture_flux_uJy", "aperture_flux_unc_uJy"


def _load_raw_spectra(args: argparse.Namespace) -> pd.DataFrame:
    spectra_path = args.spectra_path or (args.run_dir / "spectra" / "target_spectra.parquet")
    if not spectra_path.exists():
        raise SystemExit(f"Missing spectra parquet: {spectra_path}")
    df = pd.read_parquet(spectra_path)
    try:
        assert_science_wavelengths(df, spectra_path, allow_approx=args.allow_approx_wavelengths)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    flux_col, unc_col = _flux_columns(args.flux_kind)
    required = {"target_id", "image_id", "cwave_um", "cband_um", flux_col, unc_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required spectra columns: {', '.join(missing)}")
    cols = [
        "target_id",
        "image_id",
        "cwave_um",
        "cband_um",
        flux_col,
        unc_col,
        "fatal_flag_present",
        "detector",
        "observation_id",
        "input_file_path",
        "original_input_file_path",
        "path_override_applied",
    ]
    df = df[[col for col in cols if col in df.columns]].copy()
    df["score_flux_uJy"] = pd.to_numeric(df[flux_col], errors="coerce")
    df["score_flux_unc_uJy"] = pd.to_numeric(df[unc_col], errors="coerce")
    df["score_mode"] = "raw_spectrum"
    return df


def _load_scoring_table(args: argparse.Namespace) -> pd.DataFrame:
    if args.baseline_run_dir and args.injected_run_dir:
        paired = _load_pair(args)
        paired["score_flux_uJy"] = pd.to_numeric(paired["delta_flux_uJy"], errors="coerce")
        paired["score_flux_unc_uJy"] = pd.to_numeric(paired["delta_flux_unc_uJy"], errors="coerce")
        paired["score_mode"] = "paired_delta"
        return paired
    return _load_raw_spectra(args)


def _robust_continuum(
    flux: np.ndarray,
    wave_um: np.ndarray,
    template: np.ndarray,
    line_um: float,
    local_window_um: float,
    exclude_template_above: float,
) -> tuple[float, float, int]:
    local = np.isfinite(wave_um) & np.isfinite(flux) & (np.abs(wave_um - line_um) <= local_window_um)
    continuum_mask = local & (template <= exclude_template_above)
    values = flux[continuum_mask]
    if values.size < 3:
        values = flux[local & np.isfinite(flux)]
    if values.size < 3:
        values = flux[np.isfinite(flux)]
    if values.size == 0:
        return float("nan"), float("nan"), 0
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    rms = 1.4826 * mad if mad > 0 else float(np.nanstd(values))
    return median, rms, int(values.size)


def _score_one_target(target_id: str, rows: pd.DataFrame, lines: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = rows.sort_values("cwave_um").copy()
    if args.ignore_flagged and "fatal_flag_present" in rows.columns:
        rows = rows[~rows["fatal_flag_present"].fillna(False).astype(bool)]
    wave = pd.to_numeric(rows["cwave_um"], errors="coerce").to_numpy(dtype=float)
    cband = pd.to_numeric(rows["cband_um"], errors="coerce").to_numpy(dtype=float)
    flux = pd.to_numeric(rows["score_flux_uJy"], errors="coerce").to_numpy(dtype=float)
    unc = pd.to_numeric(rows["score_flux_unc_uJy"], errors="coerce").to_numpy(dtype=float)
    image_ids = rows.get("image_id", pd.Series([""] * len(rows), index=rows.index)).astype(str).to_numpy()
    if "detector" in rows:
        detectors_raw = pd.to_numeric(rows["detector"], errors="coerce").to_numpy(dtype=float)
    else:
        detectors_raw = np.full(len(rows), np.nan, dtype=float)
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
        continuum, local_rms, continuum_points = _robust_continuum(
            flux,
            wave,
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
        if not math.isfinite(matched_snr):
            continue
        near_idx = np.where(near)[0]
        best_idx = near_idx[np.argsort(template[near_idx])[::-1][:5]]
        best_frame_ids = ",".join(str(image_ids[i]) for i in best_idx if str(image_ids[i]))
        detector_values = sorted({str(int(detectors_raw[i])) for i in near_idx if np.isfinite(detectors_raw[i])})
        flagged_nearby = 0
        if "fatal_flag_present" in rows.columns:
            nearby = np.abs(wave - line_um) <= args.local_window_um
            flagged_nearby = int(np.count_nonzero(rows["fatal_flag_present"].fillna(False).astype(bool).to_numpy() & nearby))
        if args.save_all_scores or matched_snr >= args.min_snr:
            out.append(
                {
                    "target_id": target_id,
                    "candidate_line_nm": line_nm,
                    "line_family": "blind",
                    "nominal_line_nm": line_nm,
                    "offset_nm": 0.0,
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
                    "best_frame_ids": best_frame_ids,
                    "detectors": ",".join(detector_values),
                    "candidate_status": "candidate" if matched_snr >= args.min_snr else "below_threshold",
                    "score_mode": str(rows["score_mode"].iloc[0]) if "score_mode" in rows and len(rows) else "unknown",
                    "flux_kind": args.flux_kind,
                }
            )
    return out


def _cluster_candidates(candidates: pd.DataFrame, cluster_gap_nm: float) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    work = candidates.copy()
    work["candidate_line_nm"] = pd.to_numeric(work["candidate_line_nm"], errors="coerce")
    work["matched_snr"] = pd.to_numeric(work["matched_snr"], errors="coerce")
    work = work.dropna(subset=["target_id", "candidate_line_nm", "matched_snr"])
    for target_id, group in work.sort_values(["target_id", "candidate_line_nm"]).groupby("target_id", sort=False):
        cluster: list[pd.Series] = []
        last_nm: float | None = None
        cluster_index = 0
        for _, row in group.iterrows():
            line_nm = float(row["candidate_line_nm"])
            if cluster and last_nm is not None and line_nm - last_nm > cluster_gap_nm:
                rows.append(_cluster_row(str(target_id), cluster_index, cluster))
                cluster = []
                cluster_index += 1
            cluster.append(row)
            last_nm = line_nm
        if cluster:
            rows.append(_cluster_row(str(target_id), cluster_index, cluster))
    return pd.DataFrame(rows).sort_values("peak_snr", ascending=False, na_position="last") if rows else pd.DataFrame()


def _cluster_row(target_id: str, cluster_index: int, cluster: list[pd.Series]) -> dict[str, Any]:
    df = pd.DataFrame([row.to_dict() for row in cluster])
    peak = df.sort_values("matched_snr", ascending=False).iloc[0]
    detectors = sorted({str(value) for text in df.get("detectors", pd.Series(dtype=str)).dropna().astype(str) for value in text.split(",") if value})
    frames = []
    for text in df.get("best_frame_ids", pd.Series(dtype=str)).dropna().astype(str):
        frames.extend([part for part in text.split(",") if part])
    unique_frames = list(dict.fromkeys(frames))
    min_nm = float(pd.to_numeric(df["candidate_line_nm"], errors="coerce").min())
    max_nm = float(pd.to_numeric(df["candidate_line_nm"], errors="coerce").max())
    return {
        "cluster_id": f"{target_id}::blind::{cluster_index}::{float(peak['candidate_line_nm']):g}",
        "target_id": target_id,
        "peak_line_nm": float(peak["candidate_line_nm"]),
        "peak_snr": float(peak["matched_snr"]),
        "peak_flux_uJy": float(peak["matched_flux_uJy"]),
        "peak_flux_unc_uJy": float(peak["matched_flux_unc_uJy"]),
        "line_min_nm": min_nm,
        "line_max_nm": max_nm,
        "cluster_width_nm": max_nm - min_nm,
        "score_count": int(len(df)),
        "supporting_points_max": int(pd.to_numeric(df.get("n_supporting_points"), errors="coerce").max()),
        "flagged_points_sum": int(pd.to_numeric(df.get("n_flagged_nearby"), errors="coerce").fillna(0).sum()),
        "detectors": ",".join(detectors),
        "best_frame_ids": ",".join(unique_frames[:12]),
        "score_mode": peak.get("score_mode"),
        "flux_kind": peak.get("flux_kind"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind dense-wavelength matched-filter scan over recovered SPHEREx spectra.")
    parser.add_argument("--run-dir", type=Path, help="Raw spectra run directory.")
    parser.add_argument("--spectra-path", type=Path)
    parser.add_argument("--baseline-run-dir", type=Path, help="Baseline run for paired-delta mode.")
    parser.add_argument("--injected-run-dir", type=Path, help="Injected run for paired-delta mode.")
    parser.add_argument("--baseline-spectra", type=Path)
    parser.add_argument("--injected-spectra", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-line-nm", type=float, default=750.0)
    parser.add_argument("--max-line-nm", type=float, default=5000.0)
    parser.add_argument("--grid-step-nm", type=float, default=5.0)
    parser.add_argument("--line-width-nm", type=float, default=1.0)
    parser.add_argument("--cluster-gap-nm", type=float, default=15.0)
    parser.add_argument("--flux-kind", choices=["aperture", "psf"], default="aperture")
    parser.add_argument("--ignore-flagged", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--uncertainty-mode", choices=["quadrature", "injected", "baseline"], default="quadrature")
    parser.add_argument("--min-points", type=int, default=8)
    parser.add_argument("--min-supporting-points", type=int, default=2)
    parser.add_argument("--min-template-response", type=float, default=1e-3)
    parser.add_argument("--local-window-um", type=float, default=0.18)
    parser.add_argument("--continuum-exclude-template-above", type=float, default=0.25)
    parser.add_argument("--min-snr", type=float, default=5.0)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--target-id", action="append", help="Restrict scan to one target id; may be repeated.")
    parser.add_argument("--target-ids-file", type=Path, help="Text file with one target id per line.")
    parser.add_argument("--target-ids-from-manifest", type=Path, help="Restrict scan to target ids present in an injection manifest JSON.")
    parser.add_argument("--save-all-scores", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-approx-wavelengths", action="store_true")
    args = parser.parse_args()

    if not args.run_dir and not (args.baseline_run_dir and args.injected_run_dir):
        raise SystemExit("Provide either --run-dir or both --baseline-run-dir and --injected-run-dir")
    if args.run_dir and (args.baseline_run_dir or args.injected_run_dir):
        raise SystemExit("Use raw mode or paired-delta mode, not both")

    lines = _line_grid(args.min_line_nm, args.max_line_nm, args.grid_step_nm, args.line_width_nm)
    df = _load_scoring_table(args)
    target_ids = list(df["target_id"].dropna().astype(str).drop_duplicates())
    requested_targets = set(args.target_id or [])
    if args.target_ids_file:
        requested_targets.update(
            line.strip()
            for line in args.target_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if args.target_ids_from_manifest:
        manifest = json.loads(args.target_ids_from_manifest.read_text(encoding="utf-8"))
        requested_targets.update(
            str(injection.get("target_id"))
            for injection in manifest.get("injections", [])
            if injection.get("target_id")
        )
    if requested_targets:
        target_ids = [target_id for target_id in target_ids if target_id in requested_targets]
    if args.max_targets:
        target_ids = target_ids[: int(args.max_targets)]

    results: list[dict[str, Any]] = []
    grouped = df[df["target_id"].astype(str).isin(target_ids)].groupby("target_id", sort=False)
    for index, (target_id, rows) in enumerate(grouped, start=1):
        if index == 1 or index % 100 == 0:
            print(f"blind-scanned {index}/{len(target_ids)} targets; rows={len(results)}", flush=True)
        results.extend(_score_one_target(str(target_id), rows, lines, args))

    out_df = pd.DataFrame(results)
    candidates = (
        out_df[pd.to_numeric(out_df.get("matched_snr"), errors="coerce") >= args.min_snr]
        .sort_values("matched_snr", ascending=False)
        if not out_df.empty
        else pd.DataFrame()
    )
    clusters = _cluster_candidates(candidates, args.cluster_gap_nm)

    default_parent = args.run_dir if args.run_dir else args.injected_run_dir
    default_name = "blind_classifier" if args.run_dir else "blind_classifier_paired_delta"
    output_dir = args.output_dir or (default_parent / default_name)  # type: ignore[operator]
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / "blind_line_grid.parquet"
    scores_path = output_dir / "blind_matched_filter_scores.parquet"
    candidates_path = output_dir / "blind_matched_filter_candidates.parquet"
    clusters_path = output_dir / "blind_candidate_clusters.parquet"
    summary_path = output_dir / "blind_classifier_summary.json"

    pd.DataFrame(lines).to_parquet(grid_path, index=False)
    out_df.to_parquet(scores_path, index=False)
    candidates.to_parquet(candidates_path, index=False)
    clusters.to_parquet(clusters_path, index=False)

    summary = {
        "mode": "raw_spectrum" if args.run_dir else "paired_delta",
        "run_dir": str(args.run_dir) if args.run_dir else None,
        "baseline_run_dir": str(args.baseline_run_dir) if args.baseline_run_dir else None,
        "injected_run_dir": str(args.injected_run_dir) if args.injected_run_dir else None,
        "line_count": len(lines),
        "target_count": len(target_ids),
        "score_rows": int(len(out_df)),
        "candidate_rows": int(len(candidates)),
        "cluster_rows": int(len(clusters)),
        "min_snr": args.min_snr,
        "grid_step_nm": args.grid_step_nm,
        "line_width_nm": args.line_width_nm,
        "cluster_gap_nm": args.cluster_gap_nm,
        "flux_kind": args.flux_kind,
        "ignore_flagged": args.ignore_flagged,
        "save_all_scores": args.save_all_scores,
        "grid_path": str(grid_path),
        "scores_path": str(scores_path),
        "candidates_path": str(candidates_path),
        "clusters_path": str(clusters_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
