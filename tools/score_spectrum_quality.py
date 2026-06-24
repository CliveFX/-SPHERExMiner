#!/usr/bin/env python3
"""Score recovered target spectra for basic visual/science usability.

This is deliberately not ML. It is a deterministic ranking heuristic for
separating smooth, low-flag, well-measured spectra from noisy or broken ones.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RUN_ROOT = Path("/mnt/niroseti/spherex_cache/runs")


def _finite_array(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def _robust_scale(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return float("nan")
    med = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - med)))
    if math.isfinite(mad) and mad > 0:
        return 1.4826 * mad
    std = float(np.nanstd(finite))
    return std if math.isfinite(std) and std > 0 else float("nan")


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _smoothness_score(wave: np.ndarray, flux: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(wave) & np.isfinite(flux)
    if mask.sum() < 8:
        return 0.0, float("nan")
    order = np.argsort(wave[mask])
    y = flux[mask][order]
    scale = _robust_scale(y)
    if not math.isfinite(scale) or scale <= 0:
        return 0.5, 0.0

    # Local jaggedness: compare each point to the average of its immediate
    # neighbors. Smooth stellar continua have low curvature in this metric.
    residual = y[1:-1] - 0.5 * (y[:-2] + y[2:])
    roughness = float(np.nanmedian(np.abs(residual)) / scale) if residual.size else float("nan")
    score = 1.0 / (1.0 + max(roughness, 0.0))
    return _clip01(score), roughness


def _agreement_score(aperture: np.ndarray, psf: np.ndarray, flags: np.ndarray) -> tuple[float, float, float]:
    mask = np.isfinite(aperture) & np.isfinite(psf) & ~flags
    if mask.sum() < 8:
        return 0.55, float("nan"), float("nan")
    ap = aperture[mask]
    pf = psf[mask]
    ap_scale = _robust_scale(ap)
    pf_scale = _robust_scale(pf)
    if not math.isfinite(ap_scale) or not math.isfinite(pf_scale) or ap_scale <= 0 or pf_scale <= 0:
        return 0.55, float("nan"), float("nan")
    ap_z = (ap - np.nanmedian(ap)) / ap_scale
    pf_z = (pf - np.nanmedian(pf)) / pf_scale
    corr = float(np.corrcoef(ap_z, pf_z)[0, 1]) if mask.sum() >= 3 else float("nan")
    corr_score = _clip01((corr + 1.0) / 2.0) if math.isfinite(corr) else 0.55

    denom = np.maximum(np.abs(ap), np.abs(pf))
    good = np.isfinite(denom) & (denom > 0)
    frac_delta = float(np.nanmedian(np.abs(ap[good] - pf[good]) / denom[good])) if good.any() else float("nan")
    delta_score = 1.0 / (1.0 + 2.0 * max(frac_delta, 0.0)) if math.isfinite(frac_delta) else 0.55
    return _clip01(0.65 * corr_score + 0.35 * delta_score), corr, frac_delta


def _snr_score(flux: np.ndarray, unc: np.ndarray, flags: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(flux) & np.isfinite(unc) & (unc > 0) & ~flags
    if mask.sum() < 5:
        return 0.0, float("nan")
    snr = np.abs(flux[mask] / unc[mask])
    median_snr = float(np.nanmedian(snr))
    # This is a quality floor, not a signal detector. Saturate once the spectrum
    # is clearly measured above the uncertainty scale.
    return _clip01(math.log10(max(median_snr, 1.0)) / math.log10(20.0)), median_snr


def score_one_target(target_id: str, rows: pd.DataFrame) -> dict[str, Any]:
    rows = rows.sort_values("cwave_um")
    wave = _finite_array(rows["cwave_um"]) * 1000.0
    aperture = _finite_array(rows["aperture_flux_uJy"])
    aperture_unc = _finite_array(rows["aperture_flux_unc_uJy"]) if "aperture_flux_unc_uJy" in rows else np.full(len(rows), np.nan)
    psf = _finite_array(rows["psf_flux_uJy"]) if "psf_flux_uJy" in rows else np.full(len(rows), np.nan)
    flags = rows["fatal_flag_present"].fillna(False).astype(bool).to_numpy() if "fatal_flag_present" in rows else np.zeros(len(rows), dtype=bool)

    finite_ap = np.isfinite(wave) & np.isfinite(aperture)
    usable = finite_ap & ~flags
    n_measurements = int(len(rows))
    n_usable = int(usable.sum())
    flag_fraction = float(flags.sum() / n_measurements) if n_measurements else 1.0
    coverage_score = _clip01(n_usable / 80.0)
    flag_score = _clip01(1.0 - 2.0 * flag_fraction)
    smooth_score, roughness = _smoothness_score(wave[usable], aperture[usable])
    agreement_score, ap_psf_corr, ap_psf_frac_delta = _agreement_score(aperture, psf, flags)
    snr_score, median_snr = _snr_score(aperture, aperture_unc, flags)

    total = 100.0 * (
        0.28 * flag_score
        + 0.24 * smooth_score
        + 0.20 * coverage_score
        + 0.18 * agreement_score
        + 0.10 * snr_score
    )
    reasons: list[str] = []
    if n_usable < 25:
        reasons.append("too_few_usable_points")
    if flag_fraction > 0.25:
        reasons.append("too_many_fatal_flags")
    if smooth_score < 0.45:
        reasons.append("jagged_or_noisy_continuum")
    if math.isfinite(ap_psf_corr) and ap_psf_corr < 0.25:
        reasons.append("low_aperture_psf_shape_agreement")
    if math.isfinite(ap_psf_frac_delta) and ap_psf_frac_delta > 1.5:
        reasons.append("large_aperture_psf_flux_disagreement")
    if math.isfinite(median_snr) and median_snr < 1.0:
        reasons.append("low_median_snr")

    hard_reasons = {"too_few_usable_points", "too_many_fatal_flags", "jagged_or_noisy_continuum"} & set(reasons)
    if total >= 70 and not hard_reasons:
        category = "good"
    elif total >= 45 and flag_fraction <= 0.35 and n_usable >= 25:
        category = "review"
    else:
        category = "bad"

    first = rows.iloc[0]
    point_injected = _target_point_injection_mask(rows)
    point_injected_count = int(point_injected.sum())
    run_injection_applied = _first_or_none(first, "run_injection_applied")
    if run_injection_applied is None:
        run_injection_applied = bool(point_injected_count)
    return {
        "target_id": str(target_id),
        "target_type": first.get("target_type"),
        "object_name": first.get("object_name"),
        "run_kind": _first_or_none(first, "run_kind"),
        "run_injection_applied": bool(run_injection_applied),
        "point_injection_applied_count": point_injected_count,
        "point_injection_applied_fraction": float(point_injected_count / n_measurements) if n_measurements else 0.0,
        "injection_manifest_path": _first_or_none(first, "injection_manifest_path"),
        "phot_g_mean_mag": first.get("phot_g_mean_mag"),
        "bp_rp": first.get("bp_rp"),
        "spectrum_quality_score": float(total),
        "spectrum_quality_category": category,
        "spectrum_quality_reasons": ",".join(reasons),
        "n_measurements": n_measurements,
        "n_usable_measurements": n_usable,
        "flag_fraction": flag_fraction,
        "coverage_score": coverage_score,
        "flag_score": flag_score,
        "smoothness_score": smooth_score,
        "roughness_mad_ratio": roughness,
        "aperture_psf_agreement_score": agreement_score,
        "aperture_psf_corr": ap_psf_corr,
        "aperture_psf_median_frac_delta": ap_psf_frac_delta,
        "median_abs_aperture_snr": median_snr,
        "snr_score": snr_score,
        "wavelength_min_nm": float(np.nanmin(wave)) if np.isfinite(wave).any() else float("nan"),
        "wavelength_max_nm": float(np.nanmax(wave)) if np.isfinite(wave).any() else float("nan"),
    }


def _first_or_none(row: pd.Series, column: str) -> Any:
    if column not in row:
        return None
    value = row.get(column)
    return None if pd.isna(value) else value


def _target_point_injection_mask(rows: pd.DataFrame) -> np.ndarray:
    injected = np.zeros(len(rows), dtype=bool)
    for column in ("point_injection_applied", "injection_applied", "path_override_applied"):
        if column in rows:
            injected |= rows[column].fillna(False).astype(bool).to_numpy()
    if {"input_file_path", "original_input_file_path"} <= set(rows.columns):
        input_path = rows["input_file_path"].fillna("").astype(str)
        original_path = rows["original_input_file_path"].fillna("").astype(str)
        injected |= (original_path.ne("") & input_path.ne(original_path)).to_numpy()
    return injected


def score_spectra_table(spectra: pd.DataFrame) -> pd.DataFrame:
    required = {"target_id", "cwave_um", "aperture_flux_uJy"}
    missing = sorted(required - set(spectra.columns))
    if missing:
        raise ValueError(f"missing required spectra columns: {missing}")
    rows = [score_one_target(str(target_id), group) for target_id, group in spectra.groupby("target_id", dropna=False)]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["spectrum_quality_category", "spectrum_quality_score", "n_usable_measurements"],
        ascending=[True, False, False],
        kind="mergesort",
    )


def score_run(run_dir: Path) -> pd.DataFrame:
    spectra_path = run_dir / "spectra" / "target_spectra.parquet"
    if not spectra_path.exists():
        raise FileNotFoundError(spectra_path)
    spectra = pd.read_parquet(spectra_path)
    scored = score_spectra_table(spectra)
    scored.insert(0, "run_name", run_dir.name)
    scored.insert(1, "run_dir", str(run_dir))
    out_path = run_dir / "spectra" / "spectrum_quality.parquet"
    scored.to_parquet(out_path, index=False)
    summary = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "target_count": int(len(scored)),
        "category_counts": scored["spectrum_quality_category"].value_counts().sort_index().to_dict()
        if not scored.empty
        else {},
        "output": str(out_path),
    }
    (run_dir / "spectra" / "spectrum_quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return scored


def _run_dirs_from_args(args: argparse.Namespace) -> list[Path]:
    dirs: list[Path] = []
    for run in args.run_dir or []:
        dirs.append(run)
    if args.campaign:
        dirs.extend(sorted(path for path in args.run_root.glob(f"{args.campaign}_*") if (path / "spectra" / "target_spectra.parquet").exists()))
    return list(dict.fromkeys(dirs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", help="Run directory containing spectra/target_spectra.parquet.")
    parser.add_argument("--campaign", help="Score every matching run under --run-root.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--combined-output", type=Path, help="Optional combined .parquet or .csv output.")
    parser.add_argument("--top", type=int, default=20, help="Print this many best spectra after scoring.")
    args = parser.parse_args()

    run_dirs = _run_dirs_from_args(args)
    if not run_dirs:
        raise SystemExit("Provide --run-dir or --campaign")

    tables = []
    for run_dir in run_dirs:
        try:
            scored = score_run(run_dir)
        except Exception as exc:
            print(json.dumps({"run_dir": str(run_dir), "error": str(exc)}), flush=True)
            continue
        tables.append(scored)
        print(
            json.dumps(
                {
                    "run": run_dir.name,
                    "targets": int(len(scored)),
                    "category_counts": scored["spectrum_quality_category"].value_counts().sort_index().to_dict(),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    combined = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if args.combined_output is not None:
        args.combined_output.parent.mkdir(parents=True, exist_ok=True)
        if args.combined_output.suffix.lower() == ".csv":
            combined.to_csv(args.combined_output, index=False)
        else:
            combined.to_parquet(args.combined_output, index=False)
    if not combined.empty:
        best = combined.sort_values("spectrum_quality_score", ascending=False).head(args.top)
        cols = [
            "spectrum_quality_score",
            "spectrum_quality_category",
            "n_usable_measurements",
            "flag_fraction",
            "smoothness_score",
            "aperture_psf_corr",
            "median_abs_aperture_snr",
            "phot_g_mean_mag",
            "target_id",
            "run_name",
        ]
        print(best[[col for col in cols if col in best.columns]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
