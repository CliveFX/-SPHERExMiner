from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import warp as wp
except Exception as exc:  # pragma: no cover
    wp = None
    _WARP_IMPORT_ERROR = exc
else:
    _WARP_IMPORT_ERROR = None

RESPONSE_MODEL_VERSION = "gaussian_cwave_cband_v1"
SCIENCE_WAVELENGTH_SOURCE = "spectral_wcs_CWAVE_CBAND"


def _assert_science_wavelengths(df: pd.DataFrame, source_path: Path, *, allow_approx: bool = False) -> None:
    if allow_approx:
        return
    if "wavelength_source" not in df.columns:
        raise ValueError(
            f"{source_path} has no wavelength_source column. Rebuild spectra from measurements generated with "
            f"{SCIENCE_WAVELENGTH_SOURCE}."
        )
    sources = set(df["wavelength_source"].dropna().astype(str).unique())
    if sources != {SCIENCE_WAVELENGTH_SOURCE}:
        raise ValueError(
            f"{source_path} uses wavelength_source={sorted(sources)}. Science/injection paths require "
            f"{SCIENCE_WAVELENGTH_SOURCE}; rerun photometry or pass --allow-approx-wavelengths."
        )
    missing = [
        column
        for column in ("wavelength_calibration_file", "wavelength_calibration_collection", "wavelength_detector")
        if column not in df.columns
    ]
    if missing:
        raise ValueError(f"{source_path} is missing wavelength provenance columns: {', '.join(missing)}")


@wp.kernel(enable_backward=False)  # type: ignore[union-attr]
def _score_both_flux_kinds_kernel(
    offsets: wp.array(dtype=wp.int32),
    lengths: wp.array(dtype=wp.int32),
    wave_um: wp.array(dtype=wp.float32),
    cband_um: wp.array(dtype=wp.float32),
    aperture_flux: wp.array(dtype=wp.float32),
    aperture_unc: wp.array(dtype=wp.float32),
    psf_flux: wp.array(dtype=wp.float32),
    psf_unc: wp.array(dtype=wp.float32),
    fatal_flag: wp.array(dtype=wp.int32),
    line_nm: wp.array(dtype=wp.float32),
    n_lines: int,
    line_width_nm: float,
    local_window_um: float,
    guard_template_above: float,
    min_template_response: float,
    min_support: int,
    ignore_flagged: int,
    out_ap_amp: wp.array(dtype=wp.float32),
    out_ap_unc: wp.array(dtype=wp.float32),
    out_ap_rho: wp.array(dtype=wp.float32),
    out_ap_q: wp.array(dtype=wp.float32),
    out_ap_support: wp.array(dtype=wp.int32),
    out_ap_continuum: wp.array(dtype=wp.float32),
    out_psf_amp: wp.array(dtype=wp.float32),
    out_psf_unc: wp.array(dtype=wp.float32),
    out_psf_rho: wp.array(dtype=wp.float32),
    out_psf_q: wp.array(dtype=wp.float32),
    out_psf_support: wp.array(dtype=wp.int32),
    out_psf_continuum: wp.array(dtype=wp.float32),
    out_flagged_support: wp.array(dtype=wp.int32),
) -> None:
    gid = wp.tid()
    target_idx = gid / n_lines
    line_idx = gid - target_idx * n_lines
    start = offsets[target_idx]
    length = lengths[target_idx]
    line_um = line_nm[line_idx] * float(0.001)
    width_um = line_width_nm * float(0.001)

    ap_sum = float(0.0)
    psf_sum = float(0.0)
    cont_n = int(0)
    i = int(0)
    while i < length:
        idx = start + i
        w = wave_um[idx]
        cb = cband_um[idx]
        ap_f = aperture_flux[idx]
        psf_f = psf_flux[idx]
        flagged = fatal_flag[idx]
        use_row = int(1)
        if ignore_flagged == int(1) and flagged != int(0):
            use_row = int(0)
        if (
            use_row == int(1)
            and wp.isfinite(w)
            and wp.isfinite(cb)
            and wp.isfinite(ap_f)
            and wp.isfinite(psf_f)
            and wp.abs(w - line_um) <= local_window_um
        ):
            fwhm = wp.sqrt(wp.max(cb, float(0.0)) * wp.max(cb, float(0.0)) + width_um * width_um)
            sigma = fwhm / float(2.354820045)
            if sigma > float(0.0):
                template = wp.exp(float(-0.5) * ((w - line_um) / sigma) * ((w - line_um) / sigma))
                if template <= guard_template_above:
                    ap_sum += ap_f
                    psf_sum += psf_f
                    cont_n += 1
        i += 1

    ap_cont = float(0.0)
    psf_cont = float(0.0)
    if cont_n > int(0):
        ap_cont = ap_sum / float(cont_n)
        psf_cont = psf_sum / float(cont_n)

    ap_s = float(0.0)
    ap_n = float(0.0)
    psf_s = float(0.0)
    psf_n = float(0.0)
    ap_support = int(0)
    psf_support = int(0)
    flagged_support = int(0)
    i = int(0)
    while i < length:
        idx = start + i
        w = wave_um[idx]
        cb = cband_um[idx]
        flagged = fatal_flag[idx]
        use_row = int(1)
        if ignore_flagged == int(1) and flagged != int(0):
            use_row = int(0)
        if use_row == int(1) and wp.isfinite(w) and wp.isfinite(cb):
            fwhm = wp.sqrt(wp.max(cb, float(0.0)) * wp.max(cb, float(0.0)) + width_um * width_um)
            sigma = fwhm / float(2.354820045)
            if sigma > float(0.0):
                template = wp.exp(float(-0.5) * ((w - line_um) / sigma) * ((w - line_um) / sigma))
                if template >= min_template_response:
                    if flagged != int(0):
                        flagged_support += 1
                    ap_f = aperture_flux[idx]
                    ap_u = aperture_unc[idx]
                    if wp.isfinite(ap_f) and wp.isfinite(ap_u) and ap_u > float(0.0):
                        wt = float(1.0) / (ap_u * ap_u)
                        ap_s += template * (ap_f - ap_cont) * wt
                        ap_n += template * template * wt
                        ap_support += 1
                    psf_f = psf_flux[idx]
                    psf_u = psf_unc[idx]
                    if wp.isfinite(psf_f) and wp.isfinite(psf_u) and psf_u > float(0.0):
                        wt2 = float(1.0) / (psf_u * psf_u)
                        psf_s += template * (psf_f - psf_cont) * wt2
                        psf_n += template * template * wt2
                        psf_support += 1
        i += 1

    ap_amp = float(-3.402823e38)
    ap_amp_unc = float(3.402823e38)
    ap_rho = float(-3.402823e38)
    ap_q = float(0.0)
    if ap_support >= min_support and ap_n > float(0.0):
        ap_amp = ap_s / ap_n
        ap_amp_unc = float(1.0) / wp.sqrt(ap_n)
        ap_rho = ap_s / wp.sqrt(ap_n)
        if ap_rho > float(0.0):
            ap_q = ap_rho * ap_rho

    psf_amp = float(-3.402823e38)
    psf_amp_unc = float(3.402823e38)
    psf_rho = float(-3.402823e38)
    psf_q = float(0.0)
    if psf_support >= min_support and psf_n > float(0.0):
        psf_amp = psf_s / psf_n
        psf_amp_unc = float(1.0) / wp.sqrt(psf_n)
        psf_rho = psf_s / wp.sqrt(psf_n)
        if psf_rho > float(0.0):
            psf_q = psf_rho * psf_rho

    out_ap_amp[gid] = ap_amp
    out_ap_unc[gid] = ap_amp_unc
    out_ap_rho[gid] = ap_rho
    out_ap_q[gid] = ap_q
    out_ap_support[gid] = ap_support
    out_ap_continuum[gid] = ap_cont
    out_psf_amp[gid] = psf_amp
    out_psf_unc[gid] = psf_amp_unc
    out_psf_rho[gid] = psf_rho
    out_psf_q[gid] = psf_q
    out_psf_support[gid] = psf_support
    out_psf_continuum[gid] = psf_cont
    out_flagged_support[gid] = flagged_support


@wp.kernel(enable_backward=False)  # type: ignore[union-attr]
def _topk_joint_kernel(
    ap_q: wp.array(dtype=wp.float32),
    psf_q: wp.array(dtype=wp.float32),
    ap_rho: wp.array(dtype=wp.float32),
    psf_rho: wp.array(dtype=wp.float32),
    n_lines: int,
    top_k: int,
    min_joint_q: float,
    min_separation_bins: int,
    upcross_level_q: float,
    out_line_idx: wp.array(dtype=wp.int32),
    out_joint_q: wp.array(dtype=wp.float32),
    out_joint_rho: wp.array(dtype=wp.float32),
    out_upcross: wp.array(dtype=wp.int32),
    out_q_max: wp.array(dtype=wp.float32),
) -> None:
    target_idx = wp.tid()
    base = target_idx * n_lines
    out_base = target_idx * top_k

    up = int(0)
    prev = float(0.0)
    qmax = float(0.0)
    li0 = int(0)
    while li0 < n_lines:
        idx0 = base + li0
        q0 = wp.min(ap_q[idx0], psf_q[idx0])
        if q0 > qmax:
            qmax = q0
        if prev < upcross_level_q and q0 >= upcross_level_q:
            up += 1
        prev = q0
        li0 += 1
    out_upcross[target_idx] = up
    out_q_max[target_idx] = qmax

    k = int(0)
    while k < top_k:
        best_line = int(-1)
        best_q = min_joint_q
        li = int(0)
        while li < n_lines:
            idx = base + li
            q = wp.min(ap_q[idx], psf_q[idx])
            if q >= best_q:
                allowed = int(1)
                prev_k = int(0)
                while prev_k < k:
                    chosen = out_line_idx[out_base + prev_k]
                    if chosen >= int(0) and wp.abs(li - chosen) <= min_separation_bins:
                        allowed = int(0)
                    prev_k += 1
                if allowed == int(1):
                    left = float(-1.0)
                    right = float(-1.0)
                    if li > int(0):
                        left = wp.min(ap_q[idx - int(1)], psf_q[idx - int(1)])
                    if li + int(1) < n_lines:
                        right = wp.min(ap_q[idx + int(1)], psf_q[idx + int(1)])
                    if q >= left and q >= right:
                        best_q = q
                        best_line = li
            li += 1

        out_idx = out_base + k
        out_line_idx[out_idx] = best_line
        if best_line >= int(0):
            score_idx = base + best_line
            out_joint_q[out_idx] = best_q
            out_joint_rho[out_idx] = wp.sqrt(best_q)
        else:
            out_joint_q[out_idx] = float(-1.0)
            out_joint_rho[out_idx] = float(-3.402823e38)
        k += 1


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
                "frames_written": int(injection.get("frames_written") or 0),
            }
        )
    return pd.DataFrame(rows)


def _num(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _line_grid(min_nm: float, max_nm: float, step_nm: float) -> np.ndarray:
    if step_nm <= 0.0:
        raise ValueError("--grid-step-nm must be positive")
    return np.arange(min_nm, max_nm + step_nm * 0.5, step_nm, dtype=np.float32)


def _load_spectra(run_dir: Path, spectra_path: Path | None, allow_approx_wavelengths: bool) -> pd.DataFrame:
    path = spectra_path or (run_dir / "spectra" / "target_spectra.parquet")
    if not path.exists():
        raise SystemExit(f"Missing spectra parquet: {path}")
    df = pd.read_parquet(path)
    try:
        _assert_science_wavelengths(df, path, allow_approx=allow_approx_wavelengths)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    required = {
        "target_id",
        "cwave_um",
        "cband_um",
        "aperture_flux_uJy",
        "aperture_flux_unc_uJy",
        "psf_flux_uJy",
        "psf_flux_unc_uJy",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required spectra columns: {', '.join(missing)}")
    keep = [
        "target_id",
        "cwave_um",
        "cband_um",
        "aperture_flux_uJy",
        "aperture_flux_unc_uJy",
        "psf_flux_uJy",
        "psf_flux_unc_uJy",
        "fatal_flag_present",
        "detector",
        "image_id",
        "observation_id",
        "x_pix",
        "y_pix",
        "edge_distance_pix",
        "phot_g_mean_mag",
    ]
    return df[[col for col in keep if col in df.columns]].copy()


def _apply_spectrum_quality_gate(df: pd.DataFrame, run_dir: Path, require_good: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not require_good:
        target_count = int(df["target_id"].dropna().astype(str).nunique())
        return df, {
            "require_good_spectrum": False,
            "quality_path": None,
            "input_target_count": target_count,
            "kept_target_count": target_count,
        }
    quality_path = run_dir / "spectra" / "spectrum_quality.parquet"
    if not quality_path.exists():
        raise SystemExit(
            f"Missing spectrum quality gate file: {quality_path}. Run spectrum quality scoring first, "
            "or pass --allow-non-good-spectra for diagnostic/debug scans."
        )
    quality = pd.read_parquet(quality_path)
    required = {"target_id", "spectrum_quality_category"}
    missing = sorted(required - set(quality.columns))
    if missing:
        raise SystemExit(f"{quality_path} is missing required quality columns: {', '.join(missing)}")
    q = quality.copy()
    if "wavelength_span_nm" not in q and {"wavelength_min_nm", "wavelength_max_nm"} <= set(q.columns):
        q["wavelength_span_nm"] = pd.to_numeric(q["wavelength_max_nm"], errors="coerce") - pd.to_numeric(
            q["wavelength_min_nm"], errors="coerce"
        )
    good_mask = q["spectrum_quality_category"].astype(str).eq("good")
    if "n_usable_measurements" in q:
        good_mask &= pd.to_numeric(q["n_usable_measurements"], errors="coerce").fillna(0).ge(50)
    if "wavelength_span_nm" in q:
        good_mask &= pd.to_numeric(q["wavelength_span_nm"], errors="coerce").fillna(0).ge(4000.0)
    if "aperture_psf_corr" in q:
        corr = pd.to_numeric(q["aperture_psf_corr"], errors="coerce")
        good_mask &= corr.isna() | corr.ge(0.75)
    if "aperture_psf_median_frac_delta" in q:
        frac_delta = pd.to_numeric(q["aperture_psf_median_frac_delta"], errors="coerce")
        good_mask &= frac_delta.isna() | frac_delta.le(1.0)
    input_target_set = set(df["target_id"].dropna().astype(str))
    input_targets = len(input_target_set)
    good_targets = set(q.loc[good_mask, "target_id"].dropna().astype(str))
    kept_targets = good_targets & input_target_set
    gated = df[df["target_id"].astype(str).isin(kept_targets)].copy()
    return gated, {
        "require_good_spectrum": True,
        "quality_path": str(quality_path),
        "input_target_count": input_targets,
        "kept_target_count": int(gated["target_id"].dropna().astype(str).nunique()),
        "rejected_target_count": int(max(0, input_targets - len(kept_targets))),
        "min_usable_points": 50,
        "min_wavelength_span_nm": 4000.0,
        "min_aperture_psf_corr": 0.75,
        "max_aperture_psf_frac_delta": 1.0,
        "quality_category_counts": quality["spectrum_quality_category"].astype(str).value_counts().sort_index().to_dict(),
    }


def _target_ids(df: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    target_ids = list(df["target_id"].dropna().astype(str).drop_duplicates())
    requested: set[str] = set(args.target_id or [])
    if args.target_ids_file:
        requested.update(
            line.strip()
            for line in args.target_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if args.manifest:
        truth = _read_truth(args.manifest)
        requested.update(truth["target_id"].dropna().astype(str))
    if requested:
        target_ids = [target_id for target_id in target_ids if target_id in requested]
    if args.max_targets:
        target_ids = target_ids[: int(args.max_targets)]
    return target_ids


def _write_empty_outputs(args: argparse.Namespace, run_dir: Path, quality_gate: dict[str, Any], started: float) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = args.output_dir / "narrowband_candidates.parquet"
    pd.DataFrame().to_parquet(candidates_path, index=False)
    summary = {
        "detector": "warp_narrowband_detector",
        "spec": "docs/gpu_narrowband_detector_spec.md",
        "response_model_version_hash": RESPONSE_MODEL_VERSION,
        "run_dir": str(run_dir),
        "manifest": str(args.manifest) if args.manifest else None,
        "output_dir": str(args.output_dir),
        "device": args.device,
        "target_count": 0,
        "measurement_rows": 0,
        "line_count": 0,
        "score_rows": 0,
        "candidate_count": 0,
        "science_pass_count": 0,
        "quality_pass_count": 0,
        "spectrum_quality_gate": quality_gate,
        "total_sec": time.perf_counter() - started,
        "candidates_path": str(candidates_path),
        "diagnostic_line_scores_path": None,
    }
    (args.output_dir / "narrowband_detector_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _pack(df: pd.DataFrame, target_ids: list[str]) -> tuple[np.ndarray, ...]:
    grouped = {str(t): rows.sort_values("cwave_um") for t, rows in df.groupby("target_id", sort=False)}
    offsets: list[int] = []
    lengths: list[int] = []
    waves: list[np.ndarray] = []
    cbands: list[np.ndarray] = []
    ap_flux: list[np.ndarray] = []
    ap_unc: list[np.ndarray] = []
    psf_flux: list[np.ndarray] = []
    psf_unc: list[np.ndarray] = []
    fatal: list[np.ndarray] = []
    cursor = 0
    for target_id in target_ids:
        rows = grouped.get(str(target_id))
        if rows is None:
            continue
        offsets.append(cursor)
        lengths.append(int(len(rows)))
        cursor += int(len(rows))
        waves.append(pd.to_numeric(rows["cwave_um"], errors="coerce").to_numpy(dtype=np.float32))
        cbands.append(pd.to_numeric(rows["cband_um"], errors="coerce").to_numpy(dtype=np.float32))
        ap_flux.append(pd.to_numeric(rows["aperture_flux_uJy"], errors="coerce").to_numpy(dtype=np.float32))
        ap_unc.append(pd.to_numeric(rows["aperture_flux_unc_uJy"], errors="coerce").to_numpy(dtype=np.float32))
        psf_flux.append(pd.to_numeric(rows["psf_flux_uJy"], errors="coerce").to_numpy(dtype=np.float32))
        psf_unc.append(pd.to_numeric(rows["psf_flux_unc_uJy"], errors="coerce").to_numpy(dtype=np.float32))
        if "fatal_flag_present" in rows:
            fatal.append(rows["fatal_flag_present"].fillna(False).astype(bool).to_numpy(dtype=np.int32))
        else:
            fatal.append(np.zeros(len(rows), dtype=np.int32))
    if not waves:
        raise SystemExit("No target rows selected")
    return (
        np.asarray(offsets, dtype=np.int32),
        np.asarray(lengths, dtype=np.int32),
        np.concatenate(waves).astype(np.float32),
        np.concatenate(cbands).astype(np.float32),
        np.concatenate(ap_flux).astype(np.float32),
        np.concatenate(ap_unc).astype(np.float32),
        np.concatenate(psf_flux).astype(np.float32),
        np.concatenate(psf_unc).astype(np.float32),
        np.concatenate(fatal).astype(np.int32),
    )


def _p_local(q: float) -> float:
    if not math.isfinite(q) or q <= 0.0:
        return 0.5
    return 0.5 * math.erfc(math.sqrt(q / 2.0))


def _p_global(q: float, n0: float, u0: float) -> float:
    if not math.isfinite(q) or q <= 0.0:
        return 1.0
    return min(1.0, _p_local(q) + max(n0, 0.0) * math.exp(-0.5 * (q - u0)))


def _enrich_candidate_rows(candidates: pd.DataFrame, measurements: pd.DataFrame, *, local_window_um: float) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    by_target = {str(t): rows.sort_values("cwave_um") for t, rows in measurements.groupby("target_id", sort=False)}
    out_rows: list[dict[str, Any]] = []
    for row in candidates.to_dict(orient="records"):
        target_id = str(row["target_id"])
        rows = by_target.get(target_id)
        out = dict(row)
        if rows is not None:
            line_um = float(row["line_nm_source_frame"]) / 1000.0
            wave = pd.to_numeric(rows["cwave_um"], errors="coerce").to_numpy(dtype=float)
            near = np.isfinite(wave) & (np.abs(wave - line_um) <= local_window_um)
            if "detector" in rows:
                detectors = pd.to_numeric(rows["detector"], errors="coerce").to_numpy(dtype=float)
                out["detectors"] = ",".join(sorted({str(int(d)) for d in detectors[near] if np.isfinite(d)}))
            if "image_id" in rows:
                frames = rows.loc[near, "image_id"].dropna().astype(str).drop_duplicates().head(12).tolist()
                out["frame_ids"] = ",".join(frames)
            if "observation_id" in rows:
                out["n_visits"] = int(rows.loc[near, "observation_id"].dropna().astype(str).nunique())
            if "fatal_flag_present" in rows:
                out["flagged_points_sum"] = int(rows.loc[near, "fatal_flag_present"].fillna(False).astype(bool).sum())
        out_rows.append(out)
    return pd.DataFrame(out_rows)


def _apply_quality_filter(candidates: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    out = candidates.copy()
    out["target_candidate_count"] = out.groupby("target_id")["target_id"].transform("size")
    ap_amp = pd.to_numeric(out["aperture_amp_uJy"], errors="coerce").abs()
    psf_amp = pd.to_numeric(out["psf_amp_uJy"], errors="coerce").abs()
    out["aperture_psf_amp_ratio"] = np.maximum(ap_amp, psf_amp) / np.maximum(np.minimum(ap_amp, psf_amp), 1e-9)

    reasons: list[str] = []
    quality_pass = pd.Series(True, index=out.index)

    base_science = out["tier"].astype(str).eq("science_pass")
    quality_pass &= base_science
    reasons.append("p_global_above_alpha")

    support_ok = pd.to_numeric(out["support_count"], errors="coerce").fillna(0).ge(int(args.quality_min_support))
    quality_pass &= support_ok
    reasons.append("support_below_min")

    flagged_ok = pd.to_numeric(out["flagged_points_sum"], errors="coerce").fillna(0).le(int(args.quality_max_flagged_points))
    quality_pass &= flagged_ok
    reasons.append("too_many_flagged_points")

    ambiguity_ok = pd.to_numeric(out["target_candidate_count"], errors="coerce").fillna(999999).le(
        int(args.quality_max_candidates_per_target)
    )
    quality_pass &= ambiguity_ok
    reasons.append("too_many_competing_peaks")

    ratio_ok = pd.to_numeric(out["aperture_psf_amp_ratio"], errors="coerce").fillna(np.inf).le(
        float(args.quality_max_aperture_psf_ratio)
    )
    quality_pass &= ratio_ok
    reasons.append("aperture_psf_ratio_out_of_range")

    reject_reasons: list[str] = []
    for idx in out.index:
        row_reasons: list[str] = []
        if not bool(base_science.loc[idx]):
            row_reasons.append("p_global_above_alpha")
        if not bool(support_ok.loc[idx]):
            row_reasons.append("support_below_min")
        if not bool(flagged_ok.loc[idx]):
            row_reasons.append("too_many_flagged_points")
        if not bool(ambiguity_ok.loc[idx]):
            row_reasons.append("too_many_competing_peaks")
        if not bool(ratio_ok.loc[idx]):
            row_reasons.append("aperture_psf_ratio_out_of_range")
        reject_reasons.append(",".join(row_reasons))
    out["quality_pass"] = quality_pass.astype(bool)
    joint_rho = pd.to_numeric(out["joint_rho"], errors="coerce").fillna(0.0)
    p_global = pd.to_numeric(out["p_global"], errors="coerce").fillna(1.0)
    support = pd.to_numeric(out["support_count"], errors="coerce").fillna(0)
    flagged = pd.to_numeric(out["flagged_points_sum"], errors="coerce").fillna(999999)
    competing = pd.to_numeric(out["target_candidate_count"], errors="coerce").fillna(999999)
    ratio = pd.to_numeric(out["aperture_psf_amp_ratio"], errors="coerce").fillna(np.inf)

    s_tier = (
        out["quality_pass"]
        & p_global.le(float(args.s_tier_max_p_global))
        & joint_rho.ge(float(args.s_tier_min_joint_rho))
        & support.ge(int(args.s_tier_min_support))
        & flagged.le(int(args.s_tier_max_flagged_points))
        & competing.le(int(args.s_tier_max_candidates_per_target))
        & ratio.le(float(args.s_tier_max_aperture_psf_ratio))
    )
    a_tier = (
        out["quality_pass"]
        & ~s_tier
        & p_global.le(float(args.a_tier_max_p_global))
        & support.ge(int(args.a_tier_min_support))
        & competing.le(int(args.a_tier_max_candidates_per_target))
    )
    b_tier = out["quality_pass"] & ~s_tier & ~a_tier
    c_tier = ~out["quality_pass"] & base_science

    out["review_grade"] = np.select(
        [s_tier, a_tier, b_tier, c_tier],
        ["S", "A", "B", "C"],
        default="D",
    )
    out["quality_tier"] = np.select(
        [s_tier, a_tier, b_tier, c_tier],
        ["S_go_look", "A_priority", "B_review", "C_suspicious_science",],
        default="D_raw_debug",
    )
    out["quality_reject_reasons"] = reject_reasons
    out["reject_reasons"] = out["quality_reject_reasons"]
    return out


def _score_recovery(
    candidates: pd.DataFrame,
    manifest: Path,
    tolerance_nm: float,
    scanned_target_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    truth = _read_truth(manifest)
    rows: list[dict[str, Any]] = []
    for truth_row in truth.to_dict(orient="records"):
        target_id = str(truth_row["target_id"])
        line_nm = _num(truth_row.get("injected_line_nm"))
        matched = pd.DataFrame()
        matched_quality = pd.DataFrame()
        if line_nm is not None and not candidates.empty:
            same = candidates[candidates["target_id"].astype(str).eq(target_id)].copy()
            if not same.empty:
                delta = (pd.to_numeric(same["line_nm_source_frame"], errors="coerce") - line_nm).abs()
                matched = same[delta <= tolerance_nm].copy()
                if not matched.empty:
                    matched = matched.sort_values("joint_q", ascending=False, na_position="last")
                    if "quality_pass" in matched.columns:
                        matched_quality = matched[matched["quality_pass"].fillna(False).astype(bool)].copy()
        out = dict(truth_row)
        out["target_scanned"] = target_id in scanned_target_ids
        if matched.empty:
            out.update(
                {
                    "gpu_narrowband_match": False,
                    "gpu_narrowband_line_nm": None,
                    "gpu_narrowband_joint_q": None,
                    "gpu_narrowband_joint_rho": None,
                    "gpu_narrowband_p_global": None,
                    "gpu_narrowband_tier": "miss",
                    "gpu_narrowband_quality_match": False,
                    "gpu_narrowband_quality_tier": "miss",
                    "gpu_narrowband_quality_reject_reasons": "no_match",
                }
            )
        else:
            best = matched.iloc[0].to_dict()
            out.update(
                {
                    "gpu_narrowband_match": True,
                    "gpu_narrowband_line_nm": _num(best.get("line_nm_source_frame")),
                    "gpu_narrowband_joint_q": _num(best.get("joint_q")),
                    "gpu_narrowband_joint_rho": _num(best.get("joint_rho")),
                    "gpu_narrowband_p_global": _num(best.get("p_global")),
                    "gpu_narrowband_tier": best.get("tier"),
                    "gpu_narrowband_quality_match": not matched_quality.empty,
                    "gpu_narrowband_quality_tier": best.get("quality_tier"),
                    "gpu_narrowband_quality_reject_reasons": best.get("quality_reject_reasons"),
                }
            )
        rows.append(out)
    scored = pd.DataFrame(rows)
    scanned = scored[scored["target_scanned"].fillna(False).astype(bool)] if not scored.empty else scored
    summary = {
        "truth_count": int(len(scored)),
        "scanned_truth_count": int(len(scanned)),
        "match_count": int(scored.get("gpu_narrowband_match", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "match_fraction": float(scored.get("gpu_narrowband_match", pd.Series(dtype=bool)).fillna(False).astype(bool).mean())
        if len(scored)
        else 0.0,
        "scanned_match_count": int(
            scanned.get("gpu_narrowband_match", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
        ),
        "scanned_match_fraction": float(
            scanned.get("gpu_narrowband_match", pd.Series(dtype=bool)).fillna(False).astype(bool).mean()
        )
        if len(scanned)
        else 0.0,
        "quality_match_count": int(
            scored.get("gpu_narrowband_quality_match", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
        ),
        "quality_match_fraction": float(
            scored.get("gpu_narrowband_quality_match", pd.Series(dtype=bool)).fillna(False).astype(bool).mean()
        )
        if len(scored)
        else 0.0,
        "scanned_quality_match_count": int(
            scanned.get("gpu_narrowband_quality_match", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
        ),
        "scanned_quality_match_fraction": float(
            scanned.get("gpu_narrowband_quality_match", pd.Series(dtype=bool)).fillna(False).astype(bool).mean()
        )
        if len(scanned)
        else 0.0,
        "wavelength_tolerance_nm": float(tolerance_nm),
    }
    return scored, summary


def _diagnostic_line_scores(
    candidates: pd.DataFrame,
    target_ids: list[str],
    lines: np.ndarray,
    ap_rho: np.ndarray,
    psf_rho: np.ndarray,
    ap_amp: np.ndarray,
    ap_unc: np.ndarray,
    psf_amp: np.ndarray,
    psf_unc: np.ndarray,
    ap_support: np.ndarray,
    psf_support: np.ndarray,
    flagged_support: np.ndarray,
    *,
    half_window_nm: float,
    max_rows_per_candidate: int,
) -> pd.DataFrame:
    if candidates.empty or half_window_nm <= 0.0 or max_rows_per_candidate <= 0:
        return pd.DataFrame()
    target_index = {str(target_id): idx for idx, target_id in enumerate(target_ids)}
    out_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for candidate in candidates.to_dict(orient="records"):
        target_id = str(candidate.get("target_id") or "")
        line_nm = _num(candidate.get("line_nm_source_frame"))
        ti = target_index.get(target_id)
        if ti is None or line_nm is None:
            continue
        key = (target_id, round(line_nm, 6))
        if key in seen:
            continue
        seen.add(key)
        near = np.flatnonzero(np.abs(lines.astype(float) - float(line_nm)) <= float(half_window_nm))
        if near.size == 0:
            continue
        if near.size > max_rows_per_candidate:
            order = np.argsort(np.abs(lines[near].astype(float) - float(line_nm)))
            near = np.sort(near[order[:max_rows_per_candidate]])
        for li in near.tolist():
            ap_q = float(ap_rho[ti, li]) * float(ap_rho[ti, li]) if math.isfinite(float(ap_rho[ti, li])) else 0.0
            psf_q = float(psf_rho[ti, li]) * float(psf_rho[ti, li]) if math.isfinite(float(psf_rho[ti, li])) else 0.0
            joint_q = min(ap_q, psf_q)
            out_rows.append(
                {
                    "joint_candidate_id": candidate.get("joint_candidate_id"),
                    "target_id": target_id,
                    "candidate_line_nm": float(line_nm),
                    "line_nm": float(lines[li]),
                    "delta_nm": float(lines[li]) - float(line_nm),
                    "aperture_snr": float(ap_rho[ti, li]),
                    "psf_snr": float(psf_rho[ti, li]),
                    "joint_rho": math.sqrt(joint_q) if joint_q > 0.0 else 0.0,
                    "aperture_q": ap_q,
                    "psf_q": psf_q,
                    "joint_q": joint_q,
                    "aperture_amp_uJy": float(ap_amp[ti, li]),
                    "aperture_amp_unc_uJy": float(ap_unc[ti, li]),
                    "psf_amp_uJy": float(psf_amp[ti, li]),
                    "psf_amp_unc_uJy": float(psf_unc[ti, li]),
                    "aperture_support_count": int(ap_support[ti, li]),
                    "psf_support_count": int(psf_support[ti, li]),
                    "support_count": int(min(ap_support[ti, li], psf_support[ti, li])),
                    "flagged_points_sum": int(flagged_support[ti, li]),
                    "review_grade": candidate.get("review_grade"),
                    "quality_tier": candidate.get("quality_tier"),
                    "quality_pass": candidate.get("quality_pass"),
                }
            )
    return pd.DataFrame(out_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone GPU response-template narrowband detector.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--spectra-path", type=Path)
    parser.add_argument("--manifest", type=Path, help="Optional injection manifest; also restricts scan to injected targets.")
    parser.add_argument("--target-id", action="append")
    parser.add_argument("--target-ids-file", type=Path)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-line-nm", type=float, default=750.0)
    parser.add_argument("--max-line-nm", type=float, default=5000.0)
    parser.add_argument("--grid-step-nm", type=float, default=1.0)
    parser.add_argument("--line-width-nm", type=float, default=1.0)
    parser.add_argument("--local-window-um", type=float, default=0.18)
    parser.add_argument("--guard-template-above", type=float, default=0.25)
    parser.add_argument("--min-template-response", type=float, default=1e-3)
    parser.add_argument("--min-supporting-points", type=int, default=2)
    parser.add_argument("--top-k-per-target", type=int, default=10)
    parser.add_argument("--top-k-min-separation-nm", type=float, default=10.0)
    parser.add_argument("--min-joint-rho", type=float, default=1.5)
    parser.add_argument("--upcross-level-q", type=float, default=4.0)
    parser.add_argument("--n0", type=float, help="Override Gross-Vitells up-crossing factor.")
    parser.add_argument("--alpha-global", type=float, default=1e-3)
    parser.add_argument("--quality-min-support", type=int, default=3)
    parser.add_argument("--quality-max-flagged-points", type=int, default=3)
    parser.add_argument("--quality-max-candidates-per-target", type=int, default=5)
    parser.add_argument("--quality-max-aperture-psf-ratio", type=float, default=3.0)
    parser.add_argument("--s-tier-max-p-global", type=float, default=1e-12)
    parser.add_argument("--s-tier-min-joint-rho", type=float, default=12.0)
    parser.add_argument("--s-tier-min-support", type=int, default=4)
    parser.add_argument("--s-tier-max-flagged-points", type=int, default=0)
    parser.add_argument("--s-tier-max-candidates-per-target", type=int, default=1)
    parser.add_argument("--s-tier-max-aperture-psf-ratio", type=float, default=1.5)
    parser.add_argument("--a-tier-max-p-global", type=float, default=1e-6)
    parser.add_argument("--a-tier-min-support", type=int, default=3)
    parser.add_argument("--a-tier-max-candidates-per-target", type=int, default=2)
    parser.add_argument("--recovery-tolerance-nm", type=float, default=10.0)
    parser.add_argument("--diagnostic-line-half-window-nm", type=float, default=80.0)
    parser.add_argument("--diagnostic-line-max-rows-per-candidate", type=int, default=201)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ignore-flagged", action="store_true", default=True)
    parser.add_argument("--include-flagged", dest="ignore_flagged", action="store_false")
    parser.add_argument("--require-good-spectrum", action="store_true", default=True)
    parser.add_argument("--allow-non-good-spectra", dest="require_good_spectrum", action="store_false")
    parser.add_argument("--allow-approx-wavelengths", action="store_true")
    args = parser.parse_args()

    if wp is None:
        raise SystemExit(f"Warp is not available: {_WARP_IMPORT_ERROR}")
    wp.init()

    t0 = time.perf_counter()
    measurements = _load_spectra(args.run_dir, args.spectra_path, args.allow_approx_wavelengths)
    measurements, spectrum_quality_gate = _apply_spectrum_quality_gate(measurements, args.run_dir, bool(args.require_good_spectrum))
    target_ids = _target_ids(measurements, args)
    if not target_ids:
        _write_empty_outputs(args, args.run_dir, spectrum_quality_gate, t0)
        print(json.dumps({"status": "no_targets_after_spectrum_quality_gate", "output_dir": str(args.output_dir)}, sort_keys=True), flush=True)
        return
    selected = measurements[measurements["target_id"].astype(str).isin(target_ids)].copy()
    lines = _line_grid(args.min_line_nm, args.max_line_nm, args.grid_step_nm)
    offsets, lengths, wave, cband, ap_f, ap_u, psf_f, psf_u, fatal = _pack(selected, target_ids)
    pack_sec = time.perf_counter() - t0

    n_targets = len(target_ids)
    n_lines = len(lines)
    n_scores = int(n_targets * n_lines)
    min_joint_q = float(args.min_joint_rho) * float(args.min_joint_rho)

    t1 = time.perf_counter()
    offsets_d = wp.array(offsets, dtype=wp.int32, device=args.device)
    lengths_d = wp.array(lengths, dtype=wp.int32, device=args.device)
    wave_d = wp.array(wave, dtype=wp.float32, device=args.device)
    cband_d = wp.array(cband, dtype=wp.float32, device=args.device)
    ap_f_d = wp.array(ap_f, dtype=wp.float32, device=args.device)
    ap_u_d = wp.array(ap_u, dtype=wp.float32, device=args.device)
    psf_f_d = wp.array(psf_f, dtype=wp.float32, device=args.device)
    psf_u_d = wp.array(psf_u, dtype=wp.float32, device=args.device)
    fatal_d = wp.array(fatal, dtype=wp.int32, device=args.device)
    line_d = wp.array(lines, dtype=wp.float32, device=args.device)

    ap_amp_d = wp.empty(n_scores, dtype=wp.float32, device=args.device)
    ap_unc_d = wp.empty(n_scores, dtype=wp.float32, device=args.device)
    ap_rho_d = wp.empty(n_scores, dtype=wp.float32, device=args.device)
    ap_q_d = wp.empty(n_scores, dtype=wp.float32, device=args.device)
    ap_support_d = wp.empty(n_scores, dtype=wp.int32, device=args.device)
    ap_cont_d = wp.empty(n_scores, dtype=wp.float32, device=args.device)
    psf_amp_d = wp.empty(n_scores, dtype=wp.float32, device=args.device)
    psf_unc_d = wp.empty(n_scores, dtype=wp.float32, device=args.device)
    psf_rho_d = wp.empty(n_scores, dtype=wp.float32, device=args.device)
    psf_q_d = wp.empty(n_scores, dtype=wp.float32, device=args.device)
    psf_support_d = wp.empty(n_scores, dtype=wp.int32, device=args.device)
    psf_cont_d = wp.empty(n_scores, dtype=wp.float32, device=args.device)
    flagged_support_d = wp.empty(n_scores, dtype=wp.int32, device=args.device)

    wp.launch(
        _score_both_flux_kinds_kernel,
        dim=n_scores,
        inputs=[
            offsets_d,
            lengths_d,
            wave_d,
            cband_d,
            ap_f_d,
            ap_u_d,
            psf_f_d,
            psf_u_d,
            fatal_d,
            line_d,
            int(n_lines),
            float(args.line_width_nm),
            float(args.local_window_um),
            float(args.guard_template_above),
            float(args.min_template_response),
            int(args.min_supporting_points),
            int(1 if args.ignore_flagged else 0),
            ap_amp_d,
            ap_unc_d,
            ap_rho_d,
            ap_q_d,
            ap_support_d,
            ap_cont_d,
            psf_amp_d,
            psf_unc_d,
            psf_rho_d,
            psf_q_d,
            psf_support_d,
            psf_cont_d,
            flagged_support_d,
        ],
        device=args.device,
    )
    wp.synchronize_device(args.device)
    score_kernel_sec = time.perf_counter() - t1

    top_k = max(1, int(args.top_k_per_target))
    min_sep_bins = max(1, int(math.ceil(float(args.top_k_min_separation_nm) / max(float(args.grid_step_nm), 1e-9))))
    compact = int(n_targets * top_k)
    top_line_d = wp.empty(compact, dtype=wp.int32, device=args.device)
    top_q_d = wp.empty(compact, dtype=wp.float32, device=args.device)
    top_rho_d = wp.empty(compact, dtype=wp.float32, device=args.device)
    upcross_d = wp.empty(n_targets, dtype=wp.int32, device=args.device)
    qmax_d = wp.empty(n_targets, dtype=wp.float32, device=args.device)

    t2 = time.perf_counter()
    wp.launch(
        _topk_joint_kernel,
        dim=n_targets,
        inputs=[
            ap_q_d,
            psf_q_d,
            ap_rho_d,
            psf_rho_d,
            int(n_lines),
            int(top_k),
            float(min_joint_q),
            int(min_sep_bins),
            float(args.upcross_level_q),
            top_line_d,
            top_q_d,
            top_rho_d,
            upcross_d,
            qmax_d,
        ],
        device=args.device,
    )
    wp.synchronize_device(args.device)
    topk_kernel_sec = time.perf_counter() - t2

    t3 = time.perf_counter()
    top_line = top_line_d.numpy().reshape((n_targets, top_k))
    top_q = top_q_d.numpy().reshape((n_targets, top_k))
    top_rho = top_rho_d.numpy().reshape((n_targets, top_k))
    upcross = upcross_d.numpy()
    qmax = qmax_d.numpy()
    ap_amp = ap_amp_d.numpy().reshape((n_targets, n_lines))
    ap_unc = ap_unc_d.numpy().reshape((n_targets, n_lines))
    ap_rho = ap_rho_d.numpy().reshape((n_targets, n_lines))
    ap_support = ap_support_d.numpy().reshape((n_targets, n_lines))
    psf_amp = psf_amp_d.numpy().reshape((n_targets, n_lines))
    psf_unc = psf_unc_d.numpy().reshape((n_targets, n_lines))
    psf_rho = psf_rho_d.numpy().reshape((n_targets, n_lines))
    psf_support = psf_support_d.numpy().reshape((n_targets, n_lines))
    flagged_support = flagged_support_d.numpy().reshape((n_targets, n_lines))
    copyback_sec = time.perf_counter() - t3

    n0 = float(args.n0) if args.n0 is not None else float(np.mean(upcross)) if len(upcross) else 0.0
    rows: list[dict[str, Any]] = []
    for ti, target_id in enumerate(target_ids):
        for ki in range(top_k):
            li = int(top_line[ti, ki])
            if li < 0:
                continue
            q = float(top_q[ti, ki])
            p_local = _p_local(q)
            p_global = _p_global(q, n0, float(args.upcross_level_q))
            tier = "science_pass" if p_global < float(args.alpha_global) else "review"
            rows.append(
                {
                    "joint_candidate_id": f"{target_id}::narrowband::{float(lines[li]):g}",
                    "target_id": target_id,
                    "line_nm_source_frame": float(lines[li]),
                    "peak_line_nm": float(lines[li]),
                    "candidate_line_nm": float(lines[li]),
                    "template_id": RESPONSE_MODEL_VERSION,
                    "tier": tier,
                    "q_max": float(qmax[ti]),
                    "joint_q": q,
                    "joint_rho": float(top_rho[ti, ki]),
                    "rank_score": float(top_rho[ti, ki]),
                    "p_local": p_local,
                    "p_global": p_global,
                    "N0_used": n0,
                    "upcross_count": int(upcross[ti]),
                    "aperture_snr": float(ap_rho[ti, li]),
                    "psf_snr": float(psf_rho[ti, li]),
                    "aperture_peak_snr": float(ap_rho[ti, li]),
                    "psf_peak_snr": float(psf_rho[ti, li]),
                    "aperture_amp_uJy": float(ap_amp[ti, li]),
                    "aperture_amp_unc_uJy": float(ap_unc[ti, li]),
                    "psf_amp_uJy": float(psf_amp[ti, li]),
                    "psf_amp_unc_uJy": float(psf_unc[ti, li]),
                    "response_bin_chi2": np.nan,
                    "source_frame_recurrence_pass": False,
                    "support_count": int(min(ap_support[ti, li], psf_support[ti, li])),
                    "aperture_support_count": int(ap_support[ti, li]),
                    "psf_support_count": int(psf_support[ti, li]),
                    "aperture_support": int(ap_support[ti, li]),
                    "psf_support": int(psf_support[ti, li]),
                    "flagged_points_sum": int(flagged_support[ti, li]),
                    "sigma_ratio_flag": False,
                    "response_model_version_hash": RESPONSE_MODEL_VERSION,
                    "reject_reasons": "" if tier == "science_pass" else "p_global_above_alpha_or_no_recurrence",
                }
            )
    candidates = pd.DataFrame(rows).sort_values("joint_q", ascending=False, na_position="last") if rows else pd.DataFrame()
    candidates = _enrich_candidate_rows(candidates, selected, local_window_um=float(args.local_window_um))
    candidates = _apply_quality_filter(candidates, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = args.output_dir / "narrowband_candidates.parquet"
    candidates.to_parquet(candidates_path, index=False)
    diagnostic = _diagnostic_line_scores(
        candidates,
        target_ids,
        lines,
        ap_rho,
        psf_rho,
        ap_amp,
        ap_unc,
        psf_amp,
        psf_unc,
        ap_support,
        psf_support,
        flagged_support,
        half_window_nm=float(args.diagnostic_line_half_window_nm),
        max_rows_per_candidate=int(args.diagnostic_line_max_rows_per_candidate),
    )
    diagnostic_path = args.output_dir / "narrowband_line_scores.parquet"
    if not diagnostic.empty:
        diagnostic.to_parquet(diagnostic_path, index=False)

    recovery_summary: dict[str, Any] | None = None
    if args.manifest:
        recovery, recovery_summary = _score_recovery(
            candidates,
            args.manifest,
            float(args.recovery_tolerance_nm),
            set(str(target_id) for target_id in target_ids),
        )
        recovery.to_parquet(args.output_dir / "narrowband_recovery.parquet", index=False)

    summary = {
        "detector": "warp_narrowband_detector",
        "spec": "docs/gpu_narrowband_detector_spec.md",
        "response_model_version_hash": RESPONSE_MODEL_VERSION,
        "run_dir": str(args.run_dir),
        "manifest": str(args.manifest) if args.manifest else None,
        "output_dir": str(args.output_dir),
        "device": args.device,
        "target_count": int(n_targets),
        "measurement_rows": int(len(wave)),
        "line_count": int(n_lines),
        "score_rows": int(n_scores),
        "candidate_count": int(len(candidates)),
        "science_pass_count": int(candidates["tier"].eq("science_pass").sum()) if not candidates.empty else 0,
        "quality_pass_count": int(candidates["quality_pass"].fillna(False).astype(bool).sum()) if not candidates.empty else 0,
        "quality_config": {
            "require_good_spectrum": bool(args.require_good_spectrum),
            "quality_min_support": int(args.quality_min_support),
            "quality_max_flagged_points": int(args.quality_max_flagged_points),
            "quality_max_candidates_per_target": int(args.quality_max_candidates_per_target),
            "quality_max_aperture_psf_ratio": float(args.quality_max_aperture_psf_ratio),
            "s_tier_max_p_global": float(args.s_tier_max_p_global),
            "s_tier_min_joint_rho": float(args.s_tier_min_joint_rho),
            "s_tier_min_support": int(args.s_tier_min_support),
            "s_tier_max_flagged_points": int(args.s_tier_max_flagged_points),
            "s_tier_max_candidates_per_target": int(args.s_tier_max_candidates_per_target),
            "s_tier_max_aperture_psf_ratio": float(args.s_tier_max_aperture_psf_ratio),
            "a_tier_max_p_global": float(args.a_tier_max_p_global),
            "a_tier_min_support": int(args.a_tier_min_support),
            "a_tier_max_candidates_per_target": int(args.a_tier_max_candidates_per_target),
        },
        "spectrum_quality_gate": spectrum_quality_gate,
        "n0_used": n0,
        "upcross_level_q": float(args.upcross_level_q),
        "alpha_global": float(args.alpha_global),
        "min_joint_rho": float(args.min_joint_rho),
        "grid_step_nm": float(args.grid_step_nm),
        "pack_sec": pack_sec,
        "score_kernel_sec": score_kernel_sec,
        "topk_kernel_sec": topk_kernel_sec,
        "copyback_sec": copyback_sec,
        "total_sec": time.perf_counter() - t0,
        "scores_per_sec_kernel": float(n_scores / score_kernel_sec) if score_kernel_sec > 0 else None,
        "candidates_path": str(candidates_path),
        "diagnostic_line_scores_path": str(diagnostic_path) if not diagnostic.empty else None,
        "diagnostic_line_score_rows": int(len(diagnostic)),
        "recovery": recovery_summary,
    }
    (args.output_dir / "narrowband_detector_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
