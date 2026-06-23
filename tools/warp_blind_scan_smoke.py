from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.blind_scan_matched_filter import _cluster_candidates, _line_grid, _load_raw_spectra  # noqa: E402
from tools.classify_paired_delta_matched_filter import _load_pair  # noqa: E402

try:
    import warp as wp
except Exception as exc:  # pragma: no cover
    wp = None
    _WARP_IMPORT_ERROR = exc
else:
    _WARP_IMPORT_ERROR = None


@wp.kernel(enable_backward=False)  # type: ignore[union-attr]
def _blind_score_kernel(
    offsets: wp.array(dtype=wp.int32),
    lengths: wp.array(dtype=wp.int32),
    wave_um: wp.array(dtype=wp.float32),
    cband_um: wp.array(dtype=wp.float32),
    flux_uJy: wp.array(dtype=wp.float32),
    unc_uJy: wp.array(dtype=wp.float32),
    line_nm: wp.array(dtype=wp.float32),
    n_lines: int,
    line_width_nm: float,
    min_template_response: float,
    local_window_um: float,
    continuum_exclude_template_above: float,
    min_supporting_points: int,
    out_amp: wp.array(dtype=wp.float32),
    out_unc: wp.array(dtype=wp.float32),
    out_snr: wp.array(dtype=wp.float32),
    out_support: wp.array(dtype=wp.int32),
    out_continuum: wp.array(dtype=wp.float32),
) -> None:
    gid = wp.tid()
    target_idx = gid / n_lines
    line_idx = gid - target_idx * n_lines
    start = offsets[target_idx]
    length = lengths[target_idx]
    line_um = line_nm[line_idx] * 0.001
    width_um = line_width_nm * 0.001

    cont_sum = float(0.0)
    cont_n = int(0)
    i = int(0)
    while i < length:
        idx = start + i
        w = wave_um[idx]
        f = flux_uJy[idx]
        cb = cband_um[idx]
        if wp.isfinite(w) and wp.isfinite(f) and wp.isfinite(cb):
            dw = wp.abs(w - line_um)
            if dw <= local_window_um:
                fwhm = wp.sqrt(wp.max(cb, float(0.0)) * wp.max(cb, float(0.0)) + width_um * width_um)
                sigma = fwhm / float(2.354820045)
                if sigma > float(0.0):
                    t = wp.exp(float(-0.5) * ((w - line_um) / sigma) * ((w - line_um) / sigma))
                    if t <= continuum_exclude_template_above:
                        cont_sum += f
                        cont_n += 1
        i += 1
    continuum = float(0.0)
    if cont_n > 0:
        continuum = cont_sum / float(cont_n)

    numer = float(0.0)
    denom = float(0.0)
    support = int(0)
    i = int(0)
    while i < length:
        idx = start + i
        w = wave_um[idx]
        cb = cband_um[idx]
        f = flux_uJy[idx]
        u = unc_uJy[idx]
        if wp.isfinite(w) and wp.isfinite(cb) and wp.isfinite(f) and wp.isfinite(u) and u > float(0.0):
            fwhm = wp.sqrt(wp.max(cb, float(0.0)) * wp.max(cb, float(0.0)) + width_um * width_um)
            sigma = fwhm / float(2.354820045)
            if sigma > float(0.0):
                t = wp.exp(float(-0.5) * ((w - line_um) / sigma) * ((w - line_um) / sigma))
                if t >= min_template_response:
                    wt = float(1.0) / (u * u)
                    numer += t * (f - continuum) * wt
                    denom += t * t * wt
                    support += 1
        i += 1
    amp = float(-3.402823e38)
    amp_unc = float(3.402823e38)
    snr = float(-3.402823e38)
    if support >= min_supporting_points and denom > float(0.0):
        amp = numer / denom
        amp_unc = float(1.0) / wp.sqrt(denom)
        snr = amp / amp_unc
    out_amp[gid] = amp
    out_unc[gid] = amp_unc
    out_snr[gid] = snr
    out_support[gid] = support
    out_continuum[gid] = continuum


def _pack_targets(df: pd.DataFrame, target_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    offsets: list[int] = []
    lengths: list[int] = []
    waves: list[np.ndarray] = []
    cbands: list[np.ndarray] = []
    fluxes: list[np.ndarray] = []
    uncs: list[np.ndarray] = []
    cursor = 0
    grouped = {str(target_id): rows.sort_values("cwave_um") for target_id, rows in df.groupby("target_id", sort=False)}
    for target_id in target_ids:
        rows = grouped.get(str(target_id))
        if rows is None:
            continue
        w = pd.to_numeric(rows["cwave_um"], errors="coerce").to_numpy(dtype=np.float32)
        cb = pd.to_numeric(rows["cband_um"], errors="coerce").to_numpy(dtype=np.float32)
        f = pd.to_numeric(rows["score_flux_uJy"], errors="coerce").to_numpy(dtype=np.float32)
        u = pd.to_numeric(rows["score_flux_unc_uJy"], errors="coerce").to_numpy(dtype=np.float32)
        offsets.append(cursor)
        lengths.append(int(len(rows)))
        cursor += int(len(rows))
        waves.append(w)
        cbands.append(cb)
        fluxes.append(f)
        uncs.append(u)
    return (
        np.asarray(offsets, dtype=np.int32),
        np.asarray(lengths, dtype=np.int32),
        np.concatenate(waves).astype(np.float32),
        np.concatenate(cbands).astype(np.float32),
        np.concatenate(fluxes).astype(np.float32),
        np.concatenate(uncs).astype(np.float32),
    )


def _target_ids_from_args(df: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    target_ids = list(df["target_id"].dropna().astype(str).drop_duplicates())
    requested = set(args.target_id or [])
    if args.target_ids_file:
        requested.update(
            line.strip()
            for line in args.target_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        requested.update(str(inj.get("target_id")) for inj in manifest.get("injections", []) if inj.get("target_id"))
    if requested:
        target_ids = [target_id for target_id in target_ids if target_id in requested]
    if args.max_targets:
        target_ids = target_ids[: int(args.max_targets)]
    return target_ids


def _load_scoring_table(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    if args.baseline_run_dir and args.injected_run_dir:
        paired_args = argparse.Namespace(
            baseline_run_dir=args.baseline_run_dir,
            injected_run_dir=args.injected_run_dir,
            baseline_spectra=args.baseline_spectra,
            injected_spectra=args.injected_spectra,
            flux_kind=args.flux_kind,
            uncertainty_mode=args.uncertainty_mode,
            allow_approx_wavelengths=args.allow_approx_wavelengths,
        )
        df = _load_pair(paired_args)
        df["score_flux_uJy"] = pd.to_numeric(df["delta_flux_uJy"], errors="coerce")
        df["score_flux_unc_uJy"] = pd.to_numeric(df["delta_flux_unc_uJy"], errors="coerce")
        return df, "paired_delta_warp"
    if args.run_dir:
        df = _load_raw_spectra(args)
        return df, "raw_spectrum_warp"
    raise SystemExit("Provide either --run-dir or both --baseline-run-dir and --injected-run-dir")


def _enrich_candidates(
    candidates: pd.DataFrame,
    measurements: pd.DataFrame,
    *,
    line_width_nm: float,
    min_template_response: float,
    local_window_um: float,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    by_target = {str(target_id): rows.sort_values("cwave_um") for target_id, rows in measurements.groupby("target_id", sort=False)}
    enriched = []
    for row in candidates.to_dict(orient="records"):
        target_id = str(row.get("target_id") or "")
        line_nm = float(row.get("candidate_line_nm") or np.nan)
        rows = by_target.get(target_id)
        out = dict(row)
        if rows is None or not np.isfinite(line_nm):
            enriched.append(out)
            continue
        wave = pd.to_numeric(rows["cwave_um"], errors="coerce").to_numpy(dtype=float)
        cband = pd.to_numeric(rows["cband_um"], errors="coerce").to_numpy(dtype=float)
        line_um = line_nm / 1000.0
        width_um = float(line_width_nm) / 1000.0
        fwhm = np.sqrt(np.maximum(cband, 0.0) ** 2 + max(width_um, 0.0) ** 2)
        sigma = fwhm / 2.354820045
        template = np.zeros_like(wave, dtype=float)
        good = np.isfinite(wave) & np.isfinite(sigma) & (sigma > 0.0)
        template[good] = np.exp(-0.5 * ((wave[good] - line_um) / sigma[good]) ** 2)
        near = good & (template >= min_template_response)
        near_idx = np.where(near)[0]
        if near_idx.size:
            best_idx = near_idx[np.argsort(template[near_idx])[::-1][:5]]
            image_ids = rows.get("image_id", pd.Series([""] * len(rows), index=rows.index)).astype(str).to_numpy()
            out["best_frame_ids"] = ",".join(str(image_ids[i]) for i in best_idx if str(image_ids[i]))
            if "detector" in rows:
                detectors = pd.to_numeric(rows["detector"], errors="coerce").to_numpy(dtype=float)
                out["detectors"] = ",".join(sorted({str(int(detectors[i])) for i in near_idx if np.isfinite(detectors[i])}))
        if "fatal_flag_present" in rows:
            flagged = rows["fatal_flag_present"].fillna(False).astype(bool).to_numpy()
            nearby = np.isfinite(wave) & (np.abs(wave - line_um) <= local_window_um)
            out["n_flagged_nearby"] = int(np.count_nonzero(flagged & nearby))
        enriched.append(out)
    return pd.DataFrame(enriched)


def main() -> None:
    parser = argparse.ArgumentParser(description="Warp dense blind matched-filter scoring over SPHEREx spectra.")
    parser.add_argument("--run-dir", type=Path, help="Raw spectra run directory.")
    parser.add_argument("--spectra-path", type=Path)
    parser.add_argument("--baseline-run-dir", type=Path, help="Baseline run for paired-delta mode.")
    parser.add_argument("--injected-run-dir", type=Path, help="Injected run for paired-delta mode.")
    parser.add_argument("--baseline-spectra", type=Path)
    parser.add_argument("--injected-spectra", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-line-nm", type=float, default=750.0)
    parser.add_argument("--max-line-nm", type=float, default=5000.0)
    parser.add_argument("--grid-step-nm", type=float, default=5.0)
    parser.add_argument("--line-width-nm", type=float, default=1.0)
    parser.add_argument("--cluster-gap-nm", type=float, default=15.0)
    parser.add_argument("--flux-kind", choices=["aperture", "psf"], default="aperture")
    parser.add_argument("--uncertainty-mode", choices=["quadrature", "injected", "baseline"], default="quadrature")
    parser.add_argument("--min-template-response", type=float, default=1e-3)
    parser.add_argument("--local-window-um", type=float, default=0.18)
    parser.add_argument("--continuum-exclude-template-above", type=float, default=0.25)
    parser.add_argument("--min-supporting-points", type=int, default=2)
    parser.add_argument("--min-snr", type=float, default=1.5)
    parser.add_argument("--target-id", action="append")
    parser.add_argument("--target-ids-file", type=Path)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-approx-wavelengths", action="store_true")
    args = parser.parse_args()

    if wp is None:
        raise SystemExit(f"Warp is not available: {_WARP_IMPORT_ERROR}")
    wp.init()

    t0 = time.perf_counter()
    measurements, mode = _load_scoring_table(args)
    target_ids = _target_ids_from_args(measurements, args)
    if not target_ids:
        raise SystemExit("No targets selected")

    lines = _line_grid(args.min_line_nm, args.max_line_nm, args.grid_step_nm, args.line_width_nm)
    line_nm = np.asarray([float(line["candidate_line_nm"]) for line in lines], dtype=np.float32)
    selected = measurements[measurements["target_id"].astype(str).isin(target_ids)].copy()
    offsets, lengths, wave, cband, flux, unc = _pack_targets(selected, target_ids)
    pack_sec = time.perf_counter() - t0

    n_scores = int(len(target_ids) * len(lines))
    t1 = time.perf_counter()
    offsets_dev = wp.array(offsets, dtype=wp.int32, device=args.device)
    lengths_dev = wp.array(lengths, dtype=wp.int32, device=args.device)
    wave_dev = wp.array(wave, dtype=wp.float32, device=args.device)
    cband_dev = wp.array(cband, dtype=wp.float32, device=args.device)
    flux_dev = wp.array(flux, dtype=wp.float32, device=args.device)
    unc_dev = wp.array(unc, dtype=wp.float32, device=args.device)
    line_dev = wp.array(line_nm, dtype=wp.float32, device=args.device)
    amp_dev = wp.empty(n_scores, dtype=wp.float32, device=args.device)
    amp_unc_dev = wp.empty(n_scores, dtype=wp.float32, device=args.device)
    snr_dev = wp.empty(n_scores, dtype=wp.float32, device=args.device)
    support_dev = wp.empty(n_scores, dtype=wp.int32, device=args.device)
    continuum_dev = wp.empty(n_scores, dtype=wp.float32, device=args.device)
    wp.launch(
        _blind_score_kernel,
        dim=n_scores,
        inputs=[
            offsets_dev,
            lengths_dev,
            wave_dev,
            cband_dev,
            flux_dev,
            unc_dev,
            line_dev,
            int(len(lines)),
            float(args.line_width_nm),
            float(args.min_template_response),
            float(args.local_window_um),
            float(args.continuum_exclude_template_above),
            int(args.min_supporting_points),
            amp_dev,
            amp_unc_dev,
            snr_dev,
            support_dev,
            continuum_dev,
        ],
        device=args.device,
    )
    wp.synchronize_device(args.device)
    kernel_sec = time.perf_counter() - t1

    t2 = time.perf_counter()
    snr = snr_dev.numpy().reshape((len(target_ids), len(lines)))
    amp = amp_dev.numpy().reshape((len(target_ids), len(lines)))
    amp_unc = amp_unc_dev.numpy().reshape((len(target_ids), len(lines)))
    support = support_dev.numpy().reshape((len(target_ids), len(lines)))
    continuum = continuum_dev.numpy().reshape((len(target_ids), len(lines)))
    rows = []
    target_arr, line_arr = np.where(snr >= float(args.min_snr))
    for ti, li in zip(target_arr.tolist(), line_arr.tolist(), strict=True):
        rows.append(
            {
                "target_id": target_ids[ti],
                "candidate_line_nm": float(line_nm[li]),
                "line_family": "blind",
                "nominal_line_nm": float(line_nm[li]),
                "offset_nm": 0.0,
                "line_width_nm": float(args.line_width_nm),
                "matched_flux_uJy": float(amp[ti, li]),
                "matched_flux_unc_uJy": float(amp_unc[ti, li]),
                "matched_snr": float(snr[ti, li]),
                "score": float(snr[ti, li]),
                "n_supporting_points": int(support[ti, li]),
                "n_flagged_nearby": 0,
                "local_continuum_uJy": float(continuum[ti, li]),
                "local_residual_rms_uJy": np.nan,
                "continuum_points": np.nan,
                "wavelength_min_um": float(args.min_line_nm / 1000.0),
                "wavelength_max_um": float(args.max_line_nm / 1000.0),
                "best_frame_ids": "",
                "detectors": "",
                "candidate_status": "candidate",
                "score_mode": f"{mode}_mean_continuum",
                "flux_kind": args.flux_kind,
            }
        )
    candidates = pd.DataFrame(rows).sort_values("matched_snr", ascending=False) if rows else pd.DataFrame()
    candidates = _enrich_candidates(
        candidates,
        selected,
        line_width_nm=args.line_width_nm,
        min_template_response=args.min_template_response,
        local_window_um=args.local_window_um,
    )
    clusters = _cluster_candidates(candidates, args.cluster_gap_nm)
    collect_sec = time.perf_counter() - t2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(lines).to_parquet(args.output_dir / "blind_line_grid.parquet", index=False)
    candidates.to_parquet(args.output_dir / "blind_matched_filter_candidates.parquet", index=False)
    clusters.to_parquet(args.output_dir / "blind_candidate_clusters.parquet", index=False)
    summary = {
        "mode": mode,
        "device": args.device,
        "run_dir": str(args.run_dir) if args.run_dir else None,
        "baseline_run_dir": str(args.baseline_run_dir) if args.baseline_run_dir else None,
        "injected_run_dir": str(args.injected_run_dir) if args.injected_run_dir else None,
        "target_count": len(target_ids),
        "line_count": len(lines),
        "score_rows": n_scores,
        "candidate_rows": int(len(candidates)),
        "cluster_rows": int(len(clusters)),
        "pack_sec": pack_sec,
        "kernel_sec": kernel_sec,
        "collect_sec": collect_sec,
        "total_sec": time.perf_counter() - t0,
        "scores_per_sec_kernel": float(n_scores / kernel_sec) if kernel_sec > 0 else None,
        "min_snr": args.min_snr,
        "grid_step_nm": args.grid_step_nm,
        "continuum": "mean_local_excluding_template_core",
    }
    (args.output_dir / "blind_classifier_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
