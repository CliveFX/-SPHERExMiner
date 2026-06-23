#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

from spherex_laser_miner.calibration import image_to_ujy_per_pixel, load_sapm, variance_to_ujy2
from spherex_laser_miner.config import MinerConfig
from spherex_laser_miner.photometry.psf_forced import (
    extract_native_psf_kernel,
    fit_psf_cpu,
    fit_psf_cpu_with_grid,
    fit_psf_warp_grid_batch,
)


SPEXPI_SRC = Path("/mnt/niroseti/spherex_cache/external/source/spexpi/src")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare forced PSF photometry implementations on existing run spectra rows. "
            "This is a correctness harness, not the production extraction path."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("/mnt/niroseti/spherex_cache"))
    parser.add_argument("--release", default="qr2")
    parser.add_argument("--target-id", action="append", default=[])
    parser.add_argument("--mag-min", type=float)
    parser.add_argument("--mag-max", type=float)
    parser.add_argument("--max-rows", type=int, default=80)
    parser.add_argument("--max-rows-per-target", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--kernel-radius-native", type=int, default=5)
    parser.add_argument("--grid-half-range-pix", type=float, default=1.0)
    parser.add_argument("--grid-step-pix", type=float, default=0.5)
    parser.add_argument("--grid-metric", choices=("snr", "chi2"), default="snr")
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--skip-spexpi", action="store_true")
    parser.add_argument("--enable-gpu", action="store_true")
    parser.add_argument("--gpu-kernel-build-mode", choices=("gpu_bilinear", "gpu_spline", "cpu_scipy"), default="gpu_bilinear")
    parser.add_argument(
        "--allow-experimental-warp",
        action="store_true",
        help="Actually run the current experimental Warp PSF kernel. It is known to be unstable and may hang.",
    )
    parser.add_argument("--warp-device", default="cuda:0")
    args = parser.parse_args()

    output_dir = args.output_dir or args.run_dir / "benchmarks" / "psf_correctness"
    output_dir.mkdir(parents=True, exist_ok=True)
    spectra_path = args.run_dir / "spectra" / "target_spectra.parquet"
    if not spectra_path.exists():
        raise FileNotFoundError(spectra_path)

    rows = _select_rows(
        pd.read_parquet(spectra_path),
        target_ids=args.target_id,
        mag_min=args.mag_min,
        mag_max=args.mag_max,
        max_rows=args.max_rows,
        max_rows_per_target=args.max_rows_per_target,
        random_seed=args.random_seed,
    )
    if rows.empty:
        raise SystemExit("No rows selected for PSF comparison")

    print(f"selected {len(rows)} rows from {spectra_path}", flush=True)
    spexpi = None if args.skip_spexpi else _load_spexpi()
    cfg = MinerConfig(cache_root=args.cache_root, release=args.release)
    result_rows: list[dict[str, object]] = []
    timings: dict[str, float] = {"cpu_fixed_sec": 0.0, "cpu_grid_sec": 0.0, "spexpi_sec": 0.0, "gpu_sec": 0.0}

    for file_idx, (input_file_path, group) in enumerate(rows.groupby("input_file_path", dropna=False), start=1):
        input_path = Path(str(input_file_path))
        print(f"[{file_idx}] {input_path.name}: {len(group)} row(s)", flush=True)
        with fits.open(input_path, memmap=True) as hdul:
            image_hdu = hdul["IMAGE"]
            detector = int(image_hdu.header.get("DETECTOR", group["detector"].iloc[0]))
            sapm_data, sapm_header, sapm_path = load_sapm(args.cache_root, args.release, detector)
            flux_ujy = image_to_ujy_per_pixel(image_hdu.data, image_hdu.header, sapm_data, sapm_header)
            var_ujy2 = (
                variance_to_ujy2(hdul["VARIANCE"].data, image_hdu.header, sapm_data, sapm_header)
                if "VARIANCE" in hdul
                else None
            )
            flags = hdul["FLAGS"].data if "FLAGS" in hdul else None

            gpu_items = []
            for _, row in group.iterrows():
                base = _base_row(row, input_path, detector, sapm_path)
                x = float(row["x_pix"])
                y = float(row["y_pix"])
                try:
                    t0 = time.perf_counter()
                    kernel = extract_native_psf_kernel(
                        hdul=hdul,
                        x_pix=x,
                        y_pix=y,
                        kernel_radius_native=args.kernel_radius_native,
                    )
                    fixed = fit_psf_cpu(
                        flux_ujy=flux_ujy,
                        var_ujy2=var_ujy2,
                        flags=flags,
                        psf_kernel=kernel.data,
                        x_pix=x,
                        y_pix=y,
                        x_init=x,
                        y_init=y,
                        fatal_flag_bits=cfg.fatal_flag_bits,
                        sector=kernel.sector,
                        backend="cpu_fixed",
                    )
                    timings["cpu_fixed_sec"] += time.perf_counter() - t0
                    result_rows.append({**base, **_prefix(fixed.to_dict(), "cpu_fixed")})
                    gpu_items.append((base, kernel, x, y))
                except Exception as exc:
                    result_rows.append({**base, "backend": "cpu_fixed", "status": f"error:{type(exc).__name__}", "error": str(exc)})

                if not args.skip_grid:
                    t0 = time.perf_counter()
                    try:
                        grid, _grid_kernel = fit_psf_cpu_with_grid(
                            hdul=hdul,
                            flux_ujy=flux_ujy,
                            var_ujy2=var_ujy2,
                            flags=flags,
                            x_pix=x,
                            y_pix=y,
                            fatal_flag_bits=cfg.fatal_flag_bits,
                            kernel_radius_native=args.kernel_radius_native,
                            half_range_pix=args.grid_half_range_pix,
                            step_pix=args.grid_step_pix,
                            metric=args.grid_metric,
                        )
                        result_rows.append({**base, **_prefix(grid.to_dict(), "cpu_grid")})
                    except Exception as exc:
                        result_rows.append({**base, "backend": "cpu_grid", "status": f"error:{type(exc).__name__}", "error": str(exc)})
                    timings["cpu_grid_sec"] += time.perf_counter() - t0

                if spexpi is not None:
                    t0 = time.perf_counter()
                    try:
                        hdul_like = fits.HDUList([fits.PrimaryHDU(), image_hdu, hdul["PSF"]])
                        sp_flux, sp_unc, sp_info = spexpi.psf_flux_fit_with_local_grid_search(
                            hdul_like,
                            flux_ujy,
                            var_ujy2,
                            flags,
                            x,
                            y,
                            detector=detector,
                            cfg=_spexpi_cfg(spexpi, args),
                        )
                        result_rows.append({**base, **_spexpi_row(sp_flux, sp_unc, sp_info)})
                    except Exception as exc:
                        result_rows.append({**base, "backend": "spexpi", "status": f"error:{type(exc).__name__}", "error": str(exc)})
                    timings["spexpi_sec"] += time.perf_counter() - t0

            if args.enable_gpu and not args.allow_experimental_warp and gpu_items:
                for base, _kernel, _x, _y in gpu_items:
                    result_rows.append(
                        {
                            **base,
                            "backend": "gpu_grid",
                            "status": "skipped_experimental_warp_disabled",
                            "error": "Use --allow-experimental-warp only while debugging the Warp PSF grid kernel.",
                        }
                    )
            elif args.enable_gpu and gpu_items:
                t0 = time.perf_counter()
                try:
                    gpu = fit_psf_warp_grid_batch(
                        hdul=hdul,
                        flux_ujy=flux_ujy,
                        var_ujy2=var_ujy2,
                        flags=flags,
                        x_pix=np.asarray([item[2] for item in gpu_items], dtype=float),
                        y_pix=np.asarray([item[3] for item in gpu_items], dtype=float),
                        fatal_flag_bits=cfg.fatal_flag_bits,
                        kernel_radius_native=args.kernel_radius_native,
                        half_range_pix=args.grid_half_range_pix,
                        step_pix=args.grid_step_pix,
                        metric=args.grid_metric,
                        kernel_build_mode=args.gpu_kernel_build_mode,
                        device=args.warp_device,
                    )
                    for (base, _kernel, _x, _y), result in zip(gpu_items, gpu.rows, strict=True):
                        result_rows.append({**base, **_prefix(result.to_dict(), "gpu_grid")})
                except Exception as exc:
                    for base, _kernel, _x, _y in gpu_items:
                        result_rows.append({**base, "backend": "gpu_grid", "status": f"error:{type(exc).__name__}", "error": str(exc)})
                timings["gpu_sec"] += time.perf_counter() - t0

    out = pd.DataFrame(result_rows)
    csv_path = output_dir / "psf_comparison_rows.csv"
    parquet_path = output_dir / "psf_comparison_rows.parquet"
    out.to_csv(csv_path, index=False)
    out.to_parquet(parquet_path, index=False)
    summary = _summarize(out, timings, args)
    summary_path = output_dir / "psf_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_summary(out, output_dir)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {csv_path}", flush=True)


def _select_rows(
    df: pd.DataFrame,
    *,
    target_ids: list[str],
    mag_min: float | None,
    mag_max: float | None,
    max_rows: int,
    max_rows_per_target: int,
    random_seed: int,
) -> pd.DataFrame:
    required = {"target_id", "input_file_path", "x_pix", "y_pix", "detector"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"spectra table is missing required columns: {missing}")
    work = df.copy()
    work = work[work["input_file_path"].notna()]
    work = work[np.isfinite(pd.to_numeric(work["x_pix"], errors="coerce"))]
    work = work[np.isfinite(pd.to_numeric(work["y_pix"], errors="coerce"))]
    if target_ids:
        work = work[work["target_id"].astype(str).isin(set(target_ids))]
    if mag_min is not None and "phot_g_mean_mag" in work:
        work = work[pd.to_numeric(work["phot_g_mean_mag"], errors="coerce").ge(float(mag_min))]
    if mag_max is not None and "phot_g_mean_mag" in work:
        work = work[pd.to_numeric(work["phot_g_mean_mag"], errors="coerce").le(float(mag_max))]

    pieces = []
    manual_ids = {"simp0136", "ucs_0972"}
    manual = work[work["target_id"].astype(str).isin(manual_ids)]
    if not manual.empty:
        pieces.append(manual.groupby("target_id", group_keys=False).head(max_rows_per_target))
    if target_ids:
        pieces.append(work.groupby("target_id", group_keys=False).head(max_rows_per_target))
    elif "phot_g_mean_mag" in work:
        binned = work.copy()
        binned["mag_bin"] = np.floor(pd.to_numeric(binned["phot_g_mean_mag"], errors="coerce"))
        binned = binned[np.isfinite(binned["mag_bin"])]
        pieces.append(
            binned.groupby("mag_bin", group_keys=False).apply(
                lambda group: group.sample(
                    n=min(len(group), min(5, max_rows_per_target)),
                    random_state=random_seed,
                    replace=False,
                )
            )
        )
    else:
        pieces.append(work.sample(n=min(len(work), max_rows), random_state=random_seed))

    selected = pd.concat(pieces, ignore_index=True).drop_duplicates(["target_id", "input_file_path", "x_pix", "y_pix"])
    if len(selected) > max_rows:
        selected = selected.sample(n=max_rows, random_state=random_seed)
    sort_cols = ["input_file_path", "target_id"]
    if "cwave_um" in selected:
        sort_cols.append("cwave_um")
    return selected.sort_values(sort_cols)


def _load_spexpi():
    if not SPEXPI_SRC.exists():
        return None
    sys.path.insert(0, str(SPEXPI_SRC))
    from spexpi import spherex_pipeline  # type: ignore

    return spherex_pipeline


def _spexpi_cfg(spexpi, args: argparse.Namespace):
    cfg = spexpi.PipelineConfig()
    cfg.photometry_method = "psf"
    cfg.psf_kernel_radius_native = int(args.kernel_radius_native)
    cfg.psf_shift_sign = 1.0
    cfg.psf_fit_background_gradient = False
    cfg.enable_psf_local_grid_search = not bool(args.skip_grid)
    cfg.psf_local_grid_half_range_pix = float(args.grid_half_range_pix)
    cfg.psf_local_grid_step_pix = float(args.grid_step_pix)
    cfg.psf_local_grid_metric = str(args.grid_metric)
    return cfg


def _base_row(row: pd.Series, input_path: Path, detector: int, sapm_path: Path) -> dict[str, object]:
    return {
        "target_id": str(row.get("target_id", "")),
        "target_type": row.get("target_type", ""),
        "phot_g_mean_mag": _maybe_float(row.get("phot_g_mean_mag")),
        "cwave_um": _maybe_float(row.get("cwave_um")),
        "cband_um": _maybe_float(row.get("cband_um")),
        "aperture_flux_uJy": _maybe_float(row.get("aperture_flux_uJy")),
        "aperture_flux_unc_uJy": _maybe_float(row.get("aperture_flux_unc_uJy")),
        "detector": int(detector),
        "image_id": row.get("image_id", input_path.stem),
        "input_file_path": str(input_path),
        "x_pix": _maybe_float(row.get("x_pix")),
        "y_pix": _maybe_float(row.get("y_pix")),
        "sapm_file_path": str(sapm_path),
    }


def _prefix(values: dict[str, object], backend: str) -> dict[str, object]:
    return {
        "backend": backend,
        "status": values.get("status"),
        "flux_uJy": values.get("flux_uJy"),
        "flux_unc_uJy": values.get("flux_unc_uJy"),
        "snr": _safe_ratio(values.get("flux_uJy"), values.get("flux_unc_uJy")),
        "background_uJy_per_pix": values.get("background_uJy_per_pix"),
        "chi2": values.get("chi2"),
        "reduced_chi2": values.get("reduced_chi2"),
        "dof": values.get("dof"),
        "n_valid": values.get("n_valid"),
        "sector": values.get("sector"),
        "x_fit": values.get("x_fit"),
        "y_fit": values.get("y_fit"),
        "fit_offset_pix": values.get("fit_offset_pix"),
        "kernel_sum": values.get("kernel_sum"),
        "kernel_shape_y": values.get("kernel_shape_y"),
        "kernel_shape_x": values.get("kernel_shape_x"),
        "grid_status": values.get("grid_status"),
        "grid_metric": values.get("grid_metric"),
        "grid_n_trials": values.get("grid_n_trials"),
        "grid_n_valid": values.get("grid_n_valid"),
        "grid_best_score": values.get("grid_best_score"),
    }


def _spexpi_row(flux_uJy: float, unc_uJy: float, info: dict[str, object]) -> dict[str, object]:
    return {
        "backend": "spexpi",
        "status": info.get("psf_status"),
        "flux_uJy": float(flux_uJy) if np.isfinite(flux_uJy) else np.nan,
        "flux_unc_uJy": float(unc_uJy) if np.isfinite(unc_uJy) else np.nan,
        "snr": _safe_ratio(flux_uJy, unc_uJy),
        "background_uJy_per_pix": info.get("psf_background_const"),
        "chi2": info.get("psf_chi2"),
        "reduced_chi2": info.get("psf_reduced_chi2"),
        "dof": np.nan,
        "n_valid": info.get("psf_n_valid"),
        "sector": np.nan,
        "x_fit": info.get("psf_x_fit"),
        "y_fit": info.get("psf_y_fit"),
        "fit_offset_pix": info.get("psf_fit_offset_pix"),
        "kernel_sum": info.get("psf_kernel_sum"),
        "kernel_shape_y": info.get("psf_kernel_shape_y"),
        "kernel_shape_x": info.get("psf_kernel_shape_x"),
        "grid_status": info.get("psf_grid_status"),
        "grid_metric": info.get("psf_grid_metric"),
        "grid_n_trials": info.get("psf_grid_n_trials"),
        "grid_n_valid": info.get("psf_grid_n_valid"),
        "grid_best_score": info.get("psf_grid_best_score"),
    }


def _summarize(df: pd.DataFrame, timings: dict[str, float], args: argparse.Namespace) -> dict[str, object]:
    ok = df[df["status"].astype(str).eq("ok")].copy() if "status" in df else pd.DataFrame()
    pivot = ok.pivot_table(index=["target_id", "input_file_path", "x_pix", "y_pix"], columns="backend", values="flux_uJy", aggfunc="first")
    comparisons: dict[str, object] = {}
    for other in ("cpu_grid", "gpu_fixed", "gpu_grid", "spexpi"):
        if "cpu_fixed" in pivot and other in pivot:
            diff = pivot[other] - pivot["cpu_fixed"]
            denom = pivot["cpu_fixed"].abs().replace(0, np.nan)
            comparisons[f"{other}_minus_cpu_fixed_median_uJy"] = _maybe_float(diff.median())
            comparisons[f"{other}_minus_cpu_fixed_median_frac"] = _maybe_float((diff / denom).median())
            comparisons[f"{other}_compared_rows"] = int(diff.notna().sum())
        if "cpu_grid" in pivot and other in pivot and other != "cpu_grid":
            diff = pivot[other] - pivot["cpu_grid"]
            denom = pivot["cpu_grid"].abs().replace(0, np.nan)
            comparisons[f"{other}_minus_cpu_grid_median_uJy"] = _maybe_float(diff.median())
            comparisons[f"{other}_minus_cpu_grid_median_frac"] = _maybe_float((diff / denom).median())
            comparisons[f"{other}_vs_cpu_grid_rows"] = int(diff.notna().sum())
    return {
        "run_dir": str(args.run_dir),
        "rows": int(len(df)),
        "targets": int(df["target_id"].nunique()) if "target_id" in df else 0,
        "backend_counts": df["backend"].value_counts(dropna=False).to_dict() if "backend" in df else {},
        "status_counts": _status_counts(df),
        "timings": timings,
        "comparisons": comparisons,
        "notes": "Correctness harness. CPU grid and SPExPI rebuild PSF kernels per grid trial; not production performance.",
    }


def _status_counts(df: pd.DataFrame) -> dict[str, int]:
    if not {"backend", "status"} <= set(df.columns):
        return {}
    counts = df.groupby(["backend", "status"]).size().astype(int)
    return {f"{backend}:{status}": int(value) for (backend, status), value in counts.items()}


def _plot_summary(df: pd.DataFrame, output_dir: Path) -> None:
    ok = df[df["status"].astype(str).eq("ok")].copy()
    if ok.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for backend, group in ok.groupby("backend"):
        ax.scatter(group["cwave_um"], group["flux_uJy"], s=18, alpha=0.7, label=str(backend))
    ax.set_xlabel("wavelength (um)")
    ax.set_ylabel("PSF flux (uJy)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "psf_flux_by_backend.png", dpi=160)
    plt.close(fig)


def _safe_ratio(a: object, b: object) -> float:
    aa = _maybe_float(a)
    bb = _maybe_float(b)
    if aa is None or bb is None or bb == 0:
        return np.nan
    return float(aa / bb)


def _maybe_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


if __name__ == "__main__":
    main()
