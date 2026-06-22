from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from spherex_laser_miner.calibration import image_to_ujy_per_pixel, load_sapm, variance_to_ujy2
from spherex_laser_miner.config import load_config

try:
    import warp as wp
except Exception as exc:  # pragma: no cover - dependency check path
    wp = None
    _WARP_IMPORT_ERROR = exc
else:
    _WARP_IMPORT_ERROR = None


MAX_RADIUS_PIX = 6


if wp is not None:

    @wp.kernel
    def _aperture_mean_kernel(
        flux: wp.array(dtype=wp.float32),
        var: wp.array(dtype=wp.float32),
        flags: wp.array(dtype=wp.uint32),
        width: int,
        height: int,
        x_targets: wp.array(dtype=wp.float32),
        y_targets: wp.array(dtype=wp.float32),
        bad_mask_value: wp.uint32,
        aperture_radius_pix: float,
        annulus_inner_pix: float,
        annulus_outer_pix: float,
        out_flux: wp.array(dtype=wp.float32),
        out_unc: wp.array(dtype=wp.float32),
        out_bkg: wp.array(dtype=wp.float32),
        out_area: wp.array(dtype=wp.float32),
        out_bad: wp.array(dtype=wp.int32),
        out_status: wp.array(dtype=wp.int32),
    ):
        tid = wp.tid()
        x0 = x_targets[tid]
        y0 = y_targets[tid]
        ix0 = int(wp.floor(x0))
        iy0 = int(wp.floor(y0))
        ap_sum = float(0.0)
        ap_var_sum = float(0.0)
        ap_area = float(0.0)
        bad_ap = int(0)
        ann_sum = float(0.0)
        ann_sumsq = float(0.0)
        ann_count = int(0)

        for dy in range(-MAX_RADIUS_PIX, MAX_RADIUS_PIX + 1):
            for dx in range(-MAX_RADIUS_PIX, MAX_RADIUS_PIX + 1):
                ix = ix0 + dx
                iy = iy0 + dy
                if ix >= 0 and ix < width and iy >= 0 and iy < height:
                    ddx = float(ix) - x0
                    ddy = float(iy) - y0
                    rr = wp.sqrt(ddx * ddx + ddy * ddy)
                    flat = iy * width + ix
                    value = flux[flat]
                    good = (value == value) and ((flags[flat] & bad_mask_value) == wp.uint32(0))
                    if rr <= aperture_radius_pix:
                        if good:
                            ap_sum += value
                            ap_var_sum += var[flat]
                            ap_area += 1.0
                        else:
                            bad_ap += 1
                    if rr > annulus_inner_pix and rr <= annulus_outer_pix and good:
                        ann_sum += value
                        ann_sumsq += value * value
                        ann_count += 1

        if ap_area <= 0.0 or ann_count <= 0:
            out_flux[tid] = 0.0
            out_unc[tid] = 0.0
            out_bkg[tid] = 0.0
            out_area[tid] = ap_area
            out_bad[tid] = bad_ap
            out_status[tid] = 1
            return

        bkg = ann_sum / float(ann_count)
        ann_var = wp.max(float(0.0), ann_sumsq / float(ann_count) - bkg * bkg)
        bkg_var_per_pix = ann_var / float(ann_count)
        measured = ap_sum - bkg * ap_area
        total_var = ap_var_sum + ap_area * ap_area * bkg_var_per_pix
        out_flux[tid] = measured
        out_unc[tid] = wp.sqrt(wp.max(float(0.0), total_var))
        out_bkg[tid] = bkg
        out_area[tid] = ap_area
        out_bad[tid] = bad_ap
        out_status[tid] = 0


def cpu_center_mean_aperture(
    flux: np.ndarray,
    var: np.ndarray,
    flags: np.ndarray,
    x_targets: np.ndarray,
    y_targets: np.ndarray,
    bad_mask_value: int,
    aperture_radius_pix: float,
    annulus_inner_pix: float,
    annulus_outer_pix: float,
) -> dict[str, np.ndarray]:
    height, width = flux.shape
    out_flux = np.full(len(x_targets), np.nan, dtype=np.float64)
    out_unc = np.full(len(x_targets), np.nan, dtype=np.float64)
    out_bkg = np.full(len(x_targets), np.nan, dtype=np.float64)
    out_area = np.zeros(len(x_targets), dtype=np.float64)
    out_bad = np.zeros(len(x_targets), dtype=np.int32)
    out_status = np.ones(len(x_targets), dtype=np.int32)
    for i, (x0, y0) in enumerate(zip(x_targets, y_targets, strict=True)):
        ix0 = int(np.floor(x0))
        iy0 = int(np.floor(y0))
        ap_sum = 0.0
        ap_var_sum = 0.0
        ap_area = 0.0
        bad_ap = 0
        ann = []
        for iy in range(iy0 - MAX_RADIUS_PIX, iy0 + MAX_RADIUS_PIX + 1):
            if iy < 0 or iy >= height:
                continue
            for ix in range(ix0 - MAX_RADIUS_PIX, ix0 + MAX_RADIUS_PIX + 1):
                if ix < 0 or ix >= width:
                    continue
                rr = float(np.hypot(ix - x0, iy - y0))
                value = float(flux[iy, ix])
                good = np.isfinite(value) and (int(flags[iy, ix]) & bad_mask_value) == 0
                if rr <= aperture_radius_pix:
                    if good:
                        ap_sum += value
                        ap_var_sum += float(var[iy, ix])
                        ap_area += 1.0
                    else:
                        bad_ap += 1
                if annulus_inner_pix < rr <= annulus_outer_pix and good:
                    ann.append(value)
        out_area[i] = ap_area
        out_bad[i] = bad_ap
        if ap_area <= 0 or not ann:
            continue
        ann_values = np.asarray(ann, dtype=np.float64)
        bkg = float(np.mean(ann_values))
        ann_var = float(np.var(ann_values))
        measured = ap_sum - bkg * ap_area
        total_var = ap_var_sum + ap_area * ap_area * ann_var / len(ann_values)
        out_flux[i] = measured
        out_unc[i] = float(np.sqrt(max(0.0, total_var)))
        out_bkg[i] = bkg
        out_status[i] = 0
    return {
        "flux_uJy": out_flux,
        "unc_uJy": out_unc,
        "background_uJy_per_pix": out_bkg,
        "area_pix": out_area,
        "bad_aperture_pixels": out_bad,
        "status": out_status,
    }


def warp_center_mean_aperture(
    flux: np.ndarray,
    var: np.ndarray,
    flags: np.ndarray,
    x_targets: np.ndarray,
    y_targets: np.ndarray,
    bad_mask_value: int,
    aperture_radius_pix: float,
    annulus_inner_pix: float,
    annulus_outer_pix: float,
    device: str,
) -> dict[str, np.ndarray]:
    if wp is None:
        raise RuntimeError(f"warp is not available: {_WARP_IMPORT_ERROR}")
    wp.init()
    height, width = flux.shape
    n = len(x_targets)
    flux_flat = np.ascontiguousarray(flux.astype(np.float32).ravel())
    var_flat = np.ascontiguousarray(np.nan_to_num(var, nan=0.0).astype(np.float32).ravel())
    flags_flat = np.ascontiguousarray(flags.astype(np.uint32).ravel())
    out_flux = wp.empty(n, dtype=wp.float32, device=device)
    out_unc = wp.empty(n, dtype=wp.float32, device=device)
    out_bkg = wp.empty(n, dtype=wp.float32, device=device)
    out_area = wp.empty(n, dtype=wp.float32, device=device)
    out_bad = wp.empty(n, dtype=wp.int32, device=device)
    out_status = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        _aperture_mean_kernel,
        dim=n,
        inputs=[
            wp.array(flux_flat, dtype=wp.float32, device=device),
            wp.array(var_flat, dtype=wp.float32, device=device),
            wp.array(flags_flat, dtype=wp.uint32, device=device),
            int(width),
            int(height),
            wp.array(np.asarray(x_targets, dtype=np.float32), dtype=wp.float32, device=device),
            wp.array(np.asarray(y_targets, dtype=np.float32), dtype=wp.float32, device=device),
            wp.uint32(bad_mask_value),
            float(aperture_radius_pix),
            float(annulus_inner_pix),
            float(annulus_outer_pix),
            out_flux,
            out_unc,
            out_bkg,
            out_area,
            out_bad,
            out_status,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    return {
        "flux_uJy": out_flux.numpy().astype(np.float64),
        "unc_uJy": out_unc.numpy().astype(np.float64),
        "background_uJy_per_pix": out_bkg.numpy().astype(np.float64),
        "area_pix": out_area.numpy().astype(np.float64),
        "bad_aperture_pixels": out_bad.numpy(),
        "status": out_status.numpy(),
    }


def _load_known_frame(run_dir: Path, image_id: str | None, n_targets: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    shard_root = run_dir / "field_shards"
    paths = sorted(shard_root.glob("image_id=*/measurements.parquet"))
    if image_id is not None:
        paths = [p for p in paths if p.parent.name == f"image_id={image_id}" or image_id in p.parent.name]
    if not paths:
        raise FileNotFoundError(f"No measurement shards found under {shard_root}")
    for path in paths:
        rows = pd.read_parquet(path)
        rows = rows[np.isfinite(pd.to_numeric(rows["x_pix"], errors="coerce"))].copy()
        if len(rows) >= n_targets:
            selected = rows.head(n_targets).reset_index(drop=True)
            fits_path = Path(str(selected["input_file_path"].iloc[0]))
            detector = int(selected["detector"].iloc[0])
            with fits.open(fits_path, memmap=True) as hdul:
                image_hdu = hdul["IMAGE"]
                cfg = load_config()
                sapm_data, sapm_header, _ = load_sapm(cfg.cache_root, cfg.release, detector)
                flux = image_to_ujy_per_pixel(image_hdu.data, image_hdu.header, sapm_data, sapm_header)
                if "VARIANCE" in hdul:
                    var = variance_to_ujy2(hdul["VARIANCE"].data, image_hdu.header, sapm_data, sapm_header)
                else:
                    var = np.zeros_like(flux, dtype=np.float32)
                if "FLAGS" in hdul:
                    flags = np.asarray(hdul["FLAGS"].data, dtype=np.uint32)
                else:
                    flags = np.zeros_like(flux, dtype=np.uint32)
            return selected, np.asarray(flux), np.asarray(var), flags
    raise RuntimeError(f"No shard under {run_dir} had at least {n_targets} targets")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare standalone Warp aperture photometry with matching CPU math.")
    parser.add_argument("--run-dir", type=Path, default=Path("/mnt/niroseti/spherex_cache/runs/ucs0972_g14_16_n10_f40_test1"))
    parser.add_argument("--image-id", default=None)
    parser.add_argument("--n-targets", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--aperture-radius-pix", type=float, default=2.0)
    parser.add_argument("--annulus-inner-pix", type=float, default=4.0)
    parser.add_argument("--annulus-outer-pix", type=float, default=6.0)
    args = parser.parse_args()

    cfg = load_config()
    bad_mask_value = sum(1 << int(bit) for bit in cfg.fatal_flag_bits)
    rows, flux, var, flags = _load_known_frame(args.run_dir, args.image_id, args.n_targets)
    x = rows["x_pix"].to_numpy(dtype=np.float64)
    y = rows["y_pix"].to_numpy(dtype=np.float64)
    repeats = max(1, int(args.repeat))
    cpu_times = []
    gpu_times = []
    cpu = {}
    gpu = {}
    for _ in range(repeats):
        t0 = time.perf_counter()
        cpu = cpu_center_mean_aperture(
            flux,
            var,
            flags,
            x,
            y,
            bad_mask_value,
            args.aperture_radius_pix,
            args.annulus_inner_pix,
            args.annulus_outer_pix,
        )
        cpu_times.append(time.perf_counter() - t0)
    for _ in range(repeats):
        t0 = time.perf_counter()
        gpu = warp_center_mean_aperture(
            flux,
            var,
            flags,
            x,
            y,
            bad_mask_value,
            args.aperture_radius_pix,
            args.annulus_inner_pix,
            args.annulus_outer_pix,
            args.device,
        )
        gpu_times.append(time.perf_counter() - t0)
    diff = gpu["flux_uJy"] - cpu["flux_uJy"]
    rel = diff / np.where(np.abs(cpu["flux_uJy"]) > 0, np.abs(cpu["flux_uJy"]), np.nan)
    target_rows = []
    for i, row in rows.iterrows():
        target_rows.append(
            {
                "target_id": str(row["target_id"]),
                "target_type": str(row.get("target_type", "")),
                "x_pix": float(row["x_pix"]),
                "y_pix": float(row["y_pix"]),
                "main_calibrated_flux_uJy": float(row.get("aperture_flux_uJy", np.nan)),
                "cpu_center_mean_flux_uJy": float(cpu["flux_uJy"][i]),
                "warp_center_mean_flux_uJy": float(gpu["flux_uJy"][i]),
                "warp_minus_cpu_uJy": float(diff[i]),
                "warp_minus_cpu_fraction": None if not np.isfinite(rel[i]) else float(rel[i]),
                "cpu_unc_uJy": float(cpu["unc_uJy"][i]),
                "warp_unc_uJy": float(gpu["unc_uJy"][i]),
                "cpu_status": int(cpu["status"][i]),
                "warp_status": int(gpu["status"][i]),
                "fatal_flag_present": bool(row.get("fatal_flag_present", False)),
            }
        )
    summary = {
        "run_dir": str(args.run_dir),
        "image_id": str(rows["image_id"].iloc[0]),
        "input_file_path": str(rows["input_file_path"].iloc[0]),
        "device": args.device,
        "n_targets": int(len(rows)),
        "algorithm": "center-pixel aperture with mean annulus background; standalone prototype, not SPExPI exact",
        "repeat": repeats,
        "timing": {
            "cpu_median_ms": float(np.median(cpu_times) * 1000.0),
            "cpu_min_ms": float(np.min(cpu_times) * 1000.0),
            "warp_median_ms_including_transfer": float(np.median(gpu_times) * 1000.0),
            "warp_min_ms_including_transfer": float(np.min(gpu_times) * 1000.0),
            "targets_per_sec_cpu_median": float(len(rows) / np.median(cpu_times)),
            "targets_per_sec_warp_median_including_transfer": float(len(rows) / np.median(gpu_times)),
        },
        "max_abs_flux_diff_uJy": float(np.nanmax(np.abs(diff))),
        "median_abs_flux_diff_uJy": float(np.nanmedian(np.abs(diff))),
        "max_abs_flux_rel_diff": float(np.nanmax(np.abs(rel))),
        "targets": target_rows,
    }
    text = json.dumps(summary, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
