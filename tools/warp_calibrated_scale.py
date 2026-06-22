from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spherex_laser_miner.config import load_config
from tools.warp_photometry_scale import _load_frame, _make_targets, _parse_ints, _resolve_devices

try:
    import warp as wp
except Exception as exc:  # pragma: no cover - dependency check path
    wp = None
    _WARP_IMPORT_ERROR = exc
else:
    _WARP_IMPORT_ERROR = None


MAX_RADIUS_PIX = 6
SUBPIX = 5
ANNULUS_MAX_PIX = 128
SIGMA_CLIP_ITERS = 5


if wp is not None:

    @wp.kernel
    def _weighted_calibrated_aperture_kernel(
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
        ann_values = wp.zeros(shape=ANNULUS_MAX_PIX, dtype=wp.float32)
        ann_active = wp.zeros(shape=ANNULUS_MAX_PIX, dtype=wp.int32)
        clipped_values = wp.zeros(shape=ANNULUS_MAX_PIX, dtype=wp.float32)

        for dy in range(-MAX_RADIUS_PIX, MAX_RADIUS_PIX + 1):
            for dx in range(-MAX_RADIUS_PIX, MAX_RADIUS_PIX + 1):
                ix = ix0 + dx
                iy = iy0 + dy
                if ix >= 0 and ix < width and iy >= 0 and iy < height:
                    flat = iy * width + ix
                    value = flux[flat]
                    good = (value == value) and ((flags[flat] & bad_mask_value) == wp.uint32(0))
                    ddx_center = float(ix) - x0
                    ddy_center = float(iy) - y0
                    rr_center = wp.sqrt(ddx_center * ddx_center + ddy_center * ddy_center)

                    if rr_center <= aperture_radius_pix and not good:
                        bad_ap += 1

                    if rr_center > annulus_inner_pix and rr_center <= annulus_outer_pix and good:
                        ann_sum += value
                        ann_sumsq += value * value
                        if ann_count < ANNULUS_MAX_PIX:
                            ann_values[ann_count] = value
                            ann_active[ann_count] = 1
                        ann_count += 1

                    if good:
                        inside = int(0)
                        for sy in range(SUBPIX):
                            for sx in range(SUBPIX):
                                sub_x = float(ix) - 0.5 + (float(sx) + 0.5) / float(SUBPIX)
                                sub_y = float(iy) - 0.5 + (float(sy) + 0.5) / float(SUBPIX)
                                ddx = sub_x - x0
                                ddy = sub_y - y0
                                rr = wp.sqrt(ddx * ddx + ddy * ddy)
                                if rr <= aperture_radius_pix:
                                    inside += 1
                        if inside > 0:
                            weight = float(inside) / float(SUBPIX * SUBPIX)
                            ap_sum += value * weight
                            # Match the current CPU implementation's variance-image aperture sum.
                            ap_var_sum += var[flat] * weight
                            ap_area += weight

        if ap_area <= 0.0 or ann_count <= 0:
            out_flux[tid] = 0.0
            out_unc[tid] = 0.0
            out_bkg[tid] = 0.0
            out_area[tid] = ap_area
            out_bad[tid] = bad_ap
            out_status[tid] = 1
            return

        usable_ann_count = wp.min(ann_count, ANNULUS_MAX_PIX)
        bkg = float(0.0)
        ann_std = float(0.0)
        active_count = int(0)
        for clip_iter in range(SIGMA_CLIP_ITERS):
            active_count = int(0)
            for i in range(ANNULUS_MAX_PIX):
                if i < usable_ann_count and ann_active[i] == 1:
                    clipped_values[active_count] = ann_values[i]
                    active_count += 1
            if active_count <= 0:
                out_flux[tid] = 0.0
                out_unc[tid] = 0.0
                out_bkg[tid] = 0.0
                out_area[tid] = ap_area
                out_bad[tid] = bad_ap
                out_status[tid] = 1
                return
            for i in range(ANNULUS_MAX_PIX):
                if i < active_count:
                    min_j = i
                    min_v = clipped_values[i]
                    for j in range(ANNULUS_MAX_PIX):
                        if j >= i and j < active_count:
                            v = clipped_values[j]
                            if v < min_v:
                                min_v = v
                                min_j = j
                    tmp = clipped_values[i]
                    clipped_values[i] = clipped_values[min_j]
                    clipped_values[min_j] = tmp
            mid = active_count / 2
            bkg = clipped_values[mid]
            if active_count > 1 and active_count - 2 * mid == 0:
                bkg = 0.5 * (clipped_values[mid - 1] + clipped_values[mid])

            mean = float(0.0)
            sumsq = float(0.0)
            for i in range(ANNULUS_MAX_PIX):
                if i < active_count:
                    v = clipped_values[i]
                    mean += v
                    sumsq += v * v
            mean = mean / float(active_count)
            ann_var = wp.max(float(0.0), sumsq / float(active_count) - mean * mean)
            ann_std = wp.sqrt(ann_var)

            changed = int(0)
            if ann_std > 0.0:
                lower = bkg - 3.0 * ann_std
                upper = bkg + 3.0 * ann_std
                for i in range(ANNULUS_MAX_PIX):
                    if i < usable_ann_count and ann_active[i] == 1:
                        v = ann_values[i]
                        if v < lower or v > upper:
                            ann_active[i] = 0
                            changed += 1
            if changed == 0:
                break

        total_var = ap_var_sum + ap_area * ap_area * ann_std * ann_std / float(active_count)
        out_flux[tid] = ap_sum - bkg * ap_area
        out_unc[tid] = wp.sqrt(wp.max(float(0.0), total_var))
        out_bkg[tid] = bkg
        out_area[tid] = ap_area
        out_bad[tid] = bad_ap
        out_status[tid] = 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a more CPU-like Warp aperture kernel: subpixel aperture weights, calibrated "
            "uJy image, bad-pixel mask, variance propagation, and sigma-clipped median annulus background. The CPU "
            "reference still uses photutils exact aperture + sigma-clipped median background."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/mnt/niroseti/spherex_cache/runs/ucs0972_no_diag_g14_16_n100_f80"),
    )
    parser.add_argument("--image-id", default=None)
    parser.add_argument("--target-counts", default="100,1000,10000,50000")
    parser.add_argument("--chunk-sizes", default="65536")
    parser.add_argument("--devices", default="cuda:0")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--compare-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1972)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--aperture-radius-pix", type=float, default=2.0)
    parser.add_argument("--annulus-inner-pix", type=float, default=4.0)
    parser.add_argument("--annulus-outer-pix", type=float, default=6.0)
    args = parser.parse_args()

    if wp is None:
        raise RuntimeError(f"warp is not available: {_WARP_IMPORT_ERROR}")
    wp.init()

    cfg = load_config()
    bad_mask_value = sum(1 << int(bit) for bit in cfg.fatal_flag_bits)
    frame = _load_frame(args.run_dir, args.image_id)
    target_counts = _parse_ints(args.target_counts)
    chunk_sizes = _parse_ints(args.chunk_sizes)
    devices = _resolve_devices(args.devices)
    output_dir = args.output_dir or args.run_dir / "benchmarks" / "warp_calibrated"
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for n_targets in target_counts:
        x, y, _source_indices = _make_targets(frame.rows, frame.flux.shape, n_targets, args.seed + n_targets)
        for device in devices:
            for chunk_size in chunk_sizes:
                for _ in range(max(0, args.warmup)):
                    _run_warp(frame, x, y, bad_mask_value, device, chunk_size, args)
                timings = []
                last = None
                for _ in range(max(1, args.repeat)):
                    t0 = time.perf_counter()
                    last = _run_warp(frame, x, y, bad_mask_value, device, chunk_size, args)
                    total_sec = time.perf_counter() - t0
                    timings.append({"total_sec": total_sec, **last["timing"]})
                correctness = _correctness_against_source_rows(
                    frame,
                    bad_mask_value,
                    device,
                    min(args.compare_count, len(frame.rows)),
                    args,
                )
                row = {
                    "n_targets": n_targets,
                    "device": device,
                    "chunk_size": chunk_size,
                    "repeat": max(1, args.repeat),
                    "chunks": last["chunks"],
                    **_summarize(timings, n_targets),
                    **correctness,
                }
                results.append(row)
                print(
                    f"device={device} n={n_targets} chunk={chunk_size} "
                    f"total_ms={row['total_ms_median']:.3f} kernel_ms={row['kernel_ms_median']:.3f} "
                    f"rate={row['targets_per_sec_total_median']:.1f}/s "
                    f"median_flux_err={row['flux_median_abs_diff_uJy']:.3f}uJy",
                    flush=True,
                )
        best = max(
            [row for row in results if row["n_targets"] == n_targets],
            key=lambda row: row["targets_per_sec_total_median"],
        )
        summaries.append(best)

    payload = {
        "run_dir": str(args.run_dir),
        "image_id": frame.image_id,
        "input_file_path": frame.input_file_path,
        "algorithm": (
            "Warp calibrated-ish aperture: 5x5 subpixel aperture weighting, center-pixel annulus, "
            "sigma-clipped median background, bad-pixel mask, variance propagation. Not yet photutils exact geometry."
        ),
        "target_counts": target_counts,
        "devices": devices,
        "chunk_sizes": chunk_sizes,
        "summaries": summaries,
        "results": results,
    }
    json_path = output_dir / "warp_calibrated_summary.json"
    csv_path = output_dir / "warp_calibrated_results.csv"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(csv_path, results)
    print(json.dumps({"summary_json": str(json_path), "results_csv": str(csv_path), "summaries": summaries}, indent=2))


def _run_warp(
    frame: Any,
    x: np.ndarray,
    y: np.ndarray,
    bad_mask_value: int,
    device: str,
    chunk_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    height, width = frame.flux.shape
    t = time.perf_counter()
    flux_dev = wp.array(np.ascontiguousarray(frame.flux.astype(np.float32).ravel()), dtype=wp.float32, device=device)
    var_dev = wp.array(
        np.ascontiguousarray(np.nan_to_num(frame.variance, nan=0.0).astype(np.float32).ravel()),
        dtype=wp.float32,
        device=device,
    )
    flags_dev = wp.array(np.ascontiguousarray(frame.flags.astype(np.uint32).ravel()), dtype=wp.uint32, device=device)
    wp.synchronize_device(device)
    image_upload_sec = time.perf_counter() - t

    outs = {
        "flux": [],
        "unc": [],
        "bkg": [],
        "area": [],
        "bad": [],
        "status": [],
    }
    target_upload_sec = 0.0
    kernel_sec = 0.0
    download_sec = 0.0
    chunks = 0
    for start in range(0, len(x), chunk_size):
        stop = min(start + chunk_size, len(x))
        xs = np.asarray(x[start:stop], dtype=np.float32)
        ys = np.asarray(y[start:stop], dtype=np.float32)
        n = len(xs)
        t = time.perf_counter()
        x_dev = wp.array(xs, dtype=wp.float32, device=device)
        y_dev = wp.array(ys, dtype=wp.float32, device=device)
        out_flux = wp.empty(n, dtype=wp.float32, device=device)
        out_unc = wp.empty(n, dtype=wp.float32, device=device)
        out_bkg = wp.empty(n, dtype=wp.float32, device=device)
        out_area = wp.empty(n, dtype=wp.float32, device=device)
        out_bad = wp.empty(n, dtype=wp.int32, device=device)
        out_status = wp.empty(n, dtype=wp.int32, device=device)
        wp.synchronize_device(device)
        target_upload_sec += time.perf_counter() - t

        t = time.perf_counter()
        wp.launch(
            _weighted_calibrated_aperture_kernel,
            dim=n,
            inputs=[
                flux_dev,
                var_dev,
                flags_dev,
                int(width),
                int(height),
                x_dev,
                y_dev,
                wp.uint32(bad_mask_value),
                float(args.aperture_radius_pix),
                float(args.annulus_inner_pix),
                float(args.annulus_outer_pix),
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
        kernel_sec += time.perf_counter() - t

        t = time.perf_counter()
        outs["flux"].append(out_flux.numpy().astype(np.float64))
        outs["unc"].append(out_unc.numpy().astype(np.float64))
        outs["bkg"].append(out_bkg.numpy().astype(np.float64))
        outs["area"].append(out_area.numpy().astype(np.float64))
        outs["bad"].append(out_bad.numpy())
        outs["status"].append(out_status.numpy())
        download_sec += time.perf_counter() - t
        chunks += 1

    return {
        "flux": np.concatenate(outs["flux"]),
        "unc": np.concatenate(outs["unc"]),
        "bkg": np.concatenate(outs["bkg"]),
        "area": np.concatenate(outs["area"]),
        "bad": np.concatenate(outs["bad"]),
        "status": np.concatenate(outs["status"]),
        "chunks": chunks,
        "timing": {
            "image_upload_sec": image_upload_sec,
            "target_upload_sec": target_upload_sec,
            "kernel_sec": kernel_sec,
            "download_sec": download_sec,
        },
    }


def _cpu_reference_from_source_rows(frame: Any, n: int) -> dict[str, np.ndarray]:
    rows = frame.rows.head(n)
    return {
        "flux": rows["aperture_flux_uJy"].to_numpy(dtype=np.float64),
        "unc": rows["aperture_flux_unc_uJy"].to_numpy(dtype=np.float64),
        "bkg": rows["background_uJy_per_pix"].to_numpy(dtype=np.float64),
        "area": rows["aperture_area_pix_exact"].to_numpy(dtype=np.float64),
        "bad": rows["n_bad_aperture_pixels_calibrated"].to_numpy(dtype=np.int32),
        "status": (rows["calibrated_aperture_status"].to_numpy() == "ok").astype(np.int32),
    }


def _correctness_against_source_rows(
    frame: Any,
    bad_mask_value: int,
    device: str,
    n: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows = frame.rows.head(n)
    x = rows["x_pix"].to_numpy(dtype=np.float64)
    y = rows["y_pix"].to_numpy(dtype=np.float64)
    cpu = _cpu_reference_from_source_rows(frame, n)
    warp_out = _run_warp(frame, x, y, bad_mask_value, device, max(n, 1), args)
    out: dict[str, Any] = {"compare_count": int(n)}
    for name in ("flux", "unc", "bkg", "area"):
        diff = warp_out[name] - cpu[name]
        out[f"{name}_median_abs_diff_uJy" if name != "area" else "area_median_abs_diff_pix"] = float(
            np.nanmedian(np.abs(diff))
        )
        out[f"{name}_max_abs_diff_uJy" if name != "area" else "area_max_abs_diff_pix"] = float(
            np.nanmax(np.abs(diff))
        )
    out["bad_pixel_count_mismatch"] = int(np.count_nonzero(warp_out["bad"] != cpu["bad"]))
    out["status_mismatch"] = int(np.count_nonzero((warp_out["status"] == 0) != (cpu["status"] == 1)))
    return out


def _summarize(timings: list[dict[str, float]], n_targets: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("image_upload_sec", "target_upload_sec", "kernel_sec", "download_sec", "total_sec"):
        vals = np.asarray([row[key] for row in timings], dtype=np.float64)
        stem = key.removesuffix("_sec")
        out[f"{stem}_ms_median"] = float(np.median(vals) * 1000.0)
        out[f"{stem}_ms_min"] = float(np.min(vals) * 1000.0)
    out["targets_per_sec_total_median"] = float(n_targets / (out["total_ms_median"] / 1000.0))
    out["targets_per_sec_kernel_median"] = float(n_targets / (out["kernel_ms_median"] / 1000.0))
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
