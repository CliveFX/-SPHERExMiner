from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spherex_laser_miner.calibration import image_to_ujy_per_pixel, load_sapm, variance_to_ujy2
from spherex_laser_miner.config import load_config
from tools.warp_photometry_compare import (
    _aperture_mean_kernel,
    cpu_center_mean_aperture,
)

try:
    import warp as wp
except Exception as exc:  # pragma: no cover - dependency check path
    wp = None
    _WARP_IMPORT_ERROR = exc
else:
    _WARP_IMPORT_ERROR = None


@dataclass(frozen=True)
class FrameData:
    rows: pd.DataFrame
    flux: np.ndarray
    variance: np.ndarray
    flags: np.ndarray
    image_id: str
    input_file_path: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scale-test approximate Warp aperture photometry. This benchmark uses the current "
            "center-pixel aperture + mean-annulus prototype, not the final photutils-equivalent math."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/mnt/niroseti/spherex_cache/runs/ucs0972_no_diag_g14_16_n100_f80"),
    )
    parser.add_argument("--image-id", default=None, help="Optional image id substring to benchmark.")
    parser.add_argument("--target-counts", default="100,1000,10000,50000")
    parser.add_argument("--chunk-sizes", default="1024,4096,16384,65536")
    parser.add_argument("--devices", default="auto", help="Comma list like cuda:0,cuda:1 or 'auto'.")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--compare-cpu-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=972)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--aperture-radius-pix", type=float, default=2.0)
    parser.add_argument("--annulus-inner-pix", type=float, default=4.0)
    parser.add_argument("--annulus-outer-pix", type=float, default=6.0)
    args = parser.parse_args()

    if wp is None:
        raise RuntimeError(f"warp is not available: {_WARP_IMPORT_ERROR}")
    wp.init()

    target_counts = _parse_ints(args.target_counts)
    chunk_sizes = _parse_ints(args.chunk_sizes)
    devices = _resolve_devices(args.devices)
    if not devices:
        raise RuntimeError("No Warp CUDA devices found. Pass --devices cuda:0 if auto-detection failed.")

    cfg = load_config()
    bad_mask_value = sum(1 << int(bit) for bit in cfg.fatal_flag_bits)
    frame = _load_frame(args.run_dir, args.image_id)
    output_dir = args.output_dir or args.run_dir / "benchmarks" / "warp_scale"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for n_targets in target_counts:
        x, y, source_indices = _make_targets(frame.rows, frame.flux.shape, n_targets, args.seed + n_targets)
        cpu_summary = _cpu_correctness_sample(
            frame,
            x,
            y,
            source_indices,
            bad_mask_value,
            args,
        )
        single_results = []
        for device in devices:
            for chunk_size in chunk_sizes:
                result = _bench_device(
                    frame,
                    x,
                    y,
                    bad_mask_value,
                    device=device,
                    chunk_size=chunk_size,
                    repeat=max(1, args.repeat),
                    warmup=max(0, args.warmup),
                    aperture_radius_pix=args.aperture_radius_pix,
                    annulus_inner_pix=args.annulus_inner_pix,
                    annulus_outer_pix=args.annulus_outer_pix,
                )
                result.update(cpu_summary)
                result["mode"] = "single_device"
                result["n_targets"] = n_targets
                all_rows.append(result)
                single_results.append(result)
                print(_one_line(result), flush=True)

        multi_result = None
        if len(devices) > 1:
            best_by_device = _best_single_by_device(single_results)
            for chunk_size in chunk_sizes:
                result = _bench_multi_device(
                    frame,
                    x,
                    y,
                    bad_mask_value,
                    devices=devices,
                    best_by_device=best_by_device,
                    chunk_size=chunk_size,
                    repeat=max(1, args.repeat),
                    warmup=max(0, args.warmup),
                    aperture_radius_pix=args.aperture_radius_pix,
                    annulus_inner_pix=args.annulus_inner_pix,
                    annulus_outer_pix=args.annulus_outer_pix,
                )
                result.update(cpu_summary)
                result["mode"] = "multi_device"
                result["n_targets"] = n_targets
                all_rows.append(result)
                print(_one_line(result), flush=True)
                if multi_result is None or result["targets_per_sec_total_median"] > multi_result["targets_per_sec_total_median"]:
                    multi_result = result

        best_single = max(single_results, key=lambda row: row["targets_per_sec_total_median"])
        best_overall = best_single
        if multi_result is not None and multi_result["targets_per_sec_total_median"] > best_overall["targets_per_sec_total_median"]:
            best_overall = multi_result
        summaries.append(
            {
                "n_targets": n_targets,
                "best_mode": best_overall["mode"],
                "best_devices": best_overall["devices"],
                "best_chunk_size": best_overall["chunk_size"],
                "best_targets_per_sec_total_median": best_overall["targets_per_sec_total_median"],
                "best_total_ms_median": best_overall["total_ms_median"],
                "cpu_compare_count": cpu_summary["cpu_compare_count"],
                "cpu_compare_ms": cpu_summary["cpu_compare_ms"],
                "approx_cpu_vs_main_max_abs_flux_diff_uJy": best_overall.get(
                    "approx_cpu_vs_main_max_abs_flux_diff_uJy"
                ),
                "approx_cpu_vs_main_median_abs_flux_diff_uJy": best_overall.get(
                    "approx_cpu_vs_main_median_abs_flux_diff_uJy"
                ),
            }
        )

    payload = {
        "run_dir": str(args.run_dir),
        "image_id": frame.image_id,
        "input_file_path": frame.input_file_path,
        "algorithm": "approximate center-pixel aperture with mean annulus background; not final calibrated photutils-equivalent math",
        "devices": devices,
        "target_counts": target_counts,
        "chunk_sizes": chunk_sizes,
        "repeat": max(1, args.repeat),
        "warmup": max(0, args.warmup),
        "summaries": summaries,
        "results": all_rows,
    }
    json_path = output_dir / "warp_scale_summary.json"
    csv_path = output_dir / "warp_scale_results.csv"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(csv_path, all_rows)
    print(json.dumps({"summary_json": str(json_path), "results_csv": str(csv_path), "summaries": summaries}, indent=2))


def _bench_device(
    frame: FrameData,
    x: np.ndarray,
    y: np.ndarray,
    bad_mask_value: int,
    *,
    device: str,
    chunk_size: int,
    repeat: int,
    warmup: int,
    aperture_radius_pix: float,
    annulus_inner_pix: float,
    annulus_outer_pix: float,
) -> dict[str, Any]:
    for _ in range(warmup):
        _bench_device_once(
            frame,
            x,
            y,
            bad_mask_value,
            device,
            chunk_size,
            aperture_radius_pix,
            annulus_inner_pix,
            annulus_outer_pix,
        )
    timings = []
    last = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        state = _upload_image(frame, device)
        image_upload_sec = time.perf_counter() - t0
        chunk_stats = _run_chunks(
            state,
            x,
            y,
            bad_mask_value,
            device,
            chunk_size,
            aperture_radius_pix,
            annulus_inner_pix,
            annulus_outer_pix,
        )
        total_sec = time.perf_counter() - t0
        timings.append(
            {
                "image_upload_sec": image_upload_sec,
                **chunk_stats,
                "total_sec": total_sec,
            }
        )
        last = chunk_stats
    return _summarize_timings(
        timings,
        n_targets=len(x),
        devices=device,
        chunk_size=chunk_size,
        chunks=last["chunks"] if last else 0,
    )


def _bench_multi_device(
    frame: FrameData,
    x: np.ndarray,
    y: np.ndarray,
    bad_mask_value: int,
    *,
    devices: list[str],
    best_by_device: dict[str, dict[str, Any]],
    chunk_size: int,
    repeat: int,
    warmup: int,
    aperture_radius_pix: float,
    annulus_inner_pix: float,
    annulus_outer_pix: float,
) -> dict[str, Any]:
    splits = _split_by_rates(len(x), devices, best_by_device)
    for _ in range(warmup):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = []
            for device, start, stop in splits:
                futures.append(
                    pool.submit(
                        _bench_device_once,
                        frame,
                        x[start:stop],
                        y[start:stop],
                        bad_mask_value,
                        device,
                        chunk_size,
                        aperture_radius_pix,
                        annulus_inner_pix,
                        annulus_outer_pix,
                    )
                )
            for future in futures:
                future.result()
    timings = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = []
            for device, start, stop in splits:
                futures.append(
                    pool.submit(
                        _bench_device_once,
                        frame,
                        x[start:stop],
                        y[start:stop],
                        bad_mask_value,
                        device,
                        chunk_size,
                        aperture_radius_pix,
                        annulus_inner_pix,
                        annulus_outer_pix,
                    )
                )
            parts = [future.result() for future in futures]
        total_sec = time.perf_counter() - t0
        timings.append(
            {
                "image_upload_sec": max(part["image_upload_sec"] for part in parts),
                "target_upload_sec": max(part["target_upload_sec"] for part in parts),
                "kernel_sec": max(part["kernel_sec"] for part in parts),
                "download_sec": max(part["download_sec"] for part in parts),
                "total_sec": total_sec,
            }
        )
    out = _summarize_timings(
        timings,
        n_targets=len(x),
        devices=",".join(devices),
        chunk_size=chunk_size,
        chunks=int(sum(math.ceil((stop - start) / chunk_size) for _, start, stop in splits)),
    )
    out["device_splits"] = [
        {"device": device, "start": start, "stop": stop, "n_targets": stop - start}
        for device, start, stop in splits
    ]
    return out


def _bench_device_once(
    frame: FrameData,
    x: np.ndarray,
    y: np.ndarray,
    bad_mask_value: int,
    device: str,
    chunk_size: int,
    aperture_radius_pix: float,
    annulus_inner_pix: float,
    annulus_outer_pix: float,
) -> dict[str, float]:
    t0 = time.perf_counter()
    state = _upload_image(frame, device)
    image_upload_sec = time.perf_counter() - t0
    chunk_stats = _run_chunks(
        state,
        x,
        y,
        bad_mask_value,
        device,
        chunk_size,
        aperture_radius_pix,
        annulus_inner_pix,
        annulus_outer_pix,
    )
    return {"image_upload_sec": image_upload_sec, **chunk_stats}


def _run_chunks(
    state: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    bad_mask_value: int,
    device: str,
    chunk_size: int,
    aperture_radius_pix: float,
    annulus_inner_pix: float,
    annulus_outer_pix: float,
) -> dict[str, Any]:
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
            _aperture_mean_kernel,
            dim=n,
            inputs=[
                state["flux_dev"],
                state["var_dev"],
                state["flags_dev"],
                state["width"],
                state["height"],
                x_dev,
                y_dev,
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
        kernel_sec += time.perf_counter() - t

        t = time.perf_counter()
        _ = out_flux.numpy()
        _ = out_unc.numpy()
        _ = out_status.numpy()
        download_sec += time.perf_counter() - t
        chunks += 1
    return {
        "target_upload_sec": target_upload_sec,
        "kernel_sec": kernel_sec,
        "download_sec": download_sec,
        "chunks": chunks,
    }


def _upload_image(frame: FrameData, device: str) -> dict[str, Any]:
    height, width = frame.flux.shape
    flux_flat = np.ascontiguousarray(frame.flux.astype(np.float32).ravel())
    var_flat = np.ascontiguousarray(np.nan_to_num(frame.variance, nan=0.0).astype(np.float32).ravel())
    flags_flat = np.ascontiguousarray(frame.flags.astype(np.uint32).ravel())
    return {
        "flux_dev": wp.array(flux_flat, dtype=wp.float32, device=device),
        "var_dev": wp.array(var_flat, dtype=wp.float32, device=device),
        "flags_dev": wp.array(flags_flat, dtype=wp.uint32, device=device),
        "width": int(width),
        "height": int(height),
    }


def _cpu_correctness_sample(
    frame: FrameData,
    x: np.ndarray,
    y: np.ndarray,
    source_indices: np.ndarray,
    bad_mask_value: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    n = min(len(x), max(0, int(args.compare_cpu_count)))
    if n <= 0:
        return {"cpu_compare_count": 0, "cpu_compare_ms": 0.0}
    sample = np.linspace(0, len(x) - 1, n, dtype=np.int64)
    t0 = time.perf_counter()
    cpu = cpu_center_mean_aperture(
        frame.flux,
        frame.variance,
        frame.flags,
        x[sample],
        y[sample],
        bad_mask_value,
        args.aperture_radius_pix,
        args.annulus_inner_pix,
        args.annulus_outer_pix,
    )
    cpu_ms = (time.perf_counter() - t0) * 1000.0
    source_flux = frame.rows.iloc[source_indices[sample]]["aperture_flux_uJy"].to_numpy(dtype=np.float64)
    diff = cpu["flux_uJy"] - source_flux
    return {
        "cpu_compare_count": int(n),
        "cpu_compare_ms": float(cpu_ms),
        "approx_cpu_vs_main_max_abs_flux_diff_uJy": float(np.nanmax(np.abs(diff))),
        "approx_cpu_vs_main_median_abs_flux_diff_uJy": float(np.nanmedian(np.abs(diff))),
        "status_bad_count": int(np.count_nonzero(cpu["status"] != 0)),
    }


def _summarize_timings(
    timings: list[dict[str, float]],
    *,
    n_targets: int,
    devices: str,
    chunk_size: int,
    chunks: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "devices": devices,
        "chunk_size": int(chunk_size),
        "chunks": int(chunks),
    }
    for key in ("image_upload_sec", "target_upload_sec", "kernel_sec", "download_sec", "total_sec"):
        vals = np.asarray([row[key] for row in timings], dtype=np.float64)
        out[f"{key.removesuffix('_sec')}_ms_median"] = float(np.median(vals) * 1000.0)
        out[f"{key.removesuffix('_sec')}_ms_min"] = float(np.min(vals) * 1000.0)
    total = out["total_ms_median"] / 1000.0
    kernel = out["kernel_ms_median"] / 1000.0
    out["targets_per_sec_total_median"] = float(n_targets / total) if total > 0 else None
    out["targets_per_sec_kernel_median"] = float(n_targets / kernel) if kernel > 0 else None
    return out


def _load_frame(run_dir: Path, image_id: str | None) -> FrameData:
    shard_root = run_dir / "field_shards"
    paths = sorted(shard_root.glob("image_id=*/measurements.parquet"))
    if image_id is not None:
        paths = [p for p in paths if p.parent.name == f"image_id={image_id}" or image_id in p.parent.name]
    if not paths:
        raise FileNotFoundError(f"No measurement shards found under {shard_root}")
    best_rows = None
    best_path = None
    for path in paths:
        rows = pd.read_parquet(path)
        rows = rows[np.isfinite(pd.to_numeric(rows["x_pix"], errors="coerce"))].copy()
        if best_rows is None or len(rows) > len(best_rows):
            best_rows = rows
            best_path = path
    if best_rows is None or best_path is None or best_rows.empty:
        raise RuntimeError(f"No usable target rows found under {run_dir}")
    rows = best_rows.reset_index(drop=True)
    fits_path = Path(str(rows["input_file_path"].iloc[0]))
    detector = int(rows["detector"].iloc[0])
    cfg = load_config()
    with fits.open(fits_path, memmap=True) as hdul:
        image_hdu = hdul["IMAGE"]
        sapm_data, sapm_header, _ = load_sapm(cfg.cache_root, cfg.release, detector)
        flux = image_to_ujy_per_pixel(image_hdu.data, image_hdu.header, sapm_data, sapm_header)
        if "VARIANCE" in hdul:
            variance = variance_to_ujy2(hdul["VARIANCE"].data, image_hdu.header, sapm_data, sapm_header)
        else:
            variance = np.zeros_like(flux, dtype=np.float32)
        if "FLAGS" in hdul:
            flags = np.asarray(hdul["FLAGS"].data, dtype=np.uint32)
        else:
            flags = np.zeros_like(flux, dtype=np.uint32)
    return FrameData(
        rows=rows,
        flux=np.asarray(flux),
        variance=np.asarray(variance),
        flags=flags,
        image_id=str(rows["image_id"].iloc[0]),
        input_file_path=str(fits_path),
    )


def _make_targets(rows: pd.DataFrame, shape: tuple[int, int], n_targets: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base_x = rows["x_pix"].to_numpy(dtype=np.float64)
    base_y = rows["y_pix"].to_numpy(dtype=np.float64)
    source_indices = np.arange(n_targets, dtype=np.int64) % len(rows)
    repeats = source_indices // len(rows)
    # Jitter repeated targets so 50k measurements do real nearby pixel work instead of exact duplicates.
    jitter_x = rng.uniform(-0.35, 0.35, size=n_targets) + (repeats % 7) * 0.011
    jitter_y = rng.uniform(-0.35, 0.35, size=n_targets) + (repeats % 11) * 0.009
    height, width = shape
    x = np.clip(base_x[source_indices] + jitter_x, 6.0, width - 7.0)
    y = np.clip(base_y[source_indices] + jitter_y, 6.0, height - 7.0)
    return x, y, source_indices


def _resolve_devices(value: str) -> list[str]:
    if value != "auto":
        return [part.strip() for part in value.split(",") if part.strip()]
    if wp is None:
        return []
    devices = []
    for device in wp.get_devices():
        name = str(device)
        if "cuda" in name.lower():
            devices.append(device.alias if hasattr(device, "alias") else name)
    if not devices:
        # Warp device string support is stable even when get_devices repr changes.
        for i in range(8):
            try:
                wp.get_device(f"cuda:{i}")
            except Exception:
                continue
            devices.append(f"cuda:{i}")
    return devices


def _best_single_by_device(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        device = str(row["devices"])
        if device not in best or row["targets_per_sec_total_median"] > best[device]["targets_per_sec_total_median"]:
            best[device] = row
    return best


def _split_by_rates(n_targets: int, devices: list[str], best_by_device: dict[str, dict[str, Any]]) -> list[tuple[str, int, int]]:
    rates = np.asarray(
        [float(best_by_device.get(device, {}).get("targets_per_sec_total_median") or 1.0) for device in devices],
        dtype=np.float64,
    )
    weights = rates / np.sum(rates)
    counts = np.floor(weights * n_targets).astype(np.int64)
    while int(np.sum(counts)) < n_targets:
        counts[int(np.argmax(weights * n_targets - counts))] += 1
    splits = []
    start = 0
    for device, count in zip(devices, counts, strict=True):
        stop = start + int(count)
        splits.append((device, start, stop))
        start = stop
    return splits


def _parse_ints(value: str) -> list[int]:
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out or any(item <= 0 for item in out):
        raise ValueError(f"Expected positive comma-separated integers, got {value!r}")
    return out


def _one_line(row: dict[str, Any]) -> str:
    return (
        f"{row['mode']} devices={row['devices']} n={row['n_targets']} chunk={row['chunk_size']} "
        f"total_ms={row['total_ms_median']:.3f} kernel_ms={row['kernel_ms_median']:.3f} "
        f"rate={row['targets_per_sec_total_median']:.1f}/s"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys() if key != "device_splits"})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


if __name__ == "__main__":
    main()
