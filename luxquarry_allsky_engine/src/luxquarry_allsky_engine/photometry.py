from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits

from .calibration import (
    SPECTRAL_WCS_COLLECTION,
    bilinear_sample,
    image_to_ujy_per_pixel,
    load_sapm,
    load_spectral_wcs_maps,
    variance_to_ujy2,
)


DEFAULT_FATAL_FLAG_BITS = (0, 1, 2, 4, 6, 7, 9, 10, 11, 14, 15, 17, 19, 22, 24, 26, 27, 28, 29)


@dataclass(frozen=True)
class ApertureConfig:
    cache_root: Path
    aperture_radius_pix: float = 2.0
    annulus_inner_pix: float = 4.0
    annulus_outer_pix: float = 6.0
    edge_margin_pix: float = 6.0
    fatal_flag_bits: tuple[int, ...] = DEFAULT_FATAL_FLAG_BITS


def run_cpu_aperture(
    *,
    manifest_path: Path,
    projected_targets_path: Path,
    output_path: Path,
    config: ApertureConfig,
    limit_frames: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    manifest = pd.read_parquet(manifest_path)
    targets = pd.read_parquet(projected_targets_path)
    if limit_frames is not None:
        frame_ids = set(manifest.head(limit_frames)["frame_group_id"].astype(str))
        manifest = manifest[manifest["frame_group_id"].astype(str).isin(frame_ids)].copy()
        targets = targets[targets["frame_group_id"].astype(str).isin(frame_ids)].copy()

    rows: list[pd.DataFrame] = []
    frame_timings: list[dict[str, Any]] = []
    for frame in manifest.to_dict(orient="records"):
        frame_group_id = str(frame.get("frame_group_id"))
        frame_targets = targets[
            targets["frame_group_id"].astype(str).eq(frame_group_id) & targets["in_frame"].astype(bool)
        ].copy()
        if frame_targets.empty:
            continue
        t0 = time.perf_counter()
        measured = _measure_one_frame(frame, frame_targets, config)
        frame_timings.append(
            {
                "frame_group_id": frame_group_id,
                "image_id": frame.get("image_id"),
                "input_target_count": int(len(frame_targets)),
                "measurement_count": int(len(measured)),
                "ok_count": int((measured["aperture_status"] == "ok").sum()),
                "wall_time_sec": time.perf_counter() - t0,
            }
        )
        rows.append(measured)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=_measurement_columns())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    out.to_parquet(output_path, index=False)
    write_wall = time.perf_counter() - t0
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "projected_targets_path": str(projected_targets_path),
        "output_path": str(output_path),
        "frame_count": int(len(manifest)),
        "input_projected_rows": int(len(targets)),
        "measurement_rows": int(len(out)),
        "ok_measurement_rows": int((out["aperture_status"] == "ok").sum()) if len(out) else 0,
        "total_wall_sec": time.perf_counter() - started,
        "write_wall_sec": write_wall,
        "frame_timings": frame_timings,
        "aperture_radius_pix": config.aperture_radius_pix,
        "annulus_inner_pix": config.annulus_inner_pix,
        "annulus_outer_pix": config.annulus_outer_pix,
        "edge_margin_pix": config.edge_margin_pix,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_profile(output_path.with_suffix(".profile.json"), summary)
    return summary


def _measure_one_frame(frame: dict[str, Any], targets: pd.DataFrame, config: ApertureConfig) -> pd.DataFrame:
    path = Path(str(frame["path"]))
    detector = int(frame["detector"])
    release = str(frame.get("release") or "qr2")
    with fits.open(path, memmap=True) as hdul:
        image_hdu = hdul["IMAGE"]
        image = np.asarray(image_hdu.data, dtype=float)
        variance = np.asarray(hdul["VARIANCE"].data, dtype=float) if "VARIANCE" in hdul else None
        flags = np.asarray(hdul["FLAGS"].data, dtype=np.uint32) if "FLAGS" in hdul else None
        sapm_data, sapm_header, sapm_path = load_sapm(str(config.cache_root), release, detector)
        cwave_map, cband_map, wavelength_path = load_spectral_wcs_maps(str(config.cache_root), release, detector)
        flux_ujy = image_to_ujy_per_pixel(image, image_hdu.header, sapm_data, sapm_header)
        var_ujy2 = variance_to_ujy2(variance, image_hdu.header, sapm_data, sapm_header) if variance is not None else None

    x_zero = targets["x_pix"].to_numpy(dtype=float) - 1.0
    y_zero = targets["y_pix"].to_numpy(dtype=float) - 1.0
    cwave_um = bilinear_sample(cwave_map, x_zero, y_zero)
    cband_um = bilinear_sample(cband_map, x_zero, y_zero)
    edge_zero = np.minimum.reduce([x_zero, y_zero, flux_ujy.shape[1] - 1 - x_zero, flux_ujy.shape[0] - 1 - y_zero])

    measured_rows: list[dict[str, Any]] = []
    base = targets.reset_index(drop=True)
    for i, target in base.iterrows():
        if not np.isfinite(edge_zero[i]) or edge_zero[i] < config.edge_margin_pix:
            measurement = _bad_measurement("outside_edge_margin")
        else:
            measurement = _aperture_one(
                flux_ujy=flux_ujy,
                var_ujy2=var_ujy2,
                flags=flags,
                x=float(x_zero[i]),
                y=float(y_zero[i]),
                config=config,
            )
        measured_rows.append(
            {
                "frame_group_id": target["frame_group_id"],
                "image_id": target["image_id"],
                "fits_path": str(path),
                "catalog": target["catalog"],
                "target_id": target["target_id"],
                "source_id": target["source_id"],
                "ra_deg": float(target["ra_deg"]),
                "dec_deg": float(target["dec_deg"]),
                "mag_primary": float(target["mag_primary"]) if pd.notna(target["mag_primary"]) else np.nan,
                "mag_primary_band": target["mag_primary_band"],
                "x_pix": float(target["x_pix"]),
                "y_pix": float(target["y_pix"]),
                "edge_distance_pix": float(edge_zero[i]),
                "detector": detector,
                "release": release,
                "cwave_um": float(cwave_um[i]) if np.isfinite(cwave_um[i]) else np.nan,
                "cband_um": float(cband_um[i]) if np.isfinite(cband_um[i]) else np.nan,
                "wavelength_source": "spectral_wcs_CWAVE_CBAND",
                "wavelength_calibration_file": wavelength_path,
                "wavelength_calibration_collection": SPECTRAL_WCS_COLLECTION,
                "sapm_file": sapm_path,
                **measurement,
            }
        )
    return pd.DataFrame(measured_rows, columns=_measurement_columns())


def _aperture_one(
    *,
    flux_ujy: np.ndarray,
    var_ujy2: np.ndarray | None,
    flags: np.ndarray | None,
    x: float,
    y: float,
    config: ApertureConfig,
) -> dict[str, Any]:
    if not np.isfinite(x) or not np.isfinite(y):
        return _bad_measurement("bad_position")
    y0, y1, x0, x1, ap_weight, ann_mask = _aperture_cutout_masks(
        shape=flux_ujy.shape,
        x=x,
        y=y,
        aperture_radius=config.aperture_radius_pix,
        annulus_inner=config.annulus_inner_pix,
        annulus_outer=config.annulus_outer_pix,
    )
    flux_cut = flux_ujy[y0:y1, x0:x1]
    var_cut = var_ujy2[y0:y1, x0:x1] if var_ujy2 is not None else None
    flags_cut = flags[y0:y1, x0:x1] if flags is not None else None
    bad_mask = _bad_pixel_mask(flags_cut, flux_cut, config.fatal_flag_bits)
    valid_weight = np.where(bad_mask, 0.0, ap_weight)
    ap_area = float(np.nansum(valid_weight))
    if ap_area <= 0 or not np.isfinite(ap_area):
        return _bad_measurement("no_good_aperture_area")
    ann_values = flux_cut[ann_mask & ~bad_mask & np.isfinite(flux_cut)]
    if ann_values.size == 0:
        return _bad_measurement("bad_background")
    clipped = _sigma_clip(ann_values)
    if clipped.size == 0:
        return _bad_measurement("bad_background")
    bkg = float(np.nanmedian(clipped))
    bkg_std = float(np.nanstd(clipped))
    flux = float(np.nansum(flux_cut * valid_weight) - bkg * ap_area)
    var_sum = 0.0
    if var_cut is not None:
        var_sum = float(np.nansum(np.where(np.isfinite(var_cut), var_cut, 0.0) * valid_weight))
    elif bkg_std > 0:
        var_sum = bkg_std**2 * ap_area
    var_bkg = bkg_std**2 / float(clipped.size) if clipped.size else 0.0
    unc = math.sqrt(max(0.0, var_sum + ap_area**2 * var_bkg))
    flags_summary = _flags_summary(flags_cut, ap_weight > 0)
    fatal_mask = _fatal_mask(config.fatal_flag_bits)
    n_bad = int(np.count_nonzero(((flags_cut & fatal_mask) != 0) & (ap_weight > 0))) if flags_cut is not None else 0
    return {
        "aperture_flux_uJy": flux,
        "aperture_flux_unc_uJy": unc,
        "aperture_flux_unit": "uJy",
        "aperture_area_pix": ap_area,
        "background_uJy_per_pix": bkg,
        "background_unc_uJy_per_pix": bkg_std,
        "n_bad_aperture_pixels": n_bad,
        "flags_summary": flags_summary,
        "aperture_status": "ok",
    }


def _aperture_cutout_masks(
    *,
    shape: tuple[int, int],
    x: float,
    y: float,
    aperture_radius: float,
    annulus_inner: float,
    annulus_outer: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    margin = int(math.ceil(max(aperture_radius, annulus_outer) + 1.0))
    x0 = max(0, int(math.floor(x)) - margin)
    x1 = min(width, int(math.floor(x)) + margin + 1)
    y0 = max(0, int(math.floor(y)) - margin)
    y1 = min(height, int(math.floor(y)) + margin + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - x, yy - y)
    ann_mask = (rr > annulus_inner) & (rr <= annulus_outer)

    subpix = 5
    weight = np.zeros(rr.shape, dtype=float)
    offsets = (np.arange(subpix, dtype=float) + 0.5) / subpix - 0.5
    for dy in offsets:
        for dx in offsets:
            sub_rr = np.hypot((xx + dx) - x, (yy + dy) - y)
            weight += sub_rr <= aperture_radius
    return y0, y1, x0, x1, weight / float(subpix * subpix), ann_mask


def _sigma_clip(values: np.ndarray, *, sigma: float = 3.0, maxiters: int = 5) -> np.ndarray:
    clipped = np.asarray(values, dtype=float)
    clipped = clipped[np.isfinite(clipped)]
    for _ in range(maxiters):
        if clipped.size < 2:
            return clipped
        med = np.nanmedian(clipped)
        std = np.nanstd(clipped)
        if not np.isfinite(std) or std <= 0:
            return clipped
        keep = np.abs(clipped - med) <= sigma * std
        if int(keep.sum()) == clipped.size:
            return clipped
        clipped = clipped[keep]
    return clipped


def _bad_pixel_mask(flags: np.ndarray | None, flux: np.ndarray, fatal_flag_bits: tuple[int, ...]) -> np.ndarray:
    bad = ~np.isfinite(flux)
    if flags is not None:
        bad |= (flags & _fatal_mask(fatal_flag_bits)) != 0
    return bad


def _fatal_mask(bits: tuple[int, ...]) -> int:
    value = 0
    for bit in bits:
        value |= 1 << int(bit)
    return value


def _flags_summary(flags: np.ndarray | None, mask: np.ndarray) -> int:
    if flags is None:
        return 0
    summary = 0
    for value in flags[mask]:
        summary |= int(value)
    return int(summary)


def _bad_measurement(status: str) -> dict[str, Any]:
    return {
        "aperture_flux_uJy": np.nan,
        "aperture_flux_unc_uJy": np.nan,
        "aperture_flux_unit": "uJy",
        "aperture_area_pix": np.nan,
        "background_uJy_per_pix": np.nan,
        "background_unc_uJy_per_pix": np.nan,
        "n_bad_aperture_pixels": 0,
        "flags_summary": 0,
        "aperture_status": status,
    }


def _measurement_columns() -> list[str]:
    return [
        "frame_group_id",
        "image_id",
        "fits_path",
        "catalog",
        "target_id",
        "source_id",
        "ra_deg",
        "dec_deg",
        "mag_primary",
        "mag_primary_band",
        "x_pix",
        "y_pix",
        "edge_distance_pix",
        "detector",
        "release",
        "cwave_um",
        "cband_um",
        "wavelength_source",
        "wavelength_calibration_file",
        "wavelength_calibration_collection",
        "sapm_file",
        "aperture_flux_uJy",
        "aperture_flux_unc_uJy",
        "aperture_flux_unit",
        "aperture_area_pix",
        "background_uJy_per_pix",
        "background_unc_uJy_per_pix",
        "n_bad_aperture_pixels",
        "flags_summary",
        "aperture_status",
    ]


def _write_profile(path: Path, summary: dict[str, Any]) -> None:
    total = max(float(summary.get("total_wall_sec") or 0.0), 1e-12)
    phot_wall = sum(float(row["wall_time_sec"]) for row in summary.get("frame_timings", []))
    write_wall = float(summary.get("write_wall_sec") or 0.0)
    profile = {
        "created_utc": summary.get("created_utc"),
        "rows": [
            {
                "stage": "cpu_aperture_photometry",
                "function_or_script": "luxquarry_allsky_engine.photometry.run_cpu_aperture",
                "wall_time_sec": phot_wall,
                "wall_time_pct": 100.0 * phot_wall / total,
                "backend": "numpy_cpu_frame_batch",
                "rows_out": summary.get("measurement_rows"),
            },
            {
                "stage": "write_measurements",
                "function_or_script": "luxquarry_allsky_engine.photometry.run_cpu_aperture",
                "wall_time_sec": write_wall,
                "wall_time_pct": 100.0 * write_wall / total,
                "backend": "pandas_pyarrow",
                "rows_out": summary.get("measurement_rows"),
            },
        ],
    }
    path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
