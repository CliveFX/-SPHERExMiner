from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from spherex_laser_miner.photometry.calibrated_aperture import CalibratedApertureMeasurement

try:
    import warp as wp
except Exception as exc:  # pragma: no cover - optional GPU dependency
    wp = None
    _WARP_IMPORT_ERROR = exc
else:
    _WARP_IMPORT_ERROR = None


MAX_RADIUS_PIX = 6
SUBPIX = 5
ANNULUS_MAX_PIX = 128
SIGMA_CLIP_ITERS = 5


@dataclass(frozen=True)
class WarpBatchResult:
    measurements: list[CalibratedApertureMeasurement]
    device: str


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
        out_bkg_std: wp.array(dtype=wp.float32),
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
                            ap_var_sum += var[flat] * weight
                            ap_area += weight

        if ap_area <= 0.0 or ann_count <= 0:
            out_flux[tid] = 0.0
            out_unc[tid] = 0.0
            out_bkg[tid] = 0.0
            out_bkg_std[tid] = 0.0
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
                out_bkg_std[tid] = 0.0
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
        out_bkg_std[tid] = ann_std
        out_area[tid] = ap_area
        out_bad[tid] = bad_ap
        out_status[tid] = 0


def warp_calibrated_aperture_batch(
    *,
    flux_ujy: np.ndarray,
    var_ujy2: np.ndarray | None,
    flags: np.ndarray | None,
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    aperture_radius_pix: float,
    annulus_inner_pix: float,
    annulus_outer_pix: float,
    fatal_flag_bits: Iterable[int],
    devices: tuple[str, ...],
    worker_name: str,
) -> WarpBatchResult:
    if wp is None:
        raise RuntimeError(f"Warp is not available: {_WARP_IMPORT_ERROR}")
    wp.init()
    x = np.asarray(x_pix, dtype=np.float32)
    y = np.asarray(y_pix, dtype=np.float32)
    if len(x) == 0:
        return WarpBatchResult(measurements=[], device=_select_device(devices, worker_name))
    if var_ujy2 is None:
        var_ujy2 = np.zeros_like(flux_ujy, dtype=np.float32)
    if flags is None:
        flags = np.zeros_like(flux_ujy, dtype=np.uint32)
    device = _select_device(devices, worker_name)
    bad_mask_value = sum(1 << int(bit) for bit in fatal_flag_bits)
    height, width = flux_ujy.shape
    flux_dev = wp.array(np.ascontiguousarray(flux_ujy.astype(np.float32).ravel()), dtype=wp.float32, device=device)
    var_dev = wp.array(
        np.ascontiguousarray(np.nan_to_num(var_ujy2, nan=0.0).astype(np.float32).ravel()),
        dtype=wp.float32,
        device=device,
    )
    flags_dev = wp.array(np.ascontiguousarray(np.asarray(flags, dtype=np.uint32).ravel()), dtype=wp.uint32, device=device)
    x_dev = wp.array(x, dtype=wp.float32, device=device)
    y_dev = wp.array(y, dtype=wp.float32, device=device)
    n = len(x)
    out_flux = wp.empty(n, dtype=wp.float32, device=device)
    out_unc = wp.empty(n, dtype=wp.float32, device=device)
    out_bkg = wp.empty(n, dtype=wp.float32, device=device)
    out_bkg_std = wp.empty(n, dtype=wp.float32, device=device)
    out_area = wp.empty(n, dtype=wp.float32, device=device)
    out_bad = wp.empty(n, dtype=wp.int32, device=device)
    out_status = wp.empty(n, dtype=wp.int32, device=device)
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
            float(aperture_radius_pix),
            float(annulus_inner_pix),
            float(annulus_outer_pix),
            out_flux,
            out_unc,
            out_bkg,
            out_bkg_std,
            out_area,
            out_bad,
            out_status,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    flux = out_flux.numpy().astype(float)
    unc = out_unc.numpy().astype(float)
    bkg = out_bkg.numpy().astype(float)
    bkg_std = out_bkg_std.numpy().astype(float)
    area = out_area.numpy().astype(float)
    bad = out_bad.numpy().astype(int)
    status = out_status.numpy().astype(int)
    measurements = [
        CalibratedApertureMeasurement(
            aperture_flux_uJy=float(flux[i]) if status[i] == 0 else float("nan"),
            aperture_flux_unc_uJy=float(unc[i]) if status[i] == 0 else float("nan"),
            aperture_flux_unit_calibrated="uJy",
            aperture_area_pix_exact=float(area[i]) if status[i] == 0 else float("nan"),
            background_uJy_per_pix=float(bkg[i]) if status[i] == 0 else float("nan"),
            background_unc_uJy_per_pix=float(bkg_std[i]) if status[i] == 0 else float("nan"),
            n_bad_aperture_pixels_calibrated=int(bad[i]),
            calibrated_aperture_status="ok" if status[i] == 0 else "bad_background",
        )
        for i in range(n)
    ]
    return WarpBatchResult(measurements=measurements, device=device)


def _select_device(devices: tuple[str, ...], worker_name: str) -> str:
    if not devices:
        return "cuda:0"
    match = re.search(r"_(\d+)$", worker_name)
    if match is None:
        return devices[0]
    return devices[int(match.group(1)) % len(devices)]
