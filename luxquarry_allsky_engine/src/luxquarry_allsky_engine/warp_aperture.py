from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

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
class WarpApertureBatch:
    flux_uJy: np.ndarray
    flux_unc_uJy: np.ndarray
    aperture_area_pix: np.ndarray
    background_uJy_per_pix: np.ndarray
    background_unc_uJy_per_pix: np.ndarray
    n_bad_aperture_pixels: np.ndarray
    flags_summary: np.ndarray
    status: np.ndarray
    device: str


@dataclass(frozen=True)
class WarpFrameApertureBatch:
    flux_uJy: np.ndarray
    flux_unc_uJy: np.ndarray
    aperture_area_pix: np.ndarray
    background_uJy_per_pix: np.ndarray
    background_unc_uJy_per_pix: np.ndarray
    n_bad_aperture_pixels: np.ndarray
    flags_summary: np.ndarray
    cwave_um: np.ndarray
    cband_um: np.ndarray
    status: np.ndarray
    device: str


@dataclass(frozen=True)
class WarpFrameApertureDeviceBatch:
    columns: dict[str, object]
    status: object
    device: str


@dataclass(frozen=True)
class WarpFrameCalibrationDevice:
    sapm: object
    cwave: object
    cband: object
    width: int
    height: int
    device: str


if wp is not None:

    @wp.kernel
    def _aperture_kernel(
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
        out_area: wp.array(dtype=wp.float32),
        out_bkg: wp.array(dtype=wp.float32),
        out_bkg_std: wp.array(dtype=wp.float32),
        out_bad: wp.array(dtype=wp.int32),
        out_flags_summary: wp.array(dtype=wp.uint32),
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
        flags_summary = wp.uint32(0)
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

                    if rr_center <= aperture_radius_pix:
                        flags_summary = flags_summary | flags[flat]
                        if not good:
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
            out_area[tid] = ap_area
            out_bkg[tid] = 0.0
            out_bkg_std[tid] = 0.0
            out_bad[tid] = bad_ap
            out_flags_summary[tid] = flags_summary
            out_status[tid] = 1
            return

        usable_ann_count = wp.min(ann_count, ANNULUS_MAX_PIX)
        bkg = float(0.0)
        ann_std = float(0.0)
        active_count = int(0)
        for _clip_iter in range(SIGMA_CLIP_ITERS):
            active_count = int(0)
            for i in range(ANNULUS_MAX_PIX):
                if i < usable_ann_count and ann_active[i] == 1:
                    clipped_values[active_count] = ann_values[i]
                    active_count += 1
            if active_count <= 0:
                out_flux[tid] = 0.0
                out_unc[tid] = 0.0
                out_area[tid] = ap_area
                out_bkg[tid] = 0.0
                out_bkg_std[tid] = 0.0
                out_bad[tid] = bad_ap
                out_flags_summary[tid] = flags_summary
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
        out_area[tid] = ap_area
        out_bkg[tid] = bkg
        out_bkg_std[tid] = ann_std
        out_bad[tid] = bad_ap
        out_flags_summary[tid] = flags_summary
        out_status[tid] = 0


    @wp.kernel
    def _frame_aperture_kernel(
        image: wp.array(dtype=wp.float32),
        variance: wp.array(dtype=wp.float32),
        flags: wp.array(dtype=wp.uint32),
        sapm: wp.array(dtype=wp.float32),
        cwave: wp.array(dtype=wp.float32),
        cband: wp.array(dtype=wp.float32),
        width: int,
        height: int,
        image_to_ujy_arcsec2: float,
        x_targets: wp.array(dtype=wp.float32),
        y_targets: wp.array(dtype=wp.float32),
        bad_mask_value: wp.uint32,
        aperture_radius_pix: float,
        annulus_inner_pix: float,
        annulus_outer_pix: float,
        out_flux: wp.array(dtype=wp.float32),
        out_unc: wp.array(dtype=wp.float32),
        out_area: wp.array(dtype=wp.float32),
        out_bkg: wp.array(dtype=wp.float32),
        out_bkg_std: wp.array(dtype=wp.float32),
        out_bad: wp.array(dtype=wp.int32),
        out_flags_summary: wp.array(dtype=wp.uint32),
        out_cwave: wp.array(dtype=wp.float32),
        out_cband: wp.array(dtype=wp.float32),
        out_status: wp.array(dtype=wp.int32),
    ):
        tid = wp.tid()
        x0 = x_targets[tid]
        y0 = y_targets[tid]
        ix0 = int(wp.floor(x0))
        iy0 = int(wp.floor(y0))

        # Bilinear sample wavelength maps at the target position.
        if x0 >= 0.0 and y0 >= 0.0 and x0 <= float(width - 1) and y0 <= float(height - 1):
            sx0 = int(wp.floor(x0))
            sy0 = int(wp.floor(y0))
            sx1 = wp.min(sx0 + 1, width - 1)
            sy1 = wp.min(sy0 + 1, height - 1)
            dxs = x0 - float(sx0)
            dys = y0 - float(sy0)
            f00 = sy0 * width + sx0
            f10 = sy0 * width + sx1
            f01 = sy1 * width + sx0
            f11 = sy1 * width + sx1
            w00 = (1.0 - dxs) * (1.0 - dys)
            w10 = dxs * (1.0 - dys)
            w01 = (1.0 - dxs) * dys
            w11 = dxs * dys
            out_cwave[tid] = cwave[f00] * w00 + cwave[f10] * w10 + cwave[f01] * w01 + cwave[f11] * w11
            out_cband[tid] = cband[f00] * w00 + cband[f10] * w10 + cband[f01] * w01 + cband[f11] * w11
        else:
            out_cwave[tid] = 0.0
            out_cband[tid] = 0.0

        ap_sum = float(0.0)
        ap_var_sum = float(0.0)
        ap_area = float(0.0)
        bad_ap = int(0)
        flags_summary = wp.uint32(0)
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
                    scale = image_to_ujy_arcsec2 * sapm[flat]
                    value = image[flat] * scale
                    var_value = variance[flat] * scale * scale
                    good = (value == value) and ((flags[flat] & bad_mask_value) == wp.uint32(0))
                    ddx_center = float(ix) - x0
                    ddy_center = float(iy) - y0
                    rr_center = wp.sqrt(ddx_center * ddx_center + ddy_center * ddy_center)

                    if rr_center <= aperture_radius_pix:
                        flags_summary = flags_summary | flags[flat]
                        if not good:
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
                                ddx_sub = sub_x - x0
                                ddy_sub = sub_y - y0
                                rr_sub = wp.sqrt(ddx_sub * ddx_sub + ddy_sub * ddy_sub)
                                if rr_sub <= aperture_radius_pix:
                                    inside += 1
                        if inside > 0:
                            weight = float(inside) / float(SUBPIX * SUBPIX)
                            ap_sum += value * weight
                            ap_var_sum += var_value * weight
                            ap_area += weight

        if ap_area <= 0.0 or ann_count <= 0:
            out_flux[tid] = 0.0
            out_unc[tid] = 0.0
            out_area[tid] = ap_area
            out_bkg[tid] = 0.0
            out_bkg_std[tid] = 0.0
            out_bad[tid] = bad_ap
            out_flags_summary[tid] = flags_summary
            out_status[tid] = 1
            return

        usable_ann_count = wp.min(ann_count, ANNULUS_MAX_PIX)
        bkg = float(0.0)
        ann_std = float(0.0)
        active_count = int(0)
        for _clip_iter in range(SIGMA_CLIP_ITERS):
            active_count = int(0)
            for i in range(ANNULUS_MAX_PIX):
                if i < usable_ann_count and ann_active[i] == 1:
                    clipped_values[active_count] = ann_values[i]
                    active_count += 1
            if active_count <= 0:
                out_flux[tid] = 0.0
                out_unc[tid] = 0.0
                out_area[tid] = ap_area
                out_bkg[tid] = 0.0
                out_bkg_std[tid] = 0.0
                out_bad[tid] = bad_ap
                out_flags_summary[tid] = flags_summary
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
        out_area[tid] = ap_area
        out_bkg[tid] = bkg
        out_bkg_std[tid] = ann_std
        out_bad[tid] = bad_ap
        out_flags_summary[tid] = flags_summary
        out_status[tid] = 0


def run_warp_aperture_batch(
    *,
    flux_ujy: np.ndarray,
    var_ujy2: np.ndarray | None,
    flags: np.ndarray | None,
    x_zero_based: np.ndarray,
    y_zero_based: np.ndarray,
    aperture_radius_pix: float,
    annulus_inner_pix: float,
    annulus_outer_pix: float,
    fatal_flag_bits: Iterable[int],
    device: str = "cuda:0",
) -> WarpApertureBatch:
    if wp is None:
        raise RuntimeError(f"Warp is not available: {_WARP_IMPORT_ERROR}")
    wp.init()
    x = np.asarray(x_zero_based, dtype=np.float32)
    y = np.asarray(y_zero_based, dtype=np.float32)
    n = len(x)
    if n == 0:
        empty_f = np.array([], dtype=float)
        empty_i = np.array([], dtype=int)
        return WarpApertureBatch(empty_f, empty_f, empty_f, empty_f, empty_f, empty_i, empty_i, empty_i, device)
    if var_ujy2 is None:
        var_ujy2 = np.zeros_like(flux_ujy, dtype=np.float32)
    if flags is None:
        flags = np.zeros_like(flux_ujy, dtype=np.uint32)

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
    out_flux = wp.empty(n, dtype=wp.float32, device=device)
    out_unc = wp.empty(n, dtype=wp.float32, device=device)
    out_area = wp.empty(n, dtype=wp.float32, device=device)
    out_bkg = wp.empty(n, dtype=wp.float32, device=device)
    out_bkg_std = wp.empty(n, dtype=wp.float32, device=device)
    out_bad = wp.empty(n, dtype=wp.int32, device=device)
    out_flags_summary = wp.empty(n, dtype=wp.uint32, device=device)
    out_status = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        _aperture_kernel,
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
            out_area,
            out_bkg,
            out_bkg_std,
            out_bad,
            out_flags_summary,
            out_status,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    return WarpApertureBatch(
        flux_uJy=out_flux.numpy().astype(float),
        flux_unc_uJy=out_unc.numpy().astype(float),
        aperture_area_pix=out_area.numpy().astype(float),
        background_uJy_per_pix=out_bkg.numpy().astype(float),
        background_unc_uJy_per_pix=out_bkg_std.numpy().astype(float),
        n_bad_aperture_pixels=out_bad.numpy().astype(int),
        flags_summary=out_flags_summary.numpy().astype(int),
        status=out_status.numpy().astype(int),
        device=device,
    )


def run_warp_frame_aperture_cupy(
    *,
    image: np.ndarray,
    variance: np.ndarray | None,
    flags: np.ndarray | None,
    sapm: np.ndarray,
    cwave: np.ndarray,
    cband: np.ndarray,
    image_to_ujy_arcsec2: float,
    x_zero_based: np.ndarray,
    y_zero_based: np.ndarray,
    aperture_radius_pix: float,
    annulus_inner_pix: float,
    annulus_outer_pix: float,
    fatal_flag_bits: Iterable[int],
    device: str = "cuda:0",
) -> WarpFrameApertureDeviceBatch:
    if wp is None:
        raise RuntimeError(f"Warp is not available: {_WARP_IMPORT_ERROR}")

    wp.init()
    calibration = upload_frame_calibration(sapm=sapm, cwave=cwave, cband=cband, device=device)
    return run_warp_frame_aperture_resident_cupy(
        image=image,
        variance=variance,
        flags=flags,
        calibration=calibration,
        image_to_ujy_arcsec2=image_to_ujy_arcsec2,
        x_zero_based=x_zero_based,
        y_zero_based=y_zero_based,
        aperture_radius_pix=aperture_radius_pix,
        annulus_inner_pix=annulus_inner_pix,
        annulus_outer_pix=annulus_outer_pix,
        fatal_flag_bits=fatal_flag_bits,
        device=device,
    )


def upload_frame_calibration(
    *,
    sapm: np.ndarray,
    cwave: np.ndarray,
    cband: np.ndarray,
    device: str = "cuda:0",
) -> WarpFrameCalibrationDevice:
    if wp is None:
        raise RuntimeError(f"Warp is not available: {_WARP_IMPORT_ERROR}")
    wp.init()
    height, width = sapm.shape
    if cwave.shape != sapm.shape or cband.shape != sapm.shape:
        raise ValueError(f"Calibration map shapes do not match: sapm={sapm.shape} cwave={cwave.shape} cband={cband.shape}")
    return WarpFrameCalibrationDevice(
        sapm=wp.array(np.ascontiguousarray(sapm.astype(np.float32).ravel()), dtype=wp.float32, device=device),
        cwave=wp.array(np.ascontiguousarray(cwave.astype(np.float32).ravel()), dtype=wp.float32, device=device),
        cband=wp.array(np.ascontiguousarray(cband.astype(np.float32).ravel()), dtype=wp.float32, device=device),
        width=int(width),
        height=int(height),
        device=device,
    )


def run_warp_frame_aperture_resident_cupy(
    *,
    image: np.ndarray,
    variance: np.ndarray | None,
    flags: np.ndarray | None,
    calibration: WarpFrameCalibrationDevice,
    image_to_ujy_arcsec2: float,
    x_zero_based: np.ndarray,
    y_zero_based: np.ndarray,
    aperture_radius_pix: float,
    annulus_inner_pix: float,
    annulus_outer_pix: float,
    fatal_flag_bits: Iterable[int],
    device: str = "cuda:0",
) -> WarpFrameApertureDeviceBatch:
    if wp is None:
        raise RuntimeError(f"Warp is not available: {_WARP_IMPORT_ERROR}")
    import cupy as cp

    wp.init()
    if calibration.device != device:
        raise ValueError(f"Calibration lives on {calibration.device}, requested {device}")
    x = np.asarray(x_zero_based, dtype=np.float32)
    y = np.asarray(y_zero_based, dtype=np.float32)
    n = len(x)
    if variance is None:
        variance = np.zeros_like(image, dtype=np.float32)
    if flags is None:
        flags = np.zeros_like(image, dtype=np.uint32)
    height, width = image.shape
    if width != calibration.width or height != calibration.height:
        raise ValueError(f"Frame shape {(height, width)} does not match calibration {(calibration.height, calibration.width)}")
    bad_mask_value = sum(1 << int(bit) for bit in fatal_flag_bits)

    image_dev = wp.array(np.ascontiguousarray(image.astype(np.float32).ravel()), dtype=wp.float32, device=device)
    variance_dev = wp.array(
        np.ascontiguousarray(np.nan_to_num(variance, nan=0.0).astype(np.float32).ravel()),
        dtype=wp.float32,
        device=device,
    )
    flags_dev = wp.array(np.ascontiguousarray(np.asarray(flags, dtype=np.uint32).ravel()), dtype=wp.uint32, device=device)
    x_dev = wp.array(x, dtype=wp.float32, device=device)
    y_dev = wp.array(y, dtype=wp.float32, device=device)
    out_flux = wp.empty(n, dtype=wp.float32, device=device)
    out_unc = wp.empty(n, dtype=wp.float32, device=device)
    out_area = wp.empty(n, dtype=wp.float32, device=device)
    out_bkg = wp.empty(n, dtype=wp.float32, device=device)
    out_bkg_std = wp.empty(n, dtype=wp.float32, device=device)
    out_bad = wp.empty(n, dtype=wp.int32, device=device)
    out_flags_summary = wp.empty(n, dtype=wp.uint32, device=device)
    out_cwave = wp.empty(n, dtype=wp.float32, device=device)
    out_cband = wp.empty(n, dtype=wp.float32, device=device)
    out_status = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        _frame_aperture_kernel,
        dim=n,
        inputs=[
            image_dev,
            variance_dev,
            flags_dev,
            calibration.sapm,
            calibration.cwave,
            calibration.cband,
            int(width),
            int(height),
            float(image_to_ujy_arcsec2),
            x_dev,
            y_dev,
            wp.uint32(bad_mask_value),
            float(aperture_radius_pix),
            float(annulus_inner_pix),
            float(annulus_outer_pix),
            out_flux,
            out_unc,
            out_area,
            out_bkg,
            out_bkg_std,
            out_bad,
            out_flags_summary,
            out_cwave,
            out_cband,
            out_status,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    columns = {
        "aperture_flux_uJy": cp.from_dlpack(out_flux),
        "aperture_flux_unc_uJy": cp.from_dlpack(out_unc),
        "aperture_area_pix": cp.from_dlpack(out_area),
        "background_uJy_per_pix": cp.from_dlpack(out_bkg),
        "background_unc_uJy_per_pix": cp.from_dlpack(out_bkg_std),
        "n_bad_aperture_pixels": cp.from_dlpack(out_bad),
        "flags_summary": cp.from_dlpack(out_flags_summary),
        "cwave_um": cp.from_dlpack(out_cwave),
        "cband_um": cp.from_dlpack(out_cband),
    }
    return WarpFrameApertureDeviceBatch(columns=columns, status=cp.from_dlpack(out_status), device=device)
