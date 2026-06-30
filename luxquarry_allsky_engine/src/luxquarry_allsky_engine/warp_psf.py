from __future__ import annotations

import time
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


MAX_PSF_PIXELS = 144


@dataclass(frozen=True)
class WarpFramePsfDeviceBatch:
    columns: dict[str, object]
    status: object
    device: str
    timings: dict[str, float]


if wp is not None:

    @wp.func
    def _cubic_bspline_weight(t: float, slot: int):
        if slot == 0:
            a = 1.0 - t
            return a * a * a / 6.0
        if slot == 1:
            return (3.0 * t * t * t - 6.0 * t * t + 4.0) / 6.0
        if slot == 2:
            return (-3.0 * t * t * t + 3.0 * t * t + 3.0 * t + 1.0) / 6.0
        return t * t * t / 6.0

    @wp.kernel(enable_backward=False)
    def _psf_build_kernel_bank(
        psf_cube: wp.array(dtype=wp.float32),
        n_planes: int,
        psf_height: int,
        psf_width: int,
        image_width: int,
        image_height: int,
        ngrid: int,
        oversamp: int,
        kernel_radius_native: int,
        kernel_size: int,
        shift_sign: float,
        sample_mode: int,
        x_targets: wp.array(dtype=wp.float32),
        y_targets: wp.array(dtype=wp.float32),
        dx_offsets: wp.array(dtype=wp.float32),
        dy_offsets: wp.array(dtype=wp.float32),
        n_offsets: int,
        out_kernels: wp.array(dtype=wp.float32),
        out_sectors: wp.array(dtype=wp.int32),
        out_kernel_sum: wp.array(dtype=wp.float32),
    ):
        cid = wp.tid()
        target_index = int(cid / n_offsets)
        offset_index = cid - target_index * n_offsets
        x = x_targets[target_index] + dx_offsets[offset_index]
        y = y_targets[target_index] + dy_offsets[offset_index]
        p = int(0)
        while p < MAX_PSF_PIXELS:
            out_kernels[cid * MAX_PSF_PIXELS + p] = 0.0
            p += 1

        if x < 0.0 or x >= float(image_width) or y < 0.0 or y >= float(image_height):
            out_sectors[cid] = -1
            out_kernel_sum[cid] = 0.0
            return

        zone_x = int(wp.floor(x / float(image_width) * float(ngrid)))
        zone_y = int(wp.floor(y / float(image_height) * float(ngrid)))
        if zone_x < 0:
            zone_x = 0
        if zone_x >= ngrid:
            zone_x = ngrid - 1
        if zone_y < 0:
            zone_y = 0
        if zone_y >= ngrid:
            zone_y = ngrid - 1
        sector = zone_y * ngrid + zone_x
        if sector < 0:
            sector = 0
        if sector >= n_planes:
            sector = n_planes - 1
        out_sectors[cid] = sector

        frac_x = x - wp.floor(x)
        frac_y = y - wp.floor(y)
        shift_x = shift_sign * frac_x * float(oversamp)
        shift_y = shift_sign * frac_y * float(oversamp)
        cy = int(psf_height / 2)
        cx = int(psf_width / 2)
        radius_super = kernel_radius_native * oversamp
        start_y = cy - radius_super
        start_x = cx - radius_super
        base = sector * psf_height * psf_width

        total = float(0.0)
        ky = int(0)
        while ky < kernel_size:
            kx = int(0)
            while kx < kernel_size:
                acc = float(0.0)
                sy = int(0)
                while sy < oversamp:
                    sx = int(0)
                    while sx < oversamp:
                        src_y = float(start_y + ky * oversamp + sy) - shift_y
                        src_x = float(start_x + kx * oversamp + sx) - shift_x
                        y0f = wp.floor(src_y)
                        x0f = wp.floor(src_x)
                        y0 = int(y0f)
                        x0 = int(x0f)
                        fy = src_y - y0f
                        fx = src_x - x0f
                        if sample_mode == 1:
                            sample = float(0.0)
                            wy_i = int(0)
                            while wy_i < 4:
                                yy = y0 + wy_i - 1
                                if yy >= 0 and yy < psf_height:
                                    wy = _cubic_bspline_weight(fy, wy_i)
                                    wx_i = int(0)
                                    while wx_i < 4:
                                        xx = x0 + wx_i - 1
                                        if xx >= 0 and xx < psf_width:
                                            wx = _cubic_bspline_weight(fx, wx_i)
                                            sample += wy * wx * psf_cube[base + yy * psf_width + xx]
                                        wx_i += 1
                                wy_i += 1
                            if sample > 0.0:
                                acc += sample
                        else:
                            if x0 >= 0 and x0 + 1 < psf_width and y0 >= 0 and y0 + 1 < psf_height:
                                v00 = wp.max(float(0.0), psf_cube[base + y0 * psf_width + x0])
                                v10 = wp.max(float(0.0), psf_cube[base + y0 * psf_width + x0 + 1])
                                v01 = wp.max(float(0.0), psf_cube[base + (y0 + 1) * psf_width + x0])
                                v11 = wp.max(float(0.0), psf_cube[base + (y0 + 1) * psf_width + x0 + 1])
                                acc += (1.0 - fy) * ((1.0 - fx) * v00 + fx * v10) + fy * (
                                    (1.0 - fx) * v01 + fx * v11
                                )
                        sx += 1
                    sy += 1
                out_kernels[cid * MAX_PSF_PIXELS + ky * 12 + kx] = acc
                total += acc
                kx += 1
            ky += 1

        if total > 0.0:
            ky = int(0)
            while ky < kernel_size:
                kx = int(0)
                while kx < kernel_size:
                    idx = cid * MAX_PSF_PIXELS + ky * 12 + kx
                    out_kernels[idx] = out_kernels[idx] / total
                    kx += 1
                ky += 1
        out_kernel_sum[cid] = total

    @wp.kernel(enable_backward=False)
    def _psf_grid_candidate_calibrated_kernel(
        image: wp.array(dtype=wp.float32),
        variance: wp.array(dtype=wp.float32),
        flags: wp.array(dtype=wp.uint32),
        sapm: wp.array(dtype=wp.float32),
        kernels: wp.array(dtype=wp.float32),
        width: int,
        height: int,
        image_to_ujy_arcsec2: float,
        kernel_size: int,
        x_targets: wp.array(dtype=wp.float32),
        y_targets: wp.array(dtype=wp.float32),
        dx_offsets: wp.array(dtype=wp.float32),
        dy_offsets: wp.array(dtype=wp.float32),
        n_offsets: int,
        bad_mask_value: wp.uint32,
        out_flux: wp.array(dtype=wp.float32),
        out_unc: wp.array(dtype=wp.float32),
        out_bkg: wp.array(dtype=wp.float32),
        out_chi2: wp.array(dtype=wp.float32),
        out_dof: wp.array(dtype=wp.int32),
        out_valid: wp.array(dtype=wp.int32),
        out_status: wp.array(dtype=wp.int32),
    ):
        cid = wp.tid()
        target_index = int(cid / n_offsets)
        offset_index = cid - target_index * n_offsets
        x = x_targets[target_index] + dx_offsets[offset_index]
        y = y_targets[target_index] + dy_offsets[offset_index]
        x0 = int(wp.floor(x + float(0.5)))
        y0 = int(wp.floor(y + float(0.5)))
        half = int(kernel_size / 2)
        s_pp = float(0.0)
        s_p1 = float(0.0)
        s_11 = float(0.0)
        s_yp = float(0.0)
        s_y1 = float(0.0)
        n_valid = int(0)

        ky = int(0)
        while ky < kernel_size:
            kx = int(0)
            while kx < kernel_size:
                ix = x0 + kx - half
                iy = y0 + ky - half
                if ix >= 0 and ix < width and iy >= 0 and iy < height:
                    flat = iy * width + ix
                    kval = kernels[cid * MAX_PSF_PIXELS + ky * 12 + kx]
                    scale = image_to_ujy_arcsec2 * sapm[flat]
                    f = image[flat] * scale
                    v = variance[flat] * scale * scale
                    good = (
                        (kval == kval)
                        and (f == f)
                        and (v == v)
                        and kval > 0.0
                        and v > 0.0
                        and ((flags[flat] & bad_mask_value) == wp.uint32(0))
                    )
                    if good:
                        w = 1.0 / v
                        s_pp += w * kval * kval
                        s_p1 += w * kval
                        s_11 += w
                        s_yp += w * f * kval
                        s_y1 += w * f
                        n_valid += 1
                kx += 1
            ky += 1

        det = s_pp * s_11 - s_p1 * s_p1
        if n_valid < 3 or det <= 0.0:
            out_status[cid] = 1
            out_flux[cid] = 0.0
            out_unc[cid] = 0.0
            out_bkg[cid] = 0.0
            out_chi2[cid] = 0.0
            out_dof[cid] = 0
            out_valid[cid] = n_valid
            return

        fit_flux = (s_yp * s_11 - s_y1 * s_p1) / det
        fit_bkg = (s_pp * s_y1 - s_p1 * s_yp) / det
        fit_unc = wp.sqrt(s_11 / det)
        chi2 = float(0.0)
        ky = int(0)
        while ky < kernel_size:
            kx = int(0)
            while kx < kernel_size:
                ix = x0 + kx - half
                iy = y0 + ky - half
                if ix >= 0 and ix < width and iy >= 0 and iy < height:
                    flat = iy * width + ix
                    kval = kernels[cid * MAX_PSF_PIXELS + ky * 12 + kx]
                    scale = image_to_ujy_arcsec2 * sapm[flat]
                    f = image[flat] * scale
                    v = variance[flat] * scale * scale
                    good = (
                        (kval == kval)
                        and (f == f)
                        and (v == v)
                        and kval > 0.0
                        and v > 0.0
                        and ((flags[flat] & bad_mask_value) == wp.uint32(0))
                    )
                    if good:
                        resid = f - (fit_flux * kval + fit_bkg)
                        chi2 += resid * resid / v
                kx += 1
            ky += 1

        out_status[cid] = 0
        out_flux[cid] = fit_flux
        out_unc[cid] = fit_unc
        out_bkg[cid] = fit_bkg
        out_chi2[cid] = chi2
        out_dof[cid] = n_valid - 2
        out_valid[cid] = n_valid

    @wp.kernel(enable_backward=False)
    def _psf_grid_reduce_kernel(
        cand_flux: wp.array(dtype=wp.float32),
        cand_unc: wp.array(dtype=wp.float32),
        cand_chi2: wp.array(dtype=wp.float32),
        cand_status: wp.array(dtype=wp.int32),
        n_offsets: int,
        metric_code: int,
        out_best_index: wp.array(dtype=wp.int32),
        out_grid_valid: wp.array(dtype=wp.int32),
        out_best_score: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        start = tid * n_offsets
        best_i = int(-1)
        n_valid = int(0)
        best_score = float(0.0)
        if metric_code == 1:
            best_score = float(3.4028234663852886e38)
        else:
            best_score = float(-3.4028234663852886e38)

        j = int(0)
        while j < n_offsets:
            cid = start + j
            if cand_status[cid] == 0:
                unc = cand_unc[cid]
                score = float(0.0)
                good_score = bool(False)
                if metric_code == 1:
                    score = cand_chi2[cid]
                    good_score = score == score
                else:
                    if unc > 0.0:
                        score = cand_flux[cid] / unc
                        good_score = score == score
                if good_score:
                    n_valid += 1
                    if metric_code == 1:
                        if score < best_score:
                            best_score = score
                            best_i = cid
                    else:
                        if score > best_score:
                            best_score = score
                            best_i = cid
            j += 1

        out_best_index[tid] = best_i
        out_grid_valid[tid] = n_valid
        out_best_score[tid] = best_score


def run_warp_frame_psf_grid_resident_cupy(
    *,
    image: np.ndarray,
    variance: np.ndarray | None,
    flags: np.ndarray | None,
    frame_data: object | None = None,
    sapm: object,
    psf_cube: np.ndarray,
    image_to_ujy_arcsec2: float,
    x_zero_based: np.ndarray,
    y_zero_based: np.ndarray,
    fatal_flag_bits: Iterable[int],
    kernel_radius_native: int = 5,
    half_range_pix: float = 1.0,
    step_pix: float = 0.5,
    metric: str = "snr",
    kernel_build_mode: str = "gpu_spline",
    shift_sign: float = 1.0,
    oversamp: int | None = None,
    device: str = "cuda:0",
) -> WarpFramePsfDeviceBatch:
    total_started = time.perf_counter()
    if wp is None:
        raise RuntimeError(f"Warp is not available: {_WARP_IMPORT_ERROR}")
    if metric not in {"snr", "chi2"}:
        raise ValueError(f"unsupported PSF grid metric: {metric}")
    if kernel_build_mode not in {"gpu_spline", "gpu_bilinear"}:
        raise ValueError(f"unsupported PSF kernel build mode: {kernel_build_mode}")

    import cupy as cp

    wp.init()
    x_pix = np.asarray(x_zero_based, dtype=float)
    y_pix = np.asarray(y_zero_based, dtype=float)
    if x_pix.shape != y_pix.shape:
        raise ValueError("x_zero_based and y_zero_based must have matching shape")
    n_targets = int(x_pix.size)
    if n_targets == 0:
        empty = cp.asarray([], dtype=cp.float32)
        empty_i = cp.asarray([], dtype=cp.int32)
        columns = _columns_from_arrays(
            empty,
            empty,
            empty,
            empty,
            empty_i,
            empty_i,
            empty_i,
            empty_i,
            empty,
            empty,
            empty,
        )
        return WarpFramePsfDeviceBatch(columns=columns, status=empty_i, device=device, timings={"total_wall_sec": 0.0})

    if frame_data is None:
        if variance is None:
            variance = np.zeros_like(image, dtype=np.float32)
        if flags is None:
            flags = np.zeros_like(image, dtype=np.uint32)
        height, width = image.shape
    else:
        if getattr(frame_data, "device", device) != device:
            raise ValueError(f"Frame data lives on {getattr(frame_data, 'device', None)}, requested {device}")
        height = int(getattr(frame_data, "height"))
        width = int(getattr(frame_data, "width"))
    psf_source = np.asarray(psf_cube, dtype=np.float32)
    if psf_source.ndim != 3:
        raise ValueError(f"expected 3D PSF cube, got {psf_source.shape}")
    n_planes, psf_height, psf_width = (int(psf_source.shape[0]), int(psf_source.shape[1]), int(psf_source.shape[2]))
    ngrid = int(round(np.sqrt(n_planes)))
    if ngrid * ngrid != n_planes:
        ngrid = 11
    oversamp_value = int(oversamp or round(psf_height / 10))
    oversamp_value = max(1, oversamp_value)
    kernel_size = int((2 * int(kernel_radius_native) * oversamp_value + 1) // oversamp_value)
    if kernel_size > 12:
        raise ValueError("Warp PSF grid expects kernel_size <= 12")

    t_candidates = time.perf_counter()
    offsets = np.arange(
        -float(half_range_pix),
        float(half_range_pix) + 0.5 * float(step_pix),
        float(step_pix),
        dtype=np.float32,
    )
    dx_grid, dy_grid = np.meshgrid(offsets, offsets, indexing="xy")
    dx_offsets = dx_grid.ravel().astype(np.float32, copy=False)
    dy_offsets = dy_grid.ravel().astype(np.float32, copy=False)
    n_offsets = int(dx_offsets.size)
    n_candidates = int(n_targets * n_offsets)
    x_targets = np.ascontiguousarray(x_pix.astype(np.float32, copy=False))
    y_targets = np.ascontiguousarray(y_pix.astype(np.float32, copy=False))
    dx_offsets = np.ascontiguousarray(dx_offsets, dtype=np.float32)
    dy_offsets = np.ascontiguousarray(dy_offsets, dtype=np.float32)
    candidate_grid_wall = time.perf_counter() - t_candidates

    psf_source = np.nan_to_num(psf_source, nan=0.0, posinf=0.0, neginf=0.0)
    sample_mode = 0
    spline_coeff_wall = 0.0
    if kernel_build_mode == "gpu_spline":
        from scipy.ndimage import spline_filter

        t_spline = time.perf_counter()
        coeff_cube = np.empty_like(psf_source, dtype=np.float32)
        for plane_index in range(n_planes):
            coeff_cube[plane_index] = spline_filter(
                psf_source[plane_index],
                order=3,
                output=np.float64,
                mode="constant",
            ).astype(np.float32)
        psf_source = coeff_cube
        sample_mode = 1
        spline_coeff_wall = time.perf_counter() - t_spline

    bad_mask_value = sum(1 << int(bit) for bit in fatal_flag_bits)
    t_upload = time.perf_counter()
    if frame_data is None:
        image_dev = wp.array(_flat_float32(image), dtype=wp.float32, device=device)
        variance_dev = wp.array(_flat_float32(np.nan_to_num(variance, nan=0.0)), dtype=wp.float32, device=device)
        flags_dev = wp.array(_flat_uint32(flags), dtype=wp.uint32, device=device)
    else:
        image_dev = frame_data.image
        variance_dev = frame_data.variance
        flags_dev = frame_data.flags
    psf_dev = wp.array(np.ascontiguousarray(psf_source.ravel()), dtype=wp.float32, device=device)
    x_dev = wp.array(x_targets, dtype=wp.float32, device=device)
    y_dev = wp.array(y_targets, dtype=wp.float32, device=device)
    dx_dev = wp.array(dx_offsets, dtype=wp.float32, device=device)
    dy_dev = wp.array(dy_offsets, dtype=wp.float32, device=device)
    upload_wall = time.perf_counter() - t_upload

    t_device = time.perf_counter()
    kernels_dev = wp.empty(n_candidates * MAX_PSF_PIXELS, dtype=wp.float32, device=device)
    sectors_dev = wp.empty(n_candidates, dtype=wp.int32, device=device)
    kernel_sum_dev = wp.empty(n_candidates, dtype=wp.float32, device=device)
    wp.launch(
        _psf_build_kernel_bank,
        dim=n_candidates,
        inputs=[
            psf_dev,
            int(n_planes),
            int(psf_height),
            int(psf_width),
            int(width),
            int(height),
            int(ngrid),
            int(oversamp_value),
            int(kernel_radius_native),
            int(kernel_size),
            float(shift_sign),
            int(sample_mode),
            x_dev,
            y_dev,
            dx_dev,
            dy_dev,
            int(n_offsets),
            kernels_dev,
            sectors_dev,
            kernel_sum_dev,
        ],
        device=device,
    )

    cand_flux = wp.empty(n_candidates, dtype=wp.float32, device=device)
    cand_unc = wp.empty(n_candidates, dtype=wp.float32, device=device)
    cand_bkg = wp.empty(n_candidates, dtype=wp.float32, device=device)
    cand_chi2 = wp.empty(n_candidates, dtype=wp.float32, device=device)
    cand_dof = wp.empty(n_candidates, dtype=wp.int32, device=device)
    cand_valid = wp.empty(n_candidates, dtype=wp.int32, device=device)
    cand_status = wp.empty(n_candidates, dtype=wp.int32, device=device)
    wp.launch(
        _psf_grid_candidate_calibrated_kernel,
        dim=n_candidates,
        inputs=[
            image_dev,
            variance_dev,
            flags_dev,
            sapm,
            kernels_dev,
            int(width),
            int(height),
            float(image_to_ujy_arcsec2),
            int(kernel_size),
            x_dev,
            y_dev,
            dx_dev,
            dy_dev,
            int(n_offsets),
            wp.uint32(bad_mask_value),
            cand_flux,
            cand_unc,
            cand_bkg,
            cand_chi2,
            cand_dof,
            cand_valid,
            cand_status,
        ],
        device=device,
    )

    best_index_dev = wp.empty(n_targets, dtype=wp.int32, device=device)
    grid_valid_dev = wp.empty(n_targets, dtype=wp.int32, device=device)
    best_score_dev = wp.empty(n_targets, dtype=wp.float32, device=device)
    wp.launch(
        _psf_grid_reduce_kernel,
        dim=n_targets,
        inputs=[
            cand_flux,
            cand_unc,
            cand_chi2,
            cand_status,
            int(n_offsets),
            1 if metric == "chi2" else 0,
            best_index_dev,
            grid_valid_dev,
            best_score_dev,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    device_submit_sync_wall = time.perf_counter() - t_device

    t_gather = time.perf_counter()
    best_index = cp.from_dlpack(best_index_dev)
    valid_best = best_index >= 0
    safe_index = cp.where(valid_best, best_index, cp.zeros_like(best_index))
    flux = cp.from_dlpack(cand_flux)[safe_index]
    unc = cp.from_dlpack(cand_unc)[safe_index]
    bkg = cp.from_dlpack(cand_bkg)[safe_index]
    chi2 = cp.from_dlpack(cand_chi2)[safe_index]
    dof = cp.from_dlpack(cand_dof)[safe_index]
    valid = cp.from_dlpack(cand_valid)[safe_index]
    sectors = cp.from_dlpack(sectors_dev)[safe_index]
    kernel_sums = cp.from_dlpack(kernel_sum_dev)[safe_index]
    score = cp.from_dlpack(best_score_dev)
    grid_valid = cp.from_dlpack(grid_valid_dev)
    best_target = safe_index // int(n_offsets)
    best_offset = safe_index - best_target * int(n_offsets)
    x_target_gpu = cp.from_dlpack(x_dev)
    y_target_gpu = cp.from_dlpack(y_dev)
    dx_gpu = cp.from_dlpack(dx_dev)
    dy_gpu = cp.from_dlpack(dy_dev)
    dx_fit = dx_gpu[best_offset]
    dy_fit = dy_gpu[best_offset]
    x_fit = x_target_gpu[best_target] + dx_fit
    y_fit = y_target_gpu[best_target] + dy_fit
    fit_offset = cp.sqrt(dx_fit * dx_fit + dy_fit * dy_fit)
    status = cp.where(valid_best, cp.asarray(0, dtype=cp.int32), cp.asarray(1, dtype=cp.int32))

    nan_f = cp.asarray(cp.nan, dtype=cp.float32)
    flux = cp.where(valid_best, flux, nan_f)
    unc = cp.where(valid_best, unc, nan_f)
    bkg = cp.where(valid_best, bkg, nan_f)
    chi2 = cp.where(valid_best, chi2, nan_f)
    reduced = cp.where((valid_best) & (dof > 0), chi2 / dof.astype(cp.float32), nan_f)
    sectors = cp.where(valid_best, sectors, cp.asarray(-1, dtype=cp.int32))
    valid = cp.where(valid_best, valid, cp.asarray(0, dtype=cp.int32))
    kernel_sums = cp.where(valid_best, kernel_sums, nan_f)
    fit_offset = cp.where(valid_best, fit_offset, nan_f)
    gather_wall = time.perf_counter() - t_gather

    columns = _columns_from_arrays(
        flux,
        unc,
        bkg,
        chi2,
        dof,
        valid,
        sectors,
        grid_valid,
        score,
        x_fit,
        y_fit,
        reduced=reduced,
        fit_offset=fit_offset,
        kernel_sums=kernel_sums,
        n_offsets=n_offsets,
        kernel_size=kernel_size,
    )
    return WarpFramePsfDeviceBatch(
        columns=columns,
        status=status,
        device=device,
        timings={
            "candidate_grid_wall_sec": candidate_grid_wall,
            "spline_coeff_wall_sec": spline_coeff_wall,
            "upload_wall_sec": upload_wall,
            "device_submit_sync_wall_sec": device_submit_sync_wall,
            "gather_wall_sec": gather_wall,
            "total_wall_sec": time.perf_counter() - total_started,
            "target_count": float(n_targets),
            "candidate_count": float(n_candidates),
            "grid_offsets": float(n_offsets),
        },
    )


def _columns_from_arrays(
    flux,
    unc,
    bkg,
    chi2,
    dof,
    valid,
    sectors,
    grid_valid,
    score,
    x_fit,
    y_fit,
    *,
    reduced=None,
    fit_offset=None,
    kernel_sums=None,
    n_offsets: int = 0,
    kernel_size: int = 0,
) -> dict[str, object]:
    import cupy as cp

    n = len(flux)
    if reduced is None:
        reduced = cp.asarray([], dtype=cp.float32)
    if fit_offset is None:
        fit_offset = cp.asarray([], dtype=cp.float32)
    if kernel_sums is None:
        kernel_sums = cp.asarray([], dtype=cp.float32)
    return {
        "psf_flux_uJy": flux,
        "psf_flux_unc_uJy": unc,
        "psf_background_uJy_per_pix": bkg,
        "psf_chi2": chi2,
        "psf_reduced_chi2": reduced,
        "psf_dof": dof,
        "psf_n_valid": valid,
        "psf_sector": sectors,
        "psf_grid_n_valid": grid_valid,
        "psf_grid_best_score": score,
        "psf_x_fit": x_fit,
        "psf_y_fit": y_fit,
        "psf_fit_offset_pix": fit_offset,
        "psf_kernel_sum": kernel_sums,
        "psf_grid_n_trials": cp.full(n, int(n_offsets), dtype=cp.int32),
        "psf_kernel_shape_y": cp.full(n, int(kernel_size), dtype=cp.int32),
        "psf_kernel_shape_x": cp.full(n, int(kernel_size), dtype=cp.int32),
    }


def _flat_float32(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float32).ravel())


def _flat_uint32(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.uint32).ravel())
