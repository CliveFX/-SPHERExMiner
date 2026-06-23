from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from astropy.io import fits
from scipy.ndimage import shift as ndi_shift
from scipy.ndimage import spline_filter

try:
    import warp as wp
except Exception as exc:  # pragma: no cover - optional GPU dependency
    wp = None
    _WARP_IMPORT_ERROR = exc
else:
    _WARP_IMPORT_ERROR = None


@dataclass(frozen=True)
class PsfFitResult:
    flux_uJy: float
    flux_unc_uJy: float
    background_uJy_per_pix: float
    chi2: float
    reduced_chi2: float
    dof: int
    n_valid: int
    status: str
    sector: int | None
    x_fit: float
    y_fit: float
    x_init: float
    y_init: float
    fit_offset_pix: float
    kernel_sum: float
    kernel_shape_y: int
    kernel_shape_x: int
    grid_status: str
    grid_metric: str
    grid_n_trials: int
    grid_n_valid: int
    grid_best_score: float
    backend: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PsfKernel:
    data: np.ndarray
    sector: int
    oversamp: int


@dataclass(frozen=True)
class WarpPsfBatchResult:
    rows: list[PsfFitResult]
    device: str


MAX_PSF_PIXELS = 144


def extract_native_psf_kernel(
    *,
    hdul: fits.HDUList,
    x_pix: float,
    y_pix: float,
    kernel_radius_native: int = 5,
    shift_sign: float = 1.0,
) -> PsfKernel:
    """Return a normalized native-pixel PSF kernel for a target position.

    This follows the SPExPI convention: select one plane from the FITS PSF
    extension's spatial grid, shift the oversampled PSF by the target subpixel
    phase, crop, rebin to native pixels, and normalize to unit sum.
    """
    if "PSF" not in hdul:
        raise ValueError("missing PSF extension")
    if "IMAGE" not in hdul:
        raise ValueError("missing IMAGE extension")

    psf_cube = np.asarray(hdul["PSF"].data, dtype=float)
    if psf_cube.ndim != 3:
        raise ValueError(f"expected 3D PSF cube, got {psf_cube.shape}")

    image_shape = hdul["IMAGE"].data.shape
    ny, nx = int(image_shape[0]), int(image_shape[1])
    nplanes = int(psf_cube.shape[0])
    ngrid = int(round(np.sqrt(nplanes)))
    if ngrid * ngrid != nplanes:
        ngrid = 11

    zone_x = min(ngrid - 1, max(0, int(np.floor(float(x_pix) / nx * ngrid))))
    zone_y = min(ngrid - 1, max(0, int(np.floor(float(y_pix) / ny * ngrid))))
    sector = min(nplanes - 1, max(0, zone_y * ngrid + zone_x))

    oversamp = int(hdul["PSF"].header.get("OVERSAMP", 10))
    psf_super = np.nan_to_num(psf_cube[sector], nan=0.0, posinf=0.0, neginf=0.0)
    psf_super = np.clip(psf_super, 0.0, None)
    psf_super = _normalize_positive(psf_super, "selected PSF plane")

    frac_x = float(x_pix) - np.floor(float(x_pix))
    frac_y = float(y_pix) - np.floor(float(y_pix))
    shift_x = float(shift_sign) * frac_x * oversamp
    shift_y = float(shift_sign) * frac_y * oversamp
    shifted = ndi_shift(psf_super, shift=(shift_y, shift_x), order=3, mode="constant", cval=0.0, prefilter=True)
    shifted = np.clip(np.nan_to_num(shifted, nan=0.0), 0.0, None)
    shifted = _normalize_positive(shifted, "shifted PSF plane")

    cy = shifted.shape[0] // 2
    cx = shifted.shape[1] // 2
    radius_super = int(kernel_radius_native * oversamp)
    cut = shifted[
        max(0, cy - radius_super) : min(shifted.shape[0], cy + radius_super + 1),
        max(0, cx - radius_super) : min(shifted.shape[1], cx + radius_super + 1),
    ]
    size = min((cut.shape[0] // oversamp) * oversamp, (cut.shape[1] // oversamp) * oversamp)
    if size < oversamp:
        raise ValueError("PSF crop too small")
    cut = cut[:size, :size]
    native_n = size // oversamp
    native = cut.reshape(native_n, oversamp, native_n, oversamp).sum(axis=(1, 3))
    native = _normalize_positive(native, "native PSF kernel")
    return PsfKernel(data=native, sector=int(sector), oversamp=int(oversamp))


def fit_psf_cpu(
    *,
    flux_ujy: np.ndarray,
    var_ujy2: np.ndarray | None,
    flags: np.ndarray | None,
    psf_kernel: np.ndarray,
    x_pix: float,
    y_pix: float,
    x_init: float,
    y_init: float,
    fatal_flag_bits: Iterable[int],
    sector: int | None,
    backend: str = "cpu_psf_fit",
) -> PsfFitResult:
    """Fit source flux and constant background with inverse-variance weights."""
    if not np.isfinite(x_pix) or not np.isfinite(y_pix):
        return _bad("bad_position", x_pix, y_pix, x_init, y_init, sector, backend)

    psf = _normalize_positive(np.asarray(psf_kernel, dtype=float), "fit PSF kernel")
    ph, pw = psf.shape
    x0 = int(np.round(float(x_pix)))
    y0 = int(np.round(float(y_pix)))
    left_x = pw // 2
    left_y = ph // 2
    x1 = max(0, x0 - left_x)
    y1 = max(0, y0 - left_y)
    x2 = min(flux_ujy.shape[1], x0 + (pw - left_x))
    y2 = min(flux_ujy.shape[0], y0 + (ph - left_y))
    cut = np.asarray(flux_ujy[y1:y2, x1:x2], dtype=float)
    psf_x1 = left_x - (x0 - x1)
    psf_y1 = left_y - (y0 - y1)
    psf_cut = psf[psf_y1 : psf_y1 + cut.shape[0], psf_x1 : psf_x1 + cut.shape[1]]
    if cut.shape != psf_cut.shape or cut.size == 0:
        return _bad("shape_mismatch", x_pix, y_pix, x_init, y_init, sector, backend)

    if var_ujy2 is None:
        finite = np.isfinite(cut)
        local_var = np.nanvar(cut[finite]) if np.count_nonzero(finite) > 3 else np.nan
        var_cut = np.full(cut.shape, local_var, dtype=float)
    else:
        var_cut = np.asarray(var_ujy2[y1:y2, x1:x2], dtype=float)

    valid = np.isfinite(cut) & np.isfinite(psf_cut) & np.isfinite(var_cut) & (var_cut > 0)
    if flags is not None:
        fatal_mask = sum(1 << int(bit) for bit in fatal_flag_bits)
        valid &= (np.asarray(flags[y1:y2, x1:x2]) & fatal_mask) == 0

    n_valid = int(np.count_nonzero(valid))
    if n_valid < 3:
        return _bad("too_few_valid_pixels", x_pix, y_pix, x_init, y_init, sector, backend, n_valid=n_valid)

    p = psf_cut[valid]
    one = np.ones(n_valid, dtype=float)
    y = cut[valid]
    w = 1.0 / var_cut[valid]
    s_pp = float(np.sum(w * p * p))
    s_p1 = float(np.sum(w * p * one))
    s_11 = float(np.sum(w))
    s_yp = float(np.sum(w * y * p))
    s_y1 = float(np.sum(w * y))
    det = s_pp * s_11 - s_p1 * s_p1
    if not np.isfinite(det) or det <= 0.0:
        return _bad("singular_fit", x_pix, y_pix, x_init, y_init, sector, backend, n_valid=n_valid)

    flux = (s_yp * s_11 - s_y1 * s_p1) / det
    bkg = (s_pp * s_y1 - s_p1 * s_yp) / det
    unc = np.sqrt(s_11 / det)
    model = flux * p + bkg
    chi2 = float(np.sum((y - model) ** 2 * w))
    dof = int(n_valid - 2)
    reduced = chi2 / dof if dof > 0 else np.nan
    return PsfFitResult(
        flux_uJy=float(flux),
        flux_unc_uJy=float(unc),
        background_uJy_per_pix=float(bkg),
        chi2=chi2,
        reduced_chi2=float(reduced) if np.isfinite(reduced) else np.nan,
        dof=dof,
        n_valid=n_valid,
        status="ok",
        sector=sector,
        x_fit=float(x_pix),
        y_fit=float(y_pix),
        x_init=float(x_init),
        y_init=float(y_init),
        fit_offset_pix=float(np.hypot(float(x_pix) - float(x_init), float(y_pix) - float(y_init))),
        kernel_sum=float(np.nansum(psf_kernel)),
        kernel_shape_y=int(psf.shape[0]),
        kernel_shape_x=int(psf.shape[1]),
        grid_status="disabled",
        grid_metric="none",
        grid_n_trials=1,
        grid_n_valid=1,
        grid_best_score=float(flux / unc) if unc > 0 else np.nan,
        backend=backend,
    )


def fit_psf_cpu_with_grid(
    *,
    hdul: fits.HDUList,
    flux_ujy: np.ndarray,
    var_ujy2: np.ndarray | None,
    flags: np.ndarray | None,
    x_pix: float,
    y_pix: float,
    fatal_flag_bits: Iterable[int],
    kernel_radius_native: int = 5,
    shift_sign: float = 1.0,
    half_range_pix: float = 1.0,
    step_pix: float = 0.5,
    metric: str = "snr",
) -> tuple[PsfFitResult, PsfKernel]:
    """Try a local grid of centroids and keep the best valid PSF fit."""
    offsets = np.arange(-float(half_range_pix), float(half_range_pix) + 0.5 * float(step_pix), float(step_pix))
    best: PsfFitResult | None = None
    best_kernel: PsfKernel | None = None
    best_score = -np.inf if metric != "chi2" else np.inf
    n_trials = 0
    n_valid = 0
    fallback_result: PsfFitResult | None = None
    fallback_kernel: PsfKernel | None = None

    for dy in offsets:
        for dx in offsets:
            x_try = float(x_pix + dx)
            y_try = float(y_pix + dy)
            if not (0 <= x_try < flux_ujy.shape[1] and 0 <= y_try < flux_ujy.shape[0]):
                continue
            n_trials += 1
            try:
                kernel = extract_native_psf_kernel(
                    hdul=hdul,
                    x_pix=x_try,
                    y_pix=y_try,
                    kernel_radius_native=kernel_radius_native,
                    shift_sign=shift_sign,
                )
                result = fit_psf_cpu(
                    flux_ujy=flux_ujy,
                    var_ujy2=var_ujy2,
                    flags=flags,
                    psf_kernel=kernel.data,
                    x_pix=x_try,
                    y_pix=y_try,
                    x_init=x_pix,
                    y_init=y_pix,
                    fatal_flag_bits=fatal_flag_bits,
                    sector=kernel.sector,
                    backend="cpu_psf_grid",
                )
            except Exception:
                continue
            if dx == 0 and dy == 0:
                fallback_result = result
                fallback_kernel = kernel
            if result.status != "ok" or not np.isfinite(result.flux_unc_uJy) or result.flux_unc_uJy <= 0:
                continue
            n_valid += 1
            if metric == "chi2":
                score = result.reduced_chi2
                better = np.isfinite(score) and score < best_score
            else:
                score = result.flux_uJy / result.flux_unc_uJy
                better = np.isfinite(score) and score > best_score
            if best is None or better:
                best = result
                best_kernel = kernel
                best_score = float(score)

    if best is None:
        if fallback_result is not None and fallback_kernel is not None:
            return _with_grid_metadata(fallback_result, "fallback_initial_position", metric, n_trials, n_valid, np.nan), fallback_kernel
        bad = _bad("no_valid_grid_trials", x_pix, y_pix, x_pix, y_pix, None, "cpu_psf_grid")
        return _with_grid_metadata(bad, "no_valid_trials", metric, n_trials, n_valid, np.nan), PsfKernel(np.zeros((0, 0)), -1, 10)
    return _with_grid_metadata(best, "ok", metric, n_trials, n_valid, best_score), best_kernel  # type: ignore[arg-type]


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
        x_candidates: wp.array(dtype=wp.float32),
        y_candidates: wp.array(dtype=wp.float32),
        out_kernels: wp.array(dtype=wp.float32),
        out_sectors: wp.array(dtype=wp.int32),
        out_kernel_sum: wp.array(dtype=wp.float32),
    ):
        cid = wp.tid()
        x = x_candidates[cid]
        y = y_candidates[cid]
        if x < 0.0 or x >= float(image_width) or y < 0.0 or y >= float(image_height):
            out_sectors[cid] = -1
            out_kernel_sum[cid] = 0.0
            p = int(0)
            while p < MAX_PSF_PIXELS:
                out_kernels[cid * MAX_PSF_PIXELS + p] = 0.0
                p += 1
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
        p = int(0)
        while p < MAX_PSF_PIXELS:
            out_kernels[cid * MAX_PSF_PIXELS + p] = 0.0
            p += 1

        frac_x = x - wp.floor(x)
        frac_y = y - wp.floor(y)
        shift_x = shift_sign * frac_x * float(oversamp)
        shift_y = shift_sign * frac_y * float(oversamp)
        cy = int(psf_height / 2)
        cx = int(psf_width / 2)
        radius_super = kernel_radius_native * oversamp
        start_y = cy - radius_super
        start_x = cx - radius_super

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
                        base = sector * psf_height * psf_width
                        if sample_mode == 1:
                            y0f = wp.floor(src_y)
                            x0f = wp.floor(src_x)
                            y0 = int(y0f)
                            x0 = int(x0f)
                            fy = src_y - y0f
                            fx = src_x - x0f
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
                            y0f = wp.floor(src_y)
                            x0f = wp.floor(src_x)
                            y0 = int(y0f)
                            x0 = int(x0f)
                            fy = src_y - y0f
                            fx = src_x - x0f
                            if x0 >= 0 and x0 + 1 < psf_width and y0 >= 0 and y0 + 1 < psf_height:
                                v00 = psf_cube[base + y0 * psf_width + x0]
                                v10 = psf_cube[base + y0 * psf_width + x0 + 1]
                                v01 = psf_cube[base + (y0 + 1) * psf_width + x0]
                                v11 = psf_cube[base + (y0 + 1) * psf_width + x0 + 1]
                                if v00 < 0.0:
                                    v00 = 0.0
                                if v10 < 0.0:
                                    v10 = 0.0
                                if v01 < 0.0:
                                    v01 = 0.0
                                if v11 < 0.0:
                                    v11 = 0.0
                                acc += (1.0 - fy) * ((1.0 - fx) * v00 + fx * v10) + fy * ((1.0 - fx) * v01 + fx * v11)
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
    def _psf_fit_kernel(
        flux: wp.array(dtype=wp.float32),
        var: wp.array(dtype=wp.float32),
        flags: wp.array(dtype=wp.uint32),
        kernels: wp.array(dtype=wp.float32),
        width: int,
        height: int,
        kernel_size: int,
        x_targets: wp.array(dtype=wp.float32),
        y_targets: wp.array(dtype=wp.float32),
        sectors: wp.array(dtype=wp.int32),
        bad_mask_value: wp.uint32,
        out_flux: wp.array(dtype=wp.float32),
        out_unc: wp.array(dtype=wp.float32),
        out_bkg: wp.array(dtype=wp.float32),
        out_chi2: wp.array(dtype=wp.float32),
        out_dof: wp.array(dtype=wp.int32),
        out_valid: wp.array(dtype=wp.int32),
        out_status: wp.array(dtype=wp.int32),
    ):
        tid = wp.tid()
        x = x_targets[tid]
        y = y_targets[tid]
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
                    kval = kernels[tid * MAX_PSF_PIXELS + ky * 12 + kx]
                    v = var[flat]
                    f = flux[flat]
                    good = (kval == kval) and (f == f) and (v == v) and v > 0.0 and ((flags[flat] & bad_mask_value) == wp.uint32(0))
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
            out_status[tid] = 1
            out_flux[tid] = 0.0
            out_unc[tid] = 0.0
            out_bkg[tid] = 0.0
            out_chi2[tid] = 0.0
            out_dof[tid] = 0
            out_valid[tid] = n_valid
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
                    kval = kernels[tid * MAX_PSF_PIXELS + ky * 12 + kx]
                    v = var[flat]
                    f = flux[flat]
                    good = (kval == kval) and (f == f) and (v == v) and v > 0.0 and ((flags[flat] & bad_mask_value) == wp.uint32(0))
                    if good:
                        resid = f - (fit_flux * kval + fit_bkg)
                        chi2 += resid * resid / v
                kx += 1
            ky += 1

        out_status[tid] = 0
        out_flux[tid] = fit_flux
        out_unc[tid] = fit_unc
        out_bkg[tid] = fit_bkg
        out_chi2[tid] = chi2
        out_dof[tid] = n_valid - 2
        out_valid[tid] = n_valid

    @wp.kernel(enable_backward=False)
    def _psf_grid_candidate_kernel(
        flux: wp.array(dtype=wp.float32),
        var: wp.array(dtype=wp.float32),
        flags: wp.array(dtype=wp.uint32),
        kernels: wp.array(dtype=wp.float32),
        width: int,
        height: int,
        kernel_size: int,
        x_candidates: wp.array(dtype=wp.float32),
        y_candidates: wp.array(dtype=wp.float32),
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
        x = x_candidates[cid]
        y = y_candidates[cid]
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
                    v = var[flat]
                    f = flux[flat]
                    good = (kval == kval) and (f == f) and (v == v) and v > 0.0 and ((flags[flat] & bad_mask_value) == wp.uint32(0))
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
                    v = var[flat]
                    f = flux[flat]
                    good = (kval == kval) and (f == f) and (v == v) and v > 0.0 and ((flags[flat] & bad_mask_value) == wp.uint32(0))
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


def fit_psf_warp_batch(
    *,
    flux_ujy: np.ndarray,
    var_ujy2: np.ndarray | None,
    flags: np.ndarray | None,
    kernels: list[PsfKernel],
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    x_init: np.ndarray,
    y_init: np.ndarray,
    fatal_flag_bits: Iterable[int],
    device: str = "cuda:0",
) -> WarpPsfBatchResult:
    """Run the weighted PSF linear solve on the GPU for prebuilt kernels."""
    if wp is None:
        raise RuntimeError(f"Warp is not available: {_WARP_IMPORT_ERROR}")
    if not kernels:
        return WarpPsfBatchResult(rows=[], device=device)
    wp.init()
    kernel_size = int(kernels[0].data.shape[0])
    if kernel_size > 12 or any(k.data.shape != (kernel_size, kernel_size) for k in kernels):
        raise ValueError("Warp PSF benchmark expects same-size kernels with size <= 12")
    if var_ujy2 is None:
        var_ujy2 = np.full_like(flux_ujy, np.nanvar(flux_ujy[np.isfinite(flux_ujy)]), dtype=float)
    if flags is None:
        flags = np.zeros_like(flux_ujy, dtype=np.uint32)

    n = len(kernels)
    packed = np.zeros((n, MAX_PSF_PIXELS), dtype=np.float32)
    for i, kernel in enumerate(kernels):
        block = np.zeros((12, 12), dtype=np.float32)
        block[:kernel_size, :kernel_size] = kernel.data.astype(np.float32)
        packed[i] = block.ravel()

    height, width = flux_ujy.shape
    bad_mask_value = sum(1 << int(bit) for bit in fatal_flag_bits)
    flux_dev = wp.array(np.ascontiguousarray(flux_ujy.astype(np.float32).ravel()), dtype=wp.float32, device=device)
    var_dev = wp.array(np.ascontiguousarray(np.nan_to_num(var_ujy2, nan=0.0).astype(np.float32).ravel()), dtype=wp.float32, device=device)
    flags_dev = wp.array(np.ascontiguousarray(np.asarray(flags, dtype=np.uint32).ravel()), dtype=wp.uint32, device=device)
    kernels_dev = wp.array(packed.ravel(), dtype=wp.float32, device=device)
    x_dev = wp.array(np.asarray(x_pix, dtype=np.float32), dtype=wp.float32, device=device)
    y_dev = wp.array(np.asarray(y_pix, dtype=np.float32), dtype=wp.float32, device=device)
    sectors_dev = wp.array(np.asarray([k.sector for k in kernels], dtype=np.int32), dtype=wp.int32, device=device)
    out_flux = wp.empty(n, dtype=wp.float32, device=device)
    out_unc = wp.empty(n, dtype=wp.float32, device=device)
    out_bkg = wp.empty(n, dtype=wp.float32, device=device)
    out_chi2 = wp.empty(n, dtype=wp.float32, device=device)
    out_dof = wp.empty(n, dtype=wp.int32, device=device)
    out_valid = wp.empty(n, dtype=wp.int32, device=device)
    out_status = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        _psf_fit_kernel,
        dim=n,
        inputs=[
            flux_dev,
            var_dev,
            flags_dev,
            kernels_dev,
            int(width),
            int(height),
            int(kernel_size),
            x_dev,
            y_dev,
            sectors_dev,
            wp.uint32(bad_mask_value),
            out_flux,
            out_unc,
            out_bkg,
            out_chi2,
            out_dof,
            out_valid,
            out_status,
        ],
        device=device,
    )
    wp.synchronize_device(device)

    flux = out_flux.numpy().astype(float)
    unc = out_unc.numpy().astype(float)
    bkg = out_bkg.numpy().astype(float)
    chi2 = out_chi2.numpy().astype(float)
    dof = out_dof.numpy().astype(int)
    valid = out_valid.numpy().astype(int)
    status = out_status.numpy().astype(int)
    rows = []
    for i in range(n):
        ok = status[i] == 0
        reduced = chi2[i] / dof[i] if ok and dof[i] > 0 else np.nan
        rows.append(
            PsfFitResult(
                flux_uJy=float(flux[i]) if ok else np.nan,
                flux_unc_uJy=float(unc[i]) if ok else np.nan,
                background_uJy_per_pix=float(bkg[i]) if ok else np.nan,
                chi2=float(chi2[i]) if ok else np.nan,
                reduced_chi2=float(reduced) if np.isfinite(reduced) else np.nan,
                dof=int(dof[i]) if ok else 0,
                n_valid=int(valid[i]),
                status="ok" if ok else "bad_fit",
                sector=int(kernels[i].sector),
                x_fit=float(x_pix[i]),
                y_fit=float(y_pix[i]),
                x_init=float(x_init[i]),
                y_init=float(y_init[i]),
                fit_offset_pix=float(np.hypot(float(x_pix[i]) - float(x_init[i]), float(y_pix[i]) - float(y_init[i]))),
                kernel_sum=float(np.nansum(kernels[i].data)),
                kernel_shape_y=int(kernels[i].data.shape[0]),
                kernel_shape_x=int(kernels[i].data.shape[1]),
                grid_status="cpu_prepared_kernel",
                grid_metric="none",
                grid_n_trials=0,
                grid_n_valid=0,
                grid_best_score=float(flux[i] / unc[i]) if ok and unc[i] > 0 else np.nan,
                backend="warp_psf_fit",
            )
        )
    return WarpPsfBatchResult(rows=rows, device=device)


def fit_psf_warp_grid_batch(
    *,
    hdul: fits.HDUList,
    flux_ujy: np.ndarray,
    var_ujy2: np.ndarray | None,
    flags: np.ndarray | None,
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    fatal_flag_bits: Iterable[int],
    kernel_radius_native: int = 5,
    shift_sign: float = 1.0,
    half_range_pix: float = 1.0,
    step_pix: float = 0.5,
    metric: str = "snr",
    kernel_build_mode: str = "gpu_bilinear",
    device: str = "cuda:0",
) -> WarpPsfBatchResult:
    """Run SPExPI-style local-grid forced PSF fits on the GPU.

    The launch shape is flattened as ``targets * grid_offsets``. Each candidate
    fit is independent, then a second small kernel picks the best candidate for
    each target.
    """
    if wp is None:
        raise RuntimeError(f"Warp is not available: {_WARP_IMPORT_ERROR}")
    wp.init()
    if metric not in {"snr", "chi2"}:
        raise ValueError(f"unsupported grid metric: {metric}")
    if kernel_build_mode not in {"gpu_bilinear", "gpu_spline", "cpu_scipy"}:
        raise ValueError(f"unsupported kernel build mode: {kernel_build_mode}")

    x_pix = np.asarray(x_pix, dtype=float)
    y_pix = np.asarray(y_pix, dtype=float)
    if x_pix.shape != y_pix.shape:
        raise ValueError("x_pix and y_pix must have matching shape")
    n_targets = int(x_pix.size)
    if n_targets == 0:
        return WarpPsfBatchResult(rows=[], device=device)
    if var_ujy2 is None:
        var_ujy2 = np.full_like(flux_ujy, np.nanvar(flux_ujy[np.isfinite(flux_ujy)]), dtype=float)
    if flags is None:
        flags = np.zeros_like(flux_ujy, dtype=np.uint32)

    offsets = np.arange(-float(half_range_pix), float(half_range_pix) + 0.5 * float(step_pix), float(step_pix))
    xy_offsets = [(float(dx), float(dy)) for dy in offsets for dx in offsets]
    n_offsets = len(xy_offsets)
    n_candidates = n_targets * n_offsets

    candidate_x64 = np.empty(n_candidates, dtype=float)
    candidate_y64 = np.empty(n_candidates, dtype=float)
    candidate_x = np.empty(n_candidates, dtype=np.float32)
    candidate_y = np.empty(n_candidates, dtype=np.float32)

    for target_index in range(n_targets):
        for offset_index, (dx, dy) in enumerate(xy_offsets):
            cid = target_index * n_offsets + offset_index
            x_try = float(x_pix[target_index] + dx)
            y_try = float(y_pix[target_index] + dy)
            candidate_x64[cid] = x_try
            candidate_y64[cid] = y_try
            candidate_x[cid] = x_try
            candidate_y[cid] = y_try

    oversamp = int(hdul["PSF"].header.get("OVERSAMP", 10))
    kernel_size = int((2 * int(kernel_radius_native) * oversamp + 1) // oversamp)
    if kernel_size > 12:
        raise ValueError("Warp PSF benchmark expects kernel size <= 12")

    height, width = flux_ujy.shape
    bad_mask_value = sum(1 << int(bit) for bit in fatal_flag_bits)
    flux_dev = wp.array(np.ascontiguousarray(flux_ujy.astype(np.float32).ravel()), dtype=wp.float32, device=device)
    var_dev = wp.array(np.ascontiguousarray(np.nan_to_num(var_ujy2, nan=0.0).astype(np.float32).ravel()), dtype=wp.float32, device=device)
    flags_dev = wp.array(np.ascontiguousarray(np.asarray(flags, dtype=np.uint32).ravel()), dtype=wp.uint32, device=device)
    x_dev = wp.array(candidate_x, dtype=wp.float32, device=device)
    y_dev = wp.array(candidate_y, dtype=wp.float32, device=device)

    if kernel_build_mode in {"gpu_bilinear", "gpu_spline"}:
        psf_cube = np.asarray(hdul["PSF"].data, dtype=np.float32)
        if psf_cube.ndim != 3:
            raise ValueError(f"expected 3D PSF cube, got {psf_cube.shape}")
        n_planes, psf_height, psf_width = (int(psf_cube.shape[0]), int(psf_cube.shape[1]), int(psf_cube.shape[2]))
        ngrid = int(round(np.sqrt(n_planes)))
        if ngrid * ngrid != n_planes:
            ngrid = 11
        psf_source = np.nan_to_num(psf_cube, nan=0.0, posinf=0.0, neginf=0.0)
        sample_mode = 0
        if kernel_build_mode == "gpu_spline":
            coeff_cube = np.empty_like(psf_source, dtype=np.float32)
            for plane_index in range(n_planes):
                coeff_cube[plane_index] = spline_filter(psf_source[plane_index], order=3, output=np.float64, mode="constant").astype(np.float32)
            psf_source = coeff_cube
            sample_mode = 1
        psf_dev = wp.array(np.ascontiguousarray(psf_source.ravel()), dtype=wp.float32, device=device)
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
                int(oversamp),
                int(kernel_radius_native),
                int(kernel_size),
                float(shift_sign),
                int(sample_mode),
                x_dev,
                y_dev,
                kernels_dev,
                sectors_dev,
                kernel_sum_dev,
            ],
            device=device,
        )
    else:
        sectors = np.empty(n_candidates, dtype=np.int32)
        kernels: list[np.ndarray] = []
        for cid, (x_try, y_try) in enumerate(zip(candidate_x64, candidate_y64, strict=True)):
            if not (0 <= x_try < width and 0 <= y_try < height):
                sectors[cid] = -1
                kernels.append(np.zeros((kernel_size, kernel_size), dtype=np.float32))
                continue
            try:
                kernel = extract_native_psf_kernel(
                    hdul=hdul,
                    x_pix=float(x_try),
                    y_pix=float(y_try),
                    kernel_radius_native=kernel_radius_native,
                    shift_sign=shift_sign,
                )
            except Exception:
                sectors[cid] = -1
                kernels.append(np.zeros((kernel_size, kernel_size), dtype=np.float32))
                continue
            if kernel.data.shape != (kernel_size, kernel_size):
                raise ValueError(f"mixed PSF kernel shapes are not supported: {(kernel_size, kernel_size)} and {kernel.data.shape}")
            sectors[cid] = int(kernel.sector)
            kernels.append(kernel.data.astype(np.float32))
        packed = np.zeros((n_candidates, MAX_PSF_PIXELS), dtype=np.float32)
        kernel_sums = np.zeros(n_candidates, dtype=np.float32)
        for i, kernel in enumerate(kernels):
            block = np.zeros((12, 12), dtype=np.float32)
            block[:kernel_size, :kernel_size] = kernel[:kernel_size, :kernel_size]
            packed[i] = block.ravel()
            kernel_sums[i] = float(np.nansum(kernel))
        kernels_dev = wp.array(packed.ravel(), dtype=wp.float32, device=device)
        sectors_dev = wp.array(sectors, dtype=wp.int32, device=device)
        kernel_sum_dev = wp.array(kernel_sums, dtype=wp.float32, device=device)

    cand_flux = wp.empty(n_candidates, dtype=wp.float32, device=device)
    cand_unc = wp.empty(n_candidates, dtype=wp.float32, device=device)
    cand_bkg = wp.empty(n_candidates, dtype=wp.float32, device=device)
    cand_chi2 = wp.empty(n_candidates, dtype=wp.float32, device=device)
    cand_dof = wp.empty(n_candidates, dtype=wp.int32, device=device)
    cand_valid = wp.empty(n_candidates, dtype=wp.int32, device=device)
    cand_status = wp.empty(n_candidates, dtype=wp.int32, device=device)
    wp.launch(
        _psf_grid_candidate_kernel,
        dim=n_candidates,
        inputs=[
            flux_dev,
            var_dev,
            flags_dev,
            kernels_dev,
            int(width),
            int(height),
            int(kernel_size),
            x_dev,
            y_dev,
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

    flux = cand_flux.numpy().astype(float)
    unc = cand_unc.numpy().astype(float)
    bkg = cand_bkg.numpy().astype(float)
    chi2 = cand_chi2.numpy().astype(float)
    dof = cand_dof.numpy().astype(int)
    valid = cand_valid.numpy().astype(int)
    status = cand_status.numpy().astype(int)
    best_index = best_index_dev.numpy().astype(int)
    grid_valid = grid_valid_dev.numpy().astype(int)
    best_score = best_score_dev.numpy().astype(float)
    sectors = sectors_dev.numpy().astype(int)
    kernel_sums = kernel_sum_dev.numpy().astype(float)

    rows = []
    for i in range(n_targets):
        cid = int(best_index[i])
        if cid < 0:
            rows.append(
                _with_grid_metadata(
                    _bad("no_valid_grid_trials", x_pix[i], y_pix[i], x_pix[i], y_pix[i], None, f"warp_psf_grid:{kernel_build_mode}"),
                    "no_valid_trials",
                    metric,
                    n_offsets,
                    int(grid_valid[i]),
                    np.nan,
                )
            )
            continue
        reduced = chi2[cid] / dof[cid] if dof[cid] > 0 else np.nan
        row = PsfFitResult(
            flux_uJy=float(flux[cid]) if status[cid] == 0 else np.nan,
            flux_unc_uJy=float(unc[cid]) if status[cid] == 0 else np.nan,
            background_uJy_per_pix=float(bkg[cid]) if status[cid] == 0 else np.nan,
            chi2=float(chi2[cid]) if status[cid] == 0 else np.nan,
            reduced_chi2=float(reduced) if np.isfinite(reduced) else np.nan,
            dof=int(dof[cid]) if status[cid] == 0 else 0,
            n_valid=int(valid[cid]),
            status="ok" if status[cid] == 0 else "bad_fit",
            sector=int(sectors[cid]) if sectors[cid] >= 0 else None,
            x_fit=float(candidate_x64[cid]),
            y_fit=float(candidate_y64[cid]),
            x_init=float(x_pix[i]),
            y_init=float(y_pix[i]),
            fit_offset_pix=float(np.hypot(float(candidate_x64[cid]) - float(x_pix[i]), float(candidate_y64[cid]) - float(y_pix[i]))),
            kernel_sum=float(kernel_sums[cid]) if np.isfinite(kernel_sums[cid]) else np.nan,
            kernel_shape_y=int(kernel_size),
            kernel_shape_x=int(kernel_size),
            grid_status="ok",
            grid_metric=metric,
            grid_n_trials=int(n_offsets),
            grid_n_valid=int(grid_valid[i]),
            grid_best_score=float(best_score[i]) if np.isfinite(best_score[i]) else np.nan,
            backend=f"warp_psf_grid:{kernel_build_mode}",
        )
        rows.append(row)
    return WarpPsfBatchResult(rows=rows, device=device)


def _normalize_positive(values: np.ndarray, label: str) -> np.ndarray:
    total = float(np.nansum(values))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{label} has non-positive sum")
    return np.asarray(values, dtype=float) / total


def _with_grid_metadata(
    result: PsfFitResult,
    grid_status: str,
    metric: str,
    n_trials: int,
    n_valid: int,
    best_score: float,
) -> PsfFitResult:
    values = result.to_dict()
    values.update(
        {
            "grid_status": grid_status,
            "grid_metric": metric,
            "grid_n_trials": int(n_trials),
            "grid_n_valid": int(n_valid),
            "grid_best_score": float(best_score) if np.isfinite(best_score) else np.nan,
        }
    )
    return PsfFitResult(**values)


def _bad(
    status: str,
    x_fit: float,
    y_fit: float,
    x_init: float,
    y_init: float,
    sector: int | None,
    backend: str,
    *,
    n_valid: int = 0,
) -> PsfFitResult:
    return PsfFitResult(
        flux_uJy=np.nan,
        flux_unc_uJy=np.nan,
        background_uJy_per_pix=np.nan,
        chi2=np.nan,
        reduced_chi2=np.nan,
        dof=0,
        n_valid=int(n_valid),
        status=status,
        sector=sector,
        x_fit=float(x_fit),
        y_fit=float(y_fit),
        x_init=float(x_init),
        y_init=float(y_init),
        fit_offset_pix=float(np.hypot(float(x_fit) - float(x_init), float(y_fit) - float(y_init))),
        kernel_sum=np.nan,
        kernel_shape_y=0,
        kernel_shape_x=0,
        grid_status="disabled",
        grid_metric="none",
        grid_n_trials=0,
        grid_n_valid=0,
        grid_best_score=np.nan,
        backend=backend,
    )


def select_warp_device(devices: tuple[str, ...], worker_name: str = "worker_0") -> str:
    if not devices:
        return "cuda:0"
    match = re.search(r"_(\d+)$", worker_name)
    if match is None:
        return devices[0]
    return devices[int(match.group(1)) % len(devices)]
