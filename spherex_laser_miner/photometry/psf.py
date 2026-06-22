from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import shift as ndi_shift


@dataclass(frozen=True)
class PsfMeasurement:
    psf_flux_uJy: float
    psf_flux_unc_uJy: float
    psf_flux_unit: str
    psf_fit_status: str
    psf_chi2: float
    psf_dof: int
    psf_n_valid: int
    psf_model_id: str
    psf_sector: int | None
    psf_background_uJy_per_pix: float
    centroid_dx_pix: float
    centroid_dy_pix: float

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def psf_measure(
    flux_ujy: np.ndarray,
    var_ujy2: np.ndarray | None,
    flags: np.ndarray | None,
    psf_cube: np.ndarray | None,
    x_pix: float,
    y_pix: float,
    fatal_flag_bits: tuple[int, ...],
    kernel_radius_native: int = 5,
    fit_background_gradient: bool = False,
) -> PsfMeasurement:
    if psf_cube is None:
        return _bad("missing_psf")
    try:
        psf_kernel, sector = _extract_native_psf(psf_cube, flux_ujy.shape, x_pix, y_pix, kernel_radius_native)
        return _fit_psf(
            flux_ujy=flux_ujy,
            var_ujy2=var_ujy2,
            flags=flags,
            psf_kernel=psf_kernel,
            x_pix=x_pix,
            y_pix=y_pix,
            fatal_flag_bits=fatal_flag_bits,
            fit_background_gradient=fit_background_gradient,
            sector=sector,
        )
    except Exception as exc:
        out = _bad(f"failed:{type(exc).__name__}")
        return out


def psf_not_run(status: str = "disabled") -> PsfMeasurement:
    return _bad(status)


def _extract_native_psf(
    psf_cube: np.ndarray,
    image_shape: tuple[int, int],
    x_pix: float,
    y_pix: float,
    kernel_radius_native: int,
) -> tuple[np.ndarray, int]:
    cube = np.asarray(psf_cube, dtype=float)
    if cube.ndim != 3:
        raise ValueError("PSF cube is not 3D")
    nplanes = cube.shape[0]
    ngrid = int(round(np.sqrt(nplanes)))
    if ngrid * ngrid != nplanes:
        ngrid = 11
    ny, nx = image_shape
    zone_x = min(ngrid - 1, max(0, int(np.floor(x_pix / nx * ngrid))))
    zone_y = min(ngrid - 1, max(0, int(np.floor(y_pix / ny * ngrid))))
    # QR2 erratum says planes are x-fast in stored order.
    sector = zone_y * ngrid + zone_x
    sector = min(nplanes - 1, max(0, sector))
    psf_super = np.nan_to_num(cube[sector], nan=0.0, posinf=0.0, neginf=0.0)
    psf_super = np.clip(psf_super, 0.0, None)
    total = float(np.sum(psf_super))
    if total <= 0 or not np.isfinite(total):
        raise ValueError("non-positive PSF plane")
    psf_super = psf_super / total

    oversamp = max(1, int(round(psf_super.shape[0] / 10)))
    shift_y = (y_pix - np.floor(y_pix)) * oversamp
    shift_x = (x_pix - np.floor(x_pix)) * oversamp
    psf_super = ndi_shift(psf_super, shift=(shift_y, shift_x), order=3, mode="constant", cval=0.0, prefilter=True)
    psf_super = np.clip(np.nan_to_num(psf_super, nan=0.0), 0.0, None)
    total = float(np.sum(psf_super))
    if total <= 0 or not np.isfinite(total):
        raise ValueError("non-positive shifted PSF")
    psf_super = psf_super / total

    cy = psf_super.shape[0] // 2
    cx = psf_super.shape[1] // 2
    r = int(kernel_radius_native * oversamp)
    cut = psf_super[max(0, cy - r) : min(psf_super.shape[0], cy + r + 1), max(0, cx - r) : min(psf_super.shape[1], cx + r + 1)]
    size = min((cut.shape[0] // oversamp) * oversamp, (cut.shape[1] // oversamp) * oversamp)
    if size < oversamp:
        raise ValueError("PSF crop too small")
    cut = cut[:size, :size]
    n = size // oversamp
    native = cut.reshape(n, oversamp, n, oversamp).sum(axis=(1, 3))
    native_sum = float(np.sum(native))
    if native_sum <= 0 or not np.isfinite(native_sum):
        raise ValueError("non-positive native PSF")
    return native / native_sum, sector


def _fit_psf(
    flux_ujy: np.ndarray,
    var_ujy2: np.ndarray | None,
    flags: np.ndarray | None,
    psf_kernel: np.ndarray,
    x_pix: float,
    y_pix: float,
    fatal_flag_bits: tuple[int, ...],
    fit_background_gradient: bool,
    sector: int,
) -> PsfMeasurement:
    ph, pw = psf_kernel.shape
    x0 = int(round(x_pix))
    y0 = int(round(y_pix))
    lx = pw // 2
    ly = ph // 2
    x1 = max(0, x0 - lx)
    x2 = min(flux_ujy.shape[1], x0 + (pw - lx))
    y1 = max(0, y0 - ly)
    y2 = min(flux_ujy.shape[0], y0 + (ph - ly))
    cut = flux_ujy[y1:y2, x1:x2].astype(float)
    px1 = lx - (x0 - x1)
    py1 = ly - (y0 - y1)
    psf_cut = psf_kernel[py1 : py1 + cut.shape[0], px1 : px1 + cut.shape[1]]
    if cut.shape != psf_cut.shape or cut.size == 0:
        return _bad("shape_mismatch")

    if var_ujy2 is not None:
        var_cut = var_ujy2[y1:y2, x1:x2].astype(float)
    else:
        var_cut = np.full(cut.shape, np.nanvar(cut[np.isfinite(cut)]), dtype=float)
    valid = np.isfinite(cut) & np.isfinite(psf_cut) & np.isfinite(var_cut) & (var_cut > 0)
    if flags is not None:
        flag_cut = flags[y1:y2, x1:x2]
        fatal_mask = sum(1 << int(bit) for bit in fatal_flag_bits)
        valid &= (flag_cut & fatal_mask) == 0
    n_valid = int(np.count_nonzero(valid))
    n_params = 4 if fit_background_gradient else 2
    if n_valid <= n_params:
        return _bad("too_few_valid_pixels", n_valid=n_valid, sector=sector)

    yy, xx = np.indices(cut.shape)
    cols = [psf_cut[valid], np.ones(n_valid)]
    if fit_background_gradient:
        cols.extend([(xx[valid] - (x_pix - x1)), (yy[valid] - (y_pix - y1))])
    a = np.vstack(cols).T
    y = cut[valid]
    w = 1.0 / var_cut[valid]
    aw = a * np.sqrt(w)[:, None]
    yw = y * np.sqrt(w)
    try:
        normal = aw.T @ aw
        cov = np.linalg.inv(normal)
        pars = cov @ (aw.T @ yw)
    except np.linalg.LinAlgError:
        return _bad("singular_fit", n_valid=n_valid, sector=sector)
    model = a @ pars
    chi2 = float(np.sum((y - model) ** 2 * w))
    dof = int(n_valid - len(pars))
    unc = float(np.sqrt(cov[0, 0])) if cov[0, 0] > 0 else float("nan")
    return PsfMeasurement(
        psf_flux_uJy=float(pars[0]),
        psf_flux_unc_uJy=unc,
        psf_flux_unit="uJy",
        psf_fit_status="ok",
        psf_chi2=chi2,
        psf_dof=dof,
        psf_n_valid=n_valid,
        psf_model_id="mef_psf_cube_native_linear_v1",
        psf_sector=int(sector),
        psf_background_uJy_per_pix=float(pars[1]),
        centroid_dx_pix=0.0,
        centroid_dy_pix=0.0,
    )


def _bad(status: str, n_valid: int = 0, sector: int | None = None) -> PsfMeasurement:
    return PsfMeasurement(
        psf_flux_uJy=float("nan"),
        psf_flux_unc_uJy=float("nan"),
        psf_flux_unit="uJy",
        psf_fit_status=status,
        psf_chi2=float("nan"),
        psf_dof=0,
        psf_n_valid=n_valid,
        psf_model_id="mef_psf_cube_native_linear_v1",
        psf_sector=sector,
        psf_background_uJy_per_pix=float("nan"),
        centroid_dx_pix=float("nan"),
        centroid_dy_pix=float("nan"),
    )
