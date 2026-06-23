from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from astropy.stats import SigmaClip
from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture, aperture_photometry


@dataclass(frozen=True)
class CalibratedApertureMeasurement:
    aperture_flux_uJy: float
    aperture_flux_unc_uJy: float
    aperture_flux_unit_calibrated: str
    aperture_area_pix_exact: float
    background_uJy_per_pix: float
    background_unc_uJy_per_pix: float
    n_bad_aperture_pixels_calibrated: int
    flags_summary: int
    calibrated_aperture_status: str

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def calibrated_aperture_measure(
    flux_ujy: np.ndarray,
    var_ujy2: np.ndarray | None,
    flags: np.ndarray | None,
    x_pix: float,
    y_pix: float,
    aperture_radius_pix: float,
    annulus_inner_pix: float,
    annulus_outer_pix: float,
    fatal_flag_bits: tuple[int, ...],
) -> CalibratedApertureMeasurement:
    if not np.isfinite(x_pix) or not np.isfinite(y_pix):
        return _bad("bad_position")
    ny, nx = flux_ujy.shape
    if x_pix < 0 or y_pix < 0 or x_pix >= nx or y_pix >= ny:
        return _bad("outside_image")

    aperture = CircularAperture((x_pix, y_pix), r=float(aperture_radius_pix))
    annulus = CircularAnnulus((x_pix, y_pix), r_in=float(annulus_inner_pix), r_out=float(annulus_outer_pix))
    bad_mask = make_bad_pixel_mask(flags, flux_ujy, fatal_flag_bits)

    sigclip = SigmaClip(sigma=3.0, maxiters=5)
    stats = ApertureStats(flux_ujy, annulus, mask=bad_mask, sigma_clip=sigclip)
    bkg_per_pix = stats.median
    if bkg_per_pix is None or not np.isfinite(bkg_per_pix):
        return _bad("bad_background")

    ap_weight = aperture.to_mask(method="exact").to_image(flux_ujy.shape)
    if ap_weight is None:
        return _bad("bad_aperture_mask")
    valid_weight = np.where(bad_mask, 0.0, ap_weight)
    ap_area = float(np.nansum(valid_weight))
    if not np.isfinite(ap_area) or ap_area <= 0:
        return _bad("no_good_aperture_area")

    phot = aperture_photometry(flux_ujy, aperture, mask=bad_mask, method="exact")
    flux = float(phot["aperture_sum"][0]) - float(bkg_per_pix) * ap_area

    if var_ujy2 is not None:
        var_mask = bad_mask | ~np.isfinite(var_ujy2)
        vtab = aperture_photometry(var_ujy2, aperture, mask=var_mask, method="exact")
        var_sum = float(vtab["aperture_sum"][0])
    else:
        var_sum = 0.0

    ann_mask_img = annulus.to_mask(method="center").to_image(flux_ujy.shape)
    n_bg = int(np.count_nonzero((ann_mask_img > 0) & (~bad_mask))) if ann_mask_img is not None else 0
    stats_std = float(stats.std) if stats.std is not None and np.isfinite(stats.std) else 0.0
    var_bkg_per_pix = stats_std**2 / n_bg if n_bg > 0 else 0.0
    if var_ujy2 is None and stats_std > 0:
        var_sum = stats_std**2 * ap_area
    total_var = var_sum + ap_area**2 * var_bkg_per_pix
    err = float(np.sqrt(total_var)) if total_var >= 0 else float("nan")

    return CalibratedApertureMeasurement(
        aperture_flux_uJy=float(flux),
        aperture_flux_unc_uJy=err,
        aperture_flux_unit_calibrated="uJy",
        aperture_area_pix_exact=ap_area,
        background_uJy_per_pix=float(bkg_per_pix),
        background_unc_uJy_per_pix=stats_std,
        n_bad_aperture_pixels_calibrated=count_bad_aperture_flags(
            flags, x_pix, y_pix, aperture_radius_pix, fatal_flag_bits
        ),
        flags_summary=summarize_aperture_flags(flags, x_pix, y_pix, aperture_radius_pix),
        calibrated_aperture_status="ok",
    )


def make_bad_pixel_mask(flags: np.ndarray | None, flux: np.ndarray, bad_flag_bits: Iterable[int]) -> np.ndarray:
    bad_mask = ~np.isfinite(flux)
    if flags is not None:
        mask_value = sum(1 << int(bit) for bit in bad_flag_bits)
        bad_mask |= (np.asarray(flags) & mask_value) != 0
    return bad_mask


def count_bad_aperture_flags(
    flags: np.ndarray | None,
    x_pix: float,
    y_pix: float,
    radius_pix: float,
    flag_bits: Iterable[int],
) -> int:
    if flags is None:
        return 0
    aperture = CircularAperture((x_pix, y_pix), r=float(radius_pix))
    ap_mask = aperture.to_mask(method="center").to_image(flags.shape)
    if ap_mask is None:
        return 0
    ap_flags = np.asarray(flags)[ap_mask > 0]
    mask_value = sum(1 << int(bit) for bit in flag_bits)
    return int(np.count_nonzero((ap_flags & mask_value) != 0))


def summarize_aperture_flags(
    flags: np.ndarray | None,
    x_pix: float,
    y_pix: float,
    radius_pix: float,
) -> int:
    if flags is None:
        return 0
    aperture = CircularAperture((x_pix, y_pix), r=float(radius_pix))
    ap_mask = aperture.to_mask(method="center").to_image(flags.shape)
    if ap_mask is None:
        return 0
    summary = 0
    for value in np.asarray(flags)[ap_mask > 0]:
        summary |= int(value)
    return int(summary)


def _bad(status: str) -> CalibratedApertureMeasurement:
    return CalibratedApertureMeasurement(
        aperture_flux_uJy=float("nan"),
        aperture_flux_unc_uJy=float("nan"),
        aperture_flux_unit_calibrated="uJy",
        aperture_area_pix_exact=float("nan"),
        background_uJy_per_pix=float("nan"),
        background_unc_uJy_per_pix=float("nan"),
        n_bad_aperture_pixels_calibrated=0,
        flags_summary=0,
        calibrated_aperture_status=status,
    )
