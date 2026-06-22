from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture, aperture_photometry


@dataclass(frozen=True)
class ApertureMeasurement:
    aperture_flux: float
    aperture_flux_unc: float
    aperture_flux_unit: str
    background: float
    background_unc: float
    variance_model_unc: float
    empirical_background_unc: float
    n_aperture_pixels: int
    n_good_aperture_pixels: int
    n_bad_aperture_pixels: int
    bad_pixel_fraction_aperture: float
    bad_pixel_fraction_annulus: float
    flags_summary: int
    fatal_flag_present: bool
    image_value_raw: float
    zodi_model_at_target: float | None
    zodi_aperture_mean: float | None
    zodi_annulus_mean: float | None
    aperture_status: str

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def aperture_measure(
    image: np.ndarray,
    variance: np.ndarray | None,
    flags: np.ndarray | None,
    zodi: np.ndarray | None,
    x_pix: float,
    y_pix: float,
    aperture_radius_pix: float,
    annulus_inner_pix: float,
    annulus_outer_pix: float,
    fatal_flag_bits: tuple[int, ...],
) -> ApertureMeasurement:
    bitmask = _bitmask(fatal_flag_bits)
    image_mask = ~np.isfinite(image)
    if flags is not None:
        image_mask |= (flags.astype(np.int64) & bitmask) != 0

    aperture = CircularAperture((x_pix, y_pix), r=aperture_radius_pix)
    annulus = CircularAnnulus((x_pix, y_pix), r_in=annulus_inner_pix, r_out=annulus_outer_pix)

    aperture_mask = aperture.to_mask(method="center").to_image(image.shape).astype(bool)
    annulus_mask = annulus.to_mask(method="center").to_image(image.shape).astype(bool)
    good_aperture = aperture_mask & ~image_mask
    good_annulus = annulus_mask & ~image_mask

    n_ap = int(aperture_mask.sum())
    n_good_ap = int(good_aperture.sum())
    n_bad_ap = n_ap - n_good_ap
    n_ann = int(annulus_mask.sum())
    n_good_ann = int(good_annulus.sum())

    if n_good_ap == 0 or n_good_ann < 3:
        return ApertureMeasurement(
            aperture_flux=float("nan"),
            aperture_flux_unc=float("nan"),
            aperture_flux_unit="MJy/sr*pix_uncalibrated",
            background=float("nan"),
            background_unc=float("nan"),
            variance_model_unc=float("nan"),
            empirical_background_unc=float("nan"),
            n_aperture_pixels=n_ap,
            n_good_aperture_pixels=n_good_ap,
            n_bad_aperture_pixels=n_bad_ap,
            bad_pixel_fraction_aperture=_fraction(n_bad_ap, n_ap),
            bad_pixel_fraction_annulus=_fraction(n_ann - n_good_ann, n_ann),
            flags_summary=_flags_summary(flags, aperture_mask),
            fatal_flag_present=n_bad_ap > 0,
            image_value_raw=_sample(image, x_pix, y_pix),
            zodi_model_at_target=_sample_or_none(zodi, x_pix, y_pix),
            zodi_aperture_mean=_masked_mean_or_none(zodi, good_aperture),
            zodi_annulus_mean=_masked_mean_or_none(zodi, good_annulus),
            aperture_status="insufficient_good_pixels",
        )

    annulus_values = image[good_annulus]
    _, background, background_std = sigma_clipped_stats(annulus_values, sigma=3.0, maxiters=5)
    phot = aperture_photometry(image, aperture, mask=image_mask, method="center")
    raw_sum = float(phot["aperture_sum"][0])
    aperture_flux = raw_sum - float(background) * n_good_ap

    variance_unc = float("nan")
    if variance is not None:
        variance_good = np.where(np.isfinite(variance) & good_aperture, variance, 0.0)
        variance_unc = float(np.sqrt(np.sum(variance_good)))
    empirical_unc = float(background_std * np.sqrt(n_good_ap))
    if np.isfinite(variance_unc):
        flux_unc = float(np.sqrt(variance_unc**2 + empirical_unc**2))
    else:
        flux_unc = empirical_unc

    return ApertureMeasurement(
        aperture_flux=float(aperture_flux),
        aperture_flux_unc=flux_unc,
        aperture_flux_unit="MJy/sr*pix_uncalibrated",
        background=float(background),
        background_unc=float(background_std),
        variance_model_unc=variance_unc,
        empirical_background_unc=empirical_unc,
        n_aperture_pixels=n_ap,
        n_good_aperture_pixels=n_good_ap,
        n_bad_aperture_pixels=n_bad_ap,
        bad_pixel_fraction_aperture=_fraction(n_bad_ap, n_ap),
        bad_pixel_fraction_annulus=_fraction(n_ann - n_good_ann, n_ann),
        flags_summary=_flags_summary(flags, aperture_mask),
        fatal_flag_present=n_bad_ap > 0,
        image_value_raw=_sample(image, x_pix, y_pix),
        zodi_model_at_target=_sample_or_none(zodi, x_pix, y_pix),
        zodi_aperture_mean=_masked_mean_or_none(zodi, good_aperture),
        zodi_annulus_mean=_masked_mean_or_none(zodi, good_annulus),
        aperture_status="ok",
    )


def _bitmask(bits: tuple[int, ...]) -> int:
    mask = 0
    for bit in bits:
        mask |= 1 << int(bit)
    return mask


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _sample(data: np.ndarray, x_pix: float, y_pix: float) -> float:
    y = int(round(y_pix))
    x = int(round(x_pix))
    if y < 0 or x < 0 or y >= data.shape[0] or x >= data.shape[1]:
        return float("nan")
    return float(data[y, x])


def _sample_or_none(data: np.ndarray | None, x_pix: float, y_pix: float) -> float | None:
    if data is None:
        return None
    return _sample(data, x_pix, y_pix)


def _masked_mean_or_none(data: np.ndarray | None, mask: np.ndarray) -> float | None:
    if data is None:
        return None
    values = data[mask & np.isfinite(data)]
    if len(values) == 0:
        return None
    return float(np.mean(values))


def _flags_summary(flags: np.ndarray | None, mask: np.ndarray) -> int:
    if flags is None:
        return 0
    values = flags[mask]
    summary = 0
    for value in values:
        summary |= int(value)
    return summary
