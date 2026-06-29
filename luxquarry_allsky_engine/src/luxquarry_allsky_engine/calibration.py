from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.units import Unit


SAPM_COLLECTION = "cal-sapm-v2-2025-164"
SPECTRAL_WCS_COLLECTION = "cal-wcs-v4-2025-254"


@lru_cache(maxsize=16)
def load_sapm(cache_root: str, release: str, detector: int) -> tuple[np.ndarray, fits.Header, str]:
    path = sapm_cache_path(Path(cache_root), release, detector)
    if not path.exists():
        raise FileNotFoundError(f"Missing solid-angle pixel map: {path}")
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul["IMAGE"] if "IMAGE" in hdul else hdul[0]
        return np.asarray(hdu.data, dtype=float), hdu.header.copy(), str(path)


@lru_cache(maxsize=16)
def load_spectral_wcs_maps(cache_root: str, release: str, detector: int) -> tuple[np.ndarray, np.ndarray, str]:
    path = spectral_wcs_cache_path(Path(cache_root), release, detector)
    if not path.exists():
        raise FileNotFoundError(f"Missing spectral WCS map: {path}")
    with fits.open(path, memmap=True) as hdul:
        return (
            np.asarray(hdul["CWAVE"].data, dtype=float),
            np.asarray(hdul["CBAND"].data, dtype=float),
            str(path),
        )


def sapm_cache_path(cache_root: Path, release: str, detector: int) -> Path:
    filename = f"solid_angle_pixel_map_D{int(detector)}_spx_{SAPM_COLLECTION}.fits"
    return cache_root / "calibration" / release / "solid_angle_pixel_map" / SAPM_COLLECTION / str(int(detector)) / filename


def spectral_wcs_cache_path(cache_root: Path, release: str, detector: int) -> Path:
    filename = f"spectral_wcs_D{int(detector)}_spx_{SPECTRAL_WCS_COLLECTION}.fits"
    return cache_root / "calibration" / release / "spectral_wcs" / SPECTRAL_WCS_COLLECTION / str(int(detector)) / filename


def image_to_ujy_per_pixel(
    image_data: np.ndarray,
    image_header: fits.Header,
    sapm_data: np.ndarray,
    sapm_header: fits.Header,
) -> np.ndarray:
    bunit = Unit(image_header.get("BUNIT", "MJy/sr"))
    sapm_unit = Unit(sapm_header.get("BUNIT", "arcsec2"))
    return ((np.asarray(image_data, dtype=float) * bunit).to(u.uJy / u.arcsec**2) * (sapm_data * sapm_unit)).value


def image_to_ujy_arcsec2_scale(image_header: fits.Header) -> float:
    bunit = Unit(image_header.get("BUNIT", "MJy/sr"))
    return float((1.0 * bunit).to(u.uJy / u.arcsec**2).value)


def variance_to_ujy2(
    variance_data: np.ndarray,
    image_header: fits.Header,
    sapm_data: np.ndarray,
    sapm_header: fits.Header,
) -> np.ndarray:
    bunit = Unit(image_header.get("BUNIT", "MJy/sr"))
    sapm_unit = Unit(sapm_header.get("BUNIT", "arcsec2"))
    scale = ((1.0 * bunit).to(u.uJy / u.arcsec**2) * (sapm_data * sapm_unit)).value
    return np.asarray(variance_data, dtype=float) * scale**2


def bilinear_sample(image: np.ndarray, x_zero_based: np.ndarray, y_zero_based: np.ndarray) -> np.ndarray:
    data = np.asarray(image, dtype=float)
    x = np.asarray(x_zero_based, dtype=float)
    y = np.asarray(y_zero_based, dtype=float)
    out = np.full(np.broadcast_shapes(x.shape, y.shape), np.nan, dtype=float)
    x = np.broadcast_to(x, out.shape)
    y = np.broadcast_to(y, out.shape)
    height, width = data.shape
    good = np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (y >= 0.0) & (x <= width - 1) & (y <= height - 1)
    if not np.any(good):
        return out
    x0 = np.floor(x[good]).astype(int)
    y0 = np.floor(y[good]).astype(int)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    dx = x[good] - x0
    dy = y[good] - y0
    v00 = data[y0, x0]
    v10 = data[y0, x1]
    v01 = data[y1, x0]
    v11 = data[y1, x1]
    out[good] = (
        v00 * (1.0 - dx) * (1.0 - dy)
        + v10 * dx * (1.0 - dy)
        + v01 * (1.0 - dx) * dy
        + v11 * dx * dy
    )
    return out
