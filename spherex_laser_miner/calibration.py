from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.units import Unit

from spherex_laser_miner.downloader import download_file


SAPM_COLLECTION = "cal-sapm-v2-2025-164"
SPECTRAL_WCS_COLLECTION = "cal-wcs-v4-2025-254"


def sapm_url(release: str, detector: int, collection: str = SAPM_COLLECTION) -> str:
    dataset = "solid_angle_pixel_map"
    filename = f"{dataset}_D{int(detector)}_spx_{collection}.fits"
    return (
        "https://irsa.ipac.caltech.edu/ibe/data/spherex/"
        f"{release}/{dataset}/{collection}/{int(detector)}/{filename}"
    )


def sapm_cache_path(cache_root: Path, release: str, detector: int, collection: str = SAPM_COLLECTION) -> Path:
    filename = f"solid_angle_pixel_map_D{int(detector)}_spx_{collection}.fits"
    return cache_root / "calibration" / release / "solid_angle_pixel_map" / collection / str(int(detector)) / filename


def load_sapm(cache_root: Path, release: str, detector: int) -> tuple[np.ndarray, fits.Header, Path]:
    path = sapm_cache_path(cache_root, release, detector)
    download_file(sapm_url(release, detector), path, redownload=False)
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul["IMAGE"] if "IMAGE" in hdul else hdul[0]
        return np.asarray(hdu.data, dtype=float), hdu.header.copy(), path


def spectral_wcs_url(release: str, detector: int, collection: str = SPECTRAL_WCS_COLLECTION) -> str:
    dataset = "spectral_wcs"
    filename = f"{dataset}_D{int(detector)}_spx_{collection}.fits"
    return (
        "https://irsa.ipac.caltech.edu/ibe/data/spherex/"
        f"{release}/{dataset}/{collection}/{int(detector)}/{filename}"
    )


def spectral_wcs_cache_path(
    cache_root: Path,
    release: str,
    detector: int,
    collection: str = SPECTRAL_WCS_COLLECTION,
) -> Path:
    filename = f"spectral_wcs_D{int(detector)}_spx_{collection}.fits"
    return cache_root / "calibration" / release / "spectral_wcs" / collection / str(int(detector)) / filename


def load_spectral_wcs_maps(cache_root: Path, release: str, detector: int) -> tuple[np.ndarray, np.ndarray, fits.Header, Path]:
    path = spectral_wcs_cache_path(cache_root, release, detector)
    download_file(spectral_wcs_url(release, detector), path, redownload=False)
    with fits.open(path, memmap=True) as hdul:
        cwave = np.asarray(hdul["CWAVE"].data, dtype=float)
        cband = np.asarray(hdul["CBAND"].data, dtype=float)
        return cwave, cband, hdul["CWAVE"].header.copy(), path


def sample_wavelength_maps(
    cwave_um: np.ndarray,
    cband_um: np.ndarray,
    x_pix: np.ndarray,
    y_pix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return _bilinear_sample(cwave_um, x_pix, y_pix), _bilinear_sample(cband_um, x_pix, y_pix)


def _bilinear_sample(image: np.ndarray, x_pix: np.ndarray, y_pix: np.ndarray) -> np.ndarray:
    data = np.asarray(image, dtype=float)
    x = np.asarray(x_pix, dtype=float)
    y = np.asarray(y_pix, dtype=float)
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


def image_to_ujy_per_pixel(
    image_data: np.ndarray,
    image_header: fits.Header,
    sapm_data: np.ndarray,
    sapm_header: fits.Header,
) -> np.ndarray:
    bunit = Unit(image_header.get("BUNIT", "MJy/sr"))
    sapm_unit = Unit(sapm_header.get("BUNIT", "arcsec2"))
    return ((np.asarray(image_data, dtype=float) * bunit).to(u.uJy / u.arcsec**2) * (sapm_data * sapm_unit)).value


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
