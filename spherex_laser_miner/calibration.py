from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.units import Unit

from spherex_laser_miner.downloader import download_file


SAPM_COLLECTION = "cal-sapm-v2-2025-164"


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
