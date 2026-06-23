from __future__ import annotations

from pathlib import Path

import pandas as pd


SCIENCE_WAVELENGTH_SOURCE = "spectral_wcs_CWAVE_CBAND"


def assert_science_wavelengths(df: pd.DataFrame, source_path: Path | str, *, allow_approx: bool = False) -> None:
    if allow_approx:
        return
    path = Path(source_path)
    if "wavelength_source" not in df.columns:
        raise ValueError(
            f"{path} has no wavelength_source column. Rebuild spectra from measurements generated with "
            f"{SCIENCE_WAVELENGTH_SOURCE}."
        )
    sources = set(df["wavelength_source"].dropna().astype(str).unique())
    if sources != {SCIENCE_WAVELENGTH_SOURCE}:
        raise ValueError(
            f"{path} uses wavelength_source={sorted(sources)}. Science/injection paths require "
            f"{SCIENCE_WAVELENGTH_SOURCE}; rerun photometry or pass an explicit approximate-wavelength override."
        )
    required = ["wavelength_calibration_file", "wavelength_calibration_collection", "wavelength_detector"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing wavelength provenance columns: {', '.join(missing)}")
