from __future__ import annotations

import warnings

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from erfa import ErfaWarning
import numpy as np

from spherex_laser_miner.catalog.manual_targets import ManualTarget


def propagate_target(target: ManualTarget, obs_mid_mjd: float | None) -> tuple[float, float, str]:
    if obs_mid_mjd is None:
        return target.ra_deg, target.dec_deg, "no_observation_epoch"

    pmra = 0.0 if target.pmra_masyr is None else target.pmra_masyr
    pmdec = 0.0 if target.pmdec_masyr is None else target.pmdec_masyr
    c0 = SkyCoord(
        ra=target.ra_deg * u.deg,
        dec=target.dec_deg * u.deg,
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        obstime=Time(target.reference_epoch_yr, format="jyear"),
        frame="icrs",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErfaWarning)
        c_obs = c0.apply_space_motion(new_obstime=Time(float(obs_mid_mjd), format="mjd"))
    return float(c_obs.ra.deg), float(c_obs.dec.deg), "astropy_space_motion"


def propagate_coordinate(
    ra_deg: float,
    dec_deg: float,
    reference_epoch_yr: float | None,
    pmra_masyr: float | None,
    pmdec_masyr: float | None,
    obs_mid_mjd: float | None,
) -> tuple[float, float, str]:
    if obs_mid_mjd is None or reference_epoch_yr is None:
        return float(ra_deg), float(dec_deg), "no_observation_or_reference_epoch"

    pmra = 0.0 if pmra_masyr is None else float(pmra_masyr)
    pmdec = 0.0 if pmdec_masyr is None else float(pmdec_masyr)
    c0 = SkyCoord(
        ra=float(ra_deg) * u.deg,
        dec=float(dec_deg) * u.deg,
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        obstime=Time(float(reference_epoch_yr), format="jyear"),
        frame="icrs",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErfaWarning)
        c_obs = c0.apply_space_motion(new_obstime=Time(float(obs_mid_mjd), format="mjd"))
    return float(c_obs.ra.deg), float(c_obs.dec.deg), "astropy_space_motion"


def propagate_coordinates(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    reference_epoch_yr: np.ndarray,
    pmra_masyr: np.ndarray,
    pmdec_masyr: np.ndarray,
    obs_mid_mjd: float | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Propagate arrays of ICRS coordinates to one observation epoch."""
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    ref_epoch = np.asarray(reference_epoch_yr, dtype=float)
    pmra = np.nan_to_num(np.asarray(pmra_masyr, dtype=float), nan=0.0)
    pmdec = np.nan_to_num(np.asarray(pmdec_masyr, dtype=float), nan=0.0)
    out_ra = ra.copy()
    out_dec = dec.copy()
    statuses = ["no_observation_or_reference_epoch"] * len(ra)
    if obs_mid_mjd is None or len(ra) == 0:
        return out_ra, out_dec, statuses

    valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(ref_epoch)
    if not np.any(valid):
        return out_ra, out_dec, statuses

    c0 = SkyCoord(
        ra=ra[valid] * u.deg,
        dec=dec[valid] * u.deg,
        pm_ra_cosdec=pmra[valid] * u.mas / u.yr,
        pm_dec=pmdec[valid] * u.mas / u.yr,
        obstime=Time(ref_epoch[valid], format="jyear"),
        frame="icrs",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErfaWarning)
        c_obs = c0.apply_space_motion(new_obstime=Time(float(obs_mid_mjd), format="mjd"))
    out_ra[valid] = c_obs.ra.deg
    out_dec[valid] = c_obs.dec.deg
    for idx in np.flatnonzero(valid):
        statuses[int(idx)] = "astropy_space_motion"
    return out_ra, out_dec, statuses


def edge_distance_pix(x_pix: float, y_pix: float, shape: tuple[int, int]) -> float:
    height, width = shape
    return float(min(x_pix, y_pix, width - 1 - x_pix, height - 1 - y_pix))
