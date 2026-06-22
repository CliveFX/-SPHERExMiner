from __future__ import annotations

import warnings

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from erfa import ErfaWarning

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


def edge_distance_pix(x_pix: float, y_pix: float, shape: tuple[int, int]) -> float:
    height, width = shape
    return float(min(x_pix, y_pix, width - 1 - x_pix, height - 1 - y_pix))
