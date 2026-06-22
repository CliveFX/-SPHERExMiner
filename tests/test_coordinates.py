from __future__ import annotations

import numpy as np

from spherex_laser_miner.coordinates import propagate_coordinate, propagate_coordinates


def test_vector_coordinate_propagation_matches_scalar() -> None:
    ra = np.array([84.857137, 85.0, 86.5])
    dec = np.array([40.641523, 40.0, 39.5])
    ref_epoch = np.array([2016.0, 2016.0, np.nan])
    pmra = np.array([646.149592, 1.0, np.nan])
    pmdec = np.array([-834.491923, 2.0, np.nan])
    obs_mid_mjd = 61000.0

    vec_ra, vec_dec, statuses = propagate_coordinates(ra, dec, ref_epoch, pmra, pmdec, obs_mid_mjd)

    for i in range(2):
        scalar_ra, scalar_dec, scalar_status = propagate_coordinate(
            ra_deg=float(ra[i]),
            dec_deg=float(dec[i]),
            reference_epoch_yr=float(ref_epoch[i]),
            pmra_masyr=float(pmra[i]),
            pmdec_masyr=float(pmdec[i]),
            obs_mid_mjd=obs_mid_mjd,
        )
        assert statuses[i] == scalar_status
        assert np.isclose(vec_ra[i], scalar_ra, rtol=0, atol=1e-12)
        assert np.isclose(vec_dec[i], scalar_dec, rtol=0, atol=1e-12)

    assert statuses[2] == "no_observation_or_reference_epoch"
    assert vec_ra[2] == ra[2]
    assert vec_dec[2] == dec[2]
