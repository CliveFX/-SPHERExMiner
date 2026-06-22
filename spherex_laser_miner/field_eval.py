from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from spherex_laser_miner.cache import cache_path_for_access_url, sha256_file
from spherex_laser_miner.calibration import image_to_ujy_per_pixel, load_sapm, variance_to_ujy2
from spherex_laser_miner.catalog.manual_targets import ManualTarget
from spherex_laser_miner.config import MinerConfig
from spherex_laser_miner.coordinates import edge_distance_pix, propagate_target
from spherex_laser_miner.downloader import download_file
from spherex_laser_miner.manifest.sia import SpherexImageCandidate, query_sia_candidates
from spherex_laser_miner.photometry.aperture import aperture_measure
from spherex_laser_miner.photometry.calibrated_aperture import calibrated_aperture_measure
from spherex_laser_miner.qa import write_smoke_artifacts


def evaluate_target_fields(
    target: ManualTarget,
    cfg: MinerConfig,
    limit_fields: int,
    redownload: bool = False,
    max_eval_workers: int = 1,
) -> list[dict[str, object]]:
    # Query around a current-ish propagated position. Exact per-image propagation is
    # still recomputed using each candidate's observation time below.
    query_ra, query_dec, _ = propagate_target(target, Time.now().mjd)
    candidates = query_sia_candidates(query_ra, query_dec)
    cfg.smoke_run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cfg.smoke_run_dir / "candidate_fields.json"
    manifest_path.write_text(
        json.dumps([candidate.to_json_dict() for candidate in candidates], indent=2),
        encoding="utf-8",
    )

    selected = candidates[:limit_fields]
    trials_by_index: dict[int, dict[str, object]] = {}
    worker_count = max(1, min(int(max_eval_workers), len(selected) or 1))
    if worker_count == 1:
        for idx, candidate in enumerate(selected):
            trials_by_index[idx] = evaluate_one_candidate(target, cfg, candidate, redownload=redownload)
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="candidate-eval") as executor:
            futures = {
                executor.submit(evaluate_one_candidate, target, cfg, candidate, redownload): idx
                for idx, candidate in enumerate(selected)
            }
            for future in as_completed(futures):
                trials_by_index[futures[future]] = future.result()
                partial = [trials_by_index[idx] for idx in sorted(trials_by_index)]
                (cfg.smoke_run_dir / "simp_field_trials.partial.json").write_text(
                    json.dumps(partial, indent=2),
                    encoding="utf-8",
                )
    trials = [trials_by_index[idx] for idx in sorted(trials_by_index)]
    trial_path = cfg.smoke_run_dir / "simp_field_trials.json"
    trial_path.write_text(json.dumps(trials, indent=2), encoding="utf-8")
    write_smoke_artifacts(cfg.smoke_run_dir, trials)
    return trials


def evaluate_one_candidate(
    target: ManualTarget,
    cfg: MinerConfig,
    candidate: SpherexImageCandidate,
    redownload: bool = False,
) -> dict[str, object]:
    local_path = cache_path_for_access_url(cfg.cache_root, candidate.access_url)
    trial: dict[str, object] = {
        "target_id": target.target_id,
        "object_name": target.object_name,
        "candidate": candidate.to_json_dict(),
        "local_path": str(local_path),
    }
    try:
        local_path, download_status = download_file(candidate.access_url, local_path, redownload=redownload)
        trial["download_status"] = download_status
        trial["file_size"] = local_path.stat().st_size
        trial["sha256"] = sha256_file(local_path)

        with fits.open(local_path, memmap=True) as hdul:
            image_hdu = hdul["IMAGE"]
            image = image_hdu.data
            variance = hdul["VARIANCE"].data if "VARIANCE" in hdul else None
            flags = hdul["FLAGS"].data if "FLAGS" in hdul else None
            zodi = hdul["ZODI"].data if "ZODI" in hdul else None
            detector = int(image_hdu.header.get("DETECTOR", candidate.detector))
            sapm_data, sapm_header, sapm_path = load_sapm(cfg.cache_root, cfg.release, detector)
            flux_ujy = image_to_ujy_per_pixel(image, image_hdu.header, sapm_data, sapm_header)
            var_ujy2 = (
                variance_to_ujy2(variance, image_hdu.header, sapm_data, sapm_header)
                if variance is not None
                else None
            )

            obs_mid = _obs_mid_mjd(candidate, image_hdu.header)
            ra_epoch, dec_epoch, propagation_status = propagate_target(target, obs_mid)
            spatial_wcs = WCS(image_hdu.header)
            x_pix, y_pix = spatial_wcs.world_to_pixel(SkyCoord(ra_epoch * u.deg, dec_epoch * u.deg))
            edge_pix = edge_distance_pix(float(x_pix), float(y_pix), image.shape)
            cwave_um, cband_um = _wavelength_at(hdul, image_hdu.header, float(x_pix), float(y_pix))

            trial.update(
                {
                    "status": "projected",
                    "obs_mid_mjd": obs_mid,
                    "obs_mid_isot": Time(obs_mid, format="mjd").isot if obs_mid is not None else None,
                    "ra_epoch_deg": ra_epoch,
                    "dec_epoch_deg": dec_epoch,
                    "coordinate_propagation": propagation_status,
                    "x_pix": float(x_pix),
                    "y_pix": float(y_pix),
                    "inside_image": _inside_image(float(x_pix), float(y_pix), image.shape),
                    "edge_distance_pix": edge_pix,
                    "detector": detector,
                    "image_shape": [int(image.shape[0]), int(image.shape[1])],
                    "image_unit": str(image_hdu.header.get("BUNIT", "")),
                    "cwave_um": cwave_um,
                    "cband_um": cband_um,
                    "wavelength_source": "MEF WCS-WAVE",
                    "sapm_file_path": str(sapm_path),
                    "aperture_radius_pix": cfg.aperture_radius_pix,
                    "annulus_inner_pix": cfg.annulus_inner_pix,
                    "annulus_outer_pix": cfg.annulus_outer_pix,
                }
            )
            if not trial["inside_image"] or edge_pix < cfg.edge_margin_pix:
                trial["status"] = "rejected_edge"
                return trial

            measurement = aperture_measure(
                image=image,
                variance=variance,
                flags=flags,
                zodi=zodi,
                x_pix=float(x_pix),
                y_pix=float(y_pix),
                aperture_radius_pix=cfg.aperture_radius_pix,
                annulus_inner_pix=cfg.annulus_inner_pix,
                annulus_outer_pix=cfg.annulus_outer_pix,
                fatal_flag_bits=cfg.fatal_flag_bits,
            )
            trial["aperture"] = measurement.to_json_dict()
            calibrated = calibrated_aperture_measure(
                flux_ujy=flux_ujy,
                var_ujy2=var_ujy2,
                flags=flags,
                x_pix=float(x_pix),
                y_pix=float(y_pix),
                aperture_radius_pix=cfg.aperture_radius_pix,
                annulus_inner_pix=cfg.annulus_inner_pix,
                annulus_outer_pix=cfg.annulus_outer_pix,
                fatal_flag_bits=cfg.fatal_flag_bits,
            )
            trial["calibrated_aperture"] = calibrated.to_json_dict()
            trial["status"] = "measured" if measurement.aperture_status == "ok" else measurement.aperture_status
            return trial
    except Exception as exc:
        trial["status"] = "failed"
        trial["error_type"] = type(exc).__name__
        trial["error_message"] = str(exc)
        return trial


def _obs_mid_mjd(candidate: SpherexImageCandidate, header: fits.Header) -> float | None:
    if candidate.obs_mid_mjd is not None:
        return candidate.obs_mid_mjd
    for key in ("MJD-AVG", "MJDAVG", "MJD-MID", "MJD_OBS", "MJD-OBS"):
        if key in header:
            return float(header[key])
    if "DATE" in header:
        return float(Time(header["DATE"], format="isot").mjd)
    return None


def _inside_image(x_pix: float, y_pix: float, shape: tuple[int, int]) -> bool:
    height, width = shape
    return 0 <= x_pix < width and 0 <= y_pix < height


def _wavelength_at(
    hdul: fits.HDUList,
    header: fits.Header,
    x_pix: float,
    y_pix: float,
) -> tuple[float | None, float | None]:
    try:
        spectral_wcs = WCS(header=header, fobj=hdul, key="W")
        spectral_wcs.sip = None
        cwave, cband = spectral_wcs.pixel_to_world(x_pix, y_pix)
        return float(cwave.to(u.um).value), float(cband.to(u.um).value)
    except Exception:
        return None, None
