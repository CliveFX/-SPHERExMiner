from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from spherex_laser_miner.calibration import image_to_ujy_per_pixel, load_sapm, variance_to_ujy2
from spherex_laser_miner.catalog.gaia import query_gaia_for_s_region
from spherex_laser_miner.catalog.manual_targets import ManualTarget
from spherex_laser_miner.config import MinerConfig
from spherex_laser_miner.coordinates import edge_distance_pix, propagate_coordinates
from spherex_laser_miner.live_status import mark_frame, mark_frame_perf, mark_targets, reset_live_status
from spherex_laser_miner.photometry.aperture import aperture_measure
from spherex_laser_miner.photometry.calibrated_aperture import calibrated_aperture_measure
from spherex_laser_miner.photometry.warp_calibrated import warp_calibrated_aperture_batch
from spherex_laser_miner.photometry.psf import psf_measure, psf_not_run
from spherex_laser_miner.qa import write_smoke_artifacts


def run_best_trial_field_worker(
    target: ManualTarget,
    cfg: MinerConfig,
    trials: list[dict[str, object]],
    max_gaia_sources: int = 500,
) -> dict[str, object]:
    best = _best_trial(trials)
    if best is None:
        raise RuntimeError("No measured SIMP trial available for field worker.")
    field_job = run_trial_field_worker(
        target=target,
        cfg=cfg,
        trial=best,
        output_dir=cfg.smoke_run_dir,
        max_gaia_sources=max_gaia_sources,
        job_kind="smoke_simp_best_field",
    )
    _update_qa_after_field_worker(cfg.smoke_run_dir, field_job, len(pd.read_parquet(cfg.smoke_run_dir / "measurements.parquet")))
    return field_job


def run_trial_field_worker(
    target: ManualTarget,
    cfg: MinerConfig,
    trial: dict[str, object],
    output_dir: Path,
    max_gaia_sources: int = 500,
    job_kind: str = "field_trial",
    target_rows_override: list[dict[str, Any]] | None = None,
    gaia_g_min: float = 8.0,
    gaia_g_max: float = 19.0,
) -> dict[str, object]:
    candidate = dict(trial["candidate"])
    local_path = Path(str(trial["local_path"]))
    gaia_cache_path = (
        cfg.cache_root
        / "gaia"
        / "target_index"
        / f"{candidate['obs_id']}_D{trial['detector']}_top{max_gaia_sources}_tiled_v2.parquet"
    )
    if target_rows_override is None:
        gaia = query_gaia_for_s_region(
            s_region=str(candidate["s_region"]),
            cache_path=gaia_cache_path,
            max_sources=max_gaia_sources,
            g_min=gaia_g_min,
            g_max=gaia_g_max,
        )
        target_rows = _combined_targets(target, gaia)
    else:
        target_rows = target_rows_override

    selection_rows: list[dict[str, Any]] = []
    measurement_rows: list[dict[str, Any]] = []
    image_id = local_path.stem
    perf_start = time.perf_counter()
    perf: dict[str, Any] = {
        "worker_name": threading.current_thread().name,
        "fits_open_sec": 0.0,
        "calibration_sec": 0.0,
        "selection_sec": 0.0,
        "photometry_sec": 0.0,
        "aperture_sec": 0.0,
        "calibrated_aperture_sec": 0.0,
        "psf_sec": 0.0,
        "status_sec": 0.0,
        "write_sec": 0.0,
    }
    mark_frame(
        cfg.smoke_run_dir,
        image_id=image_id,
        status="active",
        worker_name=threading.current_thread().name,
        input_file_path=str(local_path),
        detector=int(trial["detector"]),
        observation_id=str(candidate["obs_id"]),
    )
    try:
        t0 = time.perf_counter()
        with fits.open(local_path, memmap=True) as hdul:
            image_hdu = hdul["IMAGE"]
            image = image_hdu.data
            variance = hdul["VARIANCE"].data if "VARIANCE" in hdul else None
            flags = hdul["FLAGS"].data if "FLAGS" in hdul else None
            zodi = hdul["ZODI"].data if "ZODI" in hdul else None
            psf_cube = hdul["PSF"].data if "PSF" in hdul else None
            spatial_wcs = WCS(image_hdu.header)
            spectral_wcs = _make_spectral_wcs(hdul, image_hdu.header)
            obs_mid_mjd = float(trial["obs_mid_mjd"])
            detector = int(trial["detector"])
            perf["fits_open_sec"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            sapm_data, sapm_header, sapm_path = load_sapm(cfg.cache_root, cfg.release, detector)
            flux_ujy = image_to_ujy_per_pixel(image, image_hdu.header, sapm_data, sapm_header)
            var_ujy2 = (
                variance_to_ujy2(variance, image_hdu.header, sapm_data, sapm_header)
                if variance is not None
                else None
            )
            perf["calibration_sec"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            ra_ref = np.asarray([float(row["ra_reference_deg"]) for row in target_rows], dtype=float)
            dec_ref = np.asarray([float(row["dec_reference_deg"]) for row in target_rows], dtype=float)
            ref_epoch = np.asarray(
                [
                    np.nan if _optional_float(row.get("reference_epoch_yr")) is None else float(row["reference_epoch_yr"])
                    for row in target_rows
                ],
                dtype=float,
            )
            pmra = np.asarray(
                [np.nan if _optional_float(row.get("pmra_masyr")) is None else float(row["pmra_masyr"]) for row in target_rows],
                dtype=float,
            )
            pmdec = np.asarray(
                [np.nan if _optional_float(row.get("pmdec_masyr")) is None else float(row["pmdec_masyr"]) for row in target_rows],
                dtype=float,
            )
            ra_epoch_arr, dec_epoch_arr, coord_statuses = propagate_coordinates(
                ra_deg=ra_ref,
                dec_deg=dec_ref,
                reference_epoch_yr=ref_epoch,
                pmra_masyr=pmra,
                pmdec_masyr=pmdec,
                obs_mid_mjd=obs_mid_mjd,
            )
            x_arr, y_arr = spatial_wcs.world_to_pixel(SkyCoord(ra_epoch_arr * u.deg, dec_epoch_arr * u.deg))
            x_arr = np.asarray(x_arr, dtype=float)
            y_arr = np.asarray(y_arr, dtype=float)
            edge_arr = np.minimum.reduce(
                [x_arr, y_arr, image.shape[1] - 1 - x_arr, image.shape[0] - 1 - y_arr]
            )
            inside_arr = (0 <= x_arr) & (x_arr < image.shape[1]) & (0 <= y_arr) & (y_arr < image.shape[0])
            selected_arr = inside_arr & (edge_arr >= cfg.edge_margin_pix)
            for idx, row in enumerate(target_rows):
                ra_epoch = float(ra_epoch_arr[idx])
                dec_epoch = float(dec_epoch_arr[idx])
                coord_status = coord_statuses[idx]
                x_pix = float(x_arr[idx])
                y_pix = float(y_arr[idx])
                edge_pix = float(edge_arr[idx])
                inside = bool(inside_arr[idx])
                selected = bool(selected_arr[idx])
                selection = {
                    **row,
                    "ra_epoch_deg": ra_epoch,
                    "dec_epoch_deg": dec_epoch,
                    "coordinate_propagation": coord_status,
                    "x_pix": x_pix,
                    "y_pix": y_pix,
                    "inside_image": inside,
                    "edge_distance_pix": edge_pix,
                    "selected_for_photometry": selected,
                    "rejection_reason": None if selected else "outside_or_edge",
                }
                selection_rows.append(selection)
            selected_rows = [item for item in selection_rows if item["selected_for_photometry"]]
            ts = time.perf_counter()
            mark_targets(cfg.smoke_run_dir, image_id=image_id, targets=selected_rows, status="queued")
            perf["status_sec"] += time.perf_counter() - ts
            perf["selection_sec"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            ts = time.perf_counter()
            mark_targets(cfg.smoke_run_dir, image_id=image_id, targets=selected_rows, status="active")
            perf["status_sec"] += time.perf_counter() - ts
            calibrated_batch = None
            if cfg.photometry_backend == "warp_calibrated" and selected_rows:
                tp = time.perf_counter()
                calibrated_batch_result = warp_calibrated_aperture_batch(
                    flux_ujy=flux_ujy,
                    var_ujy2=var_ujy2,
                    flags=flags,
                    x_pix=np.asarray([float(row["x_pix"]) for row in selected_rows], dtype=float),
                    y_pix=np.asarray([float(row["y_pix"]) for row in selected_rows], dtype=float),
                    aperture_radius_pix=cfg.aperture_radius_pix,
                    annulus_inner_pix=cfg.annulus_inner_pix,
                    annulus_outer_pix=cfg.annulus_outer_pix,
                    fatal_flag_bits=cfg.fatal_flag_bits,
                    devices=cfg.warp_devices,
                    worker_name=threading.current_thread().name,
                )
                perf["calibrated_aperture_sec"] += time.perf_counter() - tp
                perf["warp_device"] = calibrated_batch_result.device
                calibrated_batch = calibrated_batch_result.measurements
            for row_idx, row in enumerate(selected_rows):
                x_pix = float(row["x_pix"])
                y_pix = float(row["y_pix"])
                cwave_um, cband_um = _wavelength_at(spectral_wcs, x_pix, y_pix)
                aperture_json = {
                    "aperture_status": "diagnostic_disabled",
                    "aperture_flux": None,
                    "aperture_flux_unc": None,
                    "aperture_flux_unit": None,
                    "flags_summary": None,
                }
                if cfg.enable_diagnostic_aperture:
                    tp = time.perf_counter()
                    aperture = aperture_measure(
                        image=image,
                        variance=variance,
                        flags=flags,
                        zodi=zodi,
                        x_pix=x_pix,
                        y_pix=y_pix,
                        aperture_radius_pix=cfg.aperture_radius_pix,
                        annulus_inner_pix=cfg.annulus_inner_pix,
                        annulus_outer_pix=cfg.annulus_outer_pix,
                        fatal_flag_bits=cfg.fatal_flag_bits,
                    )
                    perf["aperture_sec"] += time.perf_counter() - tp
                    aperture_json = aperture.to_json_dict()
                if calibrated_batch is not None:
                    calibrated = calibrated_batch[row_idx]
                else:
                    tp = time.perf_counter()
                    calibrated = calibrated_aperture_measure(
                        flux_ujy=flux_ujy,
                        var_ujy2=var_ujy2,
                        flags=flags,
                        x_pix=x_pix,
                        y_pix=y_pix,
                        aperture_radius_pix=cfg.aperture_radius_pix,
                        annulus_inner_pix=cfg.annulus_inner_pix,
                        annulus_outer_pix=cfg.annulus_outer_pix,
                        fatal_flag_bits=cfg.fatal_flag_bits,
                    )
                    perf["calibrated_aperture_sec"] += time.perf_counter() - tp
                tp = time.perf_counter()
                if cfg.enable_psf_photometry:
                    psf = psf_measure(
                        flux_ujy=flux_ujy,
                        var_ujy2=var_ujy2,
                        flags=flags,
                        psf_cube=psf_cube,
                        x_pix=x_pix,
                        y_pix=y_pix,
                        fatal_flag_bits=cfg.fatal_flag_bits,
                    )
                else:
                    psf = psf_not_run()
                perf["psf_sec"] += time.perf_counter() - tp
                measurement = {
                    **row,
                    "image_id": image_id,
                    "release": cfg.release,
                    "processing_version": _processing_version(local_path),
                    "planning_period": _planning_period(local_path),
                    "detector": detector,
                    "observation_id": candidate["obs_id"],
                    "obs_mid_time": Time(obs_mid_mjd, format="mjd").isot,
                    "filter_profile": cfg.filter_profile,
                    "cwave_um": cwave_um,
                    "cband_um": cband_um,
                    "wavelength_source": "MEF WCS-WAVE",
                    "wavelength_calibration_file": None,
                    "sapm_file_path": str(sapm_path),
                    "image_unit": str(image_hdu.header.get("BUNIT", "")),
                    "photometry_backend": cfg.photometry_backend,
                    "diagnostic_aperture_enabled": cfg.enable_diagnostic_aperture,
                    "input_file_path": str(local_path),
                    "pipeline_version": "0.1.0",
                    **aperture_json,
                    **calibrated.to_json_dict(),
                    "fatal_flag_present": calibrated.n_bad_aperture_pixels_calibrated > 0,
                    **psf.to_json_dict(),
                }
                measurement_rows.append(measurement)
            ts = time.perf_counter()
            mark_targets(cfg.smoke_run_dir, image_id=image_id, targets=measurement_rows, status="done")
            perf["status_sec"] += time.perf_counter() - ts
            perf["photometry_sec"] += time.perf_counter() - t0
    except Exception as exc:
        mark_frame(cfg.smoke_run_dir, image_id=image_id, status="error", error=f"{type(exc).__name__}: {exc}")
        raise

    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selection_rows).to_parquet(output_dir / "target_selection.parquet", index=False)
    pd.DataFrame(measurement_rows).to_parquet(output_dir / "measurements.parquet", index=False)
    perf["write_sec"] += time.perf_counter() - t0
    perf["targets_selected"] = sum(1 for row in selection_rows if row.get("selected_for_photometry"))
    perf["targets_measured"] = len(measurement_rows)
    perf["elapsed_sec"] = time.perf_counter() - perf_start
    perf["target_rate_per_sec"] = len(measurement_rows) / perf["elapsed_sec"] if perf["elapsed_sec"] > 0 else None
    first_measurement = measurement_rows[0] if measurement_rows else {}
    mark_frame(
        cfg.smoke_run_dir,
        image_id=image_id,
        status="done",
        cwave_um=_optional_float(first_measurement.get("cwave_um")),
        cband_um=_optional_float(first_measurement.get("cband_um")),
    )
    mark_frame_perf(cfg.smoke_run_dir, image_id=image_id, perf=perf)
    field_job = {
        "job_kind": job_kind,
        "manual_target_id": target.target_id,
        "image_id": local_path.stem,
        "input_file_path": str(local_path),
        "candidate": candidate,
        "gaia_cache_path": str(gaia_cache_path),
        "output_dir": str(output_dir),
        "measurement_path": str(output_dir / "measurements.parquet"),
        "target_selection_path": str(output_dir / "target_selection.parquet"),
        "targets_considered": len(selection_rows),
        "targets_measured": len(measurement_rows),
        "simp_measured": any(row["target_id"] == target.target_id for row in measurement_rows),
        "performance": perf,
    }
    (output_dir / "field_job.json").write_text(json.dumps(field_job, indent=2), encoding="utf-8")
    return field_job


def run_multi_trial_field_workers(
    target: ManualTarget,
    cfg: MinerConfig,
    trials: list[dict[str, object]],
    max_gaia_sources: int = 500,
    include_fatal_simp_trials: bool = True,
    max_field_workers: int = 4,
    target_rows_override: list[dict[str, Any]] | None = None,
    gaia_g_min: float = 8.0,
    gaia_g_max: float = 19.0,
) -> list[dict[str, object]]:
    reset_live_status(cfg.smoke_run_dir)
    shard_root = cfg.smoke_run_dir / "field_shards"
    selected_trials = []
    for trial in trials:
        if trial.get("status") != "measured":
            continue
        if not include_fatal_simp_trials and dict(trial.get("aperture") or {}).get("fatal_flag_present"):
            continue
        selected_trials.append(trial)

    jobs: list[dict[str, object]] = []

    def run_one(trial: dict[str, object]) -> dict[str, object]:
        image_id = Path(str(trial["local_path"])).stem
        shard_dir = shard_root / f"image_id={image_id}"
        return run_trial_field_worker(
            target=target,
            cfg=cfg,
            trial=trial,
            output_dir=shard_dir,
            max_gaia_sources=max_gaia_sources,
            job_kind="smoke_simp_multifield",
            target_rows_override=target_rows_override,
            gaia_g_min=gaia_g_min,
            gaia_g_max=gaia_g_max,
        )

    worker_count = max(1, min(int(max_field_workers), len(selected_trials) or 1))
    if worker_count == 1:
        jobs = [run_one(trial) for trial in selected_trials]
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="field-worker") as executor:
            futures = [executor.submit(run_one, trial) for trial in selected_trials]
            for future in as_completed(futures):
                jobs.append(future.result())
    jobs.sort(key=lambda job: str(job.get("image_id", "")))
    (cfg.smoke_run_dir / "field_jobs.json").write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    return jobs


def build_fixed_target_rows_from_trial(
    target: ManualTarget,
    cfg: MinerConfig,
    trial: dict[str, object],
    max_gaia_sources: int,
    gaia_g_min: float = 7.0,
    gaia_g_max: float = 10.0,
) -> list[dict[str, Any]]:
    candidate = dict(trial["candidate"])
    cache_path = (
        cfg.cache_root
        / "gaia"
        / "target_index"
        / f"fixed_depth_{candidate['obs_id']}_D{trial['detector']}_top{max_gaia_sources}_g{gaia_g_min:.1f}_{gaia_g_max:.1f}_tiled_v2.parquet"
    )
    gaia = query_gaia_for_s_region(
        s_region=str(candidate["s_region"]),
        cache_path=cache_path,
        max_sources=max_gaia_sources,
        g_min=gaia_g_min,
        g_max=gaia_g_max,
    )
    return _combined_targets(target, gaia)


def _combined_targets(target: ManualTarget, gaia: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "target_id": target.target_id,
            "target_type": target.target_type,
            "source_id": None,
            "object_name": target.object_name,
            "ra_reference_deg": target.ra_deg,
            "dec_reference_deg": target.dec_deg,
            "reference_epoch_yr": target.reference_epoch_yr,
            "pmra_masyr": target.pmra_masyr,
            "pmdec_masyr": target.pmdec_masyr,
            "parallax_mas": target.parallax_mas,
            "priority_score": target.priority_score,
            "target_filter_flags": "manual_include",
        }
    ]
    for _, src in gaia.iterrows():
        source_id = str(src["source_id"])
        rows.append(
            {
                "target_id": f"gaia_dr3_{source_id}",
                "target_type": "gaia_dr3",
                "source_id": source_id,
                "object_name": None,
                "ra_reference_deg": float(src["ra"]),
                "dec_reference_deg": float(src["dec"]),
                "reference_epoch_yr": _optional_float(src.get("ref_epoch")),
                "pmra_masyr": _optional_float(src.get("pmra")),
                "pmdec_masyr": _optional_float(src.get("pmdec")),
                "parallax_mas": _optional_float(src.get("parallax")),
                "priority_score": float(100.0 - min(float(src.get("phot_g_mean_mag", 99.0)), 99.0)),
                "phot_g_mean_mag": _optional_float(src.get("phot_g_mean_mag")),
                "phot_bp_mean_mag": _optional_float(src.get("phot_bp_mean_mag")),
                "phot_rp_mean_mag": _optional_float(src.get("phot_rp_mean_mag")),
                "bp_rp": _optional_float(src.get("bp_rp")),
                "ruwe": _optional_float(src.get("ruwe")),
                "duplicated_source": bool(src.get("duplicated_source")),
                "astrometric_params_solved": _optional_float(src.get("astrometric_params_solved")),
                "target_filter_flags": "gaia_broad_debug",
            }
        )
    return rows


def _best_trial(trials: list[dict[str, object]]) -> dict[str, object] | None:
    eligible = [
        trial
        for trial in trials
        if trial.get("status") == "measured" and not dict(trial.get("aperture") or {}).get("fatal_flag_present")
    ]
    if not eligible:
        eligible = [trial for trial in trials if trial.get("status") == "measured"]
    if not eligible:
        return None
    return max(eligible, key=lambda trial: float(trial.get("edge_distance_pix") or -1.0))


def _optional_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _processing_version(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("l2b-"):
            return part
    return None


def _planning_period(path: Path) -> str | None:
    parts = path.parts
    for idx, part in enumerate(parts):
        if part == "level2" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _make_spectral_wcs(hdul: fits.HDUList, header: fits.Header) -> WCS | None:
    try:
        spectral_wcs = WCS(header=header, fobj=hdul, key="W")
        spectral_wcs.sip = None
        return spectral_wcs
    except Exception:
        return None


def _wavelength_at(spectral_wcs: WCS | None, x_pix: float, y_pix: float) -> tuple[float | None, float | None]:
    if spectral_wcs is None:
        return None, None
    try:
        cwave, cband = spectral_wcs.pixel_to_world(x_pix, y_pix)
        return float(cwave.to(u.um).value), float(cband.to(u.um).value)
    except Exception:
        return None, None


def _update_qa_after_field_worker(run_dir: Path, field_job: dict[str, object], measurement_rows: int) -> None:
    qa_path = run_dir / "qa.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8")) if qa_path.exists() else {}
    qa.update(
        {
            "field_worker_image_id": field_job["image_id"],
            "field_worker_targets_considered": field_job["targets_considered"],
            "field_worker_targets_measured": field_job["targets_measured"],
            "field_worker_simp_measured": field_job["simp_measured"],
            "measurement_rows": measurement_rows,
            "measurement_path": str(run_dir / "measurements.parquet"),
            "target_selection_path": str(run_dir / "target_selection.parquet"),
            "field_job_path": str(run_dir / "field_job.json"),
        }
    )
    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
