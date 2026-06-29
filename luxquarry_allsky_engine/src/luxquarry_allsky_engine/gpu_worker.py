from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits

from .calibration import (
    SPECTRAL_WCS_COLLECTION,
    image_to_ujy_arcsec2_scale,
    load_sapm,
    load_spectral_wcs_maps,
)
from .photometry import ApertureConfig
from .warp_aperture import (
    WarpFrameCalibrationDevice,
    run_warp_frame_aperture_resident_cupy,
    upload_frame_calibration,
)


@dataclass(frozen=True)
class PersistentWorkerConfig:
    aperture: ApertureConfig
    device: str = "cuda:0"
    worker_index: int = 0
    worker_count: int = 1
    write_combined_output: bool = False
    rmm_pool: bool = True


@dataclass(frozen=True)
class ResidentCalibration:
    release: str
    detector: int
    sapm_path: str
    wavelength_path: str
    device_arrays: WarpFrameCalibrationDevice
    upload_wall_sec: float


class PersistentGpuFrameWorker:
    def __init__(self, config: PersistentWorkerConfig):
        self.config = config
        self._calibration: dict[tuple[str, int], ResidentCalibration] = {}
        self._cudf = None
        self._init_gpu_runtime()

    def run(
        self,
        *,
        manifest_path: Path,
        projected_targets_path: Path,
        output_dir: Path,
        run_id: str,
        limit_frames: int | None = None,
        status_path: Path | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        output_dir.mkdir(parents=True, exist_ok=True)
        shards_dir = output_dir / "measurement_shards"
        shards_dir.mkdir(exist_ok=True)
        status_path = status_path or output_dir / "run_status.json"

        manifest = pd.read_parquet(manifest_path)
        if limit_frames is not None:
            manifest = manifest.head(limit_frames).copy()
        manifest = manifest.reset_index(drop=True)
        manifest = manifest.iloc[
            [i for i in range(len(manifest)) if i % self.config.worker_count == self.config.worker_index]
        ].copy()
        targets = pd.read_parquet(projected_targets_path)
        frame_ids = set(manifest["frame_group_id"].astype(str))
        targets = targets[targets["frame_group_id"].astype(str).isin(frame_ids)].copy()

        summary: dict[str, Any] = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "manifest_path": str(manifest_path),
            "projected_targets_path": str(projected_targets_path),
            "output_dir": str(output_dir),
            "device": self.config.device,
            "worker_index": self.config.worker_index,
            "worker_count": self.config.worker_count,
            "frame_count": int(len(manifest)),
            "input_projected_rows": int(len(targets)),
            "measurement_rows": 0,
            "ok_measurement_rows": 0,
            "failed_frames": 0,
            "calibration_upload_count": 0,
            "backend": "persistent_warp_frame_kernel_plus_cudf_shards",
            "frame_timings": [],
            "shards": [],
        }
        self._write_status(status_path, summary, started, state="running")

        shard_frames = []
        for frame_ordinal, frame in enumerate(manifest.to_dict(orient="records")):
            frame_group_id = str(frame.get("frame_group_id"))
            frame_targets = targets[
                targets["frame_group_id"].astype(str).eq(frame_group_id) & targets["in_frame"].astype(bool)
            ].copy()
            if frame_targets.empty:
                continue
            t0 = time.perf_counter()
            try:
                gdf, frame_stats = self._measure_frame(frame, frame_targets)
                shard_path = shards_dir / f"{run_id}.{frame_group_id}.parquet"
                tw = time.perf_counter()
                gdf.to_parquet(shard_path, index=False)
                write_wall = time.perf_counter() - tw
                rows = int(len(gdf))
                ok_count = int(frame_stats["ok_count"])
                summary["measurement_rows"] += rows
                summary["ok_measurement_rows"] += ok_count
                shard_row = {
                    "frame_group_id": frame_group_id,
                    "image_id": frame.get("image_id"),
                    "path": str(shard_path),
                    "rows": rows,
                    "ok_rows": ok_count,
                    "write_wall_sec": write_wall,
                }
                summary["shards"].append(shard_row)
                if self.config.write_combined_output:
                    shard_frames.append(gdf)
                timing = {
                    "frame_group_id": frame_group_id,
                    "image_id": frame.get("image_id"),
                    "input_target_count": int(len(frame_targets)),
                    "measurement_count": rows,
                    "ok_count": ok_count,
                    "wall_time_sec": time.perf_counter() - t0,
                    "write_wall_sec": write_wall,
                    **frame_stats,
                }
                summary["frame_timings"].append(timing)
            except Exception as exc:
                summary["failed_frames"] += 1
                summary["frame_timings"].append(
                    {
                        "frame_group_id": frame_group_id,
                        "image_id": frame.get("image_id"),
                        "input_target_count": int(len(frame_targets)),
                        "measurement_count": 0,
                        "ok_count": 0,
                        "wall_time_sec": time.perf_counter() - t0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            summary["calibration_upload_count"] = len(self._calibration)
            summary["completed_frames"] = len(summary["frame_timings"])
            summary["total_wall_sec"] = time.perf_counter() - started
            self._write_status(status_path, summary, started, state="running")

        if self.config.write_combined_output and shard_frames:
            combined_path = output_dir / f"{run_id}.measurements.parquet"
            tw = time.perf_counter()
            self._cudf.concat(shard_frames, ignore_index=True).to_parquet(combined_path, index=False)
            summary["combined_output_path"] = str(combined_path)
            summary["combined_write_wall_sec"] = time.perf_counter() - tw

        summary["total_wall_sec"] = time.perf_counter() - started
        summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
        summary_path = output_dir / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._write_status(status_path, summary, started, state="complete")
        return summary

    def _measure_frame(self, frame: dict[str, Any], targets: pd.DataFrame):
        import cudf

        frame_started = time.perf_counter()
        path = Path(str(frame["path"]))
        detector = int(frame["detector"])
        release = str(frame.get("release") or "qr2")
        calibration = self._get_calibration(release=release, detector=detector)
        t_read = time.perf_counter()
        with fits.open(path, memmap=True) as hdul:
            image_hdu = hdul["IMAGE"]
            image = np.asarray(image_hdu.data, dtype=float)
            variance = np.asarray(hdul["VARIANCE"].data, dtype=float) if "VARIANCE" in hdul else None
            flags = np.asarray(hdul["FLAGS"].data, dtype=np.uint32) if "FLAGS" in hdul else None
            unit_scale = image_to_ujy_arcsec2_scale(image_hdu.header)
        fits_read_wall = time.perf_counter() - t_read

        t_select = time.perf_counter()
        x_zero = targets["x_pix"].to_numpy(dtype=float) - 1.0
        y_zero = targets["y_pix"].to_numpy(dtype=float) - 1.0
        edge_zero = np.minimum.reduce([x_zero, y_zero, image.shape[1] - 1 - x_zero, image.shape[0] - 1 - y_zero])
        selected = np.isfinite(edge_zero) & (edge_zero >= self.config.aperture.edge_margin_pix)
        selected_targets = targets.loc[selected].reset_index(drop=True)
        selected_edge = edge_zero[selected]
        x_selected = x_zero[selected]
        y_selected = y_zero[selected]
        selection_wall = time.perf_counter() - t_select

        t_kernel = time.perf_counter()
        batch = run_warp_frame_aperture_resident_cupy(
            image=image,
            variance=variance,
            flags=flags,
            calibration=calibration.device_arrays,
            image_to_ujy_arcsec2=unit_scale,
            x_zero_based=x_selected,
            y_zero_based=y_selected,
            aperture_radius_pix=self.config.aperture.aperture_radius_pix,
            annulus_inner_pix=self.config.aperture.annulus_inner_pix,
            annulus_outer_pix=self.config.aperture.annulus_outer_pix,
            fatal_flag_bits=self.config.aperture.fatal_flag_bits,
            device=self.config.device,
        )
        kernel_wall = time.perf_counter() - t_kernel

        t_table = time.perf_counter()
        gdf = cudf.from_pandas(
            selected_targets[
                [
                    "frame_group_id",
                    "image_id",
                    "catalog",
                    "target_id",
                    "source_id",
                    "ra_deg",
                    "dec_deg",
                    "mag_primary",
                    "mag_primary_band",
                    "x_pix",
                    "y_pix",
                ]
            ]
        )
        gdf["fits_path"] = str(path)
        gdf["edge_distance_pix"] = selected_edge
        gdf["detector"] = detector
        gdf["release"] = release
        gdf["wavelength_source"] = "spectral_wcs_CWAVE_CBAND"
        gdf["wavelength_calibration_file"] = calibration.wavelength_path
        gdf["wavelength_calibration_collection"] = SPECTRAL_WCS_COLLECTION
        gdf["sapm_file"] = calibration.sapm_path
        for name, values in batch.columns.items():
            gdf[name] = values
        gdf["aperture_flux_unit"] = "uJy"
        gdf["aperture_status_code"] = batch.status
        gdf["aperture_status"] = "ok"
        gdf.loc[gdf["aperture_status_code"] != 0, "aperture_status"] = "bad_background"
        ok_count = int((batch.status == 0).sum().get())
        table_wall = time.perf_counter() - t_table
        return gdf, {
            "ok_count": ok_count,
            "selected_target_count": int(len(selected_targets)),
            "fits_read_wall_sec": fits_read_wall,
            "selection_wall_sec": selection_wall,
            "kernel_wall_sec": kernel_wall,
            "table_wall_sec": table_wall,
            "frame_compute_wall_sec": time.perf_counter() - frame_started,
        }

    def _get_calibration(self, *, release: str, detector: int) -> ResidentCalibration:
        key = (release, int(detector))
        cached = self._calibration.get(key)
        if cached is not None:
            return cached
        t0 = time.perf_counter()
        sapm, _sapm_header, sapm_path = load_sapm(str(self.config.aperture.cache_root), release, detector)
        cwave, cband, wavelength_path = load_spectral_wcs_maps(str(self.config.aperture.cache_root), release, detector)
        device_arrays = upload_frame_calibration(sapm=sapm, cwave=cwave, cband=cband, device=self.config.device)
        cached = ResidentCalibration(
            release=release,
            detector=int(detector),
            sapm_path=sapm_path,
            wavelength_path=wavelength_path,
            device_arrays=device_arrays,
            upload_wall_sec=time.perf_counter() - t0,
        )
        self._calibration[key] = cached
        return cached

    def _init_gpu_runtime(self) -> None:
        device_index = _device_index(self.config.device)
        if self.config.rmm_pool:
            import rmm

            try:
                rmm.reinitialize(pool_allocator=True, devices=device_index)
            except TypeError:
                rmm.reinitialize(pool_allocator=True)
        import cupy as cp
        import cudf
        import warp as wp

        cp.cuda.Device(device_index).use()
        wp.init()
        self._cudf = cudf

    @staticmethod
    def _write_status(path: Path, summary: dict[str, Any], started: float, *, state: str) -> None:
        status = {
            "state": state,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": time.perf_counter() - started,
            **summary,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)


def run_persistent_gpu_worker(
    *,
    manifest_path: Path,
    projected_targets_path: Path,
    output_dir: Path,
    run_id: str,
    config: PersistentWorkerConfig,
    limit_frames: int | None = None,
    status_path: Path | None = None,
) -> dict[str, Any]:
    worker = PersistentGpuFrameWorker(config)
    return worker.run(
        manifest_path=manifest_path,
        projected_targets_path=projected_targets_path,
        output_dir=output_dir,
        run_id=run_id,
        limit_frames=limit_frames,
        status_path=status_path,
    )


def _device_index(device: str) -> int:
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    return 0
