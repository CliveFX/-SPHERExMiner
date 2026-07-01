from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from concurrent.futures import Future, ThreadPoolExecutor
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
from .object_store import is_s3_uri, stage_input_file
from .warp_aperture import (
    WarpFrameCalibrationDevice,
    run_warp_frame_aperture_resident_cupy,
    upload_frame_data,
    upload_frame_calibration,
)
from .warp_psf import run_warp_frame_psf_grid_resident_cupy


MEASUREMENT_COLUMN_PROFILES = ("full", "compact")
MEASUREMENT_PARQUET_COMPRESSIONS = ("snappy", "none")

COMPACT_MEASUREMENT_COLUMNS = (
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
    "fits_path",
    "edge_distance_pix",
    "detector",
    "release",
    "aperture_radius_pix",
    "annulus_inner_pix",
    "annulus_outer_pix",
    "edge_margin_pix",
    "wavelength_source",
    "wavelength_calibration_file",
    "wavelength_calibration_collection",
    "sapm_file",
    "aperture_flux_uJy",
    "aperture_flux_unc_uJy",
    "aperture_area_pix",
    "background_uJy_per_pix",
    "background_unc_uJy_per_pix",
    "n_bad_aperture_pixels",
    "flags_summary",
    "cwave_um",
    "cband_um",
    "psf_flux_uJy",
    "psf_flux_unc_uJy",
    "psf_background_uJy_per_pix",
    "psf_chi2",
    "psf_reduced_chi2",
    "psf_dof",
    "psf_n_valid",
    "psf_sector",
    "psf_grid_n_valid",
    "psf_grid_best_score",
    "psf_x_fit",
    "psf_y_fit",
    "psf_fit_offset_pix",
    "psf_kernel_sum",
    "psf_grid_n_trials",
    "psf_kernel_shape_y",
    "psf_kernel_shape_x",
    "psf_status_code",
    "aperture_flux_unit",
    "aperture_status_code",
    "psf_flux_unit",
)


@dataclass(frozen=True)
class PersistentWorkerConfig:
    aperture: ApertureConfig
    device: str = "cuda:0"
    worker_index: int = 0
    worker_count: int = 1
    write_combined_output: bool = False
    rmm_pool: bool = True
    shard_batch_frames: int = 1
    prefetch_frames: int = 0
    status_interval_frames: int = 1
    local_cache_dir: Path | None = None
    async_shard_writes: bool = False
    batch_table_assembly: bool = False
    discard_measurement_shards: bool = False
    measurement_column_profile: str = "full"
    measurement_parquet_compression: str = "snappy"
    enable_psf: bool = False
    psf_kernel_build_mode: str = "gpu_spline"
    psf_kernel_radius_native: int = 5
    psf_grid_half_range_pix: float = 1.0
    psf_grid_step_pix: float = 0.5
    psf_grid_metric: str = "snr"


@dataclass(frozen=True)
class ResidentCalibration:
    release: str
    detector: int
    sapm_path: str
    wavelength_path: str
    device_arrays: WarpFrameCalibrationDevice
    upload_wall_sec: float


@dataclass(frozen=True)
class FramePayload:
    source_path: str
    read_path: str
    image: np.ndarray
    variance: np.ndarray | None
    flags: np.ndarray | None
    psf_cube: np.ndarray | None
    psf_oversamp: int | None
    unit_scale: float
    staging_wall_sec: float
    fits_read_wall_sec: float
    staged_bytes: int


@dataclass(frozen=True)
class FrameMeasurement:
    metadata: pd.DataFrame
    columns: dict[str, object]
    status: object
    ok_count: int

    def __len__(self) -> int:
        return len(self.metadata)


@dataclass(frozen=True)
class FramePayloadResult:
    frame: dict[str, Any]
    payload: FramePayload
    payload_wait_wall_sec: float
    prefetched: bool


class PersistentGpuFrameWorker:
    def __init__(self, config: PersistentWorkerConfig):
        self.config = config
        self._calibration: dict[tuple[str, int], ResidentCalibration] = {}
        self._calibration_cache_hits = 0
        self._calibration_cache_misses = 0
        self._calibration_load_wall_sec = 0.0
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
        manifest = pd.read_parquet(manifest_path)
        if limit_frames is not None:
            manifest = manifest.head(limit_frames).copy()
        manifest = manifest.reset_index(drop=True)
        manifest = manifest.iloc[
            [i for i in range(len(manifest)) if i % self.config.worker_count == self.config.worker_index]
        ].copy()
        targets = pd.read_parquet(projected_targets_path)
        return self.process_frame_batch(
            manifest=manifest,
            targets=targets,
            output_dir=output_dir,
            run_id=run_id,
            manifest_path=manifest_path,
            projected_targets_path=projected_targets_path,
            status_path=status_path,
        )

    def process_frame_batch(
        self,
        *,
        manifest: pd.DataFrame,
        targets: pd.DataFrame,
        output_dir: Path,
        run_id: str,
        manifest_path: Path | str | None = None,
        projected_targets_path: Path | str | None = None,
        status_path: Path | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if self.config.write_combined_output and self.config.batch_table_assembly:
            raise ValueError("--write-combined-output is not supported with --batch-table-assembly yet")
        if self.config.write_combined_output and self.config.discard_measurement_shards:
            raise ValueError("--write-combined-output cannot be combined with --discard-measurement-shards")
        if self.config.measurement_column_profile not in MEASUREMENT_COLUMN_PROFILES:
            raise ValueError(
                f"measurement_column_profile must be one of {MEASUREMENT_COLUMN_PROFILES}: "
                f"{self.config.measurement_column_profile!r}"
            )
        if self.config.measurement_parquet_compression not in MEASUREMENT_PARQUET_COMPRESSIONS:
            raise ValueError(
                f"measurement_parquet_compression must be one of {MEASUREMENT_PARQUET_COMPRESSIONS}: "
                f"{self.config.measurement_parquet_compression!r}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        shards_dir = output_dir / "measurement_shards"
        shards_dir.mkdir(exist_ok=True)
        status_path = status_path or output_dir / "run_status.json"
        calibration_start_hits = self._calibration_cache_hits
        calibration_start_misses = self._calibration_cache_misses
        calibration_start_load_wall = self._calibration_load_wall_sec

        manifest = manifest.reset_index(drop=True).copy()
        targets = targets.copy()
        frame_ids = set(manifest["frame_group_id"].astype(str))
        targets = targets[targets["frame_group_id"].astype(str).isin(frame_ids)].copy()
        targets_by_frame = {
            str(frame_group_id): frame_targets
            for frame_group_id, frame_targets in targets.groupby("frame_group_id", sort=False)
        }

        summary: dict[str, Any] = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "manifest_path": str(manifest_path) if manifest_path is not None else None,
            "projected_targets_path": str(projected_targets_path) if projected_targets_path is not None else None,
            "output_dir": str(output_dir),
            "device": self.config.device,
            "worker_index": self.config.worker_index,
            "worker_count": self.config.worker_count,
            "frame_count": int(len(manifest)),
            "input_projected_rows": int(len(targets)),
            "measurement_rows": 0,
            "ok_measurement_rows": 0,
            "failed_frames": 0,
            "calibration_upload_count": len(self._calibration),
            "resident_calibration_count": len(self._calibration),
            "calibration_cache_hits": self._calibration_cache_hits,
            "calibration_cache_misses": self._calibration_cache_misses,
            "calibration_load_wall_sec": self._calibration_load_wall_sec,
            "batch_calibration_cache_hits": 0,
            "batch_calibration_cache_misses": 0,
            "batch_calibration_load_wall_sec": 0.0,
            "backend": "persistent_warp_frame_kernel_plus_cudf_shards",
            "local_cache_dir": str(self.config.local_cache_dir) if self.config.local_cache_dir else None,
            "async_shard_writes": self.config.async_shard_writes,
            "batch_table_assembly": self.config.batch_table_assembly,
            "discard_measurement_shards": self.config.discard_measurement_shards,
            "measurement_column_profile": self.config.measurement_column_profile,
            "measurement_parquet_compression": self.config.measurement_parquet_compression,
            "aperture_radius_pix": float(self.config.aperture.aperture_radius_pix),
            "annulus_inner_pix": float(self.config.aperture.annulus_inner_pix),
            "annulus_outer_pix": float(self.config.aperture.annulus_outer_pix),
            "edge_margin_pix": float(self.config.aperture.edge_margin_pix),
            "enable_psf": self.config.enable_psf,
            "psf_kernel_build_mode": self.config.psf_kernel_build_mode if self.config.enable_psf else None,
            "psf_grid_half_range_pix": self.config.psf_grid_half_range_pix if self.config.enable_psf else None,
            "psf_grid_step_pix": self.config.psf_grid_step_pix if self.config.enable_psf else None,
            "psf_grid_metric": self.config.psf_grid_metric if self.config.enable_psf else None,
            "psf_measurement_rows": 0,
            "ok_psf_rows": 0,
            "frame_timings": [],
            "shards": [],
        }
        self._write_status(status_path, summary, started, state="running")

        shard_frames = []
        shard_group: list[dict[str, Any]] = []
        pending_gdfs = []
        pending_shard_writes: list[Future[dict[str, Any]]] = []
        writer_pool = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="shard-writer")
            if self.config.async_shard_writes
            else None
        )
        frames = manifest.to_dict(orient="records")
        try:
            frame_payloads = self._iter_frame_payloads(frames)
            for frame_ordinal, payload_result in enumerate(frame_payloads):
                frame = payload_result.frame
                payload = payload_result.payload
                frame_group_id = str(frame.get("frame_group_id"))
                frame_targets_all = targets_by_frame.get(frame_group_id)
                if frame_targets_all is None or frame_targets_all.empty:
                    continue
                frame_targets = frame_targets_all[frame_targets_all["in_frame"].astype(bool)]
                if frame_targets.empty:
                    continue
                t0 = time.perf_counter()
                try:
                    measurement, frame_stats = self._measure_frame(frame, frame_targets, payload)
                    rows = int(len(measurement))
                    ok_count = int(frame_stats["ok_count"])
                    summary["measurement_rows"] += rows
                    summary["ok_measurement_rows"] += ok_count
                    summary["psf_measurement_rows"] += int(frame_stats.get("psf_measurement_count", 0))
                    summary["ok_psf_rows"] += int(frame_stats.get("ok_psf_count", 0))
                    if self.config.discard_measurement_shards:
                        summary["discarded_measurement_rows"] = int(
                            summary.get("discarded_measurement_rows", 0)
                        ) + rows
                    else:
                        pending_gdfs.append(measurement)
                        shard_group.append(
                            {
                                "frame_group_id": frame_group_id,
                                "image_id": frame.get("image_id"),
                                "rows": rows,
                                "ok_rows": ok_count,
                            }
                        )
                    if self.config.write_combined_output:
                        shard_frames.append(measurement)
                    write_wall = 0.0
                    shard_submit_wall = 0.0
                    async_write_queued = False
                    shard_path = None
                    if (not self.config.discard_measurement_shards) and len(pending_gdfs) >= self.config.shard_batch_frames:
                        shard_path, flush_wall, async_write_queued = self._flush_shard_batch(
                            pending_gdfs=pending_gdfs,
                            shard_group=shard_group,
                            shards_dir=shards_dir,
                            run_id=run_id,
                            summary=summary,
                            writer_pool=writer_pool,
                            pending_writes=pending_shard_writes,
                        )
                        if async_write_queued:
                            shard_submit_wall = flush_wall
                        else:
                            write_wall = flush_wall
                    timing = {
                        "frame_group_id": frame_group_id,
                        "image_id": frame.get("image_id"),
                        "input_target_count": int(len(frame_targets)),
                        "measurement_count": rows,
                        "ok_count": ok_count,
                        "wall_time_sec": time.perf_counter() - t0,
                        "payload_wait_wall_sec": payload_result.payload_wait_wall_sec,
                        "payload_prefetched": payload_result.prefetched,
                        "write_wall_sec": write_wall,
                        "shard_submit_wall_sec": shard_submit_wall,
                        "async_write_queued": async_write_queued,
                        "deferred_write": shard_path is None,
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
                            "payload_wait_wall_sec": payload_result.payload_wait_wall_sec,
                            "payload_prefetched": payload_result.prefetched,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                self._update_calibration_summary(
                    summary,
                    start_hits=calibration_start_hits,
                    start_misses=calibration_start_misses,
                    start_load_wall=calibration_start_load_wall,
                )
                summary["completed_frames"] = len(summary["frame_timings"])
                summary["queued_shard_writes"] = len(pending_shard_writes)
                summary["total_wall_sec"] = time.perf_counter() - started
                if self._should_write_status(len(summary["frame_timings"])):
                    self._write_status(status_path, summary, started, state="running")

            if pending_gdfs:
                self._flush_shard_batch(
                    pending_gdfs=pending_gdfs,
                    shard_group=shard_group,
                    shards_dir=shards_dir,
                    run_id=run_id,
                    summary=summary,
                    writer_pool=writer_pool,
                    pending_writes=pending_shard_writes,
                )

            if pending_shard_writes:
                summary["async_shard_write_wait_wall_sec"] = self._collect_shard_writes(
                    pending_shard_writes,
                    summary,
                )
                summary["queued_shard_writes"] = 0
            self._update_calibration_summary(
                summary,
                start_hits=calibration_start_hits,
                start_misses=calibration_start_misses,
                start_load_wall=calibration_start_load_wall,
            )
        finally:
            if writer_pool is not None:
                writer_pool.shutdown(wait=True)

        if self.config.write_combined_output and shard_frames:
            combined_path = output_dir / f"{run_id}.measurements.parquet"
            tw = time.perf_counter()
            self._cudf.concat(shard_frames, ignore_index=True).to_parquet(combined_path, index=False)
            summary["combined_output_path"] = str(combined_path)
            summary["combined_write_wall_sec"] = time.perf_counter() - tw

        self._update_calibration_summary(
            summary,
            start_hits=calibration_start_hits,
            start_misses=calibration_start_misses,
            start_load_wall=calibration_start_load_wall,
        )
        summary["total_wall_sec"] = time.perf_counter() - started
        summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
        summary_path = output_dir / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._write_status(status_path, summary, started, state="complete")
        return summary

    def _measure_frame(self, frame: dict[str, Any], targets: pd.DataFrame, payload: FramePayload):
        frame_started = time.perf_counter()
        path = Path(str(frame["path"]))
        detector = int(frame["detector"])
        release = str(frame.get("release") or "qr2")
        calibration = self._get_calibration(release=release, detector=detector)

        t_select = time.perf_counter()
        x_zero = targets["x_pix"].to_numpy(dtype=np.float32) - np.float32(1.0)
        y_zero = targets["y_pix"].to_numpy(dtype=np.float32) - np.float32(1.0)
        edge_zero = np.minimum.reduce(
            [x_zero, y_zero, payload.image.shape[1] - 1 - x_zero, payload.image.shape[0] - 1 - y_zero]
        )
        selected = np.isfinite(edge_zero) & (edge_zero >= self.config.aperture.edge_margin_pix)
        selected_targets = targets.loc[selected].reset_index(drop=True)
        selected_edge = edge_zero[selected]
        x_selected = x_zero[selected]
        y_selected = y_zero[selected]
        selection_wall = time.perf_counter() - t_select

        t_frame_upload = time.perf_counter()
        frame_data = upload_frame_data(
            image=payload.image,
            variance=payload.variance,
            flags=payload.flags,
            device=self.config.device,
        )
        frame_upload_wall = time.perf_counter() - t_frame_upload

        t_kernel = time.perf_counter()
        batch = run_warp_frame_aperture_resident_cupy(
            image=payload.image,
            variance=payload.variance,
            flags=payload.flags,
            frame_data=frame_data,
            calibration=calibration.device_arrays,
            image_to_ujy_arcsec2=payload.unit_scale,
            x_zero_based=x_selected,
            y_zero_based=y_selected,
            aperture_radius_pix=self.config.aperture.aperture_radius_pix,
            annulus_inner_pix=self.config.aperture.annulus_inner_pix,
            annulus_outer_pix=self.config.aperture.annulus_outer_pix,
            fatal_flag_bits=self.config.aperture.fatal_flag_bits,
            device=self.config.device,
        )
        kernel_wall = time.perf_counter() - t_kernel
        psf_kernel_wall = 0.0
        psf_ok_count = 0
        psf_timings: dict[str, float] = {}
        if self.config.enable_psf:
            if payload.psf_cube is None:
                raise ValueError(f"PSF photometry requested but FITS frame has no PSF HDU: {payload.read_path}")
            t_psf = time.perf_counter()
            psf_batch = run_warp_frame_psf_grid_resident_cupy(
                image=payload.image,
                variance=payload.variance,
                flags=payload.flags,
                frame_data=frame_data,
                sapm=calibration.device_arrays.sapm,
                psf_cube=payload.psf_cube,
                image_to_ujy_arcsec2=payload.unit_scale,
                x_zero_based=x_selected,
                y_zero_based=y_selected,
                fatal_flag_bits=self.config.aperture.fatal_flag_bits,
                kernel_radius_native=self.config.psf_kernel_radius_native,
                half_range_pix=self.config.psf_grid_half_range_pix,
                step_pix=self.config.psf_grid_step_pix,
                metric=self.config.psf_grid_metric,
                kernel_build_mode=self.config.psf_kernel_build_mode,
                oversamp=payload.psf_oversamp,
                device=self.config.device,
            )
            psf_kernel_wall = time.perf_counter() - t_psf
            batch.columns.update(psf_batch.columns)
            batch.columns["psf_status_code"] = psf_batch.status
            psf_ok_count = int((psf_batch.status == 0).sum().get())
            psf_timings = {f"psf_{name}": float(value) for name, value in psf_batch.timings.items()}

        t_table = time.perf_counter()
        metadata = self._measurement_metadata(
            frame=frame,
            selected_targets=selected_targets,
            selected_edge=selected_edge,
            payload=payload,
            calibration=calibration,
            aperture=self.config.aperture,
        )
        ok_count = int((batch.status == 0).sum().get())
        if self.config.batch_table_assembly:
            measurement = FrameMeasurement(
                metadata=metadata,
                columns=batch.columns,
                status=batch.status,
                ok_count=ok_count,
            )
        else:
            measurement = self._measurement_to_cudf(
                metadata=metadata,
                columns=batch.columns,
                status=batch.status,
                device=self.config.device,
            )
        table_wall = time.perf_counter() - t_table
        return measurement, {
            "ok_count": ok_count,
            "selected_target_count": int(len(selected_targets)),
            "fits_read_wall_sec": payload.fits_read_wall_sec,
            "staging_wall_sec": payload.staging_wall_sec,
            "staged_bytes": payload.staged_bytes,
            "selection_wall_sec": selection_wall,
            "frame_upload_wall_sec": frame_upload_wall,
            "kernel_wall_sec": kernel_wall,
            "aperture_kernel_wall_sec": kernel_wall,
            "psf_kernel_wall_sec": psf_kernel_wall,
            "psf_measurement_count": int(len(selected_targets)) if self.config.enable_psf else 0,
            "ok_psf_count": psf_ok_count,
            **psf_timings,
            "table_wall_sec": table_wall,
            "frame_compute_wall_sec": time.perf_counter() - frame_started,
        }

    @staticmethod
    def _measurement_metadata(
        *,
        frame: dict[str, Any],
        selected_targets: pd.DataFrame,
        selected_edge: np.ndarray,
        payload: FramePayload,
        calibration: ResidentCalibration,
        aperture: ApertureConfig,
    ) -> pd.DataFrame:
        detector = int(frame["detector"])
        release = str(frame.get("release") or "qr2")
        metadata = selected_targets[
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
        ].copy()
        metadata["fits_path"] = str(frame["path"])
        metadata["local_fits_path"] = payload.read_path
        metadata["edge_distance_pix"] = selected_edge.astype(np.float32, copy=False)
        metadata["detector"] = detector
        metadata["release"] = release
        metadata["aperture_radius_pix"] = float(aperture.aperture_radius_pix)
        metadata["annulus_inner_pix"] = float(aperture.annulus_inner_pix)
        metadata["annulus_outer_pix"] = float(aperture.annulus_outer_pix)
        metadata["edge_margin_pix"] = float(aperture.edge_margin_pix)
        metadata["wavelength_source"] = "spectral_wcs_CWAVE_CBAND"
        metadata["wavelength_calibration_file"] = calibration.wavelength_path
        metadata["wavelength_calibration_collection"] = SPECTRAL_WCS_COLLECTION
        metadata["sapm_file"] = calibration.sapm_path
        return metadata

    @staticmethod
    def _measurement_to_cudf(
        *,
        metadata: pd.DataFrame,
        columns: dict[str, object],
        status: object,
        device: str,
        column_profile: str = "full",
    ) -> tuple[Any, dict[str, float]]:
        import cudf
        import cupy as cp

        timings = {
            "metadata_to_cudf_wall_sec": 0.0,
            "column_attach_wall_sec": 0.0,
            "status_attach_wall_sec": 0.0,
        }
        cp.cuda.Device(_device_index(device)).use()
        compact = column_profile == "compact"
        if compact:
            metadata = metadata[[column for column in metadata.columns if column in COMPACT_MEASUREMENT_COLUMNS]]
            columns = {name: values for name, values in columns.items() if name in COMPACT_MEASUREMENT_COLUMNS}
        t_metadata = time.perf_counter()
        gdf = cudf.from_pandas(metadata)
        timings["metadata_to_cudf_wall_sec"] = time.perf_counter() - t_metadata
        t_columns = time.perf_counter()
        for name, values in columns.items():
            gdf[name] = values
        timings["column_attach_wall_sec"] = time.perf_counter() - t_columns
        t_status = time.perf_counter()
        if not compact or "aperture_flux_unit" in COMPACT_MEASUREMENT_COLUMNS:
            gdf["aperture_flux_unit"] = "uJy"
        if not compact or "aperture_status_code" in COMPACT_MEASUREMENT_COLUMNS:
            gdf["aperture_status_code"] = status
        if not compact:
            gdf["aperture_status"] = "ok"
            gdf.loc[gdf["aperture_status_code"] != 0, "aperture_status"] = "bad_background"
        if "psf_status_code" in gdf.columns:
            if not compact or "psf_flux_unit" in COMPACT_MEASUREMENT_COLUMNS:
                gdf["psf_flux_unit"] = "uJy"
            if not compact:
                gdf["psf_status"] = "ok"
                gdf.loc[gdf["psf_status_code"] != 0, "psf_status"] = "bad_fit"
        timings["status_attach_wall_sec"] = time.perf_counter() - t_status
        return gdf, timings

    def _get_calibration(self, *, release: str, detector: int) -> ResidentCalibration:
        key = (release, int(detector))
        cached = self._calibration.get(key)
        if cached is not None:
            self._calibration_cache_hits += 1
            return cached
        self._calibration_cache_misses += 1
        t0 = time.perf_counter()
        sapm, _sapm_header, sapm_path = load_sapm(str(self.config.aperture.cache_root), release, detector)
        cwave, cband, wavelength_path = load_spectral_wcs_maps(str(self.config.aperture.cache_root), release, detector)
        device_arrays = upload_frame_calibration(sapm=sapm, cwave=cwave, cband=cband, device=self.config.device)
        load_wall_sec = time.perf_counter() - t0
        self._calibration_load_wall_sec += load_wall_sec
        cached = ResidentCalibration(
            release=release,
            detector=int(detector),
            sapm_path=sapm_path,
            wavelength_path=wavelength_path,
            device_arrays=device_arrays,
            upload_wall_sec=load_wall_sec,
        )
        self._calibration[key] = cached
        return cached

    def _update_calibration_summary(
        self,
        summary: dict[str, Any],
        *,
        start_hits: int,
        start_misses: int,
        start_load_wall: float,
    ) -> None:
        summary["calibration_upload_count"] = len(self._calibration)
        summary["resident_calibration_count"] = len(self._calibration)
        summary["calibration_cache_hits"] = int(self._calibration_cache_hits)
        summary["calibration_cache_misses"] = int(self._calibration_cache_misses)
        summary["calibration_load_wall_sec"] = float(self._calibration_load_wall_sec)
        summary["batch_calibration_cache_hits"] = int(self._calibration_cache_hits - start_hits)
        summary["batch_calibration_cache_misses"] = int(self._calibration_cache_misses - start_misses)
        summary["batch_calibration_load_wall_sec"] = float(self._calibration_load_wall_sec - start_load_wall)

    def _iter_frame_payloads(self, frames: list[dict[str, Any]]):
        if self.config.prefetch_frames <= 0:
            for frame in frames:
                t_payload = time.perf_counter()
                yield FramePayloadResult(
                    frame=frame,
                    payload=self._read_frame_payload(frame),
                    payload_wait_wall_sec=time.perf_counter() - t_payload,
                    prefetched=False,
                )
            return
        max_workers = max(1, int(self.config.prefetch_frames))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="fits-prefetch") as pool:
            pending: dict[int, Future[FramePayload]] = {}
            next_submit = 0
            for _ in range(min(max_workers, len(frames))):
                pending[next_submit] = pool.submit(self._read_frame_payload, frames[next_submit])
                next_submit += 1
            for idx, frame in enumerate(frames):
                future = pending.pop(idx)
                if next_submit < len(frames):
                    pending[next_submit] = pool.submit(self._read_frame_payload, frames[next_submit])
                    next_submit += 1
                t_wait = time.perf_counter()
                yield FramePayloadResult(
                    frame=frame,
                    payload=future.result(),
                    payload_wait_wall_sec=time.perf_counter() - t_wait,
                    prefetched=True,
                )

    @staticmethod
    def _read_fits_arrays(
        path: Path,
        *,
        include_psf: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, int | None, float]:
        with fits.open(path, memmap=True) as hdul:
            image_hdu = hdul["IMAGE"]
            image = np.asarray(image_hdu.data, dtype=np.float32)
            variance = np.asarray(hdul["VARIANCE"].data, dtype=np.float32) if "VARIANCE" in hdul else None
            flags = np.asarray(hdul["FLAGS"].data, dtype=np.uint32) if "FLAGS" in hdul else None
            psf_cube = None
            psf_oversamp = None
            if include_psf:
                if "PSF" not in hdul:
                    raise ValueError(f"FITS file has no PSF extension: {path}")
                psf_hdu = hdul["PSF"]
                psf_cube = np.asarray(psf_hdu.data, dtype=np.float32)
                psf_oversamp = int(psf_hdu.header.get("OVERSAMP", image_hdu.header.get("OVERSAMP", 0)) or 0) or None
            unit_scale = image_to_ujy_arcsec2_scale(image_hdu.header)
        return image, variance, flags, psf_cube, psf_oversamp, unit_scale

    def _read_frame_payload(self, frame: dict[str, Any]) -> FramePayload:
        source_uri = str(frame["path"])
        staging_wall = 0.0
        staged_bytes = 0
        if is_s3_uri(source_uri) and self.config.local_cache_dir is None:
            raise ValueError("S3 FITS paths require PersistentWorkerConfig.local_cache_dir")
        read_path = Path(source_uri)
        if self.config.local_cache_dir is not None:
            t_stage = time.perf_counter()
            read_path, staged_bytes = self._stage_fits(source_uri, self.config.local_cache_dir)
            staging_wall = time.perf_counter() - t_stage
        t_read = time.perf_counter()
        image, variance, flags, psf_cube, psf_oversamp, unit_scale = self._read_fits_arrays(
            read_path,
            include_psf=self.config.enable_psf,
        )
        return FramePayload(
            source_path=source_uri,
            read_path=str(read_path),
            image=image,
            variance=variance,
            flags=flags,
            psf_cube=psf_cube,
            psf_oversamp=psf_oversamp,
            unit_scale=unit_scale,
            staging_wall_sec=staging_wall,
            fits_read_wall_sec=time.perf_counter() - t_read,
            staged_bytes=staged_bytes,
        )

    @staticmethod
    def _stage_fits(source_path: str | Path, cache_dir: Path) -> tuple[Path, int]:
        return stage_input_file(source_path, cache_dir)

    def _flush_shard_batch(
        self,
        *,
        pending_gdfs: list[Any],
        shard_group: list[dict[str, Any]],
        shards_dir: Path,
        run_id: str,
        summary: dict[str, Any],
        writer_pool: ThreadPoolExecutor | None = None,
        pending_writes: list[Future[dict[str, Any]]] | None = None,
    ) -> tuple[Path, float, bool]:
        if not pending_gdfs:
            return shards_dir / f"{run_id}.empty.parquet", 0.0, False
        tw = time.perf_counter()
        gdfs = list(pending_gdfs)
        group = list(shard_group)
        shard_path = self._shard_path(shards_dir=shards_dir, run_id=run_id, shard_group=group)
        pending_gdfs.clear()
        shard_group.clear()
        if writer_pool is not None:
            if pending_writes is None:
                raise ValueError("pending_writes is required when writer_pool is provided")
            future = writer_pool.submit(
                self._write_shard_batch,
                gdfs,
                group,
                shards_dir,
                run_id,
                self.config.device,
                self.config.measurement_column_profile,
                self.config.measurement_parquet_compression,
            )
            pending_writes.append(future)
            return shard_path, time.perf_counter() - tw, True

        result = self._write_shard_batch(
            gdfs,
            group,
            shards_dir,
            run_id,
            self.config.device,
            self.config.measurement_column_profile,
            self.config.measurement_parquet_compression,
        )
        summary["shards"].append(result)
        return Path(result["path"]), float(result["write_wall_sec"]), False

    @staticmethod
    def _shard_path(*, shards_dir: Path, run_id: str, shard_group: list[dict[str, Any]]) -> Path:
        first = shard_group[0]["frame_group_id"]
        last = shard_group[-1]["frame_group_id"]
        return shards_dir / f"{run_id}.{first}_to_{last}.parquet"

    @staticmethod
    def _write_shard_batch(
        measurements: list[Any],
        shard_group: list[dict[str, Any]],
        shards_dir: Path,
        run_id: str,
        device: str,
        column_profile: str = "full",
        parquet_compression: str = "snappy",
    ) -> dict[str, Any]:
        import cudf
        import cupy as cp

        cp.cuda.Device(_device_index(device)).use()
        shard_path = PersistentGpuFrameWorker._shard_path(
            shards_dir=shards_dir,
            run_id=run_id,
            shard_group=shard_group,
        )
        tw = time.perf_counter()
        t_assemble = time.perf_counter()
        out, assemble_timings = PersistentGpuFrameWorker._assemble_shard_table(
            measurements,
            device=device,
            column_profile=column_profile,
        )
        shard_table_assembly_wall_sec = time.perf_counter() - t_assemble
        t_profile = time.perf_counter()
        out = PersistentGpuFrameWorker._apply_measurement_column_profile(out, column_profile)
        shard_column_profile_wall_sec = time.perf_counter() - t_profile
        compression_arg = None if parquet_compression == "none" else parquet_compression
        t_parquet = time.perf_counter()
        out.to_parquet(shard_path, index=False, compression=compression_arg)
        parquet_write_wall_sec = time.perf_counter() - t_parquet
        write_wall_sec = time.perf_counter() - tw
        byte_count = shard_path.stat().st_size if shard_path.exists() else 0
        return {
            "path": str(shard_path),
            "frame_group_ids": [row["frame_group_id"] for row in shard_group],
            "image_ids": [row["image_id"] for row in shard_group],
            "rows": int(sum(row["rows"] for row in shard_group)),
            "ok_rows": int(sum(row["ok_rows"] for row in shard_group)),
            "frame_count": len(shard_group),
            "column_profile": column_profile,
            "column_count": int(len(out.columns)),
            "parquet_compression": parquet_compression,
            "bytes": int(byte_count),
            "write_wall_sec": write_wall_sec,
            "shard_table_assembly_wall_sec": shard_table_assembly_wall_sec,
            **assemble_timings,
            "shard_column_profile_wall_sec": shard_column_profile_wall_sec,
            "parquet_write_wall_sec": parquet_write_wall_sec,
        }

    @staticmethod
    def _assemble_shard_table(measurements: list[Any], *, device: str, column_profile: str = "full"):
        import cudf
        import cupy as cp

        cp.cuda.Device(_device_index(device)).use()
        timings = {
            "metadata_concat_wall_sec": 0.0,
            "device_column_concat_wall_sec": 0.0,
            "status_concat_wall_sec": 0.0,
            "metadata_to_cudf_wall_sec": 0.0,
            "column_attach_wall_sec": 0.0,
            "status_attach_wall_sec": 0.0,
        }
        if not measurements:
            return cudf.DataFrame(), timings
        first = measurements[0]
        if not isinstance(first, FrameMeasurement):
            t_concat = time.perf_counter()
            table = measurements[0] if len(measurements) == 1 else cudf.concat(measurements, ignore_index=True)
            timings["device_column_concat_wall_sec"] = time.perf_counter() - t_concat
            return table, timings

        t_metadata = time.perf_counter()
        metadata = pd.concat([measurement.metadata for measurement in measurements], ignore_index=True)
        timings["metadata_concat_wall_sec"] = time.perf_counter() - t_metadata
        t_columns = time.perf_counter()
        columns = {
            name: cp.concatenate([measurement.columns[name] for measurement in measurements])
            for name in first.columns
        }
        timings["device_column_concat_wall_sec"] = time.perf_counter() - t_columns
        t_status = time.perf_counter()
        status = cp.concatenate([measurement.status for measurement in measurements])
        timings["status_concat_wall_sec"] = time.perf_counter() - t_status
        table, cudf_timings = PersistentGpuFrameWorker._measurement_to_cudf(
            metadata=metadata,
            columns=columns,
            status=status,
            device=device,
            column_profile=column_profile,
        )
        timings.update(cudf_timings)
        return table, timings

    @staticmethod
    def _apply_measurement_column_profile(table: Any, column_profile: str):
        if column_profile == "full":
            return table
        if column_profile != "compact":
            raise ValueError(f"Unknown measurement column profile: {column_profile!r}")
        columns = [column for column in COMPACT_MEASUREMENT_COLUMNS if column in table.columns]
        return table[columns]

    @staticmethod
    def _collect_shard_writes(pending_writes: list[Future[dict[str, Any]]], summary: dict[str, Any]) -> float:
        tw = time.perf_counter()
        for future in pending_writes:
            summary["shards"].append(future.result())
        pending_writes.clear()
        return time.perf_counter() - tw

    def _should_write_status(self, completed_frames: int) -> bool:
        interval = max(1, int(self.config.status_interval_frames))
        return completed_frames <= 1 or completed_frames % interval == 0

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
