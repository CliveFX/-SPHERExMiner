from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .dispatch import DispatchPlanConfig, build_dispatch_plan, collect_dispatch_run, write_dispatch_plan
from .local_runner import (
    LocalDispatchRunConfig,
    LocalPlanWorkerRunConfig,
    run_dispatch_plan_workers,
    run_local_dispatch,
)


@dataclass(frozen=True)
class DispatchBenchmarkSweepConfig:
    manifest_path: Path
    projected_targets_path: Path
    output_dir: Path
    run_id: str
    devices: tuple[str, ...]
    workers_per_device_values: tuple[int, ...]
    limit_frame_values: tuple[int, ...]
    shard_batch_frame_values: tuple[int, ...]
    prefetch_frame_values: tuple[int, ...]
    repetitions: int = 1
    cache_root: Path = Path("/mnt/niroseti/spherex_cache")
    local_cache_dir: Path | None = None
    executable: str = ".venv/bin/luxquarry-allsky"
    async_shard_writes: bool = True
    batch_table_assembly: bool = True
    discard_measurement_shards: bool = False
    measurement_column_profile: str = "full"
    measurement_parquet_compression: str = "snappy"
    materialize_worker_inputs: bool = True
    aperture_radius_pix: float = 2.0
    annulus_inner_pix: float = 4.0
    annulus_outer_pix: float = 6.0
    edge_margin_pix: float = 6.0
    enable_psf: bool = False
    psf_kernel_build_mode: str = "gpu_spline"
    psf_kernel_radius_native: int = 5
    psf_grid_half_range_pix: float = 1.0
    psf_grid_step_pix: float = 0.5
    psf_grid_metric: str = "snr"
    finalize_device: str = "cuda:0"
    score_baseline: bool = False
    candidate_min_abs_zscore: float = 5.0
    candidate_min_measurements: int = 10
    candidate_max_rows: int | None = None
    status_snapshot_interval_sec: float = 0.0
    continue_on_error: bool = False
    worker_only: bool = False


def run_dispatch_benchmark_sweep(config: DispatchBenchmarkSweepConfig) -> dict[str, Any]:
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []

    for trial_index, settings in enumerate(_settings(config), start=1):
        trial_run_id = _trial_run_id(config.run_id, trial_index, settings)
        trial_dir = config.output_dir / "trials" / trial_run_id
        trial_started = time.perf_counter()
        try:
            summary = _run_trial(config, settings, trial_dir, trial_run_id)
            row = _trial_row(
                config=config,
                settings=settings,
                trial_index=trial_index,
                trial_run_id=trial_run_id,
                trial_dir=trial_dir,
                summary=summary,
                status="complete",
                error=None,
                elapsed=time.perf_counter() - trial_started,
            )
        except Exception as exc:
            row = _failed_trial_row(
                config=config,
                settings=settings,
                trial_index=trial_index,
                trial_run_id=trial_run_id,
                trial_dir=trial_dir,
                error=f"{type(exc).__name__}: {exc}",
                elapsed=time.perf_counter() - trial_started,
            )
            rows.append(row)
            if not config.continue_on_error:
                _write_outputs(config, rows, profile_rows, started)
                raise
            continue
        rows.append(row)
        profile_rows.extend(_profile_rows_for_trial(row, summary))

    return _write_outputs(config, rows, profile_rows, started)


def _run_trial(
    config: DispatchBenchmarkSweepConfig,
    settings: dict[str, int],
    trial_dir: Path,
    trial_run_id: str,
) -> dict[str, Any]:
    if config.worker_only:
        return _run_worker_only_trial(config, settings, trial_dir, trial_run_id)
    return run_local_dispatch(
        LocalDispatchRunConfig(
            manifest_path=config.manifest_path,
            projected_targets_path=config.projected_targets_path,
            output_dir=trial_dir,
            run_id=trial_run_id,
            devices=config.devices,
            workers_per_device=int(settings["workers_per_device"]),
            cache_root=config.cache_root,
            limit_frames=int(settings["limit_frames"]),
            executable=config.executable,
            shard_batch_frames=int(settings["shard_batch_frames"]),
            prefetch_frames=int(settings["prefetch_frames"]),
            local_cache_dir=config.local_cache_dir,
            async_shard_writes=config.async_shard_writes,
            batch_table_assembly=config.batch_table_assembly,
            discard_measurement_shards=config.discard_measurement_shards,
            measurement_column_profile=config.measurement_column_profile,
            measurement_parquet_compression=config.measurement_parquet_compression,
            materialize_worker_inputs=config.materialize_worker_inputs,
            aperture_radius_pix=config.aperture_radius_pix,
            annulus_inner_pix=config.annulus_inner_pix,
            annulus_outer_pix=config.annulus_outer_pix,
            edge_margin_pix=config.edge_margin_pix,
            enable_psf=config.enable_psf,
            psf_kernel_build_mode=config.psf_kernel_build_mode,
            psf_kernel_radius_native=config.psf_kernel_radius_native,
            psf_grid_half_range_pix=config.psf_grid_half_range_pix,
            psf_grid_step_pix=config.psf_grid_step_pix,
            psf_grid_metric=config.psf_grid_metric,
            finalize_device=config.finalize_device,
            score_baseline=config.score_baseline,
            candidate_min_abs_zscore=config.candidate_min_abs_zscore,
            candidate_min_measurements=config.candidate_min_measurements,
            candidate_max_rows=config.candidate_max_rows,
            status_snapshot_interval_sec=config.status_snapshot_interval_sec,
        )
    )


def _run_worker_only_trial(
    config: DispatchBenchmarkSweepConfig,
    settings: dict[str, int],
    trial_dir: Path,
    trial_run_id: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    trial_dir.mkdir(parents=True, exist_ok=True)
    plan_path = trial_dir / "dispatch_plan.json"
    t_plan = time.perf_counter()
    plan = build_dispatch_plan(
        DispatchPlanConfig(
            manifest_path=config.manifest_path,
            projected_targets_path=config.projected_targets_path,
            output_dir=trial_dir,
            run_id=trial_run_id,
            devices=config.devices,
            workers_per_device=int(settings["workers_per_device"]),
            cache_root=config.cache_root,
            limit_frames=int(settings["limit_frames"]),
            executable=config.executable,
            shard_batch_frames=int(settings["shard_batch_frames"]),
            prefetch_frames=int(settings["prefetch_frames"]),
            status_interval_frames=1,
            local_cache_dir=config.local_cache_dir,
            async_shard_writes=config.async_shard_writes,
            batch_table_assembly=config.batch_table_assembly,
            discard_measurement_shards=config.discard_measurement_shards,
            measurement_column_profile=config.measurement_column_profile,
            measurement_parquet_compression=config.measurement_parquet_compression,
            materialize_worker_inputs=config.materialize_worker_inputs,
            aperture_radius_pix=config.aperture_radius_pix,
            annulus_inner_pix=config.annulus_inner_pix,
            annulus_outer_pix=config.annulus_outer_pix,
            edge_margin_pix=config.edge_margin_pix,
            enable_psf=config.enable_psf,
            psf_kernel_build_mode=config.psf_kernel_build_mode,
            psf_kernel_radius_native=config.psf_kernel_radius_native,
            psf_grid_half_range_pix=config.psf_grid_half_range_pix,
            psf_grid_step_pix=config.psf_grid_step_pix,
            psf_grid_metric=config.psf_grid_metric,
        )
    )
    write_dispatch_plan(plan, plan_path)
    plan_wall = time.perf_counter() - t_plan
    worker = run_dispatch_plan_workers(
        LocalPlanWorkerRunConfig(
            plan_path=plan_path,
            logs_dir=trial_dir / "worker_logs",
            status_snapshot_interval_sec=config.status_snapshot_interval_sec,
        )
    )
    aggregate = collect_dispatch_run(plan_path)
    elapsed = time.perf_counter() - started
    measurement_rows = int(aggregate.get("measurement_rows") or 0)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "luxquarry_worker_only_benchmark_trial",
        "status": worker.get("status"),
        "run_id": trial_run_id,
        "plan_path": str(plan_path),
        "output_dir": str(trial_dir),
        "devices": list(config.devices),
        "resume": False,
        "status_snapshot_interval_sec": config.status_snapshot_interval_sec,
        "worker_count": int(plan.get("worker_count") or 0),
        "launched_worker_count": int(worker.get("launched_worker_count") or 0),
        "skipped_worker_count": int(worker.get("skipped_worker_count") or 0),
        "materialize_worker_inputs": bool(plan.get("materialize_worker_inputs")),
        "plan_wall_sec": plan_wall,
        "worker_wall_sec": float(worker.get("worker_wall_sec") or 0.0),
        "finalize_wall_sec": 0.0,
        "total_wall_sec": elapsed,
        "measurement_rows": measurement_rows,
        "measurements_per_sec": _rate(measurement_rows, elapsed),
        "failed_worker_count": int(worker.get("failed_worker_count") or 0),
        "worker_results": list(worker.get("worker_results") or []),
        "finalize": {
            "aggregate": aggregate,
            "measurement_rows": measurement_rows,
            "target_count": 0,
            "spectra": {},
            "baseline_scoring": {},
        },
    }


def _settings(config: DispatchBenchmarkSweepConfig):
    for repetition in range(1, config.repetitions + 1):
        for workers_per_device, limit_frames, shard_batch_frames, prefetch_frames in itertools.product(
            config.workers_per_device_values,
            config.limit_frame_values,
            config.shard_batch_frame_values,
            config.prefetch_frame_values,
        ):
            yield {
                "repetition": repetition,
                "workers_per_device": workers_per_device,
                "limit_frames": limit_frames,
                "shard_batch_frames": shard_batch_frames,
                "prefetch_frames": prefetch_frames,
            }


def _trial_run_id(base: str, trial_index: int, settings: dict[str, int]) -> str:
    return (
        f"{base}_t{trial_index:03d}"
        f"_f{settings['limit_frames']}"
        f"_w{settings['workers_per_device']}"
        f"_s{settings['shard_batch_frames']}"
        f"_p{settings['prefetch_frames']}"
        f"_r{settings['repetition']}"
    )


def _trial_row(
    *,
    config: DispatchBenchmarkSweepConfig,
    settings: dict[str, int],
    trial_index: int,
    trial_run_id: str,
    trial_dir: Path,
    summary: dict[str, Any],
    status: str,
    error: str | None,
    elapsed: float,
) -> dict[str, Any]:
    finalize = summary.get("finalize") or {}
    aggregate = finalize.get("aggregate") or {}
    spectra = finalize.get("spectra") or {}
    baseline_scoring = finalize.get("baseline_scoring") or {}
    measurement_rows = int(summary.get("measurement_rows") or finalize.get("measurement_rows") or 0)
    completed_frames = int(aggregate.get("completed_frames") or 0)
    worker_count = int(summary.get("worker_count") or 0)
    gpu_count = len(config.devices)
    worker_wall = _num(summary.get("worker_wall_sec"))
    finalize_wall = _num(summary.get("finalize_wall_sec"))
    total_wall = _num(summary.get("total_wall_sec")) or elapsed
    worker_max_wall = _num(aggregate.get("worker_max_wall_sec"))
    worker_launch_overhead = max(0.0, worker_wall - worker_max_wall)
    psf_candidate_count = _num(aggregate.get("psf_candidate_count"))
    psf_device_submit_sync_wall = _num(aggregate.get("psf_device_submit_sync_wall_sec"))
    return {
        "trial_index": trial_index,
        "run_id": trial_run_id,
        "status": status,
        "error": error,
        "output_dir": str(trial_dir),
        "devices": ",".join(config.devices),
        "gpu_count": gpu_count,
        "workers_per_device": int(settings["workers_per_device"]),
        "worker_count": worker_count,
        "limit_frames": int(settings["limit_frames"]),
        "completed_frames": completed_frames,
        "shard_batch_frames": int(settings["shard_batch_frames"]),
        "prefetch_frames": int(settings["prefetch_frames"]),
        "repetition": int(settings["repetition"]),
        "measurement_rows": measurement_rows,
        "psf_measurement_rows": int(aggregate.get("psf_measurement_rows") or 0),
        "ok_psf_rows": int(aggregate.get("ok_psf_rows") or 0),
        "psf_candidate_count": psf_candidate_count,
        "target_count": int(finalize.get("target_count") or 0),
        "enable_psf": bool(config.enable_psf),
        "discard_measurement_shards": bool(config.discard_measurement_shards),
        "measurement_column_profile": config.measurement_column_profile,
        "measurement_parquet_compression": config.measurement_parquet_compression,
        "aperture_radius_pix": float(config.aperture_radius_pix),
        "annulus_inner_pix": float(config.annulus_inner_pix),
        "annulus_outer_pix": float(config.annulus_outer_pix),
        "psf_kernel_build_mode": config.psf_kernel_build_mode if config.enable_psf else None,
        "psf_grid_half_range_pix": float(config.psf_grid_half_range_pix) if config.enable_psf else 0.0,
        "psf_grid_step_pix": float(config.psf_grid_step_pix) if config.enable_psf else 0.0,
        "plan_wall_sec": _num(summary.get("plan_wall_sec")),
        "worker_wall_sec": worker_wall,
        "finalize_wall_sec": finalize_wall,
        "total_wall_sec": total_wall,
        "collect_wall_sec": _num(aggregate.get("collect_wall_sec")),
        "worker_min_wall_sec": _num(aggregate.get("worker_min_wall_sec")),
        "worker_avg_wall_sec": _num(aggregate.get("worker_avg_wall_sec")),
        "worker_max_wall_sec": worker_max_wall,
        "worker_sum_wall_sec": _num(aggregate.get("worker_sum_wall_sec")),
        "worker_parallel_efficiency": _num(aggregate.get("worker_parallel_efficiency")),
        "worker_wall_skew_ratio": _num(aggregate.get("worker_wall_skew_ratio")),
        "worker_launch_overhead_sec": worker_launch_overhead,
        "read_shards_wall_sec": _num(spectra.get("read_shards_wall_sec")),
        "sort_wall_sec": _num(spectra.get("sort_wall_sec")),
        "write_spectra_wall_sec": _num(spectra.get("write_spectra_wall_sec")),
        "target_summary_wall_sec": _num(spectra.get("target_summary_wall_sec")),
        "spectra_total_wall_sec": _num(spectra.get("total_wall_sec")),
        "baseline_score_wall_sec": _num(baseline_scoring.get("total_wall_sec")),
        "baseline_candidate_count": int(baseline_scoring.get("candidate_count") or 0),
        "staging_wall_sec": _num(aggregate.get("staging_wall_sec")),
        "payload_wait_wall_sec": _num(aggregate.get("payload_wait_wall_sec")),
        "frame_upload_wall_sec": _num(aggregate.get("frame_upload_wall_sec")),
        "aperture_kernel_wall_sec": _num(aggregate.get("aperture_kernel_wall_sec")),
        "fits_read_wall_sec": _num(aggregate.get("fits_read_wall_sec")),
        "selection_wall_sec": _num(aggregate.get("selection_wall_sec")),
        "table_wall_sec": _num(aggregate.get("table_wall_sec")),
        "frame_compute_wall_sec": _num(aggregate.get("frame_compute_wall_sec")),
        "shard_submit_wall_sec": _num(aggregate.get("shard_submit_wall_sec")),
        "async_shard_write_wait_wall_sec": _num(aggregate.get("async_shard_write_wait_wall_sec")),
        "write_wall_sec": _num(aggregate.get("shard_write_wall_sec") or aggregate.get("write_wall_sec")),
        "shard_total_bytes": _num(aggregate.get("shard_total_bytes")),
        "shard_bytes_per_measurement": _num(aggregate.get("shard_bytes_per_measurement")),
        "max_worker_staging_wall_sec": _num(aggregate.get("max_worker_staging_wall_sec")),
        "max_worker_payload_wait_wall_sec": _num(aggregate.get("max_worker_payload_wait_wall_sec")),
        "max_worker_fits_read_wall_sec": _num(aggregate.get("max_worker_fits_read_wall_sec")),
        "max_worker_frame_upload_wall_sec": _num(aggregate.get("max_worker_frame_upload_wall_sec")),
        "max_worker_aperture_kernel_wall_sec": _num(aggregate.get("max_worker_aperture_kernel_wall_sec")),
        "max_worker_psf_kernel_wall_sec": _num(aggregate.get("max_worker_psf_kernel_wall_sec")),
        "max_worker_psf_device_submit_sync_wall_sec": _num(
            aggregate.get("max_worker_psf_device_submit_sync_wall_sec")
        ),
        "max_worker_psf_spline_coeff_wall_sec": _num(aggregate.get("max_worker_psf_spline_coeff_wall_sec")),
        "max_worker_psf_gather_wall_sec": _num(aggregate.get("max_worker_psf_gather_wall_sec")),
        "max_worker_table_wall_sec": _num(aggregate.get("max_worker_table_wall_sec")),
        "max_worker_shard_write_wall_sec": _num(aggregate.get("max_worker_shard_write_wall_sec")),
        "max_worker_shard_bytes": _num(aggregate.get("max_worker_shard_bytes")),
        "max_worker_async_shard_write_wait_wall_sec": _num(
            aggregate.get("max_worker_async_shard_write_wait_wall_sec")
        ),
        "psf_candidate_grid_wall_sec": _num(aggregate.get("psf_candidate_grid_wall_sec")),
        "psf_kernel_wall_sec": _num(aggregate.get("psf_kernel_wall_sec")),
        "psf_device_submit_sync_wall_sec": psf_device_submit_sync_wall,
        "psf_spline_coeff_wall_sec": _num(aggregate.get("psf_spline_coeff_wall_sec")),
        "psf_upload_wall_sec": _num(aggregate.get("psf_upload_wall_sec")),
        "psf_gather_wall_sec": _num(aggregate.get("psf_gather_wall_sec")),
        "measurements_per_sec_total": _rate(measurement_rows, total_wall),
        "measurements_per_sec_worker_phase": _rate(measurement_rows, worker_wall),
        "measurements_per_sec_worker_payload": _rate(measurement_rows, worker_max_wall),
        "measurements_per_sec_per_gpu_total": _rate(measurement_rows, total_wall * gpu_count),
        "measurements_per_sec_per_gpu_payload": _rate(measurement_rows, worker_max_wall * gpu_count),
        "frames_per_sec_total": _rate(completed_frames, total_wall),
        "frames_per_sec_worker_phase": _rate(completed_frames, worker_wall),
        "frames_per_sec_worker_payload": _rate(completed_frames, worker_max_wall),
        "psf_candidates_per_sec_total": _rate(psf_candidate_count, total_wall),
        "psf_candidates_per_sec_worker_payload": _rate(psf_candidate_count, worker_max_wall),
        "psf_candidates_per_sec_device_submit_sync": _rate(psf_candidate_count, psf_device_submit_sync_wall),
    }


def _failed_trial_row(
    *,
    config: DispatchBenchmarkSweepConfig,
    settings: dict[str, int],
    trial_index: int,
    trial_run_id: str,
    trial_dir: Path,
    error: str,
    elapsed: float,
) -> dict[str, Any]:
    row = _trial_row(
        config=config,
        settings=settings,
        trial_index=trial_index,
        trial_run_id=trial_run_id,
        trial_dir=trial_dir,
        summary={},
        status="failed",
        error=error,
        elapsed=elapsed,
    )
    row["total_wall_sec"] = elapsed
    return row


def _profile_rows_for_trial(row: dict[str, Any], summary: dict[str, Any]) -> list[dict[str, Any]]:
    total = float(row.get("total_wall_sec") or 0.0)
    stages = [
        ("plan_dispatch", row.get("plan_wall_sec"), 0, 0),
        ("worker_phase", row.get("worker_wall_sec"), 0, row.get("measurement_rows")),
        ("staging", row.get("staging_wall_sec"), 0, row.get("measurement_rows")),
        ("payload_wait", row.get("payload_wait_wall_sec"), 0, row.get("measurement_rows")),
        ("fits_read", row.get("fits_read_wall_sec"), 0, row.get("measurement_rows")),
        ("selection", row.get("selection_wall_sec"), row.get("measurement_rows"), row.get("measurement_rows")),
        ("frame_upload", row.get("frame_upload_wall_sec"), row.get("measurement_rows"), row.get("measurement_rows")),
        ("aperture_kernel", row.get("aperture_kernel_wall_sec"), row.get("measurement_rows"), row.get("measurement_rows")),
        ("psf_candidate_grid", row.get("psf_candidate_grid_wall_sec"), row.get("psf_measurement_rows"), row.get("psf_candidate_count")),
        ("psf_kernel_total", row.get("psf_kernel_wall_sec"), row.get("measurement_rows"), row.get("psf_measurement_rows")),
        ("psf_device_submit_sync", row.get("psf_device_submit_sync_wall_sec"), row.get("psf_candidate_count"), row.get("psf_measurement_rows")),
        ("psf_spline_coeff", row.get("psf_spline_coeff_wall_sec"), row.get("psf_measurement_rows"), row.get("psf_measurement_rows")),
        ("psf_upload", row.get("psf_upload_wall_sec"), row.get("psf_measurement_rows"), row.get("psf_candidate_count")),
        ("psf_gather", row.get("psf_gather_wall_sec"), row.get("psf_candidate_count"), row.get("psf_measurement_rows")),
        ("table_assembly", row.get("table_wall_sec"), row.get("measurement_rows"), row.get("measurement_rows")),
        ("shard_submit", row.get("shard_submit_wall_sec"), row.get("measurement_rows"), row.get("measurement_rows")),
        ("async_shard_write_wait", row.get("async_shard_write_wait_wall_sec"), row.get("measurement_rows"), row.get("measurement_rows")),
        ("write_measurement_shards", row.get("write_wall_sec"), row.get("measurement_rows"), row.get("measurement_rows")),
        ("finalize_total", row.get("finalize_wall_sec"), row.get("measurement_rows"), row.get("measurement_rows")),
        ("collect_shards", row.get("collect_wall_sec"), 0, row.get("measurement_rows")),
        ("assemble_spectra", row.get("spectra_total_wall_sec"), row.get("measurement_rows"), row.get("measurement_rows")),
        ("baseline_score", row.get("baseline_score_wall_sec"), row.get("measurement_rows"), row.get("baseline_candidate_count")),
    ]
    out: list[dict[str, Any]] = []
    for stage, wall, rows_in, rows_out in stages:
        wall_time = _num(wall)
        if wall_time <= 0:
            continue
        out.append(
            {
                "run_id": row["run_id"],
                "stage": stage,
                "function_or_script": "luxquarry-allsky",
                "wall_time_sec": wall_time,
                "wall_time_pct": float(100.0 * wall_time / total) if total > 0 else 0.0,
                "cpu_time_sec": None,
                "gpu_time_sec": None,
                "io_wait_sec": None,
                "call_count": 1,
                "rows_in": int(rows_in or 0),
                "rows_out": int(rows_out or 0),
                "bytes_in": None,
                "bytes_out": None,
                "backend": _stage_backend(stage, summary),
            }
        )
    return out


def _stage_backend(stage: str, summary: dict[str, Any]) -> str:
    if stage == "assemble_spectra":
        return str(((summary.get("finalize") or {}).get("spectra") or {}).get("backend") or "cudf_spectra_assembly")
    if stage == "baseline_score":
        return str(
            ((summary.get("finalize") or {}).get("baseline_scoring") or {}).get("backend")
            or "cudf_simple_target_zscore_scorer"
        )
    if stage == "worker_phase":
        return "persistent_gpu_frame_worker"
    if stage.startswith("psf_"):
        return "warp_gpu_psf_grid"
    if stage == "aperture_kernel":
        return "warp_gpu_aperture"
    if stage in {"fits_read", "staging", "write_measurement_shards", "shard_submit", "async_shard_write_wait"}:
        return "fits_parquet_io"
    if stage == "frame_upload":
        return "warp_frame_device_upload"
    if stage in {"payload_wait", "selection", "table_assembly"}:
        return "persistent_gpu_frame_worker"
    return "luxquarry_local_dispatch_runner"


def _write_outputs(
    config: DispatchBenchmarkSweepConfig,
    rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows, columns=_result_columns())
    profiles = pd.DataFrame(profile_rows, columns=_profile_columns())
    results_path = config.output_dir / "sweep_results.parquet"
    results_json_path = config.output_dir / "sweep_results.json"
    profile_path = config.output_dir / "profile_summary.parquet"
    profile_json_path = config.output_dir / "profile_summary.json"
    results.to_parquet(results_path, index=False)
    results_json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    profiles.to_parquet(profile_path, index=False)
    profile_json_path.write_text(json.dumps({"rows": profile_rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    complete = results[results["status"] == "complete"] if not results.empty else results
    best = {}
    best_payload = {}
    if not complete.empty:
        best = complete.sort_values("measurements_per_sec_total", ascending=False).iloc[0].to_dict()
        best_payload = complete.sort_values("measurements_per_sec_worker_payload", ascending=False).iloc[0].to_dict()
    total_wall = time.perf_counter() - started
    perf = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "luxquarry_dispatch_benchmark_sweep",
        "run_id": config.run_id,
        "output_dir": str(config.output_dir),
        "manifest_path": str(config.manifest_path),
        "projected_targets_path": str(config.projected_targets_path),
        "devices": list(config.devices),
        "worker_only": config.worker_only,
        "trial_count": len(rows),
        "complete_trial_count": int((results["status"] == "complete").sum()) if not results.empty else 0,
        "failed_trial_count": int((results["status"] == "failed").sum()) if not results.empty else 0,
        "total_wall_sec": total_wall,
        "best_trial": _json_safe(best),
        "best_payload_trial": _json_safe(best_payload),
        "outputs": {
            "sweep_results_parquet": str(results_path),
            "sweep_results_json": str(results_json_path),
            "profile_summary_parquet": str(profile_path),
            "profile_summary_json": str(profile_json_path),
            "perf_summary_json": str(config.output_dir / "perf_summary.json"),
        },
    }
    (config.output_dir / "perf_summary.json").write_text(
        json.dumps(perf, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return perf


def _json_safe(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            out[key] = None
        elif hasattr(value, "item"):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def _result_columns() -> list[str]:
    return [
        "trial_index",
        "run_id",
        "status",
        "error",
        "output_dir",
        "devices",
        "gpu_count",
        "workers_per_device",
        "worker_count",
        "limit_frames",
        "completed_frames",
        "shard_batch_frames",
        "prefetch_frames",
        "repetition",
        "measurement_rows",
        "psf_measurement_rows",
        "ok_psf_rows",
        "psf_candidate_count",
        "target_count",
        "enable_psf",
        "discard_measurement_shards",
        "measurement_column_profile",
        "measurement_parquet_compression",
        "aperture_radius_pix",
        "annulus_inner_pix",
        "annulus_outer_pix",
        "psf_kernel_build_mode",
        "psf_grid_half_range_pix",
        "psf_grid_step_pix",
        "plan_wall_sec",
        "worker_wall_sec",
        "finalize_wall_sec",
        "total_wall_sec",
        "collect_wall_sec",
        "worker_min_wall_sec",
        "worker_avg_wall_sec",
        "worker_max_wall_sec",
        "worker_sum_wall_sec",
        "worker_parallel_efficiency",
        "worker_wall_skew_ratio",
        "worker_launch_overhead_sec",
        "read_shards_wall_sec",
        "sort_wall_sec",
        "write_spectra_wall_sec",
        "target_summary_wall_sec",
        "spectra_total_wall_sec",
        "baseline_score_wall_sec",
        "baseline_candidate_count",
        "staging_wall_sec",
        "payload_wait_wall_sec",
        "frame_upload_wall_sec",
        "aperture_kernel_wall_sec",
        "fits_read_wall_sec",
        "selection_wall_sec",
        "table_wall_sec",
        "frame_compute_wall_sec",
        "shard_submit_wall_sec",
        "async_shard_write_wait_wall_sec",
        "write_wall_sec",
        "shard_total_bytes",
        "shard_bytes_per_measurement",
        "max_worker_staging_wall_sec",
        "max_worker_payload_wait_wall_sec",
        "max_worker_fits_read_wall_sec",
        "max_worker_frame_upload_wall_sec",
        "max_worker_aperture_kernel_wall_sec",
        "max_worker_psf_kernel_wall_sec",
        "max_worker_psf_device_submit_sync_wall_sec",
        "max_worker_psf_spline_coeff_wall_sec",
        "max_worker_psf_gather_wall_sec",
        "max_worker_table_wall_sec",
        "max_worker_shard_write_wall_sec",
        "max_worker_shard_bytes",
        "max_worker_async_shard_write_wait_wall_sec",
        "psf_candidate_grid_wall_sec",
        "psf_kernel_wall_sec",
        "psf_device_submit_sync_wall_sec",
        "psf_spline_coeff_wall_sec",
        "psf_upload_wall_sec",
        "psf_gather_wall_sec",
        "measurements_per_sec_total",
        "measurements_per_sec_worker_phase",
        "measurements_per_sec_worker_payload",
        "measurements_per_sec_per_gpu_total",
        "measurements_per_sec_per_gpu_payload",
        "frames_per_sec_total",
        "frames_per_sec_worker_phase",
        "frames_per_sec_worker_payload",
        "psf_candidates_per_sec_total",
        "psf_candidates_per_sec_worker_payload",
        "psf_candidates_per_sec_device_submit_sync",
    ]


def _profile_columns() -> list[str]:
    return [
        "run_id",
        "stage",
        "function_or_script",
        "wall_time_sec",
        "wall_time_pct",
        "cpu_time_sec",
        "gpu_time_sec",
        "io_wait_sec",
        "call_count",
        "rows_in",
        "rows_out",
        "bytes_in",
        "bytes_out",
        "backend",
    ]


def _num(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _rate(count: int | float, elapsed_sec: float) -> float:
    return float(count / elapsed_sec) if elapsed_sec > 0 and count else 0.0
