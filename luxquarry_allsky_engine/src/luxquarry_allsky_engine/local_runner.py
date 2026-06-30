from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dispatch import DispatchPlanConfig, build_dispatch_plan, write_dispatch_plan
from .finalize import FinalizeDispatchConfig, finalize_dispatch_run
from .status import DispatchStatusConfig, write_dispatch_status_snapshot


@dataclass(frozen=True)
class LocalDispatchRunConfig:
    manifest_path: Path
    projected_targets_path: Path
    output_dir: Path
    run_id: str
    devices: tuple[str, ...]
    workers_per_device: int = 1
    cache_root: Path = Path("/mnt/niroseti/spherex_cache")
    limit_frames: int | None = None
    executable: str = ".venv/bin/luxquarry-allsky"
    shard_batch_frames: int = 1
    prefetch_frames: int = 0
    status_interval_frames: int = 1
    local_cache_dir: Path | None = None
    async_shard_writes: bool = False
    batch_table_assembly: bool = False
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
    spectra_out_dir: Path | None = None
    spectra_run_id: str | None = None
    only_ok: bool = False
    allow_incomplete_finalize: bool = False
    campaign_id: str | None = None
    campaign_contract_out: Path | None = None
    candidate_dir: Path | None = None
    score_baseline: bool = False
    candidate_min_abs_zscore: float = 5.0
    candidate_min_measurements: int = 10
    candidate_max_rows: int | None = None
    score_injected: bool = False
    injected_spectra_dir: Path | None = None
    injection_truth_path: Path | None = None
    recover_injections: bool = False
    recovery_min_score: float = 5.0
    recovery_wavelength_tolerance_nm: float = 10.0
    recovery_require_line_family: bool = False
    resume: bool = False
    status_snapshot_interval_sec: float = 1.0


@dataclass(frozen=True)
class LocalPlanWorkerRunConfig:
    plan_path: Path
    logs_dir: Path | None = None
    resume: bool = False
    status_snapshot_interval_sec: float = 1.0
    allow_failed_workers: bool = False


def run_local_dispatch(config: LocalDispatchRunConfig) -> dict[str, Any]:
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = config.output_dir / "dispatch_plan.json"
    logs_dir = config.output_dir / "worker_logs"
    logs_dir.mkdir(exist_ok=True)

    t_plan = time.perf_counter()
    plan = build_dispatch_plan(
        DispatchPlanConfig(
            manifest_path=config.manifest_path,
            projected_targets_path=config.projected_targets_path,
            output_dir=config.output_dir,
            run_id=config.run_id,
            devices=config.devices,
            workers_per_device=config.workers_per_device,
            cache_root=config.cache_root,
            limit_frames=config.limit_frames,
            executable=config.executable,
            shard_batch_frames=config.shard_batch_frames,
            prefetch_frames=config.prefetch_frames,
            status_interval_frames=config.status_interval_frames,
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

    t_workers = time.perf_counter()
    worker_results = _run_workers(
        plan,
        logs_dir,
        plan_path=plan_path,
        resume=config.resume,
        status_snapshot_interval_sec=config.status_snapshot_interval_sec,
    )
    worker_wall = time.perf_counter() - t_workers
    failed_workers = [row for row in worker_results if row["returncode"] != 0]
    if failed_workers and not config.allow_incomplete_finalize:
        summary = _summary(
            config=config,
            plan_path=plan_path,
            plan=plan,
            plan_wall=plan_wall,
            worker_wall=worker_wall,
            worker_results=worker_results,
            finalize=None,
            started=started,
            status="worker_failed",
        )
        _write_summary(config.output_dir / "local_dispatch_summary.json", summary)
        raise RuntimeError(f"{len(failed_workers)} local worker(s) failed; see {logs_dir}")

    t_finalize = time.perf_counter()
    finalize = finalize_dispatch_run(
        FinalizeDispatchConfig(
            plan_path=plan_path,
            device=config.finalize_device,
            spectra_out_dir=config.spectra_out_dir or config.output_dir / "spectra",
            spectra_run_id=config.spectra_run_id or config.run_id,
            only_ok=config.only_ok,
            allow_incomplete=config.allow_incomplete_finalize,
            campaign_id=config.campaign_id or f"{config.run_id}_campaign",
            campaign_contract_out=config.campaign_contract_out or config.output_dir / "campaign_contract.json",
            candidate_dir=config.candidate_dir or config.output_dir / "candidates",
            score_baseline=config.score_baseline,
            candidate_min_abs_zscore=config.candidate_min_abs_zscore,
            candidate_min_measurements=config.candidate_min_measurements,
            candidate_max_rows=config.candidate_max_rows,
            score_injected=config.score_injected,
            injected_spectra_dir=config.injected_spectra_dir,
            injection_truth_path=config.injection_truth_path,
            recover_injections=config.recover_injections,
            recovery_min_score=config.recovery_min_score,
            recovery_wavelength_tolerance_nm=config.recovery_wavelength_tolerance_nm,
            recovery_require_line_family=config.recovery_require_line_family,
        )
    )
    finalize_wall = time.perf_counter() - t_finalize
    summary = _summary(
        config=config,
        plan_path=plan_path,
        plan=plan,
        plan_wall=plan_wall,
        worker_wall=worker_wall,
        worker_results=worker_results,
        finalize=finalize,
        started=started,
        status="complete",
    )
    summary["finalize_wall_sec"] = finalize_wall
    _write_summary(config.output_dir / "local_dispatch_summary.json", summary)
    return summary


def run_dispatch_plan_workers(config: LocalPlanWorkerRunConfig) -> dict[str, Any]:
    started = time.perf_counter()
    plan = json.loads(config.plan_path.read_text(encoding="utf-8"))
    run_output_dir = _run_output_dir(config.plan_path, plan)
    logs_dir = config.logs_dir or run_output_dir / "worker_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    t_workers = time.perf_counter()
    worker_results = _run_workers(
        plan,
        logs_dir,
        plan_path=config.plan_path,
        resume=config.resume,
        status_snapshot_interval_sec=config.status_snapshot_interval_sec,
    )
    worker_wall = time.perf_counter() - t_workers
    failed_workers = [row for row in worker_results if row["returncode"] != 0]
    status = "complete" if not failed_workers else "worker_failed"
    summary = _worker_only_summary(
        config=config,
        plan=plan,
        logs_dir=logs_dir,
        worker_wall=worker_wall,
        worker_results=worker_results,
        started=started,
        status=status,
    )
    summary_path = run_output_dir / "worker_only_summary.json"
    _write_summary(summary_path, summary)
    if failed_workers and not config.allow_failed_workers:
        raise RuntimeError(f"{len(failed_workers)} local worker(s) failed; see {logs_dir}")
    return summary


def _run_workers(
    plan: dict[str, Any],
    logs_dir: Path,
    *,
    plan_path: Path,
    resume: bool,
    status_snapshot_interval_sec: float,
) -> list[dict[str, Any]]:
    running: list[tuple[dict[str, Any], subprocess.Popen[bytes], Any, Any, float]] = []
    results: list[dict[str, Any]] = []
    for worker in plan.get("workers") or []:
        if resume:
            skipped = _completed_worker_result(worker, logs_dir)
            if skipped is not None:
                results.append(skipped)
                continue
        worker_id = str(worker["worker_id"])
        stdout_path = logs_dir / f"{worker_id}.stdout.log"
        stderr_path = logs_dir / f"{worker_id}.stderr.log"
        stdout_file = stdout_path.open("wb")
        stderr_file = stderr_path.open("wb")
        started = time.perf_counter()
        proc = subprocess.Popen(list(worker["argv"]), stdout=stdout_file, stderr=stderr_file)
        running.append((worker, proc, stdout_file, stderr_file, started))

    next_snapshot = 0.0
    while running:
        now = time.perf_counter()
        if status_snapshot_interval_sec > 0 and now >= next_snapshot:
            write_dispatch_status_snapshot(DispatchStatusConfig(plan_path=plan_path))
            next_snapshot = now + status_snapshot_interval_sec

        still_running: list[tuple[dict[str, Any], subprocess.Popen[bytes], Any, Any, float]] = []
        for worker, proc, stdout_file, stderr_file, started in running:
            returncode = proc.poll()
            if returncode is None:
                still_running.append((worker, proc, stdout_file, stderr_file, started))
                continue
            results.append(_worker_process_result(worker, logs_dir, int(returncode), started))
            stdout_file.close()
            stderr_file.close()
        running = still_running
        if running:
            time.sleep(_poll_sleep(status_snapshot_interval_sec))

    write_dispatch_status_snapshot(DispatchStatusConfig(plan_path=plan_path))
    return results


def _worker_process_result(worker: dict[str, Any], logs_dir: Path, returncode: int, started: float) -> dict[str, Any]:
    worker_id = str(worker["worker_id"])
    return {
        "worker_id": worker_id,
        "worker_index": worker.get("worker_index"),
        "device": worker.get("device"),
        "returncode": returncode,
        "skipped": False,
        "wall_sec": time.perf_counter() - started,
        "stdout_log": str(logs_dir / f"{worker_id}.stdout.log"),
        "stderr_log": str(logs_dir / f"{worker_id}.stderr.log"),
        "argv": list(worker["argv"]),
    }


def _poll_sleep(status_snapshot_interval_sec: float) -> float:
    if status_snapshot_interval_sec <= 0:
        return 0.25
    return min(0.25, max(0.05, status_snapshot_interval_sec / 4.0))


def _completed_worker_result(worker: dict[str, Any], logs_dir: Path) -> dict[str, Any] | None:
    worker_id = str(worker["worker_id"])
    output_dir = Path(str(worker.get("output_dir") or ""))
    summary_path = output_dir / "run_summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not summary.get("completed_utc") or int(summary.get("failed_frames") or 0) != 0:
        return None
    return {
        "worker_id": worker_id,
        "worker_index": worker.get("worker_index"),
        "device": worker.get("device"),
        "returncode": 0,
        "skipped": True,
        "wall_sec": 0.0,
        "stdout_log": str(logs_dir / f"{worker_id}.stdout.log"),
        "stderr_log": str(logs_dir / f"{worker_id}.stderr.log"),
        "argv": list(worker["argv"]),
        "summary_path": str(summary_path),
        "completed_frames": int(summary.get("completed_frames") or 0),
        "measurement_rows": int(summary.get("measurement_rows") or 0),
        "ok_measurement_rows": int(summary.get("ok_measurement_rows") or 0),
    }


def _summary(
    *,
    config: LocalDispatchRunConfig,
    plan_path: Path,
    plan: dict[str, Any],
    plan_wall: float,
    worker_wall: float,
    worker_results: list[dict[str, Any]],
    finalize: dict[str, Any] | None,
    started: float,
    status: str,
) -> dict[str, Any]:
    measurement_rows = int((finalize or {}).get("measurement_rows") or 0)
    elapsed = time.perf_counter() - started
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "luxquarry_local_dispatch_runner",
        "status": status,
        "run_id": config.run_id,
        "plan_path": str(plan_path),
        "output_dir": str(config.output_dir),
        "devices": list(config.devices),
        "resume": config.resume,
        "status_snapshot_interval_sec": config.status_snapshot_interval_sec,
        "worker_count": int(plan.get("worker_count") or len(worker_results)),
        "launched_worker_count": sum(1 for row in worker_results if not row.get("skipped")),
        "skipped_worker_count": sum(1 for row in worker_results if row.get("skipped")),
        "materialize_worker_inputs": bool(plan.get("materialize_worker_inputs")),
        "plan_wall_sec": plan_wall,
        "worker_wall_sec": worker_wall,
        "finalize_wall_sec": None,
        "total_wall_sec": elapsed,
        "measurement_rows": measurement_rows,
        "measurements_per_sec": float(measurement_rows / elapsed) if elapsed > 0 and measurement_rows else 0.0,
        "failed_worker_count": sum(1 for row in worker_results if row["returncode"] != 0),
        "worker_results": worker_results,
        "finalize": finalize,
    }


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_output_dir(plan_path: Path, plan: dict[str, Any]) -> Path:
    configured = Path(str(plan.get("output_dir") or plan_path.parent))
    if configured.is_absolute() or configured.exists():
        return configured
    if plan_path.parent.name == configured.name:
        return plan_path.parent
    return configured


def _worker_only_summary(
    *,
    config: LocalPlanWorkerRunConfig,
    plan: dict[str, Any],
    logs_dir: Path,
    worker_wall: float,
    worker_results: list[dict[str, Any]],
    started: float,
    status: str,
) -> dict[str, Any]:
    elapsed = time.perf_counter() - started
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "luxquarry_worker_only_runner",
        "status": status,
        "run_id": str(plan.get("run_id") or config.plan_path.parent.name),
        "plan_path": str(config.plan_path),
        "output_dir": str(_run_output_dir(config.plan_path, plan)),
        "logs_dir": str(logs_dir),
        "devices": list(plan.get("devices") or []),
        "resume": config.resume,
        "status_snapshot_interval_sec": config.status_snapshot_interval_sec,
        "worker_count": int(plan.get("worker_count") or len(worker_results)),
        "launched_worker_count": sum(1 for row in worker_results if not row.get("skipped")),
        "skipped_worker_count": sum(1 for row in worker_results if row.get("skipped")),
        "materialize_worker_inputs": bool(plan.get("materialize_worker_inputs")),
        "worker_wall_sec": worker_wall,
        "total_wall_sec": elapsed,
        "failed_worker_count": sum(1 for row in worker_results if row["returncode"] != 0),
        "worker_results": worker_results,
    }
