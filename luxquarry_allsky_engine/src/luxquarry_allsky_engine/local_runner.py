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
    materialize_worker_inputs: bool = True
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
    resume: bool = False
    status_snapshot_interval_sec: float = 1.0


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
            materialize_worker_inputs=config.materialize_worker_inputs,
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
