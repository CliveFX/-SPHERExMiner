from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DispatchStatusConfig:
    plan_path: Path
    output_path: Path | None = None


def write_dispatch_status_snapshot(config: DispatchStatusConfig) -> dict[str, Any]:
    started = time.perf_counter()
    plan = _read_json(config.plan_path)
    output_dir = _run_output_dir(config.plan_path, plan)
    output_path = config.output_path or output_dir / "dispatch_status.json"
    workers = [_worker_status(config.plan_path, worker) for worker in plan.get("workers") or []]
    states = _state_counts(workers)
    completed_frames = sum(int(worker.get("completed_frames") or 0) for worker in workers)
    frame_count = sum(int(worker.get("frame_count") or 0) for worker in workers)
    measurement_rows = sum(int(worker.get("measurement_rows") or 0) for worker in workers)
    ok_measurement_rows = sum(int(worker.get("ok_measurement_rows") or 0) for worker in workers)
    failed_frames = sum(int(worker.get("failed_frames") or 0) for worker in workers)
    active_workers = sum(1 for worker in workers if worker["state"] == "running")
    complete_workers = sum(1 for worker in workers if worker["state"] == "complete")
    errored_workers = sum(1 for worker in workers if worker["state"] == "error")
    missing_workers = sum(1 for worker in workers if worker["state"] == "missing")
    worker_count = len(workers)
    complete = worker_count > 0 and complete_workers == worker_count and failed_frames == 0
    snapshot = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "luxquarry_dispatch_status_snapshot",
        "plan_path": str(config.plan_path),
        "run_id": plan.get("run_id"),
        "output_dir": str(output_dir),
        "worker_count": worker_count,
        "active_workers": active_workers,
        "complete_workers": complete_workers,
        "errored_workers": errored_workers,
        "missing_workers": missing_workers,
        "complete": complete,
        "state_counts": states,
        "frame_count": frame_count,
        "completed_frames": completed_frames,
        "failed_frames": failed_frames,
        "progress_fraction": float(completed_frames / frame_count) if frame_count else 0.0,
        "measurement_rows": measurement_rows,
        "ok_measurement_rows": ok_measurement_rows,
        "queued_shard_writes": sum(int(worker.get("queued_shard_writes") or 0) for worker in workers),
        "latest_worker_update_utc": _latest([worker.get("updated_utc") for worker in workers]),
        "snapshot_wall_sec": time.perf_counter() - started,
        "workers": workers,
    }
    _write_json_atomic(output_path, snapshot)
    return snapshot


def _worker_status(plan_path: Path, worker: dict[str, Any]) -> dict[str, Any]:
    worker_id = str(worker.get("worker_id"))
    status_path = _resolve_path(plan_path, worker.get("status_path"))
    summary_path = _worker_output_dir(plan_path, worker) / "run_summary.json"
    status = _read_json_if_exists(status_path)
    summary = _read_json_if_exists(summary_path)
    source = "missing"
    payload: dict[str, Any] = {}
    if status:
        source = "status"
        payload = status
    elif summary:
        source = "summary"
        payload = summary

    state = str(payload.get("state") or ("complete" if payload.get("completed_utc") else "missing"))
    if source == "missing":
        state = "missing"
    elif summary_path.exists() and summary and not summary.get("completed_utc"):
        state = "incomplete"
    failed_frames = int(payload.get("failed_frames") or 0)
    if failed_frames and state == "complete":
        state = "error"

    frame_count = int(payload.get("frame_count") or _planned_frame_count(worker) or 0)
    completed_frames = int(payload.get("completed_frames") or 0)
    measurement_rows = int(payload.get("measurement_rows") or 0)
    ok_measurement_rows = int(payload.get("ok_measurement_rows") or 0)
    elapsed_sec = float(payload.get("elapsed_sec") or payload.get("total_wall_sec") or 0.0)
    return {
        "worker_id": worker_id,
        "worker_index": worker.get("worker_index"),
        "device": worker.get("device"),
        "state": state,
        "source": source,
        "status_path": str(status_path),
        "summary_path": str(summary_path),
        "frame_count": frame_count,
        "completed_frames": completed_frames,
        "failed_frames": failed_frames,
        "progress_fraction": float(completed_frames / frame_count) if frame_count else 0.0,
        "measurement_rows": measurement_rows,
        "ok_measurement_rows": ok_measurement_rows,
        "queued_shard_writes": int(payload.get("queued_shard_writes") or 0),
        "elapsed_sec": elapsed_sec,
        "updated_utc": payload.get("updated_utc") or payload.get("completed_utc") or payload.get("created_utc"),
        "error": payload.get("error"),
    }


def _state_counts(workers: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for worker in workers:
        state = str(worker.get("state") or "unknown")
        counts[state] = counts.get(state, 0) + 1
    return counts


def _latest(values: list[Any]) -> str | None:
    clean = [str(value) for value in values if value]
    return max(clean) if clean else None


def _planned_frame_count(worker: dict[str, Any]) -> int | None:
    manifest_path = Path(str(worker.get("manifest_path") or ""))
    if not manifest_path.exists():
        return None
    try:
        import pandas as pd

        return int(len(pd.read_parquet(manifest_path, columns=[])))
    except Exception:
        return None


def _run_output_dir(plan_path: Path, plan: dict[str, Any]) -> Path:
    configured = Path(str(plan.get("output_dir") or plan_path.parent))
    if configured.is_absolute() or configured.exists():
        return configured
    if plan_path.parent.name == configured.name:
        return plan_path.parent
    return configured


def _worker_output_dir(plan_path: Path, worker: dict[str, Any]) -> Path:
    configured = Path(str(worker.get("output_dir") or ""))
    if configured.is_absolute() or configured.exists():
        return configured
    fallback = plan_path.parent / "workers" / str(worker.get("worker_id"))
    return fallback if fallback.exists() else configured


def _resolve_path(plan_path: Path, value: object) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute() or path.exists():
        return path
    for root in [Path.cwd(), *plan_path.parents]:
        candidate = root / path
        if candidate.exists():
            return candidate
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except Exception as exc:
        return {"state": "error", "error": f"{type(exc).__name__}: {exc}"}


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
