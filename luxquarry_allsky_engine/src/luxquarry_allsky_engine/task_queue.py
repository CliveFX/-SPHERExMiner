from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .gpu_worker import PersistentGpuFrameWorker, PersistentWorkerConfig


@dataclass(frozen=True)
class TaskQueueWriteConfig:
    manifest_path: Path
    projected_targets_path: Path
    output_dir: Path
    campaign_id: str
    frames_per_task: int = 25
    limit_frames: int | None = None


@dataclass(frozen=True)
class GpuWorkerServiceConfig:
    queue_dir: Path
    output_dir: Path
    run_id: str
    worker_id: str
    worker_config: PersistentWorkerConfig
    max_tasks: int | None = None


@dataclass(frozen=True)
class TaskQueueCollectConfig:
    queue_dir: Path
    output_path: Path


def write_task_queue(config: TaskQueueWriteConfig) -> dict[str, Any]:
    if config.frames_per_task <= 0:
        raise ValueError("frames_per_task must be positive")

    started = time.perf_counter()
    queue_dir = config.output_dir
    for child in ["pending", "leased", "complete", "failed", "task_inputs"]:
        (queue_dir / child).mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(config.manifest_path)
    if config.limit_frames is not None:
        manifest = manifest.head(config.limit_frames).copy()
    manifest = manifest.reset_index(drop=True)
    targets = pd.read_parquet(config.projected_targets_path)
    targets["frame_group_id"] = targets["frame_group_id"].astype(str)

    tasks: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(manifest), config.frames_per_task)):
        frame_slice = manifest.iloc[start : start + config.frames_per_task].copy()
        frame_ids = [str(value) for value in frame_slice["frame_group_id"].tolist()]
        target_slice = targets[targets["frame_group_id"].isin(frame_ids)].copy()
        task_id = f"task_{batch_index:06d}"
        input_dir = queue_dir / "task_inputs" / task_id
        input_dir.mkdir(parents=True, exist_ok=True)
        task_manifest_path = input_dir / "frame_manifest.parquet"
        task_targets_path = input_dir / "projected_targets.parquet"
        frame_slice.to_parquet(task_manifest_path, index=False)
        target_slice.to_parquet(task_targets_path, index=False)
        task = {
            "task_id": task_id,
            "campaign_id": config.campaign_id,
            "batch_index": batch_index,
            "frame_group_ids": frame_ids,
            "frame_count": int(len(frame_slice)),
            "projected_target_rows": int(len(target_slice)),
            "manifest_path": str(task_manifest_path),
            "projected_targets_path": str(task_targets_path),
            "attempt": 0,
            "status": "pending",
            "created_utc": _utc_now(),
        }
        _write_json_atomic(queue_dir / "pending" / f"{task_id}.json", task)
        tasks.append(task)

    queue_manifest = {
        "created_utc": _utc_now(),
        "campaign_id": config.campaign_id,
        "queue_dir": str(queue_dir),
        "source_manifest_path": str(config.manifest_path),
        "source_projected_targets_path": str(config.projected_targets_path),
        "frames_per_task": int(config.frames_per_task),
        "frame_count": int(len(manifest)),
        "projected_target_rows": int(len(targets)),
        "task_count": len(tasks),
        "write_wall_sec": time.perf_counter() - started,
        "tasks": [
            {
                "task_id": task["task_id"],
                "frame_count": task["frame_count"],
                "projected_target_rows": task["projected_target_rows"],
                "frame_group_ids": task["frame_group_ids"],
            }
            for task in tasks
        ],
    }
    _write_json_atomic(queue_dir / "queue_manifest.json", queue_manifest)
    return queue_manifest


def run_gpu_worker_service(config: GpuWorkerServiceConfig) -> dict[str, Any]:
    started = time.perf_counter()
    queue_dir = config.queue_dir
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    status_dir = output_dir / "status" / "workers"
    status_dir.mkdir(parents=True, exist_ok=True)
    task_output_root = output_dir / "tasks"
    task_output_root.mkdir(parents=True, exist_ok=True)

    worker = PersistentGpuFrameWorker(config.worker_config)
    summary: dict[str, Any] = {
        "created_utc": _utc_now(),
        "run_id": config.run_id,
        "worker_id": config.worker_id,
        "queue_dir": str(queue_dir),
        "output_dir": str(output_dir),
        "device": config.worker_config.device,
        "tasks_completed": 0,
        "tasks_failed": 0,
        "frames_completed": 0,
        "measurement_rows": 0,
        "ok_measurement_rows": 0,
        "failed_frames": 0,
        "task_summaries": [],
    }
    _write_service_status(status_dir / f"{config.worker_id}.json", summary, started, state="running")

    while config.max_tasks is None or summary["tasks_completed"] + summary["tasks_failed"] < config.max_tasks:
        claimed = _claim_next_task(queue_dir, config.worker_id)
        if claimed is None:
            break
        lease_path, task = claimed
        task_id = str(task["task_id"])
        task_started = time.perf_counter()
        task_out = task_output_root / task_id
        task_status_path = output_dir / "status" / "tasks" / f"{task_id}.json"
        task_run_id = f"{config.run_id}.{task_id}.{config.worker_id}"
        try:
            task_summary = worker.run(
                manifest_path=Path(str(task["manifest_path"])),
                projected_targets_path=Path(str(task["projected_targets_path"])),
                output_dir=task_out,
                run_id=task_run_id,
                status_path=task_status_path,
            )
            completed = {
                **task,
                "status": "complete",
                "completed_utc": _utc_now(),
                "worker_id": config.worker_id,
                "task_output_dir": str(task_out),
                "summary_path": str(task_out / "run_summary.json"),
                "task_wall_sec": time.perf_counter() - task_started,
                "worker_summary": _compact_worker_summary(task_summary),
            }
            _write_json_atomic(queue_dir / "complete" / f"{task_id}.json", completed)
            lease_path.unlink(missing_ok=True)
            summary["tasks_completed"] += 1
            summary["frames_completed"] += int(task_summary.get("completed_frames") or 0)
            summary["measurement_rows"] += int(task_summary.get("measurement_rows") or 0)
            summary["ok_measurement_rows"] += int(task_summary.get("ok_measurement_rows") or 0)
            summary["failed_frames"] += int(task_summary.get("failed_frames") or 0)
            summary["task_summaries"].append(completed)
        except Exception as exc:
            failed = {
                **task,
                "status": "failed",
                "failed_utc": _utc_now(),
                "worker_id": config.worker_id,
                "task_output_dir": str(task_out),
                "task_wall_sec": time.perf_counter() - task_started,
                "error": f"{type(exc).__name__}: {exc}",
            }
            _write_json_atomic(queue_dir / "failed" / f"{task_id}.{config.worker_id}.json", failed)
            lease_path.unlink(missing_ok=True)
            summary["tasks_failed"] += 1
            summary["task_summaries"].append(failed)
        summary["total_wall_sec"] = time.perf_counter() - started
        _write_service_status(status_dir / f"{config.worker_id}.json", summary, started, state="running")

    summary["completed_utc"] = _utc_now()
    summary["total_wall_sec"] = time.perf_counter() - started
    _write_json_atomic(output_dir / f"{config.worker_id}.service_summary.json", summary)
    _write_service_status(status_dir / f"{config.worker_id}.json", summary, started, state="complete")
    return summary


def collect_task_queue_run(config: TaskQueueCollectConfig) -> dict[str, Any]:
    started = time.perf_counter()
    complete_paths = sorted((config.queue_dir / "complete").glob("*.json"))
    failed_paths = sorted((config.queue_dir / "failed").glob("*.json"))
    shard_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []

    for complete_path in complete_paths:
        task = json.loads(complete_path.read_text(encoding="utf-8"))
        worker_summary = task.get("worker_summary") or {}
        task_rows.append(
            {
                "task_id": task.get("task_id"),
                "status": "complete",
                "worker_id": task.get("worker_id"),
                "frame_count": int(worker_summary.get("frame_count") or task.get("frame_count") or 0),
                "completed_frames": int(worker_summary.get("completed_frames") or 0),
                "measurement_rows": int(worker_summary.get("measurement_rows") or 0),
                "ok_measurement_rows": int(worker_summary.get("ok_measurement_rows") or 0),
                "failed_frames": int(worker_summary.get("failed_frames") or 0),
                "task_wall_sec": float(task.get("task_wall_sec") or 0.0),
                "summary_path": task.get("summary_path"),
            }
        )
        for shard in worker_summary.get("shards") or []:
            shard_path = Path(str(shard.get("path")))
            shard_rows.append(
                {
                    "worker_id": task.get("worker_id"),
                    "worker_index": 0,
                    "device": worker_summary.get("device"),
                    "path": str(shard_path),
                    "rows": int(shard.get("rows") or 0),
                    "ok_rows": int(shard.get("ok_rows") or 0),
                    "frame_count": int(shard.get("frame_count") or 0),
                    "frame_group_ids": ",".join(str(v) for v in shard.get("frame_group_ids") or []),
                    "image_ids": ",".join(str(v) for v in shard.get("image_ids") or []),
                    "write_wall_sec": float(shard.get("write_wall_sec") or 0.0),
                    "exists": shard_path.exists(),
                    "task_id": task.get("task_id"),
                }
            )

    for failed_path in failed_paths:
        task = json.loads(failed_path.read_text(encoding="utf-8"))
        task_rows.append(
            {
                "task_id": task.get("task_id"),
                "status": "failed",
                "worker_id": task.get("worker_id"),
                "frame_count": int(task.get("frame_count") or 0),
                "completed_frames": 0,
                "measurement_rows": 0,
                "ok_measurement_rows": 0,
                "failed_frames": 0,
                "task_wall_sec": float(task.get("task_wall_sec") or 0.0),
                "summary_path": None,
                "error": task.get("error"),
            }
        )

    pending_count = len(list((config.queue_dir / "pending").glob("*.json")))
    leased_count = len(list((config.queue_dir / "leased").glob("*.json")))
    missing_shards = sum(1 for row in shard_rows if not row["exists"])
    shard_manifest_path = config.output_path.with_name("measurement_shard_manifest.parquet")
    shard_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(shard_rows).to_parquet(shard_manifest_path, index=False)
    task_table_path = config.output_path.with_name("task_queue_tasks.parquet")
    pd.DataFrame(task_rows).to_parquet(task_table_path, index=False)

    complete_tasks = sum(1 for row in task_rows if row["status"] == "complete")
    failed_tasks = sum(1 for row in task_rows if row["status"] == "failed")
    summary = {
        "created_utc": _utc_now(),
        "queue_dir": str(config.queue_dir),
        "complete": pending_count == 0 and leased_count == 0 and failed_tasks == 0 and missing_shards == 0,
        "pending_tasks": pending_count,
        "leased_tasks": leased_count,
        "complete_tasks": complete_tasks,
        "failed_tasks": failed_tasks,
        "completed_frames": sum(int(row["completed_frames"]) for row in task_rows),
        "measurement_rows": sum(int(row["measurement_rows"]) for row in task_rows),
        "ok_measurement_rows": sum(int(row["ok_measurement_rows"]) for row in task_rows),
        "failed_frames": sum(int(row["failed_frames"]) for row in task_rows),
        "shard_count": len(shard_rows),
        "missing_shards": missing_shards,
        "worker_sum_wall_sec": sum(float(row["task_wall_sec"]) for row in task_rows),
        "collect_wall_sec": time.perf_counter() - started,
        "shard_manifest_path": str(shard_manifest_path),
        "task_table_path": str(task_table_path),
        "tasks": task_rows,
        "shards": shard_rows,
    }
    _write_json_atomic(config.output_path, summary)
    return summary


def _claim_next_task(queue_dir: Path, worker_id: str) -> tuple[Path, dict[str, Any]] | None:
    for pending_path in sorted((queue_dir / "pending").glob("*.json")):
        lease_path = queue_dir / "leased" / f"{pending_path.stem}.{worker_id}.json"
        try:
            pending_path.replace(lease_path)
        except FileNotFoundError:
            continue
        task = json.loads(lease_path.read_text(encoding="utf-8"))
        task["status"] = "leased"
        task["lease_owner"] = worker_id
        task["leased_utc"] = _utc_now()
        _write_json_atomic(lease_path, task)
        return lease_path, task
    return None


def _compact_worker_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "created_utc",
        "completed_utc",
        "run_id",
        "manifest_path",
        "projected_targets_path",
        "output_dir",
        "device",
        "frame_count",
        "completed_frames",
        "input_projected_rows",
        "measurement_rows",
        "ok_measurement_rows",
        "failed_frames",
        "calibration_upload_count",
        "backend",
        "local_cache_dir",
        "async_shard_writes",
        "batch_table_assembly",
        "total_wall_sec",
        "async_shard_write_wait_wall_sec",
        "queued_shard_writes",
    }
    compact = {key: summary.get(key) for key in keep if key in summary}
    compact["shards"] = summary.get("shards") or []
    frame_timings = summary.get("frame_timings") or []
    compact["frame_timings"] = frame_timings
    return compact


def _write_service_status(path: Path, summary: dict[str, Any], started: float, *, state: str) -> None:
    pending = len(list((Path(summary["queue_dir"]) / "pending").glob("*.json")))
    leased = len(list((Path(summary["queue_dir"]) / "leased").glob("*.json")))
    complete = len(list((Path(summary["queue_dir"]) / "complete").glob("*.json")))
    failed = len(list((Path(summary["queue_dir"]) / "failed").glob("*.json")))
    status = {
        "state": state,
        "updated_utc": _utc_now(),
        "elapsed_sec": time.perf_counter() - started,
        "queue_pending_tasks": pending,
        "queue_leased_tasks": leased,
        "queue_complete_tasks": complete,
        "queue_failed_tasks": failed,
        **summary,
    }
    _write_json_atomic(path, status)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
