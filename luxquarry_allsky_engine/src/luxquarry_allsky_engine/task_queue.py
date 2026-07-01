from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
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
    materialize_task_inputs: bool = True


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
    targets = None
    if config.materialize_task_inputs:
        targets = pd.read_parquet(config.projected_targets_path)
        targets["frame_group_id"] = targets["frame_group_id"].astype(str)

    tasks: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(manifest), config.frames_per_task)):
        frame_slice = manifest.iloc[start : start + config.frames_per_task].copy()
        frame_ids = [str(value) for value in frame_slice["frame_group_id"].tolist()]
        task_id = f"task_{batch_index:06d}"
        task = {
            "task_id": task_id,
            "campaign_id": config.campaign_id,
            "batch_index": batch_index,
            "frame_group_ids": frame_ids,
            "frame_count": int(len(frame_slice)),
            "materialized_inputs": bool(config.materialize_task_inputs),
            "attempt": 0,
            "status": "pending",
            "created_utc": _utc_now(),
        }
        if config.materialize_task_inputs:
            if targets is None:
                raise ValueError("targets must be loaded when materialize_task_inputs is enabled")
            target_slice = targets[targets["frame_group_id"].isin(frame_ids)].copy()
            input_dir = queue_dir / "task_inputs" / task_id
            input_dir.mkdir(parents=True, exist_ok=True)
            task_manifest_path = input_dir / "frame_manifest.parquet"
            task_targets_path = input_dir / "projected_targets.parquet"
            frame_slice.to_parquet(task_manifest_path, index=False)
            target_slice.to_parquet(task_targets_path, index=False)
            task["projected_target_rows"] = int(len(target_slice))
            task["manifest_path"] = str(task_manifest_path)
            task["projected_targets_path"] = str(task_targets_path)
        _write_json_atomic(queue_dir / "pending" / f"{task_id}.json", task)
        tasks.append(task)

    queue_manifest = {
        "created_utc": _utc_now(),
        "campaign_id": config.campaign_id,
        "queue_dir": str(queue_dir),
        "source_manifest_path": str(config.manifest_path),
        "source_projected_targets_path": str(config.projected_targets_path),
        "materialized_task_inputs": bool(config.materialize_task_inputs),
        "frames_per_task": int(config.frames_per_task),
        "frame_count": int(len(manifest)),
        "projected_target_rows": int(len(targets)) if targets is not None else None,
        "task_count": len(tasks),
        "write_wall_sec": time.perf_counter() - started,
        "tasks": [
            {
                "task_id": task["task_id"],
                "frame_count": task["frame_count"],
                "projected_target_rows": task.get("projected_target_rows"),
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

    queue_manifest = _read_queue_manifest(queue_dir)
    source_manifest: pd.DataFrame | None = None
    source_manifest_by_frame: pd.DataFrame | None = None
    source_targets: pd.DataFrame | None = None
    source_target_indices_by_frame: dict[str, np.ndarray] | None = None
    source_input_load_wall = 0.0
    source_input_index_wall = 0.0
    materialized_queue = bool(queue_manifest.get("materialized_task_inputs", True)) if queue_manifest else True
    if not materialized_queue:
        source_input_started = time.perf_counter()
        source_manifest_path = _resolve_queue_path(queue_dir, queue_manifest["source_manifest_path"])
        source_targets_path = _resolve_queue_path(queue_dir, queue_manifest["source_projected_targets_path"])
        source_manifest = pd.read_parquet(source_manifest_path).reset_index(drop=True)
        source_targets = pd.read_parquet(source_targets_path)
        source_manifest["frame_group_id"] = source_manifest["frame_group_id"].astype(str)
        source_targets["frame_group_id"] = source_targets["frame_group_id"].astype(str)
        source_input_load_wall = time.perf_counter() - source_input_started
        source_index_started = time.perf_counter()
        source_manifest_by_frame = source_manifest.set_index("frame_group_id", drop=False)
        source_target_indices_by_frame = {
            str(frame_id): indices
            for frame_id, indices in source_targets.groupby("frame_group_id", sort=False).indices.items()
        }
        source_input_index_wall = time.perf_counter() - source_index_started

    worker = PersistentGpuFrameWorker(config.worker_config)
    summary: dict[str, Any] = {
        "created_utc": _utc_now(),
        "run_id": config.run_id,
        "worker_id": config.worker_id,
        "queue_dir": str(queue_dir),
        "output_dir": str(output_dir),
        "device": config.worker_config.device,
        "materialized_task_inputs": materialized_queue,
        "resident_source_inputs": not materialized_queue,
        "source_input_load_wall_sec": source_input_load_wall,
        "source_input_index_wall_sec": source_input_index_wall,
        "resident_source_frame_count": int(len(source_manifest)) if source_manifest is not None else 0,
        "resident_source_target_rows": int(len(source_targets)) if source_targets is not None else 0,
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
            task_manifest_path: Path | str | None
            task_targets_path: Path | str | None
            input_read_wall = 0.0
            input_select_started = time.perf_counter()
            if "manifest_path" in task and "projected_targets_path" in task:
                input_read_started = time.perf_counter()
                task_manifest_path = Path(str(task["manifest_path"]))
                task_targets_path = Path(str(task["projected_targets_path"]))
                manifest = pd.read_parquet(task_manifest_path)
                targets = pd.read_parquet(task_targets_path)
                input_read_wall = time.perf_counter() - input_read_started
            else:
                if (
                    source_manifest is None
                    or source_manifest_by_frame is None
                    or source_targets is None
                    or source_target_indices_by_frame is None
                ):
                    raise ValueError("Task has no materialized inputs and no resident source inputs are loaded")
                frame_ids = [str(value) for value in task.get("frame_group_ids") or []]
                if not frame_ids:
                    raise ValueError(f"Task {task_id} has no frame_group_ids")
                missing_manifest = [frame_id for frame_id in frame_ids if frame_id not in source_manifest_by_frame.index]
                if missing_manifest:
                    raise ValueError(
                        f"Task {task_id} references {len(missing_manifest)} frame ids absent from source manifest"
                    )
                manifest = source_manifest_by_frame.loc[frame_ids].copy()
                target_index_parts = [
                    source_target_indices_by_frame[frame_id]
                    for frame_id in frame_ids
                    if frame_id in source_target_indices_by_frame
                ]
                if target_index_parts:
                    targets = source_targets.take(np.concatenate(target_index_parts)).copy()
                else:
                    targets = source_targets.iloc[0:0].copy()
                task_manifest_path = queue_manifest.get("source_manifest_path") if queue_manifest else None
                task_targets_path = queue_manifest.get("source_projected_targets_path") if queue_manifest else None
            input_select_wall = time.perf_counter() - input_select_started
            task_summary = worker.process_frame_batch(
                manifest=manifest,
                targets=targets,
                output_dir=task_out,
                run_id=task_run_id,
                manifest_path=task_manifest_path,
                projected_targets_path=task_targets_path,
                status_path=task_status_path,
            )
            task_summary["task_input_read_wall_sec"] = input_read_wall
            task_summary["task_input_select_wall_sec"] = input_select_wall
            completed = {
                **task,
                "status": "complete",
                "completed_utc": _utc_now(),
                "worker_id": config.worker_id,
                "task_output_dir": str(task_out),
                "summary_path": str(task_out / "run_summary.json"),
                "task_input_read_wall_sec": input_read_wall,
                "task_input_select_wall_sec": input_select_wall,
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
    frame_rows: list[dict[str, Any]] = []

    for complete_path in complete_paths:
        task = json.loads(complete_path.read_text(encoding="utf-8"))
        worker_summary = task.get("worker_summary") or {}
        task_frame_rows = _frame_timing_rows(task=task, worker_summary=worker_summary)
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
                "async_shard_writes": bool(worker_summary.get("async_shard_writes", False)),
                "async_shard_write_wait_wall_sec": float(
                    worker_summary.get("async_shard_write_wait_wall_sec") or 0.0
                ),
                "resident_calibration_count": int(worker_summary.get("resident_calibration_count") or 0),
                "calibration_cache_hits": int(worker_summary.get("calibration_cache_hits") or 0),
                "calibration_cache_misses": int(worker_summary.get("calibration_cache_misses") or 0),
                "calibration_load_wall_sec": float(worker_summary.get("calibration_load_wall_sec") or 0.0),
                "batch_calibration_cache_hits": int(worker_summary.get("batch_calibration_cache_hits") or 0),
                "batch_calibration_cache_misses": int(worker_summary.get("batch_calibration_cache_misses") or 0),
                "batch_calibration_load_wall_sec": float(worker_summary.get("batch_calibration_load_wall_sec") or 0.0),
                "task_input_read_wall_sec": float(task.get("task_input_read_wall_sec") or 0.0),
                "task_input_select_wall_sec": float(task.get("task_input_select_wall_sec") or 0.0),
                "task_wall_sec": float(task.get("task_wall_sec") or 0.0),
                "payload_wait_wall_sec": sum(float(row["payload_wait_wall_sec"]) for row in task_frame_rows),
                "staging_wall_sec": sum(float(row["staging_wall_sec"]) for row in task_frame_rows),
                "staged_bytes": sum(int(row["staged_bytes"]) for row in task_frame_rows),
                "fits_read_wall_sec": sum(float(row["fits_read_wall_sec"]) for row in task_frame_rows),
                "kernel_wall_sec": sum(float(row["kernel_wall_sec"]) for row in task_frame_rows),
                "frame_compute_wall_sec": sum(float(row["frame_compute_wall_sec"]) for row in task_frame_rows),
                "summary_path": task.get("summary_path"),
            }
        )
        frame_rows.extend(task_frame_rows)
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
                    "bytes": int(shard.get("bytes") or 0),
                    "column_profile": shard.get("column_profile"),
                    "column_count": int(shard.get("column_count") or 0),
                    "parquet_compression": shard.get("parquet_compression"),
                    "async_shard_writes": bool(worker_summary.get("async_shard_writes", False)),
                    "frame_group_ids": ",".join(str(v) for v in shard.get("frame_group_ids") or []),
                    "image_ids": ",".join(str(v) for v in shard.get("image_ids") or []),
                    "write_wall_sec": float(shard.get("write_wall_sec") or 0.0),
                    "shard_table_assembly_wall_sec": float(shard.get("shard_table_assembly_wall_sec") or 0.0),
                    "metadata_concat_wall_sec": float(shard.get("metadata_concat_wall_sec") or 0.0),
                    "device_column_concat_wall_sec": float(shard.get("device_column_concat_wall_sec") or 0.0),
                    "status_concat_wall_sec": float(shard.get("status_concat_wall_sec") or 0.0),
                    "metadata_to_cudf_wall_sec": float(shard.get("metadata_to_cudf_wall_sec") or 0.0),
                    "column_attach_wall_sec": float(shard.get("column_attach_wall_sec") or 0.0),
                    "status_attach_wall_sec": float(shard.get("status_attach_wall_sec") or 0.0),
                    "shard_column_profile_wall_sec": float(shard.get("shard_column_profile_wall_sec") or 0.0),
                    "parquet_write_wall_sec": float(shard.get("parquet_write_wall_sec") or 0.0),
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
                "resident_calibration_count": 0,
                "calibration_cache_hits": 0,
                "calibration_cache_misses": 0,
                "calibration_load_wall_sec": 0.0,
                "batch_calibration_cache_hits": 0,
                "batch_calibration_cache_misses": 0,
                "batch_calibration_load_wall_sec": 0.0,
                "task_input_read_wall_sec": 0.0,
                "task_input_select_wall_sec": 0.0,
                "task_wall_sec": float(task.get("task_wall_sec") or 0.0),
                "payload_wait_wall_sec": 0.0,
                "staging_wall_sec": 0.0,
                "staged_bytes": 0,
                "fits_read_wall_sec": 0.0,
                "kernel_wall_sec": 0.0,
                "frame_compute_wall_sec": 0.0,
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
    frame_table_path = config.output_path.with_name("task_queue_frames.parquet")
    pd.DataFrame(frame_rows).to_parquet(frame_table_path, index=False)

    complete_tasks = sum(1 for row in task_rows if row["status"] == "complete")
    failed_tasks = sum(1 for row in task_rows if row["status"] == "failed")
    payload_wait_total = sum(float(row["payload_wait_wall_sec"]) for row in frame_rows)
    staging_wall_total = sum(float(row["staging_wall_sec"]) for row in frame_rows)
    staged_bytes_total = sum(int(row["staged_bytes"]) for row in frame_rows)
    fits_read_wall_total = sum(float(row["fits_read_wall_sec"]) for row in frame_rows)
    kernel_wall_total = sum(float(row["kernel_wall_sec"]) for row in frame_rows)
    frame_compute_wall_total = sum(float(row["frame_compute_wall_sec"]) for row in frame_rows)
    completed_frame_count = sum(int(row["completed_frames"]) for row in task_rows)
    complete_worker_rows = [row for row in task_rows if row["status"] == "complete"]
    worker_payload_rows = _worker_payload_rows(complete_worker_rows)
    worker_payload_wall_values = [float(row["task_wall_sec"]) for row in worker_payload_rows]
    worker_payload_sum_wall = sum(worker_payload_wall_values)
    worker_payload_max_wall = max(worker_payload_wall_values, default=0.0)
    measurement_rows = sum(int(row["measurement_rows"]) for row in task_rows)
    summary = {
        "created_utc": _utc_now(),
        "queue_dir": str(config.queue_dir),
        "complete": pending_count == 0 and leased_count == 0 and failed_tasks == 0 and missing_shards == 0,
        "pending_tasks": pending_count,
        "leased_tasks": leased_count,
        "complete_tasks": complete_tasks,
        "failed_tasks": failed_tasks,
        "completed_frames": completed_frame_count,
        "measurement_rows": measurement_rows,
        "ok_measurement_rows": sum(int(row["ok_measurement_rows"]) for row in task_rows),
        "failed_frames": sum(int(row["failed_frames"]) for row in task_rows),
        "calibration_cache_hits": sum(int(row.get("batch_calibration_cache_hits") or 0) for row in task_rows),
        "calibration_cache_misses": sum(int(row.get("batch_calibration_cache_misses") or 0) for row in task_rows),
        "calibration_load_wall_sec": sum(float(row.get("batch_calibration_load_wall_sec") or 0.0) for row in task_rows),
        "max_worker_resident_calibration_count": max(
            (int(row.get("resident_calibration_count") or 0) for row in task_rows),
            default=0,
        ),
        "frame_timing_rows": len(frame_rows),
        "payload_wait_wall_sec": payload_wait_total,
        "payload_wait_mean_wall_sec": payload_wait_total / completed_frame_count if completed_frame_count else 0.0,
        "staging_wall_sec": staging_wall_total,
        "staged_bytes": staged_bytes_total,
        "fits_read_wall_sec": fits_read_wall_total,
        "kernel_wall_sec": kernel_wall_total,
        "frame_compute_wall_sec": frame_compute_wall_total,
        "shard_count": len(shard_rows),
        "shard_total_bytes": sum(int(row.get("bytes") or 0) for row in shard_rows),
        "shard_bytes_per_measurement": (
            sum(int(row.get("bytes") or 0) for row in shard_rows) / measurement_rows if measurement_rows else 0.0
        ),
        "missing_shards": missing_shards,
        "worker_sum_wall_sec": sum(float(row["task_wall_sec"]) for row in task_rows),
        "worker_payload_rows": worker_payload_rows,
        "worker_payload_sum_wall_sec": worker_payload_sum_wall,
        "worker_payload_max_wall_sec": worker_payload_max_wall,
        "worker_parallel_efficiency": (
            worker_payload_sum_wall / (len(worker_payload_rows) * worker_payload_max_wall)
            if worker_payload_rows and worker_payload_max_wall > 0
            else 0.0
        ),
        "measurements_per_sec_worker_payload": (
            measurement_rows / worker_payload_max_wall if worker_payload_max_wall > 0 else 0.0
        ),
        "frames_per_sec_worker_payload": (
            completed_frame_count / worker_payload_max_wall if worker_payload_max_wall > 0 else 0.0
        ),
        "collect_wall_sec": time.perf_counter() - started,
        "shard_manifest_path": str(shard_manifest_path),
        "task_table_path": str(task_table_path),
        "frame_table_path": str(frame_table_path),
        "tasks": task_rows,
        "shards": shard_rows,
    }
    _write_json_atomic(config.output_path, summary)
    return summary


def _worker_payload_rows(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in task_rows:
        worker_id = str(row.get("worker_id") or "unknown")
        item = grouped.setdefault(
            worker_id,
            {
                "worker_id": worker_id,
                "task_count": 0,
                "completed_frames": 0,
                "measurement_rows": 0,
                "ok_measurement_rows": 0,
                "task_wall_sec": 0.0,
                "fits_read_wall_sec": 0.0,
                "frame_compute_wall_sec": 0.0,
                "calibration_cache_hits": 0,
                "calibration_cache_misses": 0,
                "calibration_load_wall_sec": 0.0,
                "resident_calibration_count": 0,
            },
        )
        item["task_count"] += 1
        item["completed_frames"] += int(row.get("completed_frames") or 0)
        item["measurement_rows"] += int(row.get("measurement_rows") or 0)
        item["ok_measurement_rows"] += int(row.get("ok_measurement_rows") or 0)
        item["task_wall_sec"] += float(row.get("task_wall_sec") or 0.0)
        item["fits_read_wall_sec"] += float(row.get("fits_read_wall_sec") or 0.0)
        item["frame_compute_wall_sec"] += float(row.get("frame_compute_wall_sec") or 0.0)
        item["calibration_cache_hits"] += int(row.get("batch_calibration_cache_hits") or 0)
        item["calibration_cache_misses"] += int(row.get("batch_calibration_cache_misses") or 0)
        item["calibration_load_wall_sec"] += float(row.get("batch_calibration_load_wall_sec") or 0.0)
        item["resident_calibration_count"] = max(
            int(item["resident_calibration_count"]),
            int(row.get("resident_calibration_count") or 0),
        )
    return sorted(grouped.values(), key=lambda item: item["worker_id"])


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


def _frame_timing_rows(*, task: dict[str, Any], worker_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ordinal, timing in enumerate(worker_summary.get("frame_timings") or []):
        rows.append(
            {
                "task_id": task.get("task_id"),
                "worker_id": task.get("worker_id"),
                "device": worker_summary.get("device"),
                "frame_ordinal": ordinal,
                "frame_group_id": timing.get("frame_group_id"),
                "image_id": timing.get("image_id"),
                "input_target_count": int(timing.get("input_target_count") or 0),
                "measurement_count": int(timing.get("measurement_count") or 0),
                "ok_count": int(timing.get("ok_count") or 0),
                "selected_target_count": int(timing.get("selected_target_count") or 0),
                "payload_prefetched": bool(timing.get("payload_prefetched", False)),
                "payload_wait_wall_sec": float(timing.get("payload_wait_wall_sec") or 0.0),
                "staging_wall_sec": float(timing.get("staging_wall_sec") or 0.0),
                "staged_bytes": int(timing.get("staged_bytes") or 0),
                "fits_read_wall_sec": float(timing.get("fits_read_wall_sec") or 0.0),
                "selection_wall_sec": float(timing.get("selection_wall_sec") or 0.0),
                "coordinate_extract_wall_sec": float(timing.get("coordinate_extract_wall_sec") or 0.0),
                "edge_filter_wall_sec": float(timing.get("edge_filter_wall_sec") or 0.0),
                "target_row_select_wall_sec": float(timing.get("target_row_select_wall_sec") or 0.0),
                "frame_upload_wall_sec": float(timing.get("frame_upload_wall_sec") or 0.0),
                "kernel_wall_sec": float(timing.get("kernel_wall_sec") or 0.0),
                "aperture_kernel_wall_sec": float(timing.get("aperture_kernel_wall_sec") or 0.0),
                "psf_measurement_count": int(timing.get("psf_measurement_count") or 0),
                "ok_psf_count": int(timing.get("ok_psf_count") or 0),
                "psf_target_count": int(timing.get("psf_target_count") or 0),
                "psf_grid_offsets": int(timing.get("psf_grid_offsets") or 0),
                "psf_candidate_count": int(timing.get("psf_candidate_count") or 0),
                "psf_candidate_grid_wall_sec": float(timing.get("psf_candidate_grid_wall_sec") or 0.0),
                "psf_kernel_wall_sec": float(timing.get("psf_kernel_wall_sec") or 0.0),
                "psf_total_wall_sec": float(timing.get("psf_total_wall_sec") or 0.0),
                "psf_device_submit_sync_wall_sec": float(timing.get("psf_device_submit_sync_wall_sec") or 0.0),
                "psf_spline_coeff_wall_sec": float(timing.get("psf_spline_coeff_wall_sec") or 0.0),
                "psf_upload_wall_sec": float(timing.get("psf_upload_wall_sec") or 0.0),
                "psf_gather_wall_sec": float(timing.get("psf_gather_wall_sec") or 0.0),
                "metadata_build_wall_sec": float(timing.get("metadata_build_wall_sec") or 0.0),
                "table_wall_sec": float(timing.get("table_wall_sec") or 0.0),
                "frame_compute_wall_sec": float(timing.get("frame_compute_wall_sec") or 0.0),
                "wall_time_sec": float(timing.get("wall_time_sec") or 0.0),
                "write_wall_sec": float(timing.get("write_wall_sec") or 0.0),
                "shard_submit_wall_sec": float(timing.get("shard_submit_wall_sec") or 0.0),
                "async_write_queued": bool(timing.get("async_write_queued", False)),
                "deferred_write": bool(timing.get("deferred_write", False)),
                "error": timing.get("error"),
            }
        )
    return rows


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
        "in_frame_projected_rows",
        "target_setup_wall_sec",
        "measurement_rows",
        "ok_measurement_rows",
        "failed_frames",
        "calibration_upload_count",
        "backend",
        "local_cache_dir",
        "async_shard_writes",
        "batch_table_assembly",
        "total_wall_sec",
        "task_input_read_wall_sec",
        "task_input_select_wall_sec",
        "resident_calibration_count",
        "calibration_cache_hits",
        "calibration_cache_misses",
        "calibration_load_wall_sec",
        "batch_calibration_cache_hits",
        "batch_calibration_cache_misses",
        "batch_calibration_load_wall_sec",
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


def _read_queue_manifest(queue_dir: Path) -> dict[str, Any]:
    path = queue_dir / "queue_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_queue_path(queue_dir: Path, value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute() or path.exists():
        return path
    candidate = queue_dir / path
    if candidate.exists():
        return candidate
    return path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
