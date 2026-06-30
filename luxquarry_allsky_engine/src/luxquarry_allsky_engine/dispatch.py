from __future__ import annotations

import json
import shlex
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DispatchPlanConfig:
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
    materialize_worker_inputs: bool = False
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


def build_dispatch_plan(config: DispatchPlanConfig) -> dict[str, Any]:
    if not config.devices:
        raise ValueError("At least one GPU device is required")
    if config.workers_per_device <= 0:
        raise ValueError("workers_per_device must be positive")
    total_workers = len(config.devices) * config.workers_per_device
    materialized_inputs, materialize_summary = _materialize_worker_inputs(config, total_workers)
    workers = []
    worker_index = 0
    for device in config.devices:
        for local_slot in range(config.workers_per_device):
            worker_id = f"{config.run_id}.w{worker_index:04d}.{device.replace(':', '')}.s{local_slot}"
            worker_out = config.output_dir / "workers" / worker_id
            worker_manifest = config.manifest_path
            worker_projected_targets = config.projected_targets_path
            runtime_worker_index = worker_index
            runtime_worker_count = total_workers
            if config.materialize_worker_inputs:
                worker_input = materialized_inputs[worker_index]
                worker_manifest = Path(worker_input["manifest_path"])
                worker_projected_targets = Path(worker_input["projected_targets_path"])
                runtime_worker_index = 0
                runtime_worker_count = 1
            status_path = worker_out / "run_status.json"
            argv = [
                config.executable,
                "run-persistent-gpu-worker",
                "--manifest",
                str(worker_manifest),
                "--projected-targets",
                str(worker_projected_targets),
                "--out-dir",
                str(worker_out),
                "--run-id",
                worker_id,
                "--cache-root",
                str(config.cache_root),
                "--device",
                device,
                "--worker-index",
                str(runtime_worker_index),
                "--worker-count",
                str(runtime_worker_count),
                "--status-path",
                str(status_path),
                "--shard-batch-frames",
                str(config.shard_batch_frames),
                "--prefetch-frames",
                str(config.prefetch_frames),
                "--status-interval-frames",
                str(config.status_interval_frames),
                "--aperture-radius-pix",
                str(config.aperture_radius_pix),
                "--annulus-inner-pix",
                str(config.annulus_inner_pix),
                "--annulus-outer-pix",
                str(config.annulus_outer_pix),
                "--edge-margin-pix",
                str(config.edge_margin_pix),
            ]
            if config.limit_frames is not None and not config.materialize_worker_inputs:
                argv.extend(["--limit-frames", str(config.limit_frames)])
            if config.local_cache_dir is not None:
                argv.extend(["--local-cache-dir", str(config.local_cache_dir)])
            if config.async_shard_writes:
                argv.append("--async-shard-writes")
            if config.batch_table_assembly:
                argv.append("--batch-table-assembly")
            if config.discard_measurement_shards:
                argv.append("--discard-measurement-shards")
            if config.measurement_column_profile != "full":
                argv.extend(["--measurement-column-profile", config.measurement_column_profile])
            if config.measurement_parquet_compression != "snappy":
                argv.extend(["--measurement-parquet-compression", config.measurement_parquet_compression])
            if config.enable_psf:
                argv.append("--enable-psf")
                argv.extend(["--psf-kernel-build-mode", config.psf_kernel_build_mode])
                argv.extend(["--psf-kernel-radius-native", str(config.psf_kernel_radius_native)])
                argv.extend(["--psf-grid-half-range-pix", str(config.psf_grid_half_range_pix)])
                argv.extend(["--psf-grid-step-pix", str(config.psf_grid_step_pix)])
                argv.extend(["--psf-grid-metric", config.psf_grid_metric])
            workers.append(
                {
                    "worker_id": worker_id,
                    "worker_index": worker_index,
                    "worker_count": total_workers,
                    "runtime_worker_index": runtime_worker_index,
                    "runtime_worker_count": runtime_worker_count,
                    "device": device,
                    "local_slot": local_slot,
                    "manifest_path": str(worker_manifest),
                    "projected_targets_path": str(worker_projected_targets),
                    "output_dir": str(worker_out),
                    "status_path": str(status_path),
                    "argv": argv,
                    "shell": " ".join(shlex.quote(part) for part in argv),
                }
            )
            worker_index += 1
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config.run_id,
        "manifest_path": str(config.manifest_path),
        "projected_targets_path": str(config.projected_targets_path),
        "output_dir": str(config.output_dir),
        "cache_root": str(config.cache_root),
        "devices": list(config.devices),
        "workers_per_device": config.workers_per_device,
        "worker_count": total_workers,
        "limit_frames": config.limit_frames,
        "local_cache_dir": str(config.local_cache_dir) if config.local_cache_dir else None,
        "async_shard_writes": config.async_shard_writes,
        "batch_table_assembly": config.batch_table_assembly,
        "discard_measurement_shards": config.discard_measurement_shards,
        "measurement_column_profile": config.measurement_column_profile,
        "measurement_parquet_compression": config.measurement_parquet_compression,
        "materialize_worker_inputs": config.materialize_worker_inputs,
        "aperture_radius_pix": config.aperture_radius_pix,
        "annulus_inner_pix": config.annulus_inner_pix,
        "annulus_outer_pix": config.annulus_outer_pix,
        "edge_margin_pix": config.edge_margin_pix,
        "enable_psf": config.enable_psf,
        "psf_kernel_build_mode": config.psf_kernel_build_mode if config.enable_psf else None,
        "psf_grid_half_range_pix": config.psf_grid_half_range_pix if config.enable_psf else None,
        "psf_grid_step_pix": config.psf_grid_step_pix if config.enable_psf else None,
        "psf_grid_metric": config.psf_grid_metric if config.enable_psf else None,
        "materialized_inputs": materialize_summary,
        "contract": {
            "partitioning": (
                "pre-materialized per-worker frame and target parquets"
                if config.materialize_worker_inputs
                else "frame ordinal modulo worker_count equals worker_index"
            ),
            "output": "each worker writes independent measurement_shards and run_summary.json",
            "status": "each worker atomically rewrites run_status.json",
            "coordination": "no live database or shared lock required in the hot path",
        },
        "workers": workers,
    }


def _materialize_worker_inputs(
    config: DispatchPlanConfig,
    total_workers: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any] | None]:
    if not config.materialize_worker_inputs:
        return {}, None
    started = time.perf_counter()
    import pandas as pd

    manifest = pd.read_parquet(config.manifest_path)
    if config.limit_frames is not None:
        manifest = manifest.head(config.limit_frames).copy()
    manifest = manifest.reset_index(drop=True)
    projected_targets = pd.read_parquet(config.projected_targets_path)
    worker_inputs_dir = config.output_dir / "worker_inputs"
    worker_inputs_dir.mkdir(parents=True, exist_ok=True)

    materialized: dict[int, dict[str, Any]] = {}
    total_manifest_rows = 0
    total_target_rows = 0
    for worker_index in range(total_workers):
        worker_dir = worker_inputs_dir / f"w{worker_index:04d}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        worker_manifest = manifest.iloc[
            [i for i in range(len(manifest)) if i % total_workers == worker_index]
        ].copy()
        frame_ids = set(worker_manifest["frame_group_id"].astype(str))
        worker_targets = projected_targets[
            projected_targets["frame_group_id"].astype(str).isin(frame_ids)
        ].copy()
        manifest_path = worker_dir / "frame_manifest.parquet"
        projected_targets_path = worker_dir / "projected_targets.parquet"
        worker_manifest.to_parquet(manifest_path, index=False)
        worker_targets.to_parquet(projected_targets_path, index=False)
        materialized[worker_index] = {
            "worker_index": worker_index,
            "manifest_path": str(manifest_path),
            "projected_targets_path": str(projected_targets_path),
            "frame_count": int(len(worker_manifest)),
            "projected_target_rows": int(len(worker_targets)),
        }
        total_manifest_rows += int(len(worker_manifest))
        total_target_rows += int(len(worker_targets))

    summary = {
        "worker_inputs_dir": str(worker_inputs_dir),
        "worker_count": total_workers,
        "source_manifest_path": str(config.manifest_path),
        "source_projected_targets_path": str(config.projected_targets_path),
        "source_manifest_rows": int(len(manifest)),
        "source_projected_target_rows": int(len(projected_targets)),
        "materialized_manifest_rows": total_manifest_rows,
        "materialized_projected_target_rows": total_target_rows,
        "wall_sec": time.perf_counter() - started,
        "workers": list(materialized.values()),
    }
    return materialized, summary


def write_dispatch_plan(plan: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shell_path = output_path.with_suffix(".sh")
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for worker in plan["workers"]:
        lines.append(worker["shell"] + " &")
    lines.append("wait")
    shell_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shell_path.chmod(0o755)


def collect_dispatch_run(plan_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    run_output_dir = _run_output_dir(plan_path, plan)
    output_path = output_path or run_output_dir / "aggregate_summary.json"
    workers = list(plan.get("workers") or [])
    worker_rows: list[dict[str, Any]] = []
    shard_rows: list[dict[str, Any]] = []

    for worker in workers:
        worker_id = str(worker.get("worker_id"))
        worker_out = _worker_output_dir(plan_path, worker)
        summary_path = worker_out / "run_summary.json"
        status = "missing"
        summary: dict[str, Any] = {}
        error = None
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                status = "complete" if summary.get("completed_utc") else "incomplete"
            except Exception as exc:
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
        frame_timings = list(summary.get("frame_timings") or [])
        worker_row = {
            "worker_id": worker_id,
            "worker_index": worker.get("worker_index"),
            "worker_count": worker.get("worker_count"),
            "device": worker.get("device"),
            "output_dir": str(worker_out),
            "summary_path": str(summary_path),
            "status": status,
            "error": error,
            "frame_count": int(summary.get("frame_count") or 0),
            "completed_frames": int(summary.get("completed_frames") or 0),
            "measurement_rows": int(summary.get("measurement_rows") or 0),
            "ok_measurement_rows": int(summary.get("ok_measurement_rows") or 0),
            "psf_measurement_rows": int(summary.get("psf_measurement_rows") or 0),
            "ok_psf_rows": int(summary.get("ok_psf_rows") or 0),
            "psf_candidate_count": _sum_frame_timings(frame_timings, "psf_candidate_count"),
            "psf_candidate_grid_wall_sec": _sum_frame_timings(frame_timings, "psf_candidate_grid_wall_sec"),
            "psf_kernel_wall_sec": _sum_frame_timings(frame_timings, "psf_kernel_wall_sec"),
            "psf_device_submit_sync_wall_sec": _sum_frame_timings(frame_timings, "psf_device_submit_sync_wall_sec"),
            "psf_spline_coeff_wall_sec": _sum_frame_timings(frame_timings, "psf_spline_coeff_wall_sec"),
            "psf_upload_wall_sec": _sum_frame_timings(frame_timings, "psf_upload_wall_sec"),
            "psf_gather_wall_sec": _sum_frame_timings(frame_timings, "psf_gather_wall_sec"),
            "aperture_kernel_wall_sec": _sum_frame_timings(frame_timings, "aperture_kernel_wall_sec"),
            "frame_upload_wall_sec": _sum_frame_timings(frame_timings, "frame_upload_wall_sec"),
            "staging_wall_sec": _sum_frame_timings(frame_timings, "staging_wall_sec"),
            "payload_wait_wall_sec": _sum_frame_timings(frame_timings, "payload_wait_wall_sec"),
            "fits_read_wall_sec": _sum_frame_timings(frame_timings, "fits_read_wall_sec"),
            "write_wall_sec": _sum_frame_timings(frame_timings, "write_wall_sec"),
            "shard_submit_wall_sec": _sum_frame_timings(frame_timings, "shard_submit_wall_sec"),
            "selection_wall_sec": _sum_frame_timings(frame_timings, "selection_wall_sec"),
            "table_wall_sec": _sum_frame_timings(frame_timings, "table_wall_sec"),
            "frame_compute_wall_sec": _sum_frame_timings(frame_timings, "frame_compute_wall_sec"),
            "async_shard_write_wait_wall_sec": float(summary.get("async_shard_write_wait_wall_sec") or 0.0),
            "failed_frames": int(summary.get("failed_frames") or 0),
            "total_wall_sec": float(summary.get("total_wall_sec") or 0.0),
        }
        worker_rows.append(worker_row)
        for shard in summary.get("shards") or []:
            shard_path = _resolve_shard_path(worker_out, shard)
            frame_group_ids = _shard_values(shard, plural_key="frame_group_ids", singular_key="frame_group_id")
            image_ids = _shard_values(shard, plural_key="image_ids", singular_key="image_id")
            shard_bytes = shard_path.stat().st_size if shard_path.exists() else 0
            shard_rows.append(
                {
                    "worker_id": worker_id,
                    "worker_index": worker.get("worker_index"),
                    "device": worker.get("device"),
                    "path": str(shard_path),
                    "bytes": int(shard_bytes),
                    "column_profile": shard.get("column_profile"),
                    "column_count": int(shard.get("column_count") or 0),
                    "parquet_compression": shard.get("parquet_compression"),
                    "rows": int(shard.get("rows") or 0),
                    "ok_rows": int(shard.get("ok_rows") or 0),
                    "frame_count": int(shard.get("frame_count") or len(frame_group_ids)),
                    "frame_group_ids": ",".join(str(v) for v in frame_group_ids),
                    "image_ids": ",".join(str(v) for v in image_ids),
                    "write_wall_sec": float(shard.get("write_wall_sec") or 0.0),
                    "exists": shard_path.exists(),
                }
            )

    complete_workers = sum(1 for row in worker_rows if row["status"] == "complete")
    missing_workers = sum(1 for row in worker_rows if row["status"] == "missing")
    errored_workers = sum(1 for row in worker_rows if row["status"] == "error")
    incomplete_workers = sum(1 for row in worker_rows if row["status"] == "incomplete")
    failed_frames = sum(row["failed_frames"] for row in worker_rows)
    missing_shards = sum(1 for row in shard_rows if not row["exists"])
    shard_write_by_worker: dict[str, float] = {}
    for row in shard_rows:
        worker_id = str(row.get("worker_id"))
        shard_write_by_worker[worker_id] = shard_write_by_worker.get(worker_id, 0.0) + float(
            row.get("write_wall_sec") or 0.0
        )
    for row in worker_rows:
        row["shard_write_wall_sec"] = shard_write_by_worker.get(str(row.get("worker_id")), 0.0)
        row["shard_bytes"] = sum(
            int(shard.get("bytes") or 0)
            for shard in shard_rows
            if str(shard.get("worker_id")) == str(row.get("worker_id"))
        )
    shard_write_wall_sec = sum(shard_write_by_worker.values())
    shard_total_bytes = sum(int(row.get("bytes") or 0) for row in shard_rows)
    worker_wall_times = [float(row["total_wall_sec"] or 0.0) for row in worker_rows]
    worker_max_wall_sec = max(worker_wall_times, default=0.0)
    worker_min_wall_sec = min(worker_wall_times, default=0.0)
    worker_sum_wall_sec = sum(worker_wall_times)
    worker_avg_wall_sec = worker_sum_wall_sec / len(worker_wall_times) if worker_wall_times else 0.0
    shard_manifest_path = output_path.with_name("measurement_shard_manifest.parquet")
    _write_shard_manifest(shard_manifest_path, shard_rows)
    aggregate = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(plan_path),
        "run_id": plan.get("run_id"),
        "output_dir": str(run_output_dir),
        "plan_worker_count": int(plan.get("worker_count") or len(workers)),
        "worker_count": len(workers),
        "complete_workers": complete_workers,
        "missing_workers": missing_workers,
        "incomplete_workers": incomplete_workers,
        "errored_workers": errored_workers,
        "complete": (
            complete_workers == len(workers)
            and missing_workers == 0
            and incomplete_workers == 0
            and errored_workers == 0
            and failed_frames == 0
            and missing_shards == 0
        ),
        "frame_count": sum(row["frame_count"] for row in worker_rows),
        "completed_frames": sum(row["completed_frames"] for row in worker_rows),
        "failed_frames": failed_frames,
        "measurement_rows": sum(row["measurement_rows"] for row in worker_rows),
        "ok_measurement_rows": sum(row["ok_measurement_rows"] for row in worker_rows),
        "psf_measurement_rows": sum(row["psf_measurement_rows"] for row in worker_rows),
        "ok_psf_rows": sum(row["ok_psf_rows"] for row in worker_rows),
        "psf_candidate_count": sum(row["psf_candidate_count"] for row in worker_rows),
        "psf_candidate_grid_wall_sec": sum(row["psf_candidate_grid_wall_sec"] for row in worker_rows),
        "psf_kernel_wall_sec": sum(row["psf_kernel_wall_sec"] for row in worker_rows),
        "psf_device_submit_sync_wall_sec": sum(row["psf_device_submit_sync_wall_sec"] for row in worker_rows),
        "psf_spline_coeff_wall_sec": sum(row["psf_spline_coeff_wall_sec"] for row in worker_rows),
        "psf_upload_wall_sec": sum(row["psf_upload_wall_sec"] for row in worker_rows),
        "psf_gather_wall_sec": sum(row["psf_gather_wall_sec"] for row in worker_rows),
        "aperture_kernel_wall_sec": sum(row["aperture_kernel_wall_sec"] for row in worker_rows),
        "frame_upload_wall_sec": sum(row["frame_upload_wall_sec"] for row in worker_rows),
        "staging_wall_sec": sum(row["staging_wall_sec"] for row in worker_rows),
        "payload_wait_wall_sec": sum(row["payload_wait_wall_sec"] for row in worker_rows),
        "fits_read_wall_sec": sum(row["fits_read_wall_sec"] for row in worker_rows),
        "write_wall_sec": sum(row["write_wall_sec"] for row in worker_rows),
        "shard_submit_wall_sec": sum(row["shard_submit_wall_sec"] for row in worker_rows),
        "selection_wall_sec": sum(row["selection_wall_sec"] for row in worker_rows),
        "table_wall_sec": sum(row["table_wall_sec"] for row in worker_rows),
        "frame_compute_wall_sec": sum(row["frame_compute_wall_sec"] for row in worker_rows),
        "async_shard_write_wait_wall_sec": sum(row["async_shard_write_wait_wall_sec"] for row in worker_rows),
        "shard_write_wall_sec": shard_write_wall_sec,
        "shard_total_bytes": shard_total_bytes,
        "shard_bytes_per_measurement": (
            float(shard_total_bytes / sum(row["measurement_rows"] for row in worker_rows))
            if sum(row["measurement_rows"] for row in worker_rows)
            else 0.0
        ),
        "max_worker_staging_wall_sec": _max_worker_metric(worker_rows, "staging_wall_sec"),
        "max_worker_payload_wait_wall_sec": _max_worker_metric(worker_rows, "payload_wait_wall_sec"),
        "max_worker_fits_read_wall_sec": _max_worker_metric(worker_rows, "fits_read_wall_sec"),
        "max_worker_frame_upload_wall_sec": _max_worker_metric(worker_rows, "frame_upload_wall_sec"),
        "max_worker_aperture_kernel_wall_sec": _max_worker_metric(worker_rows, "aperture_kernel_wall_sec"),
        "max_worker_psf_kernel_wall_sec": _max_worker_metric(worker_rows, "psf_kernel_wall_sec"),
        "max_worker_psf_device_submit_sync_wall_sec": _max_worker_metric(
            worker_rows, "psf_device_submit_sync_wall_sec"
        ),
        "max_worker_psf_spline_coeff_wall_sec": _max_worker_metric(worker_rows, "psf_spline_coeff_wall_sec"),
        "max_worker_psf_gather_wall_sec": _max_worker_metric(worker_rows, "psf_gather_wall_sec"),
        "max_worker_table_wall_sec": _max_worker_metric(worker_rows, "table_wall_sec"),
        "max_worker_shard_write_wall_sec": _max_worker_metric(worker_rows, "shard_write_wall_sec"),
        "max_worker_shard_bytes": _max_worker_metric(worker_rows, "shard_bytes"),
        "max_worker_async_shard_write_wait_wall_sec": _max_worker_metric(
            worker_rows, "async_shard_write_wait_wall_sec"
        ),
        "shard_count": len(shard_rows),
        "missing_shards": missing_shards,
        "worker_min_wall_sec": worker_min_wall_sec,
        "worker_avg_wall_sec": worker_avg_wall_sec,
        "worker_max_wall_sec": worker_max_wall_sec,
        "worker_sum_wall_sec": worker_sum_wall_sec,
        "worker_parallel_efficiency": (
            worker_sum_wall_sec / (worker_max_wall_sec * len(worker_wall_times))
            if worker_max_wall_sec > 0 and worker_wall_times
            else 0.0
        ),
        "worker_wall_skew_ratio": (
            worker_max_wall_sec / worker_min_wall_sec if worker_min_wall_sec > 0 else 0.0
        ),
        "collect_wall_sec": time.perf_counter() - started,
        "shard_manifest_path": str(shard_manifest_path),
        "workers": worker_rows,
        "shards": shard_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return aggregate


def _sum_frame_timings(frame_timings: list[dict[str, Any]], key: str) -> float:
    total = 0.0
    for row in frame_timings:
        try:
            total += float(row.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
    return total


def _max_worker_metric(worker_rows: list[dict[str, Any]], key: str) -> float:
    values = []
    for row in worker_rows:
        try:
            values.append(float(row.get(key) or 0.0))
        except (TypeError, ValueError):
            continue
    return max(values, default=0.0)


def _write_shard_manifest(path: Path, shard_rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(shard_rows).to_parquet(path, index=False)


def _run_output_dir(plan_path: Path, plan: dict[str, Any]) -> Path:
    configured = Path(str(plan.get("output_dir") or ""))
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


def _resolve_shard_path(worker_output_dir: Path, shard: dict[str, Any]) -> Path:
    configured = Path(str(shard.get("path") or ""))
    if configured.is_absolute() or configured.exists():
        return configured
    fallback = worker_output_dir / "measurement_shards" / configured.name
    return fallback if fallback.exists() else configured


def _shard_values(shard: dict[str, Any], *, plural_key: str, singular_key: str) -> list[Any]:
    values = shard.get(plural_key)
    if values:
        return list(values)
    value = shard.get(singular_key)
    return [value] if value is not None else []
