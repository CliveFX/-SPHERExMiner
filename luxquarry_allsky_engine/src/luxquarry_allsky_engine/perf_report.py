from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TaskQueuePerfReportConfig:
    run_dir: Path
    output_dir: Path | None = None


def summarize_task_queue_performance(config: TaskQueuePerfReportConfig) -> dict[str, Any]:
    run_dir = config.run_dir
    output_dir = config.output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    local_summary = _read_json(run_dir / "local_task_queue_summary.json")
    collect_summary = _read_json(run_dir / "task_queue_collect_summary.json")
    frame_rows = _read_parquet(run_dir / "task_queue_frames.parquet")
    shard_rows = _read_parquet(run_dir / "measurement_shard_manifest.parquet")

    worker_payload_wall = _num(collect_summary.get("worker_payload_max_wall_sec"))
    total_wall = _num(local_summary.get("total_wall_sec"))
    measurement_rows = int(collect_summary.get("measurement_rows") or 0)
    completed_frames = int(collect_summary.get("completed_frames") or 0)

    profile_rows: list[dict[str, Any]] = []
    for stage in _frame_stage_defs():
        profile_rows.append(
            _frame_stage_row(
                run_id=str(local_summary.get("run_id") or run_dir.name),
                stage=stage,
                frames=frame_rows,
                denominator_wall=worker_payload_wall,
                measurement_rows=measurement_rows,
                completed_frames=completed_frames,
            )
        )

    profile_rows.append(
        _shard_write_row(
            run_id=str(local_summary.get("run_id") or run_dir.name),
            shards=shard_rows,
            denominator_wall=worker_payload_wall,
            measurement_rows=measurement_rows,
        )
    )

    spectra = local_summary.get("spectra_summary") or {}
    if spectra:
        profile_rows.append(
            _simple_stage_row(
                run_id=str(local_summary.get("run_id") or run_dir.name),
                stage="assemble_spectra",
                backend=str(spectra.get("backend") or "cudf_spectra_assembly"),
                wall=_num(spectra.get("total_wall_sec")),
                denominator_wall=total_wall,
                rows_out=int(spectra.get("spectra_measurement_rows") or measurement_rows),
                inclusive=False,
                phase="postprocess",
            )
        )

    candidates = local_summary.get("candidate_summary") or {}
    if candidates:
        profile_rows.append(
            _simple_stage_row(
                run_id=str(local_summary.get("run_id") or run_dir.name),
                stage="score_baseline_candidates",
                backend=str(candidates.get("backend") or "cudf_simple_target_zscore_scorer"),
                wall=_num(candidates.get("total_wall_sec")),
                denominator_wall=total_wall,
                rows_out=int(candidates.get("candidate_count") or 0),
                inclusive=False,
                phase="postprocess",
            )
        )

    profile_rows = [row for row in profile_rows if row["summed_wall_sec"] > 0 or row["critical_path_wall_sec"] > 0]
    profile_rows.sort(key=lambda row: (row["phase"], -float(row["critical_path_wall_sec"]), row["stage"]))

    profile_df = pd.DataFrame(profile_rows, columns=_profile_columns())
    profile_path = output_dir / "task_queue_perf_profile.parquet"
    profile_json_path = output_dir / "task_queue_perf_profile.json"
    report_path = output_dir / "task_queue_perf_report.json"
    profile_df.to_parquet(profile_path, index=False)

    bottlenecks = [
        row
        for row in profile_rows
        if not row["inclusive"] and row["phase"] == "worker_payload" and row["critical_path_wall_pct"] >= 5.0
    ]
    bottlenecks.sort(key=lambda row: float(row["critical_path_wall_pct"]), reverse=True)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "luxquarry_task_queue_perf_report",
        "run_id": local_summary.get("run_id") or run_dir.name,
        "run_dir": str(run_dir),
        "status": local_summary.get("status"),
        "worker_count": int(local_summary.get("worker_count") or 0),
        "failed_workers": int(local_summary.get("failed_workers") or 0),
        "completed_frames": completed_frames,
        "measurement_rows": measurement_rows,
        "ok_measurement_rows": int(collect_summary.get("ok_measurement_rows") or 0),
        "shard_count": int(collect_summary.get("shard_count") or 0),
        "shard_total_bytes": int(collect_summary.get("shard_total_bytes") or 0),
        "shard_bytes_per_measurement": _num(collect_summary.get("shard_bytes_per_measurement")),
        "total_wall_sec": total_wall,
        "worker_payload_max_wall_sec": worker_payload_wall,
        "measurements_per_sec_worker_payload": _rate(measurement_rows, worker_payload_wall),
        "frames_per_sec_worker_payload": _rate(completed_frames, worker_payload_wall),
        "worker_parallel_efficiency": _num(collect_summary.get("worker_parallel_efficiency")),
        "top_worker_payload_bottlenecks": bottlenecks[:10],
        "optimization_rule": "Worker-payload stages above 5% critical-path wall time need an explicit keep/accelerate/rewrite decision.",
        "outputs": {
            "profile_parquet": str(profile_path),
            "profile_json": str(profile_json_path),
            "report_json": str(report_path),
        },
    }

    profile_json_path.write_text(json.dumps({"rows": profile_rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _frame_stage_defs() -> list[dict[str, Any]]:
    return [
        _stage("payload_wait", "payload_wait_wall_sec", "worker_prefetch_queue", "worker_payload"),
        _stage("stage_fits_object", "staging_wall_sec", "local_or_object_staging", "worker_payload"),
        _stage("read_fits", "fits_read_wall_sec", "astropy_fits_read", "worker_payload"),
        _stage("select_targets", "selection_wall_sec", "pandas_frame_target_selection", "worker_payload"),
        _stage("upload_frame_to_gpu", "frame_upload_wall_sec", "warp_device_upload", "worker_payload"),
        _stage("aperture_kernel", "aperture_kernel_wall_sec", "warp_gpu_aperture", "worker_payload"),
        _stage("psf_candidate_grid", "psf_candidate_grid_wall_sec", "warp_gpu_psf_grid", "worker_payload"),
        _stage("psf_spline_coeff", "psf_spline_coeff_wall_sec", "scipy_cpu_spline_coefficients", "worker_payload"),
        _stage("psf_upload", "psf_upload_wall_sec", "warp_gpu_psf_grid", "worker_payload"),
        _stage("psf_device_submit_sync", "psf_device_submit_sync_wall_sec", "warp_gpu_psf_grid", "worker_payload"),
        _stage("psf_gather", "psf_gather_wall_sec", "gpu_to_cpu_psf_gather", "worker_payload"),
        _stage("assemble_measurement_table", "table_wall_sec", "cudf_measurement_table", "worker_payload"),
        _stage("submit_measurement_shard", "shard_submit_wall_sec", "async_shard_writer", "worker_payload"),
        _stage("frame_compute_inclusive", "frame_compute_wall_sec", "persistent_gpu_frame_worker", "worker_payload", True),
    ]


def _stage(stage: str, field: str, backend: str, phase: str, inclusive: bool = False) -> dict[str, Any]:
    return {"stage": stage, "field": field, "backend": backend, "phase": phase, "inclusive": inclusive}


def _frame_stage_row(
    *,
    run_id: str,
    stage: dict[str, Any],
    frames: pd.DataFrame,
    denominator_wall: float,
    measurement_rows: int,
    completed_frames: int,
) -> dict[str, Any]:
    field = str(stage["field"])
    summed = _sum_column(frames, field)
    critical = _max_group_sum(frames, "worker_id", field)
    rows_out = measurement_rows
    if field == "psf_candidate_grid_wall_sec":
        rows_out = int(_sum_column(frames, "psf_candidate_count"))
    return {
        "run_id": run_id,
        "stage": stage["stage"],
        "phase": stage["phase"],
        "backend": stage["backend"],
        "inclusive": bool(stage["inclusive"]),
        "summed_wall_sec": summed,
        "critical_path_wall_sec": critical,
        "critical_path_wall_pct": _pct(critical, denominator_wall),
        "rows_in": measurement_rows,
        "rows_out": rows_out,
        "completed_frames": completed_frames,
        "throughput_rows_per_sec_critical": _rate(rows_out, critical),
        "decision_hint": _decision_hint(stage["stage"], critical, denominator_wall),
    }


def _shard_write_row(*, run_id: str, shards: pd.DataFrame, denominator_wall: float, measurement_rows: int) -> dict[str, Any]:
    summed = _sum_column(shards, "write_wall_sec")
    critical = _max_group_sum(shards, "worker_id", "write_wall_sec")
    return {
        "run_id": run_id,
        "stage": "write_measurement_shards",
        "phase": "worker_payload",
        "backend": "parquet_measurement_writer",
        "inclusive": False,
        "summed_wall_sec": summed,
        "critical_path_wall_sec": critical,
        "critical_path_wall_pct": _pct(critical, denominator_wall),
        "rows_in": measurement_rows,
        "rows_out": measurement_rows,
        "completed_frames": 0,
        "throughput_rows_per_sec_critical": _rate(measurement_rows, critical),
        "decision_hint": _decision_hint("write_measurement_shards", critical, denominator_wall),
    }


def _simple_stage_row(
    *,
    run_id: str,
    stage: str,
    backend: str,
    wall: float,
    denominator_wall: float,
    rows_out: int,
    inclusive: bool,
    phase: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "stage": stage,
        "phase": phase,
        "backend": backend,
        "inclusive": inclusive,
        "summed_wall_sec": wall,
        "critical_path_wall_sec": wall,
        "critical_path_wall_pct": _pct(wall, denominator_wall),
        "rows_in": rows_out,
        "rows_out": rows_out,
        "completed_frames": 0,
        "throughput_rows_per_sec_critical": _rate(rows_out, wall),
        "decision_hint": _decision_hint(stage, wall, denominator_wall),
    }


def _decision_hint(stage: str, critical: float, denominator: float) -> str:
    pct = _pct(critical, denominator)
    if pct < 5.0:
        return "Below 5% critical-path threshold; keep unless it scales poorly."
    if stage in {"read_fits", "stage_fits_object"}:
        return "Hot I/O path; test KvikIO/fsspec caching and overlap before rewriting science code."
    if stage in {"write_measurement_shards", "assemble_measurement_table"}:
        return "Hot durable-output path; consider larger shard batches, cuDF parquet tuning, or async writer isolation."
    if stage.startswith("psf"):
        return "Hot PSF path; keep kernels resident and remove CPU/device round trips before changing math."
    if stage == "select_targets":
        return "Hot selection path; replace per-frame pandas filtering with prepartitioned frame target batches or cuDF."
    return "Above 5% critical-path threshold; profile and decide keep/accelerate/rewrite."


def _profile_columns() -> list[str]:
    return [
        "run_id",
        "stage",
        "phase",
        "backend",
        "inclusive",
        "summed_wall_sec",
        "critical_path_wall_sec",
        "critical_path_wall_pct",
        "rows_in",
        "rows_out",
        "completed_frames",
        "throughput_rows_per_sec_critical",
        "decision_hint",
    ]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _sum_column(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[column], errors="coerce").fillna(0.0).sum())


def _max_group_sum(df: pd.DataFrame, group_column: str, value_column: str) -> float:
    if df.empty or group_column not in df.columns or value_column not in df.columns:
        return 0.0
    grouped = pd.to_numeric(df[value_column], errors="coerce").fillna(0.0).groupby(df[group_column]).sum()
    if grouped.empty:
        return 0.0
    return float(grouped.max())


def _num(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _rate(rows: int | float, wall: float) -> float:
    return float(rows) / wall if wall > 0 else 0.0


def _pct(part: float, total: float) -> float:
    return 100.0 * part / total if total > 0 else 0.0
