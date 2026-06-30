from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ReducerPlanConfig:
    partition_manifest_path: Path
    output_dir: Path
    run_id: str
    executable: str = ".venv/bin/luxquarry-allsky"
    devices: tuple[str, ...] = ("cuda:0",)
    spectra_out_dir: Path | None = None
    only_ok: bool = False
    drop_duplicate_measurements: bool = True
    max_partitions: int | None = None


@dataclass(frozen=True)
class LocalReducerRunConfig:
    plan_path: Path
    logs_dir: Path | None = None
    resume: bool = False
    max_parallel: int | None = None
    allow_failed_reducers: bool = False


@dataclass(frozen=True)
class ReducerCollectConfig:
    plan_path: Path
    output_path: Path | None = None
    allow_incomplete: bool = False


def build_reducer_plan(config: ReducerPlanConfig) -> dict[str, Any]:
    if not config.devices:
        raise ValueError("at least one reducer device is required")
    if config.max_partitions is not None and config.max_partitions <= 0:
        raise ValueError("max_partitions must be positive")

    manifest = pd.read_parquet(config.partition_manifest_path)
    if config.max_partitions is not None:
        manifest = manifest.head(config.max_partitions)

    required = {"partition_index", "measurement_shard_manifest_path", "rows", "target_count"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Partition manifest missing columns: {missing}")

    spectra_root = config.spectra_out_dir or config.output_dir / "spectra"
    reducers: list[dict[str, Any]] = []
    for ordinal, row in enumerate(manifest.sort_values("partition_index").itertuples(index=False)):
        row_dict = row._asdict()
        partition_index = int(row_dict["partition_index"])
        partition_label = f"part{partition_index:05d}"
        reducer_id = f"{config.run_id}.{partition_label}"
        device = config.devices[ordinal % len(config.devices)]
        out_dir = spectra_root / partition_label
        argv = [
            config.executable,
            "assemble-spectra",
            "--shard-manifest",
            str(row_dict["measurement_shard_manifest_path"]),
            "--out-dir",
            str(out_dir),
            "--run-id",
            reducer_id,
            "--device",
            device,
        ]
        if config.only_ok:
            argv.append("--only-ok")
        if config.drop_duplicate_measurements:
            argv.append("--drop-duplicate-measurements")

        reducers.append(
            {
                "reducer_id": reducer_id,
                "reducer_index": ordinal,
                "reducer_count": int(len(manifest)),
                "partition_index": partition_index,
                "partition_count": int(row_dict.get("partition_count") or 0),
                "device": device,
                "measurement_rows": int(row_dict["rows"]),
                "target_count": int(row_dict["target_count"]),
                "shard_manifest_path": str(row_dict["measurement_shard_manifest_path"]),
                "output_dir": str(out_dir),
                "argv": argv,
                "shell": " ".join(shlex.quote(part) for part in argv),
            }
        )

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "luxquarry_reducer_fanout_plan",
        "run_id": config.run_id,
        "partition_manifest_path": str(config.partition_manifest_path),
        "output_dir": str(config.output_dir),
        "spectra_out_dir": str(spectra_root),
        "executable": config.executable,
        "devices": list(config.devices),
        "only_ok": config.only_ok,
        "drop_duplicate_measurements": config.drop_duplicate_measurements,
        "partition_manifest_rows": int(len(manifest)),
        "reducer_count": len(reducers),
        "total_measurement_rows": sum(int(row["measurement_rows"]) for row in reducers),
        "total_target_count_by_partition": sum(int(row["target_count"]) for row in reducers),
        "reducers": reducers,
    }


def write_reducer_plan(plan: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shell_path = output_path.with_suffix(".sh")
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for reducer in plan["reducers"]:
        lines.append(str(reducer["shell"]) + " &")
    lines.append("wait")
    shell_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shell_path.chmod(0o755)


def run_reducer_plan(config: LocalReducerRunConfig) -> dict[str, Any]:
    started = time.perf_counter()
    if config.max_parallel is not None and config.max_parallel <= 0:
        raise ValueError("max_parallel must be positive")

    plan = json.loads(config.plan_path.read_text(encoding="utf-8"))
    run_output_dir = _run_output_dir(config.plan_path, plan)
    logs_dir = config.logs_dir or run_output_dir / "reducer_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    reducers = list(plan.get("reducers") or [])
    max_parallel = config.max_parallel or len(reducers) or 1
    pending = list(reducers)
    running: list[tuple[dict[str, Any], subprocess.Popen[bytes], Any, Any, float]] = []
    results: list[dict[str, Any]] = []

    while pending or running:
        while pending and len(running) < max_parallel:
            reducer = pending.pop(0)
            if config.resume:
                skipped = _completed_reducer_result(reducer, logs_dir, config.plan_path)
                if skipped is not None:
                    results.append(skipped)
                    continue
            running.append(_launch_reducer(reducer, logs_dir))

        still_running: list[tuple[dict[str, Any], subprocess.Popen[bytes], Any, Any, float]] = []
        for reducer, proc, stdout_file, stderr_file, reducer_started in running:
            returncode = proc.poll()
            if returncode is None:
                still_running.append((reducer, proc, stdout_file, stderr_file, reducer_started))
                continue
            results.append(_reducer_process_result(reducer, logs_dir, int(returncode), reducer_started, skipped=False))
            stdout_file.close()
            stderr_file.close()
        running = still_running
        if pending or running:
            time.sleep(0.1)

    failed = [row for row in results if int(row["returncode"]) != 0]
    status = "complete" if not failed else "reducer_failed"
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "luxquarry_local_reducer_runner",
        "status": status,
        "run_id": plan.get("run_id"),
        "plan_path": str(config.plan_path),
        "output_dir": str(run_output_dir),
        "logs_dir": str(logs_dir),
        "resume": config.resume,
        "max_parallel": max_parallel,
        "reducer_count": len(reducers),
        "launched_reducer_count": sum(1 for row in results if not row.get("skipped")),
        "skipped_reducer_count": sum(1 for row in results if row.get("skipped")),
        "failed_reducer_count": len(failed),
        "total_wall_sec": time.perf_counter() - started,
        "reducer_results": sorted(results, key=lambda row: int(row.get("reducer_index") or 0)),
    }
    summary_path = run_output_dir / "local_reducer_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failed and not config.allow_failed_reducers:
        raise RuntimeError(f"{len(failed)} reducer(s) failed; see {logs_dir}")
    return summary


def collect_reducer_plan(config: ReducerCollectConfig) -> dict[str, Any]:
    started = time.perf_counter()
    plan = json.loads(config.plan_path.read_text(encoding="utf-8"))
    run_output_dir = _run_output_dir(config.plan_path, plan)
    output_path = config.output_path or run_output_dir / "reducer_collect_summary.json"
    rows: list[dict[str, Any]] = []

    for reducer in plan.get("reducers") or []:
        reducer_out = _resolve_path(config.plan_path, reducer.get("output_dir"))
        summary_path = reducer_out / "assemble_summary.json"
        status = "missing"
        summary: dict[str, Any] = {}
        error = None
        spectra_path: Path | None = None
        target_summary_path: Path | None = None
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                spectra_raw = summary.get("spectra_measurements_path")
                target_summary_raw = summary.get("target_summary_path")
                spectra_path = _resolve_path(config.plan_path, spectra_raw) if spectra_raw else None
                target_summary_path = _resolve_path(config.plan_path, target_summary_raw) if target_summary_raw else None
                status = (
                    "complete"
                    if spectra_path is not None
                    and target_summary_path is not None
                    and spectra_path.exists()
                    and target_summary_path.exists()
                    else "missing_output"
                )
            except Exception as exc:
                status = "error"
                error = f"{type(exc).__name__}: {exc}"

        rows.append(
            {
                "reducer_id": reducer.get("reducer_id"),
                "reducer_index": int(reducer.get("reducer_index") or 0),
                "partition_index": int(reducer.get("partition_index") or 0),
                "device": reducer.get("device"),
                "status": status,
                "error": error,
                "output_dir": str(reducer_out),
                "summary_path": str(summary_path),
                "spectra_measurements_path": str(spectra_path) if spectra_path is not None else None,
                "target_summary_path": str(target_summary_path) if target_summary_path is not None else None,
                "input_measurement_rows": int(summary.get("input_measurement_rows") or 0),
                "spectra_measurement_rows": int(summary.get("spectra_measurement_rows") or 0),
                "target_count": int(summary.get("target_count") or 0),
                "read_shards_wall_sec": float(summary.get("read_shards_wall_sec") or 0.0),
                "sort_wall_sec": float(summary.get("sort_wall_sec") or 0.0),
                "write_spectra_wall_sec": float(summary.get("write_spectra_wall_sec") or 0.0),
                "target_summary_wall_sec": float(summary.get("target_summary_wall_sec") or 0.0),
                "total_wall_sec": float(summary.get("total_wall_sec") or 0.0),
            }
        )

    complete_count = sum(1 for row in rows if row["status"] == "complete")
    failed_count = sum(1 for row in rows if row["status"] not in {"complete"})
    reducer_manifest_path = output_path.with_name("reducer_outputs.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(reducer_manifest_path, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "luxquarry_reducer_output_collector",
        "run_id": plan.get("run_id"),
        "plan_path": str(config.plan_path),
        "output_dir": str(run_output_dir),
        "complete": failed_count == 0,
        "allow_incomplete": config.allow_incomplete,
        "reducer_count": len(rows),
        "complete_reducer_count": complete_count,
        "failed_reducer_count": failed_count,
        "input_measurement_rows": sum(int(row["input_measurement_rows"]) for row in rows),
        "spectra_measurement_rows": sum(int(row["spectra_measurement_rows"]) for row in rows),
        "target_count": sum(int(row["target_count"]) for row in rows),
        "reducer_sum_wall_sec": sum(float(row["total_wall_sec"]) for row in rows),
        "collect_wall_sec": time.perf_counter() - started,
        "reducer_manifest_path": str(reducer_manifest_path),
        "reducers": rows,
    }
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failed_count and not config.allow_incomplete:
        raise RuntimeError(f"{failed_count} reducer output(s) missing or failed")
    return summary


def _launch_reducer(
    reducer: dict[str, Any],
    logs_dir: Path,
) -> tuple[dict[str, Any], subprocess.Popen[bytes], Any, Any, float]:
    reducer_id = str(reducer["reducer_id"])
    stdout_path = logs_dir / f"{reducer_id}.stdout.log"
    stderr_path = logs_dir / f"{reducer_id}.stderr.log"
    stdout_file = stdout_path.open("wb")
    stderr_file = stderr_path.open("wb")
    started = time.perf_counter()
    proc = subprocess.Popen(list(reducer["argv"]), stdout=stdout_file, stderr=stderr_file)
    return reducer, proc, stdout_file, stderr_file, started


def _reducer_process_result(
    reducer: dict[str, Any],
    logs_dir: Path,
    returncode: int,
    started: float,
    *,
    skipped: bool,
) -> dict[str, Any]:
    reducer_id = str(reducer["reducer_id"])
    return {
        "reducer_id": reducer_id,
        "reducer_index": reducer.get("reducer_index"),
        "partition_index": reducer.get("partition_index"),
        "device": reducer.get("device"),
        "returncode": returncode,
        "skipped": skipped,
        "wall_sec": 0.0 if skipped else time.perf_counter() - started,
        "stdout_log": str(logs_dir / f"{reducer_id}.stdout.log"),
        "stderr_log": str(logs_dir / f"{reducer_id}.stderr.log"),
        "argv": list(reducer["argv"]),
    }


def _completed_reducer_result(reducer: dict[str, Any], logs_dir: Path, plan_path: Path) -> dict[str, Any] | None:
    output_dir = _resolve_path(plan_path, reducer.get("output_dir"))
    summary_path = output_dir / "assemble_summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    spectra_raw = summary.get("spectra_measurements_path")
    target_summary_raw = summary.get("target_summary_path")
    if not spectra_raw or not target_summary_raw:
        return None
    spectra_path = _resolve_path(plan_path, spectra_raw)
    target_summary_path = _resolve_path(plan_path, target_summary_raw)
    if not spectra_path.exists() or not target_summary_path.exists():
        return None
    return _reducer_process_result(reducer, logs_dir, 0, time.perf_counter(), skipped=True)


def _run_output_dir(plan_path: Path, plan: dict[str, Any]) -> Path:
    configured = Path(str(plan.get("output_dir") or plan_path.parent))
    if configured.is_absolute() or configured.exists():
        return configured
    if plan_path.parent.name == configured.name:
        return plan_path.parent
    return configured


def _resolve_path(plan_path: Path, value: object) -> Path:
    raw = Path(str(value or ""))
    if raw.is_absolute() or raw.exists():
        return raw
    for root in [Path.cwd(), *plan_path.parents]:
        candidate = root / raw
        if candidate.exists():
            return candidate
    return raw
