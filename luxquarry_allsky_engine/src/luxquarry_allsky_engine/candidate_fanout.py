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
class CandidateFanoutPlanConfig:
    reducer_outputs_path: Path
    output_dir: Path
    run_id: str
    executable: str = ".venv/bin/luxquarry-allsky"
    devices: tuple[str, ...] = ("cuda:0",)
    output_prefix: str = "baseline"
    flux_column: str = "aperture_flux_uJy"
    min_abs_zscore: float = 5.0
    min_measurements: int = 10
    include_flagged: bool = False
    max_candidates: int | None = None
    max_partitions: int | None = None


@dataclass(frozen=True)
class LocalCandidateFanoutRunConfig:
    plan_path: Path
    logs_dir: Path | None = None
    resume: bool = False
    max_parallel: int | None = None
    allow_failed_scorers: bool = False


@dataclass(frozen=True)
class CandidateFanoutCollectConfig:
    plan_path: Path
    output_path: Path | None = None
    allow_incomplete: bool = False


def build_candidate_fanout_plan(config: CandidateFanoutPlanConfig) -> dict[str, Any]:
    if not config.devices:
        raise ValueError("at least one scorer device is required")
    if config.max_partitions is not None and config.max_partitions <= 0:
        raise ValueError("max_partitions must be positive")

    reducer_outputs = pd.read_parquet(config.reducer_outputs_path)
    if "status" in reducer_outputs.columns:
        reducer_outputs = reducer_outputs[reducer_outputs["status"] == "complete"]
    if config.max_partitions is not None:
        reducer_outputs = reducer_outputs.head(config.max_partitions)

    required = {"partition_index", "spectra_measurements_path", "spectra_measurement_rows", "target_count"}
    missing = sorted(required.difference(reducer_outputs.columns))
    if missing:
        raise ValueError(f"Reducer output manifest missing columns: {missing}")

    scorers: list[dict[str, Any]] = []
    for ordinal, row in enumerate(reducer_outputs.sort_values("partition_index").itertuples(index=False)):
        row_dict = row._asdict()
        partition_index = int(row_dict["partition_index"])
        partition_label = f"part{partition_index:05d}"
        scorer_id = f"{config.run_id}.{partition_label}"
        device = config.devices[ordinal % len(config.devices)]
        out_dir = config.output_dir / "candidates" / partition_label
        argv = [
            config.executable,
            "score-spectra-candidates",
            "--spectra",
            str(row_dict["spectra_measurements_path"]),
            "--out-dir",
            str(out_dir),
            "--run-id",
            scorer_id,
            "--device",
            device,
            "--output-prefix",
            config.output_prefix,
            "--flux-column",
            config.flux_column,
            "--min-abs-zscore",
            str(config.min_abs_zscore),
            "--min-measurements",
            str(config.min_measurements),
        ]
        if config.include_flagged:
            argv.append("--include-flagged")
        if config.max_candidates is not None:
            argv.extend(["--max-candidates", str(config.max_candidates)])

        scorers.append(
            {
                "scorer_id": scorer_id,
                "scorer_index": ordinal,
                "scorer_count": int(len(reducer_outputs)),
                "partition_index": partition_index,
                "device": device,
                "spectra_measurements_path": str(row_dict["spectra_measurements_path"]),
                "spectra_measurement_rows": int(row_dict["spectra_measurement_rows"]),
                "target_count": int(row_dict["target_count"]),
                "output_dir": str(out_dir),
                "output_prefix": config.output_prefix,
                "argv": argv,
                "shell": " ".join(shlex.quote(part) for part in argv),
            }
        )

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "luxquarry_candidate_scorer_fanout_plan",
        "run_id": config.run_id,
        "reducer_outputs_path": str(config.reducer_outputs_path),
        "output_dir": str(config.output_dir),
        "executable": config.executable,
        "devices": list(config.devices),
        "output_prefix": config.output_prefix,
        "flux_column": config.flux_column,
        "min_abs_zscore": config.min_abs_zscore,
        "min_measurements": config.min_measurements,
        "include_flagged": config.include_flagged,
        "max_candidates": config.max_candidates,
        "scorer_count": len(scorers),
        "total_spectra_measurement_rows": sum(int(row["spectra_measurement_rows"]) for row in scorers),
        "total_target_count_by_partition": sum(int(row["target_count"]) for row in scorers),
        "scorers": scorers,
    }


def write_candidate_fanout_plan(plan: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shell_path = output_path.with_suffix(".sh")
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for scorer in plan["scorers"]:
        lines.append(str(scorer["shell"]) + " &")
    lines.append("wait")
    shell_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shell_path.chmod(0o755)


def run_candidate_fanout_plan(config: LocalCandidateFanoutRunConfig) -> dict[str, Any]:
    started = time.perf_counter()
    if config.max_parallel is not None and config.max_parallel <= 0:
        raise ValueError("max_parallel must be positive")
    plan = json.loads(config.plan_path.read_text(encoding="utf-8"))
    run_output_dir = _run_output_dir(config.plan_path, plan)
    logs_dir = config.logs_dir or run_output_dir / "candidate_scorer_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    scorers = list(plan.get("scorers") or [])
    max_parallel = config.max_parallel or len(scorers) or 1
    pending = list(scorers)
    running: list[tuple[dict[str, Any], subprocess.Popen[bytes], Any, Any, float]] = []
    results: list[dict[str, Any]] = []

    while pending or running:
        while pending and len(running) < max_parallel:
            scorer = pending.pop(0)
            if config.resume:
                skipped = _completed_scorer_result(scorer, logs_dir, config.plan_path)
                if skipped is not None:
                    results.append(skipped)
                    continue
            running.append(_launch_scorer(scorer, logs_dir))

        still_running: list[tuple[dict[str, Any], subprocess.Popen[bytes], Any, Any, float]] = []
        for scorer, proc, stdout_file, stderr_file, scorer_started in running:
            returncode = proc.poll()
            if returncode is None:
                still_running.append((scorer, proc, stdout_file, stderr_file, scorer_started))
                continue
            results.append(_scorer_process_result(scorer, logs_dir, int(returncode), scorer_started, skipped=False))
            stdout_file.close()
            stderr_file.close()
        running = still_running
        if pending or running:
            time.sleep(0.1)

    failed = [row for row in results if int(row["returncode"]) != 0]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "luxquarry_local_candidate_scorer_runner",
        "status": "complete" if not failed else "scorer_failed",
        "run_id": plan.get("run_id"),
        "plan_path": str(config.plan_path),
        "output_dir": str(run_output_dir),
        "logs_dir": str(logs_dir),
        "resume": config.resume,
        "max_parallel": max_parallel,
        "scorer_count": len(scorers),
        "launched_scorer_count": sum(1 for row in results if not row.get("skipped")),
        "skipped_scorer_count": sum(1 for row in results if row.get("skipped")),
        "failed_scorer_count": len(failed),
        "total_wall_sec": time.perf_counter() - started,
        "scorer_results": sorted(results, key=lambda row: int(row.get("scorer_index") or 0)),
    }
    summary_path = run_output_dir / "local_candidate_scorer_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failed and not config.allow_failed_scorers:
        raise RuntimeError(f"{len(failed)} scorer(s) failed; see {logs_dir}")
    return summary


def collect_candidate_fanout_plan(config: CandidateFanoutCollectConfig) -> dict[str, Any]:
    started = time.perf_counter()
    plan = json.loads(config.plan_path.read_text(encoding="utf-8"))
    run_output_dir = _run_output_dir(config.plan_path, plan)
    output_path = config.output_path or run_output_dir / "candidate_fanout_collect_summary.json"
    rows: list[dict[str, Any]] = []

    for scorer in plan.get("scorers") or []:
        scorer_out = _resolve_path(config.plan_path, scorer.get("output_dir"))
        output_prefix = str(scorer.get("output_prefix") or plan.get("output_prefix") or "baseline")
        summary_path = scorer_out / f"{output_prefix}_candidate_summary.json"
        status = "missing"
        summary: dict[str, Any] = {}
        error = None
        candidates_path: Path | None = None
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                candidates_raw = summary.get("candidates_path")
                candidates_path = _resolve_path(config.plan_path, candidates_raw) if candidates_raw else None
                status = "complete" if candidates_path is not None and candidates_path.exists() else "missing_output"
            except Exception as exc:
                status = "error"
                error = f"{type(exc).__name__}: {exc}"

        rows.append(
            {
                "scorer_id": scorer.get("scorer_id"),
                "scorer_index": int(scorer.get("scorer_index") or 0),
                "partition_index": int(scorer.get("partition_index") or 0),
                "device": scorer.get("device"),
                "status": status,
                "error": error,
                "output_dir": str(scorer_out),
                "summary_path": str(summary_path),
                "candidates_path": str(candidates_path) if candidates_path is not None else None,
                "input_measurement_rows": int(summary.get("input_measurement_rows") or 0),
                "filtered_measurement_rows": int(summary.get("filtered_measurement_rows") or 0),
                "target_count": int(summary.get("target_count") or 0),
                "candidate_count": int(summary.get("candidate_count") or 0),
                "candidate_target_count": int(summary.get("candidate_target_count") or 0),
                "total_wall_sec": float(summary.get("total_wall_sec") or 0.0),
            }
        )

    complete_count = sum(1 for row in rows if row["status"] == "complete")
    failed_count = sum(1 for row in rows if row["status"] not in {"complete"})
    scorer_manifest_path = output_path.with_name("candidate_scorer_outputs.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(scorer_manifest_path, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "luxquarry_candidate_scorer_output_collector",
        "run_id": plan.get("run_id"),
        "plan_path": str(config.plan_path),
        "output_dir": str(run_output_dir),
        "complete": failed_count == 0,
        "allow_incomplete": config.allow_incomplete,
        "scorer_count": len(rows),
        "complete_scorer_count": complete_count,
        "failed_scorer_count": failed_count,
        "input_measurement_rows": sum(int(row["input_measurement_rows"]) for row in rows),
        "filtered_measurement_rows": sum(int(row["filtered_measurement_rows"]) for row in rows),
        "target_count_by_partition": sum(int(row["target_count"]) for row in rows),
        "candidate_count": sum(int(row["candidate_count"]) for row in rows),
        "candidate_target_count_by_partition": sum(int(row["candidate_target_count"]) for row in rows),
        "scorer_sum_wall_sec": sum(float(row["total_wall_sec"]) for row in rows),
        "collect_wall_sec": time.perf_counter() - started,
        "scorer_manifest_path": str(scorer_manifest_path),
        "scorers": rows,
    }
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failed_count and not config.allow_incomplete:
        raise RuntimeError(f"{failed_count} candidate scorer output(s) missing or failed")
    return summary


def _launch_scorer(
    scorer: dict[str, Any],
    logs_dir: Path,
) -> tuple[dict[str, Any], subprocess.Popen[bytes], Any, Any, float]:
    scorer_id = str(scorer["scorer_id"])
    stdout_path = logs_dir / f"{scorer_id}.stdout.log"
    stderr_path = logs_dir / f"{scorer_id}.stderr.log"
    stdout_file = stdout_path.open("wb")
    stderr_file = stderr_path.open("wb")
    started = time.perf_counter()
    proc = subprocess.Popen(list(scorer["argv"]), stdout=stdout_file, stderr=stderr_file)
    return scorer, proc, stdout_file, stderr_file, started


def _scorer_process_result(
    scorer: dict[str, Any],
    logs_dir: Path,
    returncode: int,
    started: float,
    *,
    skipped: bool,
) -> dict[str, Any]:
    scorer_id = str(scorer["scorer_id"])
    return {
        "scorer_id": scorer_id,
        "scorer_index": scorer.get("scorer_index"),
        "partition_index": scorer.get("partition_index"),
        "device": scorer.get("device"),
        "returncode": returncode,
        "skipped": skipped,
        "wall_sec": 0.0 if skipped else time.perf_counter() - started,
        "stdout_log": str(logs_dir / f"{scorer_id}.stdout.log"),
        "stderr_log": str(logs_dir / f"{scorer_id}.stderr.log"),
        "argv": list(scorer["argv"]),
    }


def _completed_scorer_result(scorer: dict[str, Any], logs_dir: Path, plan_path: Path) -> dict[str, Any] | None:
    output_dir = _resolve_path(plan_path, scorer.get("output_dir"))
    output_prefix = str(scorer.get("output_prefix") or "baseline")
    summary_path = output_dir / f"{output_prefix}_candidate_summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    candidates_raw = summary.get("candidates_path")
    if not candidates_raw:
        return None
    candidates_path = _resolve_path(plan_path, candidates_raw)
    if not candidates_path.exists():
        return None
    return _scorer_process_result(scorer, logs_dir, 0, time.perf_counter(), skipped=True)


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
