from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .object_store import is_s3_uri, stage_input_file


@dataclass(frozen=True)
class ObjectStagingBenchmarkConfig:
    manifest_path: Path
    output_dir: Path
    cache_dir: Path
    concurrency_values: tuple[int, ...] = (1, 2, 4)
    limit: int | None = None
    source_column: str = "path"
    s3_region: str = "us-east-1"
    cache_mode: str = "shared"
    require_s3: bool = False


def benchmark_object_staging(config: ObjectStagingBenchmarkConfig) -> dict[str, Any]:
    if not config.concurrency_values:
        raise ValueError("at least one concurrency value is required")
    for value in config.concurrency_values:
        if value <= 0:
            raise ValueError("concurrency values must be positive")
    if config.cache_mode not in {"shared", "per-concurrency"}:
        raise ValueError("cache_mode must be one of: shared, per-concurrency")

    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(config.manifest_path)
    if config.source_column not in manifest.columns:
        raise ValueError(f"Manifest has no source column {config.source_column!r}: {config.manifest_path}")
    if config.limit is not None:
        manifest = manifest.head(config.limit).copy()
    manifest = manifest.reset_index(drop=True)
    if config.require_s3:
        manifest = manifest[manifest[config.source_column].astype(str).map(is_s3_uri)].copy().reset_index(drop=True)

    source_rows = _source_rows(manifest, config.source_column)
    if not source_rows:
        raise ValueError("No source rows available for object staging benchmark")
    run_summaries: list[dict[str, Any]] = []
    result_frames: list[pd.DataFrame] = []
    for concurrency in config.concurrency_values:
        run_summary, rows = _run_one_concurrency(
            source_rows=source_rows,
            cache_dir=_cache_dir_for(config.cache_dir, config.cache_mode, concurrency),
            concurrency=concurrency,
            s3_region=config.s3_region,
        )
        run_summaries.append(run_summary)
        result_frames.append(pd.DataFrame(rows))

    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    results_path = config.output_dir / "object_staging_results.parquet"
    results.to_parquet(results_path, index=False)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "luxquarry_object_staging_benchmark",
        "manifest_path": str(config.manifest_path),
        "output_dir": str(config.output_dir),
        "cache_dir": str(config.cache_dir),
        "source_column": config.source_column,
        "s3_region": config.s3_region,
        "cache_mode": config.cache_mode,
        "require_s3": config.require_s3,
        "input_rows": int(len(manifest)),
        "s3_rows": sum(1 for row in source_rows if row["is_s3"]),
        "local_rows": sum(1 for row in source_rows if not row["is_s3"]),
        "concurrency_values": list(config.concurrency_values),
        "run_summaries": run_summaries,
        "results_path": str(results_path),
        "total_wall_sec": time.perf_counter() - started,
    }
    summary_path = config.output_dir / "object_staging_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _source_rows(manifest: pd.DataFrame, source_column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(manifest.to_dict(orient="records")):
        source = str(row[source_column])
        rows.append(
            {
                "manifest_row_index": index,
                "frame_group_id": row.get("frame_group_id"),
                "image_id": row.get("image_id"),
                "detector": row.get("detector"),
                "source": source,
                "is_s3": is_s3_uri(source),
                "file_size_bytes": int(row.get("file_size_bytes") or 0),
            }
        )
    return rows


def _run_one_concurrency(
    *,
    source_rows: list[dict[str, Any]],
    cache_dir: Path,
    concurrency: int,
    s3_region: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix=f"stage-c{concurrency}") as pool:
        futures = {
            pool.submit(_stage_one, source_row, cache_dir, concurrency, s3_region): source_row
            for source_row in source_rows
        }
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: int(row["manifest_row_index"]))

    wall = time.perf_counter() - started
    bytes_written = sum(int(row["bytes_written"]) for row in rows)
    error_count = sum(1 for row in rows if row["status"] != "ok")
    cache_hit_count = sum(1 for row in rows if row["status"] == "ok" and int(row["bytes_written"]) == 0)
    rows_ok = len(rows) - error_count
    total_input_bytes = sum(int(row["file_size_bytes"]) for row in rows)
    summary = {
        "concurrency": concurrency,
        "cache_dir": str(cache_dir),
        "source_count": len(source_rows),
        "ok_count": rows_ok,
        "error_count": error_count,
        "cache_hit_count": cache_hit_count,
        "cache_miss_count": rows_ok - cache_hit_count,
        "input_file_size_bytes": total_input_bytes,
        "bytes_written": bytes_written,
        "wall_sec": wall,
        "sources_per_sec": len(source_rows) / wall if wall > 0 else 0.0,
        "transferred_mib_per_sec": (bytes_written / (1024 * 1024)) / wall if wall > 0 else 0.0,
        "input_mib_per_sec": (total_input_bytes / (1024 * 1024)) / wall if wall > 0 else 0.0,
        "mean_stage_wall_sec": sum(float(row["stage_wall_sec"]) for row in rows) / len(rows) if rows else 0.0,
        "p95_stage_wall_sec": _percentile([float(row["stage_wall_sec"]) for row in rows], 95.0),
    }
    for row in rows:
        row.update(
            {
                "concurrency": concurrency,
                "benchmark_cache_dir": str(cache_dir),
            }
        )
    return summary, rows


def _stage_one(
    source_row: dict[str, Any],
    cache_dir: Path,
    concurrency: int,
    s3_region: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    out = dict(source_row)
    try:
        staged_path, bytes_written = stage_input_file(out["source"], cache_dir, s3_region=s3_region)
        out.update(
            {
                "status": "ok",
                "error": None,
                "staged_path": str(staged_path),
                "bytes_written": int(bytes_written),
                "stage_wall_sec": time.perf_counter() - started,
                "cache_hit": int(bytes_written) == 0,
                "concurrency": concurrency,
            }
        )
    except Exception as exc:
        out.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "staged_path": None,
                "bytes_written": 0,
                "stage_wall_sec": time.perf_counter() - started,
                "cache_hit": False,
                "concurrency": concurrency,
            }
        )
    return out


def _cache_dir_for(root: Path, cache_mode: str, concurrency: int) -> Path:
    if cache_mode == "per-concurrency":
        return root / f"concurrency_{concurrency:04d}"
    return root


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
