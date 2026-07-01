from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ProjectedDepthSummaryConfig:
    projected_targets_path: Path
    output_path: Path | None = None
    backend: str = "auto"
    only_in_frame: bool = True


def summarize_projected_depth(config: ProjectedDepthSummaryConfig) -> dict[str, Any]:
    if config.backend not in {"auto", "cudf", "pandas"}:
        raise ValueError("backend must be auto, cudf, or pandas")
    started = time.perf_counter()
    backend = _resolve_backend(config.backend)
    if backend == "cudf":
        summary = _summarize_projected_depth_cudf(config)
    else:
        summary = _summarize_projected_depth_pandas(config)
    summary["created_utc"] = datetime.now(timezone.utc).isoformat()
    summary["projected_targets_path"] = str(config.projected_targets_path)
    summary["only_in_frame"] = bool(config.only_in_frame)
    summary["total_wall_sec"] = time.perf_counter() - started
    if config.output_path is not None:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _resolve_backend(requested: str) -> str:
    if requested == "pandas":
        return "pandas"
    if requested == "cudf":
        return "cudf"
    try:
        import cudf  # noqa: F401
    except Exception:
        return "pandas"
    return "cudf"


def _summarize_projected_depth_pandas(config: ProjectedDepthSummaryConfig) -> dict[str, Any]:
    read_started = time.perf_counter()
    targets = pd.read_parquet(config.projected_targets_path, columns=_columns(config.only_in_frame))
    read_wall = time.perf_counter() - read_started
    if config.only_in_frame and "in_frame" in targets.columns:
        targets = targets[targets["in_frame"].astype(bool)]
    group_started = time.perf_counter()
    depth = targets.groupby(["catalog", "target_id"], sort=False).size().rename("depth").reset_index()
    group_wall = time.perf_counter() - group_started
    return {
        "backend": "pandas_projected_depth_summary",
        "input_rows": int(len(targets)),
        "read_wall_sec": read_wall,
        "groupby_wall_sec": group_wall,
        **_depth_summary_from_pandas(depth),
    }


def _summarize_projected_depth_cudf(config: ProjectedDepthSummaryConfig) -> dict[str, Any]:
    import cudf

    read_started = time.perf_counter()
    targets = cudf.read_parquet(str(config.projected_targets_path), columns=_columns(config.only_in_frame))
    read_wall = time.perf_counter() - read_started
    if config.only_in_frame and "in_frame" in targets.columns:
        targets = targets[targets["in_frame"].astype("bool")]
    input_rows = int(len(targets))
    group_started = time.perf_counter()
    depth = targets.groupby(["catalog", "target_id"]).size().reset_index(name="depth")
    group_wall = time.perf_counter() - group_started
    # The reduced depth table is usually far smaller than the projected target table.
    # Convert only the grouped result so JSON assembly stays simple and portable.
    to_pandas_started = time.perf_counter()
    depth_pd = depth.to_pandas()
    to_pandas_wall = time.perf_counter() - to_pandas_started
    return {
        "backend": "cudf_projected_depth_summary",
        "input_rows": input_rows,
        "read_wall_sec": read_wall,
        "groupby_wall_sec": group_wall,
        "grouped_to_pandas_wall_sec": to_pandas_wall,
        **_depth_summary_from_pandas(depth_pd),
    }


def _depth_summary_from_pandas(depth: pd.DataFrame) -> dict[str, Any]:
    if depth.empty:
        return {
            "unique_targets": 0,
            "mean_depth": 0.0,
            "median_depth": 0.0,
            "max_depth": 0,
            "depth_histogram": {},
            "by_catalog": {},
        }
    by_catalog = {}
    for catalog, part in depth.groupby("catalog", dropna=False, sort=True):
        values = pd.to_numeric(part["depth"], errors="coerce").fillna(0).astype("int64")
        by_catalog[str(catalog)] = {
            "targets": int(len(values)),
            "mean_depth": float(values.mean()),
            "median_depth": float(values.median()),
            "max_depth": int(values.max()),
            "depth_histogram": _histogram(values),
        }
    values = pd.to_numeric(depth["depth"], errors="coerce").fillna(0).astype("int64")
    return {
        "unique_targets": int(len(values)),
        "mean_depth": float(values.mean()),
        "median_depth": float(values.median()),
        "max_depth": int(values.max()),
        "depth_histogram": _histogram(values),
        "by_catalog": by_catalog,
    }


def _histogram(values: pd.Series) -> dict[str, int]:
    counts = values.value_counts().sort_index()
    return {str(int(depth)): int(count) for depth, count in counts.items()}


def _columns(only_in_frame: bool) -> list[str]:
    columns = ["catalog", "target_id"]
    if only_in_frame:
        columns.append("in_frame")
    return columns
