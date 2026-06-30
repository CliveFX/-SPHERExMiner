from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProjectedTargetInflationConfig:
    input_path: Path
    output_path: Path
    repeat_factor: int
    jitter_pix: float = 0.0
    seed: int = 0
    preserve_original_ids: bool = False


def inflate_projected_targets(config: ProjectedTargetInflationConfig) -> dict[str, Any]:
    started = time.perf_counter()
    if config.repeat_factor <= 0:
        raise ValueError("repeat_factor must be positive")
    if config.jitter_pix < 0:
        raise ValueError("jitter_pix must be non-negative")

    read_started = time.perf_counter()
    source = pd.read_parquet(config.input_path)
    read_wall = time.perf_counter() - read_started
    _require_columns(
        source,
        [
            "target_id",
            "source_id",
            "x_pix",
            "y_pix",
            "naxis1",
            "naxis2",
            "edge_distance_pix",
        ],
    )

    build_started = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    chunks: list[pd.DataFrame] = []
    for repeat_index in range(config.repeat_factor):
        chunk = source.copy()
        if not config.preserve_original_ids or repeat_index > 0:
            suffix = f"__bench_r{repeat_index:04d}"
            chunk["target_id"] = chunk["target_id"].astype("string") + suffix
            chunk["source_id"] = chunk["source_id"].astype("string") + suffix
        if config.jitter_pix > 0:
            dx = rng.uniform(-config.jitter_pix, config.jitter_pix, size=len(chunk))
            dy = rng.uniform(-config.jitter_pix, config.jitter_pix, size=len(chunk))
            x_pix = chunk["x_pix"].to_numpy(dtype=np.float64) + dx
            y_pix = chunk["y_pix"].to_numpy(dtype=np.float64) + dy
            x_pix = np.clip(x_pix, 1.0, chunk["naxis1"].to_numpy(dtype=np.float64))
            y_pix = np.clip(y_pix, 1.0, chunk["naxis2"].to_numpy(dtype=np.float64))
            chunk["x_pix"] = x_pix
            chunk["y_pix"] = y_pix
            chunk["edge_distance_pix"] = np.minimum.reduce(
                [
                    x_pix - 1.0,
                    y_pix - 1.0,
                    chunk["naxis1"].to_numpy(dtype=np.float64) - x_pix,
                    chunk["naxis2"].to_numpy(dtype=np.float64) - y_pix,
                ]
            )
        chunk["benchmark_inflated"] = True
        chunk["benchmark_repeat_index"] = np.int32(repeat_index)
        chunk["benchmark_parent_target_id"] = source["target_id"].astype("string").to_numpy()
        chunks.append(chunk)

    inflated = pd.concat(chunks, ignore_index=True)
    build_wall = time.perf_counter() - build_started

    write_started = time.perf_counter()
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    inflated.to_parquet(config.output_path, index=False)
    write_wall = time.perf_counter() - write_started

    summary = {
        "backend": "projected_target_benchmark_inflator",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(config.input_path),
        "output_path": str(config.output_path),
        "repeat_factor": int(config.repeat_factor),
        "jitter_pix": float(config.jitter_pix),
        "seed": int(config.seed),
        "preserve_original_ids": bool(config.preserve_original_ids),
        "input_rows": int(len(source)),
        "output_rows": int(len(inflated)),
        "input_unique_targets": int(source["target_id"].nunique(dropna=False)),
        "output_unique_targets": int(inflated["target_id"].nunique(dropna=False)),
        "frame_count": int(source["frame_group_id"].nunique(dropna=False)) if "frame_group_id" in source.columns else 0,
        "read_wall_sec": read_wall,
        "build_wall_sec": build_wall,
        "write_wall_sec": write_wall,
        "total_wall_sec": time.perf_counter() - started,
        "note": (
            "Benchmark-only synthetic target multiplication. Do not use this "
            "output as a science catalog."
        ),
    }
    summary_path = config.output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"projected target table missing required columns: {missing}")
