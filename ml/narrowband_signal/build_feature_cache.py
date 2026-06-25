#!/usr/bin/env python3
"""Build parallel tensor shards for narrowband line training."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.common.ragged_spectra import FEATURE_COLUMNS, make_point_features, write_status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--cache-name", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/niroseti/spherex_cache/ml_feature_caches"))
    parser.add_argument("--split", default="train", choices=["train", "validation", "test", "all"])
    parser.add_argument("--quality-category", action="append", choices=["good", "review", "bad"], default=["good", "review"])
    parser.add_argument("--max-points", type=int, default=384)
    parser.add_argument("--max-targets-per-run", type=int)
    parser.add_argument("--line-min-nm", type=float, default=700.0)
    parser.add_argument("--line-max-nm", type=float, default=5000.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--status-every", type=int, default=1)
    args = parser.parse_args()

    started = time.perf_counter()
    out_dir = args.output_root / args.cache_name
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "cache_status.json"

    targets = pd.read_parquet(args.dataset_dir / "narrowband_targets.parquet")
    truth = _truth_map(args.dataset_dir / "injection_truth.parquet", args.line_min_nm, args.line_max_nm)
    if args.split != "all":
        targets = targets[targets["split_id"].astype(str).eq(args.split)].copy()
    if args.quality_category and "spectrum_quality_category" in targets:
        targets = targets[targets["spectrum_quality_category"].astype(str).isin(set(args.quality_category))].copy()
    targets["_example_key"] = targets["run_name"].astype(str) + "::" + targets["target_id"].astype(str)
    targets["is_positive"] = targets["_example_key"].astype(str).isin(truth)
    run_groups = [(run_name, group.copy()) for run_name, group in targets.groupby("run_name", sort=True)]
    if not run_groups:
        raise SystemExit("No targets selected for feature cache")

    completed = 0
    summaries: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(
                _build_run_shard,
                run_name,
                group,
                truth,
                args.dataset_dir,
                shard_dir,
                args.max_points,
                args.max_targets_per_run,
                args.line_min_nm,
                args.line_max_nm,
            ): run_name
            for run_name, group in run_groups
        }
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            try:
                summary = future.result()
            except Exception as exc:
                summary = {"run_name": futures[future], "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            summaries.append(summary)
            if args.status_every > 0 and completed % args.status_every == 0:
                write_status(
                    status_path,
                    {
                        "cache_name": args.cache_name,
                        "status": "running",
                        "dataset_dir": str(args.dataset_dir),
                        "run_index": completed,
                        "run_count": len(run_groups),
                        "latest_run": summary.get("run_name"),
                        "examples": int(sum(row.get("examples", 0) for row in summaries)),
                        "positives": int(sum(row.get("positives", 0) for row in summaries)),
                        "elapsed_sec": time.perf_counter() - started,
                        "workers": int(args.workers),
                    },
                )

    manifest = {
        "cache_name": args.cache_name,
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(out_dir),
        "shard_dir": str(shard_dir),
        "feature_columns": FEATURE_COLUMNS,
        "max_points": int(args.max_points),
        "line_min_nm": float(args.line_min_nm),
        "line_max_nm": float(args.line_max_nm),
        "split": args.split,
        "quality_categories": list(args.quality_category or []),
        "run_count": len(run_groups),
        "successful_run_count": sum(1 for row in summaries if row.get("status") == "ok"),
        "error_run_count": sum(1 for row in summaries if row.get("status") == "error"),
        "examples": int(sum(row.get("examples", 0) for row in summaries)),
        "positives": int(sum(row.get("positives", 0) for row in summaries)),
        "elapsed_sec": time.perf_counter() - started,
        "shards": summaries,
    }
    (out_dir / "cache_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_status(status_path, {**manifest, "status": "done"})
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


def _truth_map(path: Path, line_min_nm: float, line_max_nm: float) -> dict[str, float]:
    if not path.exists():
        return {}
    truth = pd.read_parquet(path)
    if truth.empty:
        return {}
    truth = truth.dropna(subset=["run_name", "target_id", "injected_line_nm"]).copy()
    truth["injected_line_nm"] = pd.to_numeric(truth["injected_line_nm"], errors="coerce")
    truth = truth[(truth["injected_line_nm"] >= line_min_nm) & (truth["injected_line_nm"] <= line_max_nm)]
    out: dict[str, float] = {}
    keys = truth["run_name"].astype(str) + "::" + truth["target_id"].astype(str)
    for key, rows in truth.groupby(keys, sort=False):
        out[str(key)] = float(rows["injected_line_nm"].median())
    return out


def _build_run_shard(
    run_name: str,
    targets: pd.DataFrame,
    truth: dict[str, float],
    dataset_dir: Path,
    shard_dir: Path,
    max_points: int,
    max_targets_per_run: int | None,
    line_min_nm: float,
    line_max_nm: float,
) -> dict[str, object]:
    if max_targets_per_run is not None and len(targets) > max_targets_per_run:
        injected = targets[targets["is_positive"].fillna(False).astype(bool)].copy()
        rest = targets[~targets["_example_key"].astype(str).isin(set(injected["_example_key"].astype(str)))].copy()
        sort_cols = [col for col in ["spectrum_quality_score", "n_usable_measurements"] if col in rest]
        if sort_cols:
            rest = rest.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
        targets = pd.concat([injected, rest.head(max(0, max_targets_per_run - len(injected)))], ignore_index=True)
    keep = set(targets["_example_key"].astype(str))
    points = pd.read_parquet(dataset_dir / "narrowband_points.parquet", filters=[("run_name", "=", run_name)])
    points["_example_key"] = points["run_name"].astype(str) + "::" + points["target_id"].astype(str)
    points = points[points["_example_key"].isin(keep)].copy()
    target_ids: list[str] = []
    x_chunks: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    y: list[float] = []
    line_norm: list[float] = []
    span = max(line_max_nm - line_min_nm, 1.0)
    for key, rows in points.groupby("_example_key", sort=False):
        features = make_point_features(rows)
        if len(features) < 8:
            continue
        clipped = features[:max_points]
        arr = np.zeros((max_points, len(FEATURE_COLUMNS)), dtype=np.float32)
        mask = np.zeros(max_points, dtype=bool)
        arr[: len(clipped)] = clipped
        mask[: len(clipped)] = True
        line_nm = truth.get(str(key), float("nan"))
        positive = math.isfinite(line_nm)
        x_chunks.append(arr)
        masks.append(mask)
        y.append(float(positive))
        line_norm.append(float(np.clip((line_nm - line_min_nm) / span, 0.0, 1.0)) if positive else 0.0)
        target_ids.append(str(rows["target_id"].iloc[0]))
    if not x_chunks:
        return {"run_name": run_name, "status": "empty", "examples": 0, "positives": 0}
    shard_path = shard_dir / f"{_safe_name(run_name)}.npz"
    np.savez_compressed(
        shard_path,
        x=np.stack(x_chunks).astype(np.float32),
        mask=np.stack(masks).astype(bool),
        y=np.asarray(y, dtype=np.float32),
        line_norm=np.asarray(line_norm, dtype=np.float32),
        target_ids=np.asarray(target_ids, dtype=object),
        run_name=np.asarray([run_name], dtype=object),
    )
    return {
        "run_name": run_name,
        "status": "ok",
        "path": str(shard_path),
        "examples": int(len(y)),
        "positives": int(np.asarray(y).sum()),
    }


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:180]


if __name__ == "__main__":
    main()
