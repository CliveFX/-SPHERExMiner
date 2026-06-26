#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spherex_laser_miner.viewer import (  # noqa: E402
    VIEWER_INDEX_VERSION,
    _blind_joint_path,
    _campaign_and_target_from_run,
    _clean_json,
    _false_positive_rows,
    _grid_dispatch_campaigns,
    _normalize_blind_candidates_df,
    _read_json,
    _recovery_group_rows,
)


DEFAULT_CACHE_ROOT = Path("/mnt/niroseti/spherex_cache")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lightweight viewer index shards for SPHEREx miner outputs.")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--run-dir", type=Path, action="append", help="Run directory to index. May be repeated.")
    parser.add_argument("--all-runs", action="store_true", help="Index every run under <cache-root>/runs.")
    parser.add_argument("--candidate", action="store_true", default=True)
    parser.add_argument("--recovery", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    runs_root = args.cache_root / "runs"
    if args.all_runs:
        run_dirs = sorted([path for path in runs_root.iterdir() if path.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        run_dirs = [path for path in args.run_dir or []]
    if not run_dirs:
        raise SystemExit("provide --run-dir or --all-runs")

    index_root = args.cache_root / "viewer_indexes"
    results = []
    for run_dir in run_dirs:
        run_dir = run_dir.resolve()
        if not run_dir.exists():
            results.append({"run": str(run_dir), "status": "missing"})
            continue
        row: dict[str, Any] = {"run": run_dir.name, "status": "ok"}
        if args.candidate:
            candidate_path = update_candidate_shard(index_root, runs_root, run_dir)
            row["candidate_shard"] = str(candidate_path) if candidate_path else None
        if args.recovery:
            recovery_path = update_recovery_shard(index_root, runs_root, run_dir)
            row["recovery_shard"] = str(recovery_path) if recovery_path else None
        results.append(row)
        if not args.quiet:
            print(json.dumps(row, sort_keys=True), flush=True)
    if args.quiet:
        print(json.dumps({"indexed": len(results), "results": results}, indent=2), flush=True)


def update_candidate_shard(index_root: Path, runs_root: Path, run_dir: Path) -> Path | None:
    frames = []
    for source, scope in _candidate_sources_for_run(run_dir):
        path = _blind_joint_path(run_dir, scope)
        if path is None:
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        if df.empty:
            continue
        df = _normalize_blind_candidates_df(df)
        frames.append(_annotate_candidate_frame(df, runs_root, run_dir, path, source, scope))
    out_path = index_root / "candidate_shards" / f"{run_dir.name}.parquet"
    if not frames:
        _remove_if_exists(out_path)
        return None
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["index_version"] = VIEWER_INDEX_VERSION
    _atomic_write_parquet(out, out_path)
    return out_path


def update_recovery_shard(index_root: Path, runs_root: Path, run_dir: Path) -> Path | None:
    recovery_dir = run_dir / "recovery_score_mixed_lasers"
    summary_path = recovery_dir / "recovery_summary.json"
    if not summary_path.exists():
        out_path = index_root / "recovery_shards" / f"{run_dir.name}.json"
        _remove_if_exists(out_path)
        return None
    summary = _read_json(summary_path)
    if not isinstance(summary, dict):
        return None
    campaign, target = _campaign_and_target_from_run(run_dir.name, _grid_dispatch_campaigns(runs_root))
    run = {
        "run_name": run_dir.name,
        "campaign": campaign,
        "target": target,
        "mtime": run_dir.stat().st_mtime,
        "mtime_iso": _format_time(run_dir.stat().st_mtime),
        "injection_count": int(summary.get("injection_count") or 0),
        "recovered_count": int(summary.get("recovered_count") or 0),
        "missed_count": int(summary.get("injection_count") or 0) - int(summary.get("recovered_count") or 0),
        "recovery_fraction": _maybe_float(summary.get("recovery_fraction")),
        "candidate_count_above_threshold": int(summary.get("candidate_count_above_threshold") or 0),
        "false_positive_count": int(summary.get("false_positive_count") or 0),
        "false_positives_per_injection": _maybe_float(summary.get("false_positives_per_injection")),
        "min_snr": _maybe_float(summary.get("min_snr")),
        "wavelength_tolerance_nm": _maybe_float(summary.get("wavelength_tolerance_nm")),
        "injections_url": f"/injections?run={run_dir.name}",
        "false_positive_url": f"/injections?run={run_dir.name}&status=candidate",
    }
    blind_path = _blind_joint_path(run_dir, "paired")
    if blind_path is not None:
        run["blind_candidates_url"] = f"/blind-candidates?run={run_dir.name}&scope=paired"
        if int(run["false_positive_count"] or 0) > 0:
            run["blind_false_positive_url"] = run["blind_candidates_url"]
    data = {
        "index_version": VIEWER_INDEX_VERSION,
        "run": run,
        "by_strength": _recovery_group_rows(recovery_dir / "recovery_by_strength.parquet", run_dir.name, campaign, target),
        "by_line": _recovery_group_rows(recovery_dir / "recovery_by_line.parquet", run_dir.name, campaign, target),
        "false_positives": _false_positive_rows(recovery_dir / "false_positive_candidates.parquet", run_dir.name, campaign, target, limit=10000),
    }
    out_path = index_root / "recovery_shards" / f"{run_dir.name}.json"
    _atomic_write_json(data, out_path)
    return out_path


def _candidate_sources_for_run(run_dir: Path) -> list[tuple[str, str]]:
    if run_dir.name.endswith("_baseline"):
        return [("baseline", "raw")]
    if run_dir.name.endswith("_injected"):
        return [("injected_raw", "raw"), ("paired_delta", "paired")]
    return [("baseline", "raw"), ("injected_raw", "raw"), ("paired_delta", "paired")]


def _annotate_candidate_frame(df: pd.DataFrame, runs_root: Path, run_dir: Path, path: Path, source: str, scope: str) -> pd.DataFrame:
    campaign, target = _campaign_and_target_from_run(run_dir.name, _grid_dispatch_campaigns(runs_root))
    out = df.copy()
    out["run_name"] = run_dir.name
    out["campaign"] = campaign
    out["target"] = target
    out["source"] = source
    out["scope"] = scope
    out["source_path"] = str(path)
    target_ids = out["target_id"].fillna("").astype(str) if "target_id" in out else pd.Series("", index=out.index)
    out["spectra_url"] = [
        f"/spectra?run={run_dir.name}&target={target_id}&target_id={target_id}&q={target_id}" for target_id in target_ids
    ]
    out["blind_candidates_url"] = [
        f"/blind-candidates?run={run_dir.name}&scope={scope}&tier=all&target={target_id}" for target_id in target_ids
    ]
    return out


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _atomic_write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(_clean_json(data), indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _maybe_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _format_time(epoch: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(epoch).astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


if __name__ == "__main__":
    main()
