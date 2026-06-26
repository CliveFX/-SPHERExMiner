#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = Path("/mnt/niroseti/spherex_cache")
DEFAULT_CHECKPOINT = DEFAULT_CACHE_ROOT / "ml_runs" / "science_cv_mega_v2_train10" / "checkpoints" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build science embeddings and UMAP for a grid survey campaign prefix.")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--campaign-prefix", required=True)
    parser.add_argument("--embedding-name", help="Output name under ml_outputs/science_embeddings. Defaults to <campaign-prefix>_science_v2.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--python", type=Path, default=REPO_ROOT / "ml" / ".venv" / "bin" / "python")
    parser.add_argument("--run-kind", choices=["baseline", "injected", "all"], default="baseline")
    parser.add_argument("--quality-category", action="append", choices=["good", "review", "bad"], default=["good", "review"])
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--prep-workers", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-points", type=int, default=320)
    parser.add_argument("--umap-sample-size", type=int, default=30000)
    parser.add_argument("--umap-clusters", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    name = args.embedding_name or f"{args.campaign_prefix}_science_v2"
    dataset_dir = args.cache_root / "ml_datasets" / name
    output_dir = args.cache_root / "ml_outputs" / "science_embeddings" / name
    status_path = output_dir / "grid_embedding_status.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    if not args.checkpoint.exists():
        raise SystemExit(f"Science embedding checkpoint does not exist: {args.checkpoint}")
    py = str(args.python if args.python.exists() else sys.executable)

    _write_status(status_path, _status(args, name, "starting", started))
    run_kind_args = []
    if args.run_kind != "all":
        run_kind_args = ["--run-kind", args.run_kind]
    quality_args: list[str] = []
    for category in args.quality_category or []:
        quality_args.extend(["--quality-category", category])

    dataset_cmd = [
        py,
        str(REPO_ROOT / "ml" / "datasets" / "build_ml_datasets.py"),
        "--dataset-name",
        name,
        "--run-root",
        str(args.cache_root / "runs"),
        "--output-root",
        str(args.cache_root / "ml_datasets"),
        "--campaign",
        args.campaign_prefix,
        "--science-only",
        "--workers",
        str(args.workers),
        *run_kind_args,
        *quality_args,
    ]
    if args.max_targets is not None:
        dataset_cmd.extend(["--max-targets-per-run", str(args.max_targets)])
    _run_stage("dataset", dataset_cmd, status_path, args, name, started, force=args.force or not (dataset_dir / "science_targets.parquet").exists())

    export_cmd = [
        py,
        str(REPO_ROOT / "ml" / "science_embedding" / "export_embeddings_v2.py"),
        "--dataset-dir",
        str(dataset_dir),
        "--checkpoint",
        str(args.checkpoint),
        "--output-dir",
        str(output_dir),
        "--run-name",
        name,
        "--split",
        "all",
        "--batch-size",
        str(args.batch_size),
        "--max-points",
        str(args.max_points),
        "--prep-workers",
        str(args.prep_workers),
        "--device",
        args.device,
    ]
    _run_stage("export", export_cmd, status_path, args, name, started, force=args.force or not (output_dir / "embeddings.parquet").exists())

    umap_cmd = [
        py,
        str(REPO_ROOT / "ml" / "science_embedding" / "build_umap_projection.py"),
        "--embeddings",
        str(output_dir / "embeddings.parquet"),
        "--output-dir",
        str(output_dir),
        "--sample-size",
        str(args.umap_sample_size),
        "--clusters",
        str(args.umap_clusters),
    ]
    _run_stage("umap", umap_cmd, status_path, args, name, started, force=args.force or not (output_dir / "umap_projection.parquet").exists())

    summary = _status(args, name, "done", started)
    summary.update(
        {
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "embeddings": str(output_dir / "embeddings.parquet"),
            "umap_projection": str(output_dir / "umap_projection.parquet"),
        }
    )
    _write_status(status_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _run_stage(
    stage: str,
    cmd: list[str],
    status_path: Path,
    args: argparse.Namespace,
    name: str,
    started: float,
    *,
    force: bool,
) -> None:
    if not force:
        payload = _status(args, name, f"{stage}_skipped", started)
        payload.update({"stage": stage, "reason": "existing output"})
        _write_status(status_path, payload)
        return
    payload = _status(args, name, f"{stage}_running", started)
    payload.update({"stage": stage, "cmd": cmd})
    _write_status(status_path, payload)
    print(json.dumps({"status": "start", "stage": stage, "cmd": cmd}), flush=True)
    proc = subprocess.run([str(part) for part in cmd], cwd=REPO_ROOT, check=False)
    payload = _status(args, name, f"{stage}_done" if proc.returncode == 0 else f"{stage}_failed", started)
    payload.update({"stage": stage, "returncode": proc.returncode})
    _write_status(status_path, payload)
    print(json.dumps({"status": "done" if proc.returncode == 0 else "failed", "stage": stage, "returncode": proc.returncode}), flush=True)
    if proc.returncode != 0:
        raise SystemExit(f"{stage} failed with exit {proc.returncode}")


def _status(args: argparse.Namespace, name: str, status: str, started: float) -> dict[str, Any]:
    return {
        "status": status,
        "campaign_prefix": args.campaign_prefix,
        "embedding_name": name,
        "run_kind": args.run_kind,
        "checkpoint": str(args.checkpoint),
        "elapsed_sec": time.perf_counter() - started,
    }


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


if __name__ == "__main__":
    main()
