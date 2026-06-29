from __future__ import annotations

import json
import shlex
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


def build_dispatch_plan(config: DispatchPlanConfig) -> dict[str, Any]:
    if not config.devices:
        raise ValueError("At least one GPU device is required")
    if config.workers_per_device <= 0:
        raise ValueError("workers_per_device must be positive")
    total_workers = len(config.devices) * config.workers_per_device
    workers = []
    worker_index = 0
    for device in config.devices:
        for local_slot in range(config.workers_per_device):
            worker_id = f"{config.run_id}.w{worker_index:04d}.{device.replace(':', '')}.s{local_slot}"
            worker_out = config.output_dir / "workers" / worker_id
            status_path = worker_out / "run_status.json"
            argv = [
                config.executable,
                "run-persistent-gpu-worker",
                "--manifest",
                str(config.manifest_path),
                "--projected-targets",
                str(config.projected_targets_path),
                "--out-dir",
                str(worker_out),
                "--run-id",
                worker_id,
                "--cache-root",
                str(config.cache_root),
                "--device",
                device,
                "--worker-index",
                str(worker_index),
                "--worker-count",
                str(total_workers),
                "--status-path",
                str(status_path),
            ]
            if config.limit_frames is not None:
                argv.extend(["--limit-frames", str(config.limit_frames)])
            workers.append(
                {
                    "worker_id": worker_id,
                    "worker_index": worker_index,
                    "worker_count": total_workers,
                    "device": device,
                    "local_slot": local_slot,
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
        "contract": {
            "partitioning": "frame ordinal modulo worker_count equals worker_index",
            "output": "each worker writes independent measurement_shards and run_summary.json",
            "status": "each worker atomically rewrites run_status.json",
            "coordination": "no live database or shared lock required in the hot path",
        },
        "workers": workers,
    }


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
