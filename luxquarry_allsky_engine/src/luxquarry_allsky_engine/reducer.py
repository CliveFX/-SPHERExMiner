from __future__ import annotations

import json
import shlex
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
