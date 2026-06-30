from __future__ import annotations

import json
import time
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SpectraAssemblyConfig:
    shard_manifest_path: Path
    output_dir: Path
    run_id: str
    device: str = "cuda:0"
    only_ok: bool = False
    drop_duplicate_measurements: bool = False


@dataclass(frozen=True)
class SpectraAssemblyValidationConfig:
    shard_manifest_path: Path
    output_dir: Path
    run_id: str
    device: str = "cuda:0"
    only_ok: bool = False
    drop_duplicate_measurements: bool = False
    repetitions: int = 2
    random_seed: int = 1729


@dataclass(frozen=True)
class SpectraRetryDedupValidationConfig:
    shard_manifest_path: Path
    output_dir: Path
    run_id: str
    device: str = "cuda:0"
    only_ok: bool = False
    duplicate_shard_count: int | None = None


@dataclass(frozen=True)
class PartitionedSpectraAssemblyConfig:
    shard_manifest_path: Path
    output_dir: Path
    run_id: str
    device: str = "cuda:0"
    partition_count: int = 64
    partition_index: int | None = None
    only_ok: bool = False
    drop_duplicate_measurements: bool = False


DEDUPE_KEY_CANDIDATES = (
    "catalog",
    "target_id",
    "frame_group_id",
    "image_id",
    "detector",
)


def assemble_spectra_from_shards(config: SpectraAssemblyConfig) -> dict[str, Any]:
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    shard_manifest = pd.read_parquet(config.shard_manifest_path)
    shard_paths = _resolve_shard_paths(config.shard_manifest_path, shard_manifest["path"].tolist())
    if not shard_paths:
        raise ValueError(f"No measurement shards found in {config.shard_manifest_path}")

    import cudf
    import cupy as cp

    cp.cuda.Device(_device_index(config.device)).use()
    t_read = time.perf_counter()
    measurements = cudf.read_parquet([str(path) for path in shard_paths])
    read_wall = time.perf_counter() - t_read

    input_rows = int(len(measurements))
    if config.only_ok:
        measurements = measurements[measurements["aperture_status_code"] == 0]
    filtered_rows = int(len(measurements))
    dedupe_wall = 0.0
    duplicate_measurement_rows_dropped = 0
    dedupe_key = _measurement_dedupe_key(measurements)
    if config.drop_duplicate_measurements:
        t_dedupe = time.perf_counter()
        before_dedupe = int(len(measurements))
        measurements = measurements.drop_duplicates(subset=dedupe_key)
        duplicate_measurement_rows_dropped = before_dedupe - int(len(measurements))
        dedupe_wall = time.perf_counter() - t_dedupe
    output_rows = int(len(measurements))

    t_sort = time.perf_counter()
    sort_cols = ["catalog", "target_id", "cwave_um", "frame_group_id", "image_id"]
    spectra = measurements.sort_values(sort_cols, ignore_index=True)
    spectra["is_ok_measurement"] = (spectra["aperture_status_code"] == 0).astype("int32")
    sort_wall = time.perf_counter() - t_sort

    spectra_path = config.output_dir / f"{config.run_id}.spectra_measurements.parquet"
    t_write_spectra = time.perf_counter()
    spectra.to_parquet(spectra_path, index=False)
    write_spectra_wall = time.perf_counter() - t_write_spectra

    t_summary = time.perf_counter()
    target_summary = _build_target_summary(spectra)
    target_summary_path = config.output_dir / f"{config.run_id}.target_summary.parquet"
    target_summary.to_parquet(target_summary_path, index=False)
    summary_wall = time.perf_counter() - t_summary

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config.run_id,
        "backend": "cudf_spectra_assembly",
        "device": config.device,
        "shard_manifest_path": str(config.shard_manifest_path),
        "output_dir": str(config.output_dir),
        "spectra_measurements_path": str(spectra_path),
        "target_summary_path": str(target_summary_path),
        "only_ok": config.only_ok,
        "drop_duplicate_measurements": config.drop_duplicate_measurements,
        "measurement_dedupe_key": dedupe_key,
        "shard_count": len(shard_paths),
        "input_measurement_rows": input_rows,
        "filtered_measurement_rows": filtered_rows,
        "duplicate_measurement_rows_dropped": duplicate_measurement_rows_dropped,
        "spectra_measurement_rows": output_rows,
        "target_count": int(len(target_summary)),
        "read_shards_wall_sec": read_wall,
        "dedupe_wall_sec": dedupe_wall,
        "sort_wall_sec": sort_wall,
        "write_spectra_wall_sec": write_spectra_wall,
        "target_summary_wall_sec": summary_wall,
        "total_wall_sec": time.perf_counter() - started,
    }
    summary_path = config.output_dir / "assemble_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def assemble_spectra_partitions(config: PartitionedSpectraAssemblyConfig) -> dict[str, Any]:
    if config.partition_count <= 0:
        raise ValueError("partition_count must be positive")
    if config.partition_index is not None and (
        config.partition_index < 0 or config.partition_index >= config.partition_count
    ):
        raise ValueError("partition_index must be in [0, partition_count)")

    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    shard_manifest = pd.read_parquet(config.shard_manifest_path)
    shard_paths = _resolve_shard_paths(config.shard_manifest_path, shard_manifest["path"].tolist())
    if not shard_paths:
        raise ValueError(f"No measurement shards found in {config.shard_manifest_path}")

    import cudf
    import cupy as cp

    cp.cuda.Device(_device_index(config.device)).use()
    t_read = time.perf_counter()
    measurements = cudf.read_parquet([str(path) for path in shard_paths])
    read_wall = time.perf_counter() - t_read

    input_rows = int(len(measurements))
    if config.only_ok:
        measurements = measurements[measurements["aperture_status_code"] == 0]
    filtered_rows = int(len(measurements))

    dedupe_wall = 0.0
    duplicate_measurement_rows_dropped = 0
    dedupe_key = _measurement_dedupe_key(measurements)
    if config.drop_duplicate_measurements:
        t_dedupe = time.perf_counter()
        before_dedupe = int(len(measurements))
        measurements = measurements.drop_duplicates(subset=dedupe_key)
        duplicate_measurement_rows_dropped = before_dedupe - int(len(measurements))
        dedupe_wall = time.perf_counter() - t_dedupe

    t_bucket = time.perf_counter()
    measurements["target_partition"] = (
        measurements[["catalog", "target_id"]].hash_values() % config.partition_count
    ).astype("int32")
    bucket_wall = time.perf_counter() - t_bucket

    partition_indices = (
        [int(config.partition_index)]
        if config.partition_index is not None
        else list(range(config.partition_count))
    )
    partition_rows: list[dict[str, Any]] = []
    for partition_index in partition_indices:
        t_partition = time.perf_counter()
        partition = measurements[measurements["target_partition"] == partition_index]
        partition_input_rows = int(len(partition))

        t_sort = time.perf_counter()
        sort_cols = ["catalog", "target_id", "cwave_um", "frame_group_id", "image_id"]
        spectra = partition.sort_values(sort_cols, ignore_index=True)
        spectra["is_ok_measurement"] = (spectra["aperture_status_code"] == 0).astype("int32")
        sort_wall = time.perf_counter() - t_sort

        spectra_path = config.output_dir / f"{config.run_id}.part{partition_index:05d}.spectra_measurements.parquet"
        t_write_spectra = time.perf_counter()
        spectra.to_parquet(spectra_path, index=False)
        write_spectra_wall = time.perf_counter() - t_write_spectra

        t_summary = time.perf_counter()
        target_summary = _build_target_summary(spectra)
        target_summary_path = config.output_dir / f"{config.run_id}.part{partition_index:05d}.target_summary.parquet"
        target_summary.to_parquet(target_summary_path, index=False)
        summary_wall = time.perf_counter() - t_summary

        partition_rows.append(
            {
                "partition_index": partition_index,
                "partition_count": config.partition_count,
                "spectra_measurements_path": str(spectra_path),
                "target_summary_path": str(target_summary_path),
                "spectra_measurement_rows": partition_input_rows,
                "target_count": int(len(target_summary)),
                "sort_wall_sec": sort_wall,
                "write_spectra_wall_sec": write_spectra_wall,
                "target_summary_wall_sec": summary_wall,
                "total_partition_wall_sec": time.perf_counter() - t_partition,
            }
        )

    partition_manifest_path = config.output_dir / (
        f"{config.run_id}.partition_manifest.part{config.partition_index:05d}.parquet"
        if config.partition_index is not None
        else f"{config.run_id}.partition_manifest.parquet"
    )
    pd.DataFrame(partition_rows).to_parquet(partition_manifest_path, index=False)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config.run_id,
        "backend": "cudf_partitioned_spectra_assembly",
        "device": config.device,
        "shard_manifest_path": str(config.shard_manifest_path),
        "output_dir": str(config.output_dir),
        "partition_count": config.partition_count,
        "partition_index": config.partition_index,
        "partition_manifest_path": str(partition_manifest_path),
        "only_ok": config.only_ok,
        "drop_duplicate_measurements": config.drop_duplicate_measurements,
        "measurement_dedupe_key": dedupe_key,
        "shard_count": len(shard_paths),
        "input_measurement_rows": input_rows,
        "filtered_measurement_rows": filtered_rows,
        "duplicate_measurement_rows_dropped": duplicate_measurement_rows_dropped,
        "spectra_measurement_rows": sum(int(row["spectra_measurement_rows"]) for row in partition_rows),
        "target_count": sum(int(row["target_count"]) for row in partition_rows),
        "partition_rows": len(partition_rows),
        "read_shards_wall_sec": read_wall,
        "dedupe_wall_sec": dedupe_wall,
        "bucket_wall_sec": bucket_wall,
        "write_spectra_wall_sec": sum(float(row["write_spectra_wall_sec"]) for row in partition_rows),
        "target_summary_wall_sec": sum(float(row["target_summary_wall_sec"]) for row in partition_rows),
        "total_wall_sec": time.perf_counter() - started,
        "partitions": partition_rows,
    }
    summary_path = config.output_dir / (
        f"partitioned_assemble_summary.part{config.partition_index:05d}.json"
        if config.partition_index is not None
        else "partitioned_assemble_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def validate_shard_order_independent_assembly(config: SpectraAssemblyValidationConfig) -> dict[str, Any]:
    if config.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(config.shard_manifest_path)
    if len(manifest) == 0:
        raise ValueError(f"No shard rows found in {config.shard_manifest_path}")

    runs: list[dict[str, Any]] = []
    base_dir = config.output_dir
    for idx in range(config.repetitions + 1):
        label = "original" if idx == 0 else f"shuffle_{idx:02d}"
        run_dir = base_dir / label
        run_dir.mkdir(parents=True, exist_ok=True)
        run_manifest = manifest if idx == 0 else manifest.sample(frac=1.0, random_state=config.random_seed + idx)
        run_manifest_path = run_dir / "measurement_shard_manifest.parquet"
        run_manifest.to_parquet(run_manifest_path, index=False)
        assembly = assemble_spectra_from_shards(
            SpectraAssemblyConfig(
                shard_manifest_path=run_manifest_path,
                output_dir=run_dir,
                run_id=f"{config.run_id}.{label}",
                device=config.device,
                only_ok=config.only_ok,
                drop_duplicate_measurements=config.drop_duplicate_measurements,
            )
        )
        spectra_hash = _parquet_logical_hash(Path(assembly["spectra_measurements_path"]))
        target_summary_hash = _parquet_logical_hash(Path(assembly["target_summary_path"]))
        runs.append(
            {
                "label": label,
                "manifest_path": str(run_manifest_path),
                "output_dir": str(run_dir),
                "spectra_measurements_path": assembly["spectra_measurements_path"],
                "target_summary_path": assembly["target_summary_path"],
                "spectra_hash": spectra_hash,
                "target_summary_hash": target_summary_hash,
                "input_measurement_rows": int(assembly["input_measurement_rows"]),
                "spectra_measurement_rows": int(assembly["spectra_measurement_rows"]),
                "target_count": int(assembly["target_count"]),
                "shard_count": int(assembly["shard_count"]),
                "total_wall_sec": float(assembly["total_wall_sec"]),
            }
        )

    baseline = runs[0]
    mismatches = [
        run
        for run in runs[1:]
        if run["spectra_hash"] != baseline["spectra_hash"]
        or run["target_summary_hash"] != baseline["target_summary_hash"]
        or run["spectra_measurement_rows"] != baseline["spectra_measurement_rows"]
        or run["target_count"] != baseline["target_count"]
    ]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config.run_id,
        "backend": "cudf_spectra_assembly_order_validation",
        "device": config.device,
        "shard_manifest_path": str(config.shard_manifest_path),
        "output_dir": str(config.output_dir),
        "only_ok": config.only_ok,
        "drop_duplicate_measurements": config.drop_duplicate_measurements,
        "repetitions": config.repetitions,
        "random_seed": config.random_seed,
        "passed": len(mismatches) == 0,
        "mismatch_count": len(mismatches),
        "shard_count": int(len(manifest)),
        "baseline_spectra_hash": baseline["spectra_hash"],
        "baseline_target_summary_hash": baseline["target_summary_hash"],
        "runs": runs,
        "total_wall_sec": time.perf_counter() - started,
    }
    summary_path = config.output_dir / "assembly_order_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if mismatches:
        raise ValueError(f"Assembly order validation failed for {len(mismatches)} shuffled manifests")
    return summary


def validate_retry_dedup_assembly(config: SpectraRetryDedupValidationConfig) -> dict[str, Any]:
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(config.shard_manifest_path)
    if len(manifest) == 0:
        raise ValueError(f"No shard rows found in {config.shard_manifest_path}")
    duplicate_count = config.duplicate_shard_count if config.duplicate_shard_count is not None else len(manifest)
    if duplicate_count <= 0:
        raise ValueError("duplicate_shard_count must be positive")
    duplicate_count = min(int(duplicate_count), int(len(manifest)))

    baseline_dir = config.output_dir / "baseline"
    duplicate_dir = config.output_dir / "duplicated_retry"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    duplicate_dir.mkdir(parents=True, exist_ok=True)

    baseline_manifest_path = baseline_dir / "measurement_shard_manifest.parquet"
    manifest.to_parquet(baseline_manifest_path, index=False)
    baseline = assemble_spectra_from_shards(
        SpectraAssemblyConfig(
            shard_manifest_path=baseline_manifest_path,
            output_dir=baseline_dir,
            run_id=f"{config.run_id}.baseline",
            device=config.device,
            only_ok=config.only_ok,
            drop_duplicate_measurements=True,
        )
    )

    duplicated_manifest = pd.concat([manifest, manifest.head(duplicate_count)], ignore_index=True)
    duplicated_manifest_path = duplicate_dir / "measurement_shard_manifest.parquet"
    duplicated_manifest.to_parquet(duplicated_manifest_path, index=False)
    duplicated = assemble_spectra_from_shards(
        SpectraAssemblyConfig(
            shard_manifest_path=duplicated_manifest_path,
            output_dir=duplicate_dir,
            run_id=f"{config.run_id}.duplicated_retry",
            device=config.device,
            only_ok=config.only_ok,
            drop_duplicate_measurements=True,
        )
    )

    baseline_spectra_hash = _parquet_logical_hash(Path(baseline["spectra_measurements_path"]))
    duplicated_spectra_hash = _parquet_logical_hash(Path(duplicated["spectra_measurements_path"]))
    baseline_target_hash = _parquet_logical_hash(Path(baseline["target_summary_path"]))
    duplicated_target_hash = _parquet_logical_hash(Path(duplicated["target_summary_path"]))
    passed = (
        baseline_spectra_hash == duplicated_spectra_hash
        and baseline_target_hash == duplicated_target_hash
        and int(baseline["spectra_measurement_rows"]) == int(duplicated["spectra_measurement_rows"])
        and int(baseline["target_count"]) == int(duplicated["target_count"])
        and int(duplicated["duplicate_measurement_rows_dropped"]) > 0
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config.run_id,
        "backend": "cudf_spectra_retry_dedup_validation",
        "device": config.device,
        "shard_manifest_path": str(config.shard_manifest_path),
        "output_dir": str(config.output_dir),
        "only_ok": config.only_ok,
        "passed": passed,
        "source_shard_count": int(len(manifest)),
        "duplicate_shard_count": duplicate_count,
        "duplicated_manifest_shard_count": int(len(duplicated_manifest)),
        "baseline_spectra_hash": baseline_spectra_hash,
        "duplicated_spectra_hash": duplicated_spectra_hash,
        "baseline_target_summary_hash": baseline_target_hash,
        "duplicated_target_summary_hash": duplicated_target_hash,
        "baseline": baseline,
        "duplicated_retry": duplicated,
        "total_wall_sec": time.perf_counter() - started,
    }
    summary_path = config.output_dir / "retry_dedup_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise ValueError("Retry duplicate assembly validation failed")
    return summary


def _build_target_summary(spectra):
    keys = ["catalog", "target_id"]
    out = spectra.groupby(keys).agg(
        {
            "source_id": "first",
            "ra_deg": "first",
            "dec_deg": "first",
            "mag_primary": "first",
            "mag_primary_band": "first",
            "aperture_flux_uJy": ["count", "mean", "std"],
            "cwave_um": ["min", "max"],
            "is_ok_measurement": "sum",
            "flags_summary": "max",
        }
    ).reset_index()
    out.columns = [_flatten_column_name(column) for column in out.columns]
    out = out.rename(
        columns={
            "source_id_first": "source_id",
            "ra_deg_first": "ra_deg",
            "dec_deg_first": "dec_deg",
            "mag_primary_first": "mag_primary",
            "mag_primary_band_first": "mag_primary_band",
            "aperture_flux_uJy_count": "measurement_count",
            "aperture_flux_uJy_mean": "aperture_flux_mean_uJy",
            "aperture_flux_uJy_std": "aperture_flux_std_uJy",
            "cwave_um_min": "cwave_min_um",
            "cwave_um_max": "cwave_max_um",
            "is_ok_measurement_sum": "ok_measurement_count",
            "flags_summary_max": "flags_summary_or",
        }
    )
    out["cwave_span_um"] = out["cwave_max_um"] - out["cwave_min_um"]
    out["ok_fraction"] = out["ok_measurement_count"] / out["measurement_count"]
    return out.sort_values(["catalog", "target_id"], ignore_index=True)


def _flatten_column_name(column: Any) -> str:
    if isinstance(column, tuple):
        return "_".join(str(part) for part in column if part)
    return str(column)


def _resolve_shard_paths(manifest_path: Path, raw_paths: list[str]) -> list[Path]:
    roots = [Path.cwd(), *manifest_path.parents]
    paths: list[Path] = []
    for raw in raw_paths:
        path = Path(str(raw))
        candidates = [path] if path.is_absolute() else [root / path for root in roots]
        resolved = next((candidate for candidate in candidates if candidate.exists()), None)
        if resolved is None:
            raise FileNotFoundError(f"Missing measurement shard from manifest: {raw}")
        paths.append(resolved)
    return paths


def _measurement_dedupe_key(measurements: Any) -> list[str]:
    available = set(measurements.columns)
    keys = [column for column in DEDUPE_KEY_CANDIDATES if column in available]
    missing = [column for column in DEDUPE_KEY_CANDIDATES if column not in available]
    if missing:
        raise ValueError(f"Measurement table missing dedupe key columns: {missing}")
    return keys


def _parquet_logical_hash(path: Path) -> str:
    frame = pd.read_parquet(path)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns), separators=(",", ":")).encode("utf-8"))
    digest.update(str(len(frame)).encode("utf-8"))
    if len(frame) > 0:
        row_hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="uint64", copy=False)
        digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _device_index(device: str) -> int:
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    return 0
