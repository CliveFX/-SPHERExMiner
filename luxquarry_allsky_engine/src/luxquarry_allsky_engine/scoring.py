from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CandidateScoringConfig:
    spectra_path: Path
    output_dir: Path
    run_id: str
    device: str = "cuda:0"
    output_prefix: str = "baseline"
    flux_column: str = "aperture_flux_uJy"
    min_abs_zscore: float = 5.0
    min_measurements: int = 10
    only_ok: bool = True
    max_candidates: int | None = None


def score_spectra_candidates(config: CandidateScoringConfig) -> dict[str, Any]:
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    import cudf
    import cupy as cp

    cp.cuda.Device(_device_index(config.device)).use()

    t_read = time.perf_counter()
    spectra = cudf.read_parquet(str(config.spectra_path))
    read_wall = time.perf_counter() - t_read
    input_rows = int(len(spectra))
    if config.flux_column not in spectra.columns:
        raise ValueError(f"Flux column {config.flux_column!r} is missing from {config.spectra_path}")

    t_filter = time.perf_counter()
    filtered = spectra
    if config.only_ok:
        if "is_ok_measurement" in filtered.columns:
            filtered = filtered[filtered["is_ok_measurement"] == 1]
        elif "aperture_status_code" in filtered.columns:
            filtered = filtered[filtered["aperture_status_code"] == 0]
    filtered = filtered[filtered[config.flux_column].notnull()]
    filtered_rows = int(len(filtered))
    filter_wall = time.perf_counter() - t_filter

    t_score = time.perf_counter()
    candidates = _score_filtered_spectra(filtered, config)
    candidate_count_before_cap = int(len(candidates))
    if config.max_candidates is not None:
        candidates = candidates.head(config.max_candidates)
    candidate_count = int(len(candidates))
    score_wall = time.perf_counter() - t_score

    candidates_path = config.output_dir / f"{config.output_prefix}_candidates.parquet"
    t_write = time.perf_counter()
    candidates.to_parquet(candidates_path, index=False)
    write_wall = time.perf_counter() - t_write

    target_count = _safe_nunique(filtered, ["catalog", "target_id"])
    candidate_target_count = _safe_nunique(candidates, ["catalog", "target_id"])
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "cudf_simple_target_zscore_scorer",
        "run_id": config.run_id,
        "output_prefix": config.output_prefix,
        "device": config.device,
        "spectra_path": str(config.spectra_path),
        "output_dir": str(config.output_dir),
        "candidates_path": str(candidates_path),
        "flux_column": config.flux_column,
        "only_ok": config.only_ok,
        "min_abs_zscore": config.min_abs_zscore,
        "min_measurements": config.min_measurements,
        "max_candidates": config.max_candidates,
        "input_measurement_rows": input_rows,
        "filtered_measurement_rows": filtered_rows,
        "target_count": target_count,
        "candidate_count_before_cap": candidate_count_before_cap,
        "candidate_count": candidate_count,
        "candidate_target_count": candidate_target_count,
        "read_spectra_wall_sec": read_wall,
        "filter_wall_sec": filter_wall,
        "score_wall_sec": score_wall,
        "write_candidates_wall_sec": write_wall,
        "total_wall_sec": time.perf_counter() - started,
    }
    summary_path = config.output_dir / f"{config.output_prefix}_candidate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _score_filtered_spectra(spectra, config: CandidateScoringConfig):
    keys = ["catalog", "target_id"]
    flux = config.flux_column
    stats = spectra.groupby(keys).agg(
        {
            "source_id": "first",
            "ra_deg": "first",
            "dec_deg": "first",
            "mag_primary": "first",
            "mag_primary_band": "first",
            flux: ["count", "mean", "std"],
            "cwave_um": ["min", "max"],
        }
    ).reset_index()
    stats.columns = [_flatten_column_name(column) for column in stats.columns]
    rename = {
        "source_id_first": "target_source_id",
        "ra_deg_first": "target_ra_deg",
        "dec_deg_first": "target_dec_deg",
        "mag_primary_first": "target_mag_primary",
        "mag_primary_band_first": "target_mag_primary_band",
        f"{flux}_count": "target_measurement_count",
        f"{flux}_mean": "target_flux_mean_uJy",
        f"{flux}_std": "target_flux_std_uJy",
        "cwave_um_min": "target_cwave_min_um",
        "cwave_um_max": "target_cwave_max_um",
    }
    stats = stats.rename(columns=rename)
    stats["target_cwave_span_um"] = stats["target_cwave_max_um"] - stats["target_cwave_min_um"]
    stats = stats[
        (stats["target_measurement_count"] >= config.min_measurements)
        & stats["target_flux_std_uJy"].notnull()
        & (stats["target_flux_std_uJy"] > 0)
    ]

    candidates = spectra.merge(stats, on=keys, how="inner")
    candidates["zscore"] = (candidates[flux] - candidates["target_flux_mean_uJy"]) / candidates[
        "target_flux_std_uJy"
    ]
    candidates["abs_zscore"] = candidates["zscore"].abs()
    candidates = candidates[candidates["abs_zscore"] >= config.min_abs_zscore]
    candidates["score_method"] = "target_zscore"
    candidates["score_flux_column"] = flux
    candidates = candidates.sort_values(["abs_zscore", "catalog", "target_id"], ascending=[False, True, True])
    candidates = candidates.reset_index(drop=True)
    candidates["candidate_rank"] = candidates.index.astype("int64") + 1
    return candidates


def _safe_nunique(frame, keys: list[str]) -> int:
    if len(frame) == 0:
        return 0
    return int(len(frame[keys].drop_duplicates()))


def _flatten_column_name(column: Any) -> str:
    if isinstance(column, tuple):
        return "_".join(str(part) for part in column if part)
    return str(column)


def _device_index(device: str) -> int:
    if not device.startswith("cuda"):
        return 0
    _, _, suffix = device.partition(":")
    return int(suffix or "0")
