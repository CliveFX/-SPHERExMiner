from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SurveyEconomicsConfig:
    manifest_path: Path
    projected_targets_path: Path
    output_path: Path | None = None
    catalog_selection: str = "combined"
    gaia_mag_min: float = 8.0
    gaia_mag_max: float = 14.0
    twomass_all_usable: bool = True
    output_mode: str = "survey"
    raw_retention_fraction: float = 0.01
    bytes_per_measurement: float | None = None
    bytes_per_spectrum: float = 4096.0
    measurements_per_gpu_sec: float | None = None
    gpu_hourly_cost: float = 6.88
    gpu_count: int = 8
    budget_usd: float = 5000.0


def estimate_survey_economics(config: SurveyEconomicsConfig) -> dict[str, Any]:
    if config.output_mode not in {"audit", "survey"}:
        raise ValueError("output_mode must be audit or survey")
    if config.catalog_selection not in {"gaia_g_8_14", "twomass_all_usable", "combined"}:
        raise ValueError("catalog_selection must be gaia_g_8_14, twomass_all_usable, or combined")
    if not 0.0 <= config.raw_retention_fraction <= 1.0:
        raise ValueError("raw_retention_fraction must be between 0 and 1")
    if config.gpu_count <= 0:
        raise ValueError("gpu_count must be positive")

    manifest = pd.read_parquet(config.manifest_path)
    targets = pd.read_parquet(config.projected_targets_path)
    if "catalog" not in targets.columns:
        raise ValueError("projected targets must contain a catalog column")
    if "target_id" not in targets.columns:
        raise ValueError("projected targets must contain a target_id column")

    selected = _select_targets(targets, config)
    in_frame = selected
    if "in_frame" in in_frame.columns:
        in_frame = in_frame[in_frame["in_frame"].astype(bool)]

    catalog_counts = _catalog_target_counts(selected)
    in_frame_catalog_counts = _catalog_target_counts(in_frame)
    target_count = _unique_targets(selected)
    in_frame_target_count = _unique_targets(in_frame)
    measurement_count = int(len(in_frame))
    frame_count = int(manifest["frame_group_id"].nunique()) if "frame_group_id" in manifest.columns else int(len(manifest))
    spectra_count = int(in_frame_target_count)

    bytes_per_measurement = _resolve_bytes_per_measurement(config)
    raw_rows_retained = measurement_count if config.output_mode == "audit" else int(
        round(measurement_count * config.raw_retention_fraction)
    )
    retained_raw_bytes = int(round(raw_rows_retained * bytes_per_measurement))
    spectra_bytes = int(round(spectra_count * config.bytes_per_spectrum))
    estimated_output_bytes = int(retained_raw_bytes + spectra_bytes)

    gpu_seconds = None
    gpu_hours = None
    compute_cost = None
    if config.measurements_per_gpu_sec and config.measurements_per_gpu_sec > 0:
        gpu_seconds = measurement_count / float(config.measurements_per_gpu_sec)
        gpu_hours = gpu_seconds / 3600.0
        compute_cost = gpu_hours * float(config.gpu_hourly_cost)

    cluster_wall_hours = None
    if gpu_hours is not None:
        cluster_wall_hours = gpu_hours / float(config.gpu_count)

    cost_per_billion = None
    if compute_cost is not None and measurement_count > 0:
        cost_per_billion = compute_cost / (measurement_count / 1_000_000_000.0)
    cost_per_million_spectra = None
    if compute_cost is not None and spectra_count > 0:
        cost_per_million_spectra = compute_cost / (spectra_count / 1_000_000.0)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "luxquarry_survey_economics_estimator",
        "manifest_path": str(config.manifest_path),
        "projected_targets_path": str(config.projected_targets_path),
        "output_mode": config.output_mode,
        "catalog_selection": config.catalog_selection,
        "gaia_mag_min": float(config.gaia_mag_min),
        "gaia_mag_max": float(config.gaia_mag_max),
        "twomass_all_usable": bool(config.twomass_all_usable),
        "frame_count": frame_count,
        "target_count": int(target_count),
        "in_frame_target_count": int(in_frame_target_count),
        "gaia_target_count": int(catalog_counts.get("gaia_dr3", 0)),
        "twomass_target_count": int(catalog_counts.get("2mass_psc", 0)),
        "deduplicated_target_count": int(target_count),
        "in_frame_gaia_target_count": int(in_frame_catalog_counts.get("gaia_dr3", 0)),
        "in_frame_twomass_target_count": int(in_frame_catalog_counts.get("2mass_psc", 0)),
        "measurement_count": measurement_count,
        "estimated_measurement_count": measurement_count,
        "spectra_count": spectra_count,
        "retained_raw_measurement_count": int(raw_rows_retained),
        "raw_retention_fraction": float(config.raw_retention_fraction if config.output_mode == "survey" else 1.0),
        "bytes_per_measurement": float(bytes_per_measurement),
        "bytes_per_spectrum": float(config.bytes_per_spectrum),
        "estimated_retained_raw_bytes": retained_raw_bytes,
        "estimated_spectra_bytes": spectra_bytes,
        "estimated_output_bytes": estimated_output_bytes,
        "estimated_output_gib": estimated_output_bytes / float(1024**3),
        "measurements_per_gpu_sec": config.measurements_per_gpu_sec,
        "gpu_hourly_cost": float(config.gpu_hourly_cost),
        "gpu_count": int(config.gpu_count),
        "budget_usd": float(config.budget_usd),
        "estimated_gpu_seconds": gpu_seconds,
        "estimated_gpu_hours": gpu_hours,
        "estimated_cluster_wall_hours": cluster_wall_hours,
        "estimated_compute_cost_usd": compute_cost,
        "estimated_cost_per_billion_measurements": cost_per_billion,
        "estimated_cost_per_million_spectra": cost_per_million_spectra,
        "fits_budget": bool(compute_cost is not None and compute_cost <= config.budget_usd),
        "notes": [
            "Counts are derived from the supplied manifest/projected-target parquet files.",
            "Use a full accessible-sky projected target table before treating this as a cloud estimate.",
            "2MASS and Gaia deduplication is not solved here; deduplicated_target_count is exact only if input target_id semantics are already deduplicated.",
        ],
    }
    if config.output_path is not None:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _select_targets(targets: pd.DataFrame, config: SurveyEconomicsConfig) -> pd.DataFrame:
    catalog = targets["catalog"].astype(str)
    pieces = []
    if config.catalog_selection in {"gaia_g_8_14", "combined"}:
        if "mag_primary" not in targets.columns:
            raise ValueError("Gaia magnitude selection requires mag_primary")
        mag = pd.to_numeric(targets["mag_primary"], errors="coerce")
        pieces.append(
            targets[
                catalog.eq("gaia_dr3")
                & mag.between(config.gaia_mag_min, config.gaia_mag_max, inclusive="both")
            ]
        )
    if config.catalog_selection in {"twomass_all_usable", "combined"} and config.twomass_all_usable:
        pieces.append(targets[catalog.eq("2mass_psc")])
    if not pieces:
        return targets.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def _unique_targets(targets: pd.DataFrame) -> int:
    if targets.empty:
        return 0
    return int(targets[["catalog", "target_id"]].drop_duplicates().shape[0])


def _catalog_target_counts(targets: pd.DataFrame) -> dict[str, int]:
    if targets.empty:
        return {}
    grouped = targets[["catalog", "target_id"]].drop_duplicates().groupby("catalog", dropna=False).size()
    return {str(key): int(value) for key, value in grouped.items()}


def _resolve_bytes_per_measurement(config: SurveyEconomicsConfig) -> float:
    if config.bytes_per_measurement is not None:
        if config.bytes_per_measurement <= 0:
            raise ValueError("bytes_per_measurement must be positive")
        return float(config.bytes_per_measurement)
    return 158.0
