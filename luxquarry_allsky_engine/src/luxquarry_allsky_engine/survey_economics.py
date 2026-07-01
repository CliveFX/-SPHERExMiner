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
    require_all_2mass_input: bool = False


@dataclass(frozen=True)
class SurveyPlanConfig:
    manifest_path: Path
    projected_targets_path: Path
    output_dir: Path
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
    require_all_2mass_input: bool = False


@dataclass(frozen=True)
class SurveySampleExtrapolationConfig:
    plan_summary_paths: tuple[Path, ...]
    output_dir: Path
    target_cell_count: int
    sample_cell_count: int | None = None
    budget_usd: float = 5000.0


def plan_survey_economics(config: SurveyPlanConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(config.manifest_path)
    targets = pd.read_parquet(config.projected_targets_path)
    economics_config = SurveyEconomicsConfig(
        manifest_path=config.manifest_path,
        projected_targets_path=config.projected_targets_path,
        catalog_selection=config.catalog_selection,
        gaia_mag_min=config.gaia_mag_min,
        gaia_mag_max=config.gaia_mag_max,
        twomass_all_usable=config.twomass_all_usable,
        output_mode=config.output_mode,
        raw_retention_fraction=config.raw_retention_fraction,
        bytes_per_measurement=config.bytes_per_measurement,
        bytes_per_spectrum=config.bytes_per_spectrum,
        measurements_per_gpu_sec=config.measurements_per_gpu_sec,
        gpu_hourly_cost=config.gpu_hourly_cost,
        gpu_count=config.gpu_count,
        budget_usd=config.budget_usd,
        require_all_2mass_input=config.require_all_2mass_input,
    )
    selected = _select_targets(targets, economics_config)
    selected_in_frame = _in_frame_targets(selected)
    active_frame_ids = _active_frame_ids(selected_in_frame)
    planned_manifest = _filter_manifest_to_frames(manifest, active_frame_ids)
    unique_targets = _unique_target_rows(selected)

    frames_path = config.output_dir / "survey_plan_frames.parquet"
    targets_path = config.output_dir / "survey_plan_targets.parquet"
    unique_targets_path = config.output_dir / "survey_plan_unique_targets.parquet"
    economics_path = config.output_dir / "survey_economics_summary.json"
    summary_path = config.output_dir / "survey_plan_summary.json"
    planned_targets_summary_path = targets_path.with_suffix(".summary.json")

    planned_manifest.to_parquet(frames_path, index=False)
    selected.to_parquet(targets_path, index=False)
    unique_targets.to_parquet(unique_targets_path, index=False)
    source_projected_summary = _read_json_if_exists(config.projected_targets_path.with_suffix(".summary.json"))
    planned_targets_summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_projected_targets_path": str(config.projected_targets_path),
        "source_projected_targets_summary_path": str(config.projected_targets_path.with_suffix(".summary.json")),
        "source_projected_targets_summary": source_projected_summary,
        "planned_projected_target_rows": int(len(selected)),
        "planned_unique_targets": int(len(unique_targets)),
    }
    planned_targets_summary_path.write_text(
        json.dumps(planned_targets_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    economics = estimate_survey_economics(
        SurveyEconomicsConfig(
            manifest_path=frames_path,
            projected_targets_path=targets_path,
            output_path=economics_path,
            catalog_selection=config.catalog_selection,
            gaia_mag_min=config.gaia_mag_min,
            gaia_mag_max=config.gaia_mag_max,
            twomass_all_usable=config.twomass_all_usable,
            output_mode=config.output_mode,
            raw_retention_fraction=config.raw_retention_fraction,
            bytes_per_measurement=config.bytes_per_measurement,
            bytes_per_spectrum=config.bytes_per_spectrum,
            measurements_per_gpu_sec=config.measurements_per_gpu_sec,
            gpu_hourly_cost=config.gpu_hourly_cost,
            gpu_count=config.gpu_count,
            budget_usd=config.budget_usd,
            require_all_2mass_input=config.require_all_2mass_input,
        )
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "luxquarry_survey_economics_planner",
        "source_manifest_path": str(config.manifest_path),
        "source_projected_targets_path": str(config.projected_targets_path),
        "output_dir": str(config.output_dir),
        "catalog_selection": config.catalog_selection,
        "output_mode": config.output_mode,
        "source_frame_count": int(len(manifest)),
        "planned_frame_count": int(len(planned_manifest)),
        "source_projected_target_rows": int(len(targets)),
        "planned_projected_target_rows": int(len(selected)),
        "planned_in_frame_target_rows": int(len(selected_in_frame)),
        "planned_unique_targets": int(len(unique_targets)),
        "outputs": {
            "survey_plan_frames": str(frames_path),
            "survey_plan_targets": str(targets_path),
            "survey_plan_targets_summary": str(planned_targets_summary_path),
            "survey_plan_unique_targets": str(unique_targets_path),
            "survey_economics_summary": str(economics_path),
            "survey_plan_summary": str(summary_path),
        },
        "economics": economics,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def extrapolate_survey_sample(config: SurveySampleExtrapolationConfig) -> dict[str, Any]:
    if not config.plan_summary_paths:
        raise ValueError("at least one plan summary path is required")
    if config.target_cell_count <= 0:
        raise ValueError("target_cell_count must be positive")
    sample_count = config.sample_cell_count or len(config.plan_summary_paths)
    if sample_count <= 0:
        raise ValueError("sample_cell_count must be positive")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, path in enumerate(config.plan_summary_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        economics = payload.get("economics") or payload
        row = {
            "sample_index": index,
            "path": str(path),
            "planned_frame_count": int(payload.get("planned_frame_count") or economics.get("frame_count") or 0),
            "planned_projected_target_rows": int(payload.get("planned_projected_target_rows") or 0),
            "planned_in_frame_target_rows": int(payload.get("planned_in_frame_target_rows") or economics.get("measurement_count") or 0),
            "planned_unique_targets": int(payload.get("planned_unique_targets") or economics.get("target_count") or 0),
            "gaia_target_count": int(economics.get("gaia_target_count") or 0),
            "twomass_target_count": int(economics.get("twomass_target_count") or 0),
            "in_frame_gaia_target_count": int(economics.get("in_frame_gaia_target_count") or 0),
            "in_frame_twomass_target_count": int(economics.get("in_frame_twomass_target_count") or 0),
            "measurement_count": int(economics.get("measurement_count") or 0),
            "spectra_count": int(economics.get("spectra_count") or 0),
            "retained_raw_measurement_count": int(economics.get("retained_raw_measurement_count") or 0),
            "estimated_output_bytes": int(economics.get("estimated_output_bytes") or 0),
            "estimated_compute_cost_usd": float(economics.get("estimated_compute_cost_usd") or 0.0),
            "estimated_gpu_hours": float(economics.get("estimated_gpu_hours") or 0.0),
        }
        rows.append(row)

    sample_df = pd.DataFrame(rows)
    scale = float(config.target_cell_count) / float(sample_count)
    summed = sample_df.sum(numeric_only=True).to_dict()
    extrapolated = {
        "target_cell_count": int(config.target_cell_count),
        "sample_cell_count": int(sample_count),
        "plan_summary_count": len(config.plan_summary_paths),
        "scale_factor": scale,
        "frame_count": int(round(float(summed.get("planned_frame_count") or 0.0) * scale)),
        "projected_target_rows": int(round(float(summed.get("planned_projected_target_rows") or 0.0) * scale)),
        "in_frame_target_rows": int(round(float(summed.get("planned_in_frame_target_rows") or 0.0) * scale)),
        "unique_targets": int(round(float(summed.get("planned_unique_targets") or 0.0) * scale)),
        "gaia_target_count": int(round(float(summed.get("gaia_target_count") or 0.0) * scale)),
        "twomass_target_count": int(round(float(summed.get("twomass_target_count") or 0.0) * scale)),
        "in_frame_gaia_target_count": int(round(float(summed.get("in_frame_gaia_target_count") or 0.0) * scale)),
        "in_frame_twomass_target_count": int(round(float(summed.get("in_frame_twomass_target_count") or 0.0) * scale)),
        "measurement_count": int(round(float(summed.get("measurement_count") or 0.0) * scale)),
        "spectra_count": int(round(float(summed.get("spectra_count") or 0.0) * scale)),
        "retained_raw_measurement_count": int(
            round(float(summed.get("retained_raw_measurement_count") or 0.0) * scale)
        ),
        "estimated_output_bytes": int(round(float(summed.get("estimated_output_bytes") or 0.0) * scale)),
        "estimated_compute_cost_usd": float(summed.get("estimated_compute_cost_usd") or 0.0) * scale,
        "estimated_gpu_hours": float(summed.get("estimated_gpu_hours") or 0.0) * scale,
    }
    extrapolated["estimated_output_gib"] = extrapolated["estimated_output_bytes"] / float(1024**3)
    extrapolated["fits_budget"] = extrapolated["estimated_compute_cost_usd"] <= float(config.budget_usd)
    extrapolated["budget_usd"] = float(config.budget_usd)
    if extrapolated["measurement_count"] > 0:
        extrapolated["estimated_cost_per_billion_measurements"] = extrapolated["estimated_compute_cost_usd"] / (
            extrapolated["measurement_count"] / 1_000_000_000.0
        )
    else:
        extrapolated["estimated_cost_per_billion_measurements"] = None
    if extrapolated["spectra_count"] > 0:
        extrapolated["estimated_cost_per_million_spectra"] = extrapolated["estimated_compute_cost_usd"] / (
            extrapolated["spectra_count"] / 1_000_000.0
        )
    else:
        extrapolated["estimated_cost_per_million_spectra"] = None

    sample_path = config.output_dir / "survey_sample_cell_summaries.parquet"
    extrapolation_path = config.output_dir / "survey_sample_extrapolation.json"
    sample_df.to_parquet(sample_path, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "luxquarry_survey_sample_extrapolator",
        "plan_summary_paths": [str(path) for path in config.plan_summary_paths],
        "output_dir": str(config.output_dir),
        "outputs": {
            "survey_sample_cell_summaries": str(sample_path),
            "survey_sample_extrapolation": str(extrapolation_path),
        },
        "sample_totals": {str(key): _json_number(value) for key, value in summed.items()},
        "extrapolated": extrapolated,
        "caveats": [
            "This is an extrapolation from supplied plan summaries, not a full projected all-sky plan.",
            "Sampling quality depends on whether the supplied cells represent dense Galactic-plane and sparse high-latitude regions.",
            "Catalog cross-match/deduplication remains catalog-ID based unless upstream target planning has already deduplicated Gaia and 2MASS physically.",
        ],
    }
    extrapolation_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def estimate_survey_economics(config: SurveyEconomicsConfig) -> dict[str, Any]:
    if config.output_mode not in {"audit", "survey"}:
        raise ValueError("output_mode must be audit or survey")
    if config.catalog_selection not in {"gaia_g_8_14", "twomass_all_usable", "combined"}:
        raise ValueError("catalog_selection must be gaia_g_8_14, twomass_all_usable, or combined")
    if not 0.0 <= config.raw_retention_fraction <= 1.0:
        raise ValueError("raw_retention_fraction must be between 0 and 1")
    if config.gpu_count <= 0:
        raise ValueError("gpu_count must be positive")

    input_audit = _audit_all_2mass_input(config)
    if config.require_all_2mass_input and not input_audit["all_2mass_input_valid"]:
        raise ValueError(
            "Projected-target input is not proven valid for all-2MASS economics: "
            f"{input_audit['all_2mass_input_status']} - {input_audit['all_2mass_input_reason']}"
        )

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
        "all_2mass_input_valid": bool(input_audit["all_2mass_input_valid"]),
        "all_2mass_input_status": input_audit["all_2mass_input_status"],
        "all_2mass_input_reason": input_audit["all_2mass_input_reason"],
        "all_2mass_input_metadata": input_audit["all_2mass_input_metadata"],
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
            "For catalog_selection=combined, Gaia is magnitude-filtered and 2MASS is not; all 2MASS means all 2mass_psc rows present in the supplied projected-target parquet.",
            "If the upstream projected-target table was built with a per-frame source cap, this is a capped sample rather than an all-2MASS estimate.",
            f"All-2MASS input audit: {input_audit['all_2mass_input_status']} - {input_audit['all_2mass_input_reason']}",
            "2MASS and Gaia deduplication is not solved here; deduplicated_target_count is exact only if input target_id semantics are already deduplicated.",
        ],
    }
    if config.output_path is not None:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _audit_all_2mass_input(config: SurveyEconomicsConfig) -> dict[str, Any]:
    if config.catalog_selection not in {"combined", "twomass_all_usable"}:
        return {
            "all_2mass_input_valid": True,
            "all_2mass_input_status": "not_applicable",
            "all_2mass_input_reason": "catalog selection does not include 2MASS all-source economics",
            "all_2mass_input_metadata": {},
        }

    metadata = _projected_target_input_metadata(config.projected_targets_path)
    frame_targets = metadata.get("frame_targets_summary")
    if not frame_targets:
        return {
            "all_2mass_input_valid": False,
            "all_2mass_input_status": "unknown",
            "all_2mass_input_reason": "no frame-target summary metadata proving uncapped/no-filter 2MASS input",
            "all_2mass_input_metadata": metadata,
        }

    twomass_uncapped = frame_targets.get("twomass_uncapped")
    twomass_filtered = frame_targets.get("twomass_magnitude_filtered")
    twomass_limit = frame_targets.get("twomass_max_sources_per_frame")
    if twomass_uncapped is True and twomass_filtered is False:
        return {
            "all_2mass_input_valid": True,
            "all_2mass_input_status": "proven",
            "all_2mass_input_reason": "frame-target summary reports uncapped 2MASS with no mag_primary filter",
            "all_2mass_input_metadata": metadata,
        }

    reason_parts = []
    if twomass_uncapped is not True:
        reason_parts.append(f"twomass_uncapped={twomass_uncapped!r}")
    if twomass_filtered is not False:
        reason_parts.append(f"twomass_magnitude_filtered={twomass_filtered!r}")
    if twomass_limit is not None:
        reason_parts.append(f"twomass_max_sources_per_frame={twomass_limit!r}")
    reason = ", ".join(reason_parts) or "frame-target summary does not prove all-2MASS input"
    status = "invalid" if twomass_uncapped is False or twomass_filtered is True or twomass_limit is not None else "unknown"
    return {
        "all_2mass_input_valid": False,
        "all_2mass_input_status": status,
        "all_2mass_input_reason": reason,
        "all_2mass_input_metadata": metadata,
    }


def _projected_target_input_metadata(projected_targets_path: Path) -> dict[str, Any]:
    summary_path = projected_targets_path.with_suffix(".summary.json")
    summary = _read_json_if_exists(summary_path)
    metadata: dict[str, Any] = {
        "projected_targets_path": str(projected_targets_path),
        "projected_targets_summary_path": str(summary_path),
        "projected_targets_summary_present": summary is not None,
    }
    if not summary:
        return metadata

    metadata["projected_targets_summary"] = _compact_summary(summary)
    source_summary = summary.get("source_projected_targets_summary")
    if isinstance(source_summary, dict):
        metadata["source_projected_targets_summary"] = _compact_summary(source_summary)
        frame_targets_summary = _extract_frame_targets_summary(source_summary, projected_targets_path.parent)
    else:
        frame_targets_summary = _extract_frame_targets_summary(summary, projected_targets_path.parent)
    if frame_targets_summary:
        metadata["frame_targets_summary"] = _compact_summary(frame_targets_summary)
    return metadata


def _extract_frame_targets_summary(summary: dict[str, Any], base_dir: Path) -> dict[str, Any] | None:
    embedded = summary.get("frame_targets_summary")
    if isinstance(embedded, dict):
        return embedded

    summary_path_value = summary.get("frame_targets_summary_path")
    if summary_path_value:
        loaded = _read_json_if_exists(_resolve_metadata_path(base_dir, summary_path_value))
        if loaded:
            return loaded

    frame_targets_path_value = summary.get("frame_targets_path")
    if frame_targets_path_value:
        frame_targets_path = _resolve_metadata_path(base_dir, frame_targets_path_value)
        loaded = _read_json_if_exists(frame_targets_path.with_suffix(".summary.json"))
        if loaded:
            return loaded
    return None


def _resolve_metadata_path(base_dir: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute() or path.exists():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return path


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "created_utc",
        "catalog",
        "manifest_path",
        "frame_targets_path",
        "frame_targets_summary_path",
        "source_projected_targets_path",
        "source_projected_targets_summary_path",
        "output_path",
        "gaia_g_min",
        "gaia_g_max",
        "twomass_mag_min",
        "twomass_mag_max",
        "max_sources_per_frame",
        "gaia_max_sources_per_frame",
        "twomass_max_sources_per_frame",
        "twomass_uncapped",
        "twomass_magnitude_filtered",
        "frame_count",
        "target_row_count",
        "input_target_rows",
        "projected_target_rows",
        "planned_projected_target_rows",
        "unique_target_count",
        "planned_unique_targets",
    ]
    return {key: summary.get(key) for key in keep if key in summary}


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


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


def _in_frame_targets(targets: pd.DataFrame) -> pd.DataFrame:
    if "in_frame" not in targets.columns:
        return targets
    return targets[targets["in_frame"].astype(bool)].copy()


def _active_frame_ids(targets: pd.DataFrame) -> set[str]:
    if targets.empty or "frame_group_id" not in targets.columns:
        return set()
    return set(targets["frame_group_id"].astype(str).dropna().unique().tolist())


def _filter_manifest_to_frames(manifest: pd.DataFrame, frame_ids: set[str]) -> pd.DataFrame:
    if not frame_ids or "frame_group_id" not in manifest.columns:
        return manifest.iloc[0:0].copy()
    frame_id_series = manifest["frame_group_id"].astype(str)
    return manifest[frame_id_series.isin(frame_ids)].copy().reset_index(drop=True)


def _unique_target_rows(targets: pd.DataFrame) -> pd.DataFrame:
    if targets.empty:
        return targets.copy()
    preferred = [
        "catalog",
        "target_id",
        "source_id",
        "ra_deg",
        "dec_deg",
        "reference_epoch_yr",
        "pmra_masyr",
        "pmdec_masyr",
        "parallax_mas",
        "mag_primary",
        "mag_primary_band",
    ]
    columns = [column for column in preferred if column in targets.columns]
    if not columns:
        return targets[["catalog", "target_id"]].drop_duplicates().reset_index(drop=True)
    return targets[columns].drop_duplicates(subset=["catalog", "target_id"]).reset_index(drop=True)


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


def _json_number(value: Any) -> int | float:
    try:
        numeric = float(value)
    except Exception:
        return 0
    if numeric.is_integer():
        return int(numeric)
    return numeric
