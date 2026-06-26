from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from spherex_laser_miner.catalog.local_gaia_lite import GAIA_LITE_COLUMNS, query_local_gaia_lite_duckdb

from .healpix_tiles import HealpixTile


def query_tile_gaia(
    tile: HealpixTile,
    *,
    cache_root: Path,
    g_min: float,
    g_max: float,
    max_sources: int,
) -> pd.DataFrame:
    df = query_local_gaia_lite_duckdb(
        tile.s_region,
        cache_root=cache_root,
        max_sources=max_sources,
        g_min=g_min,
        g_max=g_max,
    )
    if df.empty:
        return df
    out = df.copy()
    out.insert(0, "survey_tile_id", tile.tile_id)
    out.insert(1, "survey_hpx", int(tile.hpx))
    out.insert(2, "survey_nside", int(tile.nside))
    return out


def gaia_to_manual_targets(df: pd.DataFrame, tile: HealpixTile) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(df.to_dict(orient="records"), start=1):
        source_id = str(row["source_id"])
        g_mag = _float_or_none(row.get("phot_g_mean_mag"))
        rows.append(
            {
                "target_id": f"grid_{tile.tile_id}_gaia_dr3_{source_id}",
                "target_type": "gaia_dr3_grid_survey",
                "object_name": f"Gaia DR3 {source_id} in {tile.tile_id}",
                "ra_deg": float(row["ra"]),
                "dec_deg": float(row["dec"]),
                "reference_epoch_yr": float(row.get("ref_epoch") or 2016.0),
                "pmra_masyr": _float_or_none(row.get("pmra")),
                "pmdec_masyr": _float_or_none(row.get("pmdec")),
                "parallax_mas": _float_or_none(row.get("parallax")),
                "source_catalog": "gaia_dr3",
                "source_catalog_id": source_id,
                "priority_score": float(100.0 - min(g_mag if g_mag is not None else 99.0, 99.0)),
                "notes": f"grid_survey_v1 tile={tile.tile_id} rank={rank} G={g_mag}",
            }
        )
    return rows


def write_tile_outputs(
    *,
    tile: HealpixTile,
    gaia: pd.DataFrame,
    output_dir: Path,
    cache_root: Path,
    g_min: float,
    g_max: float,
    max_sources: int,
    batch_size: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = gaia_to_manual_targets(gaia, tile)
    target_doc = {
        "name": tile.tile_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "survey_mode": "grid_survey_v1_healpix_gaia",
        "healpix": {
            "nside": tile.nside,
            "hpx": tile.hpx,
            "order": tile.order,
            "vertices_icrs_deg": tile.vertices,
            "s_region": tile.s_region,
        },
        "gaia_query": {
            "cache_root": str(cache_root),
            "g_min": g_min,
            "g_max": g_max,
            "max_sources": max_sources,
            "row_count": int(len(gaia)),
            "metrics": _jsonable(gaia.attrs.get("local_gaia_metrics", {})),
        },
        "targets": targets,
    }
    targets_yaml = output_dir / "targets.yaml"
    targets_yaml.write_text(yaml.safe_dump(target_doc, sort_keys=False), encoding="utf-8")
    batch_paths = _write_target_batches(
        output_dir=output_dir,
        base_doc={key: value for key, value in target_doc.items() if key != "targets"},
        targets=targets,
        batch_size=batch_size,
    )
    targets_parquet = output_dir / "targets.parquet"
    if len(gaia):
        gaia.to_parquet(targets_parquet, index=False)
    else:
        pd.DataFrame(columns=["survey_tile_id", "survey_hpx", "survey_nside", *GAIA_LITE_COLUMNS]).to_parquet(
            targets_parquet,
            index=False,
        )
    summary = {
        "tile_id": tile.tile_id,
        "nside": tile.nside,
        "hpx": tile.hpx,
        "order": tile.order,
        "target_count": int(len(targets)),
        "targets_yaml": str(targets_yaml),
        "target_batch_count": len(batch_paths),
        "target_batch_yamls": [str(path) for path in batch_paths],
        "targets_parquet": str(targets_parquet),
        "g_min": g_min,
        "g_max": g_max,
        "max_sources": max_sources,
        "batch_size": batch_size,
        "gaia_metrics": _jsonable(gaia.attrs.get("local_gaia_metrics", {})),
        "created_utc": target_doc["created_utc"],
    }
    (output_dir / "tile_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_target_batches(
    *,
    output_dir: Path,
    base_doc: dict[str, Any],
    targets: list[dict[str, Any]],
    batch_size: int,
) -> list[Path]:
    if batch_size <= 0 or len(targets) <= batch_size:
        return [output_dir / "targets.yaml"]
    batch_dir = output_dir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for batch_index, start in enumerate(range(0, len(targets), batch_size)):
        batch_targets = targets[start : start + batch_size]
        path = batch_dir / f"targets_batch_{batch_index:04d}.yaml"
        doc = {
            **base_doc,
            "batch": {
                "batch_index": batch_index,
                "batch_start": start,
                "batch_size": len(batch_targets),
                "target_count_total": len(targets),
            },
            "targets": batch_targets,
        }
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if pd.notna(out) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value
