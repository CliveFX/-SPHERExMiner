from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix


GAIA_COLUMNS = [
    "source_id",
    "ra",
    "dec",
    "ref_epoch",
    "pmra",
    "pmdec",
    "parallax",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "bp_rp",
]

TWOMASS_COLUMNS = [
    "target_id",
    "target_type",
    "source_id",
    "source_catalog",
    "source_catalog_id",
    "object_name",
    "ra_deg",
    "dec_deg",
    "ra_reference_deg",
    "dec_reference_deg",
    "reference_epoch_yr",
    "pmra_masyr",
    "pmdec_masyr",
    "parallax_mas",
    "mag_primary",
    "mag_primary_band",
    "priority_score",
    "target_filter_flags",
    "j_m",
    "h_m",
    "k_m",
    "ph_qual",
]


@dataclass(frozen=True)
class CatalogConfig:
    cache_root: Path
    catalog: str = "all"
    gaia_g_min: float = 11.0
    gaia_g_max: float = 16.0
    twomass_mag_min: float | None = 11.0
    twomass_mag_max: float | None = 16.0
    max_sources_per_frame: int | None = 5000
    gaia_max_sources_per_frame: int | None = None
    twomass_max_sources_per_frame: int | None = None
    bbox_pad_deg: float = 0.05


def build_frame_targets(
    *,
    manifest_path: Path,
    output_path: Path,
    config: CatalogConfig,
    limit_frames: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    manifest = pd.read_parquet(manifest_path)
    if limit_frames is not None:
        manifest = manifest.head(limit_frames).copy()
    rows: list[pd.DataFrame] = []
    timings: list[dict[str, Any]] = []
    for frame in manifest.to_dict(orient="records"):
        t0 = time.perf_counter()
        frame_rows = query_targets_for_frame(frame, config)
        timings.append(
            {
                "frame_group_id": frame.get("frame_group_id"),
                "image_id": frame.get("image_id"),
                "target_count": int(len(frame_rows)),
                "wall_time_sec": time.perf_counter() - t0,
            }
        )
        if len(frame_rows):
            rows.append(frame_rows)
    targets = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=_output_columns())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    targets.to_parquet(output_path, index=False)
    write_wall = time.perf_counter() - t0
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "output_path": str(output_path),
        "catalog": config.catalog,
        "gaia_g_min": float(config.gaia_g_min),
        "gaia_g_max": float(config.gaia_g_max),
        "twomass_mag_min": _json_float_or_none(config.twomass_mag_min),
        "twomass_mag_max": _json_float_or_none(config.twomass_mag_max),
        "max_sources_per_frame": config.max_sources_per_frame,
        "gaia_max_sources_per_frame": _json_int_or_none(_resolve_catalog_limit(config, "gaia")),
        "twomass_max_sources_per_frame": _json_int_or_none(_resolve_catalog_limit(config, "2mass")),
        "twomass_uncapped": _resolve_catalog_limit(config, "2mass") is None,
        "twomass_magnitude_filtered": config.twomass_mag_min is not None or config.twomass_mag_max is not None,
        "frame_count": int(len(manifest)),
        "target_row_count": int(len(targets)),
        "unique_target_count": int(targets["target_id"].nunique()) if len(targets) else 0,
        "total_wall_sec": time.perf_counter() - started,
        "write_wall_sec": write_wall,
        "frame_timings": timings,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_profile(output_path.with_suffix(".profile.json"), summary)
    return summary


def query_targets_for_frame(frame: dict[str, Any], config: CatalogConfig) -> pd.DataFrame:
    bounds = _frame_bounds(frame, pad_deg=config.bbox_pad_deg)
    pieces: list[pd.DataFrame] = []
    if config.catalog in {"gaia", "all"}:
        gaia = _query_gaia(bounds=bounds, config=config, max_sources=_resolve_catalog_limit(config, "gaia"))
        if len(gaia):
            pieces.append(_normalize_gaia(gaia, frame))
    if config.catalog in {"2mass", "all"}:
        twomass = _query_2mass(bounds=bounds, config=config, max_sources=_resolve_catalog_limit(config, "2mass"))
        if len(twomass):
            pieces.append(_normalize_2mass(twomass, frame))
    if not pieces:
        return pd.DataFrame(columns=_output_columns())
    out = pd.concat(pieces, ignore_index=True)
    out = out.drop_duplicates(subset=["frame_group_id", "target_id", "catalog"]).reset_index(drop=True)
    return out[_output_columns()]


def _query_gaia(*, bounds: tuple[float, float, float, float], config: CatalogConfig, max_sources: int | None) -> pd.DataFrame:
    root = config.cache_root / "gaia" / "parquet" / "dr3_source_lite"
    if not root.exists():
        return pd.DataFrame(columns=GAIA_COLUMNS)
    hpx_values = _candidate_hpx(bounds, hpx_level=3, samples=7)
    if not hpx_values:
        return pd.DataFrame(columns=GAIA_COLUMNS)
    parquet_files = _parquet_files(root=root, hpx_level=3, hpx_values=hpx_values)
    if not parquet_files:
        return pd.DataFrame(columns=GAIA_COLUMNS)
    parquet_expr = _duckdb_parquet_list(parquet_files)
    ra_clause, ra_params = _ra_where("ra", bounds[0], bounds[1])
    limit_clause, limit_params = _limit_clause(max_sources)
    params: list[Any] = [config.gaia_g_min, config.gaia_g_max, *hpx_values, *ra_params, bounds[2], bounds[3], *limit_params]
    sql = f"""
        SELECT {", ".join(GAIA_COLUMNS)}
        FROM read_parquet({parquet_expr}, hive_partitioning=true)
        WHERE phot_g_mean_mag BETWEEN ? AND ?
          AND hpx IN ({", ".join("?" for _ in hpx_values)})
          AND {ra_clause}
          AND dec BETWEEN ? AND ?
        ORDER BY phot_g_mean_mag, source_id
        {limit_clause}
    """
    return _execute_df(sql, params)


def _query_2mass(*, bounds: tuple[float, float, float, float], config: CatalogConfig, max_sources: int | None) -> pd.DataFrame:
    root = config.cache_root / "2mass" / "parquet" / "psc_lite"
    if not root.exists():
        return pd.DataFrame(columns=TWOMASS_COLUMNS)
    hpx_values = _candidate_hpx(bounds, hpx_level=5, samples=7)
    if not hpx_values:
        return pd.DataFrame(columns=TWOMASS_COLUMNS)
    parquet_files = _parquet_files(root=root, hpx_level=5, hpx_values=hpx_values)
    if not parquet_files:
        return pd.DataFrame(columns=TWOMASS_COLUMNS)
    parquet_expr = _duckdb_parquet_list(parquet_files)
    ra_clause, ra_params = _ra_where("ra_deg", bounds[0], bounds[1])
    mag_clause, mag_params = _twomass_mag_clause(config)
    limit_clause, limit_params = _limit_clause(max_sources)
    params: list[Any] = [*mag_params, *hpx_values, *ra_params, bounds[2], bounds[3], *limit_params]
    sql = f"""
        SELECT {", ".join(TWOMASS_COLUMNS)}
        FROM read_parquet({parquet_expr}, hive_partitioning=true)
        WHERE {mag_clause}
          AND hpx IN ({", ".join("?" for _ in hpx_values)})
          AND {ra_clause}
          AND dec_deg BETWEEN ? AND ?
        ORDER BY mag_primary, target_id
        {limit_clause}
    """
    return _execute_df(sql, params)


def _normalize_gaia(df: pd.DataFrame, frame: dict[str, Any]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["frame_group_id"] = str(frame.get("frame_group_id"))
    out["image_id"] = str(frame.get("image_id"))
    out["catalog"] = "gaia_dr3"
    out["target_id"] = "gaia_dr3_" + df["source_id"].astype("string")
    out["source_id"] = df["source_id"].astype("string")
    out["ra_deg"] = pd.to_numeric(df["ra"], errors="coerce")
    out["dec_deg"] = pd.to_numeric(df["dec"], errors="coerce")
    out["reference_epoch_yr"] = pd.to_numeric(df["ref_epoch"], errors="coerce")
    out["pmra_masyr"] = pd.to_numeric(df["pmra"], errors="coerce")
    out["pmdec_masyr"] = pd.to_numeric(df["pmdec"], errors="coerce")
    out["parallax_mas"] = pd.to_numeric(df["parallax"], errors="coerce")
    out["mag_primary"] = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce")
    out["mag_primary_band"] = "G"
    return out


def _normalize_2mass(df: pd.DataFrame, frame: dict[str, Any]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["frame_group_id"] = str(frame.get("frame_group_id"))
    out["image_id"] = str(frame.get("image_id"))
    out["catalog"] = "2mass_psc"
    out["target_id"] = df["target_id"].astype("string")
    out["source_id"] = df["source_id"].astype("string")
    out["ra_deg"] = pd.to_numeric(df["ra_reference_deg"], errors="coerce").fillna(pd.to_numeric(df["ra_deg"], errors="coerce"))
    out["dec_deg"] = pd.to_numeric(df["dec_reference_deg"], errors="coerce").fillna(pd.to_numeric(df["dec_deg"], errors="coerce"))
    out["reference_epoch_yr"] = pd.to_numeric(df["reference_epoch_yr"], errors="coerce")
    out["pmra_masyr"] = pd.to_numeric(df["pmra_masyr"], errors="coerce")
    out["pmdec_masyr"] = pd.to_numeric(df["pmdec_masyr"], errors="coerce")
    out["parallax_mas"] = pd.to_numeric(df["parallax_mas"], errors="coerce")
    out["mag_primary"] = pd.to_numeric(df["mag_primary"], errors="coerce")
    out["mag_primary_band"] = df["mag_primary_band"].astype("string")
    return out


def _frame_bounds(frame: dict[str, Any], *, pad_deg: float) -> tuple[float, float, float, float]:
    ra_min = _finite_or(frame.get("ra_min_deg"), frame.get("ra_center_deg"))
    ra_max = _finite_or(frame.get("ra_max_deg"), frame.get("ra_center_deg"))
    dec_min = _finite_or(frame.get("dec_min_deg"), frame.get("dec_center_deg"))
    dec_max = _finite_or(frame.get("dec_max_deg"), frame.get("dec_center_deg"))
    if ra_min is None or ra_max is None or dec_min is None or dec_max is None:
        raise ValueError(f"Frame has no usable footprint bounds: {frame.get('image_id')}")
    return ((ra_min - pad_deg) % 360.0, (ra_max + pad_deg) % 360.0, max(-90.0, dec_min - pad_deg), min(90.0, dec_max + pad_deg))


def _candidate_hpx(bounds: tuple[float, float, float, float], *, hpx_level: int, samples: int) -> list[int]:
    ra_min, ra_max, dec_min, dec_max = bounds
    ras = _sample_ra(ra_min, ra_max, samples)
    decs = np.linspace(dec_min, dec_max, samples)
    ra_grid, dec_grid = np.meshgrid(ras, decs)
    coords = SkyCoord(ra_grid.ravel() * u.deg, dec_grid.ravel() * u.deg, frame="icrs")
    hp = HEALPix(nside=2**hpx_level, order="nested", frame="icrs")
    return sorted({int(v) for v in hp.skycoord_to_healpix(coords)})


def _sample_ra(ra_min: float, ra_max: float, samples: int) -> np.ndarray:
    if _ra_wraps(ra_min, ra_max):
        end = ra_max + 360.0
        values = np.linspace(ra_min, end, samples) % 360.0
    else:
        values = np.linspace(ra_min, ra_max, samples)
    return values


def _ra_where(column: str, ra_min: float, ra_max: float) -> tuple[str, list[float]]:
    if _ra_wraps(ra_min, ra_max):
        return f"({column} >= ? OR {column} <= ?)", [ra_min, ra_max]
    return f"{column} BETWEEN ? AND ?", [ra_min, ra_max]


def _ra_wraps(ra_min: float, ra_max: float) -> bool:
    return float(ra_min) > float(ra_max)


def _resolve_catalog_limit(config: CatalogConfig, catalog: str) -> int | None:
    if catalog == "gaia":
        specific = config.gaia_max_sources_per_frame
    elif catalog == "2mass":
        specific = config.twomass_max_sources_per_frame
    else:
        raise ValueError(f"Unknown catalog limit key: {catalog}")

    if specific is not None:
        return _normalize_limit(specific)

    legacy = config.max_sources_per_frame
    if legacy is None:
        return None
    legacy_limit = _normalize_limit(legacy)
    if legacy_limit is None:
        return None
    if config.catalog == "all":
        return max(1, legacy_limit // 2)
    return legacy_limit


def _normalize_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    value = int(limit)
    if value <= 0:
        return None
    return value


def _limit_clause(max_sources: int | None) -> tuple[str, list[int]]:
    if max_sources is None:
        return "", []
    return "LIMIT ?", [int(max_sources)]


def _twomass_mag_clause(config: CatalogConfig) -> tuple[str, list[float]]:
    lower = config.twomass_mag_min
    upper = config.twomass_mag_max
    if lower is None and upper is None:
        return "TRUE", []
    if lower is not None and upper is not None:
        if float(lower) > float(upper):
            raise ValueError("twomass_mag_min must be <= twomass_mag_max")
        return "mag_primary BETWEEN ? AND ?", [float(lower), float(upper)]
    if lower is not None:
        return "mag_primary >= ?", [float(lower)]
    return "mag_primary <= ?", [float(upper)]


def _json_int_or_none(value: int | None) -> int | None:
    return None if value is None else int(value)


def _json_float_or_none(value: float | None) -> float | None:
    return None if value is None else float(value)


def _execute_df(sql: str, params: list[Any]) -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    try:
        return con.execute(sql, params).fetchdf()
    finally:
        con.close()


def _parquet_files(*, root: Path, hpx_level: int, hpx_values: list[int]) -> list[Path]:
    files: list[Path] = []
    for hpx in hpx_values:
        files.extend(sorted((root / f"hpx_level={hpx_level}" / f"hpx={int(hpx)}").glob("part-*.parquet")))
    return files


def _duckdb_parquet_list(paths: list[Path]) -> str:
    quoted = []
    for path in paths:
        value = str(path).replace("'", "''")
        quoted.append(f"'{value}'")
    return "[" + ", ".join(quoted) + "]"


def _finite_or(*values: Any) -> float | None:
    for value in values:
        try:
            out = float(value)
        except Exception:
            continue
        if math.isfinite(out):
            return out
    return None


def _output_columns() -> list[str]:
    return [
        "frame_group_id",
        "image_id",
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


def _write_profile(path: Path, summary: dict[str, Any]) -> None:
    total = max(float(summary.get("total_wall_sec") or 0.0), 1e-12)
    query_wall = sum(float(row["wall_time_sec"]) for row in summary.get("frame_timings", []))
    write_wall = float(summary.get("write_wall_sec") or 0.0)
    profile = {
        "created_utc": summary.get("created_utc"),
        "rows": [
            {
                "stage": "catalog_query",
                "function_or_script": "luxquarry_allsky_engine.catalog.build_frame_targets",
                "wall_time_sec": query_wall,
                "wall_time_pct": 100.0 * query_wall / total,
                "backend": "duckdb_parquet_cpu",
                "rows_out": summary.get("target_row_count"),
            },
            {
                "stage": "write_frame_targets",
                "function_or_script": "luxquarry_allsky_engine.catalog.build_frame_targets",
                "wall_time_sec": write_wall,
                "wall_time_pct": 100.0 * write_wall / total,
                "backend": "pandas_pyarrow",
                "rows_out": summary.get("target_row_count"),
            },
        ],
    }
    path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
