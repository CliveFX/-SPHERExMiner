from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spherex_laser_miner.catalog.local_gaia_lite import (
    _point_in_polygon_mask,
    bounds_from_vertices,
    candidate_hpx_for_polygon,
    polygon_vertices_from_s_region,
)


TWOMASS_TARGET_COLUMNS = [
    "target_id",
    "target_type",
    "source_id",
    "object_name",
    "ra_reference_deg",
    "dec_reference_deg",
    "reference_epoch_yr",
    "pmra_masyr",
    "pmdec_masyr",
    "parallax_mas",
    "priority_score",
    "target_filter_flags",
    "source_catalog",
    "source_catalog_id",
    "j_m",
    "h_m",
    "k_m",
    "ph_qual",
    "rd_flg",
    "cc_flg",
    "prox",
    "pts_key",
]


def twomass_lite_dir(cache_root: Path, dataset_name: str = "psc_lite") -> Path:
    return cache_root / "2mass" / "parquet" / dataset_name


def local_2mass_exists(cache_root: Path, dataset_name: str = "psc_lite") -> bool:
    root = twomass_lite_dir(cache_root, dataset_name)
    return any(root.glob("hpx_level=*/hpx=*/part-*.parquet"))


def query_local_2mass_duckdb(
    s_region: str,
    cache_root: Path,
    max_sources: int = 500,
    mag_min: float = 8.0,
    mag_max: float = 15.0,
    band: str = "Ks",
    quality: str = "ABC",
    dataset_name: str = "psc_lite",
    hpx_level: int = 5,
    selection: str = "stratified",
) -> pd.DataFrame:
    if max_sources <= 0:
        return pd.DataFrame()
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("duckdb is required for query_local_2mass_duckdb") from exc

    root = twomass_lite_dir(cache_root, dataset_name)
    if not local_2mass_exists(cache_root, dataset_name):
        raise FileNotFoundError(f"No local 2MASS Parquet shards found under {root}")

    vertices = polygon_vertices_from_s_region(s_region)
    bounds = bounds_from_vertices(vertices)
    ra_min, ra_max, dec_min, dec_max = bounds
    ra_lo, ra_hi = _unwrap_bounds(ra_min, ra_max)
    mag_col = {"J": "j_m", "H": "h_m", "Ks": "k_m"}[band]
    quality_index = {"J": 1, "H": 2, "Ks": 3}[band]
    glob_path = str(root / "hpx_level=*" / "hpx=*" / "part-*.parquet")
    candidate_hpx = candidate_hpx_for_polygon(vertices, bounds, hpx_level=hpx_level)
    if not candidate_hpx:
        return pd.DataFrame()

    params: list[Any] = [mag_min, mag_max]
    hpx_clause = f"AND hpx IN ({', '.join('?' for _ in candidate_hpx)})"
    params.extend(candidate_hpx)
    where_ra = "ra_deg BETWEEN ? AND ?"
    if _bounds_wrap_ra(ra_min, ra_max):
        where_ra = "(ra_deg >= ? OR ra_deg <= ?)"
        params.extend([ra_min % 360.0, ra_max % 360.0])
    else:
        params.extend([ra_lo, ra_hi])
    params.extend([dec_min, dec_max])
    quality_clause = ""
    if quality:
        allowed = sorted(set(str(quality).upper()))
        quality_clause = f"AND substr(ph_qual, {quality_index}, 1) IN ({', '.join('?' for _ in allowed)})"
        params.extend(allowed)
    selection = str(selection or "stratified").lower()
    if selection not in {"stratified", "brightest", "random"}:
        raise ValueError("2MASS selection must be one of: stratified, brightest, random")
    fetch_limit = max_sources * 20 if selection == "brightest" else max(max_sources * 500, 5000)
    params.append(fetch_limit)
    order_sql = f"{mag_col}, target_id" if selection == "brightest" else "hash(target_id)"
    sql = f"""
        SELECT *
        FROM read_parquet('{glob_path}', hive_partitioning=true)
        WHERE {mag_col} BETWEEN ? AND ?
          {hpx_clause}
          AND {where_ra}
          AND dec_deg BETWEEN ? AND ?
          {quality_clause}
        ORDER BY {order_sql}
        LIMIT ?
    """
    start = time.perf_counter()
    con = duckdb.connect(":memory:")
    try:
        df = con.execute(sql, params).fetchdf()
    finally:
        con.close()
    rows_after_bbox = len(df)
    if len(df):
        polygon_mask = _point_in_polygon_mask(df["ra_deg"], df["dec_deg"], vertices, bounds)
        df = df.loc[polygon_mask].copy()
    if len(df):
        df = df.drop_duplicates(subset=["target_id"])
        df = _select_2mass_rows(
            df,
            mag_col=mag_col,
            max_sources=max_sources,
            mag_min=mag_min,
            mag_max=mag_max,
            selection=selection,
        )
    df.attrs["local_2mass_metrics"] = {
        "index_root": str(root),
        "engine": "duckdb",
        "hpx_level": hpx_level,
        "hpx_candidate_count": len(candidate_hpx),
        "rows_after_bbox_and_magnitude": rows_after_bbox,
        "rows_after_polygon": len(df),
        "exact_polygon_filter": True,
        "query_wall_time_sec": time.perf_counter() - start,
        "band": band,
        "mag_min": mag_min,
        "mag_max": mag_max,
        "quality": quality,
        "selection": selection,
        "fetch_limit": fetch_limit,
    }
    return df


def _select_2mass_rows(
    df: pd.DataFrame,
    *,
    mag_col: str,
    max_sources: int,
    mag_min: float,
    mag_max: float,
    selection: str,
) -> pd.DataFrame:
    if len(df) <= max_sources:
        return df.sort_values([mag_col, "target_id"], kind="mergesort").reset_index(drop=True)
    if selection == "brightest":
        return df.sort_values([mag_col, "target_id"], kind="mergesort").head(max_sources).reset_index(drop=True)
    if selection == "random":
        return (
            df.assign(_sort_key=df["target_id"].map(lambda value: hash(str(value))))
            .sort_values("_sort_key", kind="mergesort")
            .drop(columns=["_sort_key"])
            .head(max_sources)
            .sort_values([mag_col, "target_id"], kind="mergesort")
            .reset_index(drop=True)
        )

    # Stratify across the requested magnitude span so wide bins do not collapse
    # to the bright edge. Remainders are filled from still-unused rows.
    bin_count = max(1, min(10, max_sources, int(math.ceil(max(mag_max - mag_min, 1e-6)))))
    edges = np.linspace(mag_min, mag_max, bin_count + 1)
    per_bin = max_sources // bin_count
    remainder = max_sources % bin_count
    selected_parts: list[pd.DataFrame] = []
    used_index: set[Any] = set()
    sorted_df = df.sort_values([mag_col, "target_id"], kind="mergesort")
    for idx in range(bin_count):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == bin_count - 1:
            mask = (sorted_df[mag_col] >= lo) & (sorted_df[mag_col] <= hi)
        else:
            mask = (sorted_df[mag_col] >= lo) & (sorted_df[mag_col] < hi)
        quota = per_bin + (1 if idx < remainder else 0)
        part = sorted_df.loc[mask & ~sorted_df.index.isin(used_index)].head(quota)
        if len(part):
            selected_parts.append(part)
            used_index.update(part.index.tolist())
    selected = pd.concat(selected_parts, ignore_index=False) if selected_parts else sorted_df.head(0)
    if len(selected) < max_sources:
        fill = sorted_df.loc[~sorted_df.index.isin(set(selected.index.tolist()))].head(max_sources - len(selected))
        selected = pd.concat([selected, fill], ignore_index=False)
    return selected.sort_values([mag_col, "target_id"], kind="mergesort").head(max_sources).reset_index(drop=True)


def twomass_to_fixed_target_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        rows.append(
            {
                "target_id": str(row.get("target_id")),
                "target_type": str(row.get("target_type") or "2mass_psc"),
                "source_id": str(row.get("source_id") or row.get("source_catalog_id") or ""),
                "object_name": row.get("object_name"),
                "ra_reference_deg": float(row.get("ra_reference_deg", row.get("ra_deg"))),
                "dec_reference_deg": float(row.get("dec_reference_deg", row.get("dec_deg"))),
                "reference_epoch_yr": _float_or_default(row.get("reference_epoch_yr"), 2000.0),
                "pmra_masyr": _float_or_none(row.get("pmra_masyr")),
                "pmdec_masyr": _float_or_none(row.get("pmdec_masyr")),
                "parallax_mas": _float_or_none(row.get("parallax_mas")),
                "priority_score": _float_or_default(row.get("priority_score"), 0.0),
                "target_filter_flags": str(row.get("target_filter_flags") or "2mass_static_position"),
                "source_catalog": str(row.get("source_catalog") or "2mass_psc"),
                "source_catalog_id": str(row.get("source_catalog_id") or row.get("designation") or ""),
                "j_m": _float_or_none(row.get("j_m")),
                "h_m": _float_or_none(row.get("h_m")),
                "k_m": _float_or_none(row.get("k_m")),
                "ph_qual": row.get("ph_qual"),
                "rd_flg": row.get("rd_flg"),
                "cc_flg": row.get("cc_flg"),
                "prox": _float_or_none(row.get("prox")),
                "pts_key": _int_or_none(row.get("pts_key")),
            }
        )
    return rows


def write_fixed_targets(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=TWOMASS_TARGET_COLUMNS).to_parquet(path, index=False)


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _float_or_default(value: Any, default: float) -> float:
    out = _float_or_none(value)
    return default if out is None else out


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _unwrap_bounds(ra_min: float, ra_max: float) -> tuple[float, float]:
    if _bounds_wrap_ra(ra_min, ra_max):
        return ra_min, ra_max
    return ra_min % 360.0, ra_max % 360.0


def _bounds_wrap_ra(ra_min: float, ra_max: float) -> bool:
    return ra_max > 360.0 or ra_min < 0.0 or ra_min > ra_max
