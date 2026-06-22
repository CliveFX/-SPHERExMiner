from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import pyvo

from spherex_laser_miner.catalog.local_gaia_lite import (
    local_gaia_lite_exists,
    query_local_gaia_lite_duckdb,
)
from spherex_laser_miner.config import DEFAULT_CACHE_ROOT


GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap"


def query_gaia_for_s_region(
    s_region: str,
    cache_path: Path,
    max_sources: int = 500,
    g_min: float = 8.0,
    g_max: float = 19.0,
) -> pd.DataFrame:
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return pd.read_parquet(cache_path)

    cache_root = _cache_root_from_query_cache(cache_path)
    if local_gaia_lite_exists(cache_root):
        df = query_local_gaia_lite_duckdb(
            s_region=s_region,
            cache_root=cache_root,
            max_sources=max_sources,
            g_min=g_min,
            g_max=g_max,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        return df

    coords = polygon_coords_from_s_region(s_region)
    vertices = polygon_vertices_from_s_region(s_region)
    df = _query_gaia_tiled(coords, vertices, max_sources=max_sources, g_min=g_min, g_max=g_max)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def _query_gaia_tiled(
    polygon_coords: str,
    vertices: list[tuple[float, float]],
    max_sources: int,
    g_min: float,
    g_max: float,
) -> pd.DataFrame:
    service = pyvo.dal.TAPService(GAIA_TAP_URL)
    if max_sources <= 0:
        return pd.DataFrame()

    ra_values = [ra for ra, _ in vertices]
    dec_values = [dec for _, dec in vertices]
    ra_min, ra_max = min(ra_values), max(ra_values)
    dec_min, dec_max = min(dec_values), max(dec_values)
    grid = 4
    per_tile = max(4, int(max_sources / (grid * grid)) + 2)
    frames = []
    for ix in range(grid):
        lo_ra = ra_min + (ra_max - ra_min) * ix / grid
        hi_ra = ra_min + (ra_max - ra_min) * (ix + 1) / grid
        for iy in range(grid):
            lo_dec = dec_min + (dec_max - dec_min) * iy / grid
            hi_dec = dec_min + (dec_max - dec_min) * (iy + 1) / grid
            query = _gaia_query(
                polygon_coords=polygon_coords,
                top_n=per_tile,
                g_min=g_min,
                g_max=g_max,
                ra_min=lo_ra,
                ra_max=hi_ra,
                dec_min=lo_dec,
                dec_max=hi_dec,
            )
            try:
                table = service.search(query, maxrec=per_tile).to_table()
            except Exception:
                continue
            if len(table):
                frames.append(table.to_pandas())

    if not frames:
        query = _gaia_query(polygon_coords=polygon_coords, top_n=max_sources, g_min=g_min, g_max=g_max)
        table = service.search(query, maxrec=max_sources).to_table()
        return table.to_pandas()

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["source_id"])
    # Deterministic distributed subsample: sort by sky tile then brightness enough
    # to keep the viewer and smoke run stable without forcing TAP to sort a large set.
    df["_ra_bin"] = pd.cut(df["ra"], bins=min(grid, max(1, df["ra"].nunique())), labels=False, duplicates="drop")
    df["_dec_bin"] = pd.cut(df["dec"], bins=min(grid, max(1, df["dec"].nunique())), labels=False, duplicates="drop")
    df = df.sort_values(["_ra_bin", "_dec_bin", "phot_g_mean_mag"]).drop(columns=["_ra_bin", "_dec_bin"])
    return df.head(max_sources).reset_index(drop=True)


def _gaia_query(
    polygon_coords: str,
    top_n: int,
    g_min: float,
    g_max: float,
    ra_min: float | None = None,
    ra_max: float | None = None,
    dec_min: float | None = None,
    dec_max: float | None = None,
) -> str:
    bounds = ""
    if ra_min is not None:
        bounds = f"""
  AND ra BETWEEN {float(ra_min)} AND {float(ra_max)}
  AND dec BETWEEN {float(dec_min)} AND {float(dec_max)}
"""
    return f"""
SELECT TOP {int(top_n)}
    source_id, ra, dec, ref_epoch, pmra, pmdec, parallax, parallax_error,
    phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp, ruwe,
    duplicated_source, astrometric_params_solved
FROM gaiadr3.gaia_source
WHERE 1=CONTAINS(POINT('ICRS', ra, dec), POLYGON('ICRS', {polygon_coords}))
  AND phot_g_mean_mag BETWEEN {float(g_min)} AND {float(g_max)}
{bounds}
"""


def polygon_coords_from_s_region(s_region: str) -> str:
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", s_region)
    if len(numbers) < 6:
        raise ValueError(f"Cannot parse POLYGON s_region: {s_region}")
    # SIA polygons commonly repeat the first point at the end. ADQL POLYGON
    # expects each vertex once.
    if len(numbers) >= 10 and numbers[0] == numbers[-2] and numbers[1] == numbers[-1]:
        numbers = numbers[:-2]
    return ", ".join(numbers)


def polygon_vertices_from_s_region(s_region: str) -> list[tuple[float, float]]:
    numbers = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", s_region)]
    if len(numbers) < 6:
        raise ValueError(f"Cannot parse POLYGON s_region: {s_region}")
    vertices = list(zip(numbers[0::2], numbers[1::2], strict=False))
    if len(vertices) >= 2 and vertices[0] == vertices[-1]:
        vertices = vertices[:-1]
    return vertices


def _cache_root_from_query_cache(cache_path: Path) -> Path:
    resolved = cache_path.resolve()
    for parent in [resolved.parent, *resolved.parents]:
        if parent.name == "runs":
            return parent.parent
    return Path(os.getenv("SPHEREX_CACHE_ROOT", DEFAULT_CACHE_ROOT))
