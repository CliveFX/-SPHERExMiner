#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix

from spherex_laser_miner.catalog.local_2mass import _select_2mass_rows
from spherex_laser_miner.catalog.manual_targets import get_manual_target


DEFAULT_CACHE_ROOT = Path("/mnt/niroseti/spherex_cache")
DEFAULT_MANUAL_TARGETS = Path("configs/manual_targets.yaml")
DEFAULT_HPX_LEVEL = 5


def main() -> int:
    args = parse_args()
    center_ra, center_dec, center_label = resolve_center(args)
    parquet_root = (
        args.parquet_root.expanduser().resolve()
        if args.parquet_root
        else args.cache_root.expanduser().resolve() / "2mass" / "parquet" / args.dataset_name
    )
    glob_path = str(parquet_root / "hpx_level=*" / "hpx=*" / "part-*.parquet")
    if not any(parquet_root.glob("hpx_level=*/hpx=*/part-*.parquet")):
        raise SystemExit(f"No 2MASS Parquet shards found under {parquet_root}")

    df = query_2mass_cone(
        glob_path=glob_path,
        ra_deg=center_ra,
        dec_deg=center_dec,
        radius_deg=args.radius_deg,
        hpx_level=args.hpx_level,
        band=args.band,
        mag_min=args.mag_min,
        mag_max=args.mag_max,
        max_sources=args.max_sources,
        quality=args.quality,
        selection=args.selection,
    )
    fixed = to_fixed_targets(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixed.to_parquet(args.output, index=False)
    summary = {
        "center": {
            "label": center_label,
            "ra_deg": center_ra,
            "dec_deg": center_dec,
            "radius_deg": args.radius_deg,
        },
        "parquet_root": str(parquet_root),
        "output": str(args.output),
        "band": args.band,
        "mag_min": args.mag_min,
        "mag_max": args.mag_max,
        "quality": args.quality,
        "selection": args.selection,
        "row_count": int(len(fixed)),
        "target_ids_preview": fixed["target_id"].head(10).tolist() if len(fixed) else [],
        "runner_hint": (
            "Pass this file to spherex-mine run-depth-test with --fixed-targets-path. "
            "Use --max-gaia-sources 0 to make the summary less confusing; fixed targets bypass Gaia."
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create run-depth-test fixed targets from local 2MASS Parquet.")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--parquet-root", type=Path)
    parser.add_argument("--dataset-name", default="psc_lite")
    parser.add_argument("--manual-target", help="Manual target id from configs/manual_targets.yaml.")
    parser.add_argument("--manual-targets-path", type=Path, default=DEFAULT_MANUAL_TARGETS)
    parser.add_argument("--ra-deg", type=float)
    parser.add_argument("--dec-deg", type=float)
    parser.add_argument("--radius-deg", type=float, default=0.25)
    parser.add_argument("--hpx-level", type=int, default=DEFAULT_HPX_LEVEL)
    parser.add_argument("--band", choices=["J", "H", "Ks"], default="Ks")
    parser.add_argument("--mag-min", type=float, default=8.0)
    parser.add_argument("--mag-max", type=float, default=15.0)
    parser.add_argument("--max-sources", type=int, default=25)
    parser.add_argument(
        "--quality",
        default="ABC",
        help="Allowed per-band ph_qual letters. Empty string disables this quality cut.",
    )
    parser.add_argument("--selection", choices=["stratified", "brightest", "random"], default="stratified")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def resolve_center(args: argparse.Namespace) -> tuple[float, float, str]:
    if args.manual_target:
        target = get_manual_target(args.manual_targets_path, args.manual_target)
        return target.ra_deg, target.dec_deg, args.manual_target
    if args.ra_deg is None or args.dec_deg is None:
        raise SystemExit("Provide either --manual-target or both --ra-deg and --dec-deg")
    return float(args.ra_deg) % 360.0, float(args.dec_deg), "manual-ra-dec"


def query_2mass_cone(
    *,
    glob_path: str,
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    hpx_level: int,
    band: str,
    mag_min: float,
    mag_max: float,
    max_sources: int,
    quality: str,
    selection: str = "stratified",
) -> pd.DataFrame:
    try:
        import duckdb
    except ImportError as exc:
        raise SystemExit("duckdb is required for 2MASS fixed-target queries") from exc

    mag_col = {"J": "j_m", "H": "h_m", "Ks": "k_m"}[band]
    quality_index = {"J": 1, "H": 2, "Ks": 3}[band]
    hp = HEALPix(nside=2**hpx_level, order="nested", frame=None)
    center = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    hpx_candidates = [int(v) for v in hp.cone_search_lonlat(center.ra, center.dec, radius_deg * u.deg)]
    if not hpx_candidates:
        return pd.DataFrame()

    dec_min = max(-90.0, dec_deg - radius_deg)
    dec_max = min(90.0, dec_deg + radius_deg)
    ra_half = radius_deg / max(math.cos(math.radians(dec_deg)), 1e-6)
    ra_min = (ra_deg - ra_half) % 360.0
    ra_max = (ra_deg + ra_half) % 360.0
    wraps = ra_min > ra_max
    ra_clause = "(ra_deg >= ? OR ra_deg <= ?)" if wraps else "ra_deg BETWEEN ? AND ?"
    quality_clause = ""
    params: list[Any] = [mag_min, mag_max, *hpx_candidates, dec_min, dec_max, ra_min, ra_max]
    if quality:
        allowed = sorted(set(quality.upper()))
        quality_clause = f"AND substr(ph_qual, {quality_index}, 1) IN ({', '.join('?' for _ in allowed)})"
        params.extend(allowed)
    selection = str(selection or "stratified").lower()
    fetch_limit = max_sources * 20 if selection == "brightest" else max(max_sources * 500, 5000)
    params.append(fetch_limit)
    order_sql = f"{mag_col}, target_id" if selection == "brightest" else "hash(target_id)"

    sql = f"""
        SELECT *
        FROM read_parquet('{glob_path}', hive_partitioning=true)
        WHERE {mag_col} BETWEEN ? AND ?
          AND hpx IN ({', '.join('?' for _ in hpx_candidates)})
          AND dec_deg BETWEEN ? AND ?
          AND {ra_clause}
          {quality_clause}
        ORDER BY {order_sql}
        LIMIT ?
    """
    con = duckdb.connect(":memory:")
    try:
        df = con.execute(sql, params).fetchdf()
    finally:
        con.close()
    if df.empty:
        return df
    c = SkyCoord(df["ra_deg"].to_numpy(dtype=float) * u.deg, df["dec_deg"].to_numpy(dtype=float) * u.deg, frame="icrs")
    df = df.loc[center.separation(c).deg <= radius_deg].copy()
    return _select_2mass_rows(
        df.drop_duplicates(subset=["target_id"]),
        mag_col=mag_col,
        max_sources=max_sources,
        mag_min=mag_min,
        mag_max=mag_max,
        selection=selection,
    )


def to_fixed_targets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
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
            ]
        )
    out = df.copy()
    defaults = {
        "target_type": "2mass_psc",
        "source_id": out["source_catalog_id"].astype(str),
        "ra_reference_deg": pd.to_numeric(out["ra_deg"], errors="coerce"),
        "dec_reference_deg": pd.to_numeric(out["dec_deg"], errors="coerce"),
        "reference_epoch_yr": 2000.0,
        "pmra_masyr": np.nan,
        "pmdec_masyr": np.nan,
        "parallax_mas": np.nan,
        "priority_score": 100.0 - pd.to_numeric(out["mag_primary"], errors="coerce").clip(upper=99.0),
        "target_filter_flags": "2mass_fixed_target",
        "source_catalog": "2mass_psc",
    }
    for column, value in defaults.items():
        if column not in out.columns:
            out[column] = value
    out["target_filter_flags"] = out["target_filter_flags"].fillna("2mass_fixed_target").astype(str)
    fixed_columns = [
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
    out = out[fixed_columns].copy()
    for column in ["j_m", "h_m", "k_m", "prox"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["pts_key"] = pd.to_numeric(out["pts_key"], errors="coerce").astype("Int64")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
