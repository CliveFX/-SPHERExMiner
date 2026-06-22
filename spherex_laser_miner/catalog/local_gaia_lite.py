from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from spherex_laser_miner import __version__


GAIA_LITE_COLUMNS = [
    "source_id",
    "ra",
    "dec",
    "ref_epoch",
    "pmra",
    "pmdec",
    "parallax",
    "parallax_error",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "bp_rp",
    "ruwe",
    "duplicated_source",
    "astrometric_params_solved",
]
DEFAULT_HPX_LEVEL = 3
GAIA_SOURCE_ID_HEALPIX_LEVEL = 12
DEFAULT_MAX_ROWS_PER_FILE = 2_500_000
DEFAULT_MAX_BUFFERED_ROWS = 5_000_000


@dataclass(frozen=True)
class GaiaHealpixPartitioner:
    """Gaia source_id-derived nested HEALPix partitioning."""

    hpx_level: int = DEFAULT_HPX_LEVEL

    @property
    def cell_count(self) -> int:
        return 12 * (4**self.hpx_level)

    @property
    def shift_from_source_id(self) -> int:
        return 35 + 2 * (GAIA_SOURCE_ID_HEALPIX_LEVEL - self.hpx_level)

    def cell_ids(self, source_id: pd.Series | np.ndarray) -> np.ndarray:
        ids = np.asarray(source_id, dtype=np.int64)
        return np.right_shift(ids, self.shift_from_source_id).astype(np.int64)

    def manifest(self) -> dict[str, Any]:
        return {
            "name": "gaia_source_id_nested_healpix",
            "hpx_level": self.hpx_level,
            "source_id_healpix_level": GAIA_SOURCE_ID_HEALPIX_LEVEL,
            "source_id_shift_bits": self.shift_from_source_id,
            "cell_count": self.cell_count,
            "note": "Gaia source_id encodes nested HEALPix level 12; hpx is derived by right-shifting source_id.",
        }


def gaia_lite_dir(cache_root: Path) -> Path:
    return cache_root / "gaia" / "parquet" / "dr3_source_lite"


def gaia_raw_dir(cache_root: Path) -> Path:
    return cache_root / "gaia" / "raw_download" / "gaia_dr3"


def local_gaia_lite_exists(cache_root: Path) -> bool:
    root = gaia_lite_dir(cache_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(manifest.get("complete_raw_file_set")) and any(root.glob("hpx_level=*/hpx=*/part-*.parquet"))


def build_gaia_lite_index(
    cache_root: Path,
    limit_files: int | None = None,
    overwrite: bool = False,
    hp_level: int = DEFAULT_HPX_LEVEL,
    max_rows_per_file: int = DEFAULT_MAX_ROWS_PER_FILE,
    max_buffered_rows: int = DEFAULT_MAX_BUFFERED_ROWS,
) -> dict[str, Any]:
    output_root = gaia_lite_dir(cache_root)
    if output_root.exists() and not overwrite:
        raise FileExistsError(f"{output_root} already exists; pass --overwrite to rebuild")
    if output_root.exists():
        _remove_existing_parquet(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(gaia_raw_dir(cache_root).glob("GaiaSource_*.csv.gz"))
    if limit_files is not None:
        raw_files = raw_files[:limit_files]
    if not raw_files:
        raise FileNotFoundError(
            f"No GaiaSource_*.csv.gz files found under {gaia_raw_dir(cache_root)}"
        )

    partitioner = GaiaHealpixPartitioner(hpx_level=hp_level)
    start = time.perf_counter()
    row_count = 0
    bytes_read = 0
    bytes_written_before = _parquet_bytes(output_root)
    part_counters: dict[int, int] = {}
    buffers: dict[int, list[pd.DataFrame]] = {}
    buffer_rows: dict[int, int] = {}
    total_buffered_rows = 0

    for raw_file in raw_files:
        bytes_read += raw_file.stat().st_size
        skip_rows = _ecsv_metadata_rows(raw_file)
        reader = pacsv.open_csv(
            raw_file,
            read_options=pacsv.ReadOptions(
                block_size=32 * 1024 * 1024,
                skip_rows=skip_rows,
            ),
            convert_options=pacsv.ConvertOptions(include_columns=GAIA_LITE_COLUMNS),
        )
        for batch in reader:
            table = pa.Table.from_batches([batch])
            if table.num_rows == 0:
                continue
            frame = table.to_pandas()
            missing = [col for col in GAIA_LITE_COLUMNS if col not in frame.columns]
            if missing:
                raise ValueError(f"{raw_file} is missing required Gaia columns: {missing}")
            frame = frame[GAIA_LITE_COLUMNS]
            frame["hpx"] = partitioner.cell_ids(frame["source_id"])
            row_count += len(frame)
            for hpx, part in frame.groupby("hpx", sort=True):
                hpx_id = int(hpx)
                clean_part = part.drop(columns=["hpx"])
                buffers.setdefault(hpx_id, []).append(clean_part)
                rows_added = len(clean_part)
                buffer_rows[hpx_id] = buffer_rows.get(hpx_id, 0) + rows_added
                total_buffered_rows += rows_added
                if buffer_rows[hpx_id] >= max_rows_per_file:
                    total_buffered_rows -= _flush_hpx_buffer(
                        output_root=output_root,
                        hpx_level=hp_level,
                        hpx=hpx_id,
                        buffers=buffers,
                        buffer_rows=buffer_rows,
                        part_counters=part_counters,
                    )
            while total_buffered_rows > max_buffered_rows and buffer_rows:
                largest_hpx = max(buffer_rows, key=buffer_rows.get)
                total_buffered_rows -= _flush_hpx_buffer(
                    output_root=output_root,
                    hpx_level=hp_level,
                    hpx=largest_hpx,
                    buffers=buffers,
                    buffer_rows=buffer_rows,
                    part_counters=part_counters,
                )

    for hpx in sorted(list(buffers)):
        _flush_hpx_buffer(
            output_root=output_root,
            hpx_level=hp_level,
            hpx=hpx,
            buffers=buffers,
            buffer_rows=buffer_rows,
            part_counters=part_counters,
        )

    elapsed = max(time.perf_counter() - start, 1e-9)
    bytes_written = _parquet_bytes(output_root) - bytes_written_before
    manifest = {
        "build_time": datetime.now(UTC).isoformat(),
        "source_raw_directory": str(gaia_raw_dir(cache_root)),
        "output_directory": str(output_root),
        "file_count_processed": len(raw_files),
        "complete_raw_file_set": limit_files is None,
        "row_count": row_count,
        "selected_columns": GAIA_LITE_COLUMNS,
        "partition_scheme": partitioner.manifest(),
        "package_version": __version__,
        "command_args": {
            "cache_root": str(cache_root),
            "limit_files": limit_files,
            "overwrite": overwrite,
            "hp_level": hp_level,
            "max_rows_per_file": max_rows_per_file,
            "max_buffered_rows": max_buffered_rows,
        },
        "performance": {
            "elapsed_sec": elapsed,
            "files_per_sec": len(raw_files) / elapsed,
            "rows_per_sec": row_count / elapsed,
            "compressed_read_mb_per_sec": bytes_read / 1024 / 1024 / elapsed,
            "parquet_write_mb_per_sec": bytes_written / 1024 / 1024 / elapsed,
        },
        "write_options": {
            "max_rows_per_file": max_rows_per_file,
            "max_buffered_rows": max_buffered_rows,
            "row_group_size": min(max_rows_per_file, 250_000),
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def query_local_gaia_lite(
    s_region: str,
    cache_root: Path,
    max_sources: int = 500,
    g_min: float = 8.0,
    g_max: float = 19.0,
) -> pd.DataFrame:
    if max_sources <= 0:
        return pd.DataFrame(columns=GAIA_LITE_COLUMNS)
    root = gaia_lite_dir(cache_root)
    manifest = _read_manifest(root)
    vertices = polygon_vertices_from_s_region(s_region)
    bounds = bounds_from_vertices(vertices)
    files = sorted(root.glob("hpx_level=*/hpx=*/part-*.parquet"))

    start = time.perf_counter()
    frames: list[pd.DataFrame] = []
    rows_scanned = 0
    for path in files:
        table = pq.read_table(path, columns=GAIA_LITE_COLUMNS)
        rows_scanned += table.num_rows
        if table.num_rows:
            frames.append(table.to_pandas())

    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame(columns=GAIA_LITE_COLUMNS)

    bbox_mask = _bbox_mask(df, bounds) if len(df) else pd.Series(dtype=bool)
    df = df.loc[bbox_mask].copy() if len(df) else df
    rows_after_bbox = len(df)
    if len(df):
        mag = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce")
        df = df.loc[mag.between(g_min, g_max, inclusive="both")].copy()
    rows_after_magnitude = len(df)
    if len(df):
        polygon_mask = _point_in_polygon_mask(df["ra"], df["dec"], vertices, bounds)
        df = df.loc[polygon_mask].copy()
    rows_after_polygon = len(df)
    if len(df):
        df = df.drop_duplicates(subset=["source_id"])
        df = distributed_sample(df, bounds, max_sources=max_sources)
        df = df[GAIA_LITE_COLUMNS].reset_index(drop=True)

    df.attrs["local_gaia_metrics"] = {
        "index_root": str(root),
        "partition_scheme": manifest.get("partition_scheme", {}),
        "partitions_considered": len({path.parent.name for path in files}),
        "partitions_scanned": len(files),
        "rows_scanned": rows_scanned,
        "rows_after_bbox": rows_after_bbox,
        "rows_after_magnitude": rows_after_magnitude,
        "rows_after_polygon": rows_after_polygon,
        "exact_polygon_filter": True,
        "query_wall_time_sec": time.perf_counter() - start,
    }
    return df


def query_local_gaia_lite_duckdb(
    s_region: str,
    cache_root: Path,
    max_sources: int = 500,
    g_min: float = 8.0,
    g_max: float = 19.0,
) -> pd.DataFrame:
    if max_sources <= 0:
        return pd.DataFrame(columns=GAIA_LITE_COLUMNS)
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("duckdb is required for query_local_gaia_lite_duckdb") from exc

    root = gaia_lite_dir(cache_root)
    manifest = _read_manifest(root)
    hpx_level = int(manifest.get("partition_scheme", {}).get("hpx_level", DEFAULT_HPX_LEVEL))
    vertices = polygon_vertices_from_s_region(s_region)
    bounds = bounds_from_vertices(vertices)
    ra_min, ra_max, dec_min, dec_max = bounds
    ra_lo, ra_hi = _unwrap_bounds(ra_min, ra_max)
    glob_path = str(root / "hpx_level=*" / "hpx=*" / "part-*.parquet")
    columns = ", ".join(GAIA_LITE_COLUMNS)
    start = time.perf_counter()

    where_ra = "ra BETWEEN ? AND ?"
    params: list[object] = [g_min, g_max]
    if _bounds_wrap_ra(ra_min, ra_max):
        where_ra = "(ra >= ? OR ra <= ?)"
        params.extend([ra_min % 360.0, ra_max % 360.0])
    else:
        params.extend([ra_lo, ra_hi])
    params.extend([dec_min, dec_max])
    candidate_hpx = candidate_hpx_for_polygon(vertices, bounds, hpx_level=hpx_level)
    hpx_clause = ""
    if candidate_hpx:
        hpx_clause = f"AND hpx IN ({', '.join('?' for _ in candidate_hpx)})"
        params.extend(candidate_hpx)
    params.append(max_sources * 20)
    sql = f"""
        SELECT {columns}
        FROM read_parquet('{glob_path}', hive_partitioning = true)
        WHERE phot_g_mean_mag BETWEEN ? AND ?
          AND {where_ra}
          AND dec BETWEEN ? AND ?
          {hpx_clause}
        ORDER BY phot_g_mean_mag, source_id
        LIMIT ?
    """
    con = duckdb.connect(":memory:")
    try:
        df = con.execute(sql, params).fetchdf()
    finally:
        con.close()

    rows_after_bbox = len(df)
    if len(df):
        polygon_mask = _point_in_polygon_mask(df["ra"], df["dec"], vertices, bounds)
        df = df.loc[polygon_mask].copy()
    rows_after_polygon = len(df)
    if len(df):
        df = df.drop_duplicates(subset=["source_id"])
        df = distributed_sample(df, bounds, max_sources=max_sources)
        df = df[GAIA_LITE_COLUMNS].reset_index(drop=True)
    df.attrs["local_gaia_metrics"] = {
        "index_root": str(root),
        "engine": "duckdb",
        "hpx_level": hpx_level,
        "hpx_candidate_count": len(candidate_hpx),
        "hpx_candidates": candidate_hpx[:32],
        "rows_after_bbox_and_magnitude": rows_after_bbox,
        "rows_after_polygon": rows_after_polygon,
        "exact_polygon_filter": True,
        "query_wall_time_sec": time.perf_counter() - start,
    }
    return df


def candidate_hpx_for_polygon(
    vertices: list[tuple[float, float]],
    bounds: tuple[float, float, float, float],
    hpx_level: int,
) -> list[int]:
    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astropy_healpix import HEALPix
    except ImportError:
        return []

    ra_values = [_unwrap_ra_value(ra, bounds[0], bounds[1]) for ra, _ in vertices]
    dec_values = [dec for _, dec in vertices]
    center_ra = (sum(ra_values) / len(ra_values)) % 360.0
    center_dec = sum(dec_values) / len(dec_values)
    center = SkyCoord(center_ra * u.deg, center_dec * u.deg, frame="icrs")
    corners = SkyCoord([(ra % 360.0) for ra, _ in vertices] * u.deg, dec_values * u.deg, frame="icrs")
    radius = max(center.separation(corners).max(), 0.01 * u.deg)
    hp = HEALPix(nside=2**hpx_level, order="nested", frame=None)
    return [int(value) for value in hp.cone_search_lonlat(center.ra, center.dec, radius)]


def distributed_sample(
    df: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    max_sources: int,
    grid: int = 4,
) -> pd.DataFrame:
    if len(df) <= max_sources:
        return df.sort_values(
            ["phot_g_mean_mag", "source_id"],
            kind="mergesort",
        ).reset_index(drop=True)
    ra_min, ra_max, dec_min, dec_max = bounds
    working = df.copy()
    working["_ra_unwrapped"] = _unwrap_ra_series(working["ra"], ra_min, ra_max)
    ra_lo, ra_hi = _unwrap_bounds(ra_min, ra_max)
    ra_span = max(ra_hi - ra_lo, 1e-9)
    dec_span = max(dec_max - dec_min, 1e-9)
    working["_tile_ra"] = (
        np.floor((working["_ra_unwrapped"] - ra_lo) / ra_span * grid)
        .astype(int)
        .clip(0, grid - 1)
    )
    working["_tile_dec"] = (
        np.floor((working["dec"] - dec_min) / dec_span * grid)
        .astype(int)
        .clip(0, grid - 1)
    )

    per_tile = max(1, max_sources // (grid * grid))
    selected_parts: list[pd.DataFrame] = []
    selected_index: set[Any] = set()
    for _, tile in working.groupby(["_tile_ra", "_tile_dec"], sort=True):
        picked = tile.sort_values(["phot_g_mean_mag", "source_id"], kind="mergesort").head(per_tile)
        selected_parts.append(picked)
        selected_index.update(picked.index.tolist())

    selected = (
        pd.concat(selected_parts, ignore_index=False)
        if selected_parts
        else working.iloc[0:0]
    )
    remaining_slots = max_sources - len(selected)
    if remaining_slots > 0:
        fill = (
            working.loc[~working.index.isin(selected_index)]
            .sort_values(["phot_g_mean_mag", "source_id"], kind="mergesort")
            .head(remaining_slots)
        )
        selected = pd.concat([selected, fill], ignore_index=False)

    return (
        selected.sort_values(
            ["_tile_ra", "_tile_dec", "phot_g_mean_mag", "source_id"],
            kind="mergesort",
        )
        .drop(columns=["_ra_unwrapped", "_tile_ra", "_tile_dec"])
        .head(max_sources)
        .reset_index(drop=True)
    )


def validate_local_gaia_result(
    df: pd.DataFrame,
    s_region: str,
    g_min: float,
    g_max: float,
) -> dict[str, Any]:
    bounds = bounds_from_vertices(polygon_vertices_from_s_region(s_region))
    columns_match = list(df.columns) == GAIA_LITE_COLUMNS
    in_bounds = bool(_bbox_mask(df, bounds).all()) if len(df) else True
    mag = (
        pd.to_numeric(df["phot_g_mean_mag"], errors="coerce")
        if "phot_g_mean_mag" in df
        else pd.Series(dtype=float)
    )
    magnitude_ok = bool(mag.between(g_min, g_max, inclusive="both").all()) if len(df) else True
    unique_sources = bool(df["source_id"].is_unique) if "source_id" in df else False
    return {
        "row_count": len(df),
        "columns_match_remote_contract": columns_match,
        "inside_conservative_bounds": in_bounds,
        "magnitude_cuts_respected": magnitude_ok,
        "source_ids_unique": unique_sources,
        "metrics": df.attrs.get("local_gaia_metrics", {}),
    }


def bounds_from_vertices(vertices: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    ra_values = [ra % 360.0 for ra, _ in vertices]
    dec_values = [dec for _, dec in vertices]
    if max(ra_values) - min(ra_values) > 180.0:
        unwrapped = [ra if ra >= 180.0 else ra + 360.0 for ra in ra_values]
        return min(unwrapped), max(unwrapped), min(dec_values), max(dec_values)
    return min(ra_values), max(ra_values), min(dec_values), max(dec_values)


def polygon_vertices_from_s_region(s_region: str) -> list[tuple[float, float]]:
    from spherex_laser_miner.catalog.gaia import polygon_vertices_from_s_region as parse_vertices

    return parse_vertices(s_region)


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Local Gaia lite manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _ecsv_metadata_rows(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle):
            if line.startswith("#"):
                continue
            header = line.rstrip("\n\r").split(",")
            missing = [col for col in GAIA_LITE_COLUMNS if col not in header]
            if missing:
                raise ValueError(f"{path} CSV header is missing required Gaia columns: {missing}")
            return line_number
    raise ValueError(f"{path} has no CSV data header")


def _bbox_mask(df: pd.DataFrame, bounds: tuple[float, float, float, float]) -> pd.Series:
    if df.empty:
        return pd.Series([], index=df.index, dtype=bool)
    ra_min, ra_max, dec_min, dec_max = bounds
    ra = _unwrap_ra_series(df["ra"], ra_min, ra_max)
    return ra.between(*_unwrap_bounds(ra_min, ra_max), inclusive="both") & df[
        "dec"
    ].between(dec_min, dec_max, inclusive="both")


def _point_in_polygon_mask(
    ra: pd.Series,
    dec: pd.Series,
    vertices: list[tuple[float, float]],
    bounds: tuple[float, float, float, float],
) -> pd.Series:
    if len(ra) == 0:
        return pd.Series([], index=ra.index, dtype=bool)
    polygon = [
        (_unwrap_ra_value(vertex_ra, bounds[0], bounds[1]), vertex_dec)
        for vertex_ra, vertex_dec in vertices
    ]
    x = _unwrap_ra_series(ra, bounds[0], bounds[1]).to_numpy(dtype=float)
    y = dec.to_numpy(dtype=float)
    inside = np.zeros(len(x), dtype=bool)
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-300) + xi
        )
        inside ^= intersects
        j = i
    return pd.Series(inside, index=ra.index)


def _unwrap_bounds(ra_min: float, ra_max: float) -> tuple[float, float]:
    if _bounds_wrap_ra(ra_min, ra_max):
        return ra_min, ra_max
    return ra_min % 360.0, ra_max % 360.0


def _bounds_wrap_ra(ra_min: float, ra_max: float) -> bool:
    return ra_max > 360.0 or ra_min < 0.0 or ra_min > ra_max


def _unwrap_ra_value(ra: float, ra_min: float, ra_max: float) -> float:
    value = ra % 360.0
    if _bounds_wrap_ra(ra_min, ra_max) and value < (ra_min % 360.0):
        return value + 360.0
    return value


def _unwrap_ra_series(ra: pd.Series, ra_min: float, ra_max: float) -> pd.Series:
    values = pd.to_numeric(ra, errors="coerce") % 360.0
    if _bounds_wrap_ra(ra_min, ra_max):
        values = values.where(values >= (ra_min % 360.0), values + 360.0)
    return values


def _parquet_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.glob("hpx_level=*/hpx=*/part-*.parquet"))


def _remove_existing_parquet(root: Path) -> None:
    for pattern in ("hpx_level=*/hpx=*/part-*.parquet", "hp_level=*/hp=*/part-*.parquet"):
        for path in root.glob(pattern):
            path.unlink()
    for pattern in ("hpx_level=*/hpx=*", "hp_level=*/hp=*"):
        for path in sorted(root.glob(pattern), reverse=True):
            path.rmdir()
    for pattern in ("hpx_level=*", "hp_level=*"):
        for path in sorted(root.glob(pattern), reverse=True):
            path.rmdir()
    (root / "manifest.json").unlink(missing_ok=True)


def _flush_hpx_buffer(
    *,
    output_root: Path,
    hpx_level: int,
    hpx: int,
    buffers: dict[int, list[pd.DataFrame]],
    buffer_rows: dict[int, int],
    part_counters: dict[int, int],
) -> int:
    parts = buffers.pop(hpx, [])
    rows = buffer_rows.pop(hpx, 0)
    if not parts:
        return 0
    frame = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    part_idx = part_counters.get(hpx, 0)
    part_counters[hpx] = part_idx + 1
    part_dir = output_root / f"hpx_level={hpx_level}" / f"hpx={hpx}"
    part_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False),
        part_dir / f"part-{part_idx:06d}.parquet",
        compression="zstd",
        row_group_size=min(max(len(frame), 1), 250_000),
    )
    return rows
