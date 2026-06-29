from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS


FITS_SUFFIXES = (".fits", ".fits.gz")


@dataclass(frozen=True)
class StageTiming:
    stage: str
    wall_time_sec: float
    rows_out: int = 0
    bytes_out: int = 0


def build_frame_manifest(
    *,
    input_roots: list[Path],
    output_path: Path,
    limit: int | None = None,
    campaign_id: str = "local_manifest",
    read_headers: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    timings: list[StageTiming] = []

    t0 = time.perf_counter()
    paths = list(iter_fits_paths(input_roots=input_roots, limit=limit))
    timings.append(StageTiming("discover_fits", time.perf_counter() - t0, rows_out=len(paths)))

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for index, path in enumerate(paths):
        rows.append(_manifest_row(index=index, path=path, read_headers=read_headers))
    timings.append(StageTiming("read_headers", time.perf_counter() - t0, rows_out=len(rows)))

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    df.to_parquet(output_path, index=False)
    parquet_wall = time.perf_counter() - t0
    out_bytes = output_path.stat().st_size if output_path.exists() else 0
    timings.append(StageTiming("write_manifest", parquet_wall, rows_out=len(df), bytes_out=out_bytes))

    summary = {
        "campaign_id": campaign_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_roots": [str(root) for root in input_roots],
        "output_path": str(output_path),
        "frame_count": int(len(df)),
        "fits_total_bytes": int(df["file_size_bytes"].sum()) if "file_size_bytes" in df and len(df) else 0,
        "total_wall_sec": time.perf_counter() - started,
        "stages": [timing.__dict__ for timing in timings],
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    profile_path = output_path.with_suffix(".profile.json")
    total = max(summary["total_wall_sec"], 1e-12)
    profile = {
        "campaign_id": campaign_id,
        "created_utc": summary["created_utc"],
        "rows": [
            {
                "stage": timing.stage,
                "function_or_script": "luxquarry_allsky_engine.manifest.build_frame_manifest",
                "wall_time_sec": timing.wall_time_sec,
                "wall_time_pct": 100.0 * timing.wall_time_sec / total,
                "cpu_time_sec": timing.wall_time_sec,
                "gpu_time_sec": 0.0,
                "io_wait_sec": None,
                "call_count": 1,
                "rows_in": None,
                "rows_out": timing.rows_out,
                "bytes_in": None,
                "bytes_out": timing.bytes_out,
                "backend": "python_astropy_pandas",
            }
            for timing in timings
        ],
    }
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def iter_fits_paths(*, input_roots: list[Path], limit: int | None = None) -> Iterable[Path]:
    emitted = 0
    for root in input_roots:
        if root.is_file() and _is_fits(root):
            yield root
            emitted += 1
            if limit is not None and emitted >= limit:
                return
            continue
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or not _is_fits(path):
                continue
            yield path
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def _manifest_row(*, index: int, path: Path, read_headers: bool) -> dict[str, Any]:
    stat = path.stat()
    row: dict[str, Any] = {
        "frame_group_id": f"fg_{index:08d}",
        "frame_index": index,
        "path": str(path),
        "file_name": path.name,
        "image_id": _image_id_from_path(path),
        "file_size_bytes": int(stat.st_size),
        "mtime_unix": float(stat.st_mtime),
        "planning_period": None,
        "processing_version": None,
        "exposure_id": None,
        "frame_in_exposure": None,
        "detector": _detector_from_name(path.name),
        "release": "qr2",
        "naxis1": None,
        "naxis2": None,
        "ra_center_deg": None,
        "dec_center_deg": None,
        "ra_min_deg": None,
        "ra_max_deg": None,
        "dec_min_deg": None,
        "dec_max_deg": None,
        "obs_mid_time": None,
        "observation_id": None,
        "header_error": None,
    }
    row.update(_path_fields(path))
    row.update(_filename_fields(path.name))
    if not read_headers:
        return row
    try:
        with fits.open(path, memmap=False, lazy_load_hdus=True) as hdul:
            header = _science_header(hdul)
            row.update(_header_fields(header))
            row.update(_footprint_fields(header))
    except Exception as exc:
        row["header_error"] = f"{type(exc).__name__}: {exc}"
    return row


def _science_header(hdul: fits.HDUList) -> fits.Header:
    for name in ("IMAGE", "PRIMARY"):
        try:
            hdu = hdul[name]
        except Exception:
            continue
        if hdu.header:
            return hdu.header
    return hdul[0].header


def _header_fields(header: fits.Header) -> dict[str, Any]:
    return {
        "naxis1": _int_or_none(header.get("NAXIS1")),
        "naxis2": _int_or_none(header.get("NAXIS2")),
        "obs_mid_time": _first_header(header, "DATE-OBS", "MJD-OBS", "OBSMID", "OBS_MID"),
        "observation_id": _first_header(header, "OBS_ID", "OBSID", "OBSERVID"),
    }


def _footprint_fields(header: fits.Header) -> dict[str, Any]:
    naxis1 = _int_or_none(header.get("NAXIS1"))
    naxis2 = _int_or_none(header.get("NAXIS2"))
    if not naxis1 or not naxis2:
        return {}
    pixels = [
        [1.0, 1.0],
        [float(naxis1), 1.0],
        [float(naxis1), float(naxis2)],
        [1.0, float(naxis2)],
        [float(naxis1) / 2.0, float(naxis2) / 2.0],
    ]
    try:
        wcs = WCS(header)
        world = wcs.all_pix2world(pixels, 1)
    except Exception as exc:
        return {"header_error": f"WCS: {type(exc).__name__}: {exc}"}
    ra = [float(item[0]) % 360.0 for item in world if len(item) >= 2]
    dec = [float(item[1]) for item in world if len(item) >= 2]
    if not ra or not dec:
        return {}
    return {
        "ra_center_deg": ra[-1],
        "dec_center_deg": dec[-1],
        "ra_min_deg": min(ra),
        "ra_max_deg": max(ra),
        "dec_min_deg": min(dec),
        "dec_max_deg": max(dec),
    }


def _is_fits(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in FITS_SUFFIXES)


def _image_id_from_path(path: Path) -> str:
    name = path.name
    for suffix in FITS_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _path_fields(path: Path) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    parts = path.parts
    for index, part in enumerate(parts):
        if re.fullmatch(r"\d{4}W\d{2}_[0-9A-Z]+", part):
            fields["planning_period"] = part
            if index + 1 < len(parts) and parts[index + 1].startswith("l2b-"):
                fields["processing_version"] = parts[index + 1]
            if index + 2 < len(parts) and parts[index + 2].isdigit():
                fields["detector_dir"] = int(parts[index + 2])
            break
    return fields


def _filename_fields(name: str) -> dict[str, Any]:
    match = re.match(
        r"level2_(?P<planning>\d{4}W\d{2}_[0-9A-Z]+)_(?P<exposure>\d+)_(?P<frame>[0-9])D(?P<detector>[1-6])_",
        name,
    )
    if not match:
        return {}
    return {
        "planning_period": match.group("planning"),
        "exposure_id": match.group("exposure"),
        "frame_in_exposure": int(match.group("frame")),
        "detector": int(match.group("detector")),
    }


def _detector_from_name(name: str) -> int | None:
    match = re.search(r"_1D([1-6])_", name)
    if match:
        return int(match.group(1))
    match = re.search(r"D([1-6])", name)
    return int(match.group(1)) if match else None


def _first_header(header: fits.Header, *keys: str) -> Any:
    for key in keys:
        if key in header:
            value = header.get(key)
            if value is not None:
                return value
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None
