#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy import units as u
from astropy_healpix import HEALPix


PSC_COLUMNS = [
    "ra",
    "dec",
    "err_maj",
    "err_min",
    "err_ang",
    "designation",
    "j_m",
    "j_cmsig",
    "j_msigcom",
    "j_snr",
    "h_m",
    "h_cmsig",
    "h_msigcom",
    "h_snr",
    "k_m",
    "k_cmsig",
    "k_msigcom",
    "k_snr",
    "ph_qual",
    "rd_flg",
    "bl_flg",
    "cc_flg",
    "ndet",
    "prox",
    "pxpa",
    "pxcntr",
    "gal_contam",
    "mp_flg",
    "pts_key",
    "hemis",
    "date",
    "scan",
    "glon",
    "glat",
    "x_scan",
    "jdate",
    "j_psfchi",
    "h_psfchi",
    "k_psfchi",
    "j_m_stdap",
    "j_msig_stdap",
    "h_m_stdap",
    "h_msig_stdap",
    "k_m_stdap",
    "k_msig_stdap",
    "dist_edge_ns",
    "dist_edge_ew",
    "dist_edge_flg",
    "dup_src",
    "use_src",
    "a",
    "dist_opt",
    "phi_opt",
    "b_m_opt",
    "vr_m_opt",
    "nopt_mchs",
    "ext_key",
    "scan_key",
    "coadd_key",
    "coadd",
]

KEEP_COLUMNS = [
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
    "designation",
    "ra",
    "dec",
    "err_maj",
    "err_min",
    "err_ang",
    "j_m",
    "j_cmsig",
    "j_msigcom",
    "j_snr",
    "h_m",
    "h_cmsig",
    "h_msigcom",
    "h_snr",
    "k_m",
    "k_cmsig",
    "k_msigcom",
    "k_snr",
    "ph_qual",
    "rd_flg",
    "bl_flg",
    "cc_flg",
    "ndet",
    "prox",
    "pxpa",
    "pxcntr",
    "gal_contam",
    "mp_flg",
    "pts_key",
    "hemis",
    "date",
    "scan",
    "glon",
    "glat",
    "x_scan",
    "jdate",
    "j_psfchi",
    "h_psfchi",
    "k_psfchi",
    "j_m_stdap",
    "j_msig_stdap",
    "h_m_stdap",
    "h_msig_stdap",
    "k_m_stdap",
    "k_msig_stdap",
    "dist_edge_ns",
    "dist_edge_ew",
    "dist_edge_flg",
    "dup_src",
    "use_src",
    "ext_key",
    "scan_key",
    "coadd_key",
    "coadd",
]

FLOAT_COLUMNS = [
    "ra",
    "dec",
    "err_maj",
    "err_min",
    "j_m",
    "j_cmsig",
    "j_msigcom",
    "j_snr",
    "h_m",
    "h_cmsig",
    "h_msigcom",
    "h_snr",
    "k_m",
    "k_cmsig",
    "k_msigcom",
    "k_snr",
    "prox",
    "pxpa",
    "glon",
    "glat",
    "x_scan",
    "jdate",
    "j_psfchi",
    "h_psfchi",
    "k_psfchi",
    "j_m_stdap",
    "j_msig_stdap",
    "h_m_stdap",
    "h_msig_stdap",
    "k_m_stdap",
    "k_msig_stdap",
    "dist_edge_ns",
    "dist_edge_ew",
]

INT_COLUMNS = [
    "err_ang",
    "pxcntr",
    "gal_contam",
    "mp_flg",
    "pts_key",
    "scan",
    "dup_src",
    "use_src",
    "nopt_mchs",
    "ext_key",
    "scan_key",
    "coadd_key",
    "coadd",
]

DEFAULT_CACHE_ROOT = Path("/mnt/niroseti/spherex_cache")
DEFAULT_HPX_LEVEL = 5
DEFAULT_CHUNK_ROWS = 500_000
DEFAULT_MAX_ROWS_PER_FILE = 500_000
DEFAULT_MAX_BUFFERED_ROWS = 4_000_000


@dataclass
class WriterState:
    part_counters: dict[int, int]
    buffers: dict[int, list[pd.DataFrame]]
    buffer_rows: dict[int, int]
    total_buffered_rows: int = 0
    parquet_files_written: int = 0
    parquet_rows_written: int = 0


def raw_psc_dir(cache_root: Path) -> Path:
    return cache_root / "2mass" / "raw_download" / "psc"


def output_parquet_dir(cache_root: Path, name: str) -> Path:
    return cache_root / "2mass" / "parquet" / name


def main() -> int:
    args = parse_args()
    cache_root = args.cache_root.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve() if args.raw_root else raw_psc_dir(cache_root)
    output_root = args.output.expanduser().resolve() if args.output else output_parquet_dir(cache_root, args.dataset_name)
    status_path = args.status_path.expanduser().resolve() if args.status_path else output_root / "build_status.json"

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    if output_root.exists() and args.resume:
        raise SystemExit("--resume is not implemented safely yet; use --overwrite for a clean rebuild")
    if output_root.exists() and not args.resume:
        raise SystemExit(f"{output_root} already exists; pass --overwrite or --resume")
    output_root.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(raw_root.glob(args.raw_glob))
    if args.limit_files is not None:
        raw_files = raw_files[: args.limit_files]
    if not raw_files:
        raise SystemExit(f"No raw 2MASS PSC files matched {raw_root / args.raw_glob}")
    if args.workers > 1:
        if args.limit_rows is not None:
            raise SystemExit("--limit-rows is only supported for single-worker builds")
        return build_parallel(args, raw_root, output_root, status_path, raw_files)

    hpx = HEALPix(nside=2**args.hpx_level, order="nested", frame=None)
    state = WriterState(part_counters={}, buffers={}, buffer_rows={})
    start = time.perf_counter()
    row_count = 0
    bytes_read = 0
    stopped_by_limit_rows = False

    print(
        f"building 2MASS parquet: files={len(raw_files)} hpx_level={args.hpx_level} "
        f"chunk_rows={args.chunk_rows} output={output_root}",
        flush=True,
    )
    write_status(
        status_path,
        phase="starting",
        start=start,
        raw_root=raw_root,
        output_root=output_root,
        file_index=0,
        file_count=len(raw_files),
        row_count=row_count,
        bytes_read=bytes_read,
        state=state,
        current_file=None,
        hpx_level=args.hpx_level,
    )

    for file_index, raw_file in enumerate(raw_files, start=1):
        file_size = raw_file.stat().st_size
        bytes_read += file_size
        print(f"[{file_index}/{len(raw_files)}] reading {raw_file.name} ({file_size / 1024 / 1024:.1f} MiB gz)", flush=True)
        chunk_iter = pd.read_csv(
            raw_file,
            sep="|",
            names=PSC_COLUMNS,
            header=None,
            na_values=[r"\N"],
            keep_default_na=True,
            chunksize=args.chunk_rows,
            low_memory=False,
        )
        for chunk_index, chunk in enumerate(chunk_iter, start=1):
            if args.limit_rows is not None:
                remaining = args.limit_rows - row_count
                if remaining <= 0:
                    stopped_by_limit_rows = True
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining].copy()
                    stopped_by_limit_rows = True
            clean = normalize_chunk(chunk, primary_band=args.primary_band)
            clean["hpx"] = hpx.lonlat_to_healpix(
                clean["ra_deg"].to_numpy(dtype=float) * u.deg,
                clean["dec_deg"].to_numpy(dtype=float) * u.deg,
            ).astype(np.int64)
            row_count += len(clean)
            append_partitioned(
                output_root=output_root,
                hpx_level=args.hpx_level,
                frame=clean,
                state=state,
                max_rows_per_file=args.max_rows_per_file,
                max_buffered_rows=args.max_buffered_rows,
                compression=args.compression,
            )
            elapsed = max(time.perf_counter() - start, 1e-9)
            print(
                f"  chunk={chunk_index} rows={row_count:,} "
                f"rate={row_count / elapsed:,.0f} rows/s "
                f"buffered={state.total_buffered_rows:,} parquet_files={state.parquet_files_written}",
                flush=True,
            )
            write_status(
                status_path,
                phase="running",
                start=start,
                raw_root=raw_root,
                output_root=output_root,
                file_index=file_index,
                file_count=len(raw_files),
                row_count=row_count,
                bytes_read=bytes_read,
                state=state,
                current_file=str(raw_file),
                hpx_level=args.hpx_level,
            )
            if stopped_by_limit_rows:
                break
        if stopped_by_limit_rows:
            break

    for hpx_id in sorted(list(state.buffers)):
        flush_hpx_buffer(
            output_root=output_root,
            hpx_level=args.hpx_level,
            hpx_id=hpx_id,
            state=state,
            compression=args.compression,
            max_rows_per_file=args.max_rows_per_file,
        )

    elapsed = max(time.perf_counter() - start, 1e-9)
    manifest = {
        "build_time": datetime.now(UTC).isoformat(),
        "source": {
            "catalog": "2MASS All-Sky Point Source Catalog",
            "raw_directory": str(raw_root),
            "raw_glob": args.raw_glob,
            "raw_file_count_processed": file_index if "file_index" in locals() else 0,
            "complete_raw_file_set": args.limit_files is None and args.limit_rows is None,
            "row_format": "pipe-delimited gzip ASCII, IRSA null token \\N",
            "official_schema_columns": PSC_COLUMNS,
        },
        "output_directory": str(output_root),
        "status_path": str(status_path),
        "row_count": row_count,
        "selected_columns": KEEP_COLUMNS,
        "canonical_columns": [
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
        ],
        "partition_scheme": {
            "name": "coordinate_nested_healpix",
            "hpx_level": args.hpx_level,
            "nside": 2**args.hpx_level,
            "order": "nested",
            "hive_columns": ["hpx_level", "hpx"],
        },
        "photometry": {
            "primary_band": args.primary_band,
            "available_bands": ["J", "H", "Ks"],
            "magnitude_note": "2MASS has band-specific J/H/Ks magnitudes rather than Gaia G; mag_primary is a convenience copy for generic target selection.",
        },
        "performance": {
            "elapsed_sec": elapsed,
            "rows_per_sec": row_count / elapsed,
            "compressed_read_mb_per_sec": bytes_read / 1024 / 1024 / elapsed,
            "parquet_files_written": state.parquet_files_written,
            "parquet_rows_written": state.parquet_rows_written,
        },
        "command_args": jsonable(vars(args)),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_status(
        status_path,
        phase="complete",
        start=start,
        raw_root=raw_root,
        output_root=output_root,
        file_index=len(raw_files),
        file_count=len(raw_files),
        row_count=row_count,
        bytes_read=bytes_read,
        state=state,
        current_file=None,
        hpx_level=args.hpx_level,
        extra={"manifest_path": str(output_root / "manifest.json")},
    )
    print(
        f"done: rows={row_count:,} files={state.parquet_files_written:,} "
        f"elapsed={elapsed:.1f}s rate={row_count / elapsed:,.0f} rows/s",
        flush=True,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HEALPix Parquet shards from raw 2MASS PSC gzip files.")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--raw-glob", default="psc_*.gz")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset-name", default="psc_lite")
    parser.add_argument("--status-path", type=Path)
    parser.add_argument("--hpx-level", type=int, default=DEFAULT_HPX_LEVEL)
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--max-rows-per-file", type=int, default=DEFAULT_MAX_ROWS_PER_FILE)
    parser.add_argument("--max-buffered-rows", type=int, default=DEFAULT_MAX_BUFFERED_ROWS)
    parser.add_argument("--limit-files", type=int)
    parser.add_argument("--limit-rows", type=int)
    parser.add_argument("--primary-band", choices=["J", "H", "Ks"], default="Ks")
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--workers", type=int, default=1, help="Raw gzip files to process in parallel.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reserved for future resumable builds; currently rejected.")
    return parser.parse_args()


def build_parallel(
    args: argparse.Namespace,
    raw_root: Path,
    output_root: Path,
    status_path: Path,
    raw_files: list[Path],
) -> int:
    workers = max(1, int(args.workers))
    start = time.perf_counter()
    status_root = output_root / "_worker_status"
    status_root.mkdir(parents=True, exist_ok=True)
    print(
        f"building 2MASS parquet in parallel: files={len(raw_files)} workers={workers} "
        f"hpx_level={args.hpx_level} output={output_root}",
        flush=True,
    )
    write_parallel_status(
        status_path=status_path,
        phase="starting",
        start=start,
        raw_root=raw_root,
        output_root=output_root,
        raw_files=raw_files,
        results=[],
        workers=workers,
        hpx_level=args.hpx_level,
    )

    worker_args = {
        "hpx_level": args.hpx_level,
        "chunk_rows": args.chunk_rows,
        "max_rows_per_file": args.max_rows_per_file,
        "max_buffered_rows": args.max_buffered_rows,
        "primary_band": args.primary_band,
        "compression": args.compression,
    }
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_raw_file_worker, raw_file, output_root, status_root, index, len(raw_files), worker_args): raw_file
            for index, raw_file in enumerate(raw_files, start=1)
        }
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=2.0, return_when=FIRST_COMPLETED)
            for future in done:
                raw_file = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    write_parallel_status(
                        status_path=status_path,
                        phase="failed",
                        start=start,
                        raw_root=raw_root,
                        output_root=output_root,
                        raw_files=raw_files,
                        results=results,
                        workers=workers,
                        hpx_level=args.hpx_level,
                        extra={"failed_file": str(raw_file), "error": f"{type(exc).__name__}: {exc}"},
                    )
                    raise
                results.append(result)
                elapsed = max(time.perf_counter() - start, 1e-9)
                print(
                    f"[done {len(results)}/{len(raw_files)}] {Path(result['raw_file']).name} "
                    f"rows={int(result['row_count']):,} aggregate_rate={sum(int(r['row_count']) for r in results) / elapsed:,.0f} rows/s",
                    flush=True,
                )
            write_parallel_status(
                status_path=status_path,
                phase="running",
                start=start,
                raw_root=raw_root,
                output_root=output_root,
                raw_files=raw_files,
                results=results,
                workers=workers,
                hpx_level=args.hpx_level,
            )

    elapsed = max(time.perf_counter() - start, 1e-9)
    row_count = sum(int(result["row_count"]) for result in results)
    bytes_read = sum(int(result["compressed_bytes_read"]) for result in results)
    parquet_files_written = sum(int(result["parquet_files_written"]) for result in results)
    parquet_rows_written = sum(int(result["parquet_rows_written"]) for result in results)
    manifest = {
        "build_time": datetime.now(UTC).isoformat(),
        "source": {
            "catalog": "2MASS All-Sky Point Source Catalog",
            "raw_directory": str(raw_root),
            "raw_glob": args.raw_glob,
            "raw_file_count_processed": len(results),
            "complete_raw_file_set": args.limit_files is None,
            "row_format": "pipe-delimited gzip ASCII, IRSA null token \\N",
            "official_schema_columns": PSC_COLUMNS,
        },
        "output_directory": str(output_root),
        "status_path": str(status_path),
        "row_count": row_count,
        "selected_columns": KEEP_COLUMNS,
        "canonical_columns": [
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
        ],
        "partition_scheme": {
            "name": "coordinate_nested_healpix",
            "hpx_level": args.hpx_level,
            "nside": 2**args.hpx_level,
            "order": "nested",
            "hive_columns": ["hpx_level", "hpx"],
            "parallel_write_note": "Each worker writes uniquely named Parquet files into shared HEALPix directories.",
        },
        "photometry": {
            "primary_band": args.primary_band,
            "available_bands": ["J", "H", "Ks"],
            "magnitude_note": "2MASS has band-specific J/H/Ks magnitudes rather than Gaia G; mag_primary is a convenience copy for generic target selection.",
        },
        "performance": {
            "elapsed_sec": elapsed,
            "rows_per_sec": row_count / elapsed,
            "compressed_read_mb_per_sec": bytes_read / 1024 / 1024 / elapsed,
            "parquet_files_written": parquet_files_written,
            "parquet_rows_written": parquet_rows_written,
            "workers": workers,
        },
        "worker_results": results,
        "command_args": jsonable(vars(args)),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_parallel_status(
        status_path=status_path,
        phase="complete",
        start=start,
        raw_root=raw_root,
        output_root=output_root,
        raw_files=raw_files,
        results=results,
        workers=workers,
        hpx_level=args.hpx_level,
        extra={"manifest_path": str(output_root / "manifest.json")},
    )
    print(
        f"done: rows={row_count:,} files={parquet_files_written:,} "
        f"elapsed={elapsed:.1f}s rate={row_count / elapsed:,.0f} rows/s workers={workers}",
        flush=True,
    )
    return 0


def process_raw_file_worker(
    raw_file: Path,
    output_root: Path,
    status_root: Path,
    file_index: int,
    file_count: int,
    args: dict[str, Any],
) -> dict[str, Any]:
    start = time.perf_counter()
    raw_file = Path(raw_file)
    output_root = Path(output_root)
    status_path = Path(status_root) / f"{raw_file.stem}.json"
    hpx = HEALPix(nside=2 ** int(args["hpx_level"]), order="nested", frame=None)
    state = WriterState(part_counters={}, buffers={}, buffer_rows={})
    row_count = 0
    compressed_bytes_read = raw_file.stat().st_size
    part_prefix = raw_file.stem
    write_status(
        status_path,
        phase="running",
        start=start,
        raw_root=raw_file.parent,
        output_root=output_root,
        file_index=file_index,
        file_count=file_count,
        row_count=row_count,
        bytes_read=compressed_bytes_read,
        state=state,
        current_file=str(raw_file),
        hpx_level=int(args["hpx_level"]),
        extra={"worker_file": raw_file.name},
    )
    chunk_iter = pd.read_csv(
        raw_file,
        sep="|",
        names=PSC_COLUMNS,
        header=None,
        na_values=[r"\N"],
        keep_default_na=True,
        chunksize=int(args["chunk_rows"]),
        low_memory=False,
    )
    for chunk_index, chunk in enumerate(chunk_iter, start=1):
        clean = normalize_chunk(chunk, primary_band=str(args["primary_band"]))
        clean["hpx"] = hpx.lonlat_to_healpix(
            clean["ra_deg"].to_numpy(dtype=float) * u.deg,
            clean["dec_deg"].to_numpy(dtype=float) * u.deg,
        ).astype(np.int64)
        row_count += len(clean)
        append_partitioned(
            output_root=output_root,
            hpx_level=int(args["hpx_level"]),
            frame=clean,
            state=state,
            max_rows_per_file=int(args["max_rows_per_file"]),
            max_buffered_rows=int(args["max_buffered_rows"]),
            compression=str(args["compression"]),
            part_prefix=part_prefix,
        )
        write_status(
            status_path,
            phase="running",
            start=start,
            raw_root=raw_file.parent,
            output_root=output_root,
            file_index=file_index,
            file_count=file_count,
            row_count=row_count,
            bytes_read=compressed_bytes_read,
            state=state,
            current_file=str(raw_file),
            hpx_level=int(args["hpx_level"]),
            extra={"worker_file": raw_file.name, "chunk_index": chunk_index},
        )
    for hpx_id in sorted(list(state.buffers)):
        flush_hpx_buffer(
            output_root=output_root,
            hpx_level=int(args["hpx_level"]),
            hpx_id=hpx_id,
            state=state,
            compression=str(args["compression"]),
            max_rows_per_file=int(args["max_rows_per_file"]),
            part_prefix=part_prefix,
        )
    elapsed = max(time.perf_counter() - start, 1e-9)
    result = {
        "raw_file": str(raw_file),
        "file_index": file_index,
        "row_count": row_count,
        "compressed_bytes_read": compressed_bytes_read,
        "parquet_files_written": state.parquet_files_written,
        "parquet_rows_written": state.parquet_rows_written,
        "elapsed_sec": elapsed,
        "rows_per_sec": row_count / elapsed,
    }
    write_status(
        status_path,
        phase="complete",
        start=start,
        raw_root=raw_file.parent,
        output_root=output_root,
        file_index=file_index,
        file_count=file_count,
        row_count=row_count,
        bytes_read=compressed_bytes_read,
        state=state,
        current_file=None,
        hpx_level=int(args["hpx_level"]),
        extra={"worker_file": raw_file.name, "result": result},
    )
    return result


def normalize_chunk(chunk: pd.DataFrame, primary_band: str) -> pd.DataFrame:
    chunk = chunk.copy()
    for col in FLOAT_COLUMNS:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("float64")
    for col in INT_COLUMNS:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("Int64")

    designation = chunk["designation"].astype("string").str.strip()
    chunk["designation"] = designation
    chunk["target_id"] = "twomass_psc_" + designation.fillna("unknown")
    chunk["target_type"] = "2mass_psc"
    chunk["source_id"] = designation
    chunk["source_catalog"] = "2mass_psc"
    chunk["source_catalog_id"] = designation
    chunk["object_name"] = "2MASS J" + designation.fillna("unknown")
    chunk["ra_deg"] = chunk["ra"]
    chunk["dec_deg"] = chunk["dec"]
    chunk["ra_reference_deg"] = chunk["ra"]
    chunk["dec_reference_deg"] = chunk["dec"]
    chunk["reference_epoch_yr"] = 2000.0
    chunk["pmra_masyr"] = np.nan
    chunk["pmdec_masyr"] = np.nan
    chunk["parallax_mas"] = np.nan
    mag_col = {"J": "j_m", "H": "h_m", "Ks": "k_m"}[primary_band]
    chunk["mag_primary"] = chunk[mag_col]
    chunk["mag_primary_band"] = primary_band
    chunk["priority_score"] = 100.0 - chunk["mag_primary"].clip(upper=99.0)
    chunk["target_filter_flags"] = "2mass_psc_catalog"

    clean = chunk[KEEP_COLUMNS].copy()
    clean = clean.loc[clean["ra_deg"].between(0.0, 360.0, inclusive="left")]
    clean = clean.loc[clean["dec_deg"].between(-90.0, 90.0, inclusive="both")]
    return clean.reset_index(drop=True)


def append_partitioned(
    output_root: Path,
    hpx_level: int,
    frame: pd.DataFrame,
    state: WriterState,
    max_rows_per_file: int,
    max_buffered_rows: int,
    compression: str,
    part_prefix: str | None = None,
) -> None:
    for hpx_id, part in frame.groupby("hpx", sort=True):
        cell = int(hpx_id)
        clean = part.drop(columns=["hpx"])
        state.buffers.setdefault(cell, []).append(clean)
        rows = len(clean)
        state.buffer_rows[cell] = state.buffer_rows.get(cell, 0) + rows
        state.total_buffered_rows += rows
        if state.buffer_rows[cell] >= max_rows_per_file:
            flush_hpx_buffer(output_root, hpx_level, cell, state, compression, max_rows_per_file, part_prefix=part_prefix)

    while state.total_buffered_rows > max_buffered_rows and state.buffer_rows:
        largest = max(state.buffer_rows, key=state.buffer_rows.get)
        flush_hpx_buffer(output_root, hpx_level, largest, state, compression, max_rows_per_file, part_prefix=part_prefix)


def flush_hpx_buffer(
    output_root: Path,
    hpx_level: int,
    hpx_id: int,
    state: WriterState,
    compression: str,
    max_rows_per_file: int,
    part_prefix: str | None = None,
) -> None:
    pieces = state.buffers.pop(hpx_id, [])
    rows = state.buffer_rows.pop(hpx_id, 0)
    if not pieces or rows <= 0:
        return
    frame = pd.concat(pieces, ignore_index=True)
    part_no = state.part_counters.get(hpx_id, 0)
    state.part_counters[hpx_id] = part_no + 1
    directory = output_root / f"hpx_level={hpx_level}" / f"hpx={hpx_id}"
    directory.mkdir(parents=True, exist_ok=True)
    if part_prefix:
        path = directory / f"part-{part_prefix}-{part_no:05d}.parquet"
    else:
        path = directory / f"part-{part_no:05d}.parquet"
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(
        table,
        path,
        compression=compression,
        row_group_size=min(max_rows_per_file, 250_000),
    )
    state.total_buffered_rows -= rows
    state.parquet_files_written += 1
    state.parquet_rows_written += rows


def write_status(
    path: Path,
    phase: str,
    start: float,
    raw_root: Path,
    output_root: Path,
    file_index: int,
    file_count: int,
    row_count: int,
    bytes_read: int,
    state: WriterState,
    current_file: str | None,
    hpx_level: int,
    extra: dict[str, Any] | None = None,
) -> None:
    elapsed = max(time.perf_counter() - start, 1e-9)
    payload: dict[str, Any] = {
        "updated_at": datetime.now(UTC).isoformat(),
        "phase": phase,
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "current_file": current_file,
        "file_index": file_index,
        "file_count": file_count,
        "file_progress_fraction": file_index / file_count if file_count else 0.0,
        "row_count": row_count,
        "rows_per_sec": row_count / elapsed,
        "compressed_bytes_read": bytes_read,
        "compressed_mib_per_sec": bytes_read / 1024 / 1024 / elapsed,
        "buffered_rows": state.total_buffered_rows,
        "open_hpx_buffers": len(state.buffers),
        "parquet_files_written": state.parquet_files_written,
        "parquet_rows_written": state.parquet_rows_written,
        "hpx_level": hpx_level,
        "nside": 2**hpx_level,
        "elapsed_sec": elapsed,
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def write_parallel_status(
    *,
    status_path: Path,
    phase: str,
    start: float,
    raw_root: Path,
    output_root: Path,
    raw_files: list[Path],
    results: list[dict[str, Any]],
    workers: int,
    hpx_level: int,
    extra: dict[str, Any] | None = None,
) -> None:
    elapsed = max(time.perf_counter() - start, 1e-9)
    worker_statuses = read_worker_statuses(output_root / "_worker_status")
    done_files = {str(result.get("raw_file")) for result in results}
    active = [
        status
        for status in worker_statuses
        if status.get("phase") == "running" and str(status.get("current_file") or "") not in done_files
    ]
    worker_rows = sum(int(status.get("row_count") or 0) for status in active)
    done_rows = sum(int(result.get("row_count") or 0) for result in results)
    worker_bytes = sum(int(status.get("compressed_bytes_read") or 0) for status in active)
    done_bytes = sum(int(result.get("compressed_bytes_read") or 0) for result in results)
    parquet_files = sum(int(result.get("parquet_files_written") or 0) for result in results) + sum(
        int(status.get("parquet_files_written") or 0) for status in active
    )
    parquet_rows = sum(int(result.get("parquet_rows_written") or 0) for result in results) + sum(
        int(status.get("parquet_rows_written") or 0) for status in active
    )
    row_count = done_rows + worker_rows
    bytes_read = done_bytes + worker_bytes
    payload: dict[str, Any] = {
        "updated_at": datetime.now(UTC).isoformat(),
        "phase": phase,
        "mode": "parallel",
        "workers": workers,
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "file_count": len(raw_files),
        "files_done": len(results),
        "files_active": len(active),
        "files_waiting": max(0, len(raw_files) - len(results) - len(active)),
        "file_progress_fraction": len(results) / len(raw_files) if raw_files else 0.0,
        "row_count": row_count,
        "done_row_count": done_rows,
        "active_row_count": worker_rows,
        "rows_per_sec": row_count / elapsed,
        "compressed_bytes_read": bytes_read,
        "compressed_mib_per_sec": bytes_read / 1024 / 1024 / elapsed,
        "parquet_files_written": parquet_files,
        "parquet_rows_written": parquet_rows,
        "hpx_level": hpx_level,
        "nside": 2**hpx_level,
        "elapsed_sec": elapsed,
        "active_files": [
            {
                "worker_file": status.get("worker_file"),
                "row_count": status.get("row_count"),
                "rows_per_sec": status.get("rows_per_sec"),
                "chunk_index": status.get("chunk_index"),
                "buffered_rows": status.get("buffered_rows"),
                "parquet_files_written": status.get("parquet_files_written"),
            }
            for status in sorted(active, key=lambda item: str(item.get("worker_file") or ""))
        ],
        "recent_done_files": [
            {
                "raw_file": Path(str(result.get("raw_file"))).name,
                "row_count": result.get("row_count"),
                "elapsed_sec": result.get("elapsed_sec"),
                "rows_per_sec": result.get("rows_per_sec"),
            }
            for result in results[-10:]
        ],
    }
    if extra:
        payload.update(extra)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = status_path.with_suffix(status_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(status_path)


def read_worker_statuses(status_root: Path) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for path in sorted(status_root.glob("*.json")):
        try:
            statuses.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return statuses


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
