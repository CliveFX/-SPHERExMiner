from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import typer

from spherex_laser_miner.cache import ensure_cache_dirs
from spherex_laser_miner.catalog.local_gaia_lite import (
    GAIA_LITE_COLUMNS,
    build_gaia_lite_index,
    query_local_gaia_lite,
    query_local_gaia_lite_duckdb,
    validate_local_gaia_result,
)
from spherex_laser_miner.catalog.manual_targets import get_manual_target, load_manual_targets
from spherex_laser_miner.config import load_config
from spherex_laser_miner.coarse_status import append_status_event, reset_coarse_status
from spherex_laser_miner.field_eval import evaluate_target_fields
from spherex_laser_miner.field_worker import (
    build_fixed_target_rows_from_trial,
    run_best_trial_field_worker,
    run_multi_trial_field_workers,
)
from spherex_laser_miner.spectra import assemble_spectra_from_jobs
from spherex_laser_miner.viewer import serve_viewer

try:
    from tools.score_spectrum_quality import score_run as score_spectrum_quality_run
except ModuleNotFoundError:
    import sys

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from tools.score_spectrum_quality import score_run as score_spectrum_quality_run

app = typer.Typer(no_args_is_help=True)
logging.getLogger("astropy").setLevel(logging.ERROR)


@app.callback()
def main() -> None:
    """SPHEREx field-first mining CLI."""


@app.command()
def doctor(cache_root: Path | None = typer.Option(None, help="Override SPHEREx cache root.")) -> None:
    """Check cache, references, and local runtime imports."""
    cfg = load_config(cache_root)
    ensure_cache_dirs(cfg.cache_root)

    checks: dict[str, object] = {
        "cache_root": str(cfg.cache_root),
        "cache_root_writable": _is_writable_dir(cfg.cache_root),
        "manual_targets_path": str(cfg.manual_targets_path),
        "manual_target_count": len(load_manual_targets(cfg.manual_targets_path)),
        "simp_target_loads": get_manual_target(cfg.manual_targets_path, "simp0136").object_name,
        "explanatory_supplement_pdf": str(cfg.docs_dir / "SPHEREx_Expsupp_QR.pdf"),
        "explanatory_supplement_pdf_exists": (cfg.docs_dir / "SPHEREx_Expsupp_QR.pdf").exists(),
        "spexpi_source": str(cfg.spexpi_dir),
        "spexpi_source_exists": cfg.spexpi_dir.exists(),
        "imports": _check_imports(),
    }
    typer.echo(json.dumps(checks, indent=2, sort_keys=True))


@app.command("build-gaia-lite")
def build_gaia_lite(
    cache_root: Path | None = typer.Option(None, help="Override SPHEREx cache root."),
    limit_files: int | None = typer.Option(
        None,
        min=1,
        help="Only process this many raw Gaia CSV files.",
    ),
    overwrite: bool = typer.Option(False, help="Replace an existing local Gaia lite index."),
    hp_level: int = typer.Option(
        3,
        "--hpx-level",
        "--hp-level",
        min=0,
        max=12,
        help="Gaia source_id HEALPix partition level.",
    ),
    max_rows_per_file: int = typer.Option(2_500_000, min=1, help="Approximate max rows per Parquet file."),
    max_buffered_rows: int = typer.Option(5_000_000, min=1, help="Flush largest buffers above this row count."),
) -> None:
    """Build a local Gaia DR3 lite Parquet index."""
    cfg = load_config(cache_root)
    manifest = build_gaia_lite_index(
        cache_root=cfg.cache_root,
        limit_files=limit_files,
        overwrite=overwrite,
        hp_level=hp_level,
        max_rows_per_file=max_rows_per_file,
        max_buffered_rows=max_buffered_rows,
    )
    typer.echo(json.dumps(manifest, indent=2, sort_keys=True))


@app.command("query-local-gaia")
def query_local_gaia(
    s_region: str = typer.Option(..., "--s-region", help="SIA POLYGON s_region to query."),
    g_min: float = typer.Option(8.0, help="Minimum Gaia G magnitude."),
    g_max: float = typer.Option(19.0, help="Maximum Gaia G magnitude."),
    max_sources: int = typer.Option(500, min=0, help="Maximum Gaia sources to return."),
    cache_root: Path | None = typer.Option(None, help="Override SPHEREx cache root."),
    output: Path | None = typer.Option(None, help="Optional .parquet or .csv output path."),
    engine: str = typer.Option("pyarrow", help="Query engine: pyarrow or duckdb."),
) -> None:
    """Query the local Gaia lite index for one SIA polygon."""
    cfg = load_config(cache_root)
    query_fn = query_local_gaia_lite_duckdb if engine == "duckdb" else query_local_gaia_lite
    if engine not in {"pyarrow", "duckdb"}:
        raise typer.BadParameter("engine must be pyarrow or duckdb")
    df = query_fn(
        s_region=s_region,
        cache_root=cfg.cache_root,
        max_sources=max_sources,
        g_min=g_min,
        g_max=g_max,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == ".csv":
            df.to_csv(output, index=False)
        else:
            df.to_parquet(output, index=False)
    summary = validate_local_gaia_result(df, s_region=s_region, g_min=g_min, g_max=g_max)
    summary["output"] = str(output) if output is not None else None
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("compare-local-gaia")
def compare_local_gaia(
    s_region: str = typer.Option(..., "--s-region", help="SIA POLYGON s_region to query."),
    g_min: float = typer.Option(8.0, help="Minimum Gaia G magnitude."),
    g_max: float = typer.Option(19.0, help="Maximum Gaia G magnitude."),
    max_sources: int = typer.Option(500, min=0, help="Maximum Gaia sources to return."),
    cache_root: Path | None = typer.Option(None, help="Override SPHEREx cache root."),
) -> None:
    """Validate local Gaia lite query behavior and deterministic sampling."""
    cfg = load_config(cache_root)
    first = query_local_gaia_lite(
        s_region=s_region,
        cache_root=cfg.cache_root,
        max_sources=max_sources,
        g_min=g_min,
        g_max=g_max,
    )
    second = query_local_gaia_lite(
        s_region=s_region,
        cache_root=cfg.cache_root,
        max_sources=max_sources,
        g_min=g_min,
        g_max=g_max,
    )
    summary = validate_local_gaia_result(first, s_region=s_region, g_min=g_min, g_max=g_max)
    summary["deterministic_repeated_query"] = first.equals(second)
    summary["validation_passed"] = all(
        [
            summary["columns_match_remote_contract"],
            summary["inside_conservative_bounds"],
            summary["magnitude_cuts_respected"],
            summary["source_ids_unique"],
            summary["deterministic_repeated_query"],
        ]
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("smoke-local-gaia-duckdb")
def smoke_local_gaia_duckdb(
    cache_root: Path | None = typer.Option(None, help="Override SPHEREx cache root."),
    max_sources: int = typer.Option(25, min=1, help="Sources to return from each query engine."),
    half_width_deg: float = typer.Option(0.25, min=0.001, help="Half-width of generated test polygon."),
) -> None:
    """Compare PyArrow and DuckDB local Gaia queries on one generated polygon."""
    import duckdb
    import pandas as pd

    cfg = load_config(cache_root)
    index = cfg.cache_root / "gaia" / "parquet" / "dr3_source_lite"
    sample_file = next(index.glob("hpx_level=*/hpx=*/part-*.parquet"), None)
    if sample_file is None:
        raise typer.BadParameter(f"No Parquet files found in {index}")
    hpx = int(sample_file.parent.name.split("=", 1)[1])
    sample = pd.read_parquet(sample_file, columns=["source_id", "ra", "dec", "phot_g_mean_mag"]).dropna(
        subset=["ra", "dec", "phot_g_mean_mag"]
    )
    if sample.empty:
        raise typer.BadParameter(f"No queryable rows found in {sample_file}")
    row = sample.iloc[len(sample) // 2]
    ra = float(row.ra)
    dec = float(row.dec)
    half = half_width_deg
    s_region = (
        f"POLYGON {ra - half} {dec - half} {ra + half} {dec - half} "
        f"{ra + half} {dec + half} {ra - half} {dec + half}"
    )
    pyarrow_df = pd.read_parquet(sample_file)
    pyarrow_df = pyarrow_df[
        pyarrow_df["ra"].between(ra - half, ra + half)
        & pyarrow_df["dec"].between(dec - half, dec + half)
        & pyarrow_df["phot_g_mean_mag"].between(8.0, 19.0)
    ].sort_values(["phot_g_mean_mag", "source_id"]).head(max_sources)
    pyarrow_df = pyarrow_df[GAIA_LITE_COLUMNS].reset_index(drop=True)
    duckdb_start = time.perf_counter()
    con = duckdb.connect(":memory:")
    try:
        duckdb_df = con.execute(
            f"""
            SELECT {", ".join(GAIA_LITE_COLUMNS)}
            FROM read_parquet('{index}/hpx_level=*/hpx=*/part-*.parquet', hive_partitioning = true)
            WHERE hpx = ?
              AND phot_g_mean_mag BETWEEN 8.0 AND 19.0
              AND ra BETWEEN ? AND ?
              AND dec BETWEEN ? AND ?
            ORDER BY phot_g_mean_mag, source_id
            LIMIT ?
            """,
            [hpx, ra - half, ra + half, dec - half, dec + half, max_sources],
        ).fetchdf()
    finally:
        con.close()
    duckdb_df.attrs["local_gaia_metrics"] = {
        "engine": "duckdb",
        "sample_file": str(sample_file),
        "hpx": hpx,
        "query_wall_time_sec": time.perf_counter() - duckdb_start,
    }
    pyarrow_df.attrs["local_gaia_metrics"] = {
        "engine": "pyarrow-single-file",
        "sample_file": str(sample_file),
        "hpx": hpx,
        "rows_scanned": len(sample),
    }
    summary = {
        "cache_root": str(cfg.cache_root),
        "index": str(index),
        "sample_file": str(sample_file),
        "hpx": hpx,
        "s_region": s_region,
        "pyarrow": validate_local_gaia_result(pyarrow_df, s_region=s_region, g_min=8.0, g_max=19.0),
        "duckdb": validate_local_gaia_result(duckdb_df, s_region=s_region, g_min=8.0, g_max=19.0),
        "same_source_ids": pyarrow_df["source_id"].tolist() == duckdb_df["source_id"].tolist(),
        "pyarrow_metrics": pyarrow_df.attrs.get("local_gaia_metrics", {}),
        "duckdb_metrics": duckdb_df.attrs.get("local_gaia_metrics", {}),
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("run-field-smoke-test")
def run_field_smoke_test(
    target: str = typer.Option("simp0136", help="Manual target id."),
    release: str = typer.Option("qr2", help="SPHEREx release."),
    limit_fields: int = typer.Option(3, min=1, help="Number of SIA candidates to download/evaluate."),
    max_gaia_sources: int = typer.Option(500, min=0, help="Maximum Gaia sources to select in the best field."),
    enable_psf: bool = typer.Option(False, help="Run experimental PSF photometry."),
    cache_root: Path | None = typer.Option(None, help="Override SPHEREx cache root."),
    redownload: bool = typer.Option(False, help="Refresh cached parent MEFs."),
) -> None:
    """Evaluate parent SPHEREx fields covering a manual target."""
    cfg = load_config(cache_root)
    cfg.release = release
    cfg.enable_psf_photometry = enable_psf
    ensure_cache_dirs(cfg.cache_root)
    manual_target = get_manual_target(cfg.manual_targets_path, target)
    trials = evaluate_target_fields(
        target=manual_target,
        cfg=cfg,
        limit_fields=limit_fields,
        redownload=redownload,
        max_eval_workers=1,
    )
    field_job = run_best_trial_field_worker(
        target=manual_target,
        cfg=cfg,
        trials=trials,
        max_gaia_sources=max_gaia_sources,
    )
    measured = [trial for trial in trials if trial.get("status") == "measured"]
    summary = {
        "target": target,
        "run_name": cfg.run_name,
        "release": release,
        "trial_count": len(trials),
        "measured_count": len(measured),
        "field_worker_targets_measured": field_job["targets_measured"],
        "field_worker_simp_measured": field_job["simp_measured"],
        "run_dir": str(cfg.smoke_run_dir),
        "trial_json": str(cfg.smoke_run_dir / "simp_field_trials.json"),
        "measurements_parquet": str(cfg.smoke_run_dir / "measurements.parquet"),
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("run-multifield-smoke-test")
def run_multifield_smoke_test(
    target: str = typer.Option("simp0136", help="Manual target id."),
    release: str = typer.Option("qr2", help="SPHEREx release."),
    limit_fields: int = typer.Option(10, min=1, help="Number of SIA candidates to download/evaluate."),
    max_gaia_sources: int = typer.Option(100, min=0, help="Maximum Gaia sources per field shard."),
    max_field_workers: int = typer.Option(6, min=1, help="Concurrent parent-field workers."),
    enable_psf: bool = typer.Option(False, help="Run experimental PSF photometry."),
    cache_root: Path | None = typer.Option(None, help="Override SPHEREx cache root."),
    redownload: bool = typer.Option(False, help="Refresh cached parent MEFs."),
    include_fatal_simp_trials: bool = typer.Option(True, help="Process measured SIMP trial fields even when SIMP aperture has a fatal flag."),
) -> None:
    """Process every measured SIMP-overlap parent field as a full-field shard."""
    cfg = load_config(cache_root)
    cfg.release = release
    cfg.enable_psf_photometry = enable_psf
    ensure_cache_dirs(cfg.cache_root)
    manual_target = get_manual_target(cfg.manual_targets_path, target)
    trials = evaluate_target_fields(
        target=manual_target,
        cfg=cfg,
        limit_fields=limit_fields,
        redownload=redownload,
        max_eval_workers=max_field_workers,
    )
    jobs = run_multi_trial_field_workers(
        target=manual_target,
        cfg=cfg,
        trials=trials,
        max_gaia_sources=max_gaia_sources,
        include_fatal_simp_trials=include_fatal_simp_trials,
        max_field_workers=max_field_workers,
    )
    assembly = assemble_spectra_from_jobs(cfg.smoke_run_dir, jobs)
    spectrum_quality = _score_spectrum_quality_stage(cfg.smoke_run_dir, cfg.status_mode)
    summary = {
        "target": target,
        "release": release,
        "trial_count": len(trials),
        "field_job_count": len(jobs),
        "assembly": assembly,
        "spectrum_quality": spectrum_quality,
        "run_dir": str(cfg.smoke_run_dir),
        "viewer_hint": "spherex-mine viewer --host 0.0.0.0 --port 8765",
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("run-depth-test")
def run_depth_test(
    target: str = typer.Option("simp0136", help="Manual target id."),
    run_name: str | None = typer.Option(None, help="Output run name under cache_root/runs."),
    release: str = typer.Option("qr2", help="SPHEREx release."),
    limit_fields: int = typer.Option(220, min=1, help="Number of SIA candidates to evaluate/process."),
    max_gaia_sources: int = typer.Option(100, min=0, help="Fixed Gaia targets carried through every field."),
    gaia_g_min: float = typer.Option(7.0, help="Minimum Gaia G magnitude for fixed depth targets."),
    gaia_g_max: float = typer.Option(10.0, help="Maximum Gaia G magnitude for fixed depth targets."),
    max_field_workers: int = typer.Option(24, min=1, help="Concurrent parent-field workers."),
    photometry_backend: str = typer.Option("cpu_numpy", help="Photometry backend: cpu_numpy or warp_calibrated."),
    warp_devices: str = typer.Option("cuda:0,cuda:1,cuda:2", help="Comma-separated Warp CUDA devices."),
    status_mode: str = typer.Option("jsonl", help="Status backend: jsonl or off. Legacy live aliases to jsonl."),
    max_field_retries: int = typer.Option(0, min=0, help="Retry failed fields this many times."),
    enable_psf: bool = typer.Option(False, help="Run experimental PSF photometry."),
    psf_photometry_backend: str = typer.Option("cpu_single", help="PSF backend: cpu_single or warp_grid."),
    psf_kernel_build_mode: str = typer.Option("gpu_spline", help="PSF kernel build mode: cpu_scipy, gpu_bilinear, or gpu_spline."),
    psf_grid_half_range_pix: float = typer.Option(1.0, help="Half-width of PSF local grid search, in pixels."),
    psf_grid_step_pix: float = typer.Option(0.5, help="Step size of PSF local grid search, in pixels."),
    psf_grid_metric: str = typer.Option("snr", help="PSF grid metric: snr or chi2."),
    enable_diagnostic_aperture: bool = typer.Option(False, help="Run raw diagnostic aperture QA photometry."),
    cache_root: Path | None = typer.Option(None, help="Override SPHEREx cache root."),
    redownload: bool = typer.Option(False, help="Refresh cached parent MEFs."),
    path_overrides: Path | None = typer.Option(None, help="JSON map from raw FITS path to replacement FITS path."),
    fixed_targets_path: Path | None = typer.Option(None, help="Optional Parquet/CSV fixed target rows to use instead of querying Gaia."),
    field_launch_stagger_sec: float = typer.Option(0.0, min=0.0, help="Delay between field worker submissions; useful for GPU backend startup."),
) -> None:
    """Run a deeper SIMP-centered spectral pass using one fixed target set."""
    summary = _run_depth_pipeline(
        target=target,
        run_name=run_name,
        release=release,
        limit_fields=limit_fields,
        max_gaia_sources=max_gaia_sources,
        gaia_g_min=gaia_g_min,
        gaia_g_max=gaia_g_max,
        max_field_workers=max_field_workers,
        photometry_backend=photometry_backend,
        warp_devices=warp_devices,
        status_mode=status_mode,
        max_field_retries=max_field_retries,
        enable_psf=enable_psf,
        psf_photometry_backend=psf_photometry_backend,
        psf_kernel_build_mode=psf_kernel_build_mode,
        psf_grid_half_range_pix=psf_grid_half_range_pix,
        psf_grid_step_pix=psf_grid_step_pix,
        psf_grid_metric=psf_grid_metric,
        enable_diagnostic_aperture=enable_diagnostic_aperture,
        cache_root=cache_root,
        redownload=redownload,
        path_overrides=path_overrides,
        fixed_targets_path=fixed_targets_path,
        field_launch_stagger_sec=field_launch_stagger_sec,
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("run-benchmark")
def run_benchmark(
    target: str = typer.Option("simp0136", help="Manual target id."),
    run_name: str | None = typer.Option(None, help="Output run name under cache_root/runs."),
    release: str = typer.Option("qr2", help="SPHEREx release."),
    limit_fields: int = typer.Option(30, min=1, help="Number of SIA candidates to evaluate/process."),
    max_gaia_sources: int = typer.Option(100, min=0, help="Fixed Gaia targets carried through every field."),
    gaia_g_min: float = typer.Option(12.5, help="Minimum Gaia G magnitude for fixed benchmark targets."),
    gaia_g_max: float = typer.Option(14.0, help="Maximum Gaia G magnitude for fixed benchmark targets."),
    max_field_workers: int = typer.Option(24, min=1, help="Concurrent parent-field workers."),
    photometry_backend: str = typer.Option("cpu_numpy", help="Photometry backend: cpu_numpy or warp_calibrated."),
    warp_devices: str = typer.Option("cuda:0,cuda:1,cuda:2", help="Comma-separated Warp CUDA devices."),
    status_mode: str = typer.Option("jsonl", help="Status backend: jsonl or off. Legacy live aliases to jsonl."),
    max_field_retries: int = typer.Option(0, min=0, help="Retry failed fields this many times."),
    enable_psf: bool = typer.Option(False, help="Run experimental PSF photometry."),
    psf_photometry_backend: str = typer.Option("cpu_single", help="PSF backend: cpu_single or warp_grid."),
    psf_kernel_build_mode: str = typer.Option("gpu_spline", help="PSF kernel build mode: cpu_scipy, gpu_bilinear, or gpu_spline."),
    psf_grid_half_range_pix: float = typer.Option(1.0, help="Half-width of PSF local grid search, in pixels."),
    psf_grid_step_pix: float = typer.Option(0.5, help="Step size of PSF local grid search, in pixels."),
    psf_grid_metric: str = typer.Option("snr", help="PSF grid metric: snr or chi2."),
    enable_diagnostic_aperture: bool = typer.Option(False, help="Run raw diagnostic aperture QA photometry."),
    cache_root: Path | None = typer.Option(None, help="Override SPHEREx cache root."),
    redownload: bool = typer.Option(False, help="Refresh cached parent MEFs."),
    output: Path | None = typer.Option(None, help="Optional benchmark summary JSON output."),
    path_overrides: Path | None = typer.Option(None, help="JSON map from raw FITS path to replacement FITS path."),
    fixed_targets_path: Path | None = typer.Option(None, help="Optional Parquet/CSV fixed target rows to use instead of querying Gaia."),
    field_launch_stagger_sec: float = typer.Option(0.0, min=0.0, help="Delay between field worker submissions; useful for GPU backend startup."),
) -> None:
    """Run one controlled benchmark pass and write throughput metrics."""
    wall_start = time.perf_counter()
    summary = _run_depth_pipeline(
        target=target,
        run_name=run_name,
        release=release,
        limit_fields=limit_fields,
        max_gaia_sources=max_gaia_sources,
        gaia_g_min=gaia_g_min,
        gaia_g_max=gaia_g_max,
        max_field_workers=max_field_workers,
        photometry_backend=photometry_backend,
        warp_devices=warp_devices,
        status_mode=status_mode,
        max_field_retries=max_field_retries,
        enable_psf=enable_psf,
        psf_photometry_backend=psf_photometry_backend,
        psf_kernel_build_mode=psf_kernel_build_mode,
        psf_grid_half_range_pix=psf_grid_half_range_pix,
        psf_grid_step_pix=psf_grid_step_pix,
        psf_grid_metric=psf_grid_metric,
        enable_diagnostic_aperture=enable_diagnostic_aperture,
        cache_root=cache_root,
        redownload=redownload,
        path_overrides=path_overrides,
        fixed_targets_path=fixed_targets_path,
        field_launch_stagger_sec=field_launch_stagger_sec,
    )
    wall_elapsed = time.perf_counter() - wall_start
    measurements = int(dict(summary.get("assembly") or {}).get("measurement_rows") or 0)
    benchmark = {
        **summary,
        "benchmark": {
            "wall_elapsed_sec": wall_elapsed,
            "measurement_rows": measurements,
            "measurements_per_wall_sec": measurements / wall_elapsed if wall_elapsed > 0 else None,
            "measurements_per_worker_sec_estimate": (
                measurements / (wall_elapsed * max_field_workers) if wall_elapsed > 0 and max_field_workers > 0 else None
            ),
            "psf_enabled": enable_psf,
        },
    }
    if output is None:
        cfg = load_config(cache_root)
        output = cfg.smoke_run_dir / "benchmark_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    benchmark["benchmark"]["output"] = str(output)
    output.write_text(json.dumps(benchmark, indent=2, sort_keys=True), encoding="utf-8")
    typer.echo(json.dumps(benchmark, indent=2, sort_keys=True))


def _run_depth_pipeline(
    *,
    target: str,
    run_name: str | None,
    release: str,
    limit_fields: int,
    max_gaia_sources: int,
    gaia_g_min: float,
    gaia_g_max: float,
    max_field_workers: int,
    photometry_backend: str,
    warp_devices: str,
    status_mode: str,
    max_field_retries: int,
    enable_psf: bool,
    psf_photometry_backend: str = "cpu_single",
    psf_kernel_build_mode: str = "gpu_spline",
    psf_grid_half_range_pix: float = 1.0,
    psf_grid_step_pix: float = 0.5,
    psf_grid_metric: str = "snr",
    enable_diagnostic_aperture: bool = False,
    cache_root: Path | None = None,
    redownload: bool = False,
    path_overrides: Path | None = None,
    fixed_targets_path: Path | None = None,
    field_launch_stagger_sec: float = 0.0,
) -> dict[str, object]:
    cfg = load_config(cache_root)
    if run_name is not None:
        cfg.run_name = run_name
    cfg.release = release
    if photometry_backend not in {"cpu_numpy", "warp_calibrated"}:
        raise typer.BadParameter("photometry_backend must be cpu_numpy or warp_calibrated")
    if psf_photometry_backend not in {"cpu_single", "warp_grid"}:
        raise typer.BadParameter("psf_photometry_backend must be cpu_single or warp_grid")
    if psf_kernel_build_mode not in {"cpu_scipy", "gpu_bilinear", "gpu_spline"}:
        raise typer.BadParameter("psf_kernel_build_mode must be cpu_scipy, gpu_bilinear, or gpu_spline")
    if psf_grid_metric not in {"snr", "chi2"}:
        raise typer.BadParameter("psf_grid_metric must be snr or chi2")
    if status_mode == "live":
        status_mode = "jsonl"
    if status_mode not in {"jsonl", "off"}:
        raise typer.BadParameter("status_mode must be jsonl or off")
    cfg.photometry_backend = photometry_backend
    cfg.status_mode = status_mode
    cfg.warp_devices = tuple(part.strip() for part in warp_devices.split(",") if part.strip())
    cfg.enable_psf_photometry = enable_psf
    cfg.psf_photometry_backend = psf_photometry_backend
    cfg.psf_kernel_build_mode = psf_kernel_build_mode
    cfg.psf_grid_half_range_pix = float(psf_grid_half_range_pix)
    cfg.psf_grid_step_pix = float(psf_grid_step_pix)
    cfg.psf_grid_metric = psf_grid_metric
    cfg.enable_diagnostic_aperture = enable_diagnostic_aperture
    ensure_cache_dirs(cfg.cache_root)
    if cfg.photometry_backend == "warp_calibrated" or (
        cfg.enable_psf_photometry and cfg.psf_photometry_backend == "warp_grid"
    ):
        _prewarm_warp_modules(cfg.warp_devices, cfg.photometry_backend, cfg.enable_psf_photometry, cfg.psf_photometry_backend)
    if cfg.status_mode == "jsonl":
        reset_coarse_status(cfg.smoke_run_dir, worker_count=max_field_workers)
        append_status_event(
            cfg.smoke_run_dir,
            "pipeline_start",
            target=target,
            limit_fields=limit_fields,
            max_gaia_sources=max_gaia_sources,
            gaia_g_min=gaia_g_min,
            gaia_g_max=gaia_g_max,
            photometry_backend=photometry_backend,
            psf_photometry_backend=psf_photometry_backend,
            psf_kernel_build_mode=psf_kernel_build_mode,
            psf_grid_half_range_pix=psf_grid_half_range_pix,
            psf_grid_step_pix=psf_grid_step_pix,
            psf_grid_metric=psf_grid_metric,
            field_launch_stagger_sec=field_launch_stagger_sec,
        )
    manual_target = get_manual_target(cfg.manual_targets_path, target)
    try:
        trials = evaluate_target_fields(
            target=manual_target,
            cfg=cfg,
            limit_fields=limit_fields,
            redownload=redownload,
            max_eval_workers=max_field_workers,
        )
        measured_trials = [trial for trial in trials if trial.get("status") == "measured"]
        if not measured_trials:
            raise typer.BadParameter("No measured parent fields available for depth run.")
    except Exception as exc:
        if cfg.status_mode == "jsonl":
            append_status_event(cfg.smoke_run_dir, "run_error", error=f"{type(exc).__name__}: {exc}")
        raise
    path_override_map = _load_path_overrides(path_overrides)
    best_trial = max(
        measured_trials,
        key=lambda trial: (
            not dict(trial.get("aperture") or {}).get("fatal_flag_present", False),
            float(trial.get("edge_distance_pix") or -1.0),
        ),
    )
    if fixed_targets_path is not None:
        fixed_targets = _load_fixed_targets(fixed_targets_path)
    else:
        fixed_targets = build_fixed_target_rows_from_trial(
            target=manual_target,
            cfg=cfg,
            trial=best_trial,
            max_gaia_sources=max_gaia_sources,
            gaia_g_min=gaia_g_min,
            gaia_g_max=gaia_g_max,
        )
    jobs = run_multi_trial_field_workers(
        target=manual_target,
        cfg=cfg,
        trials=trials,
        max_gaia_sources=max_gaia_sources,
        include_fatal_simp_trials=True,
        max_field_workers=max_field_workers,
        target_rows_override=fixed_targets,
        gaia_g_min=gaia_g_min,
        gaia_g_max=gaia_g_max,
        path_overrides=path_override_map,
        max_field_retries=max_field_retries,
        field_launch_stagger_sec=field_launch_stagger_sec,
    )
    error_path = cfg.smoke_run_dir / "field_errors.json"
    field_errors = _read_json_file(error_path) if error_path.exists() else []
    assembly = assemble_spectra_from_jobs(cfg.smoke_run_dir, jobs)
    spectrum_quality = _score_spectrum_quality_stage(cfg.smoke_run_dir, cfg.status_mode)
    summary = {
        "target": target,
        "release": release,
        "trial_count": len(trials),
        "measured_trial_count": len(measured_trials),
        "fixed_target_count": len(fixed_targets),
        "gaia_g_min": gaia_g_min,
        "gaia_g_max": gaia_g_max,
        "field_job_count": len(jobs),
        "max_field_workers": max_field_workers,
        "photometry_backend": cfg.photometry_backend,
        "warp_devices": list(cfg.warp_devices),
        "status_mode": cfg.status_mode,
        "max_field_retries": max_field_retries,
        "field_error_count": len(field_errors) if isinstance(field_errors, list) else 0,
        "psf_enabled": enable_psf,
        "psf_photometry_backend": cfg.psf_photometry_backend if enable_psf else "disabled",
        "psf_kernel_build_mode": cfg.psf_kernel_build_mode if enable_psf else None,
        "psf_grid_half_range_pix": cfg.psf_grid_half_range_pix if enable_psf else None,
        "psf_grid_step_pix": cfg.psf_grid_step_pix if enable_psf else None,
        "psf_grid_metric": cfg.psf_grid_metric if enable_psf else None,
        "diagnostic_aperture_enabled": enable_diagnostic_aperture,
        "path_overrides_path": str(path_overrides) if path_overrides is not None else None,
        "path_override_count": len(path_override_map),
        "fixed_targets_path": str(fixed_targets_path) if fixed_targets_path is not None else None,
        "field_launch_stagger_sec": field_launch_stagger_sec,
        "assembly": assembly,
        "spectrum_quality": spectrum_quality,
        "run_dir": str(cfg.smoke_run_dir),
    }
    (cfg.smoke_run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _prewarm_warp_modules(
    devices: tuple[str, ...],
    photometry_backend: str,
    enable_psf: bool,
    psf_photometry_backend: str,
) -> None:
    try:
        import warp as wp

        from spherex_laser_miner.photometry import psf_forced, warp_calibrated
    except Exception:
        return
    wp.init()
    for device in devices:
        if photometry_backend == "warp_calibrated":
            wp.load_module(warp_calibrated, device=device, max_workers=0)
        if enable_psf and psf_photometry_backend == "warp_grid":
            wp.load_module(psf_forced, device=device, max_workers=0)


@app.command()
def viewer(
    host: str = typer.Option("0.0.0.0", help="Bind host."),
    port: int = typer.Option(8765, help="Bind port."),
    run_name: str | None = typer.Option(None, help="Initial run name under cache_root/runs."),
    cache_root: Path | None = typer.Option(None, help="Override SPHEREx cache root."),
) -> None:
    """Serve a local web viewer for smoke run artifacts."""
    cfg = load_config(cache_root)
    if run_name is not None:
        cfg.run_name = run_name
    serve_viewer(cfg.smoke_run_dir, host, port)


def _is_writable_dir(path: Path) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write_probe"
    try:
        probe.write_text("ok\n", encoding="utf-8")
        return probe.read_text(encoding="utf-8") == "ok\n"
    finally:
        probe.unlink(missing_ok=True)


def _load_path_overrides(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise typer.BadParameter(f"Path override JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise typer.BadParameter("Path override JSON must be an object mapping original path to replacement path")
    return {str(key): str(value) for key, value in data.items()}


def _load_fixed_targets(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise typer.BadParameter(f"Fixed target file not found: {path}")
    import pandas as pd

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    required = {"target_id", "target_type", "ra_reference_deg", "dec_reference_deg"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise typer.BadParameter(f"Fixed target file missing columns: {', '.join(missing)}")
    if df.empty:
        raise typer.BadParameter("Fixed target file is empty")
    return df.where(pd.notna(df), None).to_dict(orient="records")


def _score_spectrum_quality_stage(run_dir: Path, status_mode: str) -> dict[str, object]:
    started = time.perf_counter()
    if status_mode == "jsonl":
        append_status_event(run_dir, "spectrum_quality_start")
    try:
        scored = score_spectrum_quality_run(run_dir)
        elapsed = time.perf_counter() - started
        category_counts = scored["spectrum_quality_category"].value_counts().sort_index().to_dict() if not scored.empty else {}
        summary = {
            "status": "done",
            "elapsed_sec": elapsed,
            "target_count": int(len(scored)),
            "good_count": int(category_counts.get("good", 0)),
            "review_count": int(category_counts.get("review", 0)),
            "bad_count": int(category_counts.get("bad", 0)),
            "category_counts": {str(key): int(value) for key, value in category_counts.items()},
            "spectrum_quality_path": str(run_dir / "spectra" / "spectrum_quality.parquet"),
            "spectrum_quality_summary_path": str(run_dir / "spectra" / "spectrum_quality_summary.json"),
        }
        if status_mode == "jsonl":
            append_status_event(run_dir, "spectrum_quality_done", **summary)
        return summary
    except Exception as exc:
        elapsed = time.perf_counter() - started
        summary = {
            "status": "error",
            "elapsed_sec": elapsed,
            "error": f"{type(exc).__name__}: {exc}",
            "spectrum_quality_path": str(run_dir / "spectra" / "spectrum_quality.parquet"),
        }
        if status_mode == "jsonl":
            append_status_event(run_dir, "spectrum_quality_error", **summary)
        return summary


def _read_json_file(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_imports() -> dict[str, bool]:
    modules = ["astropy", "matplotlib", "numpy", "pandas", "photutils", "pyarrow", "pyvo", "yaml"]
    results: dict[str, bool] = {}
    for module in modules:
        try:
            __import__(module)
        except Exception:
            results[module] = False
        else:
            results[module] = True
    return results


if __name__ == "__main__":
    app()
