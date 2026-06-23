from __future__ import annotations

import json
import time
from pathlib import Path

import typer

from spherex_laser_miner.cache import ensure_cache_dirs
from spherex_laser_miner.catalog.manual_targets import get_manual_target, load_manual_targets
from spherex_laser_miner.cli import _check_imports, _is_writable_dir, _run_depth_pipeline
from spherex_laser_miner.config import load_config
from spherex_laser_miner.viewer import serve_viewer

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """GPU aperture plus GPU spline-grid PSF SPHEREx miner."""


@app.command()
def doctor(cache_root: Path | None = typer.Option(None, help="Override SPHEREx cache root.")) -> None:
    """Check cache, references, GPU imports, and local runtime imports."""
    cfg = load_config(cache_root)
    ensure_cache_dirs(cfg.cache_root)
    try:
        import warp as wp

        wp.init()
        warp_info = {
            "available": True,
            "version": getattr(wp, "__version__", None),
            "devices": [str(device) for device in wp.get_devices()],
        }
    except Exception as exc:
        warp_info = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    checks: dict[str, object] = {
        "command": "spherex-mine-gpu-psf",
        "cache_root": str(cfg.cache_root),
        "cache_root_writable": _is_writable_dir(cfg.cache_root),
        "manual_targets_path": str(cfg.manual_targets_path),
        "manual_target_count": len(load_manual_targets(cfg.manual_targets_path)),
        "ucs_target_loads": get_manual_target(cfg.manual_targets_path, "ucs_0972").object_name,
        "imports": _check_imports(),
        "warp": warp_info,
        "default_photometry_backend": "warp_calibrated",
        "default_psf_backend": "warp_grid",
        "default_psf_kernel_build_mode": "gpu_spline",
    }
    typer.echo(json.dumps(checks, indent=2, sort_keys=True))


@app.command("run-depth-test")
def run_depth_test(
    target: str = typer.Option("ucs_0972", help="Manual target id."),
    run_name: str | None = typer.Option(None, help="Output run name under cache_root/runs."),
    release: str = typer.Option("qr2", help="SPHEREx release."),
    limit_fields: int = typer.Option(200, min=1, help="Number of SIA candidates to evaluate/process."),
    max_gaia_sources: int = typer.Option(1000, min=0, help="Fixed Gaia targets carried through every field."),
    gaia_g_min: float = typer.Option(11.0, help="Minimum Gaia G magnitude for fixed depth targets."),
    gaia_g_max: float = typer.Option(16.0, help="Maximum Gaia G magnitude for fixed depth targets."),
    max_field_workers: int = typer.Option(24, min=1, help="Concurrent parent-field workers."),
    warp_devices: str = typer.Option("cuda:0,cuda:1,cuda:2", help="Comma-separated Warp CUDA devices."),
    status_mode: str = typer.Option("live", help="Status backend: live, jsonl, or off."),
    max_field_retries: int = typer.Option(1, min=0, help="Retry failed fields this many times."),
    psf_kernel_build_mode: str = typer.Option("gpu_spline", help="PSF kernel build mode: gpu_spline, gpu_bilinear, or cpu_scipy."),
    psf_grid_half_range_pix: float = typer.Option(1.0, help="Half-width of PSF local grid search, in pixels."),
    psf_grid_step_pix: float = typer.Option(0.5, help="Step size of PSF local grid search, in pixels."),
    psf_grid_metric: str = typer.Option("snr", help="PSF grid metric: snr or chi2."),
    enable_diagnostic_aperture: bool = typer.Option(False, help="Run raw diagnostic aperture QA photometry."),
    cache_root: Path | None = typer.Option(None, help="Override SPHEREx cache root."),
    redownload: bool = typer.Option(False, help="Refresh cached parent MEFs."),
    path_overrides: Path | None = typer.Option(None, help="JSON map from raw FITS path to replacement FITS path."),
    fixed_targets_path: Path | None = typer.Option(None, help="Optional Parquet/CSV fixed target rows to use instead of querying Gaia."),
    field_launch_stagger_sec: float = typer.Option(0.0, min=0.0, help="Delay between field worker submissions."),
) -> None:
    """Run the production GPU aperture + GPU spline-grid PSF miner."""
    summary = _run_depth_pipeline(
        target=target,
        run_name=run_name,
        release=release,
        limit_fields=limit_fields,
        max_gaia_sources=max_gaia_sources,
        gaia_g_min=gaia_g_min,
        gaia_g_max=gaia_g_max,
        max_field_workers=max_field_workers,
        photometry_backend="warp_calibrated",
        warp_devices=warp_devices,
        status_mode=status_mode,
        max_field_retries=max_field_retries,
        enable_psf=True,
        psf_photometry_backend="warp_grid",
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
    target: str = typer.Option("ucs_0972", help="Manual target id."),
    run_name: str | None = typer.Option(None, help="Output run name under cache_root/runs."),
    release: str = typer.Option("qr2", help="SPHEREx release."),
    limit_fields: int = typer.Option(30, min=1, help="Number of SIA candidates to evaluate/process."),
    max_gaia_sources: int = typer.Option(1000, min=0, help="Fixed Gaia targets carried through every field."),
    gaia_g_min: float = typer.Option(11.0, help="Minimum Gaia G magnitude for fixed benchmark targets."),
    gaia_g_max: float = typer.Option(16.0, help="Maximum Gaia G magnitude for fixed benchmark targets."),
    max_field_workers: int = typer.Option(24, min=1, help="Concurrent parent-field workers."),
    warp_devices: str = typer.Option("cuda:0,cuda:1,cuda:2", help="Comma-separated Warp CUDA devices."),
    status_mode: str = typer.Option("live", help="Status backend: live, jsonl, or off."),
    max_field_retries: int = typer.Option(1, min=0, help="Retry failed fields this many times."),
    output: Path | None = typer.Option(None, help="Optional benchmark summary JSON output."),
    cache_root: Path | None = typer.Option(None, help="Override SPHEREx cache root."),
) -> None:
    """Run one controlled GPU PSF benchmark pass and write throughput metrics."""
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
        photometry_backend="warp_calibrated",
        warp_devices=warp_devices,
        status_mode=status_mode,
        max_field_retries=max_field_retries,
        enable_psf=True,
        psf_photometry_backend="warp_grid",
        psf_kernel_build_mode="gpu_spline",
        psf_grid_half_range_pix=1.0,
        psf_grid_step_pix=0.5,
        psf_grid_metric="snr",
        enable_diagnostic_aperture=False,
        cache_root=cache_root,
        redownload=False,
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
            "psf_enabled": True,
            "psf_photometry_backend": "warp_grid",
            "psf_kernel_build_mode": "gpu_spline",
        },
    }
    if output is None:
        cfg = load_config(cache_root)
        if run_name is not None:
            cfg.run_name = run_name
        output = cfg.smoke_run_dir / "benchmark_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    benchmark["benchmark"]["output"] = str(output)
    output.write_text(json.dumps(benchmark, indent=2, sort_keys=True), encoding="utf-8")
    typer.echo(json.dumps(benchmark, indent=2, sort_keys=True))


@app.command()
def viewer(
    host: str = typer.Option("0.0.0.0", help="Bind host."),
    port: int = typer.Option(8765, help="Bind port."),
    run_name: str | None = typer.Option(None, help="Initial run name under cache_root/runs."),
    cache_root: Path | None = typer.Option(None, help="Override SPHEREx cache root."),
) -> None:
    """Serve the local web viewer for GPU PSF miner artifacts."""
    cfg = load_config(cache_root)
    if run_name is not None:
        cfg.run_name = run_name
    serve_viewer(cfg.smoke_run_dir, host, port)


if __name__ == "__main__":
    app()
