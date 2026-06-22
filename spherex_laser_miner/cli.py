from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from spherex_laser_miner.cache import ensure_cache_dirs
from spherex_laser_miner.catalog.manual_targets import get_manual_target, load_manual_targets
from spherex_laser_miner.config import load_config
from spherex_laser_miner.field_eval import evaluate_target_fields
from spherex_laser_miner.field_worker import (
    build_fixed_target_rows_from_trial,
    run_best_trial_field_worker,
    run_multi_trial_field_workers,
)
from spherex_laser_miner.spectra import assemble_spectra_from_jobs
from spherex_laser_miner.viewer import serve_viewer

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


@app.command("run-field-smoke-test")
def run_field_smoke_test(
    target: str = typer.Option("simp0136", help="Manual target id."),
    release: str = typer.Option("qr2", help="SPHEREx release."),
    limit_fields: int = typer.Option(3, min=1, help="Number of SIA candidates to download/evaluate."),
    max_gaia_sources: int = typer.Option(500, min=0, help="Maximum Gaia sources to select in the best field."),
    cache_root: Path | None = typer.Option(None, help="Override SPHEREx cache root."),
    redownload: bool = typer.Option(False, help="Refresh cached parent MEFs."),
) -> None:
    """Evaluate parent SPHEREx fields covering a manual target."""
    cfg = load_config(cache_root)
    cfg.release = release
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
    cache_root: Path | None = typer.Option(None, help="Override SPHEREx cache root."),
    redownload: bool = typer.Option(False, help="Refresh cached parent MEFs."),
    include_fatal_simp_trials: bool = typer.Option(True, help="Process measured SIMP trial fields even when SIMP aperture has a fatal flag."),
) -> None:
    """Process every measured SIMP-overlap parent field as a full-field shard."""
    cfg = load_config(cache_root)
    cfg.release = release
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
    summary = {
        "target": target,
        "release": release,
        "trial_count": len(trials),
        "field_job_count": len(jobs),
        "assembly": assembly,
        "run_dir": str(cfg.smoke_run_dir),
        "viewer_hint": "spherex-mine viewer --host 0.0.0.0 --port 8765",
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("run-depth-test")
def run_depth_test(
    target: str = typer.Option("simp0136", help="Manual target id."),
    release: str = typer.Option("qr2", help="SPHEREx release."),
    limit_fields: int = typer.Option(220, min=1, help="Number of SIA candidates to evaluate/process."),
    max_gaia_sources: int = typer.Option(100, min=0, help="Fixed Gaia targets carried through every field."),
    gaia_g_min: float = typer.Option(7.0, help="Minimum Gaia G magnitude for fixed depth targets."),
    gaia_g_max: float = typer.Option(10.0, help="Maximum Gaia G magnitude for fixed depth targets."),
    max_field_workers: int = typer.Option(24, min=1, help="Concurrent parent-field workers."),
    cache_root: Path | None = typer.Option(None, help="Override SPHEREx cache root."),
    redownload: bool = typer.Option(False, help="Refresh cached parent MEFs."),
) -> None:
    """Run a deeper SIMP-centered spectral pass using one fixed target set."""
    cfg = load_config(cache_root)
    cfg.release = release
    ensure_cache_dirs(cfg.cache_root)
    manual_target = get_manual_target(cfg.manual_targets_path, target)
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
    best_trial = max(
        measured_trials,
        key=lambda trial: (
            not dict(trial.get("aperture") or {}).get("fatal_flag_present", False),
            float(trial.get("edge_distance_pix") or -1.0),
        ),
    )
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
    )
    assembly = assemble_spectra_from_jobs(cfg.smoke_run_dir, jobs)
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
        "assembly": assembly,
        "run_dir": str(cfg.smoke_run_dir),
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command()
def viewer(
    host: str = typer.Option("0.0.0.0", help="Bind host."),
    port: int = typer.Option(8765, help="Bind port."),
    cache_root: Path | None = typer.Option(None, help="Override SPHEREx cache root."),
) -> None:
    """Serve a local web viewer for smoke run artifacts."""
    cfg = load_config(cache_root)
    serve_viewer(cfg.smoke_run_dir, host, port)


def _is_writable_dir(path: Path) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write_probe"
    try:
        probe.write_text("ok\n", encoding="utf-8")
        return probe.read_text(encoding="utf-8") == "ok\n"
    finally:
        probe.unlink(missing_ok=True)


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
