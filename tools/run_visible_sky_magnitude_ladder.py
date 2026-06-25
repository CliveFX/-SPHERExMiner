#!/usr/bin/env python3
"""Run baseline visible-sky depth campaigns across Gaia magnitude bins.

This is a no-injection runner intended for broad science mining and bright-star
stress testing. Each target/bin run is fail-and-continue: failed depth or
classifier stages are recorded in the campaign manifest and the next run starts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_visible_sky_injection_campaign import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    DEFAULT_TARGETS,
    _done,
    _targets,
    _write_resolved_targets,
)


@dataclass(frozen=True)
class MagBin:
    name: str
    g_min: float
    g_max: float
    max_gaia_sources: int


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--campaign-prefix", default="cv_june_mag_ladder_f500_v1")
    parser.add_argument(
        "--mag-bin",
        action="append",
        help="Magnitude bin as name:g_min:g_max:max_sources. May be repeated.",
    )
    parser.add_argument("--start-at-bin", help="Skip bins before this bin name.")
    parser.add_argument("--stop-after-bin", help="Stop after completing this bin name.")
    parser.add_argument("--limit-targets", type=int)
    parser.add_argument("--only-target", action="append", help="Run one target_id; may be passed multiple times.")
    parser.add_argument("--limit-fields", type=int, default=500)
    parser.add_argument("--max-field-workers", type=int, default=24)
    parser.add_argument("--warp-devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--resolve-gaia-anchors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--anchor-search-radius-deg", type=float, default=1.0)
    parser.add_argument("--anchor-g-min", type=float, default=12.0)
    parser.add_argument("--anchor-g-max", type=float, default=14.0)
    parser.add_argument("--anchor-preferred-g", type=float, default=13.0)
    parser.add_argument("--max-field-retries", type=int, default=1)
    parser.add_argument("--blind-scan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--blind-grid-step-nm", type=float, default=5.0)
    parser.add_argument("--blind-warp-device", default="cuda:0")
    parser.add_argument("--blind-top-k-per-target", type=int, default=10)
    parser.add_argument("--blind-top-k-min-separation-nm", type=float)
    parser.add_argument("--narrowband-min-joint-rho", type=float, default=3.0)
    parser.add_argument("--narrowband-quality-min-support", type=int, default=3)
    parser.add_argument("--narrowband-quality-max-flagged-points", type=int, default=3)
    parser.add_argument("--narrowband-quality-max-candidates-per-target", type=int, default=5)
    parser.add_argument("--narrowband-quality-max-aperture-psf-ratio", type=float, default=3.0)
    parser.add_argument("--narrowband-diagnostic-line-half-window-nm", type=float, default=80.0)
    parser.add_argument("--narrowband-diagnostic-line-max-rows-per-candidate", type=int, default=201)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    mag_bins = _select_mag_bins(_parse_mag_bins(args.mag_bin), args.start_at_bin, args.stop_after_bin)
    selected_targets = _targets(args.targets, args.limit_targets, set(args.only_target or []) or None)
    if not selected_targets:
        raise SystemExit(f"No targets selected from {args.targets}")

    campaign_root = args.cache_root / "campaigns" / args.campaign_prefix
    campaign_root.mkdir(parents=True, exist_ok=True)
    logs_root = campaign_root / "logs"
    manual_targets_path = _write_resolved_targets(selected_targets, args, campaign_root)
    target_rows = _targets(manual_targets_path, args.limit_targets if not args.resolve_gaia_anchors else None, None)
    print("Magnitude ladder order:", " -> ".join(bin.name for bin in mag_bins), flush=True)

    env = os.environ.copy()
    env["SPHEREX_MANUAL_TARGETS_PATH"] = str(manual_targets_path)
    env["SPHEREX_CACHE_ROOT"] = str(args.cache_root)
    bin_path = str(REPO_ROOT / ".venv" / "bin" / "spherex-mine")
    py = str(REPO_ROOT / ".venv" / "bin" / "python")

    manifest: list[dict[str, Any]] = []
    total = len(target_rows) * len(mag_bins)
    count = 0
    for mag_bin in mag_bins:
        for target in target_rows:
            count += 1
            target_id = str(target["target_id"])
            run_name = f"{args.campaign_prefix}_{mag_bin.name}_{target_id}_baseline"
            run_dir = args.cache_root / "runs" / run_name
            print(f"\n##### {count}/{total} {mag_bin.name} {target_id} G={mag_bin.g_min}-{mag_bin.g_max} cap={mag_bin.max_gaia_sources} #####", flush=True)
            row: dict[str, Any] = {
                "campaign_prefix": args.campaign_prefix,
                "mag_bin": mag_bin.name,
                "gaia_g_min": mag_bin.g_min,
                "gaia_g_max": mag_bin.g_max,
                "max_gaia_sources": mag_bin.max_gaia_sources,
                "target_id": target_id,
                "object_name": target.get("object_name"),
                "run_name": run_name,
                "run_dir": str(run_dir),
                "status": "started",
                "stages": [],
            }
            _write_manifest(campaign_root, manifest + [row])
            depth_ok = _run_stage(
                row,
                "depth",
                [
                    bin_path,
                    "run-depth-test",
                    "--target",
                    target_id,
                    "--run-name",
                    run_name,
                    "--release",
                    "qr2",
                    "--limit-fields",
                    str(args.limit_fields),
                    "--max-gaia-sources",
                    str(mag_bin.max_gaia_sources),
                    "--gaia-g-min",
                    str(mag_bin.g_min),
                    "--gaia-g-max",
                    str(mag_bin.g_max),
                    "--max-field-workers",
                    str(args.max_field_workers),
                    "--photometry-backend",
                    "warp_calibrated",
                    "--warp-devices",
                    args.warp_devices,
                    "--status-mode",
                    "jsonl",
                    "--max-field-retries",
                    str(args.max_field_retries),
                    "--enable-psf",
                    "--psf-photometry-backend",
                    "warp_grid",
                    "--psf-kernel-build-mode",
                    "gpu_spline",
                    "--psf-grid-half-range-pix",
                    "1.0",
                    "--psf-grid-step-pix",
                    "0.5",
                    "--psf-grid-metric",
                    "snr",
                    "--cache-root",
                    str(args.cache_root),
                ],
                log_path=logs_root / f"{mag_bin.name}_{target_id}_depth.log",
                env=env,
                dry_run=args.dry_run,
                skip_done=(not args.force and _done(run_dir / "spectra" / "target_spectra.parquet")),
            )
            if depth_ok and args.blind_scan:
                _run_stage(
                    row,
                    "narrowband_classifier",
                    _narrowband_cmd(py, run_dir, run_dir / "narrowband_detector_raw", args),
                    log_path=logs_root / f"{mag_bin.name}_{target_id}_narrowband.log",
                    env=env,
                    dry_run=args.dry_run,
                    skip_done=(not args.force and _done(run_dir / "narrowband_detector_raw" / "narrowband_candidates.parquet")),
                )
            row["status"] = _row_status(row)
            row["run_summary"] = str(run_dir / "run_summary.json")
            row["spectra"] = str(run_dir / "spectra" / "target_spectra.parquet")
            row["spectrum_quality"] = str(run_dir / "spectra" / "spectrum_quality.parquet")
            row["narrowband_candidates"] = str(run_dir / "narrowband_detector_raw" / "narrowband_candidates.parquet")
            manifest.append(row)
            _write_manifest(campaign_root, manifest)

    summary = {
        "campaign_prefix": args.campaign_prefix,
        "campaign_root": str(campaign_root),
        "target_count": len(target_rows),
        "mag_bins": [bin.__dict__ for bin in mag_bins],
        "run_count": len(manifest),
        "done_count": sum(1 for row in manifest if row.get("status") == "done"),
        "error_count": sum(1 for row in manifest if row.get("status") == "error"),
        "skipped_count": sum(1 for row in manifest if row.get("status") == "skipped"),
        "manifest": str(campaign_root / "campaign_manifest.json"),
    }
    (campaign_root / "campaign_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _parse_mag_bins(values: list[str] | None) -> list[MagBin]:
    specs = values or [
        "g11_16:11:16:20000",
        "g8_11:8:11:20000",
        "g5_8:5:8:10000",
    ]
    bins: list[MagBin] = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 4:
            raise SystemExit(f"Bad --mag-bin {spec!r}; expected name:g_min:g_max:max_sources")
        bins.append(MagBin(parts[0], float(parts[1]), float(parts[2]), int(parts[3])))
    return bins


def _select_mag_bins(bins: list[MagBin], start_at: str | None, stop_after: str | None) -> list[MagBin]:
    names = [bin.name for bin in bins]
    start = names.index(start_at) if start_at else 0
    stop = names.index(stop_after) + 1 if stop_after else len(bins)
    selected = bins[start:stop]
    if not selected:
        raise SystemExit("No magnitude bins selected")
    return selected


def _run_stage(
    row: dict[str, Any],
    stage: str,
    cmd: list[str],
    *,
    log_path: Path,
    env: dict[str, str],
    dry_run: bool,
    skip_done: bool,
) -> bool:
    started = time.time()
    stage_row: dict[str, Any] = {
        "stage": stage,
        "status": "started",
        "log": str(log_path),
        "command": " ".join(cmd),
        "started_at_unix": started,
    }
    row.setdefault("stages", []).append(stage_row)
    if skip_done:
        stage_row["status"] = "skipped"
        stage_row["reason"] = "output already exists"
        stage_row["elapsed_sec"] = 0.0
        return True
    try:
        _run(cmd, log_path=log_path, env=env, dry_run=dry_run)
    except subprocess.CalledProcessError as exc:
        stage_row["status"] = "error"
        stage_row["returncode"] = exc.returncode
        stage_row["elapsed_sec"] = time.time() - started
        return False
    except Exception as exc:
        stage_row["status"] = "error"
        stage_row["error"] = f"{type(exc).__name__}: {exc}"
        stage_row["elapsed_sec"] = time.time() - started
        return False
    stage_row["status"] = "done"
    stage_row["elapsed_sec"] = time.time() - started
    return True


def _run(cmd: list[str], *, log_path: Path, env: dict[str, str], dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(cmd)
    print(f"\n=== {printable} ===", flush=True)
    if dry_run:
        log_path.write_text(printable + "\n", encoding="utf-8")
        return
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n\n=== {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {printable} ===\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)


def _narrowband_cmd(py: str, run_dir: Path, output_dir: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        py,
        "tools/warp_narrowband_detector.py",
        "--run-dir",
        str(run_dir),
        "--output-dir",
        str(output_dir),
        "--grid-step-nm",
        str(args.blind_grid_step_nm),
        "--min-joint-rho",
        str(args.narrowband_min_joint_rho),
        "--top-k-per-target",
        str(args.blind_top_k_per_target),
        "--device",
        args.blind_warp_device,
        "--quality-min-support",
        str(args.narrowband_quality_min_support),
        "--quality-max-flagged-points",
        str(args.narrowband_quality_max_flagged_points),
        "--quality-max-candidates-per-target",
        str(args.narrowband_quality_max_candidates_per_target),
        "--quality-max-aperture-psf-ratio",
        str(args.narrowband_quality_max_aperture_psf_ratio),
        "--diagnostic-line-half-window-nm",
        str(args.narrowband_diagnostic_line_half_window_nm),
        "--diagnostic-line-max-rows-per-candidate",
        str(args.narrowband_diagnostic_line_max_rows_per_candidate),
    ]
    if args.blind_top_k_min_separation_nm is not None:
        cmd.extend(["--top-k-min-separation-nm", str(args.blind_top_k_min_separation_nm)])
    return cmd


def _row_status(row: dict[str, Any]) -> str:
    stages = row.get("stages") or []
    if any(stage.get("status") == "error" for stage in stages):
        return "error"
    if stages and all(stage.get("status") == "skipped" for stage in stages):
        return "skipped"
    return "done"


def _write_manifest(campaign_root: Path, rows: list[dict[str, Any]]) -> None:
    path = campaign_root / "campaign_manifest.json"
    path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
