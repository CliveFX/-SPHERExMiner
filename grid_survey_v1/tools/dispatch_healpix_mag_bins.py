#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_TOOL = REPO_ROOT / "grid_survey_v1" / "tools" / "build_healpix_target_manifest.py"
INJECTION_RUNNER = REPO_ROOT / "tools" / "run_visible_sky_injection_campaign.py"
MAG_LADDER_RUNNER = REPO_ROOT / "tools" / "run_visible_sky_magnitude_ladder.py"
DIRECT_INJECTION_RUNNER = REPO_ROOT / "grid_survey_v1" / "tools" / "run_direct_grid_injection_batch.py"
SPHEREX_MINE = REPO_ROOT / ".venv" / "bin" / "spherex-mine"

DEFAULT_CACHE_ROOT = Path("/mnt/niroseti/spherex_cache")
DEFAULT_OUTPUT_ROOT = DEFAULT_CACHE_ROOT / "grid_survey_v1" / "dispatches"


@dataclass(frozen=True)
class MagBin:
    name: str
    g_min: float
    g_max: float
    max_sources: int


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build HEALPix target manifests for one or more Gaia magnitude bins "
            "and optionally dispatch the existing SPHEREx pipeline per batch."
        )
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--campaign-prefix", required=True)
    parser.add_argument("--nside", type=int, required=True)
    parser.add_argument("--hpx", type=int, action="append", help="HEALPix cell id. May be repeated.")
    parser.add_argument("--start-hpx", type=int, help="First HEALPix id for sequential generation.")
    parser.add_argument("--count", type=int, default=1, help="Number of sequential HEALPix ids from --start-hpx.")
    parser.add_argument("--order", choices=["nested", "ring"], default="nested")
    parser.add_argument(
        "--mag-bin",
        action="append",
        default=[],
        help="Magnitude bin as name:g_min:g_max:max_sources. May be repeated.",
    )
    parser.add_argument("--batch-size", type=int, default=3000)
    parser.add_argument("--limit-batches", type=int, help="Only dispatch the first N batch YAMLs per tile/bin.")
    parser.add_argument("--limit-fields", type=int, default=500)
    parser.add_argument("--max-field-workers", type=int, default=24)
    parser.add_argument("--warp-devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument(
        "--pipeline",
        choices=["baseline", "direct", "anchor_ladder", "injection"],
        default="baseline",
        help="'baseline' and 'direct' run one tile/batch directly. 'anchor_ladder' is the old per-target TOI mode.",
    )
    parser.add_argument("--blind-scan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--injection-strengths-sigma", default="5,8,12")
    parser.add_argument("--injection-targets-per-cell", type=int, default=3)
    parser.add_argument("--injection-max-lines-per-target", type=int, default=1)
    parser.add_argument("--injection-line-width-nm", type=float, default=1.0)
    parser.add_argument("--injection-max-line-flux-uJy", type=float)
    parser.add_argument("--force", action="store_true", help="Pass --force to the downstream campaign runner.")
    parser.add_argument("--overwrite-manifests", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually run downstream campaigns. Without this, only writes the plan.")
    args = parser.parse_args()

    mag_bins = _parse_mag_bins(args.mag_bin)
    hpx_args = _hpx_args(args)
    if not hpx_args:
        raise SystemExit("Provide --hpx or --start-hpx")

    dispatch_root = args.output_root / args.campaign_prefix
    manifest_root = dispatch_root / "manifests"
    log_root = dispatch_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    pause_file = dispatch_root / "PAUSE"
    stop_file = dispatch_root / "STOP"

    commands: list[dict[str, Any]] = []
    for mag_bin in mag_bins:
        bin_root = manifest_root / mag_bin.name
        build_cmd = [
            sys.executable,
            str(BUILD_TOOL),
            "--cache-root",
            str(args.cache_root),
            "--output-root",
            str(bin_root),
            "--nside",
            str(args.nside),
            "--order",
            args.order,
            "--g-min",
            str(mag_bin.g_min),
            "--g-max",
            str(mag_bin.g_max),
            "--max-sources",
            str(mag_bin.max_sources),
            "--batch-size",
            str(args.batch_size),
            *hpx_args,
        ]
        if args.overwrite_manifests:
            build_cmd.append("--overwrite")
        _run_or_print({"stage": "build_manifest", "mag_bin": asdict(mag_bin), "cmd": build_cmd}, log_root, True)

        survey_manifest = bin_root / "survey_manifest.json"
        if not survey_manifest.exists():
            if args.execute:
                raise SystemExit(f"Manifest build did not create {survey_manifest}")
            print(json.dumps({"status": "plan_only_manifest_missing", "path": str(survey_manifest)}), flush=True)
            continue
        manifest_doc = json.loads(survey_manifest.read_text(encoding="utf-8"))
        for tile in manifest_doc.get("tiles", []):
            batch_paths = list(tile.get("target_batch_yamls") or [tile["targets_yaml"]])
            if args.limit_batches is not None:
                batch_paths = batch_paths[: max(0, args.limit_batches)]
            for batch_index, batch_path in enumerate(batch_paths):
                command_spec = _campaign_command(
                    args=args,
                    mag_bin=mag_bin,
                    tile_id=str(tile["tile_id"]),
                    batch_index=batch_index,
                    targets_path=Path(batch_path),
                )
                command_row = {
                    "stage": "campaign",
                    "pipeline": args.pipeline,
                    "mag_bin": asdict(mag_bin),
                    "tile_id": tile["tile_id"],
                    "batch_index": batch_index,
                    "target_count": tile.get("target_count"),
                    "targets_path": batch_path,
                    "campaign_prefix": _campaign_name(args.campaign_prefix, mag_bin.name, str(tile["tile_id"]), batch_index),
                    "cmd": command_spec["cmd"],
                    "env": command_spec.get("env", {}),
                    "fixed_targets_path": command_spec.get("fixed_targets_path"),
                    "anchor_targets_path": command_spec.get("anchor_targets_path"),
                }
                commands.append(command_row)
                _write_dispatch_plan(args=args, dispatch_root=dispatch_root, pause_file=pause_file, stop_file=stop_file, mag_bins=mag_bins, commands=commands)
                if args.execute:
                    _wait_if_paused(pause_file, stop_file)
                _run_or_print(command_row, log_root, args.execute)

    plan_path = _write_dispatch_plan(
        args=args,
        dispatch_root=dispatch_root,
        pause_file=pause_file,
        stop_file=stop_file,
        mag_bins=mag_bins,
        commands=commands,
    )
    print(json.dumps({"status": "dispatch_plan", "path": str(plan_path), "commands": len(commands)}), flush=True)


def _parse_mag_bins(values: list[str]) -> list[MagBin]:
    if not values:
        raise SystemExit("Pass at least one --mag-bin name:g_min:g_max:max_sources")
    bins: list[MagBin] = []
    for spec in values:
        parts = spec.split(":")
        if len(parts) != 4:
            raise SystemExit(f"Bad --mag-bin {spec!r}; expected name:g_min:g_max:max_sources")
        name, g_min, g_max, max_sources = parts
        bins.append(MagBin(name=name, g_min=float(g_min), g_max=float(g_max), max_sources=int(max_sources)))
    return bins


def _hpx_args(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    for hpx in args.hpx or []:
        out.extend(["--hpx", str(hpx)])
    if args.start_hpx is not None:
        out.extend(["--start-hpx", str(args.start_hpx), "--count", str(args.count)])
    return out


def _write_dispatch_plan(
    *,
    args: argparse.Namespace,
    dispatch_root: Path,
    pause_file: Path,
    stop_file: Path,
    mag_bins: list[MagBin],
    commands: list[dict[str, Any]],
) -> Path:
    plan = {
        "created_utc": datetime.now(UTC).isoformat(),
        "campaign_prefix": args.campaign_prefix,
        "pipeline": args.pipeline,
        "execute_campaigns": bool(args.execute),
        "cache_root": str(args.cache_root),
        "dispatch_root": str(dispatch_root),
        "pause_file": str(pause_file),
        "stop_file": str(stop_file),
        "nside": args.nside,
        "order": args.order,
        "mag_bins": [asdict(bin) for bin in mag_bins],
        "command_count": len(commands),
        "commands": commands,
    }
    dispatch_root.mkdir(parents=True, exist_ok=True)
    plan_path = dispatch_root / "dispatch_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan_path


def _campaign_name(prefix: str, mag_bin_name: str, tile_id: str, batch_index: int) -> str:
    return f"{prefix}_{mag_bin_name}_{tile_id}_b{batch_index:04d}"


def _campaign_command(
    *,
    args: argparse.Namespace,
    mag_bin: MagBin,
    tile_id: str,
    batch_index: int,
    targets_path: Path,
) -> dict[str, Any]:
    campaign_prefix = _campaign_name(args.campaign_prefix, mag_bin.name, tile_id, batch_index)
    if args.pipeline in {"baseline", "direct"}:
        return _direct_tile_command(
            args=args,
            mag_bin=mag_bin,
            tile_id=tile_id,
            batch_index=batch_index,
            targets_path=targets_path,
            campaign_prefix=campaign_prefix,
        )
    if args.pipeline == "injection":
        return _direct_injection_command(
            args=args,
            mag_bin=mag_bin,
            tile_id=tile_id,
            batch_index=batch_index,
            targets_path=targets_path,
            campaign_prefix=campaign_prefix,
        )

    common = [
        "--targets",
        str(targets_path),
        "--cache-root",
        str(args.cache_root),
        "--campaign-prefix",
        campaign_prefix,
        "--limit-fields",
        str(args.limit_fields),
        "--max-field-workers",
        str(args.max_field_workers),
        "--warp-devices",
        args.warp_devices,
        "--no-resolve-gaia-anchors",
    ]
    if args.force:
        common.append("--force")

    if args.pipeline == "anchor_ladder":
        cmd = [
            sys.executable,
            str(MAG_LADDER_RUNNER),
            *common,
            "--mag-bin",
            f"{mag_bin.name}:{mag_bin.g_min}:{mag_bin.g_max}:{mag_bin.max_sources}",
            "--start-at-bin",
            mag_bin.name,
            "--stop-after-bin",
            mag_bin.name,
        ]
        cmd.append("--blind-scan" if args.blind_scan else "--no-blind-scan")
        return {"cmd": cmd}

    cmd = [
        sys.executable,
        str(INJECTION_RUNNER),
        *common,
        "--max-gaia-sources",
        str(mag_bin.max_sources),
        "--gaia-g-min",
        str(mag_bin.g_min),
        "--gaia-g-max",
        str(mag_bin.g_max),
        "--blind-scanner",
        "narrowband_gpu",
        "--blind-raw-recovery",
    ]
    cmd.append("--blind-scan" if args.blind_scan else "--no-blind-scan")
    return {"cmd": cmd}


def _direct_injection_command(
    *,
    args: argparse.Namespace,
    mag_bin: MagBin,
    tile_id: str,
    batch_index: int,
    targets_path: Path,
    campaign_prefix: str,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(DIRECT_INJECTION_RUNNER),
        "--cache-root",
        str(args.cache_root),
        "--campaign-prefix",
        campaign_prefix,
        "--tile-id",
        tile_id,
        "--batch-index",
        str(batch_index),
        "--targets-path",
        str(targets_path),
        "--mag-bin",
        f"{mag_bin.name}:{mag_bin.g_min}:{mag_bin.g_max}:{mag_bin.max_sources}",
        "--limit-fields",
        str(args.limit_fields),
        "--max-field-workers",
        str(args.max_field_workers),
        "--warp-devices",
        args.warp_devices,
        "--strengths-sigma",
        args.injection_strengths_sigma,
        "--targets-per-cell",
        str(args.injection_targets_per_cell),
        "--max-lines-per-target",
        str(args.injection_max_lines_per_target),
        "--line-width-nm",
        str(args.injection_line_width_nm),
    ]
    if args.injection_max_line_flux_uJy is not None:
        cmd.extend(["--max-line-flux-uJy", str(args.injection_max_line_flux_uJy)])
    if args.force:
        cmd.append("--force")
    return {"cmd": cmd}


def _direct_tile_command(
    *,
    args: argparse.Namespace,
    mag_bin: MagBin,
    tile_id: str,
    batch_index: int,
    targets_path: Path,
    campaign_prefix: str,
) -> dict[str, Any]:
    anchor_id, anchor_path, fixed_targets_path = _write_direct_inputs(
        tile_id=tile_id,
        targets_path=targets_path,
        campaign_prefix=campaign_prefix,
    )
    cmd = [
        str(SPHEREX_MINE if SPHEREX_MINE.exists() else "spherex-mine"),
        "run-depth-test",
        "--target",
        anchor_id,
        "--run-name",
        campaign_prefix,
        "--release",
        "qr2",
        "--limit-fields",
        str(args.limit_fields),
        "--max-gaia-sources",
        str(mag_bin.max_sources),
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
        "1",
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
        "--fixed-targets-path",
        str(fixed_targets_path),
        "--cache-root",
        str(args.cache_root),
    ]
    return {
        "cmd": cmd,
        "env": {"SPHEREX_MANUAL_TARGETS_PATH": str(anchor_path), "SPHEREX_CACHE_ROOT": str(args.cache_root)},
        "fixed_targets_path": str(fixed_targets_path),
        "anchor_targets_path": str(anchor_path),
    }


def _write_direct_inputs(*, tile_id: str, targets_path: Path, campaign_prefix: str) -> tuple[str, Path, Path]:
    direct_dir = targets_path.parent / "direct_inputs"
    direct_dir.mkdir(parents=True, exist_ok=True)
    anchor_id = f"grid_anchor_{tile_id}"
    ra_deg, dec_deg = _tile_center_from_id(tile_id)
    anchor_path = direct_dir / f"{_safe_name(campaign_prefix)}_anchor.yaml"
    fixed_targets_path = direct_dir / f"{_safe_name(campaign_prefix)}_fixed_targets.parquet"
    anchor_doc = {
        "targets": [
            {
                "target_id": anchor_id,
                "target_type": "healpix_tile_anchor",
                "object_name": f"HEALPix tile anchor {tile_id}",
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "reference_epoch_yr": 2016.0,
                "pmra_masyr": None,
                "pmdec_masyr": None,
                "parallax_mas": None,
                "source_catalog": "healpix",
                "source_catalog_id": tile_id,
                "priority_score": 0.0,
                "notes": "Synthetic tile-center anchor for direct grid survey dispatch.",
            }
        ]
    }
    anchor_path.write_text(yaml.safe_dump(anchor_doc, sort_keys=False), encoding="utf-8")
    fixed_rows = _fixed_target_rows_from_yaml(targets_path)
    pd.DataFrame(fixed_rows).to_parquet(fixed_targets_path, index=False)
    return anchor_id, anchor_path, fixed_targets_path


def _fixed_target_rows_from_yaml(targets_path: Path) -> list[dict[str, Any]]:
    doc = yaml.safe_load(targets_path.read_text(encoding="utf-8")) or {}
    targets = doc.get("targets") or []
    rows: list[dict[str, Any]] = []
    for row in targets:
        source_id = row.get("source_catalog_id")
        rows.append(
            {
                "target_id": str(row["target_id"]),
                "target_type": str(row.get("target_type") or "gaia_dr3_grid_survey"),
                "source_id": str(source_id) if source_id is not None else None,
                "object_name": row.get("object_name"),
                "ra_reference_deg": float(row["ra_deg"]),
                "dec_reference_deg": float(row["dec_deg"]),
                "reference_epoch_yr": float(row.get("reference_epoch_yr") or 2016.0),
                "pmra_masyr": row.get("pmra_masyr"),
                "pmdec_masyr": row.get("pmdec_masyr"),
                "parallax_mas": row.get("parallax_mas"),
                "priority_score": float(row.get("priority_score") or 0.0),
                "target_filter_flags": "grid_direct_fixed",
            }
        )
    if not rows:
        raise SystemExit(f"No targets in {targets_path}")
    return rows


def _tile_center_from_id(tile_id: str) -> tuple[float, float]:
    parts = tile_id.split("_")
    nside = int(parts[1].removeprefix("nside"))
    order = parts[2]
    hpx = int(parts[3])
    from astropy_healpix import HEALPix

    hp = HEALPix(nside=nside, order=order, frame=None)
    lon, lat = hp.healpix_to_lonlat(hpx)
    return float(lon.deg) % 360.0, float(lat.deg)


def _run_or_print(row: dict[str, Any], log_root: Path, execute: bool) -> None:
    cmd = [str(part) for part in row["cmd"]]
    preview = {key: value for key, value in row.items() if key != "cmd"}
    preview["cmd"] = cmd
    print(json.dumps({"status": "start" if execute else "planned", **preview}), flush=True)
    if not execute:
        return
    log_name = "_".join(str(preview.get(key, "")) for key in ("stage", "mag_bin", "tile_id", "batch_index"))
    log_name = _safe_name(log_name)[:180] or "command"
    log_path = log_root / f"{log_name}.log"
    env = None
    if row.get("env"):
        env = {**os.environ, **{str(k): str(v) for k, v in dict(row["env"]).items()}}
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False, env=env)
    print(json.dumps({"status": "done" if proc.returncode == 0 else "failed", "returncode": proc.returncode, "log": str(log_path)}), flush=True)


def _wait_if_paused(pause_file: Path, stop_file: Path) -> None:
    while pause_file.exists():
        if stop_file.exists():
            raise SystemExit(f"Stop requested: {stop_file}")
        print(json.dumps({"status": "paused", "pause_file": str(pause_file)}), flush=True)
        time.sleep(5.0)
    if stop_file.exists():
        raise SystemExit(f"Stop requested: {stop_file}")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


if __name__ == "__main__":
    main()
