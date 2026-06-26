#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grid_survey_v1.tools.dispatch_healpix_mag_bins import (  # noqa: E402
    MagBin,
    _direct_tile_command,
    _safe_name,
)


DEFAULT_CACHE_ROOT = Path("/mnt/niroseti/spherex_cache")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one direct grid baseline/injected/recovery batch.")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--campaign-prefix", required=True)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--batch-index", type=int, required=True)
    parser.add_argument("--targets-path", type=Path, required=True)
    parser.add_argument("--mag-bin", required=True, help="name:g_min:g_max:max_sources")
    parser.add_argument("--catalog", choices=["gaia", "2mass", "all"], default="gaia")
    parser.add_argument("--twomass-band", choices=["J", "H", "Ks"], default="Ks")
    parser.add_argument("--twomass-quality", default="ABC")
    parser.add_argument("--twomass-dataset-name", default="psc_lite")
    parser.add_argument("--twomass-hpx-level", type=int, default=5)
    parser.add_argument("--twomass-selection", choices=["stratified", "brightest", "random"], default="stratified")
    parser.add_argument("--limit-fields", type=int, default=500)
    parser.add_argument("--max-field-workers", type=int, default=24)
    parser.add_argument("--warp-devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--strengths-sigma", default="5,8,12")
    parser.add_argument("--targets-per-cell", type=int, default=3)
    parser.add_argument("--max-lines-per-target", type=int, default=1)
    parser.add_argument("--line-width-nm", type=float, default=1.0)
    parser.add_argument("--min-injection-measurements", type=int, default=3)
    parser.add_argument("--max-line-flux-uJy", type=float)
    parser.add_argument("--blind-grid-step-nm", type=float, default=1.0)
    parser.add_argument("--blind-min-supporting-points", type=int, default=2)
    parser.add_argument("--blind-top-k-per-target", type=int, default=8)
    parser.add_argument("--blind-min-joint-rho", type=float, default=5.0)
    parser.add_argument("--paired-min-snr", type=float, default=5.0)
    parser.add_argument("--paired-wavelength-tolerance-nm", type=float, default=10.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    mag_bin = _parse_mag_bin(args.mag_bin)
    root = args.cache_root / "grid_survey_v1" / "direct_injection" / args.campaign_prefix
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    baseline_name = f"{args.campaign_prefix}_baseline"
    injected_name = f"{args.campaign_prefix}_injected"
    baseline_run = args.cache_root / "runs" / baseline_name
    injected_run = args.cache_root / "runs" / injected_name
    injection_id = f"{args.campaign_prefix}_mixed_lasers"
    injection_root = args.cache_root / "injection_campaigns" / f"{injection_id}_s{_safe_name(args.strengths_sigma.replace(',', '_'))}"
    plan_path = args.cache_root / "injection_campaigns" / injection_id / "injection_plan.json"
    manifest_path = injection_root / "injection_manifest.json"
    overrides_path = injection_root / "path_overrides.json"

    summary: dict[str, Any] = {
        "campaign_prefix": args.campaign_prefix,
        "tile_id": args.tile_id,
        "batch_index": args.batch_index,
        "targets_path": str(args.targets_path),
        "baseline_run": str(baseline_run),
        "injected_run": str(injected_run),
        "injection_plan": str(plan_path),
        "injection_manifest": str(manifest_path),
        "catalog": args.catalog,
    }

    baseline_spec = _direct_tile_command(
        args=_direct_args(args, baseline_name),
        mag_bin=mag_bin,
        tile_id=args.tile_id,
        batch_index=args.batch_index,
        targets_path=args.targets_path,
        campaign_prefix=baseline_name,
    )
    _run_stage("baseline", baseline_spec["cmd"], logs / "baseline.log", env=baseline_spec.get("env"))
    summary["fixed_targets_path"] = baseline_spec.get("fixed_targets_path")
    summary["anchor_targets_path"] = baseline_spec.get("anchor_targets_path")

    baseline_raw_dir = baseline_run / "narrowband_detector_raw"
    if args.force or not (baseline_raw_dir / "narrowband_candidates.parquet").exists():
        _run_stage(
            "baseline_narrowband_raw",
            _narrowband_detector_cmd(args=args, run_dir=baseline_run, output_dir=baseline_raw_dir),
            logs / "baseline_narrowband_raw.log",
        )

    _run_stage(
        "make_injection_plan",
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "make_mixed_laser_injection_plan.py"),
            "--run-dir",
            str(baseline_run),
            "--campaign-id",
            injection_id,
            "--output-root",
            str(args.cache_root / "injection_campaigns"),
            "--strengths-sigma",
            args.strengths_sigma,
            "--targets-per-cell",
            str(args.targets_per_cell),
            "--max-lines-per-target",
            str(args.max_lines_per_target),
            "--line-width-nm",
            str(args.line_width_nm),
            "--min-measurements",
            str(args.min_injection_measurements),
            *([] if args.max_line_flux_uJy is None else ["--max-line-flux-uJy", str(args.max_line_flux_uJy)]),
        ],
        logs / "make_injection_plan.log",
    )

    inject_result = _run_stage(
        "inject",
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_injection_plan.py"),
            "--plan",
            str(plan_path),
            "--campaign-root",
            str(injection_root),
            "--cache-root",
            str(args.cache_root),
            "--overwrite",
            "--allow-accumulate-strengths",
        ],
        logs / "inject.log",
        allow_no_injections=True,
    )
    if inject_result == "no_injections":
        summary.update({"status": "no_eligible_injections", "reason": "Injection planner found no eligible target/line support."})
        summary_path = root / "direct_injection_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"status": "no_eligible_injections", "summary": str(summary_path), **summary}, indent=2), flush=True)
        return

    injected_spec = _direct_tile_command(
        args=_direct_args(args, injected_name),
        mag_bin=mag_bin,
        tile_id=args.tile_id,
        batch_index=args.batch_index,
        targets_path=args.targets_path,
        campaign_prefix=injected_name,
    )
    injected_cmd = [*injected_spec["cmd"], "--path-overrides", str(overrides_path)]
    _run_stage("injected", injected_cmd, logs / "injected.log", env=injected_spec.get("env"))

    injected_raw_dir = injected_run / "narrowband_detector_raw"
    if args.force or not (injected_raw_dir / "narrowband_candidates.parquet").exists():
        _run_stage(
            "injected_narrowband_raw",
            _narrowband_detector_cmd(args=args, run_dir=injected_run, output_dir=injected_raw_dir),
            logs / "injected_narrowband_raw.log",
        )

    truth_ids_path = injected_run / "blind_raw_recovery_truth_target_ids.txt"
    _write_truth_ids(manifest_path, truth_ids_path)
    detector_dir = injected_run / "narrowband_detector_truth"
    _run_stage(
        "narrowband_truth",
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "warp_narrowband_detector.py"),
            "--run-dir",
            str(injected_run),
            "--manifest",
            str(manifest_path),
            "--target-ids-file",
            str(truth_ids_path),
            "--output-dir",
            str(detector_dir),
            "--grid-step-nm",
            str(args.blind_grid_step_nm),
            "--min-supporting-points",
            str(args.blind_min_supporting_points),
            "--top-k-per-target",
            str(args.blind_top_k_per_target),
            "--min-joint-rho",
            str(args.blind_min_joint_rho),
            "--device",
            args.device,
            "--allow-non-good-spectra",
        ],
        logs / "narrowband_truth.log",
    )

    paired_dir = injected_run / "classifier_paired_delta"
    _run_stage(
        "paired_delta_classifier",
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "classify_paired_delta_matched_filter.py"),
            "--baseline-run-dir",
            str(baseline_run),
            "--injected-run-dir",
            str(injected_run),
            "--plan",
            str(plan_path),
            "--output-dir",
            str(paired_dir),
            "--flux-kind",
            "aperture",
            "--min-snr",
            str(args.paired_min_snr),
        ],
        logs / "paired_delta_classifier.log",
    )

    paired_recovery_dir = injected_run / "recovery_score_mixed_lasers"
    _run_stage(
        "paired_delta_recovery",
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "score_injection_recovery.py"),
            "--manifest",
            str(manifest_path),
            "--candidates",
            str(paired_dir / "matched_filter_candidates.parquet"),
            "--output-dir",
            str(paired_recovery_dir),
            "--min-snr",
            str(args.paired_min_snr),
            "--wavelength-tolerance-nm",
            str(args.paired_wavelength_tolerance_nm),
        ],
        logs / "paired_delta_recovery.log",
    )

    recovery_path = detector_dir / "narrowband_recovery.parquet"
    detector_summary = detector_dir / "narrowband_detector_summary.json"
    paired_summary = paired_recovery_dir / "recovery_summary.json"
    summary.update(
        {
            "injected_run": str(injected_run),
            "injection_manifest": str(manifest_path),
            "path_overrides": str(overrides_path),
            "truth_detector_dir": str(detector_dir),
            "baseline_narrowband_raw_dir": str(baseline_raw_dir),
            "baseline_narrowband_raw_summary": str(baseline_raw_dir / "narrowband_detector_summary.json"),
            "injected_narrowband_raw_dir": str(injected_raw_dir),
            "injected_narrowband_raw_summary": str(injected_raw_dir / "narrowband_detector_summary.json"),
            "narrowband_recovery": str(recovery_path),
            "narrowband_detector_summary": str(detector_summary),
            "paired_classifier_dir": str(paired_dir),
            "paired_recovery_dir": str(paired_recovery_dir),
            "paired_recovery_summary": str(paired_summary),
            "recovery": _read_detector_recovery(detector_summary),
            "paired_recovery": _read_paired_recovery(paired_summary),
        }
    )
    summary_path = root / "direct_injection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "done", "summary": str(summary_path), **summary}, indent=2), flush=True)


def _parse_mag_bin(spec: str) -> MagBin:
    name, g_min, g_max, max_sources = spec.split(":")
    return MagBin(name=name, g_min=float(g_min), g_max=float(g_max), max_sources=int(max_sources))


def _direct_args(args: argparse.Namespace, run_name: str) -> argparse.Namespace:
    return argparse.Namespace(
        cache_root=args.cache_root,
        limit_fields=args.limit_fields,
        max_field_workers=args.max_field_workers,
        warp_devices=args.warp_devices,
        force=args.force,
        campaign_prefix=run_name,
        catalog=args.catalog,
        twomass_band=args.twomass_band,
        twomass_quality=args.twomass_quality,
        twomass_dataset_name=args.twomass_dataset_name,
        twomass_hpx_level=args.twomass_hpx_level,
        twomass_selection=args.twomass_selection,
    )


def _narrowband_detector_cmd(args: argparse.Namespace, run_dir: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "tools" / "warp_narrowband_detector.py"),
        "--run-dir",
        str(run_dir),
        "--output-dir",
        str(output_dir),
        "--grid-step-nm",
        str(args.blind_grid_step_nm),
        "--min-supporting-points",
        str(args.blind_min_supporting_points),
        "--top-k-per-target",
        str(args.blind_top_k_per_target),
        "--min-joint-rho",
        str(args.blind_min_joint_rho),
        "--device",
        args.device,
    ]


def _run_stage(
    stage: str,
    cmd: list[str],
    log_path: Path,
    env: dict[str, str] | None = None,
    *,
    allow_no_injections: bool = False,
) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_env = os.environ.copy()
    if env:
        full_env.update({str(k): str(v) for k, v in env.items()})
    print(json.dumps({"status": "start", "stage": stage, "cmd": cmd, "log": str(log_path)}), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run([str(part) for part in cmd], cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, env=full_env)
    print(json.dumps({"status": "done" if proc.returncode == 0 else "failed", "stage": stage, "returncode": proc.returncode, "log": str(log_path)}), flush=True)
    if proc.returncode != 0:
        if allow_no_injections and "Plan has no injections to run" in log_path.read_text(encoding="utf-8", errors="ignore"):
            return "no_injections"
        raise SystemExit(f"{stage} failed; see {log_path}")
    return "done"


def _write_truth_ids(manifest_path: Path, output_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = sorted({str(item["target_id"]) for item in manifest.get("injections", []) if item.get("target_id")})
    output_path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def _read_detector_recovery(summary_path: Path) -> dict[str, Any] | None:
    if not summary_path.exists():
        return None
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return data.get("recovery") if isinstance(data, dict) else None


def _read_paired_recovery(summary_path: Path) -> dict[str, Any] | None:
    if not summary_path.exists():
        return None
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    keys = (
        "injection_count",
        "recovered_count",
        "recovery_fraction",
        "candidate_count_above_threshold",
        "false_positive_count",
        "false_positives_per_injection",
    )
    return {key: data.get(key) for key in keys}


if __name__ == "__main__":
    main()
