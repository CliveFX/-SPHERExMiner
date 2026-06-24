from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spherex_laser_miner.catalog.local_gaia_lite import query_local_gaia_lite_duckdb

DEFAULT_CACHE_ROOT = Path("/mnt/niroseti/spherex_cache")
DEFAULT_TARGETS = REPO_ROOT / "configs" / "castro_valley_june_survey_targets.yaml"


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


def _targets(path: Path, limit: int | None, only: set[str] | None) -> list[dict[str, Any]]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = list(doc.get("targets") or [])
    if only:
        rows = [row for row in rows if str(row.get("target_id")) in only]
    if limit:
        rows = rows[: int(limit)]
    return rows


def _s_region_square(ra_deg: float, dec_deg: float, half_deg: float) -> str:
    return (
        f"POLYGON ICRS {ra_deg - half_deg} {dec_deg - half_deg} "
        f"{ra_deg + half_deg} {dec_deg - half_deg} "
        f"{ra_deg + half_deg} {dec_deg + half_deg} "
        f"{ra_deg - half_deg} {dec_deg + half_deg} "
        f"{ra_deg - half_deg} {dec_deg - half_deg}"
    )


def _distance_score(row: Any, ra_deg: float, dec_deg: float, preferred_g: float) -> float:
    dra = (float(row["ra"]) - ra_deg) * max(0.1, abs(math.cos(math.radians(dec_deg))))
    ddec = float(row["dec"]) - dec_deg
    sep = (dra * dra + ddec * ddec) ** 0.5
    mag_penalty = abs(float(row["phot_g_mean_mag"]) - preferred_g) * 0.15
    return sep + mag_penalty


def _resolve_gaia_anchor(center: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    ra = float(center["ra_deg"])
    dec = float(center["dec_deg"])
    for g_min, g_max in [(args.anchor_g_min, args.anchor_g_max), (args.gaia_g_min, args.gaia_g_max)]:
        df = query_local_gaia_lite_duckdb(
            s_region=_s_region_square(ra, dec, args.anchor_search_radius_deg),
            cache_root=args.cache_root,
            max_sources=500,
            g_min=float(g_min),
            g_max=float(g_max),
        )
        if df.empty:
            continue
        df = df.copy()
        df["_score"] = df.apply(lambda row: _distance_score(row, ra, dec, args.anchor_preferred_g), axis=1)
        picked = df.sort_values(["_score", "phot_g_mean_mag", "source_id"]).iloc[0]
        source_id = str(picked["source_id"])
        return {
            "target_id": f"{center['target_id']}_gaia_{source_id}",
            "target_type": "gaia_dr3_safe_anchor",
            "object_name": f"Gaia DR3 {source_id} near {center['object_name']}",
            "ra_deg": float(picked["ra"]),
            "dec_deg": float(picked["dec"]),
            "reference_epoch_yr": 2016.0,
            "pmra_masyr": _none_if_nan(picked.get("pmra")),
            "pmdec_masyr": _none_if_nan(picked.get("pmdec")),
            "parallax_mas": _none_if_nan(picked.get("parallax")),
            "source_catalog": "gaia_dr3",
            "source_catalog_id": source_id,
            "priority_score": 100.0,
            "notes": (
                f"Safe Gaia anchor selected near bright sky center {center['target_id']} "
                f"({center['object_name']}); G={float(picked['phot_g_mean_mag']):.3f}. "
                "Use this source to select SPHEREx frames; science sample is Gaia magnitude cut."
            ),
            "sky_center_id": center["target_id"],
            "sky_center_object_name": center["object_name"],
            "sky_center_ra_deg": ra,
            "sky_center_dec_deg": dec,
            "anchor_phot_g_mean_mag": float(picked["phot_g_mean_mag"]),
        }
    raise RuntimeError(f"No safe Gaia anchor found near {center['target_id']} within {args.anchor_search_radius_deg} deg")


def _none_if_nan(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return None if out != out else out


def _write_resolved_targets(centers: list[dict[str, Any]], args: argparse.Namespace, campaign_root: Path) -> Path:
    path = campaign_root / "resolved_gaia_anchor_targets.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not args.resolve_gaia_anchors:
        path.write_text(yaml.safe_dump({"targets": centers}, sort_keys=False), encoding="utf-8")
        print(f"copied {len(centers)} resolved Gaia anchors -> {path}", flush=True)
        return path
    resolved = [_resolve_gaia_anchor(center, args) for center in centers]
    path.write_text(yaml.safe_dump({"targets": resolved}, sort_keys=False), encoding="utf-8")
    print(f"resolved {len(resolved)} safe Gaia anchors -> {path}", flush=True)
    return path


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _done(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _has_zero_measured_parent(run_dir: Path) -> bool:
    qa = _json(run_dir / "qa.json")
    return bool(qa) and int(qa.get("trial_count") or 0) > 0 and int(qa.get("measured_count") or 0) == 0


def _write_manifest(campaign_root: Path, manifest_rows: list[dict[str, Any]]) -> None:
    (campaign_root / "campaign_manifest.json").write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")


def _skipped_manifest_row(
    *,
    target: dict[str, Any],
    baseline_run: Path,
    injected_run: Path,
    plan_path: Path,
    manifest_path: Path,
    recovery_dir: Path,
    review_path: Path,
    reason: str,
) -> dict[str, Any]:
    return {
        "target_id": str(target["target_id"]),
        "object_name": target.get("object_name"),
        "status": "skipped",
        "skip_reason": reason,
        "baseline_run": str(baseline_run),
        "injected_run": str(injected_run),
        "injection_plan": str(plan_path),
        "injection_manifest": str(manifest_path),
        "recovery_summary": str(recovery_dir / "recovery_summary.json"),
        "false_positive_review": str(review_path),
    }


def _false_positive_review(injected_run: Path, output_path: Path, viewer_base_url: str) -> dict[str, Any]:
    recovery_dir = injected_run / "recovery_score_mixed_lasers"
    false_positive_path = recovery_dir / "false_positive_candidates.parquet"
    summary_path = recovery_dir / "recovery_summary.json"
    summary = _json(summary_path)
    review = {
        "run_name": injected_run.name,
        "injected_run_dir": str(injected_run),
        "summary_path": str(summary_path),
        "false_positive_path": str(false_positive_path),
        "false_positive_count": summary.get("false_positive_count"),
        "candidate_count_above_threshold": summary.get("candidate_count_above_threshold"),
        "review_url": f"{viewer_base_url.rstrip('/')}/injections?run={injected_run.name}&status=candidate",
        "spectra_url": f"{viewer_base_url.rstrip('/')}/spectra?run={injected_run.name}",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(review, indent=2, sort_keys=True), encoding="utf-8")
    return review


def _blind_flux_kinds(value: str) -> list[str]:
    if value == "both":
        return ["aperture", "psf"]
    return [value]


def _run_blind_raw(
    *,
    py: str,
    run_dir: Path,
    output_dir: Path,
    flux_kind: str,
    args: argparse.Namespace,
    log_path: Path,
    env: dict[str, str],
    target_ids_file: Path | None = None,
) -> None:
    cmd = [
        py,
        "tools/warp_blind_scan.py",
        "--run-dir",
        str(run_dir),
        "--output-dir",
        str(output_dir),
        "--grid-step-nm",
        str(args.blind_grid_step_nm),
        "--flux-kind",
        flux_kind,
        "--min-snr",
        str(args.blind_raw_min_snr),
        "--min-supporting-points",
        str(args.blind_min_supporting_points),
        "--device",
        args.blind_warp_device,
    ]
    if target_ids_file is not None:
        cmd.extend(["--target-ids-file", str(target_ids_file)])
    if args.blind_candidate_mode == "topk":
        cmd.extend(["--candidate-mode", "topk", "--top-k-per-target", str(args.blind_top_k_per_target)])
        if args.blind_top_k_min_separation_nm is not None:
            cmd.extend(["--top-k-min-separation-nm", str(args.blind_top_k_min_separation_nm)])
    _run(cmd, log_path=log_path, env=env, dry_run=args.dry_run)


def _run_blind_paired(
    *,
    py: str,
    baseline_run: Path,
    injected_run: Path,
    output_dir: Path,
    flux_kind: str,
    args: argparse.Namespace,
    log_path: Path,
    env: dict[str, str],
) -> None:
    cmd = [
        py,
        "tools/warp_blind_scan.py",
        "--baseline-run-dir",
        str(baseline_run),
        "--injected-run-dir",
        str(injected_run),
        "--output-dir",
        str(output_dir),
        "--grid-step-nm",
        str(args.blind_grid_step_nm),
        "--flux-kind",
        flux_kind,
        "--min-snr",
        str(args.blind_min_snr),
        "--min-supporting-points",
        str(args.blind_min_supporting_points),
        "--device",
        args.blind_warp_device,
    ]
    if args.blind_candidate_mode == "topk":
        cmd.extend(["--candidate-mode", "topk", "--top-k-per-target", str(args.blind_top_k_per_target)])
        if args.blind_top_k_min_separation_nm is not None:
            cmd.extend(["--top-k-min-separation-nm", str(args.blind_top_k_min_separation_nm)])
    _run(cmd, log_path=log_path, env=env, dry_run=args.dry_run)


def _run_blind_joint(
    *,
    py: str,
    aperture_dir: Path,
    psf_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    log_path: Path,
    env: dict[str, str],
    quality_config: Path | None = None,
) -> None:
    cmd = [
            py,
            "tools/rank_blind_candidates.py",
            "--aperture-dir",
            str(aperture_dir),
            "--psf-dir",
            str(psf_dir),
            "--output-dir",
            str(output_dir),
            "--match-tolerance-nm",
            str(args.blind_joint_tolerance_nm),
    ]
    if quality_config is not None:
        cmd.extend(["--quality-config", str(quality_config)])
    _run(cmd, log_path=log_path, env=env, dry_run=args.dry_run)


def _write_injection_target_ids(manifest_path: Path, output_path: Path) -> Path:
    manifest = _json(manifest_path)
    target_ids = sorted({str(item.get("target_id")) for item in manifest.get("injections", []) if item.get("target_id")})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(target_ids) + ("\n" if target_ids else ""), encoding="utf-8")
    return output_path


def _run_blind_raw_recovery_score(
    *,
    py: str,
    manifest_path: Path,
    injected_run: Path,
    joint_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    log_path: Path,
    env: dict[str, str],
) -> None:
    _run(
        [
            py,
            "tools/score_blind_raw_recovery.py",
            "--manifest",
            str(manifest_path),
            "--injected-run-dir",
            str(injected_run),
            "--candidates",
            str(joint_dir / "blind_joint_candidates.parquet"),
            "--output-dir",
            str(output_dir),
            "--wavelength-tolerance-nm",
            str(args.wavelength_tolerance_nm),
        ],
        log_path=log_path,
        env=env,
        dry_run=args.dry_run,
    )


def _run_narrowband_detector(
    *,
    py: str,
    run_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    log_path: Path,
    env: dict[str, str],
    manifest_path: Path | None = None,
    target_ids_file: Path | None = None,
) -> None:
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
        "--recovery-tolerance-nm",
        str(args.wavelength_tolerance_nm),
        "--diagnostic-line-half-window-nm",
        str(args.narrowband_diagnostic_line_half_window_nm),
        "--diagnostic-line-max-rows-per-candidate",
        str(args.narrowband_diagnostic_line_max_rows_per_candidate),
    ]
    if args.blind_top_k_min_separation_nm is not None:
        cmd.extend(["--top-k-min-separation-nm", str(args.blind_top_k_min_separation_nm)])
    if manifest_path is not None:
        cmd.extend(["--manifest", str(manifest_path)])
    if target_ids_file is not None:
        cmd.extend(["--target-ids-file", str(target_ids_file)])
    _run(cmd, log_path=log_path, env=env, dry_run=args.dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline, injected, paired-delta recovery campaigns for visible June sky anchors."
    )
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--campaign-prefix", default="cv_june_g11_16_f500")
    parser.add_argument("--limit-targets", type=int)
    parser.add_argument("--only-target", action="append", help="Run one target_id; may be passed multiple times.")
    parser.add_argument("--limit-fields", type=int, default=500)
    parser.add_argument("--max-gaia-sources", type=int, default=6000)
    parser.add_argument("--gaia-g-min", type=float, default=11.0)
    parser.add_argument("--gaia-g-max", type=float, default=16.0)
    parser.add_argument("--max-field-workers", type=int, default=24)
    parser.add_argument("--warp-devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--resolve-gaia-anchors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--anchor-search-radius-deg", type=float, default=1.0)
    parser.add_argument("--anchor-g-min", type=float, default=12.0)
    parser.add_argument("--anchor-g-max", type=float, default=14.0)
    parser.add_argument("--anchor-preferred-g", type=float, default=13.0)
    parser.add_argument("--strengths-sigma", default="5,8,12")
    parser.add_argument("--targets-per-cell", type=int, default=3)
    parser.add_argument("--line-width-nm", type=float, default=1.0)
    parser.add_argument("--max-line-flux-uJy", type=float)
    parser.add_argument("--min-snr", type=float, default=5.0)
    parser.add_argument("--wavelength-tolerance-nm", type=float, default=10.0)
    parser.add_argument("--blind-scan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--blind-scanner", choices=["legacy", "narrowband_gpu"], default="narrowband_gpu")
    parser.add_argument("--blind-grid-step-nm", type=float, default=5.0)
    parser.add_argument("--blind-min-snr", type=float, default=5.0)
    parser.add_argument("--blind-raw-min-snr", type=float, default=5.0)
    parser.add_argument("--blind-min-supporting-points", type=int, default=2)
    parser.add_argument("--blind-flux-kind", choices=["aperture", "psf", "both"], default="both")
    parser.add_argument("--blind-raw-scan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--blind-warp-device", default="cuda:0")
    parser.add_argument("--blind-joint-tolerance-nm", type=float, default=10.0)
    parser.add_argument("--blind-candidate-mode", choices=["exhaustive", "topk"], default="topk")
    parser.add_argument("--blind-top-k-per-target", type=int, default=10)
    parser.add_argument("--blind-top-k-min-separation-nm", type=float)
    parser.add_argument("--narrowband-min-joint-rho", type=float, default=3.0)
    parser.add_argument("--narrowband-quality-min-support", type=int, default=3)
    parser.add_argument("--narrowband-quality-max-flagged-points", type=int, default=3)
    parser.add_argument("--narrowband-quality-max-candidates-per-target", type=int, default=5)
    parser.add_argument("--narrowband-quality-max-aperture-psf-ratio", type=float, default=3.0)
    parser.add_argument("--narrowband-diagnostic-line-half-window-nm", type=float, default=80.0)
    parser.add_argument("--narrowband-diagnostic-line-max-rows-per-candidate", type=int, default=201)
    parser.add_argument("--blind-science-quality-config", type=Path, default=REPO_ROOT / "configs" / "blind_candidate_quality.yaml")
    parser.add_argument(
        "--blind-injection-quality-config",
        type=Path,
        default=REPO_ROOT / "configs" / "blind_candidate_quality_injection.yaml",
    )
    parser.add_argument("--blind-raw-recovery", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--viewer-base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--force", action="store_true", help="Rerun stages even when expected outputs exist.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    target_rows = _targets(args.targets, args.limit_targets, set(args.only_target or []) or None)
    if not target_rows:
        raise SystemExit(f"No targets selected from {args.targets}")

    campaign_root = args.cache_root / "campaigns" / args.campaign_prefix
    campaign_root.mkdir(parents=True, exist_ok=True)
    manual_targets_path = _write_resolved_targets(target_rows, args, campaign_root)
    target_rows = _targets(manual_targets_path, args.limit_targets if not args.resolve_gaia_anchors else None, None)

    env = os.environ.copy()
    env["SPHEREX_MANUAL_TARGETS_PATH"] = str(manual_targets_path)
    env["SPHEREX_CACHE_ROOT"] = str(args.cache_root)
    bin_path = str(REPO_ROOT / ".venv" / "bin" / "spherex-mine")
    py = str(REPO_ROOT / ".venv" / "bin" / "python")
    manifest_rows: list[dict[str, Any]] = []

    for index, target in enumerate(target_rows, start=1):
        target_id = str(target["target_id"])
        base_name = f"{args.campaign_prefix}_{target_id}_baseline"
        inj_name = f"{args.campaign_prefix}_{target_id}_injected"
        injection_campaign_id = f"{args.campaign_prefix}_{target_id}_mixed_lasers"
        baseline_run = args.cache_root / "runs" / base_name
        injected_run = args.cache_root / "runs" / inj_name
        injection_campaign = args.cache_root / "injection_campaigns" / f"{injection_campaign_id}_s{args.strengths_sigma.replace(',', '_')}"
        plan_path = args.cache_root / "injection_campaigns" / injection_campaign_id / "injection_plan.json"
        overrides_path = injection_campaign / "path_overrides.json"
        manifest_path = injection_campaign / "injection_manifest.json"
        classifier_dir = injected_run / "classifier_paired_delta"
        recovery_dir = injected_run / "recovery_score_mixed_lasers"
        review_path = campaign_root / "false_positive_reviews" / f"{target_id}.json"
        print(f"\n##### target {index}/{len(target_rows)} {target_id} #####", flush=True)

        if not args.force and _has_zero_measured_parent(baseline_run):
            reason = "baseline has zero measured parent fields"
            print(f"skipping {target_id}: {reason}", flush=True)
            manifest_rows.append(
                _skipped_manifest_row(
                    target=target,
                    baseline_run=baseline_run,
                    injected_run=injected_run,
                    plan_path=plan_path,
                    manifest_path=manifest_path,
                    recovery_dir=recovery_dir,
                    review_path=review_path,
                    reason=reason,
                )
            )
            _write_manifest(campaign_root, manifest_rows)
            continue

        if args.force or not _done(baseline_run / "spectra" / "target_spectra.parquet"):
            _run(
                [
                    bin_path,
                    "run-depth-test",
                    "--target",
                    target_id,
                    "--run-name",
                    base_name,
                    "--release",
                    "qr2",
                    "--limit-fields",
                    str(args.limit_fields),
                    "--max-gaia-sources",
                    str(args.max_gaia_sources),
                    "--gaia-g-min",
                    str(args.gaia_g_min),
                    "--gaia-g-max",
                    str(args.gaia_g_max),
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
                    "--cache-root",
                    str(args.cache_root),
                ],
                log_path=campaign_root / "logs" / f"{target_id}_baseline.log",
                env=env,
                dry_run=args.dry_run,
            )

        if args.blind_scan and args.blind_raw_scan and args.blind_scanner == "narrowband_gpu":
            baseline_narrowband_dir = baseline_run / "narrowband_detector_raw"
            if args.force or not _done(baseline_narrowband_dir / "narrowband_candidates.parquet"):
                _run_narrowband_detector(
                    py=py,
                    run_dir=baseline_run,
                    output_dir=baseline_narrowband_dir,
                    args=args,
                    log_path=campaign_root / "logs" / f"{target_id}_narrowband_baseline_raw.log",
                    env=env,
                )
        elif args.blind_scan and args.blind_raw_scan:
            for flux_kind in _blind_flux_kinds(args.blind_flux_kind):
                baseline_blind_dir = baseline_run / f"blind_classifier_{flux_kind}_warp"
                if args.force or not _done(baseline_blind_dir / "blind_candidate_clusters.parquet"):
                    _run_blind_raw(
                        py=py,
                        run_dir=baseline_run,
                        output_dir=baseline_blind_dir,
                        flux_kind=flux_kind,
                        args=args,
                        log_path=campaign_root / "logs" / f"{target_id}_blind_baseline_{flux_kind}.log",
                        env=env,
                    )
            if args.blind_flux_kind == "both":
                baseline_joint_dir = baseline_run / "blind_classifier_joint_warp"
                if args.force or not _done(baseline_joint_dir / "blind_joint_candidates.parquet"):
                    _run_blind_joint(
                        py=py,
                        aperture_dir=baseline_run / "blind_classifier_aperture_warp",
                        psf_dir=baseline_run / "blind_classifier_psf_warp",
                        output_dir=baseline_joint_dir,
                        args=args,
                        log_path=campaign_root / "logs" / f"{target_id}_blind_baseline_joint.log",
                        env=env,
                        quality_config=args.blind_science_quality_config,
                    )

        if args.force or not _done(plan_path):
            plan_cmd = [
                py,
                "tools/make_mixed_laser_injection_plan.py",
                "--run-dir",
                str(baseline_run),
                "--campaign-id",
                injection_campaign_id,
                "--output-root",
                str(args.cache_root / "injection_campaigns"),
                "--strengths-sigma",
                args.strengths_sigma,
                "--targets-per-cell",
                str(args.targets_per_cell),
                "--line-width-nm",
                str(args.line_width_nm),
                "--min-measurements",
                "20",
                "--max-frames-per-injection",
                "60",
            ]
            if args.max_line_flux_uJy is not None:
                plan_cmd.extend(["--max-line-flux-uJy", str(args.max_line_flux_uJy)])
            _run(
                plan_cmd,
                log_path=campaign_root / "logs" / f"{target_id}_make_plan.log",
                env=env,
                dry_run=args.dry_run,
            )

        if args.force or not _done(overrides_path):
            _run(
                [
                    py,
                    "tools/run_injection_plan.py",
                    "--plan",
                    str(plan_path),
                    "--campaign-root",
                    str(injection_campaign),
                    "--cache-root",
                    str(args.cache_root),
                    "--release",
                    "qr2",
                    "--overwrite",
                    "--allow-accumulate-strengths",
                ],
                log_path=campaign_root / "logs" / f"{target_id}_inject.log",
                env=env,
                dry_run=args.dry_run,
            )

        if args.force or not _done(injected_run / "spectra" / "target_spectra.parquet"):
            _run(
                [
                    bin_path,
                    "run-depth-test",
                    "--target",
                    target_id,
                    "--run-name",
                    inj_name,
                    "--release",
                    "qr2",
                    "--limit-fields",
                    str(args.limit_fields),
                    "--max-gaia-sources",
                    str(args.max_gaia_sources),
                    "--gaia-g-min",
                    str(args.gaia_g_min),
                    "--gaia-g-max",
                    str(args.gaia_g_max),
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
                    "--cache-root",
                    str(args.cache_root),
                    "--path-overrides",
                    str(overrides_path),
                ],
                log_path=campaign_root / "logs" / f"{target_id}_injected.log",
                env=env,
                dry_run=args.dry_run,
            )

        if args.blind_scan and args.blind_scanner == "narrowband_gpu":
            injected_narrowband_dir = injected_run / "narrowband_detector_raw"
            if args.blind_raw_scan and (args.force or not _done(injected_narrowband_dir / "narrowband_candidates.parquet")):
                _run_narrowband_detector(
                    py=py,
                    run_dir=injected_run,
                    output_dir=injected_narrowband_dir,
                    args=args,
                    log_path=campaign_root / "logs" / f"{target_id}_narrowband_injected_raw.log",
                    env=env,
                )
            for flux_kind in _blind_flux_kinds(args.blind_flux_kind):
                paired_blind_dir = injected_run / f"blind_classifier_paired_delta_{flux_kind}_warp"
                if args.force or not _done(paired_blind_dir / "blind_candidate_clusters.parquet"):
                    _run_blind_paired(
                        py=py,
                        baseline_run=baseline_run,
                        injected_run=injected_run,
                        output_dir=paired_blind_dir,
                        flux_kind=flux_kind,
                        args=args,
                        log_path=campaign_root / "logs" / f"{target_id}_blind_paired_delta_{flux_kind}.log",
                        env=env,
                    )
            if args.blind_flux_kind == "both":
                joint_scopes = [("paired_delta", injected_run)]
                if args.blind_raw_scan and args.blind_scanner == "legacy":
                    joint_scopes.insert(0, ("injected", injected_run))
                for scope, parent in joint_scopes:
                    aperture_dir = parent / ("blind_classifier_aperture_warp" if scope == "injected" else "blind_classifier_paired_delta_aperture_warp")
                    psf_dir = parent / ("blind_classifier_psf_warp" if scope == "injected" else "blind_classifier_paired_delta_psf_warp")
                    joint_dir = parent / ("blind_classifier_joint_warp" if scope == "injected" else "blind_classifier_paired_delta_joint_warp")
                    if args.force or not _done(joint_dir / "blind_joint_candidates.parquet"):
                        _run_blind_joint(
                            py=py,
                            aperture_dir=aperture_dir,
                            psf_dir=psf_dir,
                            output_dir=joint_dir,
                            args=args,
                            log_path=campaign_root / "logs" / f"{target_id}_blind_{scope}_joint.log",
                            env=env,
                            quality_config=args.blind_injection_quality_config if scope == "injected" else args.blind_injection_quality_config,
                        )
        elif args.blind_scan:
            for flux_kind in _blind_flux_kinds(args.blind_flux_kind):
                injected_blind_dir = injected_run / f"blind_classifier_{flux_kind}_warp"
                paired_blind_dir = injected_run / f"blind_classifier_paired_delta_{flux_kind}_warp"
                if args.blind_raw_scan and (args.force or not _done(injected_blind_dir / "blind_candidate_clusters.parquet")):
                    _run_blind_raw(
                        py=py,
                        run_dir=injected_run,
                        output_dir=injected_blind_dir,
                        flux_kind=flux_kind,
                        args=args,
                        log_path=campaign_root / "logs" / f"{target_id}_blind_injected_{flux_kind}.log",
                        env=env,
                    )
                if args.force or not _done(paired_blind_dir / "blind_candidate_clusters.parquet"):
                    _run_blind_paired(
                        py=py,
                        baseline_run=baseline_run,
                        injected_run=injected_run,
                        output_dir=paired_blind_dir,
                        flux_kind=flux_kind,
                        args=args,
                        log_path=campaign_root / "logs" / f"{target_id}_blind_paired_delta_{flux_kind}.log",
                        env=env,
                    )
            if args.blind_flux_kind == "both":
                joint_scopes = [("paired_delta", injected_run)]
                if args.blind_raw_scan:
                    joint_scopes.insert(0, ("injected", injected_run))
                for scope, parent in joint_scopes:
                    aperture_dir = parent / ("blind_classifier_aperture_warp" if scope == "injected" else "blind_classifier_paired_delta_aperture_warp")
                    psf_dir = parent / ("blind_classifier_psf_warp" if scope == "injected" else "blind_classifier_paired_delta_psf_warp")
                    joint_dir = parent / ("blind_classifier_joint_warp" if scope == "injected" else "blind_classifier_paired_delta_joint_warp")
                    if args.force or not _done(joint_dir / "blind_joint_candidates.parquet"):
                        _run_blind_joint(
                            py=py,
                            aperture_dir=aperture_dir,
                            psf_dir=psf_dir,
                            output_dir=joint_dir,
                            args=args,
                            log_path=campaign_root / "logs" / f"{target_id}_blind_{scope}_joint.log",
                            env=env,
                            quality_config=args.blind_injection_quality_config if scope == "injected" else args.blind_injection_quality_config,
                        )

        if args.blind_scan and args.blind_raw_recovery and args.blind_scanner == "narrowband_gpu":
            truth_ids_path = injected_run / "blind_raw_recovery_truth_target_ids.txt"
            if args.force or not _done(truth_ids_path):
                _write_injection_target_ids(manifest_path, truth_ids_path)
            truth_narrowband_dir = injected_run / "narrowband_detector_truth"
            if args.force or not _done(truth_narrowband_dir / "narrowband_recovery.parquet"):
                _run_narrowband_detector(
                    py=py,
                    run_dir=injected_run,
                    output_dir=truth_narrowband_dir,
                    args=args,
                    log_path=campaign_root / "logs" / f"{target_id}_narrowband_raw_recovery.log",
                    env=env,
                    manifest_path=manifest_path,
                    target_ids_file=truth_ids_path,
                )
        elif args.blind_scan and args.blind_raw_recovery and args.blind_flux_kind == "both":
            truth_ids_path = injected_run / "blind_raw_recovery_truth_target_ids.txt"
            if args.force or not _done(truth_ids_path):
                _write_injection_target_ids(manifest_path, truth_ids_path)
            truth_aperture_dir = injected_run / "blind_classifier_injected_raw_truth_aperture_topk"
            truth_psf_dir = injected_run / "blind_classifier_injected_raw_truth_psf_topk"
            truth_joint_dir = injected_run / "blind_classifier_injected_raw_truth_joint_topk"
            truth_score_dir = injected_run / "blind_raw_recovery_truth_topk"
            if args.force or not _done(truth_aperture_dir / "blind_candidate_clusters.parquet"):
                _run_blind_raw(
                    py=py,
                    run_dir=injected_run,
                    output_dir=truth_aperture_dir,
                    flux_kind="aperture",
                    args=args,
                    log_path=campaign_root / "logs" / f"{target_id}_blind_raw_recovery_aperture.log",
                    env=env,
                    target_ids_file=truth_ids_path,
                )
            if args.force or not _done(truth_psf_dir / "blind_candidate_clusters.parquet"):
                _run_blind_raw(
                    py=py,
                    run_dir=injected_run,
                    output_dir=truth_psf_dir,
                    flux_kind="psf",
                    args=args,
                    log_path=campaign_root / "logs" / f"{target_id}_blind_raw_recovery_psf.log",
                    env=env,
                    target_ids_file=truth_ids_path,
                )
            if args.force or not _done(truth_joint_dir / "blind_joint_candidates.parquet"):
                _run_blind_joint(
                    py=py,
                    aperture_dir=truth_aperture_dir,
                    psf_dir=truth_psf_dir,
                    output_dir=truth_joint_dir,
                    args=args,
                    log_path=campaign_root / "logs" / f"{target_id}_blind_raw_recovery_joint.log",
                    env=env,
                    quality_config=args.blind_injection_quality_config,
                )
            if args.force or not _done(truth_score_dir / "blind_raw_recovery_summary.json"):
                _run_blind_raw_recovery_score(
                    py=py,
                    manifest_path=manifest_path,
                    injected_run=injected_run,
                    joint_dir=truth_joint_dir,
                    output_dir=truth_score_dir,
                    args=args,
                    log_path=campaign_root / "logs" / f"{target_id}_blind_raw_recovery_score.log",
                    env=env,
                )

        if args.force or not _done(classifier_dir / "matched_filter_candidates.parquet"):
            _run(
                [
                    py,
                    "tools/classify_paired_delta_matched_filter.py",
                    "--baseline-run-dir",
                    str(baseline_run),
                    "--injected-run-dir",
                    str(injected_run),
                    "--plan",
                    str(plan_path),
                    "--output-dir",
                    str(classifier_dir),
                    "--flux-kind",
                    "aperture",
                    "--min-snr",
                    str(args.min_snr),
                ],
                log_path=campaign_root / "logs" / f"{target_id}_classify.log",
                env=env,
                dry_run=args.dry_run,
            )

        if args.force or not _done(recovery_dir / "recovery_summary.json"):
            _run(
                [
                    py,
                    "tools/score_injection_recovery.py",
                    "--manifest",
                    str(manifest_path),
                    "--candidates",
                    str(classifier_dir / "matched_filter_candidates.parquet"),
                    "--output-dir",
                    str(recovery_dir),
                    "--min-snr",
                    str(args.min_snr),
                    "--wavelength-tolerance-nm",
                    str(args.wavelength_tolerance_nm),
                ],
                log_path=campaign_root / "logs" / f"{target_id}_score.log",
                env=env,
                dry_run=args.dry_run,
            )

        review = {} if args.dry_run else _false_positive_review(injected_run, review_path, args.viewer_base_url)
        manifest_rows.append(
            {
                "target_id": target_id,
                "object_name": target.get("object_name"),
                "baseline_run": str(baseline_run),
                "injected_run": str(injected_run),
                "injection_plan": str(plan_path),
                "injection_manifest": str(manifest_path),
                "recovery_summary": str(recovery_dir / "recovery_summary.json"),
                "baseline_blind_joint_summary": str(
                    baseline_run
                    / ("narrowband_detector_raw/narrowband_detector_summary.json" if args.blind_scanner == "narrowband_gpu" else "blind_classifier_joint_warp/blind_joint_summary.json")
                ),
                "injected_blind_joint_summary": str(
                    injected_run
                    / ("narrowband_detector_raw/narrowband_detector_summary.json" if args.blind_scanner == "narrowband_gpu" else "blind_classifier_joint_warp/blind_joint_summary.json")
                ),
                "paired_blind_joint_summary": str(injected_run / "blind_classifier_paired_delta_joint_warp" / "blind_joint_summary.json"),
                "blind_raw_recovery_summary": str(
                    injected_run
                    / ("narrowband_detector_truth/narrowband_detector_summary.json" if args.blind_scanner == "narrowband_gpu" else "blind_raw_recovery_truth_topk/blind_raw_recovery_summary.json")
                ),
                "false_positive_review": str(review_path),
                "review_url": review.get("review_url"),
            }
        )
        _write_manifest(campaign_root, manifest_rows)

    print(json.dumps({"campaign_root": str(campaign_root), "targets": len(manifest_rows)}, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"command failed with exit {exc.returncode}: {' '.join(exc.cmd)}", file=sys.stderr)
        raise
