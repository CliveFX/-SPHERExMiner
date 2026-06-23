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
    if not args.resolve_gaia_anchors:
        return args.targets
    resolved = [_resolve_gaia_anchor(center, args) for center in centers]
    path = campaign_root / "resolved_gaia_anchor_targets.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"targets": resolved}, sort_keys=False), encoding="utf-8")
    print(f"resolved {len(resolved)} safe Gaia anchors -> {path}", flush=True)
    return path


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _done(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


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
    parser.add_argument("--min-snr", type=float, default=5.0)
    parser.add_argument("--wavelength-tolerance-nm", type=float, default=10.0)
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

        if args.force or not _done(plan_path):
            _run(
                [
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
                ],
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
                "false_positive_review": str(review_path),
                "review_url": review.get("review_url"),
            }
        )
        (campaign_root / "campaign_manifest.json").write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")

    print(json.dumps({"campaign_root": str(campaign_root), "targets": len(manifest_rows)}, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"command failed with exit {exc.returncode}: {' '.join(exc.cmd)}", file=sys.stderr)
        raise
