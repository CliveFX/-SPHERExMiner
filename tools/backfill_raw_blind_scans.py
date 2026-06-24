from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = Path("/mnt/niroseti/spherex_cache")


def _done(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(cmd)
    print(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {printable} ===", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n\n=== {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {printable} ===\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
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


def _baseline_runs(cache_root: Path, campaign_prefix: str | None) -> list[Path]:
    runs_root = cache_root / "runs"
    if not runs_root.exists():
        return []
    runs = []
    for path in runs_root.iterdir():
        if not path.is_dir() or not path.name.endswith("_baseline"):
            continue
        if campaign_prefix and not path.name.startswith(campaign_prefix):
            continue
        if not _done(path / "spectra" / "target_spectra.parquet"):
            continue
        runs.append(path)
    return sorted(runs, key=lambda path: path.stat().st_mtime)


def _scan_cmd(
    *,
    py: str,
    run_dir: Path,
    output_dir: Path,
    flux_kind: str,
    grid_step_nm: float,
    min_snr: float,
    min_supporting_points: int,
    device: str,
    candidate_mode: str,
    top_k_per_target: int,
    top_k_min_separation_nm: float | None,
) -> list[str]:
    cmd = [
        py,
        "tools/warp_blind_scan.py",
        "--run-dir",
        str(run_dir),
        "--output-dir",
        str(output_dir),
        "--grid-step-nm",
        str(grid_step_nm),
        "--flux-kind",
        flux_kind,
        "--min-snr",
        str(min_snr),
        "--min-supporting-points",
        str(min_supporting_points),
        "--device",
        device,
    ]
    if candidate_mode == "topk":
        cmd.extend(["--candidate-mode", "topk", "--top-k-per-target", str(top_k_per_target)])
        if top_k_min_separation_nm is not None:
            cmd.extend(["--top-k-min-separation-nm", str(top_k_min_separation_nm)])
    return cmd


def _rank_cmd(py: str, run_dir: Path, output_dir: Path, match_tolerance_nm: float) -> list[str]:
    return [
        py,
        "tools/rank_blind_candidates.py",
        "--aperture-dir",
        str(run_dir / "blind_classifier_aperture_warp"),
        "--psf-dir",
        str(run_dir / "blind_classifier_psf_warp"),
        "--output-dir",
        str(output_dir),
        "--match-tolerance-nm",
        str(match_tolerance_nm),
    ]


def _device_list(value: str) -> list[str]:
    devices = [item.strip() for item in value.split(",") if item.strip()]
    return devices or ["cuda:0"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill raw blind candidate scans for baseline science runs.")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--campaign-prefix", default="cv_june_g11_16_f500_wideinj")
    parser.add_argument("--limit-runs", type=int)
    parser.add_argument("--grid-step-nm", type=float, default=1.0)
    parser.add_argument("--min-snr", type=float, default=5.0)
    parser.add_argument("--min-supporting-points", type=int, default=2)
    parser.add_argument("--device", default="cuda:0", help="Single Warp device. Kept for compatibility; ignored when --devices is set.")
    parser.add_argument("--devices", default="", help="Comma-separated Warp devices for threaded scans, e.g. cuda:0,cuda:1,cuda:2.")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--candidate-mode", choices=["exhaustive", "topk"], default="topk")
    parser.add_argument("--top-k-per-target", type=int, default=10)
    parser.add_argument("--top-k-min-separation-nm", type=float)
    parser.add_argument("--match-tolerance-nm", type=float, default=10.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    py = str(REPO_ROOT / ".venv" / "bin" / "python")
    log_root = args.cache_root / "campaigns" / args.campaign_prefix / "logs"
    runs = _baseline_runs(args.cache_root, args.campaign_prefix)
    if args.limit_runs:
        runs = runs[: args.limit_runs]
    devices = _device_list(args.devices or args.device)
    max_workers = max(1, int(args.max_workers))
    print(
        f"raw blind backfill: {len(runs)} baseline runs, max_workers={max_workers}, devices={','.join(devices)}, min_snr={args.min_snr}",
        flush=True,
    )

    scan_jobs: list[tuple[Path, str, str, Path, list[str], Path]] = []
    job_index = 0
    for run_dir in runs:
        for flux_kind in ("aperture", "psf"):
            out_dir = run_dir / f"blind_classifier_{flux_kind}_warp"
            out_path = out_dir / "blind_candidate_clusters.parquet"
            if not args.force and _done(out_path):
                print(f"skip {flux_kind}: {out_path}", flush=True)
                continue
            device = devices[job_index % len(devices)]
            job_index += 1
            scan_jobs.append(
                (
                    run_dir,
                    flux_kind,
                    device,
                    out_path,
                    _scan_cmd(
                        py=py,
                        run_dir=run_dir,
                        output_dir=out_dir,
                        flux_kind=flux_kind,
                        grid_step_nm=args.grid_step_nm,
                        min_snr=args.min_snr,
                        min_supporting_points=args.min_supporting_points,
                        device=device,
                        candidate_mode=args.candidate_mode,
                        top_k_per_target=args.top_k_per_target,
                        top_k_min_separation_nm=args.top_k_min_separation_nm,
                    ),
                    log_root / f"{run_dir.name}_blind_baseline_{flux_kind}.log",
                )
            )

    print(f"scan jobs queued: {len(scan_jobs)}", flush=True)
    completed_scans = 0
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[Future[None], tuple[Path, str, str]] = {
            executor.submit(_run, cmd, log_path): (run_dir, flux_kind, device)
            for run_dir, flux_kind, device, _out_path, cmd, log_path in scan_jobs
        }
        for future in as_completed(futures):
            run_dir, flux_kind, device = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append(f"{run_dir.name}:{flux_kind}:{device}:{type(exc).__name__}:{exc}")
                print(f"FAILED {run_dir.name} {flux_kind} on {device}: {exc}", flush=True)
            else:
                completed_scans += 1
                print(f"completed scans {completed_scans}/{len(scan_jobs)}: {run_dir.name} {flux_kind} on {device}", flush=True)

    if failures:
        print("scan failures:", flush=True)
        for failure in failures:
            print(f"  {failure}", flush=True)

    completed_joint = 0
    for run_dir in runs:
        aperture_done = _done(run_dir / "blind_classifier_aperture_warp" / "blind_candidate_clusters.parquet")
        psf_done = _done(run_dir / "blind_classifier_psf_warp" / "blind_candidate_clusters.parquet")
        if not (aperture_done and psf_done):
            print(f"skip joint missing scan outputs: {run_dir.name}", flush=True)
            continue
        joint_dir = run_dir / "blind_classifier_joint_warp"
        joint_path = joint_dir / "blind_joint_candidates.parquet"
        if not args.force and _done(joint_path):
            print(f"skip joint: {joint_path}", flush=True)
            completed_joint += 1
            continue
        _run(
            _rank_cmd(py, run_dir, joint_dir, args.match_tolerance_nm),
            log_root / f"{run_dir.name}_blind_baseline_joint.log",
        )
        completed_joint += 1
        print(f"completed joint {completed_joint}/{len(runs)}: {run_dir.name}", flush=True)

    print(f"raw blind backfill complete: scans={completed_scans}/{len(scan_jobs)} joint={completed_joint}/{len(runs)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"command failed with exit {exc.returncode}: {' '.join(exc.cmd)}", file=sys.stderr)
        raise
