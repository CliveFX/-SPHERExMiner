from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any

from .campaign import CampaignContractConfig, write_campaign_contract
from .catalog import CatalogConfig, build_frame_targets
from .dispatch import DispatchPlanConfig, build_dispatch_plan, collect_dispatch_run, write_dispatch_plan
from .gpu_worker import PersistentWorkerConfig, run_persistent_gpu_worker
from .kubernetes import KubernetesJobConfig, write_kubernetes_jobs
from .manifest import build_frame_manifest
from .photometry import ApertureConfig, run_cpu_aperture, run_gpu_aperture
from .projection import project_frame_targets
from .spectra import SpectraAssemblyConfig, assemble_spectra_from_shards


OPTIONAL_MODULES = [
    "numpy",
    "pandas",
    "pyarrow",
    "astropy",
    "cupy",
    "numba",
    "warp",
    "cudf",
    "dask_cudf",
    "rmm",
    "kvikio",
    "cuspatial",
    "cuml",
    "torch",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="luxquarry-allsky")
    sub = parser.add_subparsers(dest="command", required=True)

    env_probe = sub.add_parser("env-probe", help="Report CUDA/RAPIDS/Python environment capabilities.")
    env_probe.add_argument("--out", type=Path, help="Optional JSON output path.")
    env_probe.set_defaults(func=cmd_env_probe)

    bench = sub.add_parser("benchmark-smoke", help="Create a benchmark skeleton and perf_summary.json.")
    bench.add_argument("--campaign-id", default="local_smoke")
    bench.add_argument("--out-dir", type=Path, default=Path("runs/local_smoke"))
    bench.add_argument("--frame-count", type=int, default=0)
    bench.add_argument("--target-count", type=int, default=0)
    bench.add_argument("--measurement-count", type=int, default=0)
    bench.set_defaults(func=cmd_benchmark_smoke)

    manifest = sub.add_parser("build-manifest", help="Scan FITS frames and write a frame manifest parquet.")
    manifest.add_argument("--input-root", type=Path, action="append", required=True)
    manifest.add_argument("--out", type=Path, required=True)
    manifest.add_argument("--campaign-id", default="local_manifest")
    manifest.add_argument("--limit", type=int)
    manifest.add_argument("--no-read-headers", action="store_true")
    manifest.set_defaults(func=cmd_build_manifest)

    frame_targets = sub.add_parser("build-frame-targets", help="Query catalogs for targets inside frame footprints.")
    frame_targets.add_argument("--manifest", type=Path, required=True)
    frame_targets.add_argument("--out", type=Path, required=True)
    frame_targets.add_argument("--cache-root", type=Path, default=Path("/mnt/niroseti/spherex_cache"))
    frame_targets.add_argument("--catalog", choices=["gaia", "2mass", "all"], default="all")
    frame_targets.add_argument("--gaia-g-min", type=float, default=11.0)
    frame_targets.add_argument("--gaia-g-max", type=float, default=16.0)
    frame_targets.add_argument("--twomass-mag-min", type=float, default=11.0)
    frame_targets.add_argument("--twomass-mag-max", type=float, default=16.0)
    frame_targets.add_argument("--max-sources-per-frame", type=int, default=5000)
    frame_targets.add_argument("--bbox-pad-deg", type=float, default=0.05)
    frame_targets.add_argument("--limit-frames", type=int)
    frame_targets.set_defaults(func=cmd_build_frame_targets)

    project_targets = sub.add_parser("project-frame-targets", help="Project frame target RA/Dec to detector pixels.")
    project_targets.add_argument("--manifest", type=Path, required=True)
    project_targets.add_argument("--frame-targets", type=Path, required=True)
    project_targets.add_argument("--out", type=Path, required=True)
    project_targets.add_argument("--limit-frames", type=int)
    project_targets.set_defaults(func=cmd_project_frame_targets)

    aperture = sub.add_parser("run-cpu-aperture", help="Run calibrated CPU aperture photometry for projected targets.")
    aperture.add_argument("--manifest", type=Path, required=True)
    aperture.add_argument("--projected-targets", type=Path, required=True)
    aperture.add_argument("--out", type=Path, required=True)
    aperture.add_argument("--cache-root", type=Path, default=Path("/mnt/niroseti/spherex_cache"))
    aperture.add_argument("--aperture-radius-pix", type=float, default=2.0)
    aperture.add_argument("--annulus-inner-pix", type=float, default=4.0)
    aperture.add_argument("--annulus-outer-pix", type=float, default=6.0)
    aperture.add_argument("--edge-margin-pix", type=float, default=6.0)
    aperture.add_argument("--limit-frames", type=int)
    aperture.set_defaults(func=cmd_run_cpu_aperture)

    gpu_aperture = sub.add_parser(
        "run-gpu-aperture",
        help="Run calibrated frame-level GPU aperture photometry and write cuDF parquet shards.",
    )
    gpu_aperture.add_argument("--manifest", type=Path, required=True)
    gpu_aperture.add_argument("--projected-targets", type=Path, required=True)
    gpu_aperture.add_argument("--out", type=Path, required=True)
    gpu_aperture.add_argument("--cache-root", type=Path, default=Path("/mnt/niroseti/spherex_cache"))
    gpu_aperture.add_argument("--device", default="cuda:0")
    gpu_aperture.add_argument("--aperture-radius-pix", type=float, default=2.0)
    gpu_aperture.add_argument("--annulus-inner-pix", type=float, default=4.0)
    gpu_aperture.add_argument("--annulus-outer-pix", type=float, default=6.0)
    gpu_aperture.add_argument("--edge-margin-pix", type=float, default=6.0)
    gpu_aperture.add_argument("--limit-frames", type=int)
    gpu_aperture.set_defaults(func=cmd_run_gpu_aperture)

    persistent = sub.add_parser(
        "run-persistent-gpu-worker",
        help="Run a persistent GPU frame worker with resident calibration maps and shard outputs.",
    )
    persistent.add_argument("--manifest", type=Path, required=True)
    persistent.add_argument("--projected-targets", type=Path, required=True)
    persistent.add_argument("--out-dir", type=Path, required=True)
    persistent.add_argument("--run-id", required=True)
    persistent.add_argument("--cache-root", type=Path, default=Path("/mnt/niroseti/spherex_cache"))
    persistent.add_argument("--device", default="cuda:0")
    persistent.add_argument("--worker-index", type=int, default=0)
    persistent.add_argument("--worker-count", type=int, default=1)
    persistent.add_argument("--status-path", type=Path)
    persistent.add_argument("--write-combined-output", action="store_true")
    persistent.add_argument("--no-rmm-pool", action="store_true")
    persistent.add_argument(
        "--async-shard-writes",
        action="store_true",
        help="Queue parquet shard writes on a background thread and wait for them before completion.",
    )
    persistent.add_argument(
        "--batch-table-assembly",
        action="store_true",
        help="Defer cuDF measurement table assembly until shard flush instead of once per frame.",
    )
    persistent.add_argument("--shard-batch-frames", type=int, default=1)
    persistent.add_argument("--prefetch-frames", type=int, default=0)
    persistent.add_argument("--status-interval-frames", type=int, default=1)
    persistent.add_argument(
        "--local-cache-dir",
        type=Path,
        help="Optional local SSD/NVMe directory used to stage FITS files before reading them.",
    )
    persistent.add_argument("--aperture-radius-pix", type=float, default=2.0)
    persistent.add_argument("--annulus-inner-pix", type=float, default=4.0)
    persistent.add_argument("--annulus-outer-pix", type=float, default=6.0)
    persistent.add_argument("--edge-margin-pix", type=float, default=6.0)
    persistent.add_argument("--limit-frames", type=int)
    persistent.set_defaults(func=cmd_run_persistent_gpu_worker)

    dispatch = sub.add_parser(
        "plan-gpu-dispatch",
        help="Write JSON and shell plans for horizontally partitioned persistent GPU workers.",
    )
    dispatch.add_argument("--manifest", type=Path, required=True)
    dispatch.add_argument("--projected-targets", type=Path, required=True)
    dispatch.add_argument("--out-dir", type=Path, required=True)
    dispatch.add_argument("--run-id", required=True)
    dispatch.add_argument("--plan-out", type=Path, required=True)
    dispatch.add_argument("--cache-root", type=Path, default=Path("/mnt/niroseti/spherex_cache"))
    dispatch.add_argument("--devices", default="cuda:0", help="Comma-separated GPU devices, e.g. cuda:0,cuda:1,cuda:2")
    dispatch.add_argument("--workers-per-device", type=int, default=1)
    dispatch.add_argument(
        "--async-shard-writes",
        action="store_true",
        help="Pass --async-shard-writes to every persistent worker.",
    )
    dispatch.add_argument(
        "--batch-table-assembly",
        action="store_true",
        help="Pass --batch-table-assembly to every persistent worker.",
    )
    dispatch.add_argument(
        "--materialize-worker-inputs",
        action="store_true",
        help="Write per-worker manifest/target parquets and point workers at their own input slices.",
    )
    dispatch.add_argument("--shard-batch-frames", type=int, default=1)
    dispatch.add_argument("--prefetch-frames", type=int, default=0)
    dispatch.add_argument("--status-interval-frames", type=int, default=1)
    dispatch.add_argument(
        "--local-cache-dir",
        type=Path,
        help="Optional local SSD/NVMe directory passed to every persistent worker for FITS staging.",
    )
    dispatch.add_argument("--limit-frames", type=int)
    dispatch.add_argument("--executable", default=".venv/bin/luxquarry-allsky")
    dispatch.set_defaults(func=cmd_plan_gpu_dispatch)

    collect_dispatch = sub.add_parser(
        "collect-dispatch-run",
        help="Aggregate persistent worker run summaries and write a shard manifest.",
    )
    collect_dispatch.add_argument("--plan", type=Path, required=True)
    collect_dispatch.add_argument("--out", type=Path)
    collect_dispatch.set_defaults(func=cmd_collect_dispatch_run)

    k8s_jobs = sub.add_parser(
        "write-k8s-jobs",
        help="Write one Kubernetes Job manifest per GPU worker from a dispatch plan.",
    )
    k8s_jobs.add_argument("--plan", type=Path, required=True)
    k8s_jobs.add_argument("--out-dir", type=Path, required=True)
    k8s_jobs.add_argument("--image", required=True)
    k8s_jobs.add_argument("--namespace", default="default")
    k8s_jobs.add_argument("--service-account")
    k8s_jobs.add_argument("--container-executable", default="luxquarry-allsky")
    k8s_jobs.add_argument("--working-dir")
    k8s_jobs.add_argument("--gpu-limit", type=int, default=1)
    k8s_jobs.add_argument("--cpu-request", default="4")
    k8s_jobs.add_argument("--memory-request", default="16Gi")
    k8s_jobs.add_argument("--restart-policy", default="Never")
    k8s_jobs.add_argument("--backoff-limit", type=int, default=1)
    k8s_jobs.add_argument("--pvc-name")
    k8s_jobs.add_argument("--mount-path")
    k8s_jobs.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable to place on every worker container. Repeatable.",
    )
    k8s_jobs.set_defaults(func=cmd_write_k8s_jobs)

    campaign_contract = sub.add_parser(
        "write-campaign-contract",
        help="Write a stage contract for baseline, injected, scoring, recovery, and viewer products.",
    )
    campaign_contract.add_argument("--campaign-id", required=True)
    campaign_contract.add_argument("--out", type=Path, required=True)
    campaign_contract.add_argument("--baseline-plan", type=Path, required=True)
    campaign_contract.add_argument("--baseline-spectra-dir", type=Path)
    campaign_contract.add_argument("--injected-plan", type=Path)
    campaign_contract.add_argument("--injected-spectra-dir", type=Path)
    campaign_contract.add_argument("--injection-truth", type=Path)
    campaign_contract.add_argument("--candidate-dir", type=Path)
    campaign_contract.add_argument("--viewer-index-dir", type=Path)
    campaign_contract.set_defaults(func=cmd_write_campaign_contract)

    spectra = sub.add_parser(
        "assemble-spectra",
        help="Build target-ordered spectra products from a measurement shard manifest using cuDF.",
    )
    spectra.add_argument("--shard-manifest", type=Path, required=True)
    spectra.add_argument("--out-dir", type=Path, required=True)
    spectra.add_argument("--run-id", required=True)
    spectra.add_argument("--device", default="cuda:0")
    spectra.add_argument("--only-ok", action="store_true", help="Only keep aperture_status_code == 0 measurements.")
    spectra.set_defaults(func=cmd_assemble_spectra)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


def cmd_env_probe(args: argparse.Namespace) -> int:
    report = build_env_report()
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


def cmd_benchmark_smoke(args: argparse.Namespace) -> int:
    start = time.perf_counter()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for child in ["manifest", "local_cache", "measurement_shards", "spectra", "candidates"]:
        (out_dir / child).mkdir(exist_ok=True)

    elapsed = time.perf_counter() - start
    perf = {
        "campaign_id": args.campaign_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frame_count": int(args.frame_count),
        "target_count": int(args.target_count),
        "measurement_count": int(args.measurement_count),
        "total_wall_sec": elapsed,
        "stage_wall_sec": {
            "stage_fits": 0.0,
            "catalog_query": 0.0,
            "wcs_projection": 0.0,
            "gpu_photometry": 0.0,
            "write_measurements": 0.0,
            "assemble_spectra": 0.0,
            "score_candidates": 0.0,
        },
        "throughput": {
            "frames_per_sec": _rate(args.frame_count, elapsed),
            "measurements_per_sec": _rate(args.measurement_count, elapsed),
            "measurements_per_gpu_sec": 0.0,
            "parquet_rows_per_sec": 0.0,
        },
        "io": {
            "fits_read_bytes": 0,
            "catalog_read_bytes": 0,
            "parquet_write_bytes": 0,
            "local_cache_peak_bytes": 0,
        },
        "gpu": {
            "device_count": len(_nvidia_smi_gpus()),
            "kernel_wall_sec": 0.0,
            "estimated_occupancy": None,
        },
    }
    correctness = {
        "campaign_id": args.campaign_id,
        "created_utc": perf["created_utc"],
        "reference_system": "current_target_centered_miner",
        "checks": [],
        "status": "not_run",
    }
    profile = {
        "campaign_id": args.campaign_id,
        "created_utc": perf["created_utc"],
        "rows": [],
        "note": "No instrumented stages yet. Future rows feed the 5% acceleration audit.",
    }
    _write_json(out_dir / "perf_summary.json", perf)
    _write_json(out_dir / "correctness_summary.json", correctness)
    _write_json(out_dir / "profile_summary.json", profile)
    print(json.dumps({"out_dir": str(out_dir), "perf_summary": str(out_dir / "perf_summary.json")}, indent=2))
    return 0


def cmd_build_manifest(args: argparse.Namespace) -> int:
    summary = build_frame_manifest(
        input_roots=list(args.input_root),
        output_path=args.out,
        limit=args.limit,
        campaign_id=args.campaign_id,
        read_headers=not args.no_read_headers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_build_frame_targets(args: argparse.Namespace) -> int:
    config = CatalogConfig(
        cache_root=args.cache_root,
        catalog=args.catalog,
        gaia_g_min=args.gaia_g_min,
        gaia_g_max=args.gaia_g_max,
        twomass_mag_min=args.twomass_mag_min,
        twomass_mag_max=args.twomass_mag_max,
        max_sources_per_frame=args.max_sources_per_frame,
        bbox_pad_deg=args.bbox_pad_deg,
    )
    summary = build_frame_targets(
        manifest_path=args.manifest,
        output_path=args.out,
        config=config,
        limit_frames=args.limit_frames,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_project_frame_targets(args: argparse.Namespace) -> int:
    summary = project_frame_targets(
        manifest_path=args.manifest,
        frame_targets_path=args.frame_targets,
        output_path=args.out,
        limit_frames=args.limit_frames,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_run_cpu_aperture(args: argparse.Namespace) -> int:
    summary = run_cpu_aperture(
        manifest_path=args.manifest,
        projected_targets_path=args.projected_targets,
        output_path=args.out,
        config=ApertureConfig(
            cache_root=args.cache_root,
            aperture_radius_pix=args.aperture_radius_pix,
            annulus_inner_pix=args.annulus_inner_pix,
            annulus_outer_pix=args.annulus_outer_pix,
            edge_margin_pix=args.edge_margin_pix,
        ),
        limit_frames=args.limit_frames,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_run_gpu_aperture(args: argparse.Namespace) -> int:
    summary = run_gpu_aperture(
        manifest_path=args.manifest,
        projected_targets_path=args.projected_targets,
        output_path=args.out,
        config=ApertureConfig(
            cache_root=args.cache_root,
            aperture_radius_pix=args.aperture_radius_pix,
            annulus_inner_pix=args.annulus_inner_pix,
            annulus_outer_pix=args.annulus_outer_pix,
            edge_margin_pix=args.edge_margin_pix,
        ),
        limit_frames=args.limit_frames,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_run_persistent_gpu_worker(args: argparse.Namespace) -> int:
    if args.worker_count <= 0:
        raise ValueError("--worker-count must be positive")
    if args.worker_index < 0 or args.worker_index >= args.worker_count:
        raise ValueError("--worker-index must be in [0, worker-count)")
    if args.shard_batch_frames <= 0:
        raise ValueError("--shard-batch-frames must be positive")
    if args.prefetch_frames < 0:
        raise ValueError("--prefetch-frames must be non-negative")
    if args.status_interval_frames <= 0:
        raise ValueError("--status-interval-frames must be positive")
    summary = run_persistent_gpu_worker(
        manifest_path=args.manifest,
        projected_targets_path=args.projected_targets,
        output_dir=args.out_dir,
        run_id=args.run_id,
        config=PersistentWorkerConfig(
            aperture=ApertureConfig(
                cache_root=args.cache_root,
                aperture_radius_pix=args.aperture_radius_pix,
                annulus_inner_pix=args.annulus_inner_pix,
                annulus_outer_pix=args.annulus_outer_pix,
                edge_margin_pix=args.edge_margin_pix,
            ),
            device=args.device,
            worker_index=args.worker_index,
            worker_count=args.worker_count,
            write_combined_output=args.write_combined_output,
            rmm_pool=not args.no_rmm_pool,
            async_shard_writes=args.async_shard_writes,
            batch_table_assembly=args.batch_table_assembly,
            shard_batch_frames=args.shard_batch_frames,
            prefetch_frames=args.prefetch_frames,
            status_interval_frames=args.status_interval_frames,
            local_cache_dir=args.local_cache_dir,
        ),
        limit_frames=args.limit_frames,
        status_path=args.status_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_plan_gpu_dispatch(args: argparse.Namespace) -> int:
    devices = tuple(part.strip() for part in args.devices.split(",") if part.strip())
    plan = build_dispatch_plan(
        DispatchPlanConfig(
            manifest_path=args.manifest,
            projected_targets_path=args.projected_targets,
            output_dir=args.out_dir,
            run_id=args.run_id,
            devices=devices,
            workers_per_device=args.workers_per_device,
            cache_root=args.cache_root,
            limit_frames=args.limit_frames,
            executable=args.executable,
            async_shard_writes=args.async_shard_writes,
            batch_table_assembly=args.batch_table_assembly,
            materialize_worker_inputs=args.materialize_worker_inputs,
            shard_batch_frames=args.shard_batch_frames,
            prefetch_frames=args.prefetch_frames,
            status_interval_frames=args.status_interval_frames,
            local_cache_dir=args.local_cache_dir,
        )
    )
    write_dispatch_plan(plan, args.plan_out)
    print(
        json.dumps(
            {
                "plan_out": str(args.plan_out),
                "shell_out": str(args.plan_out.with_suffix(".sh")),
                "worker_count": plan["worker_count"],
                "devices": plan["devices"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_collect_dispatch_run(args: argparse.Namespace) -> int:
    summary = collect_dispatch_run(plan_path=args.plan, output_path=args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_write_k8s_jobs(args: argparse.Namespace) -> int:
    summary = write_kubernetes_jobs(
        KubernetesJobConfig(
            plan_path=args.plan,
            output_dir=args.out_dir,
            image=args.image,
            namespace=args.namespace,
            service_account=args.service_account,
            container_executable=args.container_executable,
            working_dir=args.working_dir,
            gpu_limit=args.gpu_limit,
            cpu_request=args.cpu_request,
            memory_request=args.memory_request,
            restart_policy=args.restart_policy,
            backoff_limit=args.backoff_limit,
            pvc_name=args.pvc_name,
            mount_path=args.mount_path,
            env=_parse_env_assignments(args.env),
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_write_campaign_contract(args: argparse.Namespace) -> int:
    contract = write_campaign_contract(
        CampaignContractConfig(
            campaign_id=args.campaign_id,
            output_path=args.out,
            baseline_plan_path=args.baseline_plan,
            baseline_spectra_dir=args.baseline_spectra_dir,
            injected_plan_path=args.injected_plan,
            injected_spectra_dir=args.injected_spectra_dir,
            injection_truth_path=args.injection_truth,
            candidate_dir=args.candidate_dir,
            viewer_index_dir=args.viewer_index_dir,
        )
    )
    print(json.dumps(contract, indent=2, sort_keys=True))
    return 0


def cmd_assemble_spectra(args: argparse.Namespace) -> int:
    summary = assemble_spectra_from_shards(
        SpectraAssemblyConfig(
            shard_manifest_path=args.shard_manifest,
            output_dir=args.out_dir,
            run_id=args.run_id,
            device=args.device,
            only_ok=args.only_ok,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_env_report() -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "prefix": sys.prefix,
        },
        "nvidia_smi": _nvidia_smi_report(),
        "modules": {name: _module_report(name) for name in OPTIONAL_MODULES},
    }


def _module_report(name: str) -> dict[str, Any]:
    try:
        module = import_module(name)
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    version = getattr(module, "__version__", None)
    report: dict[str, Any] = {"available": True, "version": str(version) if version is not None else None}
    if name == "cupy":
        report.update(_cupy_details(module))
    if name == "torch":
        report.update(_torch_details(module))
    return report


def _cupy_details(module: Any) -> dict[str, Any]:
    try:
        runtime = module.cuda.runtime
        return {
            "cuda_runtime_version": runtime.runtimeGetVersion(),
            "device_count": runtime.getDeviceCount(),
        }
    except Exception as exc:
        return {"cuda_error": f"{type(exc).__name__}: {exc}"}


def _torch_details(module: Any) -> dict[str, Any]:
    try:
        return {
            "cuda_available": bool(module.cuda.is_available()),
            "cuda_device_count": int(module.cuda.device_count()),
            "cuda_version": getattr(module.version, "cuda", None),
        }
    except Exception as exc:
        return {"cuda_error": f"{type(exc).__name__}: {exc}"}


def _nvidia_smi_report() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "gpus": _parse_nvidia_smi(proc.stdout)}


def _nvidia_smi_gpus() -> list[dict[str, Any]]:
    report = _nvidia_smi_report()
    return list(report.get("gpus") or []) if report.get("available") else []


def _parse_nvidia_smi(stdout: str) -> list[dict[str, Any]]:
    gpus = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        index, name, memory_mb, driver = parts[:4]
        try:
            memory_total_mb = int(memory_mb)
        except ValueError:
            memory_total_mb = None
        gpus.append(
            {
                "index": int(index) if index.isdigit() else index,
                "name": name,
                "memory_total_mb": memory_total_mb,
                "driver_version": driver,
            }
        )
    return gpus


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_env_assignments(assignments: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in assignments:
        if "=" not in item:
            raise ValueError(f"--env must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--env key must not be empty, got {item!r}")
        env[key] = value
    return env


def _rate(count: int, elapsed_sec: float) -> float:
    return float(count / elapsed_sec) if elapsed_sec > 0 and count else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
