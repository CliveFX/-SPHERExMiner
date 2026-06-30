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

from .benchmark import DispatchBenchmarkSweepConfig, run_dispatch_benchmark_sweep
from .campaign import CampaignContractConfig, write_campaign_contract
from .candidate_fanout import (
    CandidateFanoutCollectConfig,
    CandidateFanoutPlanConfig,
    LocalCandidateFanoutRunConfig,
    build_candidate_fanout_plan,
    collect_candidate_fanout_plan,
    run_candidate_fanout_plan,
    write_candidate_fanout_plan,
)
from .catalog import CatalogConfig, build_frame_targets
from .dispatch import DispatchPlanConfig, build_dispatch_plan, collect_dispatch_run, write_dispatch_plan
from .finalize import FinalizeDispatchConfig, finalize_dispatch_run
from .gpu_worker import PersistentWorkerConfig, run_persistent_gpu_worker
from .kubernetes import (
    KubernetesJobConfig,
    KubernetesPostprocessJobConfig,
    KubernetesReducerJobConfig,
    write_kubernetes_jobs,
    write_kubernetes_postprocess_job,
    write_kubernetes_reducer_jobs,
)
from .local_runner import LocalDispatchRunConfig, LocalPlanWorkerRunConfig, run_dispatch_plan_workers, run_local_dispatch
from .manifest import build_frame_manifest, rewrite_manifest_paths_to_uri
from .photometry import ApertureConfig, run_cpu_aperture, run_gpu_aperture
from .projection import project_frame_targets
from .recovery import InjectionRecoveryConfig, score_injection_recovery
from .reducer import (
    LocalReducerRunConfig,
    ReducerCollectConfig,
    ReducerPlanConfig,
    build_reducer_plan,
    collect_reducer_plan,
    run_reducer_plan,
    write_reducer_plan,
)
from .scoring import CandidateScoringConfig, score_spectra_candidates
from .spectra import (
    MeasurementPartitionConfig,
    PartitionedSpectraAssemblyConfig,
    SpectraAssemblyConfig,
    SpectraAssemblyValidationConfig,
    SpectraRetryDedupValidationConfig,
    assemble_spectra_partitions,
    assemble_spectra_from_shards,
    partition_measurement_shards,
    validate_retry_dedup_assembly,
    validate_shard_order_independent_assembly,
)
from .status import DispatchStatusConfig, write_dispatch_status_snapshot
from .task_queue import (
    GpuWorkerServiceConfig,
    TaskQueueCollectConfig,
    TaskQueueWriteConfig,
    collect_task_queue_run,
    run_gpu_worker_service,
    write_task_queue,
)


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

    dispatch_sweep = sub.add_parser(
        "run-dispatch-benchmark-sweep",
        help="Run repeated local dispatch/finalize trials and summarize throughput by setting.",
    )
    dispatch_sweep.add_argument("--manifest", type=Path, required=True)
    dispatch_sweep.add_argument("--projected-targets", type=Path, required=True)
    dispatch_sweep.add_argument("--out-dir", type=Path, required=True)
    dispatch_sweep.add_argument("--run-id", required=True)
    dispatch_sweep.add_argument("--cache-root", type=Path, default=Path("/mnt/niroseti/spherex_cache"))
    dispatch_sweep.add_argument("--devices", default="cuda:0")
    dispatch_sweep.add_argument("--workers-per-device", default="1")
    dispatch_sweep.add_argument("--limit-frames", default="2")
    dispatch_sweep.add_argument("--shard-batch-frames", default="1")
    dispatch_sweep.add_argument("--prefetch-frames", default="0")
    dispatch_sweep.add_argument("--repetitions", type=int, default=1)
    dispatch_sweep.add_argument("--local-cache-dir", type=Path)
    dispatch_sweep.add_argument("--executable")
    dispatch_sweep.add_argument("--finalize-device", default="cuda:0")
    dispatch_sweep.add_argument("--no-async-shard-writes", action="store_true")
    dispatch_sweep.add_argument("--no-batch-table-assembly", action="store_true")
    dispatch_sweep.add_argument("--no-materialize-worker-inputs", action="store_true")
    dispatch_sweep.add_argument("--score-baseline", action="store_true")
    dispatch_sweep.add_argument("--candidate-min-abs-zscore", type=float, default=5.0)
    dispatch_sweep.add_argument("--candidate-min-measurements", type=int, default=10)
    dispatch_sweep.add_argument("--candidate-max-rows", type=int)
    dispatch_sweep.add_argument("--status-snapshot-interval-sec", type=float, default=0.0)
    dispatch_sweep.add_argument("--continue-on-error", action="store_true")
    dispatch_sweep.add_argument("--worker-only", action="store_true")
    dispatch_sweep.set_defaults(func=cmd_run_dispatch_benchmark_sweep)

    manifest = sub.add_parser("build-manifest", help="Scan FITS frames and write a frame manifest parquet.")
    manifest.add_argument("--input-root", type=Path, action="append", required=True)
    manifest.add_argument("--out", type=Path, required=True)
    manifest.add_argument("--campaign-id", default="local_manifest")
    manifest.add_argument("--limit", type=int)
    manifest.add_argument("--no-read-headers", action="store_true")
    manifest.set_defaults(func=cmd_build_manifest)

    rewrite_manifest = sub.add_parser(
        "rewrite-manifest-paths",
        help="Rewrite frame manifest paths from a local prefix to an object-store URI prefix.",
    )
    rewrite_manifest.add_argument("--manifest", type=Path, required=True)
    rewrite_manifest.add_argument("--out", type=Path, required=True)
    rewrite_manifest.add_argument("--strip-prefix", type=Path, required=True)
    rewrite_manifest.add_argument("--uri-prefix", required=True)
    rewrite_manifest.set_defaults(func=cmd_rewrite_manifest_paths)

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

    task_queue = sub.add_parser(
        "write-task-queue",
        help="Materialize a local frame-batch task queue for long-lived GPU worker services.",
    )
    task_queue.add_argument("--manifest", type=Path, required=True)
    task_queue.add_argument("--projected-targets", type=Path, required=True)
    task_queue.add_argument("--out-dir", type=Path, required=True)
    task_queue.add_argument("--campaign-id", required=True)
    task_queue.add_argument("--frames-per-task", type=int, default=25)
    task_queue.add_argument("--limit-frames", type=int)
    task_queue.add_argument(
        "--no-materialize-task-inputs",
        action="store_true",
        help="Write lightweight frame-id tasks and let the worker service keep source tables resident.",
    )
    task_queue.set_defaults(func=cmd_write_task_queue)

    worker_service = sub.add_parser(
        "run-gpu-worker-service",
        help="Run a long-lived GPU worker that claims local frame-batch tasks until the queue drains.",
    )
    worker_service.add_argument("--queue-dir", type=Path, required=True)
    worker_service.add_argument("--out-dir", type=Path, required=True)
    worker_service.add_argument("--run-id", required=True)
    worker_service.add_argument("--worker-id", default="worker-000")
    worker_service.add_argument("--cache-root", type=Path, default=Path("/mnt/niroseti/spherex_cache"))
    worker_service.add_argument("--device", default="cuda:0")
    worker_service.add_argument("--max-tasks", type=int)
    worker_service.add_argument("--no-rmm-pool", action="store_true")
    worker_service.add_argument("--async-shard-writes", action="store_true")
    worker_service.add_argument("--batch-table-assembly", action="store_true")
    worker_service.add_argument("--shard-batch-frames", type=int, default=1)
    worker_service.add_argument("--prefetch-frames", type=int, default=0)
    worker_service.add_argument("--status-interval-frames", type=int, default=1)
    worker_service.add_argument("--local-cache-dir", type=Path)
    worker_service.add_argument("--aperture-radius-pix", type=float, default=2.0)
    worker_service.add_argument("--annulus-inner-pix", type=float, default=4.0)
    worker_service.add_argument("--annulus-outer-pix", type=float, default=6.0)
    worker_service.add_argument("--edge-margin-pix", type=float, default=6.0)
    worker_service.set_defaults(func=cmd_run_gpu_worker_service)

    collect_task_queue = sub.add_parser(
        "collect-task-queue-run",
        help="Aggregate completed local task-queue summaries and write a measurement shard manifest.",
    )
    collect_task_queue.add_argument("--queue-dir", type=Path, required=True)
    collect_task_queue.add_argument("--out", type=Path, required=True)
    collect_task_queue.set_defaults(func=cmd_collect_task_queue_run)

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

    local_dispatch = sub.add_parser(
        "run-local-dispatch",
        help="Plan, launch local persistent GPU workers, and finalize the dispatch.",
    )
    local_dispatch.add_argument("--manifest", type=Path, required=True)
    local_dispatch.add_argument("--projected-targets", type=Path, required=True)
    local_dispatch.add_argument("--out-dir", type=Path, required=True)
    local_dispatch.add_argument("--run-id", required=True)
    local_dispatch.add_argument("--cache-root", type=Path, default=Path("/mnt/niroseti/spherex_cache"))
    local_dispatch.add_argument("--devices", default="cuda:0")
    local_dispatch.add_argument("--workers-per-device", type=int, default=1)
    local_dispatch.add_argument("--async-shard-writes", action="store_true")
    local_dispatch.add_argument("--batch-table-assembly", action="store_true")
    local_dispatch.add_argument("--no-materialize-worker-inputs", action="store_true")
    local_dispatch.add_argument("--shard-batch-frames", type=int, default=1)
    local_dispatch.add_argument("--prefetch-frames", type=int, default=0)
    local_dispatch.add_argument("--status-interval-frames", type=int, default=1)
    local_dispatch.add_argument("--local-cache-dir", type=Path)
    local_dispatch.add_argument("--limit-frames", type=int)
    local_dispatch.add_argument("--executable")
    local_dispatch.add_argument("--finalize-device", default="cuda:0")
    local_dispatch.add_argument("--spectra-out-dir", type=Path)
    local_dispatch.add_argument("--spectra-run-id")
    local_dispatch.add_argument("--only-ok", action="store_true")
    local_dispatch.add_argument("--allow-incomplete-finalize", action="store_true")
    local_dispatch.add_argument("--campaign-id")
    local_dispatch.add_argument("--campaign-contract-out", type=Path)
    local_dispatch.add_argument("--candidate-dir", type=Path)
    local_dispatch.add_argument("--score-baseline", action="store_true")
    local_dispatch.add_argument("--candidate-min-abs-zscore", type=float, default=5.0)
    local_dispatch.add_argument("--candidate-min-measurements", type=int, default=10)
    local_dispatch.add_argument("--candidate-max-rows", type=int)
    local_dispatch.add_argument("--score-injected", action="store_true")
    local_dispatch.add_argument("--injected-spectra-dir", type=Path)
    local_dispatch.add_argument("--injection-truth", type=Path)
    local_dispatch.add_argument("--recover-injections", action="store_true")
    local_dispatch.add_argument("--recovery-min-score", type=float, default=5.0)
    local_dispatch.add_argument("--recovery-wavelength-tolerance-nm", type=float, default=10.0)
    local_dispatch.add_argument("--recovery-require-line-family", action="store_true")
    local_dispatch.add_argument("--resume", action="store_true", help="Skip workers with complete run_summary.json.")
    local_dispatch.add_argument(
        "--status-snapshot-interval-sec",
        type=float,
        default=1.0,
        help="Refresh dispatch_status.json while workers run. Use 0 for final snapshot only.",
    )
    local_dispatch.set_defaults(func=cmd_run_local_dispatch)

    collect_dispatch = sub.add_parser(
        "collect-dispatch-run",
        help="Aggregate persistent worker run summaries and write a shard manifest.",
    )
    collect_dispatch.add_argument("--plan", type=Path, required=True)
    collect_dispatch.add_argument("--out", type=Path)
    collect_dispatch.set_defaults(func=cmd_collect_dispatch_run)

    run_plan_workers = sub.add_parser(
        "run-dispatch-plan-workers",
        help="Launch workers from an existing dispatch plan without finalizing spectra.",
    )
    run_plan_workers.add_argument("--plan", type=Path, required=True)
    run_plan_workers.add_argument("--logs-dir", type=Path)
    run_plan_workers.add_argument("--resume", action="store_true")
    run_plan_workers.add_argument("--status-snapshot-interval-sec", type=float, default=1.0)
    run_plan_workers.add_argument("--allow-failed-workers", action="store_true")
    run_plan_workers.set_defaults(func=cmd_run_dispatch_plan_workers)

    dispatch_status = sub.add_parser(
        "dispatch-status",
        help="Aggregate per-worker run_status.json files into one atomic status snapshot.",
    )
    dispatch_status.add_argument("--plan", type=Path, required=True)
    dispatch_status.add_argument("--out", type=Path)
    dispatch_status.set_defaults(func=cmd_dispatch_status)

    finalize_dispatch = sub.add_parser(
        "finalize-dispatch-run",
        help="Collect worker shards, assemble spectra, and write the campaign contract.",
    )
    finalize_dispatch.add_argument("--plan", type=Path, required=True)
    finalize_dispatch.add_argument("--device", default="cuda:0")
    finalize_dispatch.add_argument("--aggregate-out", type=Path)
    finalize_dispatch.add_argument("--spectra-out-dir", type=Path)
    finalize_dispatch.add_argument("--spectra-run-id")
    finalize_dispatch.add_argument("--only-ok", action="store_true")
    finalize_dispatch.add_argument("--allow-incomplete", action="store_true")
    finalize_dispatch.add_argument("--campaign-id")
    finalize_dispatch.add_argument("--campaign-contract-out", type=Path)
    finalize_dispatch.add_argument("--injected-plan", type=Path)
    finalize_dispatch.add_argument("--injected-spectra-dir", type=Path)
    finalize_dispatch.add_argument("--injection-truth", type=Path)
    finalize_dispatch.add_argument("--candidate-dir", type=Path)
    finalize_dispatch.add_argument("--viewer-index-dir", type=Path)
    finalize_dispatch.add_argument("--score-baseline", action="store_true")
    finalize_dispatch.add_argument("--candidate-min-abs-zscore", type=float, default=5.0)
    finalize_dispatch.add_argument("--candidate-min-measurements", type=int, default=10)
    finalize_dispatch.add_argument("--candidate-max-rows", type=int)
    finalize_dispatch.add_argument("--score-injected", action="store_true")
    finalize_dispatch.add_argument("--recover-injections", action="store_true")
    finalize_dispatch.add_argument("--recovery-min-score", type=float, default=5.0)
    finalize_dispatch.add_argument("--recovery-wavelength-tolerance-nm", type=float, default=10.0)
    finalize_dispatch.add_argument("--recovery-require-line-family", action="store_true")
    finalize_dispatch.set_defaults(func=cmd_finalize_dispatch_run)

    score_candidates = sub.add_parser(
        "score-spectra-candidates",
        help="Run the simple RAPIDS target-zscore candidate scorer on assembled spectra.",
    )
    score_candidates.add_argument("--spectra", type=Path, required=True)
    score_candidates.add_argument("--out-dir", type=Path, required=True)
    score_candidates.add_argument("--run-id", required=True)
    score_candidates.add_argument("--device", default="cuda:0")
    score_candidates.add_argument("--output-prefix", default="baseline")
    score_candidates.add_argument("--flux-column", default="aperture_flux_uJy")
    score_candidates.add_argument("--min-abs-zscore", type=float, default=5.0)
    score_candidates.add_argument("--min-measurements", type=int, default=10)
    score_candidates.add_argument("--max-candidates", type=int)
    score_candidates.add_argument("--include-flagged", action="store_true")
    score_candidates.set_defaults(func=cmd_score_spectra_candidates)

    recovery = sub.add_parser(
        "score-injection-recovery",
        help="Join injected candidate tables against an injection manifest.",
    )
    recovery.add_argument("--manifest", type=Path, required=True)
    recovery.add_argument("--candidates", type=Path, required=True)
    recovery.add_argument("--out-dir", type=Path, required=True)
    recovery.add_argument("--min-score", type=float, default=5.0)
    recovery.add_argument("--wavelength-tolerance-nm", type=float, default=10.0)
    recovery.add_argument("--require-line-family", action="store_true")
    recovery.set_defaults(func=cmd_score_injection_recovery)

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

    k8s_post = sub.add_parser(
        "write-k8s-postprocess-job",
        help="Write a Kubernetes Job manifest for finalize-dispatch-run.",
    )
    k8s_post.add_argument("--plan", type=Path, required=True)
    k8s_post.add_argument("--out-dir", type=Path, required=True)
    k8s_post.add_argument("--image", required=True)
    k8s_post.add_argument("--namespace", default="default")
    k8s_post.add_argument("--service-account")
    k8s_post.add_argument("--container-executable", default="luxquarry-allsky")
    k8s_post.add_argument("--working-dir")
    k8s_post.add_argument("--device", default="cuda:0")
    k8s_post.add_argument("--gpu-limit", type=int, default=1)
    k8s_post.add_argument("--cpu-request", default="4")
    k8s_post.add_argument("--memory-request", default="16Gi")
    k8s_post.add_argument("--restart-policy", default="Never")
    k8s_post.add_argument("--backoff-limit", type=int, default=1)
    k8s_post.add_argument("--pvc-name")
    k8s_post.add_argument("--mount-path")
    k8s_post.add_argument("--campaign-id")
    k8s_post.add_argument("--spectra-out-dir", type=Path)
    k8s_post.add_argument("--spectra-run-id")
    k8s_post.add_argument("--campaign-contract-out", type=Path)
    k8s_post.add_argument("--injected-plan", type=Path)
    k8s_post.add_argument("--injected-spectra-dir", type=Path)
    k8s_post.add_argument("--injection-truth", type=Path)
    k8s_post.add_argument("--candidate-dir", type=Path)
    k8s_post.add_argument("--viewer-index-dir", type=Path)
    k8s_post.add_argument("--score-baseline", action="store_true")
    k8s_post.add_argument("--candidate-min-abs-zscore", type=float, default=5.0)
    k8s_post.add_argument("--candidate-min-measurements", type=int, default=10)
    k8s_post.add_argument("--candidate-max-rows", type=int)
    k8s_post.add_argument("--score-injected", action="store_true")
    k8s_post.add_argument("--recover-injections", action="store_true")
    k8s_post.add_argument("--recovery-min-score", type=float, default=5.0)
    k8s_post.add_argument("--recovery-wavelength-tolerance-nm", type=float, default=10.0)
    k8s_post.add_argument("--recovery-require-line-family", action="store_true")
    k8s_post.add_argument("--only-ok", action="store_true")
    k8s_post.add_argument("--allow-incomplete", action="store_true")
    k8s_post.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable to place on the postprocess container. Repeatable.",
    )
    k8s_post.set_defaults(func=cmd_write_k8s_postprocess_job)

    k8s_reducers = sub.add_parser(
        "write-k8s-reducer-jobs",
        help="Write one Kubernetes Job per spectra reducer from a reducer fanout plan.",
    )
    k8s_reducers.add_argument("--reducer-plan", type=Path, required=True)
    k8s_reducers.add_argument("--out-dir", type=Path, required=True)
    k8s_reducers.add_argument("--image", required=True)
    k8s_reducers.add_argument("--namespace", default="default")
    k8s_reducers.add_argument("--service-account")
    k8s_reducers.add_argument("--container-executable", default="luxquarry-allsky")
    k8s_reducers.add_argument("--working-dir")
    k8s_reducers.add_argument(
        "--device",
        default="cuda:0",
        help="Device string passed inside each reducer pod. Use cuda:0 for one-GPU pods.",
    )
    k8s_reducers.add_argument("--gpu-limit", type=int, default=1)
    k8s_reducers.add_argument("--cpu-request", default="2")
    k8s_reducers.add_argument("--memory-request", default="8Gi")
    k8s_reducers.add_argument("--restart-policy", default="Never")
    k8s_reducers.add_argument("--backoff-limit", type=int, default=1)
    k8s_reducers.add_argument("--pvc-name")
    k8s_reducers.add_argument("--mount-path")
    k8s_reducers.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable to place on every reducer container. Repeatable.",
    )
    k8s_reducers.set_defaults(func=cmd_write_k8s_reducer_jobs)

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
    spectra.add_argument(
        "--drop-duplicate-measurements",
        action="store_true",
        help="Drop retry-duplicate measurement rows by catalog/target/frame/image/detector before sorting.",
    )
    spectra.set_defaults(func=cmd_assemble_spectra)

    partitioned_spectra = sub.add_parser(
        "assemble-spectra-partitions",
        help="Build target-hash partitioned spectra products from measurement shards using cuDF.",
    )
    partitioned_spectra.add_argument("--shard-manifest", type=Path, required=True)
    partitioned_spectra.add_argument("--out-dir", type=Path, required=True)
    partitioned_spectra.add_argument("--run-id", required=True)
    partitioned_spectra.add_argument("--device", default="cuda:0")
    partitioned_spectra.add_argument("--partition-count", type=int, default=64)
    partitioned_spectra.add_argument(
        "--partition-index",
        type=int,
        help="Assemble only one target-hash partition. Omit to assemble all partitions sequentially.",
    )
    partitioned_spectra.add_argument("--only-ok", action="store_true")
    partitioned_spectra.add_argument("--drop-duplicate-measurements", action="store_true")
    partitioned_spectra.set_defaults(func=cmd_assemble_spectra_partitions)

    partition_measurements = sub.add_parser(
        "partition-measurement-shards",
        help="GPU-shuffle measurement shards into target-hash parquet buckets for reducer fanout.",
    )
    partition_measurements.add_argument("--shard-manifest", type=Path, required=True)
    partition_measurements.add_argument("--out-dir", type=Path, required=True)
    partition_measurements.add_argument("--run-id", required=True)
    partition_measurements.add_argument("--device", default="cuda:0")
    partition_measurements.add_argument("--partition-count", type=int, default=64)
    partition_measurements.add_argument("--only-ok", action="store_true")
    partition_measurements.add_argument("--drop-duplicate-measurements", action="store_true")
    partition_measurements.add_argument(
        "--write-empty-partitions",
        action="store_true",
        help="Write empty parquet/manifests for empty target buckets. Default skips empty buckets.",
    )
    partition_measurements.add_argument(
        "--summary-partition-limit",
        type=int,
        default=64,
        help="Maximum partition rows to embed in JSON/stdout. Full manifest is always written to parquet.",
    )
    partition_measurements.set_defaults(func=cmd_partition_measurement_shards)

    reducer_plan = sub.add_parser(
        "write-reducer-plan",
        help="Write a local/shell spectra reducer fanout plan from a measurement partition manifest.",
    )
    reducer_plan.add_argument("--partition-manifest", type=Path, required=True)
    reducer_plan.add_argument("--out-dir", type=Path, required=True)
    reducer_plan.add_argument("--run-id", required=True)
    reducer_plan.add_argument("--plan-out", type=Path, required=True)
    reducer_plan.add_argument("--executable", default=".venv/bin/luxquarry-allsky")
    reducer_plan.add_argument("--devices", default="cuda:0")
    reducer_plan.add_argument("--spectra-out-dir", type=Path)
    reducer_plan.add_argument("--only-ok", action="store_true")
    reducer_plan.add_argument(
        "--no-drop-duplicate-measurements",
        action="store_true",
        help="Do not pass --drop-duplicate-measurements to reducer assemble-spectra jobs.",
    )
    reducer_plan.add_argument("--max-partitions", type=int)
    reducer_plan.set_defaults(func=cmd_write_reducer_plan)

    run_reducers = sub.add_parser(
        "run-reducer-plan",
        help="Launch local spectra reducers from a reducer fanout plan.",
    )
    run_reducers.add_argument("--plan", type=Path, required=True)
    run_reducers.add_argument("--logs-dir", type=Path)
    run_reducers.add_argument("--resume", action="store_true")
    run_reducers.add_argument("--max-parallel", type=int)
    run_reducers.add_argument("--allow-failed-reducers", action="store_true")
    run_reducers.set_defaults(func=cmd_run_reducer_plan)

    collect_reducers = sub.add_parser(
        "collect-reducer-plan",
        help="Collect reducer assemble-spectra outputs into a manifest and aggregate JSON.",
    )
    collect_reducers.add_argument("--plan", type=Path, required=True)
    collect_reducers.add_argument("--out", type=Path)
    collect_reducers.add_argument("--allow-incomplete", action="store_true")
    collect_reducers.set_defaults(func=cmd_collect_reducer_plan)

    candidate_fanout_plan = sub.add_parser(
        "write-candidate-fanout-plan",
        help="Write a candidate-scorer fanout plan from reducer_outputs.parquet.",
    )
    candidate_fanout_plan.add_argument("--reducer-outputs", type=Path, required=True)
    candidate_fanout_plan.add_argument("--out-dir", type=Path, required=True)
    candidate_fanout_plan.add_argument("--run-id", required=True)
    candidate_fanout_plan.add_argument("--plan-out", type=Path, required=True)
    candidate_fanout_plan.add_argument("--executable", default=".venv/bin/luxquarry-allsky")
    candidate_fanout_plan.add_argument("--devices", default="cuda:0")
    candidate_fanout_plan.add_argument("--output-prefix", default="baseline")
    candidate_fanout_plan.add_argument("--flux-column", default="aperture_flux_uJy")
    candidate_fanout_plan.add_argument("--min-abs-zscore", type=float, default=5.0)
    candidate_fanout_plan.add_argument("--min-measurements", type=int, default=10)
    candidate_fanout_plan.add_argument("--include-flagged", action="store_true")
    candidate_fanout_plan.add_argument("--max-candidates", type=int)
    candidate_fanout_plan.add_argument("--max-partitions", type=int)
    candidate_fanout_plan.set_defaults(func=cmd_write_candidate_fanout_plan)

    run_candidate_fanout = sub.add_parser(
        "run-candidate-fanout-plan",
        help="Launch local candidate scorers from a candidate fanout plan.",
    )
    run_candidate_fanout.add_argument("--plan", type=Path, required=True)
    run_candidate_fanout.add_argument("--logs-dir", type=Path)
    run_candidate_fanout.add_argument("--resume", action="store_true")
    run_candidate_fanout.add_argument("--max-parallel", type=int)
    run_candidate_fanout.add_argument("--allow-failed-scorers", action="store_true")
    run_candidate_fanout.set_defaults(func=cmd_run_candidate_fanout_plan)

    collect_candidate_fanout = sub.add_parser(
        "collect-candidate-fanout-plan",
        help="Collect partitioned candidate scorer outputs into a manifest and aggregate JSON.",
    )
    collect_candidate_fanout.add_argument("--plan", type=Path, required=True)
    collect_candidate_fanout.add_argument("--out", type=Path)
    collect_candidate_fanout.add_argument("--allow-incomplete", action="store_true")
    collect_candidate_fanout.set_defaults(func=cmd_collect_candidate_fanout_plan)

    validate_assembly = sub.add_parser(
        "validate-assembly-order",
        help="Reassemble shuffled shard manifests and verify spectra outputs are logically identical.",
    )
    validate_assembly.add_argument("--shard-manifest", type=Path, required=True)
    validate_assembly.add_argument("--out-dir", type=Path, required=True)
    validate_assembly.add_argument("--run-id", required=True)
    validate_assembly.add_argument("--device", default="cuda:0")
    validate_assembly.add_argument("--only-ok", action="store_true")
    validate_assembly.add_argument("--drop-duplicate-measurements", action="store_true")
    validate_assembly.add_argument("--repetitions", type=int, default=2)
    validate_assembly.add_argument("--random-seed", type=int, default=1729)
    validate_assembly.set_defaults(func=cmd_validate_assembly_order)

    validate_retry_dedup = sub.add_parser(
        "validate-assembly-retry-dedup",
        help="Duplicate shard manifest entries and verify retry dedup recovers the baseline spectra.",
    )
    validate_retry_dedup.add_argument("--shard-manifest", type=Path, required=True)
    validate_retry_dedup.add_argument("--out-dir", type=Path, required=True)
    validate_retry_dedup.add_argument("--run-id", required=True)
    validate_retry_dedup.add_argument("--device", default="cuda:0")
    validate_retry_dedup.add_argument("--only-ok", action="store_true")
    validate_retry_dedup.add_argument("--duplicate-shard-count", type=int)
    validate_retry_dedup.set_defaults(func=cmd_validate_assembly_retry_dedup)

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


def cmd_run_dispatch_benchmark_sweep(args: argparse.Namespace) -> int:
    devices = tuple(part.strip() for part in args.devices.split(",") if part.strip())
    executable = args.executable or sys.argv[0]
    summary = run_dispatch_benchmark_sweep(
        DispatchBenchmarkSweepConfig(
            manifest_path=args.manifest,
            projected_targets_path=args.projected_targets,
            output_dir=args.out_dir,
            run_id=args.run_id,
            devices=devices,
            workers_per_device_values=_parse_int_tuple(args.workers_per_device, name="workers-per-device"),
            limit_frame_values=_parse_int_tuple(args.limit_frames, name="limit-frames"),
            shard_batch_frame_values=_parse_int_tuple(args.shard_batch_frames, name="shard-batch-frames"),
            prefetch_frame_values=_parse_int_tuple(args.prefetch_frames, name="prefetch-frames", allow_zero=True),
            repetitions=args.repetitions,
            cache_root=args.cache_root,
            local_cache_dir=args.local_cache_dir,
            executable=executable,
            async_shard_writes=not args.no_async_shard_writes,
            batch_table_assembly=not args.no_batch_table_assembly,
            materialize_worker_inputs=not args.no_materialize_worker_inputs,
            finalize_device=args.finalize_device,
            score_baseline=args.score_baseline,
            candidate_min_abs_zscore=args.candidate_min_abs_zscore,
            candidate_min_measurements=args.candidate_min_measurements,
            candidate_max_rows=args.candidate_max_rows,
            status_snapshot_interval_sec=args.status_snapshot_interval_sec,
            continue_on_error=args.continue_on_error,
            worker_only=args.worker_only,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
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


def cmd_rewrite_manifest_paths(args: argparse.Namespace) -> int:
    summary = rewrite_manifest_paths_to_uri(
        manifest_path=args.manifest,
        output_path=args.out,
        strip_prefix=args.strip_prefix,
        uri_prefix=args.uri_prefix,
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


def cmd_write_task_queue(args: argparse.Namespace) -> int:
    if args.frames_per_task <= 0:
        raise ValueError("--frames-per-task must be positive")
    summary = write_task_queue(
        TaskQueueWriteConfig(
            manifest_path=args.manifest,
            projected_targets_path=args.projected_targets,
            output_dir=args.out_dir,
            campaign_id=args.campaign_id,
            frames_per_task=args.frames_per_task,
            limit_frames=args.limit_frames,
            materialize_task_inputs=not args.no_materialize_task_inputs,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_run_gpu_worker_service(args: argparse.Namespace) -> int:
    if args.shard_batch_frames <= 0:
        raise ValueError("--shard-batch-frames must be positive")
    if args.prefetch_frames < 0:
        raise ValueError("--prefetch-frames must be non-negative")
    if args.status_interval_frames <= 0:
        raise ValueError("--status-interval-frames must be positive")
    if args.max_tasks is not None and args.max_tasks <= 0:
        raise ValueError("--max-tasks must be positive")
    summary = run_gpu_worker_service(
        GpuWorkerServiceConfig(
            queue_dir=args.queue_dir,
            output_dir=args.out_dir,
            run_id=args.run_id,
            worker_id=args.worker_id,
            max_tasks=args.max_tasks,
            worker_config=PersistentWorkerConfig(
                aperture=ApertureConfig(
                    cache_root=args.cache_root,
                    aperture_radius_pix=args.aperture_radius_pix,
                    annulus_inner_pix=args.annulus_inner_pix,
                    annulus_outer_pix=args.annulus_outer_pix,
                    edge_margin_pix=args.edge_margin_pix,
                ),
                device=args.device,
                worker_index=0,
                worker_count=1,
                rmm_pool=not args.no_rmm_pool,
                async_shard_writes=args.async_shard_writes,
                batch_table_assembly=args.batch_table_assembly,
                shard_batch_frames=args.shard_batch_frames,
                prefetch_frames=args.prefetch_frames,
                status_interval_frames=args.status_interval_frames,
                local_cache_dir=args.local_cache_dir,
            ),
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_collect_task_queue_run(args: argparse.Namespace) -> int:
    summary = collect_task_queue_run(
        TaskQueueCollectConfig(
            queue_dir=args.queue_dir,
            output_path=args.out,
        )
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


def cmd_run_local_dispatch(args: argparse.Namespace) -> int:
    devices = tuple(part.strip() for part in args.devices.split(",") if part.strip())
    executable = args.executable or sys.argv[0]
    summary = run_local_dispatch(
        LocalDispatchRunConfig(
            manifest_path=args.manifest,
            projected_targets_path=args.projected_targets,
            output_dir=args.out_dir,
            run_id=args.run_id,
            devices=devices,
            workers_per_device=args.workers_per_device,
            cache_root=args.cache_root,
            limit_frames=args.limit_frames,
            executable=executable,
            shard_batch_frames=args.shard_batch_frames,
            prefetch_frames=args.prefetch_frames,
            status_interval_frames=args.status_interval_frames,
            local_cache_dir=args.local_cache_dir,
            async_shard_writes=args.async_shard_writes,
            batch_table_assembly=args.batch_table_assembly,
            materialize_worker_inputs=not args.no_materialize_worker_inputs,
            finalize_device=args.finalize_device,
            spectra_out_dir=args.spectra_out_dir,
            spectra_run_id=args.spectra_run_id,
            only_ok=args.only_ok,
            allow_incomplete_finalize=args.allow_incomplete_finalize,
            campaign_id=args.campaign_id,
            campaign_contract_out=args.campaign_contract_out,
            candidate_dir=args.candidate_dir,
            score_baseline=args.score_baseline,
            candidate_min_abs_zscore=args.candidate_min_abs_zscore,
            candidate_min_measurements=args.candidate_min_measurements,
            candidate_max_rows=args.candidate_max_rows,
            score_injected=args.score_injected,
            injected_spectra_dir=args.injected_spectra_dir,
            injection_truth_path=args.injection_truth,
            recover_injections=args.recover_injections,
            recovery_min_score=args.recovery_min_score,
            recovery_wavelength_tolerance_nm=args.recovery_wavelength_tolerance_nm,
            recovery_require_line_family=args.recovery_require_line_family,
            resume=args.resume,
            status_snapshot_interval_sec=args.status_snapshot_interval_sec,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_collect_dispatch_run(args: argparse.Namespace) -> int:
    summary = collect_dispatch_run(plan_path=args.plan, output_path=args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_run_dispatch_plan_workers(args: argparse.Namespace) -> int:
    summary = run_dispatch_plan_workers(
        LocalPlanWorkerRunConfig(
            plan_path=args.plan,
            logs_dir=args.logs_dir,
            resume=args.resume,
            status_snapshot_interval_sec=args.status_snapshot_interval_sec,
            allow_failed_workers=args.allow_failed_workers,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_dispatch_status(args: argparse.Namespace) -> int:
    snapshot = write_dispatch_status_snapshot(
        DispatchStatusConfig(
            plan_path=args.plan,
            output_path=args.out,
        )
    )
    print(json.dumps(snapshot, indent=2, sort_keys=True))
    return 0


def cmd_finalize_dispatch_run(args: argparse.Namespace) -> int:
    summary = finalize_dispatch_run(
        FinalizeDispatchConfig(
            plan_path=args.plan,
            device=args.device,
            aggregate_out=args.aggregate_out,
            spectra_out_dir=args.spectra_out_dir,
            spectra_run_id=args.spectra_run_id,
            only_ok=args.only_ok,
            allow_incomplete=args.allow_incomplete,
            campaign_id=args.campaign_id,
            campaign_contract_out=args.campaign_contract_out,
            injected_plan_path=args.injected_plan,
            injected_spectra_dir=args.injected_spectra_dir,
            injection_truth_path=args.injection_truth,
            candidate_dir=args.candidate_dir,
            viewer_index_dir=args.viewer_index_dir,
            score_baseline=args.score_baseline,
            candidate_min_abs_zscore=args.candidate_min_abs_zscore,
            candidate_min_measurements=args.candidate_min_measurements,
            candidate_max_rows=args.candidate_max_rows,
            score_injected=args.score_injected,
            recover_injections=args.recover_injections,
            recovery_min_score=args.recovery_min_score,
            recovery_wavelength_tolerance_nm=args.recovery_wavelength_tolerance_nm,
            recovery_require_line_family=args.recovery_require_line_family,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_score_spectra_candidates(args: argparse.Namespace) -> int:
    summary = score_spectra_candidates(
        CandidateScoringConfig(
            spectra_path=args.spectra,
            output_dir=args.out_dir,
            run_id=args.run_id,
            device=args.device,
            output_prefix=args.output_prefix,
            flux_column=args.flux_column,
            min_abs_zscore=args.min_abs_zscore,
            min_measurements=args.min_measurements,
            only_ok=not args.include_flagged,
            max_candidates=args.max_candidates,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_score_injection_recovery(args: argparse.Namespace) -> int:
    summary = score_injection_recovery(
        InjectionRecoveryConfig(
            manifest_path=args.manifest,
            candidates_path=args.candidates,
            output_dir=args.out_dir,
            min_score=args.min_score,
            wavelength_tolerance_nm=args.wavelength_tolerance_nm,
            require_line_family=args.require_line_family,
        )
    )
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


def cmd_write_k8s_postprocess_job(args: argparse.Namespace) -> int:
    summary = write_kubernetes_postprocess_job(
        KubernetesPostprocessJobConfig(
            plan_path=args.plan,
            output_dir=args.out_dir,
            image=args.image,
            namespace=args.namespace,
            service_account=args.service_account,
            container_executable=args.container_executable,
            working_dir=args.working_dir,
            device=args.device,
            gpu_limit=args.gpu_limit,
            cpu_request=args.cpu_request,
            memory_request=args.memory_request,
            restart_policy=args.restart_policy,
            backoff_limit=args.backoff_limit,
            pvc_name=args.pvc_name,
            mount_path=args.mount_path,
            campaign_id=args.campaign_id,
            spectra_out_dir=args.spectra_out_dir,
            spectra_run_id=args.spectra_run_id,
            campaign_contract_out=args.campaign_contract_out,
            injected_plan_path=args.injected_plan,
            injected_spectra_dir=args.injected_spectra_dir,
            injection_truth_path=args.injection_truth,
            candidate_dir=args.candidate_dir,
            viewer_index_dir=args.viewer_index_dir,
            score_baseline=args.score_baseline,
            candidate_min_abs_zscore=args.candidate_min_abs_zscore,
            candidate_min_measurements=args.candidate_min_measurements,
            candidate_max_rows=args.candidate_max_rows,
            score_injected=args.score_injected,
            recover_injections=args.recover_injections,
            recovery_min_score=args.recovery_min_score,
            recovery_wavelength_tolerance_nm=args.recovery_wavelength_tolerance_nm,
            recovery_require_line_family=args.recovery_require_line_family,
            only_ok=args.only_ok,
            allow_incomplete=args.allow_incomplete,
            env=_parse_env_assignments(args.env),
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_write_k8s_reducer_jobs(args: argparse.Namespace) -> int:
    summary = write_kubernetes_reducer_jobs(
        KubernetesReducerJobConfig(
            reducer_plan_path=args.reducer_plan,
            output_dir=args.out_dir,
            image=args.image,
            namespace=args.namespace,
            service_account=args.service_account,
            container_executable=args.container_executable,
            working_dir=args.working_dir,
            device=args.device,
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
            drop_duplicate_measurements=args.drop_duplicate_measurements,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_assemble_spectra_partitions(args: argparse.Namespace) -> int:
    summary = assemble_spectra_partitions(
        PartitionedSpectraAssemblyConfig(
            shard_manifest_path=args.shard_manifest,
            output_dir=args.out_dir,
            run_id=args.run_id,
            device=args.device,
            partition_count=args.partition_count,
            partition_index=args.partition_index,
            only_ok=args.only_ok,
            drop_duplicate_measurements=args.drop_duplicate_measurements,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_partition_measurement_shards(args: argparse.Namespace) -> int:
    summary = partition_measurement_shards(
        MeasurementPartitionConfig(
            shard_manifest_path=args.shard_manifest,
            output_dir=args.out_dir,
            run_id=args.run_id,
            device=args.device,
            partition_count=args.partition_count,
            only_ok=args.only_ok,
            drop_duplicate_measurements=args.drop_duplicate_measurements,
            write_empty_partitions=args.write_empty_partitions,
            summary_partition_limit=args.summary_partition_limit,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_write_reducer_plan(args: argparse.Namespace) -> int:
    devices = tuple(part.strip() for part in args.devices.split(",") if part.strip())
    plan = build_reducer_plan(
        ReducerPlanConfig(
            partition_manifest_path=args.partition_manifest,
            output_dir=args.out_dir,
            run_id=args.run_id,
            executable=args.executable,
            devices=devices,
            spectra_out_dir=args.spectra_out_dir,
            only_ok=args.only_ok,
            drop_duplicate_measurements=not args.no_drop_duplicate_measurements,
            max_partitions=args.max_partitions,
        )
    )
    write_reducer_plan(plan, args.plan_out)
    print(
        json.dumps(
            {
                "plan_out": str(args.plan_out),
                "shell_out": str(args.plan_out.with_suffix(".sh")),
                "reducer_count": plan["reducer_count"],
                "devices": plan["devices"],
                "total_measurement_rows": plan["total_measurement_rows"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_run_reducer_plan(args: argparse.Namespace) -> int:
    summary = run_reducer_plan(
        LocalReducerRunConfig(
            plan_path=args.plan,
            logs_dir=args.logs_dir,
            resume=args.resume,
            max_parallel=args.max_parallel,
            allow_failed_reducers=args.allow_failed_reducers,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_collect_reducer_plan(args: argparse.Namespace) -> int:
    summary = collect_reducer_plan(
        ReducerCollectConfig(
            plan_path=args.plan,
            output_path=args.out,
            allow_incomplete=args.allow_incomplete,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_write_candidate_fanout_plan(args: argparse.Namespace) -> int:
    devices = tuple(part.strip() for part in args.devices.split(",") if part.strip())
    plan = build_candidate_fanout_plan(
        CandidateFanoutPlanConfig(
            reducer_outputs_path=args.reducer_outputs,
            output_dir=args.out_dir,
            run_id=args.run_id,
            executable=args.executable,
            devices=devices,
            output_prefix=args.output_prefix,
            flux_column=args.flux_column,
            min_abs_zscore=args.min_abs_zscore,
            min_measurements=args.min_measurements,
            include_flagged=args.include_flagged,
            max_candidates=args.max_candidates,
            max_partitions=args.max_partitions,
        )
    )
    write_candidate_fanout_plan(plan, args.plan_out)
    print(
        json.dumps(
            {
                "plan_out": str(args.plan_out),
                "shell_out": str(args.plan_out.with_suffix(".sh")),
                "scorer_count": plan["scorer_count"],
                "devices": plan["devices"],
                "total_spectra_measurement_rows": plan["total_spectra_measurement_rows"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_run_candidate_fanout_plan(args: argparse.Namespace) -> int:
    summary = run_candidate_fanout_plan(
        LocalCandidateFanoutRunConfig(
            plan_path=args.plan,
            logs_dir=args.logs_dir,
            resume=args.resume,
            max_parallel=args.max_parallel,
            allow_failed_scorers=args.allow_failed_scorers,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_collect_candidate_fanout_plan(args: argparse.Namespace) -> int:
    summary = collect_candidate_fanout_plan(
        CandidateFanoutCollectConfig(
            plan_path=args.plan,
            output_path=args.out,
            allow_incomplete=args.allow_incomplete,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_validate_assembly_order(args: argparse.Namespace) -> int:
    summary = validate_shard_order_independent_assembly(
        SpectraAssemblyValidationConfig(
            shard_manifest_path=args.shard_manifest,
            output_dir=args.out_dir,
            run_id=args.run_id,
            device=args.device,
            only_ok=args.only_ok,
            drop_duplicate_measurements=args.drop_duplicate_measurements,
            repetitions=args.repetitions,
            random_seed=args.random_seed,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_validate_assembly_retry_dedup(args: argparse.Namespace) -> int:
    summary = validate_retry_dedup_assembly(
        SpectraRetryDedupValidationConfig(
            shard_manifest_path=args.shard_manifest,
            output_dir=args.out_dir,
            run_id=args.run_id,
            device=args.device,
            only_ok=args.only_ok,
            duplicate_shard_count=args.duplicate_shard_count,
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


def _parse_int_tuple(raw: str, *, name: str, allow_zero: bool = False) -> tuple[int, ...]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 0 or (value == 0 and not allow_zero):
            floor = "non-negative" if allow_zero else "positive"
            raise ValueError(f"--{name} values must be {floor}, got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"--{name} must contain at least one integer")
    return tuple(values)


def _rate(count: int, elapsed_sec: float) -> float:
    return float(count / elapsed_sec) if elapsed_sec > 0 and count else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
