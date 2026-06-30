from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KubernetesJobConfig:
    plan_path: Path
    output_dir: Path
    image: str
    namespace: str = "default"
    service_account: str | None = None
    container_executable: str = "luxquarry-allsky"
    working_dir: str | None = None
    gpu_limit: int = 1
    cpu_request: str = "4"
    memory_request: str = "16Gi"
    restart_policy: str = "Never"
    backoff_limit: int = 1
    pvc_name: str | None = None
    mount_path: str | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class KubernetesPostprocessJobConfig:
    plan_path: Path
    output_dir: Path
    image: str
    namespace: str = "default"
    service_account: str | None = None
    container_executable: str = "luxquarry-allsky"
    working_dir: str | None = None
    device: str = "cuda:0"
    gpu_limit: int = 1
    cpu_request: str = "4"
    memory_request: str = "16Gi"
    restart_policy: str = "Never"
    backoff_limit: int = 1
    pvc_name: str | None = None
    mount_path: str | None = None
    campaign_id: str | None = None
    spectra_out_dir: Path | None = None
    spectra_run_id: str | None = None
    campaign_contract_out: Path | None = None
    injected_plan_path: Path | None = None
    injected_spectra_dir: Path | None = None
    injection_truth_path: Path | None = None
    candidate_dir: Path | None = None
    viewer_index_dir: Path | None = None
    score_baseline: bool = False
    candidate_min_abs_zscore: float = 5.0
    candidate_min_measurements: int = 10
    candidate_max_rows: int | None = None
    score_injected: bool = False
    recover_injections: bool = False
    recovery_min_score: float = 5.0
    recovery_wavelength_tolerance_nm: float = 10.0
    recovery_require_line_family: bool = False
    only_ok: bool = False
    allow_incomplete: bool = False
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class KubernetesReducerJobConfig:
    reducer_plan_path: Path
    output_dir: Path
    image: str
    namespace: str = "default"
    service_account: str | None = None
    container_executable: str = "luxquarry-allsky"
    working_dir: str | None = None
    device: str | None = "cuda:0"
    gpu_limit: int = 1
    cpu_request: str = "2"
    memory_request: str = "8Gi"
    restart_policy: str = "Never"
    backoff_limit: int = 1
    pvc_name: str | None = None
    mount_path: str | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class KubernetesCandidateScorerJobConfig:
    candidate_plan_path: Path
    output_dir: Path
    image: str
    namespace: str = "default"
    service_account: str | None = None
    container_executable: str = "luxquarry-allsky"
    working_dir: str | None = None
    device: str | None = "cuda:0"
    gpu_limit: int = 1
    cpu_request: str = "2"
    memory_request: str = "8Gi"
    restart_policy: str = "Never"
    backoff_limit: int = 1
    pvc_name: str | None = None
    mount_path: str | None = None
    env: dict[str, str] = field(default_factory=dict)


def write_kubernetes_jobs(config: KubernetesJobConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(config.plan_path.read_text(encoding="utf-8"))
    jobs = [_worker_job(plan, worker, config) for worker in plan.get("workers") or []]
    manifest_path = config.output_dir / f"{plan.get('run_id', 'luxquarry')}.worker-jobs.yaml"
    manifest_path.write_text(_as_yaml_documents(jobs), encoding="utf-8")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "kubernetes_job_manifest_generator",
        "plan_path": str(config.plan_path),
        "run_id": plan.get("run_id"),
        "namespace": config.namespace,
        "image": config.image,
        "job_count": len(jobs),
        "manifest_path": str(manifest_path),
        "materialize_worker_inputs": bool(plan.get("materialize_worker_inputs")),
        "worker_names": [job["metadata"]["name"] for job in jobs],
    }
    summary_path = config.output_dir / "k8s_jobs_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def write_kubernetes_postprocess_job(config: KubernetesPostprocessJobConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(config.plan_path.read_text(encoding="utf-8"))
    run_id = str(plan.get("run_id") or "luxquarry")
    job = _postprocess_job(plan, config)
    manifest_path = config.output_dir / f"{run_id}.postprocess-job.yaml"
    manifest_path.write_text(_as_yaml_documents([job]), encoding="utf-8")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "kubernetes_postprocess_job_manifest_generator",
        "plan_path": str(config.plan_path),
        "run_id": run_id,
        "namespace": config.namespace,
        "image": config.image,
        "job_count": 1,
        "manifest_path": str(manifest_path),
        "job_name": job["metadata"]["name"],
    }
    summary_path = config.output_dir / "k8s_postprocess_job_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def write_kubernetes_reducer_jobs(config: KubernetesReducerJobConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(config.reducer_plan_path.read_text(encoding="utf-8"))
    jobs = [_reducer_job(plan, reducer, config) for reducer in plan.get("reducers") or []]
    run_id = str(plan.get("run_id") or "luxquarry-reducers")
    manifest_path = config.output_dir / f"{run_id}.reducer-jobs.yaml"
    manifest_path.write_text(_as_yaml_documents(jobs), encoding="utf-8")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "kubernetes_reducer_job_manifest_generator",
        "reducer_plan_path": str(config.reducer_plan_path),
        "run_id": run_id,
        "namespace": config.namespace,
        "image": config.image,
        "job_count": len(jobs),
        "manifest_path": str(manifest_path),
        "total_measurement_rows": int(plan.get("total_measurement_rows") or 0),
        "worker_names": [job["metadata"]["name"] for job in jobs],
    }
    summary_path = config.output_dir / "k8s_reducer_jobs_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def write_kubernetes_candidate_scorer_jobs(config: KubernetesCandidateScorerJobConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(config.candidate_plan_path.read_text(encoding="utf-8"))
    jobs = [_candidate_scorer_job(plan, scorer, config) for scorer in plan.get("scorers") or []]
    run_id = str(plan.get("run_id") or "luxquarry-candidate-scorers")
    manifest_path = config.output_dir / f"{run_id}.candidate-scorer-jobs.yaml"
    manifest_path.write_text(_as_yaml_documents(jobs), encoding="utf-8")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "kubernetes_candidate_scorer_job_manifest_generator",
        "candidate_plan_path": str(config.candidate_plan_path),
        "run_id": run_id,
        "namespace": config.namespace,
        "image": config.image,
        "job_count": len(jobs),
        "manifest_path": str(manifest_path),
        "total_spectra_measurement_rows": int(plan.get("total_spectra_measurement_rows") or 0),
        "total_target_count_by_partition": int(plan.get("total_target_count_by_partition") or 0),
        "worker_names": [job["metadata"]["name"] for job in jobs],
    }
    summary_path = config.output_dir / "k8s_candidate_scorer_jobs_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _worker_job(plan: dict[str, Any], worker: dict[str, Any], config: KubernetesJobConfig) -> dict[str, Any]:
    worker_id = str(worker["worker_id"])
    argv = list(worker["argv"])
    command = [config.container_executable]
    args = argv[1:] if argv else []
    env = [{"name": key, "value": value} for key, value in sorted(config.env.items())]
    container: dict[str, Any] = {
        "name": "luxquarry-worker",
        "image": config.image,
        "imagePullPolicy": "IfNotPresent",
        "command": command,
        "args": args,
        "env": env,
        "resources": {
            "limits": {"nvidia.com/gpu": config.gpu_limit},
            "requests": {
                "cpu": config.cpu_request,
                "memory": config.memory_request,
                "nvidia.com/gpu": config.gpu_limit,
            },
        },
    }
    if config.working_dir:
        container["workingDir"] = config.working_dir
    pod_spec: dict[str, Any] = {
        "restartPolicy": config.restart_policy,
        "containers": [container],
    }
    if config.service_account:
        pod_spec["serviceAccountName"] = config.service_account
    if config.pvc_name and config.mount_path:
        pod_spec["volumes"] = [{"name": "luxquarry-data", "persistentVolumeClaim": {"claimName": config.pvc_name}}]
        container["volumeMounts"] = [{"name": "luxquarry-data", "mountPath": config.mount_path}]

    labels = {
        "app.kubernetes.io/name": "luxquarry-allsky",
        "luxquarry/run-id": _label_value(str(plan.get("run_id") or "run")),
        "luxquarry/worker-index": str(worker.get("worker_index", 0)),
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": _job_name(worker_id),
            "namespace": config.namespace,
            "labels": labels,
        },
        "spec": {
            "backoffLimit": config.backoff_limit,
            "template": {
                "metadata": {"labels": labels},
                "spec": pod_spec,
            },
        },
    }


def _reducer_job(plan: dict[str, Any], reducer: dict[str, Any], config: KubernetesReducerJobConfig) -> dict[str, Any]:
    run_id = str(plan.get("run_id") or "reducers")
    reducer_id = str(reducer["reducer_id"])
    argv = list(reducer["argv"])
    args = argv[1:] if argv else []
    if config.device:
        args = _replace_option(args, "--device", config.device)
    labels = {
        "app.kubernetes.io/name": "luxquarry-allsky",
        "luxquarry/run-id": _label_value(run_id),
        "luxquarry/job-role": "spectra-reducer",
        "luxquarry/partition-index": str(reducer.get("partition_index", 0)),
    }
    container = _container(
        name="luxquarry-reducer",
        image=config.image,
        command=[config.container_executable],
        args=args,
        env=config.env,
        working_dir=config.working_dir,
        gpu_limit=config.gpu_limit,
        cpu_request=config.cpu_request,
        memory_request=config.memory_request,
        pvc_name=config.pvc_name,
        mount_path=config.mount_path,
    )
    pod_spec = _pod_spec(
        container=container,
        restart_policy=config.restart_policy,
        service_account=config.service_account,
        pvc_name=config.pvc_name,
        mount_path=config.mount_path,
    )
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": _job_name(reducer_id),
            "namespace": config.namespace,
            "labels": labels,
        },
        "spec": {
            "backoffLimit": config.backoff_limit,
            "template": {
                "metadata": {"labels": labels},
                "spec": pod_spec,
            },
        },
    }


def _candidate_scorer_job(
    plan: dict[str, Any],
    scorer: dict[str, Any],
    config: KubernetesCandidateScorerJobConfig,
) -> dict[str, Any]:
    run_id = str(plan.get("run_id") or "candidate-scorers")
    scorer_id = str(scorer["scorer_id"])
    argv = list(scorer["argv"])
    args = argv[1:] if argv else []
    if config.device:
        args = _replace_option(args, "--device", config.device)
    labels = {
        "app.kubernetes.io/name": "luxquarry-allsky",
        "luxquarry/run-id": _label_value(run_id),
        "luxquarry/job-role": "candidate-scorer",
        "luxquarry/partition-index": str(scorer.get("partition_index", 0)),
    }
    container = _container(
        name="luxquarry-candidate-scorer",
        image=config.image,
        command=[config.container_executable],
        args=args,
        env=config.env,
        working_dir=config.working_dir,
        gpu_limit=config.gpu_limit,
        cpu_request=config.cpu_request,
        memory_request=config.memory_request,
        pvc_name=config.pvc_name,
        mount_path=config.mount_path,
    )
    pod_spec = _pod_spec(
        container=container,
        restart_policy=config.restart_policy,
        service_account=config.service_account,
        pvc_name=config.pvc_name,
        mount_path=config.mount_path,
    )
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": _job_name(scorer_id),
            "namespace": config.namespace,
            "labels": labels,
        },
        "spec": {
            "backoffLimit": config.backoff_limit,
            "template": {
                "metadata": {"labels": labels},
                "spec": pod_spec,
            },
        },
    }


def _postprocess_job(plan: dict[str, Any], config: KubernetesPostprocessJobConfig) -> dict[str, Any]:
    run_id = str(plan.get("run_id") or "luxquarry")
    args = [
        "finalize-dispatch-run",
        "--plan",
        str(config.plan_path),
        "--device",
        config.device,
    ]
    if config.campaign_id:
        args.extend(["--campaign-id", config.campaign_id])
    if config.spectra_out_dir:
        args.extend(["--spectra-out-dir", str(config.spectra_out_dir)])
    if config.spectra_run_id:
        args.extend(["--spectra-run-id", config.spectra_run_id])
    if config.campaign_contract_out:
        args.extend(["--campaign-contract-out", str(config.campaign_contract_out)])
    if config.injected_plan_path:
        args.extend(["--injected-plan", str(config.injected_plan_path)])
    if config.injected_spectra_dir:
        args.extend(["--injected-spectra-dir", str(config.injected_spectra_dir)])
    if config.injection_truth_path:
        args.extend(["--injection-truth", str(config.injection_truth_path)])
    if config.candidate_dir:
        args.extend(["--candidate-dir", str(config.candidate_dir)])
    if config.viewer_index_dir:
        args.extend(["--viewer-index-dir", str(config.viewer_index_dir)])
    if config.score_baseline:
        args.append("--score-baseline")
    if config.candidate_min_abs_zscore != 5.0:
        args.extend(["--candidate-min-abs-zscore", str(config.candidate_min_abs_zscore)])
    if config.candidate_min_measurements != 10:
        args.extend(["--candidate-min-measurements", str(config.candidate_min_measurements)])
    if config.candidate_max_rows is not None:
        args.extend(["--candidate-max-rows", str(config.candidate_max_rows)])
    if config.score_injected:
        args.append("--score-injected")
    if config.recover_injections:
        args.append("--recover-injections")
    if config.recovery_min_score != 5.0:
        args.extend(["--recovery-min-score", str(config.recovery_min_score)])
    if config.recovery_wavelength_tolerance_nm != 10.0:
        args.extend(["--recovery-wavelength-tolerance-nm", str(config.recovery_wavelength_tolerance_nm)])
    if config.recovery_require_line_family:
        args.append("--recovery-require-line-family")
    if config.only_ok:
        args.append("--only-ok")
    if config.allow_incomplete:
        args.append("--allow-incomplete")

    labels = {
        "app.kubernetes.io/name": "luxquarry-allsky",
        "luxquarry/run-id": _label_value(run_id),
        "luxquarry/job-role": "postprocess",
    }
    container = _container(
        name="luxquarry-postprocess",
        image=config.image,
        command=[config.container_executable],
        args=args,
        env=config.env,
        working_dir=config.working_dir,
        gpu_limit=config.gpu_limit,
        cpu_request=config.cpu_request,
        memory_request=config.memory_request,
        pvc_name=config.pvc_name,
        mount_path=config.mount_path,
    )
    pod_spec = _pod_spec(
        container=container,
        restart_policy=config.restart_policy,
        service_account=config.service_account,
        pvc_name=config.pvc_name,
        mount_path=config.mount_path,
    )
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": _job_name(f"{run_id}.postprocess"),
            "namespace": config.namespace,
            "labels": labels,
        },
        "spec": {
            "backoffLimit": config.backoff_limit,
            "template": {
                "metadata": {"labels": labels},
                "spec": pod_spec,
            },
        },
    }


def _container(
    *,
    name: str,
    image: str,
    command: list[str],
    args: list[str],
    env: dict[str, str],
    working_dir: str | None,
    gpu_limit: int,
    cpu_request: str,
    memory_request: str,
    pvc_name: str | None,
    mount_path: str | None,
) -> dict[str, Any]:
    container: dict[str, Any] = {
        "name": name,
        "image": image,
        "imagePullPolicy": "IfNotPresent",
        "command": command,
        "args": args,
        "env": [{"name": key, "value": value} for key, value in sorted(env.items())],
        "resources": {
            "limits": {"nvidia.com/gpu": gpu_limit},
            "requests": {
                "cpu": cpu_request,
                "memory": memory_request,
                "nvidia.com/gpu": gpu_limit,
            },
        },
    }
    if working_dir:
        container["workingDir"] = working_dir
    if pvc_name and mount_path:
        container["volumeMounts"] = [{"name": "luxquarry-data", "mountPath": mount_path}]
    return container


def _pod_spec(
    *,
    container: dict[str, Any],
    restart_policy: str,
    service_account: str | None,
    pvc_name: str | None,
    mount_path: str | None,
) -> dict[str, Any]:
    pod_spec: dict[str, Any] = {
        "restartPolicy": restart_policy,
        "containers": [container],
    }
    if service_account:
        pod_spec["serviceAccountName"] = service_account
    if pvc_name and mount_path:
        pod_spec["volumes"] = [{"name": "luxquarry-data", "persistentVolumeClaim": {"claimName": pvc_name}}]
    return pod_spec


def _as_yaml_documents(objects: list[dict[str, Any]]) -> str:
    # JSON is valid YAML 1.2, and keeping the documents JSON-shaped makes the
    # generator dependency-free and easy to validate in the local venv.
    return "\n".join("---\n" + json.dumps(obj, indent=2, sort_keys=True) for obj in objects) + "\n"


def _job_name(value: str) -> str:
    lowered = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    lowered = re.sub(r"-+", "-", lowered)
    return lowered[:63].rstrip("-")


def _label_value(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned[:63].rstrip("-")


def _replace_option(args: list[str], option: str, value: str) -> list[str]:
    out = list(args)
    for idx, item in enumerate(out):
        if item == option:
            if idx + 1 < len(out):
                out[idx + 1] = value
                return out
            out.append(value)
            return out
    out.extend([option, value])
    return out
