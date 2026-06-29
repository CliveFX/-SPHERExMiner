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
