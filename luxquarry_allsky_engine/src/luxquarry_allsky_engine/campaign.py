from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CampaignContractConfig:
    campaign_id: str
    output_path: Path
    baseline_plan_path: Path
    baseline_spectra_dir: Path | None = None
    injected_plan_path: Path | None = None
    injected_spectra_dir: Path | None = None
    injection_truth_path: Path | None = None
    candidate_dir: Path | None = None
    viewer_index_dir: Path | None = None


def write_campaign_contract(config: CampaignContractConfig) -> dict[str, Any]:
    baseline_plan = _read_json(config.baseline_plan_path)
    baseline_run_id = str(baseline_plan.get("run_id") or "baseline")
    baseline_output_dir = _run_output_dir(config.baseline_plan_path, baseline_plan)
    baseline_spectra_dir = config.baseline_spectra_dir or baseline_output_dir / "spectra"
    baseline_spectra_run_id, baseline_spectra_artifacts = _spectra_artifacts(
        baseline_spectra_dir,
        fallback_run_id=baseline_run_id,
    )

    injected_plan: dict[str, Any] | None = None
    injected_run_id = "injected"
    injected_spectra_run_id: str | None = None
    injected_output_dir: Path | None = None
    injected_spectra_dir = config.injected_spectra_dir
    injected_spectra_artifacts: list[Path] = []
    if config.injected_plan_path:
        injected_plan = _read_json(config.injected_plan_path)
        injected_run_id = str(injected_plan.get("run_id") or "injected")
        injected_output_dir = _run_output_dir(config.injected_plan_path, injected_plan)
        injected_spectra_dir = injected_spectra_dir or injected_output_dir / "spectra"
    if injected_spectra_dir:
        injected_spectra_run_id, injected_spectra_artifacts = _spectra_artifacts(
            injected_spectra_dir,
            fallback_run_id=injected_run_id,
        )

    candidate_dir = config.candidate_dir or config.output_path.parent / "candidates"
    viewer_index_dir = config.viewer_index_dir or config.output_path.parent / "viewer_indexes"

    stages = [
        _stage(
            "baseline_dispatch",
            "complete",
            [
                config.baseline_plan_path,
                baseline_output_dir / "aggregate_summary.json",
                baseline_output_dir / "measurement_shard_manifest.parquet",
            ],
            "GPU frame photometry workers over the uninjected frame/target plan.",
        ),
        _stage(
            "baseline_spectra_assembly",
            "complete",
            baseline_spectra_artifacts,
            "Ragged target spectra assembled from baseline measurement shards.",
        ),
        _stage(
            "baseline_blind_scoring",
            "complete",
            [
                candidate_dir / "baseline_candidates.parquet",
                candidate_dir / "baseline_candidate_summary.json",
            ],
            "Raw and quality-gated candidate scan over baseline spectra.",
        ),
        _stage(
            "injected_dispatch",
            "complete",
            _present_paths(
                config.injected_plan_path,
                injected_output_dir / "aggregate_summary.json" if injected_output_dir else None,
                injected_output_dir / "measurement_shard_manifest.parquet" if injected_output_dir else None,
            ),
            "GPU frame photometry over the same target/frame contract with synthetic injections enabled.",
            blocked_by=None if config.injected_plan_path else "missing injected dispatch plan",
        ),
        _stage(
            "injected_spectra_assembly",
            "complete",
            injected_spectra_artifacts,
            "Ragged target spectra assembled from injected measurement shards.",
            blocked_by=None if injected_spectra_dir else "missing injected spectra directory",
        ),
        _stage(
            "injected_blind_scoring",
            "complete",
            [
                candidate_dir / "injected_candidates.parquet",
                candidate_dir / "injected_candidate_summary.json",
            ],
            "Raw and quality-gated candidate scan over injected spectra.",
        ),
        _stage(
            "truth_target_recovery",
            "complete",
            _present_paths(
                config.injection_truth_path,
                candidate_dir / "truth_recovery_summary.json",
                candidate_dir / "false_positive_summary.json",
            ),
            "Recovery accounting that joins injection truth to injected and baseline candidate tables.",
            blocked_by=None if config.injection_truth_path else "missing injection truth table",
        ),
        _stage(
            "viewer_indexes",
            "complete",
            [
                viewer_index_dir / "spectra_index.parquet",
                viewer_index_dir / "candidate_index.parquet",
                viewer_index_dir / "recovery_index.parquet",
            ],
            "Viewer-facing indexes for spectra, candidates, injections, and recovery review.",
        ),
    ]

    complete = all(stage["status"] == "complete" for stage in stages)
    contract = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "luxquarry_campaign_contract",
        "campaign_id": config.campaign_id,
        "science_complete": complete,
        "baseline_run_id": baseline_run_id,
        "baseline_spectra_run_id": baseline_spectra_run_id,
        "injected_run_id": injected_run_id if injected_plan else None,
        "injected_spectra_run_id": injected_spectra_run_id,
        "baseline_plan_path": str(config.baseline_plan_path),
        "injected_plan_path": str(config.injected_plan_path) if config.injected_plan_path else None,
        "stage_count": len(stages),
        "complete_stage_count": sum(1 for stage in stages if stage["status"] == "complete"),
        "missing_stage_count": sum(1 for stage in stages if stage["status"] != "complete"),
        "stages": stages,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contract


def _stage(
    name: str,
    complete_status: str,
    artifacts: list[Path],
    description: str,
    *,
    blocked_by: str | None = None,
) -> dict[str, Any]:
    rows = [{"path": str(path), "exists": path.exists()} for path in artifacts]
    if blocked_by:
        status = "blocked"
    elif rows and all(row["exists"] for row in rows):
        status = complete_status
    else:
        status = "missing"
    return {
        "name": name,
        "description": description,
        "status": status,
        "blocked_by": blocked_by,
        "artifacts": rows,
        "missing_artifacts": [row["path"] for row in rows if not row["exists"]],
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _present_paths(*paths: Path | None) -> list[Path]:
    return [path for path in paths if path is not None]


def _spectra_artifacts(spectra_dir: Path, *, fallback_run_id: str) -> tuple[str, list[Path]]:
    assemble_summary_path = spectra_dir / "assemble_summary.json"
    run_id = fallback_run_id
    spectra_path = spectra_dir / f"{fallback_run_id}.spectra_measurements.parquet"
    target_summary_path = spectra_dir / f"{fallback_run_id}.target_summary.parquet"
    if assemble_summary_path.exists():
        summary = _read_json(assemble_summary_path)
        run_id = str(summary.get("run_id") or fallback_run_id)
        spectra_path = _resolve_path(spectra_dir, summary.get("spectra_measurements_path")) or spectra_path
        target_summary_path = _resolve_path(spectra_dir, summary.get("target_summary_path")) or target_summary_path
    return run_id, [assemble_summary_path, spectra_path, target_summary_path]


def _run_output_dir(plan_path: Path, plan: dict[str, Any]) -> Path:
    output_dir = Path(str(plan.get("output_dir") or plan_path.parent))
    return output_dir if output_dir.is_absolute() else plan_path.parent.parent.parent / output_dir


def _resolve_path(reference_dir: Path, value: object) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    roots = [Path.cwd(), *reference_dir.parents]
    for root in roots:
        candidate = root / path
        if candidate.exists():
            return candidate
    return reference_dir / path.name
