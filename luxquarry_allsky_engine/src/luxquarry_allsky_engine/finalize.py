from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .campaign import CampaignContractConfig, write_campaign_contract
from .dispatch import collect_dispatch_run
from .recovery import InjectionRecoveryConfig, score_injection_recovery
from .scoring import CandidateScoringConfig, score_spectra_candidates
from .spectra import SpectraAssemblyConfig, assemble_spectra_from_shards


@dataclass(frozen=True)
class FinalizeDispatchConfig:
    plan_path: Path
    device: str = "cuda:0"
    aggregate_out: Path | None = None
    spectra_out_dir: Path | None = None
    spectra_run_id: str | None = None
    only_ok: bool = False
    allow_incomplete: bool = False
    campaign_id: str | None = None
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


def finalize_dispatch_run(config: FinalizeDispatchConfig) -> dict[str, Any]:
    started = time.perf_counter()
    plan = _read_json(config.plan_path)
    run_id = str(plan.get("run_id") or config.plan_path.parent.name or "luxquarry")
    run_output_dir = _run_output_dir(config.plan_path, plan)

    aggregate = collect_dispatch_run(plan_path=config.plan_path, output_path=config.aggregate_out)
    if not config.allow_incomplete and not aggregate.get("complete"):
        raise RuntimeError(
            "Dispatch run is incomplete; refusing to assemble spectra. "
            "Use --allow-incomplete for diagnostic post-processing."
        )

    shard_manifest_path = _resolve_existing(config.plan_path, aggregate["shard_manifest_path"])
    spectra_out_dir = config.spectra_out_dir or run_output_dir / "spectra"
    spectra_run_id = config.spectra_run_id or run_id
    spectra = assemble_spectra_from_shards(
        SpectraAssemblyConfig(
            shard_manifest_path=shard_manifest_path,
            output_dir=spectra_out_dir,
            run_id=spectra_run_id,
            device=config.device,
            only_ok=config.only_ok,
        )
    )

    candidate_dir = config.candidate_dir or run_output_dir / "candidates"
    baseline_scoring: dict[str, Any] | None = None
    if config.score_baseline:
        baseline_scoring = score_spectra_candidates(
            CandidateScoringConfig(
                spectra_path=Path(spectra["spectra_measurements_path"]),
                output_dir=candidate_dir,
                run_id=spectra_run_id,
                device=config.device,
                output_prefix="baseline",
                min_abs_zscore=config.candidate_min_abs_zscore,
                min_measurements=config.candidate_min_measurements,
                max_candidates=config.candidate_max_rows,
            )
        )

    injected_scoring: dict[str, Any] | None = None
    if config.score_injected:
        if config.injected_spectra_dir is None:
            raise ValueError("--score-injected requires --injected-spectra-dir")
        injected_spectra_path = _find_spectra_measurements_path(config.injected_spectra_dir)
        injected_scoring = score_spectra_candidates(
            CandidateScoringConfig(
                spectra_path=injected_spectra_path,
                output_dir=candidate_dir,
                run_id=injected_spectra_path.stem.removesuffix(".spectra_measurements"),
                device=config.device,
                output_prefix="injected",
                min_abs_zscore=config.candidate_min_abs_zscore,
                min_measurements=config.candidate_min_measurements,
                max_candidates=config.candidate_max_rows,
            )
        )

    recovery: dict[str, Any] | None = None
    if config.recover_injections:
        if config.injection_truth_path is None:
            raise ValueError("--recover-injections requires --injection-truth")
        injected_candidates_path = candidate_dir / "injected_candidates.parquet"
        recovery = score_injection_recovery(
            InjectionRecoveryConfig(
                manifest_path=config.injection_truth_path,
                candidates_path=_require_existing_path(injected_candidates_path),
                output_dir=candidate_dir,
                min_score=config.recovery_min_score,
                wavelength_tolerance_nm=config.recovery_wavelength_tolerance_nm,
                require_line_family=config.recovery_require_line_family,
            )
        )

    campaign_id = config.campaign_id or f"{run_id}_campaign"
    campaign_contract_out = config.campaign_contract_out or run_output_dir / "campaign_contract.json"
    campaign = write_campaign_contract(
        CampaignContractConfig(
            campaign_id=campaign_id,
            output_path=campaign_contract_out,
            baseline_plan_path=config.plan_path,
            baseline_spectra_dir=spectra_out_dir,
            injected_plan_path=config.injected_plan_path,
            injected_spectra_dir=config.injected_spectra_dir,
            injection_truth_path=config.injection_truth_path,
            candidate_dir=candidate_dir,
            viewer_index_dir=config.viewer_index_dir,
        )
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "luxquarry_finalize_dispatch",
        "run_id": run_id,
        "plan_path": str(config.plan_path),
        "device": config.device,
        "allow_incomplete": config.allow_incomplete,
        "aggregate_summary_path": str(config.aggregate_out or run_output_dir / "aggregate_summary.json"),
        "shard_manifest_path": str(shard_manifest_path),
        "spectra_output_dir": str(spectra_out_dir),
        "spectra_run_id": spectra_run_id,
        "spectra_summary_path": str(spectra_out_dir / "assemble_summary.json"),
        "candidate_dir": str(candidate_dir),
        "campaign_contract_path": str(campaign_contract_out),
        "dispatch_complete": bool(aggregate.get("complete")),
        "science_complete": bool(campaign.get("science_complete")),
        "measurement_rows": int(aggregate.get("measurement_rows") or 0),
        "spectra_measurement_rows": int(spectra.get("spectra_measurement_rows") or 0),
        "target_count": int(spectra.get("target_count") or 0),
        "total_wall_sec": time.perf_counter() - started,
        "aggregate": aggregate,
        "spectra": spectra,
        "baseline_scoring": baseline_scoring,
        "injected_scoring": injected_scoring,
        "recovery": recovery,
        "campaign": campaign,
    }
    summary_path = run_output_dir / "finalize_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_output_dir(plan_path: Path, plan: dict[str, Any]) -> Path:
    configured = Path(str(plan.get("output_dir") or plan_path.parent))
    if configured.is_absolute() or configured.exists():
        return configured
    if plan_path.parent.name == configured.name:
        return plan_path.parent
    return configured


def _resolve_existing(reference_path: Path, value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute() or path.exists():
        return path
    for root in [Path.cwd(), *reference_path.parents]:
        candidate = root / path
        if candidate.exists():
            return candidate
    return path


def _find_spectra_measurements_path(spectra_dir: Path) -> Path:
    summary_path = spectra_dir / "assemble_summary.json"
    if summary_path.exists():
        summary = _read_json(summary_path)
        value = summary.get("spectra_measurements_path")
        if value:
            path = Path(str(value))
            if path.exists():
                return path
            candidate = spectra_dir / path.name
            if candidate.exists():
                return candidate
    matches = sorted(spectra_dir.glob("*.spectra_measurements.parquet"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No *.spectra_measurements.parquet found in {spectra_dir}")
    raise ValueError(f"Multiple spectra measurement files found in {spectra_dir}; pass a directory with one run")


def _require_existing_path(path: Path) -> Path:
    if path.exists():
        return path
    raise FileNotFoundError(path)
