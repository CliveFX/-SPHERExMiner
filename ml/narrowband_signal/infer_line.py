#!/usr/bin/env python3
"""Run narrowband line/no-line ML inference for one miner run."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.common.ragged_spectra import FEATURE_COLUMNS, make_point_features, write_status
from ml.narrowband_signal.train_line import LineExample, build_model, _collate


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise SystemExit("PyTorch is required for narrowband line inference.") from exc
    return torch


def main() -> None:
    torch = _require_torch()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-points", type=int, default=384)
    parser.add_argument("--candidate-threshold", type=float, default=0.5)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--target-ids-file", type=Path)
    parser.add_argument("--recovery-tolerance-nm", type=float, default=10.0)
    args = parser.parse_args()

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "ml_narrowband_status.json"
    write_status(
        status_path,
        {
            "status": "running",
            "run_dir": str(args.run_dir),
            "checkpoint": str(args.checkpoint),
            "started_at_unix": time.time(),
        },
    )

    spectra_path = args.run_dir / "spectra" / "target_spectra.parquet"
    if not spectra_path.exists():
        raise SystemExit(f"Missing spectra table: {spectra_path}")
    spectra = pd.read_parquet(spectra_path)
    target_ids = _read_target_ids(args.target_ids_file)
    if target_ids:
        spectra = spectra[spectra["target_id"].astype(str).isin(target_ids)].copy()

    load_start = time.perf_counter()
    examples, meta_rows = _examples_from_spectra(spectra, args.max_points)
    feature_sec = time.perf_counter() - load_start
    if not examples:
        raise SystemExit("No examples available for ML inference")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    line_min_nm = float(ckpt.get("line_min_nm") or 700.0)
    line_max_nm = float(ckpt.get("line_max_nm") or 5000.0)
    architecture = str(ckpt.get("architecture") or "set")
    hidden_dim, embedding_dim = _infer_dims(ckpt)
    transformer_layers = int(ckpt.get("transformer_layers") or 3)
    transformer_heads = int(ckpt.get("transformer_heads") or 6)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = build_model(
        torch=torch,
        nn=torch.nn,
        architecture=architecture,
        input_dim=len(FEATURE_COLUMNS),
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    infer_start = time.perf_counter()
    probabilities: list[float] = []
    line_nm: list[float] = []
    with torch.inference_mode():
        for batch in _batches(examples, int(args.batch_size)):
            x, mask, _y, _line = _collate(batch, args.max_points, line_min_nm, line_max_nm, torch, device)
            logits, pred_line_norm = model(x, mask)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            pred = pred_line_norm.detach().cpu().numpy()
            probabilities.extend(float(v) for v in prob)
            line_nm.extend(float(line_min_nm + np.clip(v, 0.0, 1.0) * (line_max_nm - line_min_nm)) for v in pred)
        if device.type == "cuda":
            torch.cuda.synchronize()
    inference_sec = time.perf_counter() - infer_start

    scores = pd.DataFrame(meta_rows)
    scores["ml_signal_probability"] = probabilities
    scores["ml_predicted_line_nm"] = line_nm
    scores["ml_candidate"] = scores["ml_signal_probability"] >= float(args.candidate_threshold)
    scores["ml_architecture"] = architecture
    scores["ml_model_checkpoint"] = str(args.checkpoint)
    scores = scores.sort_values("ml_signal_probability", ascending=False, kind="mergesort")

    scores_path = args.output_dir / "target_scores.parquet"
    candidates_path = args.output_dir / "ml_candidates.parquet"
    scores.to_parquet(scores_path, index=False)
    scores[scores["ml_candidate"]].to_parquet(candidates_path, index=False)

    recovery = _score_manifest_recovery(scores, args.manifest, float(args.recovery_tolerance_nm))
    recovery_path = args.output_dir / "ml_injection_recovery.parquet"
    if recovery is not None:
        recovery.to_parquet(recovery_path, index=False)

    summary = _summary(
        args=args,
        scores=scores,
        recovery=recovery,
        architecture=architecture,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        feature_sec=feature_sec,
        inference_sec=inference_sec,
        total_sec=time.perf_counter() - started,
        scores_path=scores_path,
        candidates_path=candidates_path,
        recovery_path=recovery_path if recovery is not None else None,
        device=str(device),
    )
    summary_path = args.output_dir / "ml_narrowband_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    write_status(status_path, {**summary, "status": "done", "updated_at_unix": time.time()})
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _examples_from_spectra(spectra: pd.DataFrame, max_points: int) -> tuple[list[LineExample], list[dict[str, Any]]]:
    examples: list[LineExample] = []
    rows: list[dict[str, Any]] = []
    for target_id, group in spectra.groupby("target_id", sort=False):
        features = make_point_features(group)
        if len(features) < 8:
            continue
        if max_points and len(features) > max_points:
            features = features[:max_points]
        first = group.iloc[0]
        examples.append(
            LineExample(
                dataset_name="",
                run_name=str(first.get("run_name") or ""),
                split_id="inference",
                target_id=str(target_id),
                is_positive=0.0,
                line_nm=0.0,
                features=features,
            )
        )
        rows.append(
            {
                "run_name": str(first.get("run_name") or ""),
                "target_id": str(target_id),
                "source_id": None if pd.isna(first.get("source_id")) else str(first.get("source_id")),
                "object_name": None if pd.isna(first.get("object_name")) else str(first.get("object_name")),
                "n_points": int(len(group)),
                "n_model_points": int(len(features)),
                "wavelength_min_um": _num(group.get("cwave_um", pd.Series(dtype=float)).min()),
                "wavelength_max_um": _num(group.get("cwave_um", pd.Series(dtype=float)).max()),
                "phot_g_mean_mag": _num(first.get("phot_g_mean_mag")),
                "bp_rp": _num(first.get("bp_rp")),
                "flagged_fraction": float(group.get("fatal_flag_present", pd.Series(False, index=group.index)).fillna(False).astype(bool).mean()),
                "detector_count": int(group.get("detector", pd.Series(dtype=object)).nunique()),
            }
        )
    return examples, rows


def _read_target_ids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _score_manifest_recovery(scores: pd.DataFrame, manifest_path: Path | None, tolerance_nm: float) -> pd.DataFrame | None:
    if manifest_path is None or not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    truth_rows = []
    score_by_target = scores.set_index(scores["target_id"].astype(str)).to_dict(orient="index")
    for inj in manifest.get("injections", []):
        target_id = str(inj.get("target_id"))
        score = score_by_target.get(target_id, {})
        injected_line_nm = _num(inj.get("injected_line_nm"))
        predicted_line_nm = _num(score.get("ml_predicted_line_nm"))
        probability = _num(score.get("ml_signal_probability"))
        wavelength_error_nm = abs(predicted_line_nm - injected_line_nm) if predicted_line_nm is not None and injected_line_nm is not None else None
        truth_rows.append(
            {
                "injection_id": inj.get("injection_id"),
                "target_id": target_id,
                "line_family": inj.get("line_family"),
                "injected_line_nm": injected_line_nm,
                "find_me_snr": _num(inj.get("find_me_snr")),
                "line_flux_uJy": _num(inj.get("line_flux_uJy")),
                "ml_signal_probability": probability,
                "ml_predicted_line_nm": predicted_line_nm,
                "ml_wavelength_error_nm": wavelength_error_nm,
                "ml_recovered": bool(probability is not None and probability >= 0.5 and wavelength_error_nm is not None and wavelength_error_nm <= tolerance_nm),
                "ml_detected_any_line": bool(probability is not None and probability >= 0.5),
            }
        )
    return pd.DataFrame(truth_rows)


def _summary(
    *,
    args: argparse.Namespace,
    scores: pd.DataFrame,
    recovery: pd.DataFrame | None,
    architecture: str,
    hidden_dim: int,
    embedding_dim: int,
    transformer_layers: int,
    transformer_heads: int,
    feature_sec: float,
    inference_sec: float,
    total_sec: float,
    scores_path: Path,
    candidates_path: Path,
    recovery_path: Path | None,
    device: str,
) -> dict[str, Any]:
    probs = pd.to_numeric(scores["ml_signal_probability"], errors="coerce")
    thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
    summary: dict[str, Any] = {
        "run_dir": str(args.run_dir),
        "checkpoint": str(args.checkpoint),
        "architecture": architecture,
        "hidden_dim": int(hidden_dim),
        "embedding_dim": int(embedding_dim),
        "transformer_layers": int(transformer_layers) if architecture == "transformer" else None,
        "transformer_heads": int(transformer_heads) if architecture == "transformer" else None,
        "device": device,
        "scored_target_count": int(len(scores)),
        "candidate_threshold": float(args.candidate_threshold),
        "candidate_count": int((probs >= float(args.candidate_threshold)).sum()),
        "probability_mean": _num(probs.mean()),
        "probability_median": _num(probs.median()),
        "probability_max": _num(probs.max()),
        "threshold_counts": {str(threshold): int((probs >= threshold).sum()) for threshold in thresholds},
        "feature_sec": float(feature_sec),
        "inference_sec": float(inference_sec),
        "total_sec": float(total_sec),
        "spectra_per_sec_inference": float(len(scores) / inference_sec) if inference_sec > 0 else None,
        "spectra_per_sec_total": float(len(scores) / total_sec) if total_sec > 0 else None,
        "target_scores": str(scores_path),
        "candidates": str(candidates_path),
        "recovery": str(recovery_path) if recovery_path is not None else None,
        "top_candidates": _json_records(
            scores.head(12)[
                ["target_id", "ml_signal_probability", "ml_predicted_line_nm", "n_points", "phot_g_mean_mag", "flagged_fraction"]
            ]
        ),
    }
    if recovery is not None:
        recovered = int(recovery["ml_recovered"].fillna(False).astype(bool).sum()) if "ml_recovered" in recovery else 0
        detected = int(recovery["ml_detected_any_line"].fillna(False).astype(bool).sum()) if "ml_detected_any_line" in recovery else 0
        total = int(len(recovery))
        summary.update(
            {
                "injection_count": total,
                "ml_recovered_count": recovered,
                "ml_detected_any_line_count": detected,
                "ml_recovery_fraction": float(recovered / total) if total else None,
                "ml_detected_any_line_fraction": float(detected / total) if total else None,
                "ml_wavelength_mae_nm": _num(pd.to_numeric(recovery["ml_wavelength_error_nm"], errors="coerce").mean())
                if "ml_wavelength_error_nm" in recovery
                else None,
            }
        )
    return summary


def _infer_dims(ckpt: dict[str, Any]) -> tuple[int, int]:
    state = ckpt["model_state_dict"]
    if "point.0.weight" in state:
        return int(state["point.0.weight"].shape[0]), int(state["embed.2.weight"].shape[0])
    return int(state["input.0.weight"].shape[0]), int(state["embed.1.weight"].shape[0])


def _batches(items: list[LineExample], batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: _json_value(value) for key, value in row.items()} for row in df.to_dict(orient="records")]


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if pd.isna(value):
        return None
    return value


if __name__ == "__main__":
    main()
