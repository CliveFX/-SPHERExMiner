#!/usr/bin/env python3
"""Build and query LLM evidence packets for SPHEREx narrowband candidates."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PROMPT_DIR = ROOT / "prompts"
RESPONSE_DIR = ROOT / "responses"

DEFAULT_RUN_ROOT = Path("/mnt/niroseti/spherex_cache/runs")
DEFAULT_CAMPAIGN = "cv_june_g11_16_f500_diag_overnight_v1"
DEFAULT_MODEL = "qwen3.5:27b"
DEFAULT_BASE_URL = "http://spark-09dd:11434/v1"

CHALLENGE_SIZES = (5, 10, 25, 100)
SCIENCE_WAVELENGTH_SOURCE = "spectral_wcs_CWAVE_CBAND"
PACKET_FORMATS = ("verbose", "compact-v1")


SYSTEM_PROMPT = """You are a skeptical anomalous-signal investigator.

You review compact evidence packets from a spectral survey. Your job is to rank
whether each challenge spectrum contains a narrowband excess or high-value
anomaly. You are not allowed to claim discovery. You must separate signal-like
evidence from artifacts, flags, noisy continua, detector effects, and ambiguous
competing peaks.

Return strict JSON only. Do not include prose outside the JSON object.
"""


USER_PROMPTS = {
    "positive": """You are an anomalous signal investigator looking for defects, imperfections, excesses, and high-value signals in a stream.

I will provide a set of spectra with known signals acquired through other means,
then a slate of unlabeled spectra to analyze. Use the known examples to calibrate
what injected narrowband excesses look like after SPHEREx sampling, flags, PSF
photometry, and aperture photometry.

Respond with JSON matching the provided schema. Be decisive when the data support
it, but explicitly call out uncertainty and artifact risks.""",
    "negative": """You are an anomalous signal investigator looking for defects, imperfections, excesses, and high-value signals in a stream.

I will provide a set of spectra with known non-signals, rejected artifacts, and
ambiguous failures, then a slate of unlabeled spectra to analyze. Use the known
examples to calibrate what false positives, noise, flags, and bad candidate
patterns look like after SPHEREx sampling, flags, PSF photometry, and aperture
photometry.

Respond with JSON matching the provided schema. Be conservative and skeptical;
avoid promoting weak or artifact-like cases.""",
}


RETURN_SCHEMA = {
    "packet_id": "string",
    "model": "string",
    "prompt_variant": "positive|negative",
    "analyses": [
        {
            "signal_id": "string",
            "has_anomalous_excess": True,
            "confidence": 0.0,
            "priority": "S|A|B|C|D",
            "best_line_nm": 1064.0,
            "line_family_guess": "nd_yag_1064|diode_808|diode_980|telecom_1310|telecom_1550|thulium_2000|unknown|null",
            "artifact_likelihood": 0.0,
            "beam_likelihood": 0.0,
            "reason_codes": [
                "aperture_psf_agree",
                "localized_excess",
                "good_support",
                "flags_present",
                "too_many_competing_peaks",
            ],
            "evidence_summary": "short sentence",
            "recommended_next_check": "short sentence",
        }
    ],
    "global_notes": "short string",
}


@dataclass(frozen=True)
class SpectrumSource:
    run_dir: Path
    target_id: str
    injected: bool
    injection: dict[str, Any] | None
    recovery: dict[str, Any] | None
    decoy_of_signal_id: str | None = None


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_simple_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _env_default(env_values: dict[str, str], key: str, default: str) -> str:
    return os.environ.get(key) or env_values.get(key) or default


def _api_key_from_args(args: argparse.Namespace) -> str | None:
    api_key_env = getattr(args, "api_key_env", "")
    if not api_key_env:
        return None
    env_values = _read_simple_env(getattr(args, "env_file", Path(".env")))
    return os.environ.get(api_key_env) or env_values.get(api_key_env)


def _compact_float(value: Any, scale: int) -> int | None:
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return int(round(value * scale))


def _compact_spectrum(packet: dict[str, Any], *, reveal_truth: bool) -> dict[str, Any]:
    points = packet.get("points") or []
    image_ids: dict[str, str] = {}
    image_codes: list[str | None] = []
    for point in points:
        image_id = point.get("image_id")
        if not image_id:
            image_codes.append(None)
            continue
        image_id = str(image_id)
        if image_id not in image_ids:
            image_ids[image_id] = _base36(len(image_ids))
        image_codes.append(image_ids[image_id])

    compact = {
        "sid": packet.get("signal_id"),
        "run": packet.get("run"),
        "tid": packet.get("target_id"),
        "typ": packet.get("target_type"),
        "g": _compact_float(packet.get("phot_g_mean_mag"), 1000),
        "bp": _compact_float(packet.get("bp_rp"), 1000),
        "sum": {
            "n": packet.get("summary", {}).get("n_measurements"),
            "w0": _compact_float(packet.get("summary", {}).get("wave_min_nm"), 1000),
            "w1": _compact_float(packet.get("summary", {}).get("wave_max_nm"), 1000),
            "ff": packet.get("summary", {}).get("fatal_flag_count"),
            "det": packet.get("summary", {}).get("unique_detectors"),
            "map": _compact_float(packet.get("summary", {}).get("median_aperture_uJy"), 1000),
            "mps": _compact_float(packet.get("summary", {}).get("median_psf_uJy"), 1000),
        },
        "detout": {
            "m": packet.get("known_detector_output", {}).get("gpu_narrowband_match"),
            "qm": packet.get("known_detector_output", {}).get("gpu_narrowband_quality_match"),
            "ln": _compact_float(packet.get("known_detector_output", {}).get("gpu_narrowband_line_nm"), 1000),
            "rho": _compact_float(packet.get("known_detector_output", {}).get("gpu_narrowband_joint_rho"), 1000),
            "qt": packet.get("known_detector_output", {}).get("gpu_narrowband_quality_tier"),
            "qr": packet.get("known_detector_output", {}).get("gpu_narrowband_quality_reject_reasons"),
        },
        "scale": "w,bw nm*1000; flux,unc uJy*1000; ed pix*1000; g,bp*1000",
        "arr": {
            "w": [_compact_float(p.get("wave_nm"), 1000) for p in points],
            "bw": [_compact_float(p.get("bandwidth_nm"), 1000) for p in points],
            "ap": [_compact_float(p.get("aperture_uJy"), 1000) for p in points],
            "apu": [_compact_float(p.get("aperture_unc_uJy"), 1000) for p in points],
            "ps": [_compact_float(p.get("psf_uJy"), 1000) for p in points],
            "psu": [_compact_float(p.get("psf_unc_uJy"), 1000) for p in points],
            "d": [p.get("detector") for p in points],
            "ff": [1 if p.get("fatal_flag") else 0 for p in points],
            "fl": [p.get("flags_summary") for p in points],
            "ed": [_compact_float(p.get("edge_distance_pix"), 1000) for p in points],
            "img": image_codes,
        },
        "imgs": {code: image_id for image_id, code in image_ids.items()},
    }
    if reveal_truth and "truth" in packet:
        truth = packet["truth"]
        compact["truth"] = {
            "inj": bool(truth.get("is_injected")),
            "decoy": truth.get("decoy_of_signal_id"),
            "id": truth.get("injection_id"),
            "fam": truth.get("line_family"),
            "ln": _compact_float(truth.get("injected_line_nm"), 1000),
            "snr": _compact_float(truth.get("find_me_snr"), 1000),
            "flux": _compact_float(truth.get("line_flux_uJy"), 1000),
            "frames": truth.get("frames_written"),
        }
    return _json_safe(compact)


def _base36(value: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    if value == 0:
        return "0"
    out = ""
    while value:
        value, rem = divmod(value, 36)
        out = digits[rem] + out
    return out


def _compact_packet(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "packet_id": f"{payload.get('packet_id')}_compact_v1",
        "campaign": payload.get("campaign"),
        "format": "compact-v1",
        "format_notes": {
            "goal": "Lossless relative to verbose serialized spectra, except long object names are omitted from compact records.",
            "keys": {
                "sid": "signal_id",
                "tid": "target_id",
                "g": "phot_g_mean_mag*1000",
                "bp": "bp_rp*1000",
                "sum": "summary metadata",
                "detout": "GPU detector output, useful context but not truth",
                "arr": "columnar spectral arrays",
                "imgs": "image-id dictionary, keyed by arr.img codes",
            },
            "array_scales": {
                "w": "wavelength_nm*1000",
                "bw": "bandwidth_nm*1000",
                "ap": "aperture_uJy*1000",
                "apu": "aperture_unc_uJy*1000",
                "ps": "psf_uJy*1000",
                "psu": "psf_unc_uJy*1000",
                "ed": "edge_distance_pix*1000",
                "ff": "fatal_flag 0/1",
                "fl": "flags_summary bitmask",
                "d": "detector",
            },
        },
        "instructions": payload.get("instructions"),
        "known_signal_examples": [_compact_spectrum(row, reveal_truth=True) for row in payload.get("known_signal_examples", [])],
        "known_non_signal_examples": [_compact_spectrum(row, reveal_truth=True) for row in payload.get("known_non_signal_examples", [])],
        "challenge_spectra": [_compact_spectrum(row, reveal_truth=False) for row in payload.get("challenge_spectra", [])],
    }


def _packet_path(size: int, packet_format: str) -> Path:
    if packet_format == "verbose":
        return DATA_DIR / f"challenge_{size}.json"
    if packet_format == "compact-v1":
        return DATA_DIR / f"compact_challenge_{size}.json"
    raise ValueError(f"unknown packet format: {packet_format}")


def _campaign_run_dirs(run_root: Path, campaign: str) -> list[Path]:
    return sorted(path for path in run_root.glob(f"{campaign}_*_injected") if path.is_dir())


def _nearest_indices(wave_nm: np.ndarray, center_nm: float, limit: int) -> np.ndarray:
    finite = np.isfinite(wave_nm)
    if not finite.any():
        return np.array([], dtype=int)
    order = np.argsort(np.abs(wave_nm[finite] - center_nm))
    finite_idx = np.flatnonzero(finite)
    return np.sort(finite_idx[order[:limit]])


def _evenly_spaced_indices(size: int, limit: int) -> np.ndarray:
    if size <= limit:
        return np.arange(size, dtype=int)
    return np.unique(np.linspace(0, size - 1, limit, dtype=int))


def _robust_continuum(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.nanmedian(finite))


def _serialize_points(target_rows: pd.DataFrame, injection_nm: float | None, *, max_points: int) -> list[dict[str, Any]]:
    rows = target_rows.sort_values("cwave_um").reset_index(drop=True)
    wave_nm = pd.to_numeric(rows["cwave_um"], errors="coerce").to_numpy(dtype=float) * 1000.0
    if injection_nm is not None:
        local = _nearest_indices(wave_nm, float(injection_nm), max_points // 2)
        global_idx = _evenly_spaced_indices(len(rows), max_points - len(local))
        idx = np.unique(np.concatenate([local, global_idx]))
    else:
        idx = _evenly_spaced_indices(len(rows), max_points)
    selected = rows.iloc[idx].copy()
    out: list[dict[str, Any]] = []
    for row in selected.to_dict(orient="records"):
        aperture = row.get("aperture_flux_uJy")
        psf = row.get("psf_flux_uJy")
        aperture_unc = row.get("aperture_flux_unc_uJy")
        psf_unc = row.get("psf_flux_unc_uJy")
        out.append(
            {
                "wave_nm": round(float(row["cwave_um"]) * 1000.0, 3),
                "bandwidth_nm": round(float(row["cband_um"]) * 1000.0, 3) if pd.notna(row.get("cband_um")) else None,
                "aperture_uJy": round(float(aperture), 4) if pd.notna(aperture) else None,
                "aperture_unc_uJy": round(float(aperture_unc), 4) if pd.notna(aperture_unc) else None,
                "psf_uJy": round(float(psf), 4) if pd.notna(psf) else None,
                "psf_unc_uJy": round(float(psf_unc), 4) if pd.notna(psf_unc) else None,
                "detector": int(row["detector"]) if pd.notna(row.get("detector")) else None,
                "fatal_flag": bool(row.get("fatal_flag_present")) if pd.notna(row.get("fatal_flag_present")) else False,
                "flags_summary": int(row["flags_summary"]) if pd.notna(row.get("flags_summary")) else None,
                "edge_distance_pix": round(float(row["edge_distance_pix"]), 3) if pd.notna(row.get("edge_distance_pix")) else None,
                "image_id": row.get("image_id"),
            }
        )
    return out


def _target_summary(target_rows: pd.DataFrame) -> dict[str, Any]:
    aperture = pd.to_numeric(target_rows["aperture_flux_uJy"], errors="coerce").to_numpy(dtype=float)
    psf = pd.to_numeric(target_rows["psf_flux_uJy"], errors="coerce").to_numpy(dtype=float)
    return {
        "n_measurements": int(len(target_rows)),
        "wave_min_nm": round(float(target_rows["cwave_um"].min()) * 1000.0, 3),
        "wave_max_nm": round(float(target_rows["cwave_um"].max()) * 1000.0, 3),
        "fatal_flag_count": int(target_rows["fatal_flag_present"].fillna(False).astype(bool).sum()),
        "unique_detectors": sorted(int(x) for x in target_rows["detector"].dropna().unique().tolist()),
        "median_aperture_uJy": _robust_continuum(aperture),
        "median_psf_uJy": _robust_continuum(psf),
    }


def serialize_spectrum(
    source: SpectrumSource,
    spectra_cache: dict[Path, pd.DataFrame],
    *,
    reveal_truth: bool,
    max_points: int,
) -> dict[str, Any] | None:
    if source.run_dir not in spectra_cache:
        spectra_path = source.run_dir / "spectra" / "target_spectra.parquet"
        if not spectra_path.exists():
            return None
        cols = [
            "target_id",
            "target_type",
            "object_name",
            "phot_g_mean_mag",
            "bp_rp",
            "cwave_um",
            "cband_um",
            "wavelength_source",
            "aperture_flux_uJy",
            "aperture_flux_unc_uJy",
            "psf_flux_uJy",
            "psf_flux_unc_uJy",
            "detector",
            "image_id",
            "edge_distance_pix",
            "fatal_flag_present",
            "flags_summary",
        ]
        spectra_cache[source.run_dir] = pd.read_parquet(spectra_path, columns=cols)
    spectra = spectra_cache[source.run_dir]
    target_rows = spectra[spectra["target_id"].astype(str).eq(str(source.target_id))].copy()
    if target_rows.empty:
        return None
    wavelength_sources = set(target_rows["wavelength_source"].dropna().astype(str).unique().tolist())
    injection = source.injection or {}
    injection_nm = injection.get("injected_line_nm")
    packet = {
        "signal_id": f"{source.run_dir.name}::{source.target_id}",
        "run": source.run_dir.name,
        "target_id": source.target_id,
        "object_name": str(target_rows["object_name"].dropna().iloc[0]) if target_rows["object_name"].notna().any() else None,
        "target_type": str(target_rows["target_type"].dropna().iloc[0]) if target_rows["target_type"].notna().any() else None,
        "phot_g_mean_mag": _json_safe(target_rows["phot_g_mean_mag"].dropna().iloc[0]) if target_rows["phot_g_mean_mag"].notna().any() else None,
        "bp_rp": _json_safe(target_rows["bp_rp"].dropna().iloc[0]) if target_rows["bp_rp"].notna().any() else None,
        "wavelength_source": sorted(wavelength_sources),
        "summary": _target_summary(target_rows),
        "known_detector_output": {
            "gpu_narrowband_match": (source.recovery or {}).get("gpu_narrowband_match"),
            "gpu_narrowband_quality_match": (source.recovery or {}).get("gpu_narrowband_quality_match"),
            "gpu_narrowband_line_nm": (source.recovery or {}).get("gpu_narrowband_line_nm"),
            "gpu_narrowband_joint_rho": (source.recovery or {}).get("gpu_narrowband_joint_rho"),
            "gpu_narrowband_quality_tier": (source.recovery or {}).get("gpu_narrowband_quality_tier"),
            "gpu_narrowband_quality_reject_reasons": (source.recovery or {}).get("gpu_narrowband_quality_reject_reasons"),
        },
        "points": _serialize_points(target_rows, float(injection_nm) if injection_nm is not None else None, max_points=max_points),
    }
    if reveal_truth:
        packet["truth"] = {
            "is_injected": bool(source.injected),
            "decoy_of_signal_id": source.decoy_of_signal_id,
            "injection_id": injection.get("injection_id"),
            "line_family": injection.get("line_family"),
            "injected_line_nm": injection.get("injected_line_nm"),
            "find_me_snr": injection.get("find_me_snr"),
            "line_flux_uJy": injection.get("line_flux_uJy"),
            "frames_written": injection.get("frames_written"),
        }
    return _json_safe(packet)


def _load_injected_sources(run_dirs: list[Path]) -> list[SpectrumSource]:
    sources: list[SpectrumSource] = []
    for run_dir in run_dirs:
        recovery_path = run_dir / "narrowband_detector_truth" / "narrowband_recovery.parquet"
        if not recovery_path.exists():
            continue
        recovery = pd.read_parquet(recovery_path)
        for row in recovery.to_dict(orient="records"):
            injection = {
                "injection_id": row.get("injection_id"),
                "target_id": str(row.get("target_id")),
                "line_family": row.get("line_family"),
                "injected_line_nm": row.get("injected_line_nm"),
                "find_me_snr": row.get("find_me_snr"),
                "line_flux_uJy": row.get("line_flux_uJy"),
                "frames_written": row.get("frames_written"),
            }
            sources.append(
                SpectrumSource(
                    run_dir=run_dir,
                    target_id=str(row.get("target_id")),
                    injected=True,
                    injection=injection,
                    recovery=row,
                )
            )
    return sources


def _load_clean_sources(run_dirs: list[Path], injected_ids: set[str], *, limit_per_run: int = 12) -> list[SpectrumSource]:
    sources: list[SpectrumSource] = []
    for run_dir in run_dirs:
        summary_path = run_dir / "spectra" / "target_summary.parquet"
        if not summary_path.exists():
            continue
        summary = pd.read_parquet(summary_path)
        if "target_id" not in summary:
            continue
        if "fatal_flag_fraction" in summary:
            summary = summary.sort_values(["fatal_flag_fraction", "measurement_count"], ascending=[True, False])
        elif "measurement_count" in summary:
            summary = summary.sort_values("measurement_count", ascending=False)
        selected = summary[~summary["target_id"].astype(str).isin(injected_ids)].head(limit_per_run)
        for target_id in selected["target_id"].astype(str).tolist():
            sources.append(SpectrumSource(run_dir=run_dir, target_id=target_id, injected=False, injection=None, recovery=None))
    return sources


def _baseline_run_for_injected(run_dir: Path) -> Path:
    name = run_dir.name
    if name.endswith("_injected"):
        return run_dir.with_name(name[: -len("_injected")] + "_baseline")
    return run_dir


def _make_known_example_decoys(known_examples: list[dict[str, Any]], run_root: Path) -> list[SpectrumSource]:
    decoys: list[SpectrumSource] = []
    for example in known_examples:
        injected_run = run_root / str(example["run"])
        baseline_run = _baseline_run_for_injected(injected_run)
        if not baseline_run.exists():
            continue
        decoys.append(
            SpectrumSource(
                run_dir=baseline_run,
                target_id=str(example["target_id"]),
                injected=False,
                injection=None,
                recovery=None,
                decoy_of_signal_id=str(example["signal_id"]),
            )
        )
    return decoys


def build_packets(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    run_dirs = _campaign_run_dirs(args.run_root, args.campaign)
    if not run_dirs:
        raise SystemExit(f"No injected run directories found for campaign {args.campaign!r}")
    injected_sources = _load_injected_sources(run_dirs)
    injected_ids = {source.target_id for source in injected_sources}
    clean_sources = _load_clean_sources(run_dirs, injected_ids)
    random.shuffle(injected_sources)
    random.shuffle(clean_sources)
    spectra_cache: dict[Path, pd.DataFrame] = {}

    injected_library: list[dict[str, Any]] = []
    for source in injected_sources:
        packet = serialize_spectrum(source, spectra_cache, reveal_truth=True, max_points=args.max_points)
        if packet:
            injected_library.append(packet)
        if len(injected_library) >= args.injected_library_size:
            break
    if len(injected_library) < args.injected_library_size:
        raise SystemExit(f"Only built {len(injected_library)} injected spectra")

    clean_library: list[dict[str, Any]] = []
    for source in clean_sources:
        packet = serialize_spectrum(source, spectra_cache, reveal_truth=True, max_points=args.max_points)
        if packet:
            clean_library.append(packet)
        if len(clean_library) >= args.injected_library_size:
            break

    _write_jsonl(DATA_DIR / "serialized_injected_spectra.jsonl", injected_library)
    _write_jsonl(DATA_DIR / "serialized_clean_spectra.jsonl", clean_library)
    _write_json(PROMPT_DIR / "system_prompt.json", {"system_prompt": SYSTEM_PROMPT})
    _write_json(PROMPT_DIR / "user_prompts.json", USER_PROMPTS)
    _write_json(PROMPT_DIR / "return_schema.json", RETURN_SCHEMA)

    known_examples = injected_library[: min(args.known_examples, len(injected_library))]
    known_negative = clean_library[: min(args.known_negative_examples, len(clean_library))]
    decoy_sources = _make_known_example_decoys(known_examples, args.run_root)
    decoy_library: list[dict[str, Any]] = []
    for source in decoy_sources:
        packet = serialize_spectrum(source, spectra_cache, reveal_truth=True, max_points=args.max_points)
        if packet:
            decoy_library.append(packet)
    _write_jsonl(DATA_DIR / "paired_uninjected_decoy_spectra.jsonl", decoy_library)

    injected_pool = injected_library[len(known_examples) :]
    clean_pool = clean_library[len(known_negative) :]

    for size in CHALLENGE_SIZES:
        injected_count = size // 2
        decoy_count = min(len(decoy_library), max(1, size // 10))
        clean_count = size - injected_count - decoy_count
        challenge_with_truth = injected_pool[:injected_count] + decoy_library[:decoy_count] + clean_pool[:clean_count]
        random.shuffle(challenge_with_truth)
        truth = []
        challenge = []
        for packet in challenge_with_truth:
            truth_obj = packet.get("truth", {})
            truth.append(
                {
                    "signal_id": packet["signal_id"],
                    "target_id": packet["target_id"],
                    "is_injected": bool(truth_obj.get("is_injected")),
                    "decoy_of_signal_id": truth_obj.get("decoy_of_signal_id"),
                    "is_paired_decoy_uninjected_same_target": bool(truth_obj.get("decoy_of_signal_id")),
                    "line_family": truth_obj.get("line_family"),
                    "injected_line_nm": truth_obj.get("injected_line_nm"),
                    "find_me_snr": truth_obj.get("find_me_snr"),
                    "gpu_narrowband_match": packet.get("known_detector_output", {}).get("gpu_narrowband_match"),
                    "gpu_narrowband_quality_match": packet.get("known_detector_output", {}).get("gpu_narrowband_quality_match"),
                }
            )
            scrubbed = dict(packet)
            scrubbed.pop("truth", None)
            challenge.append(scrubbed)

        payload = {
            "packet_id": f"{args.campaign}_challenge_{size}",
            "campaign": args.campaign,
            "instructions": {
                "task": "Analyze the unlabeled challenge spectra and return strict JSON.",
                "known_examples_are_labeled": True,
                "challenge_truth_is_hidden": True,
                "benchmark_note": (
                    "Some challenge spectra may be controls or look similar to known examples. "
                    "Do not classify by target_id alone; classify from the spectral evidence."
                ),
                "return_schema": RETURN_SCHEMA,
            },
            "known_signal_examples": known_examples,
            "known_non_signal_examples": known_negative,
            "challenge_spectra": challenge,
        }
        _write_json(DATA_DIR / f"challenge_{size}.json", payload)
        _write_json(DATA_DIR / f"compact_challenge_{size}.json", _compact_packet(payload))
        _write_json(DATA_DIR / f"truth_{size}.json", {"packet_id": payload["packet_id"], "truth": truth})

    print(f"Built {len(injected_library)} injected spectra and {len(clean_library)} clean spectra")
    print(f"Wrote challenges: {', '.join(str(DATA_DIR / f'challenge_{s}.json') for s in CHALLENGE_SIZES)}")


def make_messages(packet: dict[str, Any], prompt_variant: str) -> list[dict[str, str]]:
    user_prompt = USER_PROMPTS[prompt_variant]
    challenge_ids = [str(row.get("signal_id") or row.get("sid")) for row in packet.get("challenge_spectra", [])]
    body = {
        "user_prompt": user_prompt,
        "hard_requirements": [
            "Return one top-level JSON object.",
            "The top-level object must contain an analyses array.",
            "The analyses array must contain exactly one object for every signal_id in challenge_signal_ids.",
            "Copy each signal_id exactly from challenge_signal_ids.",
            "Do not return known examples as analyses.",
            "Do not reveal or infer hidden truth labels; judge only from the challenge spectrum evidence.",
        ],
        "challenge_signal_ids": challenge_ids,
        "return_schema": RETURN_SCHEMA,
        "data_packet": packet,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(_json_safe(body), separators=(",", ":"))},
    ]


def call_openai_compatible(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=data,
        method="POST",
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def score_response(response: dict[str, Any], truth_path: Path) -> dict[str, Any]:
    truth_rows = _read_json(truth_path)["truth"]
    truth_by_id = {row["signal_id"]: row for row in truth_rows}
    analyses = response.get("analyses") or []
    if not isinstance(analyses, list):
        analyses = []
    tp = fp = tn = fn = 0
    details = []
    for analysis in analyses:
        signal_id = str(analysis.get("signal_id"))
        truth = truth_by_id.get(signal_id)
        if truth is None:
            continue
        predicted = bool(analysis.get("has_anomalous_excess"))
        actual = bool(truth.get("is_injected"))
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1
        details.append(
            {
                "signal_id": signal_id,
                "actual_injected": actual,
                "predicted_anomalous": predicted,
                "confidence": analysis.get("confidence"),
                "priority": analysis.get("priority"),
                "best_line_nm": analysis.get("best_line_nm"),
                "truth_line_nm": truth.get("injected_line_nm"),
                "truth_line_family": truth.get("line_family"),
            }
        )
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": precision,
        "recall": recall,
        "details": details,
    }


def run_packet(args: argparse.Namespace, *, size: int, prompt_variant: str) -> Path:
    packet_path = _packet_path(size, args.packet_format)
    truth_path = DATA_DIR / f"truth_{size}.json"
    if not packet_path.exists():
        raise SystemExit(f"Missing {packet_path}; run build first")
    packet = _read_json(packet_path)
    messages = make_messages(packet, prompt_variant)
    started = time.time()
    raw = call_openai_compatible(
        base_url=args.base_url,
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_sec=args.timeout_sec,
        api_key=_api_key_from_args(args),
        reasoning_effort=args.reasoning_effort,
    )
    elapsed = time.time() - started
    content = raw["choices"][0]["message"]["content"]
    out_path = RESPONSE_DIR / f"{packet['packet_id']}_{prompt_variant}_{args.model.replace('/', '_').replace(':', '_')}.json"
    try:
        parsed = extract_json_object(content)
    except Exception as exc:
        failed = {
            "request": {
                "model": args.model,
                "base_url": args.base_url,
                "challenge_size": size,
                "packet_format": args.packet_format,
                "prompt_variant": prompt_variant,
                "elapsed_sec": elapsed,
                "usage": raw.get("usage"),
            },
            "parse_error": str(exc),
            "content": content,
            "raw_response": raw,
        }
        failed_path = out_path.with_name(out_path.stem + "_parse_failed.json")
        _write_json(failed_path, failed)
        print(f"{prompt_variant} size={size} elapsed={elapsed:.1f}s parse_failed={exc}")
        print(f"wrote {failed_path}")
        raise
    parsed.setdefault("packet_id", packet.get("packet_id"))
    parsed.setdefault("model", args.model)
    parsed.setdefault("prompt_variant", prompt_variant)
    scored = score_response(parsed, truth_path)
    out = {
        "request": {
            "model": args.model,
            "base_url": args.base_url,
            "challenge_size": size,
            "packet_format": args.packet_format,
            "prompt_variant": prompt_variant,
            "elapsed_sec": elapsed,
            "usage": raw.get("usage"),
        },
        "response": parsed,
        "score": scored,
        "raw_response": raw,
    }
    _write_json(out_path, out)
    print(
        f"{prompt_variant} size={size} elapsed={elapsed:.1f}s "
        f"precision={scored['precision']} recall={scored['recall']} counts={scored['counts']}"
    )
    print(f"wrote {out_path}")
    return out_path


def smoke(args: argparse.Namespace) -> None:
    run_packet(args, size=args.challenge_size, prompt_variant=args.prompt_variant)


def run_examples(args: argparse.Namespace) -> None:
    for size in args.sizes:
        for prompt_variant in args.prompt_variants:
            run_packet(args, size=size, prompt_variant=prompt_variant)


def _chat_payload_for_packet(packet: dict[str, Any], args: argparse.Namespace, prompt_variant: str) -> dict[str, Any]:
    return {
        "model": args.model,
        "messages": make_messages(packet, prompt_variant),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False,
        "response_format": {"type": "json_object"},
    }


def weigh_packets(args: argparse.Namespace) -> None:
    rows = []
    for size in args.sizes:
        for packet_format in args.packet_formats:
            packet_path = _packet_path(size, packet_format)
            if not packet_path.exists():
                continue
            packet = _read_json(packet_path)
            payload = _chat_payload_for_packet(packet, args, args.prompt_variant)
            text = json.dumps(payload, separators=(",", ":"))
            rows.append(
                {
                    "size": size,
                    "format": packet_format,
                    "packet_bytes": packet_path.stat().st_size,
                    "chat_payload_chars": len(text),
                    "est_tokens_4char": round(len(text) / 4),
                    "est_tokens_3char": round(len(text) / 3),
                    "est_tokens_2p5char": round(len(text) / 2.5),
                    "path": str(packet_path),
                }
            )
    _write_json(DATA_DIR / "packet_weight_summary.json", {"rows": rows})
    for row in rows:
        print(
            f"{row['size']:>3} {row['format']:<10} packet={row['packet_bytes']/1024:8.1f} KiB "
            f"chat_chars={row['chat_payload_chars']:>8} "
            f"tok~4c={row['est_tokens_4char']:>7} tok~3c={row['est_tokens_3char']:>7} "
            f"path={row['path']}"
        )


def parse_args() -> argparse.Namespace:
    env_values = _read_simple_env(Path(".env"))
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="Build serialized spectra and challenge packets.")
    build.add_argument("--campaign", default=DEFAULT_CAMPAIGN)
    build.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--injected-library-size", type=int, default=100)
    build.add_argument("--known-examples", type=int, default=8)
    build.add_argument("--known-negative-examples", type=int, default=4)
    build.add_argument("--max-points", type=int, default=72)
    build.set_defaults(func=build_packets)

    def add_query_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--base-url", default=_env_default(env_values, "GEMINI_BASE_URL", DEFAULT_BASE_URL))
        p.add_argument("--model", default=_env_default(env_values, "GEMINI_MODEL", DEFAULT_MODEL))
        p.add_argument("--env-file", type=Path, default=Path(".env"))
        p.add_argument(
            "--api-key-env",
            default="GEMINI_API_KEY" if _env_default(env_values, "GEMINI_API_KEY", "") else "",
        )
        p.add_argument("--temperature", type=float, default=0.1)
        p.add_argument("--max-tokens", type=int, default=4096)
        p.add_argument("--timeout-sec", type=int, default=240)
        p.add_argument("--packet-format", default="verbose", choices=PACKET_FORMATS)
        p.add_argument("--reasoning-effort", default="low", choices=["none", "low", "medium", "high"])

    smoke_p = sub.add_parser("smoke", help="Run one challenge packet against the local LLM.")
    add_query_args(smoke_p)
    smoke_p.add_argument("--challenge-size", type=int, default=5, choices=CHALLENGE_SIZES)
    smoke_p.add_argument("--prompt-variant", default="positive", choices=sorted(USER_PROMPTS))
    smoke_p.set_defaults(func=smoke)

    examples = sub.add_parser("run-examples", help="Run selected packet sizes and prompt variants.")
    add_query_args(examples)
    examples.add_argument("--sizes", nargs="+", type=int, default=[5, 10], choices=CHALLENGE_SIZES)
    examples.add_argument("--prompt-variants", nargs="+", default=["positive", "negative"], choices=sorted(USER_PROMPTS))
    examples.set_defaults(func=run_examples)

    weigh = sub.add_parser("weigh", help="Estimate prompt size for verbose and compact packets.")
    add_query_args(weigh)
    weigh.add_argument("--sizes", nargs="+", type=int, default=list(CHALLENGE_SIZES), choices=CHALLENGE_SIZES)
    weigh.add_argument("--packet-formats", nargs="+", default=list(PACKET_FORMATS), choices=PACKET_FORMATS)
    weigh.add_argument("--prompt-variant", default="positive", choices=sorted(USER_PROMPTS))
    weigh.set_defaults(func=weigh_packets)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
