from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "log_wavelength_um",
    "log_cband_um",
    "aperture_flux_norm",
    "aperture_unc_norm",
    "psf_flux_norm",
    "psf_unc_norm",
    "aperture_snr_clip",
    "psf_snr_clip",
    "fatal_flag",
    "detector_norm",
    "x_norm",
    "y_norm",
    "edge_norm",
]


@dataclass(frozen=True)
class RaggedExample:
    dataset_name: str
    run_name: str
    run_kind: str
    split_id: str
    target_id: str
    source_id: str | None
    label: float
    features: np.ndarray


def load_ragged_examples(
    dataset_dir: Path,
    *,
    point_table: str,
    target_table: str,
    split_id: str,
    quality_categories: set[str] | None = None,
    label_column: str | None = None,
    max_targets: int | None = None,
    min_points: int = 8,
) -> list[RaggedExample]:
    points = pd.read_parquet(dataset_dir / point_table)
    targets = pd.read_parquet(dataset_dir / target_table)
    if split_id != "all":
        targets = targets[targets["split_id"].astype(str).eq(split_id)].copy()
    if quality_categories and "spectrum_quality_category" in targets:
        targets = targets[targets["spectrum_quality_category"].astype(str).isin(quality_categories)].copy()
    if targets.empty:
        return []
    if max_targets is not None and len(targets) > max_targets:
        targets = targets.sort_values(
            ["spectrum_quality_score", "n_usable_measurements"],
            ascending=[False, False],
            na_position="last",
            kind="mergesort",
        ).head(max_targets)
    targets["_example_key"] = targets["run_name"].astype(str) + "::" + targets["target_id"].astype(str)
    keep = set(targets["_example_key"])
    points["_example_key"] = points["run_name"].astype(str) + "::" + points["target_id"].astype(str)
    points = points[points["_example_key"].isin(keep)].copy()
    target_meta = targets.set_index("_example_key").to_dict(orient="index")

    examples: list[RaggedExample] = []
    for example_key, rows in points.groupby("_example_key", dropna=False, sort=False):
        meta = target_meta.get(str(example_key))
        if not meta:
            continue
        target_id = str(meta.get("target_id"))
        features = make_point_features(rows)
        if len(features) < min_points:
            continue
        label = float(bool(meta.get(label_column))) if label_column else 0.0
        examples.append(
            RaggedExample(
                dataset_name=str(meta.get("dataset_name") or ""),
                run_name=str(meta.get("run_name") or rows["run_name"].iloc[0]),
                run_kind=str(meta.get("run_kind") or rows.get("run_kind", pd.Series(["unknown"])).iloc[0]),
                split_id=str(meta.get("split_id") or split_id),
                target_id=target_id,
                source_id=None if pd.isna(meta.get("source_id")) else str(meta.get("source_id")),
                label=label,
                features=features,
            )
        )
    return examples


def make_point_features(rows: pd.DataFrame) -> np.ndarray:
    work = rows.sort_values("cwave_um", na_position="last").copy()
    n_rows = len(work)
    wave = _num_col(work, "cwave_um", n_rows, default=1.0)
    cband = _num_col(work, "cband_um", n_rows, default=0.01)
    aperture = _num_col(work, "aperture_flux_uJy", n_rows)
    aperture_unc = _num_col(work, "aperture_flux_unc_uJy", n_rows, default=1.0)
    psf = _num_col(work, "psf_flux_uJy", n_rows)
    psf_unc = _num_col(work, "psf_flux_unc_uJy", n_rows, default=1.0)

    aperture_scale = _robust_scale(aperture)
    psf_scale = _robust_scale(psf)
    aperture_norm = _robust_norm(aperture, aperture_scale)
    psf_norm = _robust_norm(psf, psf_scale)
    aperture_unc_norm = np.clip(aperture_unc / aperture_scale, 0.0, 10.0) if aperture_scale > 0 else np.zeros_like(aperture)
    psf_unc_norm = np.clip(psf_unc / psf_scale, 0.0, 10.0) if psf_scale > 0 else np.zeros_like(psf)

    features = np.column_stack(
        [
            np.nan_to_num(np.log10(np.clip(wave, 1e-6, None)), nan=0.0),
            np.nan_to_num(np.log10(np.clip(cband, 1e-6, None)), nan=0.0),
            aperture_norm,
            aperture_unc_norm,
            psf_norm,
            psf_unc_norm,
            _snr_feature(aperture, aperture_unc),
            _snr_feature(psf, psf_unc),
            _bool_col(work, "fatal_flag_present", n_rows),
            np.nan_to_num(_num_col(work, "detector", n_rows) / 6.0, nan=0.0),
            np.nan_to_num(_num_col(work, "x_pix", n_rows) / 2048.0, nan=0.0),
            np.nan_to_num(_num_col(work, "y_pix", n_rows) / 2048.0, nan=0.0),
            np.nan_to_num(_num_col(work, "edge_distance_pix", n_rows) / 2048.0, nan=0.0),
        ]
    ).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def random_view(features: np.ndarray, *, max_points: int, keep_fraction: float, rng: np.random.Generator) -> np.ndarray:
    n = len(features)
    if n <= 1:
        return features
    keep = max(2, min(n, int(round(n * keep_fraction))))
    keep = min(keep, max_points)
    idx = rng.choice(n, size=keep, replace=False)
    idx.sort()
    view = features[idx].copy()
    noise = rng.normal(0.0, 0.01, size=view.shape).astype(np.float32)
    view[:, 2:6] += noise[:, 2:6]
    return view


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def now_status(
    *,
    run_name: str,
    model_type: str,
    status: str,
    dataset_name: str,
    model_version: str,
    started: float,
    **extra: object,
) -> dict[str, object]:
    return {
        "run_name": run_name,
        "model_type": model_type,
        "status": status,
        "dataset_name": dataset_name,
        "model_version": model_version,
        "elapsed_sec": time.perf_counter() - started,
        "updated_at_unix": time.time(),
        **extra,
    }


def _num_col(df: pd.DataFrame, column: str, n_rows: int, default: float = 0.0) -> np.ndarray:
    if column not in df:
        return np.full(n_rows, default, dtype=np.float32)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).to_numpy(dtype=np.float32)


def _bool_col(df: pd.DataFrame, column: str, n_rows: int) -> np.ndarray:
    if column not in df:
        return np.zeros(n_rows, dtype=np.float32)
    return df[column].fillna(False).astype(bool).to_numpy(dtype=np.float32)


def _robust_scale(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return 1.0
    med = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - med)))
    scale = 1.4826 * mad if math.isfinite(mad) and mad > 0 else float(np.nanstd(finite))
    return scale if math.isfinite(scale) and scale > 0 else 1.0


def _robust_norm(values: np.ndarray, scale: float) -> np.ndarray:
    finite = values[np.isfinite(values)]
    med = float(np.nanmedian(finite)) if finite.size else 0.0
    out = (values - med) / max(scale, 1e-6)
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -12.0, 12.0)


def _snr_feature(flux: np.ndarray, unc: np.ndarray) -> np.ndarray:
    snr = np.divide(flux, unc, out=np.zeros_like(flux), where=np.isfinite(unc) & (unc > 0))
    return np.clip(np.nan_to_num(snr, nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0) / 50.0
