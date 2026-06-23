from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits


def write_smoke_artifacts(run_dir: Path, trials: list[dict[str, object]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    measured = [trial for trial in trials if trial.get("status") == "measured"]
    rows = [_measurement_row(trial) for trial in measured]
    if rows:
        pd.DataFrame(rows).to_parquet(run_dir / "measurements.parquet", index=False)
    else:
        pd.DataFrame().to_parquet(run_dir / "measurements.parquet", index=False)

    qa = {
        "trial_count": len(trials),
        "measured_count": len(measured),
        "statuses": _status_counts(trials),
        "best_trial_index": _best_trial_index(trials),
        "measurement_rows": len(rows),
        "measurement_path": str(run_dir / "measurements.parquet"),
        "trial_json": str(run_dir / "simp_field_trials.json"),
        "plots_dir": str(run_dir / "plots"),
    }
    (run_dir / "qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    _write_plots(run_dir, trials)


def _measurement_row(trial: dict[str, object]) -> dict[str, object]:
    aperture = dict(trial.get("aperture") or {})
    candidate = dict(trial.get("candidate") or {})
    row = {
        "measurement_id": f"{trial.get('target_id')}_{candidate.get('obs_id')}_D{trial.get('detector')}",
        "target_id": trial.get("target_id"),
        "target_type": "manual_coordinate",
        "source_id": None,
        "image_id": Path(str(trial.get("local_path"))).stem,
        "release": "qr2",
        "processing_version": _processing_version(str(trial.get("local_path"))),
        "planning_period": _planning_period(str(trial.get("local_path"))),
        "detector": trial.get("detector"),
        "observation_id": candidate.get("obs_id"),
        "obs_mid_time": trial.get("obs_mid_isot"),
        "filter_profile": "manual_targets",
        "ra_reference_deg": 24.2412498,
        "dec_reference_deg": 9.5630705,
        "reference_epoch_yr": 2016.0,
        "pmra_masyr": 1238.244,
        "pmdec_masyr": -16.156,
        "ra_epoch_deg": trial.get("ra_epoch_deg"),
        "dec_epoch_deg": trial.get("dec_epoch_deg"),
        "x_pix": trial.get("x_pix"),
        "y_pix": trial.get("y_pix"),
        "edge_distance_pix": trial.get("edge_distance_pix"),
        "cwave_um": trial.get("cwave_um"),
        "cband_um": trial.get("cband_um"),
        "wavelength_source": trial.get("wavelength_source"),
        "wavelength_calibration_file": trial.get("wavelength_calibration_file"),
        "wavelength_calibration_collection": trial.get("wavelength_calibration_collection"),
        "wavelength_detector": trial.get("wavelength_detector"),
        "sapm_file_path": trial.get("sapm_file_path"),
        "image_value_raw": aperture.get("image_value_raw"),
        "image_unit": trial.get("image_unit"),
        "aperture_radius_pix": trial.get("aperture_radius_pix"),
        "annulus_inner_pix": trial.get("annulus_inner_pix"),
        "annulus_outer_pix": trial.get("annulus_outer_pix"),
        "photometry_backend": "cpu_numpy",
        "input_file_path": trial.get("local_path"),
        "pipeline_version": "0.1.0",
    }
    row.update(aperture)
    row.update(dict(trial.get("calibrated_aperture") or {}))
    return row


def _processing_version(path: str) -> str | None:
    parts = Path(path).parts
    for part in parts:
        if part.startswith("l2b-"):
            return part
    return None


def _planning_period(path: str) -> str | None:
    parts = Path(path).parts
    for idx, part in enumerate(parts):
        if part == "level2" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _status_counts(trials: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trial in trials:
        status = str(trial.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _best_trial_index(trials: list[dict[str, object]]) -> int | None:
    best_idx = None
    best_edge = -np.inf
    for idx, trial in enumerate(trials):
        if trial.get("status") != "measured":
            continue
        aperture = dict(trial.get("aperture") or {})
        if aperture.get("fatal_flag_present"):
            continue
        edge = float(trial.get("edge_distance_pix") or -np.inf)
        if edge > best_edge:
            best_idx = idx
            best_edge = edge
    if best_idx is not None:
        return best_idx
    for idx, trial in enumerate(trials):
        if trial.get("status") != "measured":
            continue
        edge = float(trial.get("edge_distance_pix") or -np.inf)
        if edge > best_edge:
            best_idx = idx
            best_edge = edge
    return best_idx


def _write_plots(run_dir: Path, trials: list[dict[str, object]]) -> None:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_field_summary(plots_dir / "detector_targets.png", trials)
    best_idx = _best_trial_index(trials)
    if best_idx is not None:
        _plot_postage(plots_dir / "simp_postage_aperture.png", trials[best_idx])


def _plot_field_summary(path: Path, trials: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, 2040)
    ax.set_ylim(0, 2040)
    ax.set_aspect("equal")
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    ax.set_title("SIMP projected positions in attempted SPHEREx fields")
    for trial in trials:
        if "x_pix" not in trial or "y_pix" not in trial:
            continue
        status = trial.get("status")
        color = "tab:green" if status == "measured" else "tab:red"
        ax.scatter(float(trial["x_pix"]), float(trial["y_pix"]), c=color, s=35)
        ax.text(float(trial["x_pix"]) + 12, float(trial["y_pix"]) + 12, f"D{trial.get('detector')}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_postage(path: Path, trial: dict[str, object], half_size: int = 24) -> None:
    local_path = Path(str(trial["local_path"]))
    x_pix = float(trial["x_pix"])
    y_pix = float(trial["y_pix"])
    with fits.open(local_path, memmap=True) as hdul:
        image = hdul["IMAGE"].data
        y0 = max(0, int(round(y_pix)) - half_size)
        y1 = min(image.shape[0], int(round(y_pix)) + half_size + 1)
        x0 = max(0, int(round(x_pix)) - half_size)
        x1 = min(image.shape[1], int(round(x_pix)) + half_size + 1)
        cutout = image[y0:y1, x0:x1]

    finite = cutout[np.isfinite(cutout)]
    vmin, vmax = np.percentile(finite, [5, 99]) if len(finite) else (None, None)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cutout, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax.scatter([x_pix - x0], [y_pix - y0], s=120, facecolors="none", edgecolors="tab:red", linewidths=1.5)
    ax.set_title("SIMP aperture postage")
    ax.set_xlabel("postage x")
    ax.set_ylabel("postage y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
