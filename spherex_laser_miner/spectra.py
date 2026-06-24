from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def assemble_spectra_from_jobs(run_dir: Path, jobs: list[dict[str, object]]) -> dict[str, object]:
    tables = []
    for job in jobs:
        path = Path(str(job["measurement_path"]))
        if path.exists() and path.stat().st_size > 0:
            df = pd.read_parquet(path)
            df["shard_path"] = str(path)
            tables.append(df)

    spectra_dir = run_dir / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    if not tables:
        empty = pd.DataFrame()
        empty.to_parquet(spectra_dir / "all_measurements.parquet", index=False)
        empty.to_parquet(spectra_dir / "target_spectra.parquet", index=False)
        empty.to_parquet(spectra_dir / "target_summary.parquet", index=False)
        manifest_path = _find_injection_manifest(run_dir)
        summary = {
            "run_name": run_dir.name,
            "run_kind": _run_kind_from_name(run_dir.name),
            "run_injection_applied": bool(manifest_path is not None or _run_kind_from_name(run_dir.name) == "injected"),
            "point_injection_applied_count": 0,
            "injection_manifest_path": str(manifest_path) if manifest_path is not None else None,
            "measurement_rows": 0,
            "target_count": 0,
            "spectra_dir": str(spectra_dir),
        }
        (spectra_dir / "assembly_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    all_measurements = pd.concat(tables, ignore_index=True)
    all_measurements = all_measurements.sort_values(["target_id", "cwave_um", "obs_mid_time", "detector"])
    provenance = _add_injection_provenance(run_dir, all_measurements)
    all_measurements.to_parquet(spectra_dir / "all_measurements.parquet", index=False)

    spectra_cols = [
        "run_name",
        "run_kind",
        "run_injection_applied",
        "point_injection_applied",
        "injection_applied",
        "injection_manifest_path",
        "target_id",
        "target_type",
        "source_id",
        "object_name",
        "ra_reference_deg",
        "dec_reference_deg",
        "reference_epoch_yr",
        "pmra_masyr",
        "pmdec_masyr",
        "parallax_mas",
        "ra_epoch_deg",
        "dec_epoch_deg",
        "coordinate_propagation",
        "phot_g_mean_mag",
        "phot_bp_mean_mag",
        "phot_rp_mean_mag",
        "bp_rp",
        "ruwe",
        "duplicated_source",
        "astrometric_params_solved",
        "cwave_um",
        "cband_um",
        "wavelength_source",
        "wavelength_calibration_file",
        "wavelength_calibration_collection",
        "wavelength_detector",
        "aperture_flux_uJy",
        "aperture_flux_unc_uJy",
        "photometry_backend",
        "psf_photometry_backend",
        "psf_kernel_build_mode",
        "psf_grid_half_range_pix",
        "psf_grid_step_pix",
        "psf_grid_metric",
        "psf_flux_uJy",
        "psf_flux_unc_uJy",
        "psf_fit_status",
        "psf_chi2",
        "psf_dof",
        "psf_n_valid",
        "psf_model_id",
        "psf_sector",
        "psf_background_uJy_per_pix",
        "centroid_dx_pix",
        "centroid_dy_pix",
        "detector",
        "observation_id",
        "obs_mid_time",
        "image_id",
        "x_pix",
        "y_pix",
        "edge_distance_pix",
        "fatal_flag_present",
        "flags_summary",
        "zodi_model_at_target",
        "input_file_path",
        "original_input_file_path",
        "path_override_applied",
        "shard_path",
    ]
    available_cols = [col for col in spectra_cols if col in all_measurements.columns]
    target_spectra = all_measurements[available_cols].copy()
    target_spectra.to_parquet(spectra_dir / "target_spectra.parquet", index=False)

    grouped = all_measurements.groupby(["target_id", "target_type"], dropna=False)
    target_summary = grouped.agg(
        n_measurements=("target_id", "size"),
        wavelength_min_um=("cwave_um", "min"),
        wavelength_max_um=("cwave_um", "max"),
        median_flux_uJy=("aperture_flux_uJy", "median"),
        median_psf_flux_uJy=("psf_flux_uJy", "median"),
        max_snr_uJy=("aperture_flux_uJy", lambda s: float("nan")),
    ).reset_index()
    # Compute max SNR separately because aggregation needs two columns.
    snr = all_measurements["aperture_flux_uJy"] / all_measurements["aperture_flux_unc_uJy"]
    snr_df = all_measurements[["target_id"]].copy()
    snr_df["snr"] = snr
    snr_max = snr_df.groupby("target_id", dropna=False)["snr"].max().rename("max_snr_uJy").reset_index()
    target_summary = target_summary.drop(columns=["max_snr_uJy"]).merge(snr_max, on="target_id", how="left")
    target_summary["run_name"] = provenance["run_name"]
    target_summary["run_kind"] = provenance["run_kind"]
    target_summary["run_injection_applied"] = provenance["run_injection_applied"]
    target_summary["injection_manifest_path"] = provenance["injection_manifest_path"]
    injection_counts = (
        all_measurements.assign(point_injection_applied=all_measurements["point_injection_applied"].fillna(False).astype(bool))
        .groupby("target_id", dropna=False)
        .agg(
            point_injection_applied_count=("point_injection_applied", "sum"),
            point_measurement_count=("point_injection_applied", "size"),
        )
        .reset_index()
    )
    injection_counts["point_injection_applied_fraction"] = (
        injection_counts["point_injection_applied_count"] / injection_counts["point_measurement_count"]
    )
    target_summary = target_summary.merge(
        injection_counts.drop(columns=["point_measurement_count"]),
        on="target_id",
        how="left",
    )
    target_summary = target_summary.sort_values(["n_measurements", "max_snr_uJy"], ascending=[False, False])
    target_summary.to_parquet(spectra_dir / "target_summary.parquet", index=False)

    summary = {
        "run_name": provenance["run_name"],
        "run_kind": provenance["run_kind"],
        "run_injection_applied": provenance["run_injection_applied"],
        "point_injection_applied_count": provenance["point_injection_applied_count"],
        "injection_manifest_path": provenance["injection_manifest_path"],
        "measurement_rows": int(len(all_measurements)),
        "target_count": int(target_summary["target_id"].nunique()),
        "shard_count": int(len(tables)),
        "wavelength_sources": _unique_strings(all_measurements, "wavelength_source"),
        "wavelength_calibration_collections": _unique_strings(all_measurements, "wavelength_calibration_collection"),
        "wavelength_detectors": _unique_strings(all_measurements, "wavelength_detector"),
        "spectra_dir": str(spectra_dir),
        "all_measurements_path": str(spectra_dir / "all_measurements.parquet"),
        "target_spectra_path": str(spectra_dir / "target_spectra.parquet"),
        "target_summary_path": str(spectra_dir / "target_summary.parquet"),
    }
    (spectra_dir / "assembly_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _add_injection_provenance(run_dir: Path, df: pd.DataFrame) -> dict[str, object]:
    run_name = run_dir.name
    run_kind = _run_kind_from_name(run_name)
    manifest_path = _find_injection_manifest(run_dir)

    point_injected = pd.Series(False, index=df.index)
    if "path_override_applied" in df.columns:
        point_injected |= df["path_override_applied"].fillna(False).astype(bool)
    if {"input_file_path", "original_input_file_path"} <= set(df.columns):
        input_path = df["input_file_path"].fillna("").astype(str)
        original_path = df["original_input_file_path"].fillna("").astype(str)
        point_injected |= original_path.ne("") & input_path.ne(original_path)

    run_injected = bool(run_kind == "injected" or point_injected.any() or manifest_path is not None)
    df["run_name"] = run_name
    df["run_kind"] = run_kind
    df["run_injection_applied"] = run_injected
    df["point_injection_applied"] = point_injected.astype(bool)
    df["injection_applied"] = df["point_injection_applied"]
    df["injection_manifest_path"] = str(manifest_path) if manifest_path is not None else None

    return {
        "run_name": run_name,
        "run_kind": run_kind,
        "run_injection_applied": run_injected,
        "point_injection_applied_count": int(point_injected.sum()),
        "injection_manifest_path": str(manifest_path) if manifest_path is not None else None,
    }


def _run_kind_from_name(run_name: str) -> str:
    if run_name.endswith("_injected") or "_injected_" in run_name:
        return "injected"
    if run_name.endswith("_baseline") or "_baseline_" in run_name:
        return "baseline"
    return "unknown"


def _find_injection_manifest(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "injection_manifest.json",
        run_dir / "injections" / "injection_manifest.json",
    ]
    for summary_name in ("run_summary.json", "benchmark_summary.json"):
        summary_path = run_dir / summary_name
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        overrides_path = summary.get("path_overrides_path")
        if overrides_path:
            candidates.append(Path(str(overrides_path)).parent / "injection_manifest.json")
    for path in candidates:
        if path.exists():
            return path
    return None


def _unique_strings(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    return sorted({str(value) for value in df[column].dropna().unique()})
