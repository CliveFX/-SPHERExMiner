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
        summary = {"measurement_rows": 0, "target_count": 0, "spectra_dir": str(spectra_dir)}
        (spectra_dir / "assembly_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    all_measurements = pd.concat(tables, ignore_index=True)
    all_measurements = all_measurements.sort_values(["target_id", "cwave_um", "obs_mid_time", "detector"])
    all_measurements.to_parquet(spectra_dir / "all_measurements.parquet", index=False)

    spectra_cols = [
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
        "aperture_flux_uJy",
        "aperture_flux_unc_uJy",
        "psf_flux_uJy",
        "psf_flux_unc_uJy",
        "psf_fit_status",
        "psf_chi2",
        "psf_dof",
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
    target_summary = target_summary.sort_values(["n_measurements", "max_snr_uJy"], ascending=[False, False])
    target_summary.to_parquet(spectra_dir / "target_summary.parquet", index=False)

    summary = {
        "measurement_rows": int(len(all_measurements)),
        "target_count": int(target_summary["target_id"].nunique()),
        "shard_count": int(len(tables)),
        "spectra_dir": str(spectra_dir),
        "all_measurements_path": str(spectra_dir / "all_measurements.parquet"),
        "target_spectra_path": str(spectra_dir / "target_spectra.parquet"),
        "target_summary_path": str(spectra_dir / "target_summary.parquet"),
    }
    (spectra_dir / "assembly_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
