from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CACHE_ROOT = Path("/mnt/niroseti/spherex_cache")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize spectra quality by Gaia G magnitude bin.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-usable", type=int, default=40)
    parser.add_argument("--max-fatal-frac", type=float, default=0.6)
    parser.add_argument("--min-median-snr", type=float, default=2.0)
    args = parser.parse_args()

    spectra_path = args.run_dir / "spectra" / "target_spectra.parquet"
    if not spectra_path.exists():
        raise SystemExit(f"Missing spectra parquet: {spectra_path}")
    df = pd.read_parquet(spectra_path)
    required = {"target_id", "phot_g_mean_mag", "aperture_flux_uJy", "aperture_flux_unc_uJy", "fatal_flag_present"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing columns: {', '.join(missing)}")

    df["phot_g_mean_mag"] = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce")
    df["flux"] = pd.to_numeric(df["aperture_flux_uJy"], errors="coerce")
    df["unc"] = pd.to_numeric(df["aperture_flux_unc_uJy"], errors="coerce")
    df["fatal"] = df["fatal_flag_present"].fillna(True).astype(bool)
    df["usable"] = (~df["fatal"]) & np.isfinite(df["flux"]) & np.isfinite(df["unc"]) & (df["unc"] > 0)
    df["snr"] = df["flux"] / df["unc"].replace(0, np.nan)
    df["mag_bin"] = np.floor(df["phot_g_mean_mag"]).astype("Int64")

    target_rows = []
    for target_id, rows in df.groupby("target_id", dropna=False):
        usable = rows[rows["usable"]].copy()
        flux = usable["flux"].to_numpy(dtype=float)
        unc = usable["unc"].to_numpy(dtype=float)
        snr = usable["snr"].to_numpy(dtype=float)
        med_flux = _nanmedian(flux)
        mad_flux = _mad(flux)
        rel_mad = mad_flux / abs(med_flux) if np.isfinite(mad_flux) and np.isfinite(med_flux) and abs(med_flux) > 0 else np.nan
        target_rows.append(
            {
                "target_id": str(target_id),
                "target_type": str(rows["target_type"].iloc[0]) if "target_type" in rows else "",
                "source_id": str(rows["source_id"].iloc[0]) if "source_id" in rows else "",
                "phot_g_mean_mag": _nanmedian(rows["phot_g_mean_mag"].to_numpy(dtype=float)),
                "mag_bin": _optional_int(_nanmedian(rows["mag_bin"].dropna().astype(float).to_numpy())),
                "measurements": int(len(rows)),
                "usable_measurements": int(len(usable)),
                "fatal_fraction": float(rows["fatal"].mean()),
                "median_flux_uJy": med_flux,
                "median_unc_uJy": _nanmedian(unc),
                "median_snr": _nanmedian(snr),
                "p90_snr": _nanpercentile(snr, 90),
                "flux_mad_uJy": mad_flux,
                "relative_flux_mad": rel_mad,
                "wavelength_min_um": _nanmin(rows.get("cwave_um", pd.Series(dtype=float)).to_numpy(dtype=float)),
                "wavelength_max_um": _nanmax(rows.get("cwave_um", pd.Series(dtype=float)).to_numpy(dtype=float)),
                "detectors": int(rows["detector"].nunique()) if "detector" in rows else 0,
            }
        )
    targets = pd.DataFrame(target_rows)
    targets["allowed_default"] = (
        targets["usable_measurements"].ge(args.min_usable)
        & targets["fatal_fraction"].le(args.max_fatal_frac)
        & targets["median_snr"].ge(args.min_median_snr)
    )

    by_bin = (
        targets.groupby("mag_bin", dropna=False)
        .agg(
            targets=("target_id", "size"),
            allowed=("allowed_default", "sum"),
            median_usable=("usable_measurements", "median"),
            median_fatal_fraction=("fatal_fraction", "median"),
            median_snr=("median_snr", "median"),
            median_relative_flux_mad=("relative_flux_mad", "median"),
            best_median_snr=("median_snr", "max"),
        )
        .reset_index()
        .sort_values("mag_bin", ascending=False, na_position="last")
    )

    output_dir = args.output_dir or args.run_dir / "mag_calibration_stats"
    output_dir.mkdir(parents=True, exist_ok=True)
    targets_path = output_dir / "target_quality.csv"
    bins_path = output_dir / "mag_bin_quality.csv"
    json_path = output_dir / "summary.json"
    targets.sort_values(["mag_bin", "median_snr"], ascending=[False, False]).to_csv(targets_path, index=False)
    by_bin.to_csv(bins_path, index=False)
    summary = {
        "run_dir": str(args.run_dir),
        "spectra_path": str(spectra_path),
        "target_count": int(len(targets)),
        "measurement_rows": int(len(df)),
        "allowed_criteria": {
            "min_usable": args.min_usable,
            "max_fatal_frac": args.max_fatal_frac,
            "min_median_snr": args.min_median_snr,
        },
        "allowed_count": int(targets["allowed_default"].sum()),
        "suggested_allowed_mag_bins": [
            int(row.mag_bin)
            for row in by_bin.itertuples(index=False)
            if pd.notna(row.mag_bin) and int(row.allowed) > 0
        ],
        "target_quality_csv": str(targets_path),
        "mag_bin_quality_csv": str(bins_path),
        "bins": _json_clean(by_bin.to_dict(orient="records")),
    }
    json_path.write_text(json.dumps(_json_clean(summary), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_json_clean(summary), indent=2, sort_keys=True))


def _nanmedian(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.nanmedian(values)) if values.size else float("nan")


def _nanpercentile(values: np.ndarray, q: float) -> float:
    values = values[np.isfinite(values)]
    return float(np.nanpercentile(values, q)) if values.size else float("nan")


def _nanmin(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.nanmin(values)) if values.size else float("nan")


def _nanmax(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.nanmax(values)) if values.size else float("nan")


def _mad(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan")
    med = np.nanmedian(values)
    return float(np.nanmedian(np.abs(values - med)))


def _optional_int(value: float) -> int | None:
    return int(value) if np.isfinite(value) else None


def _json_clean(value):
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_clean(v) for v in value]
    if isinstance(value, tuple):
        return [_json_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if pd.isna(value) if not isinstance(value, (str, bytes, bool, type(None))) else False:
        return None
    return value


if __name__ == "__main__":
    main()
