from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.wavelength_guard import assert_science_wavelengths  # noqa: E402


DEFAULT_RUN_DIR = Path("/mnt/niroseti/spherex_cache/runs/ucs0972_gpu_g14_16_n1000_f500")
MAX_TARGETS = 12


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build simulator background spectra from a corrected spectra parquet.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--allow-approx-wavelengths",
        action="store_true",
        help="Allow old MEF WCS-WAVE spectra. Not valid for science-grade simulator examples.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    spectra_path = run_dir / "spectra" / "target_spectra.parquet"
    if not spectra_path.exists():
        raise SystemExit(f"Missing spectra parquet: {spectra_path}")

    cols = [
        "target_id",
        "target_type",
        "phot_g_mean_mag",
        "bp_rp",
        "cwave_um",
        "aperture_flux_uJy",
        "aperture_flux_unc_uJy",
        "fatal_flag_present",
        "detector",
        "edge_distance_pix",
    ]
    df = pd.read_parquet(spectra_path)
    try:
        assert_science_wavelengths(df, spectra_path, allow_approx=args.allow_approx_wavelengths)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing required spectra columns: {', '.join(missing)}")
    df = df[cols].copy()
    df["fatal_flag_present"] = df["fatal_flag_present"].fillna(False).astype(bool)
    df["snr"] = df["aperture_flux_uJy"] / df["aperture_flux_unc_uJy"].replace(0, pd.NA)
    grouped = (
        df.groupby("target_id")
        .agg(
            n=("target_id", "size"),
            unflagged=("fatal_flag_present", lambda s: int((~s).sum())),
            flagged=("fatal_flag_present", "sum"),
            median_flux=("aperture_flux_uJy", "median"),
            median_snr=("snr", "median"),
            g_mag=("phot_g_mean_mag", "median"),
            bp_rp=("bp_rp", "median"),
            wave_min=("cwave_um", "min"),
            wave_max=("cwave_um", "max"),
        )
        .reset_index()
    )
    grouped["flag_frac"] = grouped["flagged"] / grouped["n"]
    candidates = grouped[
        (grouped["unflagged"] >= 170)
        & (grouped["flag_frac"] <= 0.15)
        & (grouped["median_snr"] >= 30)
    ].copy()
    candidates = candidates.sort_values(
        ["unflagged", "median_snr", "median_flux"], ascending=[False, False, False]
    )
    selected = ["ucs_0972"]
    selected.extend([t for t in candidates["target_id"].tolist() if t != "ucs_0972"][: MAX_TARGETS - 1])

    payload_targets = []
    for target_id in selected:
        rows = df[df["target_id"].eq(target_id)].sort_values("cwave_um").copy()
        ok = rows[~rows["fatal_flag_present"]].copy()
        if ok.empty:
            continue
        summary = grouped[grouped["target_id"].eq(target_id)].iloc[0].to_dict()
        points = [
            {
                "wave": round(float(row.cwave_um), 6),
                "flux": round(float(row.aperture_flux_uJy), 3),
                "unc": round(float(row.aperture_flux_unc_uJy), 3),
                "detector": int(row.detector),
                "edge": round(float(row.edge_distance_pix), 3),
            }
            for row in ok.itertuples(index=False)
        ]
        label_bits = [target_id]
        if pd.notna(summary.get("g_mag")):
            label_bits.append(f"G={float(summary['g_mag']):.2f}")
        label_bits.append(f"n={int(summary['unflagged'])}")
        payload_targets.append(
            {
                "target_id": target_id,
                "label": " | ".join(label_bits),
                "target_type": str(rows["target_type"].iloc[0]),
                "n": int(summary["n"]),
                "unflagged": int(summary["unflagged"]),
                "flagged": int(summary["flagged"]),
                "median_flux_uJy": float(summary["median_flux"]),
                "median_snr": float(summary["median_snr"]),
                "phot_g_mean_mag": None
                if pd.isna(summary.get("g_mag"))
                else float(summary["g_mag"]),
                "bp_rp": None if pd.isna(summary.get("bp_rp")) else float(summary["bp_rp"]),
                "points": points,
            }
        )

    payload = {
        "source_run": str(run_dir),
        "description": "Selected clean high-coverage spectra from the UCS0972 f500 GPU run.",
        "targets": payload_targets,
    }
    out_path = Path(__file__).with_name("simulator_background_spectra.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "targets": len(payload_targets)}, indent=2))


if __name__ == "__main__":
    main()
