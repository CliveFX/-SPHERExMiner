from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RUN_DIR = Path("/mnt/niroseti/spherex_cache/runs/ucs0972_gpu_g14_16_n10000_f80")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize and plot the UCS_0972 assembled spectrum for SPExPI comparison."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--target-id", default="ucs_0972")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/spherex_compare"))
    parser.add_argument("--bins", type=int, default=28)
    args = parser.parse_args()

    spectra_path = args.run_dir / "spectra" / "target_spectra.parquet"
    if not spectra_path.exists():
        raise SystemExit(f"Missing assembled spectra parquet: {spectra_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(spectra_path)
    target = df[df["target_id"].eq(args.target_id)].sort_values("cwave_um").copy()
    if target.empty:
        raise SystemExit(f"Target {args.target_id!r} not found in {spectra_path}")

    target["flux_jy"] = target["aperture_flux_uJy"] / 1_000_000.0
    target["flux_unc_jy"] = target["aperture_flux_unc_uJy"] / 1_000_000.0
    unflagged = target[~target["fatal_flag_present"].fillna(False).astype(bool)].copy()

    binned = pd.DataFrame()
    if not unflagged.empty:
        wave_bins = pd.cut(unflagged["cwave_um"], bins=args.bins)
        binned = (
            unflagged.groupby(wave_bins, observed=True)
            .agg(cwave_um=("cwave_um", "median"), flux_jy=("flux_jy", "median"), n=("flux_jy", "size"))
            .dropna()
        )

    flagged = target[target["fatal_flag_present"].fillna(False).astype(bool)].copy()
    flagged["local_median_jy"] = flagged["cwave_um"].apply(
        lambda wave: float(
            unflagged.iloc[(unflagged["cwave_um"] - wave).abs().argsort()[:5]]["flux_jy"].median()
        )
        if not unflagged.empty
        else None
    )
    if not flagged.empty:
        flagged["abs_delta_from_local_median_jy"] = (
            flagged["flux_jy"] - flagged["local_median_jy"]
        ).abs()
        worst = flagged.sort_values("abs_delta_from_local_median_jy", ascending=False).iloc[0]
        worst_flagged = {
            "cwave_um": float(worst["cwave_um"]),
            "flux_jy": float(worst["flux_jy"]),
            "local_median_jy": float(worst["local_median_jy"]),
            "image_id": str(worst["image_id"]),
            "edge_distance_pix": float(worst["edge_distance_pix"]),
        }
    else:
        worst_flagged = None

    summary = {
        "run_dir": str(args.run_dir),
        "target_id": args.target_id,
        "rows": int(len(target)),
        "unflagged_rows": int(len(unflagged)),
        "flagged_rows": int(len(flagged)),
        "wavelength_um_min": float(target["cwave_um"].min()),
        "wavelength_um_max": float(target["cwave_um"].max()),
        "flux_jy_min": float(target["flux_jy"].min()),
        "flux_jy_median": float(target["flux_jy"].median()),
        "flux_jy_max": float(target["flux_jy"].max()),
        "unflagged_flux_jy_min": float(unflagged["flux_jy"].min()) if not unflagged.empty else None,
        "unflagged_flux_jy_median": float(unflagged["flux_jy"].median()) if not unflagged.empty else None,
        "unflagged_flux_jy_max": float(unflagged["flux_jy"].max()) if not unflagged.empty else None,
        "worst_flagged_outlier": worst_flagged,
        "note": "SPExPI reference PDF uses Jy with y-axis multiplier x10^-2; these values are Jy.",
    }

    json_path = args.out_dir / f"{args.target_id}_spectrum_compare.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 5.6))
    if not unflagged.empty:
        ax.errorbar(
            unflagged["cwave_um"],
            unflagged["flux_jy"],
            yerr=unflagged["flux_unc_jy"],
            fmt="o",
            ms=4,
            lw=0.7,
            label="unflagged measurements",
            alpha=0.85,
        )
    if not flagged.empty:
        ax.scatter(
            flagged["cwave_um"],
            flagged["flux_jy"],
            s=22,
            color="#f97316",
            label="fatal-flagged measurements",
            alpha=0.9,
        )
    if len(binned) > 1:
        ax.plot(binned["cwave_um"], binned["flux_jy"], "-", color="black", lw=1.4, label="unflagged median bins")
    ax.set_xlim(0.6, 5.2)
    ax.set_ylim(0.0, 0.08)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Flux (Jy)")
    ax.set_title(f"{args.target_id} assembled aperture spectrum")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    png_path = args.out_dir / f"{args.target_id}_spectrum_compare.png"
    fig.savefig(png_path, dpi=180)

    print(json.dumps({**summary, "json_path": str(json_path), "plot_path": str(png_path)}, indent=2))


if __name__ == "__main__":
    main()
