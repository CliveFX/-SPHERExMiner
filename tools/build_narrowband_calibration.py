from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits


DEFAULT_CACHE_ROOT = Path("/mnt/niroseti/spherex_cache")
WCS_COLLECTION = "cal-wcs-v4-2025-254"
CHANNEL_COLLECTION = "cal-sch-v1-2026-106"


def main() -> None:
    cache_root = DEFAULT_CACHE_ROOT
    channel_path = (
        cache_root
        / "calibration"
        / "qr2"
        / "spectral_channels"
        / CHANNEL_COLLECTION
        / f"spectral_channels_spx_{CHANNEL_COLLECTION}.fits.gz"
    )
    wcs_root = cache_root / "calibration" / "qr2" / "spectral_wcs" / WCS_COLLECTION
    out_path = Path(__file__).with_name("narrowband_calibration_qr2.json")

    if not channel_path.exists():
        raise SystemExit(f"Missing spectral channel map: {channel_path}")

    rows: list[dict[str, object]] = []
    with fits.open(channel_path, memmap=True) as hdul:
        channel_map = np.asarray(hdul["CHANNEL_MAP"].data)
        table = hdul["SPECTRAL_CHANNELS"].data

        for row in table:
            detector = int(row["DETECTOR"])
            subchan = int(row["SUBCHAN"])
            wcs_path = wcs_root / str(detector) / f"spectral_wcs_D{detector}_spx_{WCS_COLLECTION}.fits"
            if not wcs_path.exists():
                raise SystemExit(f"Missing spectral WCS: {wcs_path}")

            with fits.open(wcs_path, memmap=True) as wcs_hdul:
                cwave = np.asarray(wcs_hdul["CWAVE"].data, dtype=np.float32)
                cband = np.asarray(wcs_hdul["CBAND"].data, dtype=np.float32)
                mask = (
                    (channel_map == subchan)
                    & np.isfinite(cwave)
                    & np.isfinite(cband)
                    & (cband > 0)
                )
                if not np.any(mask):
                    wcs_summary = {}
                else:
                    cwave_values = cwave[mask]
                    cband_values = cband[mask]
                    r_values = cwave_values / cband_values
                    wcs_summary = {
                        "wcs_pixel_count": int(cwave_values.size),
                        "wcs_cwave_median_um": float(np.nanmedian(cwave_values)),
                        "wcs_cwave_p05_um": float(np.nanpercentile(cwave_values, 5)),
                        "wcs_cwave_p95_um": float(np.nanpercentile(cwave_values, 95)),
                        "wcs_cband_median_um": float(np.nanmedian(cband_values)),
                        "wcs_cband_p05_um": float(np.nanpercentile(cband_values, 5)),
                        "wcs_cband_p95_um": float(np.nanpercentile(cband_values, 95)),
                        "wcs_r_median": float(np.nanmedian(r_values)),
                    }

            rows.append(
                {
                    "detector": detector,
                    "subchannel": subchan,
                    "wavelength_um": float(row["WAVELENGTH"]),
                    "wl_min_um": float(row["WL_MIN"]),
                    "wl_max_um": float(row["WL_MAX"]),
                    "r": float(row["R"]),
                    "r_std": float(row["R_STD"]),
                    "bandwidth_um": float(row["BANDWIDTH"]),
                    **wcs_summary,
                }
            )

    payload = {
        "source": "SPHEREx QR2 spectral_channels and spectral_wcs calibration products from IRSA IBE",
        "spectral_channels_collection": CHANNEL_COLLECTION,
        "spectral_wcs_collection": WCS_COLLECTION,
        "channel_count": len(rows),
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "channel_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
