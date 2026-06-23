from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from spherex_laser_miner.catalog.gaia import query_gaia_for_s_region


DEFAULT_CACHE_ROOT = Path("/mnt/niroseti/spherex_cache")
DEFAULT_RUN_DIR = DEFAULT_CACHE_ROOT / "runs" / "arcturus_gaia_anchor_deep20k_f500_baseline_gpu"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a balanced Gaia fixed-target calibration set by integer G magnitude bins."
    )
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output", type=Path, required=True, help="Output Parquet path for fixed target rows.")
    parser.add_argument("--summary", type=Path, help="Optional JSON summary path.")
    parser.add_argument("--per-bin", type=int, default=10)
    parser.add_argument("--mag-min", type=int, default=5)
    parser.add_argument("--mag-max", type=int, default=16)
    parser.add_argument(
        "--trial-index",
        type=int,
        help="Optional measured trial index from simp_field_trials.json. Defaults to best edge-distance measured trial.",
    )
    parser.add_argument("--max-query-per-bin", type=int, default=500)
    args = parser.parse_args()

    trial = _select_trial(args.source_run_dir, args.trial_index)
    candidate = dict(trial["candidate"])
    s_region = str(candidate["s_region"])
    rows: list[dict[str, Any]] = []
    bins: list[dict[str, Any]] = []

    for mag in range(args.mag_max, args.mag_min - 1, -1):
        g_min = float(mag)
        g_max = float(mag + 1)
        cache_path = (
            args.cache_root
            / "gaia"
            / "target_index"
            / f"mag_cal_{candidate['obs_id']}_D{trial['detector']}_g{mag}_{mag + 1}_top{args.max_query_per_bin}.parquet"
        )
        gaia = query_gaia_for_s_region(
            s_region=s_region,
            cache_path=cache_path,
            max_sources=args.max_query_per_bin,
            g_min=g_min,
            g_max=g_max,
        )
        gaia = gaia.drop_duplicates(subset=["source_id"]).sort_values(["phot_g_mean_mag", "source_id"])
        picked = gaia.head(args.per_bin).copy()
        bins.append(
            {
                "mag_bin": mag,
                "g_min": g_min,
                "g_max": g_max,
                "available": int(len(gaia)),
                "selected": int(len(picked)),
                "cache_path": str(cache_path),
            }
        )
        for rank, (_, src) in enumerate(picked.iterrows(), start=1):
            source_id = str(src["source_id"])
            rows.append(
                {
                    "target_id": f"magcal_g{mag:02d}_{rank:02d}_gaia_dr3_{source_id}",
                    "target_type": "gaia_mag_calibration",
                    "source_id": source_id,
                    "object_name": None,
                    "ra_reference_deg": float(src["ra"]),
                    "dec_reference_deg": float(src["dec"]),
                    "reference_epoch_yr": _optional_float(src.get("ref_epoch")),
                    "pmra_masyr": _optional_float(src.get("pmra")),
                    "pmdec_masyr": _optional_float(src.get("pmdec")),
                    "parallax_mas": _optional_float(src.get("parallax")),
                    "priority_score": float(100.0 - min(float(src.get("phot_g_mean_mag", 99.0)), 99.0)),
                    "phot_g_mean_mag": _optional_float(src.get("phot_g_mean_mag")),
                    "phot_bp_mean_mag": _optional_float(src.get("phot_bp_mean_mag")),
                    "phot_rp_mean_mag": _optional_float(src.get("phot_rp_mean_mag")),
                    "bp_rp": _optional_float(src.get("bp_rp")),
                    "ruwe": _optional_float(src.get("ruwe")),
                    "duplicated_source": bool(src.get("duplicated_source")),
                    "astrometric_params_solved": _optional_float(src.get("astrometric_params_solved")),
                    "target_filter_flags": f"mag_calibration_g{mag}_{mag + 1}",
                    "mag_calibration_bin": mag,
                    "mag_calibration_rank": rank,
                    "mag_calibration_source_run": str(args.source_run_dir),
                }
            )

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    summary = {
        "source_run_dir": str(args.source_run_dir),
        "source_trial_image_id": Path(str(trial["local_path"])).stem,
        "source_trial_detector": int(trial["detector"]),
        "source_trial_edge_distance_pix": float(trial.get("edge_distance_pix") or 0.0),
        "s_region": s_region,
        "per_bin_requested": args.per_bin,
        "target_count": int(len(out)),
        "output": str(args.output),
        "bins": bins,
    }
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _select_trial(run_dir: Path, trial_index: int | None) -> dict[str, Any]:
    path = run_dir / "simp_field_trials.json"
    if not path.exists():
        raise SystemExit(f"Missing source trial file: {path}")
    trials = json.loads(path.read_text(encoding="utf-8"))
    measured = [trial for trial in trials if trial.get("status") == "measured"]
    if not measured:
        raise SystemExit(f"No measured trials in {path}")
    if trial_index is not None:
        for trial in measured:
            if int(trial.get("candidate_index", -1)) == trial_index:
                return trial
        raise SystemExit(f"No measured trial has candidate_index={trial_index}")
    return max(
        measured,
        key=lambda trial: (
            not dict(trial.get("aperture") or {}).get("fatal_flag_present", False),
            float(trial.get("edge_distance_pix") or -1.0),
        ),
    )


def _optional_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


if __name__ == "__main__":
    main()
