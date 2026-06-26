#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grid_survey_v1.grid_survey.gaia_targets import query_tile_catalog, write_tile_outputs
from grid_survey_v1.grid_survey.healpix_tiles import healpix_tile, iter_hpx


DEFAULT_CACHE_ROOT = Path("/mnt/niroseti/spherex_cache")
DEFAULT_OUTPUT_ROOT = DEFAULT_CACHE_ROOT / "grid_survey_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build catalog target manifests for HEALPix survey cells.")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--nside", type=int, required=True, help="HEALPix nside, power of two.")
    parser.add_argument("--hpx", type=int, action="append", help="HEALPix cell id. May be repeated.")
    parser.add_argument("--start-hpx", type=int, help="First HEALPix id for sequential generation.")
    parser.add_argument("--count", type=int, default=1, help="Number of sequential HEALPix ids from --start-hpx.")
    parser.add_argument("--order", choices=["nested", "ring"], default="nested")
    parser.add_argument("--catalog", choices=["gaia", "2mass", "all"], default="gaia")
    parser.add_argument("--g-min", type=float, default=11.0)
    parser.add_argument("--g-max", type=float, default=16.0)
    parser.add_argument("--twomass-band", choices=["J", "H", "Ks"], default="Ks")
    parser.add_argument("--twomass-quality", default="ABC")
    parser.add_argument("--twomass-dataset-name", default="psc_lite")
    parser.add_argument("--twomass-hpx-level", type=int, default=5)
    parser.add_argument("--twomass-selection", choices=["stratified", "brightest", "random"], default="stratified")
    parser.add_argument("--max-sources", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=3000, help="Split large target YAMLs into this many targets per batch. Use 0 to disable batching.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    hpx_ids = list(args.hpx or [])
    if args.start_hpx is not None:
        hpx_ids.extend(iter_hpx(args.start_hpx, args.count))
    hpx_ids = list(dict.fromkeys(hpx_ids))
    if not hpx_ids:
        raise SystemExit("Provide --hpx or --start-hpx")

    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    for hpx in hpx_ids:
        tile = healpix_tile(args.nside, hpx, order=args.order)
        out_dir = args.output_root / tile.tile_id
        summary_path = out_dir / "tile_summary.json"
        if summary_path.exists() and not args.overwrite:
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            print(json.dumps({"status": "skipped_existing", "tile_id": tile.tile_id, "summary": str(summary_path)}), flush=True)
            continue
        catalog_df = query_tile_catalog(
            tile,
            cache_root=args.cache_root,
            catalog=args.catalog,
            mag_min=args.g_min,
            mag_max=args.g_max,
            max_sources=args.max_sources,
            twomass_band=args.twomass_band,
            twomass_quality=args.twomass_quality,
            twomass_dataset_name=args.twomass_dataset_name,
            twomass_hpx_level=args.twomass_hpx_level,
            twomass_selection=args.twomass_selection,
        )
        summary = write_tile_outputs(
            tile=tile,
            gaia=catalog_df,
            output_dir=out_dir,
            cache_root=args.cache_root,
            g_min=args.g_min,
            g_max=args.g_max,
            max_sources=args.max_sources,
            batch_size=args.batch_size,
            catalog=args.catalog,
            twomass_band=args.twomass_band,
            twomass_quality=args.twomass_quality,
            twomass_dataset_name=args.twomass_dataset_name,
            twomass_hpx_level=args.twomass_hpx_level,
            twomass_selection=args.twomass_selection,
        )
        summaries.append(summary)
        print(json.dumps({"status": "done", "tile_id": tile.tile_id, "target_count": summary["target_count"]}), flush=True)

    manifest = {
        "survey_mode": f"grid_survey_v1_healpix_{args.catalog}",
        "created_utc": datetime.now(UTC).isoformat(),
        "cache_root": str(args.cache_root),
        "output_root": str(args.output_root),
        "nside": args.nside,
        "order": args.order,
        "catalog": args.catalog,
        "g_min": args.g_min,
        "g_max": args.g_max,
        "mag_min": args.g_min,
        "mag_max": args.g_max,
        "max_sources": args.max_sources,
        "twomass_band": args.twomass_band if args.catalog in {"2mass", "all"} else None,
        "twomass_quality": args.twomass_quality if args.catalog in {"2mass", "all"} else None,
        "twomass_selection": args.twomass_selection if args.catalog in {"2mass", "all"} else None,
        "batch_size": args.batch_size,
        "tile_count": len(summaries),
        "total_targets": int(sum(int(row.get("target_count") or 0) for row in summaries)),
        "tiles": summaries,
    }
    (args.output_root / "survey_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "survey_manifest", "path": str(args.output_root / "survey_manifest.json"), "total_targets": manifest["total_targets"]}), flush=True)


if __name__ == "__main__":
    main()
