from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS


def project_frame_targets(
    *,
    manifest_path: Path,
    frame_targets_path: Path,
    output_path: Path,
    limit_frames: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    manifest = pd.read_parquet(manifest_path)
    targets = pd.read_parquet(frame_targets_path)
    if limit_frames is not None:
        frame_ids = set(manifest.head(limit_frames)["frame_group_id"].astype(str))
        manifest = manifest[manifest["frame_group_id"].astype(str).isin(frame_ids)].copy()
        targets = targets[targets["frame_group_id"].astype(str).isin(frame_ids)].copy()

    pieces: list[pd.DataFrame] = []
    frame_timings: list[dict[str, Any]] = []
    for frame in manifest.to_dict(orient="records"):
        frame_group_id = str(frame.get("frame_group_id"))
        frame_targets = targets[targets["frame_group_id"].astype(str).eq(frame_group_id)].copy()
        if frame_targets.empty:
            continue
        t0 = time.perf_counter()
        projected = _project_one_frame(frame, frame_targets)
        frame_timings.append(
            {
                "frame_group_id": frame_group_id,
                "image_id": frame.get("image_id"),
                "target_count": int(len(frame_targets)),
                "projected_count": int(len(projected)),
                "in_frame_count": int(projected["in_frame"].sum()),
                "wall_time_sec": time.perf_counter() - t0,
            }
        )
        pieces.append(projected)

    out = pd.concat(pieces, ignore_index=True) if pieces else targets.head(0).copy()
    in_frame_count = int(out["in_frame"].sum()) if "in_frame" in out else 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    out.to_parquet(output_path, index=False)
    write_wall = time.perf_counter() - t0
    frame_targets_summary = _read_json_if_exists(frame_targets_path.with_suffix(".summary.json"))
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "frame_targets_path": str(frame_targets_path),
        "frame_targets_summary_path": str(frame_targets_path.with_suffix(".summary.json")),
        "frame_targets_summary": frame_targets_summary,
        "output_path": str(output_path),
        "frame_count": int(len(manifest)),
        "input_target_rows": int(len(targets)),
        "projected_target_rows": int(len(out)),
        "in_frame_target_rows": in_frame_count,
        "total_wall_sec": time.perf_counter() - started,
        "write_wall_sec": write_wall,
        "frame_timings": frame_timings,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_profile(output_path.with_suffix(".profile.json"), summary)
    return summary


def _project_one_frame(frame: dict[str, Any], targets: pd.DataFrame) -> pd.DataFrame:
    path = Path(str(frame["path"]))
    with fits.open(path, memmap=False, lazy_load_hdus=True) as hdul:
        header = _science_header(hdul)
        wcs = WCS(header)
        naxis1 = int(header.get("NAXIS1") or frame.get("naxis1") or 0)
        naxis2 = int(header.get("NAXIS2") or frame.get("naxis2") or 0)
    world = targets[["ra_deg", "dec_deg"]].to_numpy(dtype=float)
    pix = wcs.all_world2pix(world, 1)
    out = targets.copy()
    out["x_pix"] = pix[:, 0]
    out["y_pix"] = pix[:, 1]
    out["naxis1"] = naxis1
    out["naxis2"] = naxis2
    out["in_frame"] = (
        np.isfinite(out["x_pix"])
        & np.isfinite(out["y_pix"])
        & out["x_pix"].between(1, naxis1, inclusive="both")
        & out["y_pix"].between(1, naxis2, inclusive="both")
    )
    out["edge_distance_pix"] = np.minimum.reduce(
        [
            out["x_pix"].to_numpy(dtype=float) - 1.0,
            out["y_pix"].to_numpy(dtype=float) - 1.0,
            float(naxis1) - out["x_pix"].to_numpy(dtype=float),
            float(naxis2) - out["y_pix"].to_numpy(dtype=float),
        ]
    )
    return out


def _science_header(hdul: fits.HDUList) -> fits.Header:
    for name in ("IMAGE", "PRIMARY"):
        try:
            hdu = hdul[name]
        except Exception:
            continue
        if hdu.header:
            return hdu.header
    return hdul[0].header


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_profile(path: Path, summary: dict[str, Any]) -> None:
    total = max(float(summary.get("total_wall_sec") or 0.0), 1e-12)
    project_wall = sum(float(row["wall_time_sec"]) for row in summary.get("frame_timings", []))
    write_wall = float(summary.get("write_wall_sec") or 0.0)
    profile = {
        "created_utc": summary.get("created_utc"),
        "rows": [
            {
                "stage": "wcs_projection",
                "function_or_script": "luxquarry_allsky_engine.projection.project_frame_targets",
                "wall_time_sec": project_wall,
                "wall_time_pct": 100.0 * project_wall / total,
                "backend": "astropy_vectorized_cpu",
                "rows_out": summary.get("projected_target_rows"),
                "in_frame_rows_out": summary.get("in_frame_target_rows"),
            },
            {
                "stage": "write_projected_targets",
                "function_or_script": "luxquarry_allsky_engine.projection.project_frame_targets",
                "wall_time_sec": write_wall,
                "wall_time_pct": 100.0 * write_wall / total,
                "backend": "pandas_pyarrow",
                "rows_out": summary.get("projected_target_rows"),
            },
        ],
    }
    path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
