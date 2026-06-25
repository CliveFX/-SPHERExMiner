#!/usr/bin/env python3
"""Build ML training shards from SPHEREx miner run outputs.

The builder is intentionally conservative:

- one target row per run/target;
- one point row per spectral measurement;
- injection truth is written as a separate table instead of duplicating point
  rows for targets with multiple injected lines;
- future explicit injection provenance columns are preferred, while old runs
  fall back to path overrides, run names, summaries, and manifests.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


DEFAULT_RUN_ROOT = Path("/mnt/niroseti/spherex_cache/runs")
DEFAULT_OUTPUT_ROOT = Path("/mnt/niroseti/spherex_cache/ml_datasets")

POINT_COLUMNS = [
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
    "cwave_um",
    "cband_um",
    "wavelength_source",
    "wavelength_calibration_file",
    "wavelength_calibration_collection",
    "wavelength_detector",
    "aperture_flux_uJy",
    "aperture_flux_unc_uJy",
    "psf_flux_uJy",
    "psf_flux_unc_uJy",
    "fatal_flag_present",
    "flags_summary",
    "detector",
    "observation_id",
    "obs_mid_time",
    "image_id",
    "x_pix",
    "y_pix",
    "edge_distance_pix",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "bp_rp",
    "parallax_mas",
    "pmra_masyr",
    "pmdec_masyr",
    "ruwe",
]

QUALITY_COLUMNS = [
    "target_id",
    "spectrum_quality_score",
    "spectrum_quality_category",
    "spectrum_quality_reasons",
    "n_measurements",
    "n_usable_measurements",
    "flag_fraction",
    "smoothness_score",
    "aperture_psf_agreement_score",
    "aperture_psf_corr",
    "median_abs_aperture_snr",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True, help="Stable dataset name used in output paths.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-dir", type=Path, action="append", help="Specific run directory to include.")
    parser.add_argument("--campaign", action="append", help="Include run dirs matching '<campaign>_*'.")
    parser.add_argument("--run-glob", action="append", help="Include run dirs matching a glob under --run-root.")
    parser.add_argument("--limit-runs", type=int, help="Limit number of selected runs after sorting.")
    parser.add_argument(
        "--quality-category",
        action="append",
        choices=["good", "review", "bad"],
        help="Keep only targets with these quality categories. Omit to keep all targets.",
    )
    parser.add_argument("--exclude-injected-targets", action="store_true", help="Drop targets present in an injection manifest.")
    parser.add_argument("--science-only", action="store_true", help="Write only science target/point tables plus split manifest.")
    parser.add_argument("--max-targets-per-run", type=int, help="Optional per-run target cap for smoke datasets.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel run-table builders. Use more for large campaign dataset prep.")
    parser.add_argument("--status-every", type=int, default=1, help="Write status every N processed runs.")
    args = parser.parse_args()

    started = time.perf_counter()
    out_dir = args.output_root / args.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "build_status.json"

    run_dirs = _select_run_dirs(args)
    if not run_dirs:
        raise SystemExit("No run dirs selected")

    science_targets: list[pd.DataFrame] = []
    science_points: list[pd.DataFrame] = []
    narrowband_targets: list[pd.DataFrame] = []
    narrowband_points: list[pd.DataFrame] = []
    truth_tables: list[pd.DataFrame] = []
    split_rows: list[dict[str, object]] = []
    run_summaries: list[dict[str, object]] = []

    completed = 0

    def consume_bundle(run_dir: Path, bundle: dict[str, Any]) -> None:
        nonlocal completed
        completed += 1
        if bundle["science_targets"] is not None:
            science_targets.append(bundle["science_targets"])
            science_points.append(bundle["science_points"])
            narrowband_targets.append(bundle["narrowband_targets"])
            narrowband_points.append(bundle["narrowband_points"])
            split_rows.extend(bundle["split_rows"])
        if bundle["injection_truth"] is not None and not bundle["injection_truth"].empty:
            truth_tables.append(bundle["injection_truth"])
        run_summaries.append(bundle["summary"])
        if args.status_every > 0 and completed % args.status_every == 0:
            _write_status(
                status_path,
                {
                    "dataset_name": args.dataset_name,
                    "status": "running",
                    "run_index": completed,
                    "run_count": len(run_dirs),
                    "current_run": run_dir.name,
                    "target_rows": int(sum(len(t) for t in science_targets)),
                    "point_rows": int(sum(len(t) for t in science_points)),
                    "elapsed_sec": time.perf_counter() - started,
                    "workers": int(args.workers),
                },
            )

    def error_bundle(run_dir: Path, exc: Exception) -> dict[str, Any]:
        return {
            "science_targets": None,
            "science_points": None,
            "narrowband_targets": None,
            "narrowband_points": None,
            "injection_truth": None,
            "split_rows": [],
            "summary": {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            },
        }

    def build_one(run_dir: Path) -> dict[str, Any]:
        return _build_run_tables(
            run_dir,
            dataset_name=args.dataset_name,
            quality_categories=set(args.quality_category or []),
            max_targets=args.max_targets_per_run,
            exclude_injected_targets=args.exclude_injected_targets,
        )

    for index, run_dir in enumerate(run_dirs, start=1):
        _write_status(
            status_path,
            {
                "dataset_name": args.dataset_name,
                "status": "running",
                "run_index": index,
                "run_count": len(run_dirs),
                "current_run": run_dir.name,
                "elapsed_sec": time.perf_counter() - started,
                "workers": int(args.workers),
            },
        )
        if args.workers and args.workers > 1:
            break
        try:
            bundle = build_one(run_dir)
        except Exception as exc:
            consume_bundle(run_dir, error_bundle(run_dir, exc))
            continue
        consume_bundle(run_dir, bundle)

    if args.workers and args.workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = {executor.submit(build_one, run_dir): run_dir for run_dir in run_dirs}
            for future in concurrent.futures.as_completed(futures):
                run_dir = futures[future]
                try:
                    bundle = future.result()
                except Exception as exc:
                    bundle = error_bundle(run_dir, exc)
                consume_bundle(run_dir, bundle)

    outputs = _write_outputs(
        out_dir=out_dir,
        science_targets=_concat(science_targets),
        science_points=_concat(science_points),
        narrowband_targets=pd.DataFrame() if args.science_only else _concat(narrowband_targets),
        narrowband_points=pd.DataFrame() if args.science_only else _concat(narrowband_points),
        injection_truth=_concat(truth_tables),
        split_manifest=pd.DataFrame(split_rows),
        science_only=args.science_only,
    )
    summary = _dataset_summary(
        dataset_name=args.dataset_name,
        out_dir=out_dir,
        run_dirs=run_dirs,
        run_summaries=run_summaries,
        outputs=outputs,
        elapsed_sec=time.perf_counter() - started,
    )
    summary_path = out_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_status(status_path, {**summary, "status": "done"})
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _select_run_dirs(args: argparse.Namespace) -> list[Path]:
    dirs: list[Path] = []
    for run_dir in args.run_dir or []:
        dirs.append(run_dir)
    for campaign in args.campaign or []:
        dirs.extend(sorted(args.run_root.glob(f"{campaign}_*")))
    for pattern in args.run_glob or []:
        dirs.extend(sorted(args.run_root.glob(pattern)))
    if not dirs and args.run_root.exists():
        dirs.extend(sorted(args.run_root.iterdir()))
    out = []
    seen: set[str] = set()
    for path in dirs:
        path = path.resolve()
        if str(path) in seen or not (path / "spectra" / "target_spectra.parquet").exists():
            continue
        seen.add(str(path))
        out.append(path)
    out.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if args.limit_runs:
        out = out[: int(args.limit_runs)]
    return out


def _build_run_tables(
    run_dir: Path,
    *,
    dataset_name: str,
    quality_categories: set[str],
    max_targets: int | None,
    exclude_injected_targets: bool,
) -> dict[str, Any]:
    spectra_path = run_dir / "spectra" / "target_spectra.parquet"
    spectra = pd.read_parquet(spectra_path)
    if spectra.empty:
        return {
            "science_targets": None,
            "science_points": None,
            "narrowband_targets": None,
            "narrowband_points": None,
            "injection_truth": None,
            "split_rows": [],
            "summary": _run_summary(run_dir, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "empty"),
        }
    spectra = _normalize_spectra_provenance(run_dir, spectra)
    quality = _read_quality(run_dir)
    manifest_path = _find_injection_manifest(run_dir, spectra)
    truth = _read_injection_truth(manifest_path, run_dir.name, dataset_name)

    targets = _target_table(dataset_name, run_dir, spectra, quality, truth)
    if quality_categories and "spectrum_quality_category" in targets:
        targets = targets[targets["spectrum_quality_category"].astype(str).isin(quality_categories)].copy()
    if exclude_injected_targets and "is_injected_target" in targets:
        targets = targets[~targets["is_injected_target"].fillna(False).astype(bool)].copy()
    if max_targets is not None and len(targets) > max_targets:
        targets = _cap_targets_preserving_injections(targets, max_targets)
    keep_targets = set(targets["target_id"].astype(str))
    spectra = spectra[spectra["target_id"].astype(str).isin(keep_targets)].copy()

    split_map = {
        str(row["target_id"]): _split_for_key(row.get("source_id") or row["target_id"])
        for row in targets.to_dict(orient="records")
    }
    targets["split_id"] = targets["target_id"].astype(str).map(split_map)
    target_injection_map = targets.set_index(targets["target_id"].astype(str))["is_injected_target"].to_dict()
    points = _point_table(dataset_name, run_dir, spectra, split_map, target_injection_map)
    narrow_targets = targets.copy()
    narrow_points = points.copy()
    split_rows = [
        {
            "dataset_name": dataset_name,
            "run_name": run_dir.name,
            "target_id": target_id,
            "split_id": split_id,
        }
        for target_id, split_id in split_map.items()
    ]
    return {
        "science_targets": targets,
        "science_points": points,
        "narrowband_targets": narrow_targets,
        "narrowband_points": narrow_points,
        "injection_truth": truth,
        "split_rows": split_rows,
        "summary": _run_summary(run_dir, targets, points, truth, "ok"),
    }


def _cap_targets_preserving_injections(targets: pd.DataFrame, max_targets: int) -> pd.DataFrame:
    injected = targets[targets["is_injected_target"].fillna(False).astype(bool)].copy()
    rest = targets[~targets["target_id"].astype(str).isin(set(injected["target_id"].astype(str)))].copy()
    rest = rest.sort_values(
            ["spectrum_quality_score", "n_usable_measurements"],
            ascending=[False, False],
            na_position="last",
            kind="mergesort",
    )
    fill_count = max(0, int(max_targets) - len(injected))
    return pd.concat([injected, rest.head(fill_count)], ignore_index=True) if fill_count else injected


def _normalize_spectra_provenance(run_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "run_name" not in out:
        out["run_name"] = run_dir.name
    if "run_kind" not in out:
        out["run_kind"] = _run_kind_from_name(run_dir.name)
    if "point_injection_applied" not in out:
        point_injected = pd.Series(False, index=out.index)
        if "path_override_applied" in out:
            point_injected |= out["path_override_applied"].fillna(False).astype(bool)
        if {"input_file_path", "original_input_file_path"} <= set(out.columns):
            input_path = out["input_file_path"].fillna("").astype(str)
            original_path = out["original_input_file_path"].fillna("").astype(str)
            point_injected |= original_path.ne("") & input_path.ne(original_path)
        out["point_injection_applied"] = point_injected.astype(bool)
    if "injection_applied" not in out:
        out["injection_applied"] = out["point_injection_applied"].fillna(False).astype(bool)
    if "injection_manifest_path" not in out:
        manifest_path = _find_injection_manifest(run_dir, out)
        out["injection_manifest_path"] = str(manifest_path) if manifest_path else None
    if "run_injection_applied" not in out:
        out["run_injection_applied"] = (
            out["run_kind"].astype(str).eq("injected").any()
            or out["point_injection_applied"].fillna(False).astype(bool).any()
            or out["injection_manifest_path"].notna().any()
        )
    return out


def _read_quality(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "spectra" / "spectrum_quality.parquet"
    if not path.exists():
        return pd.DataFrame()
    quality = pd.read_parquet(path)
    cols = [col for col in QUALITY_COLUMNS if col in quality.columns]
    return quality[cols].copy() if cols else pd.DataFrame()


def _target_table(
    dataset_name: str,
    run_dir: Path,
    spectra: pd.DataFrame,
    quality: pd.DataFrame,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    first_cols = [
        "run_name",
        "run_kind",
        "run_injection_applied",
        "injection_manifest_path",
        "target_id",
        "target_type",
        "source_id",
        "object_name",
        "phot_g_mean_mag",
        "phot_bp_mean_mag",
        "phot_rp_mean_mag",
        "bp_rp",
        "parallax_mas",
        "pmra_masyr",
        "pmdec_masyr",
        "ruwe",
    ]
    first_cols = [col for col in first_cols if col in spectra.columns]
    first = spectra.sort_values(["target_id", "cwave_um"], na_position="last").groupby("target_id", dropna=False)[first_cols].first()
    counts = spectra.groupby("target_id", dropna=False).agg(
        n_points=("target_id", "size"),
        point_injection_applied_count=("point_injection_applied", "sum"),
        wavelength_min_um=("cwave_um", "min"),
        wavelength_max_um=("cwave_um", "max"),
    )
    targets = first.join(counts).reset_index(drop=True)
    targets.insert(0, "dataset_name", dataset_name)
    targets["run_dir"] = str(run_dir)
    targets["point_injection_applied_fraction"] = targets["point_injection_applied_count"] / targets["n_points"]
    if not quality.empty:
        targets = targets.merge(quality, on="target_id", how="left")
    if not truth.empty:
        inj_counts = truth.groupby("target_id", dropna=False).agg(
            injection_count=("injection_id", "size"),
            injected_line_min_nm=("injected_line_nm", "min"),
            injected_line_max_nm=("injected_line_nm", "max"),
            injected_max_find_me_snr=("find_me_snr", "max"),
        ).reset_index()
        targets = targets.merge(inj_counts, on="target_id", how="left")
    else:
        targets["injection_count"] = 0
    targets["injection_count"] = targets["injection_count"].fillna(0).astype(int)
    targets["is_injected_target"] = targets["injection_count"] > 0
    return targets


def _point_table(
    dataset_name: str,
    run_dir: Path,
    spectra: pd.DataFrame,
    split_map: dict[str, str],
    target_injection_map: dict[str, bool],
) -> pd.DataFrame:
    cols = [col for col in POINT_COLUMNS if col in spectra.columns]
    points = spectra[cols].copy()
    points.insert(0, "dataset_name", dataset_name)
    points.insert(1, "run_dir", str(run_dir))
    points.insert(2, "split_id", points["target_id"].astype(str).map(split_map))
    points.insert(3, "point_index", points.groupby("target_id", dropna=False).cumcount())
    points["is_injected_measurement"] = points.get("point_injection_applied", False)
    points["is_injected_target"] = points["target_id"].astype(str).map(target_injection_map).fillna(False).astype(bool)
    return points


def _read_injection_truth(manifest_path: Path | None, run_name: str, dataset_name: str) -> pd.DataFrame:
    if manifest_path is None or not manifest_path.exists():
        return pd.DataFrame()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for injection in manifest.get("injections", []):
        frames = injection.get("frames") or []
        rows.append(
            {
                "dataset_name": dataset_name,
                "run_name": run_name,
                "manifest_path": str(manifest_path),
                "injection_id": injection.get("injection_id"),
                "target_id": str(injection.get("target_id")),
                "line_family": injection.get("line_family"),
                "nominal_line_nm": _num(injection.get("nominal_line_nm")),
                "injected_line_nm": _num(injection.get("injected_line_nm")),
                "offset_nm": _num(injection.get("offset_nm")),
                "line_width_nm": _num(injection.get("line_width_nm")),
                "strength_mode": injection.get("strength_mode"),
                "find_me_snr": _num(injection.get("find_me_snr")),
                "line_flux_uJy": _num(injection.get("line_flux_uJy")),
                "frames_written": int(injection.get("frames_written") or 0),
                "frames_skipped": int(injection.get("frames_skipped") or 0),
                "max_frame_flux_uJy": max([_num(frame.get("injected_flux_uJy")) or 0.0 for frame in frames] or [0.0]),
            }
        )
    return pd.DataFrame(rows)


def _find_injection_manifest(run_dir: Path, spectra: pd.DataFrame | None = None) -> Path | None:
    candidates = [run_dir / "injection_manifest.json", run_dir / "injections" / "injection_manifest.json"]
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
    if spectra is not None and "injection_manifest_path" in spectra:
        for value in spectra["injection_manifest_path"].dropna().astype(str).unique():
            if value:
                candidates.append(Path(value))
    for path in candidates:
        if path.exists():
            return path
    return None


def _run_kind_from_name(run_name: str) -> str:
    if run_name.endswith("_injected") or "_injected_" in run_name or "injected" in run_name:
        return "injected"
    if run_name.endswith("_baseline") or "_baseline_" in run_name or "baseline" in run_name:
        return "baseline"
    return "unknown"


def _split_for_key(value: object) -> str:
    key = str(value)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def _write_outputs(
    *,
    out_dir: Path,
    science_targets: pd.DataFrame,
    science_points: pd.DataFrame,
    narrowband_targets: pd.DataFrame,
    narrowband_points: pd.DataFrame,
    injection_truth: pd.DataFrame,
    split_manifest: pd.DataFrame,
    science_only: bool,
) -> dict[str, str]:
    outputs = {
        "science_targets": out_dir / "science_targets.parquet",
        "science_points": out_dir / "science_points.parquet",
        "narrowband_targets": out_dir / "narrowband_targets.parquet",
        "narrowband_points": out_dir / "narrowband_points.parquet",
        "injection_truth": out_dir / "injection_truth.parquet",
        "split_manifest": out_dir / "split_manifest.parquet",
    }
    science_targets.to_parquet(outputs["science_targets"], index=False)
    science_points.to_parquet(outputs["science_points"], index=False)
    if science_only:
        pd.DataFrame().to_parquet(outputs["narrowband_targets"], index=False)
        pd.DataFrame().to_parquet(outputs["narrowband_points"], index=False)
    else:
        narrowband_targets.to_parquet(outputs["narrowband_targets"], index=False)
        narrowband_points.to_parquet(outputs["narrowband_points"], index=False)
    injection_truth.to_parquet(outputs["injection_truth"], index=False)
    split_manifest.to_parquet(outputs["split_manifest"], index=False)
    return {key: str(value) for key, value in outputs.items()}


def _dataset_summary(
    *,
    dataset_name: str,
    out_dir: Path,
    run_dirs: list[Path],
    run_summaries: list[dict[str, object]],
    outputs: dict[str, str],
    elapsed_sec: float,
) -> dict[str, object]:
    table_counts: dict[str, int] = {}
    for key, path in outputs.items():
        try:
            table_counts[key] = int(pq.ParquetFile(path).metadata.num_rows)
        except Exception:
            table_counts[key] = 0
    quality_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    injection_strength_counts: dict[str, int] = {}
    try:
        targets = pd.read_parquet(outputs["science_targets"], columns=["spectrum_quality_category", "split_id"])
        quality_counts = targets.get("spectrum_quality_category", pd.Series(dtype=str)).fillna("unknown").astype(str).value_counts().sort_index().to_dict()
        split_counts = targets.get("split_id", pd.Series(dtype=str)).fillna("unknown").astype(str).value_counts().sort_index().to_dict()
    except Exception:
        pass
    try:
        truth = pd.read_parquet(outputs["injection_truth"], columns=["find_me_snr"])
        injection_strength_counts = truth.get("find_me_snr", pd.Series(dtype=float)).fillna(-1).astype(str).value_counts().sort_index().to_dict()
    except Exception:
        pass
    return {
        "dataset_name": dataset_name,
        "output_dir": str(out_dir),
        "elapsed_sec": elapsed_sec,
        "selected_run_count": len(run_dirs),
        "successful_run_count": sum(1 for row in run_summaries if row.get("status") == "ok"),
        "error_run_count": sum(1 for row in run_summaries if row.get("status") == "error"),
        "table_counts": table_counts,
        "quality_counts": {str(key): int(value) for key, value in quality_counts.items()},
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "injection_strength_counts": {str(key): int(value) for key, value in injection_strength_counts.items()},
        "outputs": outputs,
        "runs": run_summaries,
    }


def _run_summary(run_dir: Path, targets: pd.DataFrame, points: pd.DataFrame, truth: pd.DataFrame, status: str) -> dict[str, object]:
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "status": status,
        "target_count": int(len(targets)),
        "point_count": int(len(points)),
        "injection_count": int(len(truth)),
    }


def _write_status(path: Path, payload: dict[str, object]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _concat(tables: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _num(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


if __name__ == "__main__":
    main()
