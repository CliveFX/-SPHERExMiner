from __future__ import annotations

import json
import sqlite3
import subprocess
import threading
import urllib.parse
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

from spherex_laser_miner.live_status import db_path, init_live_status


_COMPLETED_TARGET_CACHE: dict[tuple[str, str, float, float], list[dict[str, object]]] = {}
LIVE_TARGET_OVERLAY_LIMIT_PER_FRAME = 1200


def serve_viewer(run_dir: Path, host: str, port: int) -> None:
    handler = _make_handler(run_dir)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"SPHEREx viewer serving {run_dir.parent} at http://{host}:{port}")
    server.serve_forever()


def _make_handler(run_dir: Path):
    default_run_dir = run_dir.resolve()
    runs_root = default_run_dir.parent

    class ViewerHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            params = urllib.parse.parse_qs(parsed.query)
            active_run_dir = _requested_run_dir(params, runs_root, default_run_dir)
            try:
                if path == "/":
                    self._send_html(_index_html())
                elif path == "/live":
                    self._send_html(_live_html())
                elif path == "/spectra":
                    self._send_html(_spectra_html())
                elif path == "/api/runs":
                    self._send_json(_runs(runs_root, active_run_dir))
                elif path == "/api/summary":
                    self._send_json(_summary(active_run_dir))
                elif path == "/api/live/status":
                    self._send_json(_live_status(active_run_dir))
                elif path == "/api/run/status":
                    self._send_json(_overall_run_status(active_run_dir))
                elif path == "/api/fields":
                    self._send_json(_fields(active_run_dir))
                elif path.startswith("/api/live/frame/") and path.endswith(".jpg"):
                    image_id = urllib.parse.unquote(path.removeprefix("/api/live/frame/").removesuffix(".jpg"))
                    self._send_jpeg(_live_frame_image(active_run_dir, image_id))
                elif path.startswith("/api/field/") and path.endswith("/image.png"):
                    idx = int(path.split("/")[3])
                    self._send_png(_field_image(active_run_dir, idx))
                elif path.startswith("/api/field/") and path.endswith("/targets"):
                    idx = int(path.split("/")[3])
                    self._send_json(_field_targets(active_run_dir, idx))
                elif path == "/api/targets":
                    self._send_json(_targets(active_run_dir, params))
                elif path.startswith("/api/spectrum/"):
                    target_id = urllib.parse.unquote(path.removeprefix("/api/spectrum/"))
                    self._send_json(_spectrum(active_run_dir, target_id))
                elif path.startswith("/api/fits/"):
                    idx = int(path.split("/")[3])
                    self._send_json(_fits_info(active_run_dir, idx))
                else:
                    self.send_error(404)
            except Exception as exc:
                self._send_json({"error": type(exc).__name__, "message": str(exc)}, status=500)

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}")

        def _send_json(self, data: object, status: int = 200) -> None:
            body = json.dumps(_clean_json(data), indent=2, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_png(self, path: Path) -> None:
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_jpeg(self, path: Path) -> None:
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ViewerHandler


def _requested_run_dir(params: dict[str, list[str]], runs_root: Path, default_run_dir: Path) -> Path:
    names = params.get("run") or []
    if not names or not names[0]:
        return default_run_dir
    name = Path(names[0]).name
    candidate = (runs_root / name).resolve()
    try:
        candidate.relative_to(runs_root.resolve())
    except ValueError:
        return default_run_dir
    return candidate


def _runs(runs_root: Path, active_run_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not runs_root.exists():
        return rows
    for path in sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        assembly_path = path / "spectra" / "assembly_summary.json"
        summary = _read_json(assembly_path)
        status = _read_json(path / "run_summary.json")
        trial_path = path / "simp_field_trials.json"
        rows.append(
            {
                "name": path.name,
                "path": str(path),
                "active": path.resolve() == active_run_dir.resolve(),
                "mtime": path.stat().st_mtime,
                "mtime_iso": _format_time(path.stat().st_mtime),
                "has_spectra": assembly_path.exists(),
                "has_trials": trial_path.exists(),
                "target": status.get("target") if isinstance(status, dict) else None,
                "measurement_rows": summary.get("measurement_rows") if isinstance(summary, dict) else None,
                "target_count": summary.get("target_count") if isinstance(summary, dict) else None,
                "shard_count": summary.get("shard_count") if isinstance(summary, dict) else None,
            }
        )
    return rows


def _summary(run_dir: Path) -> dict[str, object]:
    qa = _read_json(run_dir / "qa.json")
    assembly = _read_json(run_dir / "spectra" / "assembly_summary.json")
    jobs = _read_json(run_dir / "field_jobs.json")
    return {"run_dir": str(run_dir), "qa": qa, "assembly": assembly, "field_job_count": len(jobs) if isinstance(jobs, list) else 0}


def _fields(run_dir: Path) -> list[dict[str, object]]:
    jobs = _jobs(run_dir)
    fields = []
    for idx, job in enumerate(jobs):
        candidate = dict(job.get("candidate") or {})
        fields.append(
            {
                "idx": idx,
                "image_id": job.get("image_id"),
                "detector": candidate.get("detector"),
                "observation_id": candidate.get("obs_id"),
                "targets_measured": job.get("targets_measured"),
                "targets_considered": job.get("targets_considered"),
                "simp_measured": job.get("simp_measured"),
                "input_file_path": job.get("input_file_path"),
                "measurement_path": job.get("measurement_path"),
                "target_selection_path": job.get("target_selection_path"),
            }
        )
    return fields


def _live_status(run_dir: Path) -> dict[str, object]:
    init_live_status(run_dir)
    path = db_path(run_dir)
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        frame_limit = 32
        frames = [
            dict(row)
            for row in con.execute(
                """
                SELECT * FROM frames
                ORDER BY
                  CASE status WHEN 'active' THEN 0 WHEN 'error' THEN 1 ELSE 2 END,
                  worker_name ASC,
                  started_at ASC,
                  updated_at DESC
                LIMIT 96
                """
            )
        ]
        has_live_frames = bool(frames)
        if not frames:
            frames = _completed_live_frames_from_jobs(run_dir, limit=frame_limit)
        else:
            active = sorted(
                [row for row in frames if row.get("status") == "active"],
                key=lambda row: (_worker_sort_key(row.get("worker_name")), float(row.get("started_at") or 0)),
            )
            recent = sorted(
                [row for row in frames if row.get("status") != "active"],
                key=lambda row: float(row.get("updated_at") or 0),
                reverse=True,
            )
            frames = (active + recent)[:frame_limit]

        frame_ids = [str(frame["image_id"]) for frame in frames if frame.get("image_id")]
        targets_by_frame: dict[str, list[dict[str, object]]] = {image_id: [] for image_id in frame_ids}
        target_counts_by_frame: dict[str, dict[str, int]] = {}
        if has_live_frames and frame_ids:
            placeholders = ",".join("?" for _ in frame_ids)
            for row in con.execute(
                f"""
                SELECT image_id, status, COUNT(*) AS count
                FROM targets
                WHERE image_id IN ({placeholders})
                GROUP BY image_id, status
                """,
                frame_ids,
            ):
                target_counts_by_frame.setdefault(str(row["image_id"]), {})[str(row["status"])] = int(row["count"])
            for row in con.execute(
                f"""
                WITH ranked AS (
                  SELECT *,
                    ROW_NUMBER() OVER (
                      PARTITION BY image_id
                      ORDER BY
                        CASE status WHEN 'active' THEN 0 WHEN 'queued' THEN 1 WHEN 'error' THEN 2 ELSE 3 END,
                        updated_at DESC
                    ) AS rn
                  FROM targets
                  WHERE image_id IN ({placeholders})
                )
                SELECT * FROM ranked
                WHERE rn <= ?
                ORDER BY image_id, rn
                """,
                [*frame_ids, LIVE_TARGET_OVERLAY_LIMIT_PER_FRAME],
            ):
                targets_by_frame.setdefault(str(row["image_id"]), []).append(dict(row))

        spectra = [
            dict(row)
            for row in con.execute(
                """
                SELECT * FROM spectra_points
                WHERE cwave_um IS NOT NULL
                ORDER BY updated_at DESC
                LIMIT 2500
                """
            )
        ]

    for frame in frames:
        image_id = str(frame.get("image_id"))
        if not frame.get("input_file_path"):
            job = _job_by_image_id(run_dir, image_id)
            if job:
                frame["input_file_path"] = job.get("input_file_path")
                candidate = dict(job.get("candidate") or {})
                frame["detector"] = candidate.get("detector")
                frame["observation_id"] = candidate.get("obs_id")
        frame["targets"] = targets_by_frame.get(image_id) or _completed_targets_for_frame(run_dir, image_id)
        counts = {"queued": 0, "active": 0, "done": 0, "error": 0}
        if image_id in target_counts_by_frame:
            for status, count in target_counts_by_frame[image_id].items():
                counts[status] = count
        else:
            for target in frame["targets"]:
                status = str(target.get("status") or "queued")
                counts[status] = counts.get(status, 0) + 1
        frame["target_status_counts"] = counts
        frame["target_count"] = sum(counts.values())
        frame["target_overlay_count"] = len(frame["targets"])
        frame["target_overlay_truncated"] = frame["target_overlay_count"] < frame["target_count"]
        frame["progress_percent"] = (100.0 * counts.get("done", 0) / frame["target_count"]) if frame["target_count"] else None
        frame["elapsed_sec"] = _elapsed_seconds(frame.get("started_at"), frame.get("finished_at"))
        perf = _frame_perf(run_dir, image_id)
        if perf:
            frame["performance"] = perf

    return {
        "run_dir": str(run_dir),
        "db_path": str(path),
        "frames": frames,
        "spectra_points": spectra,
        "frame_count": len(frames),
        "active_frame_count": sum(1 for frame in frames if frame.get("status") == "active"),
        "frame_limit": frame_limit,
    }


def _overall_run_status(run_dir: Path) -> dict[str, object]:
    trials = _read_json(run_dir / "simp_field_trials.json")
    if not isinstance(trials, list):
        trials = _read_json(run_dir / "simp_field_trials.partial.json")
    if not isinstance(trials, list):
        trials = []
    measured_trials = [trial for trial in trials if isinstance(trial, dict) and trial.get("status") == "measured"]

    shard_root = run_dir / "field_shards"
    measurement_files = list(shard_root.glob("image_id=*/measurements.parquet")) if shard_root.exists() else []
    job_files = list(shard_root.glob("image_id=*/field_job.json")) if shard_root.exists() else []

    live = _live_db_counts(run_dir)
    perf = _live_perf_summary(run_dir)
    spectra_dir = run_dir / "spectra"
    assembly_path = spectra_dir / "assembly_summary.json"
    assembly = _read_json(assembly_path)
    spectra_mtime = assembly_path.stat().st_mtime if assembly_path.exists() else None
    process = _depth_process_status()

    total_fields = len(measured_trials) or len(trials) or None
    completed_fields = int(live.get("done", 0))
    errored_fields = live.get("error", 0)
    active_fields = live.get("active", 0)
    seen_fields = live.get("total", 0)
    progress = (100.0 * min(completed_fields + errored_fields, total_fields) / total_fields) if total_fields else None

    return {
        "run_dir": str(run_dir),
        "process": process,
        "trials_total": len(trials),
        "measured_trial_count": len(measured_trials),
        "field_total_estimate": total_fields,
        "field_shards_with_measurements": len(measurement_files),
        "field_job_files": len(job_files),
        "fields_completed": completed_fields,
        "fields_seen": seen_fields,
        "live_frames_active": active_fields,
        "live_frames_done": live.get("done", 0),
        "live_frames_error": errored_fields,
        "live_frames_total_seen": live.get("total", 0),
        "live_targets_queued": live.get("target_queued", 0),
        "live_targets_active": live.get("target_active", 0),
        "live_targets_done": live.get("target_done", 0),
        "live_spectra_points": live.get("spectra_points", 0),
        "performance": perf,
        "field_progress_percent": progress,
        "assembly": assembly,
        "assembly_mtime": spectra_mtime,
        "assembly_mtime_iso": _format_time(spectra_mtime),
        "run_phase": _infer_run_phase(process, active_fields, spectra_mtime, len(measurement_files), total_fields),
    }


def _live_db_counts(run_dir: Path) -> dict[str, int]:
    init_live_status(run_dir)
    path = db_path(run_dir)
    counts: dict[str, int] = {
        "total": 0,
        "active": 0,
        "done": 0,
        "error": 0,
        "target_queued": 0,
        "target_active": 0,
        "target_done": 0,
        "spectra_points": 0,
    }
    try:
        with sqlite3.connect(path) as con:
            for status, count in con.execute("SELECT status, COUNT(*) FROM frames GROUP BY status"):
                counts[str(status)] = int(count)
                counts["total"] += int(count)
            for status, count in con.execute("SELECT status, COUNT(*) FROM targets GROUP BY status"):
                counts[f"target_{status}"] = int(count)
            row = con.execute("SELECT COUNT(*) FROM spectra_points").fetchone()
            counts["spectra_points"] = int(row[0] if row else 0)
    except sqlite3.Error:
        pass
    return counts


def _frame_perf(run_dir: Path, image_id: str) -> dict[str, object] | None:
    path = db_path(run_dir)
    try:
        with sqlite3.connect(path) as con:
            con.row_factory = sqlite3.Row
            row = con.execute("SELECT * FROM frame_perf WHERE image_id = ?", (image_id,)).fetchone()
            return dict(row) if row else None
    except sqlite3.Error:
        return None


def _live_perf_summary(run_dir: Path) -> dict[str, object]:
    init_live_status(run_dir)
    path = db_path(run_dir)
    try:
        with sqlite3.connect(path) as con:
            con.row_factory = sqlite3.Row
            rows = [dict(row) for row in con.execute("SELECT * FROM frame_perf")]
    except sqlite3.Error:
        rows = []
    if not rows:
        return {
            "frames_profiled": 0,
            "targets_profiled": 0,
            "target_rate_per_sec_sum": None,
            "target_rate_per_core_sec": None,
        }
    targets = sum(int(row.get("targets_measured") or 0) for row in rows)
    elapsed_sum = sum(float(row.get("elapsed_sec") or 0) for row in rows)
    worker_count = len({row.get("worker_name") for row in rows if row.get("worker_name")})
    wall_start = min(float(row.get("finished_at") or 0) - float(row.get("elapsed_sec") or 0) for row in rows)
    wall_end = max(float(row.get("finished_at") or 0) for row in rows)
    wall_elapsed = max(0.0, wall_end - wall_start)
    phase_cols = [
        "fits_open_sec",
        "calibration_sec",
        "selection_sec",
        "photometry_sec",
        "aperture_sec",
        "calibrated_aperture_sec",
        "psf_sec",
        "status_sec",
        "write_sec",
        "elapsed_sec",
    ]
    phase = {f"{col}_median": _median([row.get(col) for row in rows]) for col in phase_cols}
    phase.update({f"{col}_sum": _sum_float([row.get(col) for row in rows]) for col in phase_cols})
    return {
        "frames_profiled": len(rows),
        "targets_profiled": targets,
        "workers_seen": worker_count,
        "wall_elapsed_sec": wall_elapsed,
        "sum_worker_elapsed_sec": elapsed_sum,
        "target_rate_per_wall_sec": targets / wall_elapsed if wall_elapsed > 0 else None,
        "target_rate_per_worker_sec": targets / elapsed_sum if elapsed_sum > 0 else None,
        "target_rate_per_core_sec": targets / elapsed_sum if elapsed_sum > 0 else None,
        **phase,
    }


def _median(values: list[object]) -> float | None:
    arr = sorted(float(value) for value in values if value is not None)
    if not arr:
        return None
    return arr[len(arr) // 2]


def _sum_float(values: list[object]) -> float:
    return sum(float(value) for value in values if value is not None)


def _depth_process_status() -> dict[str, object]:
    try:
        proc = subprocess.run(
            ["pgrep", "-af", "spherex-mine run-depth-test"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )
    except Exception:
        return {"running": False, "processes": []}
    processes = []
    for line in proc.stdout.splitlines():
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        if "pgrep -af" in parts[1]:
            continue
        processes.append({"pid": int(parts[0]), "cmd": parts[1]})
    return {"running": bool(processes), "processes": processes}


def _infer_run_phase(
    process: dict[str, object],
    active_fields: int,
    spectra_mtime: float | None,
    measurement_files: int,
    total_fields: int | None,
) -> str:
    if process.get("running") and active_fields:
        return "field_workers_active"
    if process.get("running"):
        return "evaluating_or_assembling"
    if total_fields and measurement_files >= total_fields and spectra_mtime:
        return "assembled"
    if measurement_files:
        return "partial_or_failed"
    return "idle"


def _format_time(value: float | None) -> str | None:
    if value is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(value))


def _worker_sort_key(value: object) -> tuple[int, str]:
    text = str(value or "")
    if "_" in text:
        suffix = text.rsplit("_", 1)[-1]
        if suffix.isdigit():
            return (int(suffix), text)
    return (999999, text)


def _elapsed_seconds(started_at: object, finished_at: object | None = None) -> float | None:
    try:
        start = float(started_at)
    except Exception:
        return None
    try:
        end = float(finished_at) if finished_at is not None else time.time()
    except Exception:
        end = time.time()
    return max(0.0, end - start)


def _completed_live_frames_from_jobs(run_dir: Path, limit: int) -> list[dict[str, object]]:
    frames = []
    for job in reversed(_jobs(run_dir)):
        candidate = dict(job.get("candidate") or {})
        measurement_path = Path(str(job.get("measurement_path", "")))
        updated = measurement_path.stat().st_mtime if measurement_path.exists() else None
        cwave_um = None
        cband_um = None
        constellation = None
        if measurement_path.exists():
            try:
                m = pd.read_parquet(
                    measurement_path,
                    columns=["cwave_um", "cband_um", "ra_epoch_deg", "dec_epoch_deg"],
                ).head(1)
                if len(m):
                    cwave_um = m.iloc[0].get("cwave_um")
                    cband_um = m.iloc[0].get("cband_um")
                    ra = m.iloc[0].get("ra_epoch_deg")
                    dec = m.iloc[0].get("dec_epoch_deg")
                    if pd.notna(ra) and pd.notna(dec):
                        constellation = SkyCoord(float(ra) * u.deg, float(dec) * u.deg).get_constellation()
            except Exception:
                pass
        frames.append(
            {
                "image_id": job.get("image_id"),
                "status": "done",
                "worker_name": None,
                "input_file_path": job.get("input_file_path"),
                "detector": candidate.get("detector"),
                "observation_id": candidate.get("obs_id"),
                "cwave_um": cwave_um,
                "cband_um": cband_um,
                "constellation": constellation,
                "started_at": None,
                "updated_at": updated,
                "finished_at": updated,
                "error": None,
            }
        )
        if len(frames) >= limit:
            break
    return frames


def _job_by_image_id(run_dir: Path, image_id: str) -> dict[str, object] | None:
    for job in _jobs(run_dir):
        if str(job.get("image_id")) == image_id:
            return job
    return None


def _completed_targets_for_frame(run_dir: Path, image_id: str) -> list[dict[str, object]]:
    job = _job_by_image_id(run_dir, image_id)
    if not job:
        return []
    selection_path = Path(str(job.get("target_selection_path", "")))
    measurement_path = Path(str(job.get("measurement_path", "")))
    if not selection_path.exists():
        return []
    selection_mtime = selection_path.stat().st_mtime
    measurement_mtime = measurement_path.stat().st_mtime if measurement_path.exists() else 0.0
    key = (str(run_dir), image_id, selection_mtime, measurement_mtime)
    cached = _COMPLETED_TARGET_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_COMPLETED_TARGET_CACHE) > 120:
        _COMPLETED_TARGET_CACHE.clear()
    selected = pd.read_parquet(selection_path)
    if "selected_for_photometry" in selected.columns:
        selected = selected[selected["selected_for_photometry"] == True]  # noqa: E712
    done_ids: set[str] = set()
    measurement_cols = ["target_id", "aperture_flux_uJy", "psf_flux_uJy", "fatal_flag_present", "cwave_um"]
    measurements = pd.DataFrame()
    if measurement_path.exists():
        measurements = pd.read_parquet(measurement_path)
        done_ids = set(measurements["target_id"].astype(str))
    rows = []
    for row in selected.head(300).to_dict(orient="records"):
        target_id = str(row.get("target_id"))
        out = {
            "image_id": image_id,
            "target_id": target_id,
            "target_type": row.get("target_type"),
            "x_pix": row.get("x_pix"),
            "y_pix": row.get("y_pix"),
            "phot_g_mean_mag": row.get("phot_g_mean_mag"),
            "status": "done" if target_id in done_ids else "queued",
        }
        if target_id in done_ids and len(measurements):
            mrow = measurements[measurements["target_id"].astype(str) == target_id].head(1)
            for col in measurement_cols:
                if col in mrow.columns and len(mrow):
                    out[col] = mrow.iloc[0][col]
        rows.append(out)
    _COMPLETED_TARGET_CACHE[key] = rows
    return rows


def _field_targets(run_dir: Path, idx: int) -> list[dict[str, object]]:
    job = _jobs(run_dir)[idx]
    path = Path(str(job["target_selection_path"]))
    df = pd.read_parquet(path)
    cols = [
        "target_id",
        "target_type",
        "source_id",
        "x_pix",
        "y_pix",
        "edge_distance_pix",
        "selected_for_photometry",
        "phot_g_mean_mag",
        "ra_epoch_deg",
        "dec_epoch_deg",
    ]
    cols = [col for col in cols if col in df.columns]
    return df[cols].head(1000).to_dict(orient="records")


def _targets(run_dir: Path, params: dict[str, list[str]]) -> dict[str, object]:
    path = run_dir / "spectra" / "target_summary.parquet"
    if not path.exists():
        return {"rows": [], "total": 0, "limit": 0, "offset": 0, "sort": "measurements", "q": ""}
    limit = min(500, max(25, _query_int(params, "limit", 200)))
    offset = max(0, _query_int(params, "offset", 0))
    sort = (params.get("sort") or ["measurements"])[0]
    q = (params.get("q") or [""])[0].strip().lower()
    df = pd.read_parquet(path)
    spectra_path = run_dir / "spectra" / "target_spectra.parquet"
    if spectra_path.exists():
        spectra = pd.read_parquet(spectra_path, columns=["target_id", "fatal_flag_present"])
        flag_summary = (
            spectra.assign(fatal_flag_present=spectra["fatal_flag_present"].fillna(False).astype(bool))
            .groupby("target_id", dropna=False)
            .agg(
                flagged_measurements=("fatal_flag_present", "sum"),
                total_measurements_for_flags=("fatal_flag_present", "size"),
            )
            .reset_index()
        )
        flag_summary["flagged_fraction"] = (
            flag_summary["flagged_measurements"] / flag_summary["total_measurements_for_flags"]
        )
        df = df.merge(flag_summary, on="target_id", how="left")
    for col in ("flagged_measurements", "total_measurements_for_flags"):
        if col in df:
            df[col] = df[col].fillna(0).astype(int)
    if "flagged_fraction" in df:
        df["flagged_fraction"] = df["flagged_fraction"].fillna(0.0)
    if q:
        query_mask = df["target_id"].astype(str).str.lower().str.contains(q, regex=False)
        if "target_type" in df.columns:
            query_mask |= df["target_type"].astype(str).str.lower().str.contains(q, regex=False)
        df = df[query_mask]
    total = int(len(df))
    df = _sort_targets_df(df, sort)
    rows = df.iloc[offset : offset + limit].to_dict(orient="records")
    return {"rows": rows, "total": total, "limit": limit, "offset": offset, "sort": sort, "q": q}


def _sort_targets_df(df: pd.DataFrame, sort: str) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    if sort == "id":
        return work.sort_values("target_id", kind="mergesort")
    if sort == "snr" and "max_snr_uJy" in work:
        return work.sort_values("max_snr_uJy", ascending=False, na_position="last", kind="mergesort")
    if sort == "flux" and "median_flux_uJy" in work:
        work["_abs_flux"] = pd.to_numeric(work["median_flux_uJy"], errors="coerce").abs()
        return work.sort_values("_abs_flux", ascending=False, na_position="last", kind="mergesort").drop(columns=["_abs_flux"])
    if sort == "flags":
        for col in ("flagged_fraction", "flagged_measurements", "n_measurements"):
            if col not in work:
                work[col] = 0
        return work.sort_values(
            ["flagged_fraction", "flagged_measurements", "n_measurements"],
            ascending=[True, True, False],
            na_position="last",
            kind="mergesort",
        )
    if "n_measurements" in work:
        return work.sort_values("n_measurements", ascending=False, na_position="last", kind="mergesort")
    return work.sort_values("target_id", kind="mergesort")


def _query_int(params: dict[str, list[str]], name: str, default: int) -> int:
    try:
        return int((params.get(name) or [default])[0])
    except (TypeError, ValueError):
        return default


def _spectrum(run_dir: Path, target_id: str) -> dict[str, object]:
    path = run_dir / "spectra" / "target_spectra.parquet"
    if not path.exists():
        return {"target_id": target_id, "rows": []}
    df = pd.read_parquet(path)
    rows = df[df["target_id"] == target_id].sort_values("cwave_um")
    if "input_file_path" in rows.columns:
        rows = rows.copy()
        rows["fits_file"] = rows["input_file_path"].map(lambda value: Path(str(value)).name if pd.notna(value) else None)
    return {"target_id": target_id, "rows": rows.to_dict(orient="records")}


def _fits_info(run_dir: Path, idx: int) -> dict[str, object]:
    job = _jobs(run_dir)[idx]
    path = Path(str(job["input_file_path"]))
    with fits.open(path, memmap=True) as hdul:
        hdus = []
        for hdu in hdul:
            shape = list(hdu.data.shape) if getattr(hdu, "data", None) is not None and hasattr(hdu.data, "shape") else None
            hdus.append({"name": hdu.name, "class": type(hdu).__name__, "shape": shape})
        header = hdul["IMAGE"].header
        return {
            "path": str(path),
            "hdus": hdus,
            "detector": header.get("DETECTOR"),
            "bunit": header.get("BUNIT"),
            "obsid": header.get("OBSID"),
            "date": header.get("DATE"),
        }


def _field_image(run_dir: Path, idx: int) -> Path:
    cache_dir = run_dir / "viewer_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"field_{idx:03d}.png"
    job = _jobs(run_dir)[idx]
    image_path = Path(str(job["input_file_path"]))
    selection_path = Path(str(job["target_selection_path"]))
    image_mtime = image_path.stat().st_mtime
    selection_mtime = selection_path.stat().st_mtime
    if out.exists() and out.stat().st_mtime > max(image_mtime, selection_mtime):
        return out

    with fits.open(image_path, memmap=True) as hdul:
        image = np.asarray(hdul["IMAGE"].data, dtype=float)
    step = max(1, int(np.ceil(max(image.shape) / 1024)))
    preview = image[::step, ::step]
    finite = preview[np.isfinite(preview)]
    vmin, vmax = np.percentile(finite, [1, 99.5]) if len(finite) else (None, None)

    targets = pd.read_parquet(selection_path)
    selected = targets[targets["selected_for_photometry"] == True]  # noqa: E712
    simp = selected[selected["target_id"] == "simp0136"]
    gaia = selected[selected["target_id"] != "simp0136"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(preview, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, extent=[0, image.shape[1], 0, image.shape[0]])
    if len(gaia):
        ax.scatter(gaia["x_pix"], gaia["y_pix"], s=8, facecolors="none", edgecolors="#00bcd4", linewidths=0.7, label="Gaia")
    if len(simp):
        ax.scatter(simp["x_pix"], simp["y_pix"], s=90, marker="x", c="#ffcc00", linewidths=2.0, label="SIMP")
    ax.set_title(str(job.get("image_id")))
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _live_frame_image(run_dir: Path, image_id: str) -> Path:
    cache_dir = run_dir / "viewer_cache" / "live_jpeg"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{_safe_name(image_id)}.jpg"
    job = _job_by_image_id(run_dir, image_id)
    image_path: Path | None = None
    if job:
        image_path = Path(str(job.get("input_file_path")))
    if image_path is None:
        path = db_path(run_dir)
        if path.exists():
            with sqlite3.connect(path) as con:
                row = con.execute("SELECT input_file_path FROM frames WHERE image_id = ?", (image_id,)).fetchone()
                if row and row[0]:
                    image_path = Path(str(row[0]))
    if image_path is None or not image_path.exists():
        raise FileNotFoundError(f"No FITS path for image_id={image_id}")
    if out.exists() and out.stat().st_mtime > image_path.stat().st_mtime:
        return out

    with fits.open(image_path, memmap=True) as hdul:
        image = np.asarray(hdul["IMAGE"].data, dtype=float)
    step = max(1, int(np.ceil(max(image.shape) / 900)))
    preview = image[::step, ::step]
    finite = preview[np.isfinite(preview)]
    vmin, vmax = np.percentile(finite, [1, 99.7]) if len(finite) else (None, None)
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(preview, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, extent=[0, image.shape[1], 0, image.shape[0]])
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out, dpi=130, format="jpg", facecolor="#020617")
    plt.close(fig)
    return out


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._=-" else "_" for ch in value)


def _jobs(run_dir: Path) -> list[dict[str, object]]:
    jobs = _read_json(run_dir / "field_jobs.json")
    if not isinstance(jobs, list):
        return []
    return jobs


def _read_json(path: Path) -> object:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return str(value)


def _clean_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [_clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return _clean_json(value.item())
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _index_html() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SPHEREx Field Miner Viewer</title>
  <style>
    body { margin: 0; font: 14px system-ui, sans-serif; color: #1f2933; background: #f5f7fa; }
    header { padding: 12px 16px; background: #102a43; color: white; }
    main { display: grid; grid-template-columns: 320px 1fr; gap: 14px; padding: 14px; }
    section { background: white; border: 1px solid #d9e2ec; border-radius: 6px; padding: 12px; }
    select, button { width: 100%; padding: 7px; margin: 4px 0 10px; }
    img { max-width: 100%; border: 1px solid #d9e2ec; background: #111; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    td, th { border-bottom: 1px solid #edf2f7; padding: 4px; text-align: left; }
    .grid { display: grid; grid-template-columns: minmax(360px, 0.8fr) minmax(640px, 1.2fr); gap: 14px; }
    #plot { width: 100%; height: 620px; border: 1px solid #d9e2ec; background: white; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; white-space: pre-wrap; }
  </style>
</head>
<body>
<header><strong>SPHEREx Field Miner Viewer</strong></header>
<main>
  <section>
    <h3>Run</h3>
    <select id="runSelect" onchange="switchRun()"></select>
    <h3>Fields</h3>
    <select id="fieldSelect"></select>
    <button onclick="loadField()">Load Field</button>
    <h3>Targets</h3>
    <select id="targetSelect" size="14"></select>
    <button onclick="loadSpectrum()">Load Spectrum</button>
    <div id="summary" class="mono"></div>
  </section>
  <div>
    <section>
      <h3>FITS Field Preview</h3>
      <img id="fieldImage" alt="field preview">
      <div id="fitsInfo" class="mono"></div>
    </section>
    <div class="grid">
      <section>
        <h3>Selected Field Targets</h3>
        <div id="fieldTargets"></div>
      </section>
      <section>
        <h3>Spectrum</h3>
        <div class="mono">Aperture is SAPM-calibrated reference path. PSF points are first-pass / experimental.</div>
        <svg id="plot" viewBox="0 0 900 620"></svg>
        <div id="spectrumTable"></div>
      </section>
    </div>
  </div>
</main>
<script>
let fields = [];
let targets = [];
let activeRun = new URLSearchParams(window.location.search).get('run') || '';

function runQS(extra) {
  const p = new URLSearchParams(extra || '');
  if (activeRun) p.set('run', activeRun);
  const s = p.toString();
  return s ? '?' + s : '';
}

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

async function init() {
  const runs = await getJSON('/api/runs' + runQS());
  const rs = document.getElementById('runSelect');
  rs.innerHTML = runs.map(r => `<option value="${r.name}">${r.name} ${r.has_spectra ? '[' + (r.measurement_rows || 0) + ' rows]' : ''}</option>`).join('');
  if (!activeRun && runs.length) activeRun = runs[0].name;
  rs.value = activeRun;
  document.getElementById('summary').textContent = JSON.stringify(await getJSON('/api/summary' + runQS()), null, 2);
  fields = await getJSON('/api/fields' + runQS());
  const fs = document.getElementById('fieldSelect');
  fs.innerHTML = fields.map(f => `<option value="${f.idx}">D${f.detector} ${f.observation_id} (${f.targets_measured})</option>`).join('');
  const targetData = await getJSON('/api/targets' + runQS('limit=500&offset=0&sort=measurements'));
  targets = targetData.rows || [];
  const ts = document.getElementById('targetSelect');
  ts.innerHTML = targets.map(t => `<option value="${t.target_id}">${t.target_id} [${t.n_measurements}] ${Number(t.wavelength_min_um).toFixed(3)}-${Number(t.wavelength_max_um).toFixed(3)}um</option>`).join('');
  ts.value = targets.some(t => t.target_id === 'simp0136') ? 'simp0136' : (targets[0]?.target_id || '');
  await loadField();
  await loadSpectrum();
}

function switchRun() {
  activeRun = document.getElementById('runSelect').value;
  init();
}

async function loadField() {
  const idx = document.getElementById('fieldSelect').value || 0;
  document.getElementById('fieldImage').src = `/api/field/${idx}/image.png` + runQS(`ts=${Date.now()}`);
  document.getElementById('fitsInfo').textContent = JSON.stringify(await getJSON(`/api/fits/${idx}` + runQS()), null, 2);
  const rows = await getJSON(`/api/field/${idx}/targets` + runQS());
  document.getElementById('fieldTargets').innerHTML = makeTable(rows.slice(0, 25), ['target_id','target_type','x_pix','y_pix','selected_for_photometry']);
}

async function loadSpectrum() {
  const targetId = document.getElementById('targetSelect').value;
  const data = await getJSON(`/api/spectrum/${encodeURIComponent(targetId)}` + runQS());
  drawPlot(data.rows || []);
  document.getElementById('spectrumTable').innerHTML = makeTable((data.rows || []).slice(0, 40), ['cwave_um','aperture_flux_uJy','aperture_flux_unc_uJy','psf_flux_uJy','psf_flux_unc_uJy','psf_fit_status','detector','observation_id']);
}

function makeTable(rows, cols) {
  if (!rows.length) return '<em>No rows</em>';
  return '<table><thead><tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map(r => '<tr>' + cols.map(c => `<td>${fmt(r[c])}</td>`).join('') + '</tr>').join('') + '</tbody></table>';
}

function fmt(v) {
  if (typeof v === 'number') return Math.abs(v) > 100 ? v.toFixed(1) : v.toFixed(4);
  if (v === null || v === undefined) return '';
  return String(v);
}

function drawPlot(rows) {
  const svg = document.getElementById('plot');
  svg.innerHTML = '';
  const W = 900, H = 620, m = {l:72,r:24,t:28,b:58};
  const good = rows.filter(r => Number.isFinite(r.cwave_um) && Number.isFinite(r.aperture_flux_uJy));
  if (!good.length) return;
  const psfGood = rows.filter(r => Number.isFinite(r.cwave_um) && Number.isFinite(r.psf_flux_uJy));
  const xs = good.map(r => r.cwave_um).concat(psfGood.map(r => r.cwave_um));
  const ys = good.map(r => r.aperture_flux_uJy).concat(psfGood.map(r => r.psf_flux_uJy));
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  const x = v => m.l + (v-xmin)/(xmax-xmin || 1)*(W-m.l-m.r);
  const y = v => H-m.b - (v-ymin)/(ymax-ymin || 1)*(H-m.t-m.b);
  add('line',{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,stroke:'#334e68'});
  add('line',{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,stroke:'#334e68'});
  add('text',{x:W/2,y:H-10,'text-anchor':'middle'},'wavelength um');
  add('text',{x:12,y:H/2,transform:`rotate(-90 12 ${H/2})`,'text-anchor':'middle'},'flux uJy');
  const sorted = [...good].sort((a,b) => a.cwave_um-b.cwave_um);
  const smooth = smoothRows(sorted, 7);
  if (smooth.length > 1) {
    add('polyline',{points:smooth.map(r => `${x(r.cwave_um)},${y(r.flux)}`).join(' '),fill:'none',stroke:'#111827','stroke-width':2.5,opacity:0.75});
  }
  for (const r of sorted) {
    add('circle',{cx:x(r.cwave_um),cy:y(r.aperture_flux_uJy),r:4.5,fill:r.target_id==='simp0136'?'#d97706':'#0077b6'});
  }
  for (const r of psfGood) {
    add('path',{d:`M ${x(r.cwave_um)-5} ${y(r.psf_flux_uJy)} L ${x(r.cwave_um)} ${y(r.psf_flux_uJy)-5} L ${x(r.cwave_um)+5} ${y(r.psf_flux_uJy)} L ${x(r.cwave_um)} ${y(r.psf_flux_uJy)+5} Z`,fill:'#7c3aed',opacity:0.75});
  }
  add('text',{x:m.l,y:18,fill:'#102a43'},`${good[0].target_id}: aperture ${good.length}, PSF ${psfGood.length}`);
  add('circle',{cx:W-210,cy:18,r:4.5,fill:'#d97706'});
  add('text',{x:W-198,y:22,fill:'#102a43'},'aperture');
  add('path',{d:`M ${W-120} 18 L ${W-115} 13 L ${W-110} 18 L ${W-115} 23 Z`,fill:'#7c3aed'});
  add('text',{x:W-104,y:22,fill:'#102a43'},'PSF');
  add('line',{x1:W-58,y1:18,x2:W-28,y2:18,stroke:'#111827','stroke-width':2.5});
  add('text',{x:W-24,y:22,fill:'#102a43'},'fit');
  function add(name, attrs, text) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (const [k,v] of Object.entries(attrs)) el.setAttribute(k,v);
    if (text) el.textContent = text;
    svg.appendChild(el);
  }
}
function smoothRows(rows, width) {
  if (rows.length < 3) return rows.map(r => ({cwave_um:r.cwave_um, flux:r.aperture_flux_uJy}));
  const half = Math.max(1, Math.floor(width/2));
  return rows.map((r, i) => {
    const lo = Math.max(0, i-half), hi = Math.min(rows.length, i+half+1);
    const vals = rows.slice(lo, hi).map(x => x.aperture_flux_uJy).sort((a,b)=>a-b);
    return {cwave_um:r.cwave_um, flux:vals[Math.floor(vals.length/2)]};
  });
}
init().catch(e => alert(e.stack || e));
</script>
</body>
</html>"""


def _live_html() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SPHEREx Live Workers</title>
  <style>
    :root {
      --bg: #030712;
      --panel: #07111f;
      --line: #123046;
      --text: #d8fbff;
      --muted: #7dd3fc;
      --cyan: #22d3ee;
      --pink: #f472b6;
      --amber: #f59e0b;
      --green: #22c55e;
      --grey: #94a3b8;
      --red: #fb7185;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        linear-gradient(90deg, rgba(34,211,238,.055) 1px, transparent 1px),
        linear-gradient(rgba(244,114,182,.045) 1px, transparent 1px),
        radial-gradient(circle at 70% 20%, rgba(34,211,238,.12), transparent 32%),
        var(--bg);
      background-size: 42px 42px, 42px 42px, auto;
      color: var(--text);
      font: 13px Inter, ui-sans-serif, system-ui, sans-serif;
      letter-spacing: 0;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(3,7,18,.86);
      position: sticky;
      top: 0;
      z-index: 2;
    }
    h1 { margin: 0; font-size: 18px; color: var(--cyan); text-shadow: 0 0 14px rgba(34,211,238,.42); }
    .stats { display: flex; gap: 12px; color: var(--muted); }
    .pill { border: 1px solid var(--line); padding: 5px 8px; background: rgba(7,17,31,.9); }
    .runbar { display: grid; grid-template-columns: repeat(8, minmax(110px, 1fr)); gap: 8px; padding: 12px 12px 0; }
    .runitem { border: 1px solid var(--line); background: rgba(7,17,31,.9); padding: 8px; }
    .runitem .k { color: #7dd3fc; font-size: 10px; text-transform: uppercase; }
    .runitem .v { color: #e0f2fe; font-size: 15px; margin-top: 3px; }
    .runprogress { grid-column: 1 / -1; height: 8px; border: 1px solid var(--line); background: #020617; }
    .runprogress div { height: 100%; width: 0; background: linear-gradient(90deg, #22d3ee, #f472b6, #22c55e); }
    main { display: grid; grid-template-columns: minmax(760px, 1.55fr) minmax(340px, .65fr); gap: 12px; padding: 12px; }
    .frames { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; align-content: start; }
    .frame {
      border: 1px solid var(--line);
      background: rgba(7,17,31,.82);
      box-shadow: 0 0 22px rgba(34,211,238,.08);
      min-width: 0;
    }
    .frame-head {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      padding: 7px 8px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 10px;
    }
    .stage { position: relative; aspect-ratio: 1 / 1; overflow: hidden; background: #000; }
    .stage img { width: 100%; height: 100%; display: block; object-fit: cover; }
    .stage svg { position: absolute; inset: 0; width: 100%; height: 100%; }
    .circle { fill: transparent; stroke-width: 1.6; vector-effect: non-scaling-stroke; }
    .halo { fill: transparent; stroke: #020617; stroke-width: 2.8; opacity: .58; vector-effect: non-scaling-stroke; }
    .queued { stroke: var(--grey); opacity: .78; }
    .active { stroke: var(--amber); opacity: 1; stroke-width: 2.2; filter: drop-shadow(0 0 5px var(--amber)); }
    .done { stroke: var(--green); opacity: .95; }
    .error { stroke: var(--red); opacity: 1; stroke-width: 2.2; }
    .simp { stroke: var(--pink); stroke-width: 2.8; }
    .meta { display: grid; grid-template-columns: repeat(2, 1fr); gap: 4px 9px; padding: 8px; color: #bae6fd; font-size: 11px; }
    .meta span { color: #e0f2fe; }
    .progress { height: 4px; background: #020617; border-top: 1px solid rgba(34,211,238,.16); }
    .progress div { height: 100%; background: linear-gradient(90deg, #22d3ee, #22c55e); width: 0; }
    .side { display: grid; gap: 12px; align-content: start; }
    .panel {
      border: 1px solid var(--line);
      background: rgba(7,17,31,.86);
      padding: 10px;
      min-width: 0;
    }
    h2 { margin: 0 0 8px; color: var(--pink); font-size: 14px; }
    #plot { width: 100%; height: 360px; border: 1px solid var(--line); background: rgba(2,6,23,.72); }
    table { width: 100%; border-collapse: collapse; color: #d8fbff; font-size: 12px; }
    th, td { border-bottom: 1px solid rgba(18,48,70,.75); padding: 5px; text-align: left; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre-wrap; color: #bae6fd; }
  </style>
</head>
<body>
<header>
  <h1>SPHEREx Live Worker Deck</h1>
	  <div class="stats">
	    <select class="pill" id="runSelect" onchange="switchRun()"></select>
	    <div class="pill" id="frameStat">frames --</div>
    <div class="pill" id="workerStat">workers --</div>
    <div class="pill" id="pointStat">spectra --</div>
  </div>
</header>
<section class="runbar" id="runbar"></section>
<main>
  <section class="frames" id="frames"></section>
  <aside class="side">
    <div class="panel">
      <h2>Developing Spectra</h2>
      <svg id="plot" viewBox="0 0 680 360"></svg>
    </div>
    <div class="panel">
      <h2>Active Targets</h2>
      <div id="activeTargets"></div>
    </div>
    <div class="panel">
      <h2>Status</h2>
      <div class="mono" id="raw"></div>
    </div>
  </aside>
</main>
<script>
const W = 2048;
const H = 2048;
let lastData = null;
let activeRun = new URLSearchParams(window.location.search).get('run') || '';

function runQS(extra) {
  const p = new URLSearchParams(extra || '');
  if (activeRun) p.set('run', activeRun);
  const s = p.toString();
  return s ? '?' + s : '';
}

async function getJSON(url) {
  const r = await fetch(url, {cache: 'no-store'});
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

async function tick() {
  try {
    await refreshRuns();
    const [live, run] = await Promise.all([
      getJSON('/api/live/status' + runQS('ts=' + Date.now())),
      getJSON('/api/run/status' + runQS('ts=' + Date.now()))
    ]);
    lastData = live;
    lastData.run = run;
    render(lastData);
  } catch (e) {
    document.getElementById('raw').textContent = e.stack || String(e);
  }
}

async function refreshRuns() {
  const runs = await getJSON('/api/runs' + runQS());
  const rs = document.getElementById('runSelect');
  if (!activeRun && runs.length) activeRun = runs[0].name;
  const currentOptions = Array.from(rs.options).map(o => o.value).join('\\n');
  const nextOptions = runs.map(r => r.name).join('\\n');
  if (currentOptions !== nextOptions) {
    rs.innerHTML = runs.map(r => `<option value="${r.name}">${r.name}</option>`).join('');
  }
  rs.value = activeRun;
}

function switchRun() {
  activeRun = document.getElementById('runSelect').value;
  tick();
}

function render(data) {
  document.getElementById('frameStat').textContent = `frames ${data.frame_count}/${data.frame_limit}`;
  document.getElementById('workerStat').textContent = `active ${data.active_frame_count}`;
  document.getElementById('pointStat').textContent = `spectra ${data.spectra_points.length}`;
  renderRunbar(data.run || {});
  document.getElementById('frames').innerHTML = data.frames.map(frameHTML).join('');
  document.getElementById('activeTargets').innerHTML = activeTargetsHTML(data);
  drawSpectra(data.spectra_points || []);
  document.getElementById('raw').textContent = JSON.stringify({
    run_dir: data.run_dir,
    db_path: data.db_path,
    frames: data.frames.map(f => ({
      image_id: f.image_id,
      status: f.status,
      worker: f.worker_name,
      targets: f.target_status_counts
    }))
  }, null, 2);
}

function renderRunbar(run) {
  const pct = Number.isFinite(Number(run.field_progress_percent)) ? Number(run.field_progress_percent) : 0;
  const proc = run.process && run.process.running ? 'running' : 'stopped';
  const pid = run.process && run.process.processes && run.process.processes[0] ? run.process.processes[0].pid : '';
  const perf = run.performance || {};
  const items = [
    ['phase', run.run_phase || ''],
    ['process', pid ? `${proc} ${pid}` : proc],
    ['fields', `${run.fields_completed || 0} done + ${run.live_frames_active || 0} active / ${run.field_total_estimate || '?'}`],
    ['active', run.live_frames_active || 0],
    ['shards', run.field_shards_with_measurements || 0],
    ['rate', perf.target_rate_per_wall_sec ? `${fmt(perf.target_rate_per_wall_sec)}/s` : '--'],
    ['core rate', perf.target_rate_per_core_sec ? `${fmt(perf.target_rate_per_core_sec)}/core/s` : '--'],
    ['targets', `${run.live_targets_done || 0} done`],
    ['assembly', run.assembly_mtime_iso ? run.assembly_mtime_iso.slice(11,19) : 'pending']
  ];
  document.getElementById('runbar').innerHTML =
    items.map(([k,v]) => `<div class="runitem"><div class="k">${escapeHtml(k)}</div><div class="v">${escapeHtml(v)}</div></div>`).join('') +
    `<div class="runprogress"><div style="width:${Math.max(0, Math.min(100, pct))}%"></div></div>`;
}

function frameHTML(f) {
  const targets = f.targets || [];
  const circles = targets.map(t => {
    const cls = `${t.status || 'queued'} ${t.target_id === 'simp0136' ? 'simp' : ''}`;
    const title = `${t.target_id} ${t.status || ''} G=${fmt(t.phot_g_mean_mag)}`;
    const cx = Number(t.x_pix);
    const cy = H - Number(t.y_pix);
    const r = t.target_id === 'simp0136' ? 26 : 15;
    return `<circle class="halo" cx="${cx}" cy="${cy}" r="${r}"></circle><circle class="circle ${cls}" cx="${cx}" cy="${cy}" r="${r}"><title>${escapeHtml(title)}</title></circle>`;
  }).join('');
  const counts = f.target_status_counts || {};
  const band = Number.isFinite(Number(f.cwave_um)) ? `${Number(f.cwave_um).toFixed(3)} um` : 'band pending';
  const progress = Number.isFinite(Number(f.progress_percent)) ? Number(f.progress_percent) : 0;
  const perf = f.performance || {};
  const done = counts.done || 0;
  const total = f.target_count || Object.values(counts).reduce((a,b) => a + Number(b || 0), 0);
  const overlayNote = f.target_overlay_truncated ? `${f.target_overlay_count}/${total} drawn` : `${targets.length}/${total} drawn`;
  const elapsed = Number.isFinite(Number(f.elapsed_sec)) ? fmtDuration(Number(f.elapsed_sec)) : '';
  const worker = f.worker_name || (f.status === 'done' ? 'complete' : '');
  const shortId = String(f.image_id || '').replace(/^level2_/, '').replace(/_spx_l2b-.*/, '');
  return `<article class="frame">
    <div class="frame-head"><div title="${escapeHtml(f.image_id || '')}">${escapeHtml(shortId)}</div><div>${escapeHtml(f.status || '')}</div></div>
    <div class="stage">
      <img src="/api/live/frame/${encodeURIComponent(f.image_id)}.jpg${runQS('ts=' + Math.floor(Date.now()/30000))}" alt="">
      <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${circles}</svg>
    </div>
    <div class="progress"><div style="width:${Math.max(0, Math.min(100, progress))}%"></div></div>
    <div class="meta">
      <div>worker <span>${escapeHtml(worker)}</span></div>
      <div>elapsed <span>${elapsed}</span></div>
      <div>rate <span>${Number.isFinite(Number(perf.target_rate_per_sec)) ? fmt(perf.target_rate_per_sec) + '/s' : ''}</span></div>
      <div>phot <span>${Number.isFinite(Number(perf.photometry_sec)) ? fmt(perf.photometry_sec) + 's' : ''}</span></div>
      <div>aper <span>${Number.isFinite(Number(perf.aperture_sec)) ? fmt(perf.aperture_sec) + 's' : ''}</span></div>
      <div>psf <span>${Number.isFinite(Number(perf.psf_sec)) ? fmt(perf.psf_sec) + 's' : ''}</span></div>
      <div>detector <span>D${fmt(f.detector)}</span></div>
      <div>obs <span>${escapeHtml(f.observation_id || '')}</span></div>
      <div>band <span>${band}</span></div>
      <div>sky <span>${escapeHtml(f.constellation || '')}</span></div>
      <div>width <span>${Number.isFinite(Number(f.cband_um)) ? Number(f.cband_um).toFixed(3) + ' um' : ''}</span></div>
      <div>targets <span>${done}/${total}</span></div>
      <div>rings <span>${escapeHtml(overlayNote)}</span></div>
      <div>queued <span>${counts.queued || 0}</span></div>
      <div>active <span>${counts.active || 0}</span></div>
      <div>done <span>${counts.done || 0}</span></div>
      <div>error <span>${counts.error || 0}</span></div>
    </div>
  </article>`;
}

function activeTargetsHTML(data) {
  const rows = [];
  for (const f of data.frames || []) {
    for (const t of f.targets || []) {
      if (t.status === 'active' || t.status === 'queued') {
        rows.push({image_id: f.image_id, ...t});
      }
    }
  }
  if (!rows.length) return '<div class="mono">No active queued targets. Showing recent completed frames.</div>';
  return '<table><thead><tr><th>Frame</th><th>Target</th><th>Status</th><th>G</th></tr></thead><tbody>' +
    rows.slice(0, 40).map(r => `<tr><td>${escapeHtml(r.image_id)}</td><td>${escapeHtml(r.target_id)}</td><td>${escapeHtml(r.status)}</td><td>${fmt(r.phot_g_mean_mag)}</td></tr>`).join('') +
    '</tbody></table>';
}

function drawSpectra(points) {
  const svg = document.getElementById('plot');
  svg.innerHTML = '';
  const good = points.filter(p => Number.isFinite(p.cwave_um) && Number.isFinite(p.aperture_flux_uJy));
  if (!good.length) return;
  const targets = [...new Set(good.map(p => p.target_id))].slice(0, 8);
  const rows = good.filter(p => targets.includes(p.target_id));
  const xs = rows.map(p => p.cwave_um);
  const ys = rows.map(p => p.aperture_flux_uJy);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const m = {l:54,r:16,t:16,b:38}, w = 680, h = 360;
  const x = v => m.l + (v-xmin)/(xmax-xmin || 1)*(w-m.l-m.r);
  const y = v => h-m.b - (v-ymin)/(ymax-ymin || 1)*(h-m.t-m.b);
  add('line',{x1:m.l,y1:h-m.b,x2:w-m.r,y2:h-m.b,stroke:'#22d3ee',opacity:.6});
  add('line',{x1:m.l,y1:m.t,x2:m.l,y2:h-m.b,stroke:'#22d3ee',opacity:.6});
  const colors = ['#f472b6','#22d3ee','#22c55e','#f59e0b','#a78bfa','#fb7185','#67e8f9','#bef264'];
  targets.forEach((target, i) => {
    const tr = rows.filter(p => p.target_id === target).sort((a,b)=>a.cwave_um-b.cwave_um);
    if (tr.length > 1) add('polyline',{points:tr.map(p => `${x(p.cwave_um)},${y(p.aperture_flux_uJy)}`).join(' '),fill:'none',stroke:colors[i],opacity:.72,'stroke-width':1.0});
    for (const p of tr) add('circle',{cx:x(p.cwave_um),cy:y(p.aperture_flux_uJy),r:2.0,fill:colors[i],opacity:p.fatal_flag_present ? .35 : .95});
  });
  function add(name, attrs) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (const [k,v] of Object.entries(attrs)) el.setAttribute(k,v);
    svg.appendChild(el);
  }
}

function fmt(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return '';
  return Math.abs(n) >= 100 ? n.toFixed(0) : n.toFixed(3);
}
function fmtDuration(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return m ? `${m}m ${String(r).padStart(2,'0')}s` : `${r}s`;
}
function escapeHtml(v) {
  return String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

tick();
setInterval(tick, 1500);
</script>
</body>
</html>"""


def _spectra_html() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SPHEREx Spectra Browser</title>
  <style>
    :root {
      --bg: #08111f;
      --panel: #0f1b2d;
      --line: #24364f;
      --text: #e5eefb;
      --muted: #93a4bb;
      --accent: #38bdf8;
      --ap: #22c55e;
      --psf: #c084fc;
      --bad: #f97316;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        linear-gradient(90deg, rgba(56,189,248,.055) 1px, transparent 1px),
        linear-gradient(rgba(244,114,182,.045) 1px, transparent 1px),
        radial-gradient(circle at 80% 8%, rgba(56,189,248,.13), transparent 34%),
        radial-gradient(circle at 12% 92%, rgba(244,114,182,.1), transparent 36%),
        var(--bg);
      background-size: 44px 44px, 44px 44px, auto, auto;
      color: var(--text);
      font: 13px system-ui, sans-serif;
      letter-spacing: 0;
    }
    header { display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; border-bottom: 1px solid var(--line); background: rgba(11,20,36,.9); box-shadow: 0 0 24px rgba(56,189,248,.12); }
    h1 { margin: 0; font-size: 18px; color: var(--accent); text-shadow: 0 0 16px rgba(56,189,248,.45); }
    main { display: grid; grid-template-columns: 360px minmax(700px, 1fr); gap: 12px; padding: 12px; }
    section { background: rgba(15,27,45,.9); border: 1px solid var(--line); border-radius: 6px; padding: 10px; min-width: 0; box-shadow: inset 0 0 0 1px rgba(56,189,248,.035), 0 0 20px rgba(2,6,23,.35); }
    label { display: block; color: var(--muted); font-size: 12px; margin: 8px 0 4px; }
    input, select, button { width: 100%; padding: 8px; background: #07111f; color: var(--text); border: 1px solid var(--line); border-radius: 4px; box-shadow: inset 0 0 12px rgba(56,189,248,.035); }
    button { cursor: pointer; }
    button:hover { border-color: var(--accent); }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .toggles { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin: 8px 0; align-items: end; }
    .toggles label { display: flex; align-items: center; gap: 6px; margin: 0; color: var(--text); }
    .toggles input { width: auto; }
    #targetList { height: calc(100vh - 300px); min-height: 380px; }
    #plot { width: 100%; height: 680px; background: rgba(5,11,20,.92); border: 1px solid #1d5f7a; border-radius: 6px; box-shadow: 0 0 28px rgba(56,189,248,.12), inset 0 0 24px rgba(244,114,182,.035); }
    .summary { display: grid; grid-template-columns: repeat(5, minmax(110px, 1fr)); gap: 8px; margin-bottom: 10px; }
    .tile { border: 1px solid var(--line); background: rgba(9,21,39,.95); padding: 8px; border-radius: 4px; box-shadow: inset 0 0 14px rgba(56,189,248,.035); }
    .tile .k { color: var(--muted); font-size: 11px; text-transform: uppercase; }
    .tile .v { font-size: 16px; margin-top: 3px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { border-bottom: 1px solid var(--line); padding: 5px; text-align: left; }
    th { color: #7dd3fc; font-weight: 600; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre-wrap; color: #bdd7f4; }
    .small { color: var(--muted); font-size: 12px; }
  </style>
</head>
<body>
<header>
  <h1>SPHEREx Spectra Browser</h1>
  <div class="small"><a style="color:#93c5fd" href="/live">Live workers</a> · <a style="color:#93c5fd" href="/">Field viewer</a></div>
</header>
<main>
  <section>
    <div class="row">
      <button onclick="selectTarget('simp0136')">SIMP</button>
      <button onclick="refreshAll()">Refresh</button>
    </div>
    <label for="runSelect">Run</label>
    <select id="runSelect" onchange="switchRun()"></select>
    <label for="filter">Filter targets</label>
    <input id="filter" placeholder="target id, gaia, simp..." oninput="scheduleTargetFetch()">
    <label for="sort">Sort</label>
    <select id="sort" onchange="fetchTargets(0)">
      <option value="measurements">Most measurements</option>
      <option value="snr">Max SNR</option>
      <option value="flux">Median aperture flux</option>
      <option value="flags">Fewest flags</option>
      <option value="id">Target id</option>
    </select>
    <div class="row" style="margin-top:8px">
      <button onclick="pageTargets(-1)">Prev</button>
      <button onclick="pageTargets(1)">Next</button>
    </div>
    <div id="targetPageInfo" class="small"></div>
    <label for="targetList">Targets</label>
    <select id="targetList" size="24" onchange="loadSelected()"></select>
    <div id="runSummary" class="mono"></div>
  </section>
  <div>
    <section>
      <div class="summary" id="targetSummary"></div>
      <div class="toggles">
        <label><input type="checkbox" id="showAperture" checked onchange="redraw()"> Aperture</label>
        <label><input type="checkbox" id="showPsf" checked onchange="redraw()"> PSF</label>
        <label><input type="checkbox" id="robustScale" checked onchange="redraw()"> Robust y</label>
        <label><input type="checkbox" id="ignoreFlaggedLine" checked onchange="redraw()"> Ignore flagged</label>
        <label>Curve
          <select id="curveMode" onchange="redraw()">
            <option value="spline" selected>Median spline</option>
            <option value="median">Running median</option>
            <option value="none">Points only</option>
          </select>
        </label>
      </div>
      <svg id="plot" viewBox="0 0 1100 680"></svg>
      <div class="small">Aperture/SAPM is the reference path. PSF is still experimental. Fatal-flagged points are dimmed orange.</div>
    </section>
    <section style="margin-top:12px">
      <div id="pointTable"></div>
    </section>
  </div>
</main>
<script>
let targets = [];
let targetTotal = 0;
let targetOffset = 0;
let targetLimit = 200;
let targetFetchTimer = null;
let current = null;
let currentRows = [];
let activeRun = new URLSearchParams(window.location.search).get('run') || '';

function runQS(extra) {
  const p = new URLSearchParams(extra || '');
  if (activeRun) p.set('run', activeRun);
  const s = p.toString();
  return s ? '?' + s : '';
}

async function getJSON(url) {
  const r = await fetch(url, {cache: 'no-store'});
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

async function refreshAll() {
  const runs = await getJSON('/api/runs' + runQS());
  const rs = document.getElementById('runSelect');
  if (!activeRun && runs.length) activeRun = runs[0].name;
  rs.innerHTML = runs.map(r => `<option value="${r.name}">${r.name} ${r.has_spectra ? '[' + (r.measurement_rows || 0) + ' rows]' : ''}</option>`).join('');
  rs.value = activeRun;
  document.getElementById('runSummary').textContent = JSON.stringify(await getJSON('/api/summary' + runQS()), null, 2);
  await fetchTargets(0);
  if (!current || !targets.some(t => t.target_id === current)) {
    const preferred = targets.some(t => t.target_id === 'ucs_0972') ? 'ucs_0972' : (targets[0]?.target_id || null);
    if (preferred) selectTarget(preferred);
  }
  else await selectTarget(current);
}

function switchRun() {
  activeRun = document.getElementById('runSelect').value;
  current = null;
  currentRows = [];
  refreshAll();
}

async function fetchTargets(offset) {
  targetOffset = Math.max(0, offset || 0);
  const sort = document.getElementById('sort').value;
  const q = document.getElementById('filter').value.trim();
  const data = await getJSON('/api/targets' + runQS(`limit=${targetLimit}&offset=${targetOffset}&sort=${encodeURIComponent(sort)}&q=${encodeURIComponent(q)}`));
  targets = data.rows || [];
  targetTotal = data.total || 0;
  targetOffset = data.offset || 0;
  targetLimit = data.limit || targetLimit;
  renderTargetList();
}

function scheduleTargetFetch() {
  clearTimeout(targetFetchTimer);
  targetFetchTimer = setTimeout(() => fetchTargets(0), 180);
}

function pageTargets(direction) {
  const next = Math.min(Math.max(0, targetOffset + direction * targetLimit), Math.max(0, targetTotal - 1));
  fetchTargets(next);
}

function renderTargetList() {
  const rows = targets;
  const list = document.getElementById('targetList');
  list.innerHTML = rows.map(t => `<option value="${escapeHtml(t.target_id)}">${escapeHtml(t.target_id)}  n=${fmt(t.n_measurements)}  flags=${fmt(t.flagged_measurements || 0)}  ${fmt(t.wavelength_min_um)}-${fmt(t.wavelength_max_um)}um</option>`).join('');
  if (current) list.value = current;
  const start = targetTotal ? targetOffset + 1 : 0;
  const end = Math.min(targetTotal, targetOffset + rows.length);
  document.getElementById('targetPageInfo').textContent = `${start}-${end} of ${targetTotal} targets`;
}

async function loadSelected() {
  const value = document.getElementById('targetList').value;
  if (value) await selectTarget(value);
}

async function selectTarget(targetId) {
  current = targetId;
  document.getElementById('targetList').value = targetId;
  const data = await getJSON('/api/spectrum/' + encodeURIComponent(targetId) + runQS());
  currentRows = data.rows || [];
  renderSummary();
  redraw();
  renderTable();
}

function renderSummary() {
  const summary = targets.find(t => t.target_id === current) || {};
  const rows = currentRows;
  const fatal = rows.filter(r => r.fatal_flag_present).length;
  const tiles = [
    ['Target', current || ''],
    ['Rows', rows.length],
    ['Wave', rows.length ? `${fmt(Math.min(...rows.map(r=>num(r.cwave_um))))}-${fmt(Math.max(...rows.map(r=>num(r.cwave_um))))} um` : ''],
    ['Median flux', fmt(summary.median_flux_uJy) + ' uJy'],
    ['Fatal', rows.length ? `${fatal}/${rows.length}` : '']
  ];
  document.getElementById('targetSummary').innerHTML = tiles.map(([k,v]) => `<div class="tile"><div class="k">${k}</div><div class="v">${escapeHtml(v)}</div></div>`).join('');
}

function redraw() {
  drawPlot(currentRows || []);
}

function drawPlot(rows) {
  const svg = document.getElementById('plot');
  svg.innerHTML = '';
  const W = 1100, H = 680, m = {l:78,r:30,t:28,b:58};
  const showAp = document.getElementById('showAperture').checked;
  const showPsf = document.getElementById('showPsf').checked;
  const ap = showAp ? rows.filter(r => Number.isFinite(num(r.cwave_um)) && Number.isFinite(num(r.aperture_flux_uJy))) : [];
  const psf = showPsf ? rows.filter(r => Number.isFinite(num(r.cwave_um)) && Number.isFinite(num(r.psf_flux_uJy))) : [];
  const all = ap.map(r => [num(r.cwave_um), num(r.aperture_flux_uJy)]).concat(psf.map(r => [num(r.cwave_um), num(r.psf_flux_uJy)]));
  if (!all.length) {
    add('text',{x:W/2,y:H/2,fill:'#93a4bb','text-anchor':'middle'},'No spectrum rows');
    return;
  }
  const xmin = 0.7, xmax = 5.05;
  let ys = all.filter(([x]) => x >= xmin && x <= xmax).map(([,y]) => y).filter(Number.isFinite);
  if (!ys.length) ys = all.map(([,y]) => y).filter(Number.isFinite);
  let ymin, ymax;
  if (document.getElementById('robustScale').checked && ys.length > 8) {
    const sorted = [...ys].sort((a,b)=>a-b);
    ymin = quantile(sorted, 0.02);
    ymax = quantile(sorted, 0.98);
  } else {
    ymin = Math.min(...ys);
    ymax = Math.max(...ys);
  }
  const pad = (ymax - ymin || 1) * 0.08;
  ymin -= pad; ymax += pad;
  const x = v => m.l + (v-xmin)/(xmax-xmin)*(W-m.l-m.r);
  const y = v => H-m.b - (v-ymin)/(ymax-ymin || 1)*(H-m.t-m.b);
  grid();
  if (ap.length) {
    const sorted = [...ap].sort((a,b)=>num(a.cwave_um)-num(b.cwave_um));
    const lineRows = document.getElementById('ignoreFlaggedLine').checked ? sorted.filter(r => !r.fatal_flag_present) : sorted;
    const curveMode = document.getElementById('curveMode').value;
    if (curveMode === 'median') {
      const smooth = runningMedianRows(lineRows, 9);
      if (smooth.length > 1) add('polyline',{points:smooth.map(r => `${x(r.cwave_um)},${y(r.flux)}`).join(' '),fill:'none',stroke:'#e5eefb','stroke-width':1.45,opacity:.72});
    } else if (curveMode === 'spline') {
      const curve = medianBinnedRows(lineRows, 72);
      if (curve.length > 1) add('path',{d:monotonePath(curve, x, y),fill:'none',stroke:'#e5eefb','stroke-width':1.7,opacity:.78});
    }
    for (const r of sorted) point(
      x(num(r.cwave_um)),
      y(num(r.aperture_flux_uJy)),
      r.fatal_flag_present ? '#f97316' : '#22c55e',
      r.fatal_flag_present ? .38 : .9,
      3.2,
      pointTitle(r, 'aperture_flux_uJy')
    );
  }
  if (psf.length) {
    for (const r of psf) diamond(x(num(r.cwave_um)), y(num(r.psf_flux_uJy)), r.fatal_flag_present ? '#f97316' : '#c084fc', r.fatal_flag_present ? .3 : .75, pointTitle(r, 'psf_flux_uJy'));
  }
  add('text',{x:m.l,y:18,fill:'#e5eefb'},`${current || ''}  aperture=${ap.length} psf=${psf.length}`);
  legend();

  function grid() {
    for (let v=1; v<=5; v++) {
      add('line',{x1:x(v),y1:m.t,x2:x(v),y2:H-m.b,stroke:'#24364f',opacity:.55});
      add('text',{x:x(v),y:H-22,fill:'#93a4bb','text-anchor':'middle'},String(v));
    }
    for (let i=0; i<5; i++) {
      const yy = m.t + i*(H-m.t-m.b)/4;
      add('line',{x1:m.l,y1:yy,x2:W-m.r,y2:yy,stroke:'#24364f',opacity:.55});
    }
    add('line',{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,stroke:'#93a4bb'});
    add('line',{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,stroke:'#93a4bb'});
    add('text',{x:W/2,y:H-8,fill:'#93a4bb','text-anchor':'middle'},'wavelength (um)');
    add('text',{x:18,y:H/2,fill:'#93a4bb',transform:`rotate(-90 18 ${H/2})`,'text-anchor':'middle'},'flux (uJy)');
  }
  function legend() {
    point(W-230, 22, '#22c55e', .9, 3.2); add('text',{x:W-218,y:26,fill:'#e5eefb'},'aperture');
    diamond(W-135, 22, '#c084fc', .8); add('text',{x:W-122,y:26,fill:'#e5eefb'},'PSF');
    point(W-70, 22, '#f97316', .45, 3.2); add('text',{x:W-58,y:26,fill:'#e5eefb'},'flag');
  }
  function point(cx, cy, color, opacity, r, title) {
    if (!Number.isFinite(cx) || !Number.isFinite(cy)) return;
    add('circle',{cx,cy,r,fill:color,opacity}, title);
  }
  function diamond(cx, cy, color, opacity, title) {
    if (!Number.isFinite(cx) || !Number.isFinite(cy)) return;
    add('path',{d:`M ${cx-4} ${cy} L ${cx} ${cy-4} L ${cx+4} ${cy} L ${cx} ${cy+4} Z`,fill:color,opacity}, title);
  }
  function add(name, attrs, text) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (const [k,v] of Object.entries(attrs)) el.setAttribute(k,v);
    if (text !== undefined && name === 'text') {
      el.textContent = text;
    } else if (text !== undefined) {
      const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      title.textContent = text;
      el.appendChild(title);
    }
    svg.appendChild(el);
  }
}

function renderTable() {
  const cols = ['cwave_um','aperture_flux_uJy','aperture_flux_unc_uJy','fatal_flag_present','detector','phot_g_mean_mag','bp_rp','pmra_masyr','pmdec_masyr','coordinate_propagation','observation_id','image_id','fits_file','input_file_path'];
  document.getElementById('pointTable').innerHTML = makeTable([...currentRows].sort((a,b)=>num(a.cwave_um)-num(b.cwave_um)).slice(0, 260), cols);
}

function pointTitle(r, fluxCol) {
  return [
    `target: ${current || r.target_id || ''}`,
    `wave: ${fmt(r.cwave_um)} um`,
    `flux: ${fmt(r[fluxCol])} uJy`,
    `detector: D${fmt(r.detector)}`,
    `obs: ${r.observation_id || ''}`,
    `image: ${r.image_id || ''}`,
    `fits: ${r.fits_file || fileName(r.input_file_path) || ''}`,
    `path: ${r.input_file_path || ''}`
  ].join('\\n');
}

function fileName(path) {
  if (!path) return '';
  return String(path).split('/').pop();
}

function makeTable(rows, cols) {
  if (!rows.length) return '<div class="small">No rows</div>';
  return '<table><thead><tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map(r => '<tr>' + cols.map(c => `<td>${escapeHtml(fmt(r[c]))}</td>`).join('') + '</tr>').join('') + '</tbody></table>';
}

function runningMedianRows(rows, width) {
  const half = Math.max(1, Math.floor(width/2));
  return rows.map((r, i) => {
    const vals = rows.slice(Math.max(0, i-half), Math.min(rows.length, i+half+1)).map(x => num(x.aperture_flux_uJy)).filter(Number.isFinite).sort((a,b)=>a-b);
    return {cwave_um:num(r.cwave_um), flux:median(vals)};
  });
}
function medianBinnedRows(rows, maxBins) {
  const clean = rows.map(r => ({x:num(r.cwave_um), y:num(r.aperture_flux_uJy)})).filter(r => Number.isFinite(r.x) && Number.isFinite(r.y));
  if (clean.length <= 2) return clean.map(r => ({cwave_um:r.x, flux:r.y}));
  const xmin = Math.min(...clean.map(r => r.x));
  const xmax = Math.max(...clean.map(r => r.x));
  const binCount = Math.max(8, Math.min(maxBins, Math.ceil(clean.length / 2)));
  const bins = Array.from({length: binCount}, () => []);
  for (const r of clean) {
    const idx = Math.max(0, Math.min(binCount - 1, Math.floor((r.x - xmin) / Math.max(xmax - xmin, 1e-9) * binCount)));
    bins[idx].push(r);
  }
  return bins.filter(b => b.length).map(b => {
    const xs = b.map(r => r.x).sort((a,b)=>a-b);
    const ys = b.map(r => r.y).sort((a,b)=>a-b);
    return {cwave_um:median(xs), flux:median(ys)};
  }).sort((a,b)=>a.cwave_um-b.cwave_um);
}
function monotonePath(rows, xScale, yScale) {
  const pts = rows.map(r => ({x:num(r.cwave_um), y:num(r.flux)})).filter(p => Number.isFinite(p.x) && Number.isFinite(p.y));
  if (!pts.length) return '';
  if (pts.length === 1) return `M ${xScale(pts[0].x)} ${yScale(pts[0].y)}`;
  const n = pts.length;
  const dx = [], dy = [], slope = [];
  for (let i=0; i<n-1; i++) {
    dx[i] = pts[i+1].x - pts[i].x;
    dy[i] = pts[i+1].y - pts[i].y;
    slope[i] = Math.abs(dx[i]) > 1e-12 ? dy[i] / dx[i] : 0;
  }
  const m = new Array(n).fill(0);
  m[0] = slope[0];
  m[n-1] = slope[n-2];
  for (let i=1; i<n-1; i++) {
    if (slope[i-1] * slope[i] <= 0) m[i] = 0;
    else {
      const w1 = 2 * dx[i] + dx[i-1];
      const w2 = dx[i] + 2 * dx[i-1];
      m[i] = (w1 + w2) / (w1 / slope[i-1] + w2 / slope[i]);
    }
  }
  let d = `M ${xScale(pts[0].x)} ${yScale(pts[0].y)}`;
  for (let i=0; i<n-1; i++) {
    const x0 = pts[i].x, x1 = pts[i+1].x;
    const y0 = pts[i].y, y1 = pts[i+1].y;
    const delta = (x1 - x0) / 3;
    const c1x = x0 + delta, c1y = y0 + m[i] * delta;
    const c2x = x1 - delta, c2y = y1 - m[i+1] * delta;
    d += ` C ${xScale(c1x)} ${yScale(c1y)}, ${xScale(c2x)} ${yScale(c2y)}, ${xScale(x1)} ${yScale(y1)}`;
  }
  return d;
}
function median(sorted) {
  if (!sorted.length) return NaN;
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
}
function quantile(sorted, q) {
  const idx = Math.min(sorted.length-1, Math.max(0, Math.floor(q*(sorted.length-1))));
  return sorted[idx];
}
function num(v) { const n = Number(v); return Number.isFinite(n) ? n : NaN; }
function flagRate(t) {
  const explicit = num(t.flagged_fraction);
  if (Number.isFinite(explicit)) return explicit;
  const flags = num(t.flagged_measurements);
  const total = num(t.total_measurements_for_flags || t.n_measurements);
  return Number.isFinite(flags) && Number.isFinite(total) && total > 0 ? flags / total : 0;
}
function fmt(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return v === null || v === undefined ? '' : String(v);
  if (Math.abs(n) >= 1000) return n.toExponential(3);
  if (Math.abs(n) >= 10) return n.toFixed(2);
  return n.toFixed(4);
}
function escapeHtml(v) {
  return String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

refreshAll().catch(e => {
  document.body.insertAdjacentHTML('beforeend', '<pre style="color:#fca5a5">' + escapeHtml(e.stack || String(e)) + '</pre>');
});
</script>
</body>
</html>"""
