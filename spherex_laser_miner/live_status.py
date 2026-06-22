from __future__ import annotations

import sqlite3
import time
import hashlib
import threading
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
  image_id TEXT PRIMARY KEY,
  worker_name TEXT,
  status TEXT NOT NULL,
  input_file_path TEXT,
  detector INTEGER,
  observation_id TEXT,
  cwave_um REAL,
  cband_um REAL,
  constellation TEXT,
  started_at REAL,
  updated_at REAL,
  finished_at REAL,
  error TEXT
);
CREATE TABLE IF NOT EXISTS targets (
  image_id TEXT NOT NULL,
  target_id TEXT NOT NULL,
  target_type TEXT,
  x_pix REAL,
  y_pix REAL,
  phot_g_mean_mag REAL,
  cwave_um REAL,
  status TEXT NOT NULL,
  aperture_flux_uJy REAL,
  psf_flux_uJy REAL,
  fatal_flag_present INTEGER,
  updated_at REAL,
  PRIMARY KEY (image_id, target_id)
);
CREATE TABLE IF NOT EXISTS spectra_points (
  target_id TEXT NOT NULL,
  image_id TEXT NOT NULL,
  cwave_um REAL,
  aperture_flux_uJy REAL,
  psf_flux_uJy REAL,
  fatal_flag_present INTEGER,
  updated_at REAL,
  PRIMARY KEY (target_id, image_id)
);
CREATE INDEX IF NOT EXISTS idx_frames_status_updated ON frames(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_targets_image ON targets(image_id, status);
CREATE INDEX IF NOT EXISTS idx_spectra_target_wave ON spectra_points(target_id, cwave_um);
"""

_DB_LOCK = threading.RLock()


def db_path(run_dir: Path) -> Path:
    digest = hashlib.sha1(str(run_dir).encode("utf-8")).hexdigest()[:12]
    return Path("/tmp") / "spherex_live_status" / f"{digest}.sqlite"


def reset_live_status(run_dir: Path) -> None:
    path = db_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with _DB_LOCK, _connect(path) as con:
            con.executescript(SCHEMA)
            con.execute("DELETE FROM spectra_points")
            con.execute("DELETE FROM targets")
            con.execute("DELETE FROM frames")
    except sqlite3.Error:
        pass


def init_live_status(run_dir: Path) -> None:
    path = db_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with _DB_LOCK, _connect(path) as con:
            con.executescript(SCHEMA)
    except sqlite3.Error:
        pass


def mark_frame(
    run_dir: Path,
    *,
    image_id: str,
    status: str,
    worker_name: str | None = None,
    input_file_path: str | None = None,
    detector: int | None = None,
    observation_id: str | None = None,
    cwave_um: float | None = None,
    cband_um: float | None = None,
    constellation: str | None = None,
    error: str | None = None,
) -> None:
    now = time.time()
    finished_at = now if status in {"done", "error"} else None
    try:
        with _DB_LOCK, _connect(db_path(run_dir)) as con:
            con.executescript(SCHEMA)
            con.execute(
            """
            INSERT INTO frames (
              image_id, worker_name, status, input_file_path, detector, observation_id,
              cwave_um, cband_um, constellation, started_at, updated_at, finished_at, error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(image_id) DO UPDATE SET
              worker_name=COALESCE(excluded.worker_name, frames.worker_name),
              status=excluded.status,
              input_file_path=COALESCE(excluded.input_file_path, frames.input_file_path),
              detector=COALESCE(excluded.detector, frames.detector),
              observation_id=COALESCE(excluded.observation_id, frames.observation_id),
              cwave_um=COALESCE(excluded.cwave_um, frames.cwave_um),
              cband_um=COALESCE(excluded.cband_um, frames.cband_um),
              constellation=COALESCE(excluded.constellation, frames.constellation),
              updated_at=excluded.updated_at,
              finished_at=COALESCE(excluded.finished_at, frames.finished_at),
              error=excluded.error
            """,
                (
                    image_id,
                    worker_name,
                    status,
                    input_file_path,
                    detector,
                    observation_id,
                    cwave_um,
                    cband_um,
                    constellation,
                    now,
                    now,
                    finished_at,
                    error,
                ),
            )
    except sqlite3.Error:
        pass


def mark_target(run_dir: Path, *, image_id: str, target: dict[str, Any], status: str) -> None:
    now = time.time()
    try:
        with _DB_LOCK, _connect(db_path(run_dir)) as con:
            con.executescript(SCHEMA)
            con.execute(
            """
            INSERT INTO targets (
              image_id, target_id, target_type, x_pix, y_pix, phot_g_mean_mag,
              cwave_um, status, aperture_flux_uJy, psf_flux_uJy, fatal_flag_present, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(image_id, target_id) DO UPDATE SET
              target_type=COALESCE(excluded.target_type, targets.target_type),
              x_pix=COALESCE(excluded.x_pix, targets.x_pix),
              y_pix=COALESCE(excluded.y_pix, targets.y_pix),
              phot_g_mean_mag=COALESCE(excluded.phot_g_mean_mag, targets.phot_g_mean_mag),
              cwave_um=COALESCE(excluded.cwave_um, targets.cwave_um),
              status=excluded.status,
              aperture_flux_uJy=COALESCE(excluded.aperture_flux_uJy, targets.aperture_flux_uJy),
              psf_flux_uJy=COALESCE(excluded.psf_flux_uJy, targets.psf_flux_uJy),
              fatal_flag_present=COALESCE(excluded.fatal_flag_present, targets.fatal_flag_present),
              updated_at=excluded.updated_at
            """,
                (
                    image_id,
                    str(target.get("target_id")),
                    target.get("target_type"),
                    _optional_float(target.get("x_pix")),
                    _optional_float(target.get("y_pix")),
                    _optional_float(target.get("phot_g_mean_mag")),
                    _optional_float(target.get("cwave_um")),
                    status,
                    _optional_float(target.get("aperture_flux_uJy")),
                    _optional_float(target.get("psf_flux_uJy")),
                    _optional_int(target.get("fatal_flag_present")),
                    now,
                ),
            )
            if status == "done":
                con.execute(
                """
                INSERT INTO spectra_points (
                  target_id, image_id, cwave_um, aperture_flux_uJy, psf_flux_uJy,
                  fatal_flag_present, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(target_id, image_id) DO UPDATE SET
                  cwave_um=excluded.cwave_um,
                  aperture_flux_uJy=excluded.aperture_flux_uJy,
                  psf_flux_uJy=excluded.psf_flux_uJy,
                  fatal_flag_present=excluded.fatal_flag_present,
                  updated_at=excluded.updated_at
                """,
                    (
                        str(target.get("target_id")),
                        image_id,
                        _optional_float(target.get("cwave_um")),
                        _optional_float(target.get("aperture_flux_uJy")),
                        _optional_float(target.get("psf_flux_uJy")),
                        _optional_int(target.get("fatal_flag_present")),
                        now,
                    ),
                )
    except sqlite3.Error:
        pass


def _connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=3)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=3000")
    con.execute("PRAGMA journal_mode=DELETE")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def _optional_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(bool(value))
    except Exception:
        return None
