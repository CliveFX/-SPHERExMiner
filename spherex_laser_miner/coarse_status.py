from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any


_LOCK = threading.RLock()


def events_path(run_dir: Path) -> Path:
    return run_dir / "status_events.jsonl"


def summary_path(run_dir: Path) -> Path:
    return run_dir / "status_summary.json"


def reset_coarse_status(run_dir: Path, *, total_fields: int | None = None, worker_count: int | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        events_path(run_dir).write_text("", encoding="utf-8")
        summary = {
            "run_dir": str(run_dir),
            "started_at": time.time(),
            "updated_at": time.time(),
            "finished_at": None,
            "total_fields": total_fields,
            "worker_count": worker_count,
            "queued": int(total_fields or 0),
            "active": 0,
            "done": 0,
            "error": 0,
            "retry": 0,
            "measurements": 0,
            "targets_measured": 0,
            "last_event": None,
            "recent_events": [],
            "fields": {},
        }
        summary_path(run_dir).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        append_status_event(run_dir, "run_start", total_fields=total_fields, worker_count=worker_count)


def append_status_event(run_dir: Path, event: str, **payload: Any) -> None:
    row = {"time": time.time(), "event": event, **payload}
    with _LOCK:
        run_dir.mkdir(parents=True, exist_ok=True)
        with events_path(run_dir).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(_clean(row), sort_keys=True) + "\n")
        _update_summary(run_dir, row)


def read_coarse_summary(run_dir: Path, *, event_limit: int = 80) -> dict[str, Any]:
    path = summary_path(run_dir)
    if path.exists():
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    else:
        summary = {}
    if "recent_events" not in summary:
        summary["recent_events"] = tail_events(run_dir, limit=event_limit)
    return summary


def tail_events(run_dir: Path, *, limit: int = 100) -> list[dict[str, Any]]:
    path = events_path(run_dir)
    if not path.exists():
        return []
    rows: deque[dict[str, Any]] = deque(maxlen=limit)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(rows)


def _update_summary(run_dir: Path, row: dict[str, Any]) -> None:
    path = summary_path(run_dir)
    try:
        summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except json.JSONDecodeError:
        summary = {}
    fields = summary.setdefault("fields", {})
    event = str(row.get("event") or "")
    image_id = row.get("image_id")
    now = float(row.get("time") or time.time())
    if event == "run_start":
        summary.setdefault("started_at", now)
        summary["total_fields"] = row.get("total_fields")
        summary["worker_count"] = row.get("worker_count")
        summary["queued"] = int(row.get("total_fields") or 0)
    elif event == "field_start" and image_id:
        previous = dict(fields.get(str(image_id)) or {})
        previous.update({"status": "active", "started_at": now, "attempt": row.get("attempt"), "worker_name": row.get("worker_name")})
        fields[str(image_id)] = previous
    elif event == "field_retry" and image_id:
        summary["retry"] = int(summary.get("retry") or 0) + 1
        previous = dict(fields.get(str(image_id)) or {})
        previous.update({"status": "retry", "updated_at": now, "attempt": row.get("attempt"), "error": row.get("error")})
        fields[str(image_id)] = previous
    elif event in {"field_done", "field_error"} and image_id:
        previous = dict(fields.get(str(image_id)) or {})
        previous.update(
            {
                "status": "done" if event == "field_done" else "error",
                "finished_at": now,
                "elapsed_sec": row.get("elapsed_sec"),
                "targets_measured": row.get("targets_measured"),
                "measurements": row.get("measurements"),
                "error": row.get("error"),
            }
        )
        fields[str(image_id)] = previous
    elif event == "run_done":
        summary["finished_at"] = now
    elif event == "run_error":
        summary["finished_at"] = now
        summary["run_error"] = row.get("error")

    values = list(fields.values())
    summary["active"] = sum(1 for field in values if field.get("status") == "active")
    retrying = sum(1 for field in values if field.get("status") == "retry")
    summary["done"] = sum(1 for field in values if field.get("status") == "done")
    summary["error"] = sum(1 for field in values if field.get("status") == "error")
    total = summary.get("total_fields")
    summary["queued"] = max(0, int(total or 0) - summary["active"] - retrying - summary["done"] - summary["error"]) if total else 0
    summary["targets_measured"] = sum(int(field.get("targets_measured") or 0) for field in values)
    summary["measurements"] = sum(int(field.get("measurements") or 0) for field in values)
    started = float(summary.get("started_at") or now)
    elapsed = max(0.0, now - started)
    summary["elapsed_sec"] = elapsed
    summary["measurements_per_sec"] = summary["measurements"] / elapsed if elapsed > 0 else None
    summary["fields_per_sec"] = summary["done"] / elapsed if elapsed > 0 else None
    summary["updated_at"] = now
    summary["last_event"] = row
    recent = tail_events(run_dir, limit=80)
    summary["recent_events"] = recent
    path.write_text(json.dumps(_clean(summary), indent=2, sort_keys=True), encoding="utf-8")


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return _clean(value.item())
    except Exception:
        pass
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, Path):
        return str(value)
    return value
