#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT="${CACHE_ROOT:-/mnt/niroseti/spherex_cache}"
LOG_DIR="${LOG_DIR:-${CACHE_ROOT}/gaia/logs}"
PID_PATH="${PID_PATH:-${LOG_DIR}/gaia_lite_build.pid}"
STATUS_INTERVAL_SEC="${STATUS_INTERVAL_SEC:-20}"
RUN_ID="${RUN_ID:-manual_watch_$(date -u +%Y%m%dT%H%M%SZ)}"
STATUS_PATH="${STATUS_PATH:-${LOG_DIR}/gaia_lite_build_${RUN_ID}.status.jsonl}"
OUT_DIR="${CACHE_ROOT}/gaia/parquet/dr3_source_lite"

if [[ ! -f "${PID_PATH}" ]]; then
  echo "PID file not found: ${PID_PATH}" >&2
  exit 2
fi

builder_pid="$(tr -dc '0-9' < "${PID_PATH}")"
if [[ -z "${builder_pid}" ]] || ! kill -0 "${builder_pid}" 2>/dev/null; then
  echo "Builder PID is not running: ${builder_pid:-missing}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
start_epoch="$(date +%s)"

echo "Watching Gaia lite build pid=${builder_pid}"
echo "  cache_root=${CACHE_ROOT}"
echo "  out_dir=${OUT_DIR}"
echo "  status=${STATUS_PATH}"

while kill -0 "${builder_pid}" 2>/dev/null; do
  now_epoch="$(date +%s)"
  elapsed_sec="$((now_epoch - start_epoch))"
  parquet_files="$(find "${OUT_DIR}" -type f -name '*.parquet' 2>/dev/null | wc -l)"
  hpx_dirs="$(find "${OUT_DIR}" -mindepth 2 -maxdepth 2 -type d -path '*/hpx=*' 2>/dev/null | wc -l)"
  size_bytes="$(
    find "${OUT_DIR}" -type f -name '*.parquet' -printf '%s\n' 2>/dev/null \
      | awk '{s += $1} END {printf "%.0f", s + 0}'
  )"
  ps_line="$(ps -p "${builder_pid}" -o pid=,pcpu=,pmem=,rss=,etime=,stat= 2>/dev/null | awk '{$1=$1; print}' || true)"
  latest_file="$(
    find "${OUT_DIR}" -type f -name '*.parquet' -printf '%T@ %s %p\n' 2>/dev/null \
      | sort -n \
      | tail -1 \
      | cut -d' ' -f2-
  )"
  size_mib="$((size_bytes / 1024 / 1024))"

  .venv/bin/python - "${STATUS_PATH}" "${builder_pid}" "${elapsed_sec}" "${parquet_files}" "${hpx_dirs}" "${size_bytes}" "${size_mib}" "${ps_line}" "${latest_file}" <<'PY'
import json
import sys
import time

status_path, pid, elapsed, parquet_files, hpx_dirs, size_bytes, size_mib, ps_line, latest_file = sys.argv[1:]
status = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "pid": int(pid),
    "watch_elapsed_sec": int(elapsed),
    "parquet_files": int(parquet_files),
    "hpx_dirs": int(hpx_dirs),
    "output_bytes": int(size_bytes),
    "output_mib": int(size_mib),
    "process": ps_line,
    "latest_file": latest_file,
}
with open(status_path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(status, sort_keys=True) + "\n")
print(json.dumps(status, sort_keys=True), flush=True)
PY
  sleep "${STATUS_INTERVAL_SEC}"
done

echo "Builder PID ${builder_pid} is no longer running."
