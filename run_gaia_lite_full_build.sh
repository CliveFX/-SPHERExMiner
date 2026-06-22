#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT="${CACHE_ROOT:-/mnt/niroseti/spherex_cache}"
HPX_LEVEL="${HPX_LEVEL:-3}"
MAX_ROWS_PER_FILE="${MAX_ROWS_PER_FILE:-2500000}"
MAX_BUFFERED_ROWS="${MAX_BUFFERED_ROWS:-5000000}"
STATUS_INTERVAL_SEC="${STATUS_INTERVAL_SEC:-20}"
LOG_DIR="${LOG_DIR:-${CACHE_ROOT}/gaia/logs}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/gaia_lite_build_${RUN_ID}.log}"
STATUS_PATH="${STATUS_PATH:-${LOG_DIR}/gaia_lite_build_${RUN_ID}.status.jsonl}"
PID_PATH="${PID_PATH:-${LOG_DIR}/gaia_lite_build.pid}"

RAW_DIR="${CACHE_ROOT}/gaia/raw_download/gaia_dr3"
OUT_DIR="${CACHE_ROOT}/gaia/parquet/dr3_source_lite"

mkdir -p "${LOG_DIR}"

if [[ ! -d "${RAW_DIR}" ]]; then
  echo "Raw Gaia directory not found: ${RAW_DIR}" >&2
  exit 2
fi

raw_count="$(find "${RAW_DIR}" -maxdepth 1 -name 'GaiaSource_*.csv.gz' | wc -l)"
if [[ "${raw_count}" -lt 3000 ]]; then
  echo "Refusing full build: found only ${raw_count} GaiaSource files in ${RAW_DIR}" >&2
  exit 2
fi

echo "Starting Gaia lite full build"
echo "  cache_root=${CACHE_ROOT}"
echo "  raw_files=${raw_count}"
echo "  out_dir=${OUT_DIR}"
echo "  hpx_level=${HPX_LEVEL}"
echo "  max_rows_per_file=${MAX_ROWS_PER_FILE}"
echo "  max_buffered_rows=${MAX_BUFFERED_ROWS}"
echo "  log=${LOG_PATH}"
echo "  status=${STATUS_PATH}"
echo "  pid=${PID_PATH}"

export PYARROW_NUM_THREADS="${PYARROW_NUM_THREADS:-32}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"

start_epoch="$(date +%s)"

(
  .venv/bin/spherex-mine build-gaia-lite \
    --cache-root "${CACHE_ROOT}" \
    --overwrite \
    --hpx-level "${HPX_LEVEL}" \
    --max-rows-per-file "${MAX_ROWS_PER_FILE}" \
    --max-buffered-rows "${MAX_BUFFERED_ROWS}"
) >"${LOG_PATH}" 2>&1 &

builder_pid="$!"
printf '%s\n' "${builder_pid}" >"${PID_PATH}"

status_loop() {
  while kill -0 "${builder_pid}" 2>/dev/null; do
    now_epoch="$(date +%s)"
    elapsed_sec="$((now_epoch - start_epoch))"
    out_bytes="$(
      find "${OUT_DIR}" -type f -name '*.parquet' -printf '%s\n' 2>/dev/null \
        | awk '{s += $1} END {printf "%.0f", s + 0}'
    )"
    parquet_files="$(find "${OUT_DIR}" -type f -name '*.parquet' 2>/dev/null | wc -l)"
    hpx_dirs="$(find "${OUT_DIR}" -mindepth 2 -maxdepth 2 -type d -path '*/hpx=*' 2>/dev/null | wc -l)"
    raw_done_hint="$(grep -c 'GaiaSource_' "${LOG_PATH}" 2>/dev/null || true)"
    ps_line="$(ps -p "${builder_pid}" -o pid=,pcpu=,pmem=,rss=,etime=,stat= 2>/dev/null | awk '{$1=$1; print}' || true)"
    latest_manifest_complete="null"
    latest_manifest_rows="null"
    if [[ -f "${OUT_DIR}/manifest.json" ]]; then
      latest_manifest_complete="$(
        .venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("complete_raw_file_set"))' "${OUT_DIR}/manifest.json" 2>/dev/null || printf 'null'
      )"
      latest_manifest_rows="$(
        .venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("row_count"))' "${OUT_DIR}/manifest.json" 2>/dev/null || printf 'null'
      )"
    fi
    mb_written="$((out_bytes / 1024 / 1024))"
    mb_per_sec="0"
    if [[ "${elapsed_sec}" -gt 0 ]]; then
      mb_per_sec="$((mb_written / elapsed_sec))"
    fi

    .venv/bin/python - "${STATUS_PATH}" "${builder_pid}" "${elapsed_sec}" "${raw_count}" "${raw_done_hint}" "${out_bytes}" "${mb_written}" "${mb_per_sec}" "${parquet_files}" "${hpx_dirs}" "${latest_manifest_complete}" "${latest_manifest_rows}" "${ps_line}" <<'PY'
import ast
import json, time
import sys
(
    status_path,
    pid,
    elapsed_sec,
    raw_count,
    raw_done_hint,
    out_bytes,
    mb_written,
    mb_per_sec,
    parquet_files,
    hpx_dirs,
    latest_manifest_complete,
    latest_manifest_rows,
    ps_line,
) = sys.argv[1:]
def parse_nullable(value):
    if value == "null":
        return None
    if value in {"True", "False"}:
        return value == "True"
    try:
        return ast.literal_eval(value)
    except Exception:
        return value
status = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "pid": int(pid),
    "elapsed_sec": int(elapsed_sec),
    "raw_files_total": int(raw_count),
    "raw_done_log_hint": int(raw_done_hint),
    "output_bytes": int(out_bytes),
    "output_mib": int(mb_written),
    "output_mib_per_sec_since_start": int(mb_per_sec),
    "parquet_files": int(parquet_files),
    "hpx_dirs": int(hpx_dirs),
    "manifest_complete_raw_file_set": parse_nullable(latest_manifest_complete),
    "manifest_row_count": parse_nullable(latest_manifest_rows),
    "process": ps_line,
}
with open(status_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(status, sort_keys=True) + "\n")
print(json.dumps(status, sort_keys=True))
PY
    echo "--- recent build log ---"
    tail -20 "${LOG_PATH}" || true
    sleep "${STATUS_INTERVAL_SEC}"
  done
}

status_loop &
status_pid="$!"

set +e
wait "${builder_pid}"
exit_code="$?"
set -e

kill "${status_pid}" 2>/dev/null || true
wait "${status_pid}" 2>/dev/null || true
rm -f "${PID_PATH}"

end_epoch="$(date +%s)"
elapsed_sec="$((end_epoch - start_epoch))"

echo "Gaia lite build exited with code ${exit_code} after ${elapsed_sec}s"
echo "Final log tail:"
tail -80 "${LOG_PATH}" || true

if [[ "${exit_code}" -eq 0 ]]; then
  echo "Running DuckDB/PyArrow smoke check against completed index..."
  .venv/bin/spherex-mine smoke-local-gaia-duckdb --cache-root "${CACHE_ROOT}" --max-sources 25 | tee -a "${LOG_PATH}"
fi

exit "${exit_code}"
