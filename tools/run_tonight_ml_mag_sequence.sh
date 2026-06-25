#!/usr/bin/env bash
set -u -o pipefail

ROOT="/home/clive/dev/NIROSETI_SPHEREx"
CONFIG="$ROOT/configs/campaign_with_ml_transformer.yaml"
CACHE_ROOT="/mnt/niroseti/spherex_cache"
PY="$ROOT/.venv/bin/python"
RUNNER="$ROOT/tools/run_visible_sky_injection_campaign.py"

LIMIT_FIELDS="${LIMIT_FIELDS:-500}"
MAX_GAIA_SOURCES="${MAX_GAIA_SOURCES:-3000}"
MAX_FIELD_WORKERS="${MAX_FIELD_WORKERS:-24}"
WARP_DEVICES="${WARP_DEVICES:-cuda:0,cuda:1,cuda:2}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

run_bin() {
  local name="$1"
  local g_min="$2"
  local g_max="$3"
  local prefix="cv_june_${name}_f${LIMIT_FIELDS}_n${MAX_GAIA_SOURCES}_ml_tonight"
  local campaign_dir="$CACHE_ROOT/campaigns/$prefix"

  mkdir -p "$campaign_dir"
  echo
  echo "##### $(date -Is) starting $prefix G=$g_min-$g_max max_gaia_sources=$MAX_GAIA_SOURCES #####"
  echo

  cd "$ROOT" || return 1
  $PY "$RUNNER" \
    --config "$CONFIG" \
    --campaign-prefix "$prefix" \
    --limit-fields "$LIMIT_FIELDS" \
    --max-gaia-sources "$MAX_GAIA_SOURCES" \
    --gaia-g-min "$g_min" \
    --gaia-g-max "$g_max" \
    --max-field-workers "$MAX_FIELD_WORKERS" \
    --warp-devices "$WARP_DEVICES" \
    $EXTRA_ARGS \
    2>&1 | tee -a "$campaign_dir/campaign_stdout.log"

  local rc="${PIPESTATUS[0]}"
  echo
  echo "##### $(date -Is) finished $prefix rc=$rc #####"
  echo
  return "$rc"
}

failures=0

run_bin "g11_16" "11.0" "16.0" || failures=$((failures + 1))
run_bin "g8_11" "8.0" "11.0" || failures=$((failures + 1))
run_bin "g5_8" "5.0" "8.0" || failures=$((failures + 1))

echo "Magnitude sequence complete with $failures failed campaign command(s)."
exit "$failures"
