#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CACHE_ROOT="${CACHE_ROOT:-/mnt/niroseti/spherex_cache}"
LOG_ROOT="${LOG_ROOT:-$CACHE_ROOT/logs}"
mkdir -p "$LOG_ROOT"

BASE_RUN="${BASE_RUN:-arcturus_deep20k_f500_baseline_gpu}"
CAMPAIGN_ID="${CAMPAIGN_ID:-arcturus_deep20k_mixed_lasers}"
INJECT_STRENGTH="${INJECT_STRENGTH:-5}"
INJECTED_RUN="${INJECTED_RUN:-arcturus_deep20k_f500_injected_s${INJECT_STRENGTH}_gpu}"
CAMPAIGN_ROOT="$CACHE_ROOT/injection_campaigns/${CAMPAIGN_ID}_s${INJECT_STRENGTH}"
PLAN_PATH="$CACHE_ROOT/injection_campaigns/$CAMPAIGN_ID/injection_plan.json"
MANIFEST_PATH="$CAMPAIGN_ROOT/injection_manifest.json"
OVERRIDES_PATH="$CAMPAIGN_ROOT/path_overrides.json"
LOG_PATH="$LOG_ROOT/${INJECTED_RUN}_pipeline.log"

FIELDS="${FIELDS:-500}"
TARGETS="${TARGETS:-20000}"
WORKERS="${WORKERS:-24}"
GAIA_G_MIN="${GAIA_G_MIN:-5}"
GAIA_G_MAX="${GAIA_G_MAX:-17}"
WARP_DEVICES="${WARP_DEVICES:-cuda:0,cuda:1,cuda:2}"
RETRIES="${RETRIES:-2}"
TARGETS_PER_CELL="${TARGETS_PER_CELL:-5}"
MIN_MEASUREMENTS="${MIN_MEASUREMENTS:-80}"
MAX_FRAMES_PER_INJECTION="${MAX_FRAMES_PER_INJECTION:-80}"
MIN_SNR="${MIN_SNR:-5}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_PATH"
}

run_step() {
  log "START $*"
  "$@" 2>&1 | tee -a "$LOG_PATH"
  log "DONE $*"
}

log "Arcturus deep injection/recovery pipeline"
log "baseline_run=$BASE_RUN injected_run=$INJECTED_RUN campaign_root=$CAMPAIGN_ROOT"
log "fields=$FIELDS targets=$TARGETS workers=$WORKERS gaia_g=$GAIA_G_MIN..$GAIA_G_MAX devices=$WARP_DEVICES"
log "simple status: http://0.0.0.0:8765/simple-status?run=$BASE_RUN then run=$INJECTED_RUN"

run_step .venv/bin/spherex-mine run-depth-test \
  --target arcturus \
  --run-name "$BASE_RUN" \
  --limit-fields "$FIELDS" \
  --max-gaia-sources "$TARGETS" \
  --gaia-g-min "$GAIA_G_MIN" \
  --gaia-g-max "$GAIA_G_MAX" \
  --max-field-workers "$WORKERS" \
  --photometry-backend warp_calibrated \
  --warp-devices "$WARP_DEVICES" \
  --status-mode jsonl \
  --max-field-retries "$RETRIES" \
  --cache-root "$CACHE_ROOT"

run_step .venv/bin/python tools/make_mixed_laser_injection_plan.py \
  --run-dir "$CACHE_ROOT/runs/$BASE_RUN" \
  --campaign-id "$CAMPAIGN_ID" \
  --output-root "$CACHE_ROOT/injection_campaigns" \
  --targets-per-cell "$TARGETS_PER_CELL" \
  --strengths-sigma "3,5,8,12,20" \
  --min-measurements "$MIN_MEASUREMENTS" \
  --max-frames-per-injection "$MAX_FRAMES_PER_INJECTION"

run_step .venv/bin/python tools/run_injection_plan.py \
  --plan "$PLAN_PATH" \
  --cache-root "$CACHE_ROOT" \
  --strength-sigma "$INJECT_STRENGTH" \
  --overwrite

run_step .venv/bin/spherex-mine run-depth-test \
  --target arcturus \
  --run-name "$INJECTED_RUN" \
  --limit-fields "$FIELDS" \
  --max-gaia-sources "$TARGETS" \
  --gaia-g-min "$GAIA_G_MIN" \
  --gaia-g-max "$GAIA_G_MAX" \
  --max-field-workers "$WORKERS" \
  --photometry-backend warp_calibrated \
  --warp-devices "$WARP_DEVICES" \
  --status-mode jsonl \
  --max-field-retries "$RETRIES" \
  --path-overrides "$OVERRIDES_PATH" \
  --cache-root "$CACHE_ROOT"

run_step .venv/bin/python tools/classify_paired_delta_matched_filter.py \
  --baseline-run-dir "$CACHE_ROOT/runs/$BASE_RUN" \
  --injected-run-dir "$CACHE_ROOT/runs/$INJECTED_RUN" \
  --plan "$PLAN_PATH" \
  --output-dir "$CACHE_ROOT/runs/$INJECTED_RUN/classifier_paired_delta" \
  --min-snr "$MIN_SNR"

run_step .venv/bin/python tools/score_injection_recovery.py \
  --manifest "$MANIFEST_PATH" \
  --candidates "$CACHE_ROOT/runs/$INJECTED_RUN/classifier_paired_delta/matched_filter_candidates.parquet" \
  --output-dir "$CACHE_ROOT/runs/$INJECTED_RUN/recovery_score_mixed_lasers" \
  --min-snr "$MIN_SNR"

log "COMPLETE Arcturus deep injection/recovery pipeline"
