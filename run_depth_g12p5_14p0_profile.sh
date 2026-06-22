#!/usr/bin/env bash
set -uo pipefail
cd /home/clive/dev/NIROSETI_SPHEREx || exit 100
log=/mnt/niroseti/spherex_cache/runs/smoke_simp_field/depth_g12p5_14p0_profile.log
{
  printf 'started %s pid=%s\n' "$(date --iso-8601=seconds)" "$$"
  .venv/bin/spherex-mine run-depth-test \
    --limit-fields 30 \
    --max-gaia-sources 100 \
    --max-field-workers 24 \
    --gaia-g-min 12.5 \
    --gaia-g-max 14.0 \
    --cache-root /mnt/niroseti/spherex_cache
  code=$?
  printf 'finished %s exit=%s\n' "$(date --iso-8601=seconds)" "$code"
  exit "$code"
} >>"$log" 2>&1
