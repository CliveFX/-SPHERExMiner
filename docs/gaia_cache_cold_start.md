# Gaia Cache Cold Start

This guide rebuilds the local Gaia DR3 cache used by the SPHEREx miner. The
goal is to avoid slow remote Gaia TAP queries during campaigns.

## Paths

Raw Gaia DR3 bulk files:

```text
/mnt/niroseti/spherex_cache/gaia/raw_download/gaia_dr3/GaiaSource_*.csv.gz
```

Local Gaia lite Parquet index:

```text
/mnt/niroseti/spherex_cache/gaia/parquet/dr3_source_lite/
```

Build logs and status:

```text
/mnt/niroseti/spherex_cache/gaia/logs/
```

## What The Index Contains

The lite cache is not a full Gaia database. It keeps the Gaia columns needed for
SPHEREx target selection:

```text
source_id
ra
dec
ref_epoch
pmra
pmdec
parallax
parallax_error
phot_g_mean_mag
phot_bp_mean_mag
phot_rp_mean_mag
bp_rp
ruwe
duplicated_source
astrometric_params_solved
```

The builder writes hive-style Parquet partitions by Gaia `source_id` HEALPix:

```text
gaia/parquet/dr3_source_lite/
  manifest.json
  hpx_level=<N>/
    hpx=<cell_id>/
      part-*.parquet
```

Queries use DuckDB in the production pipeline when the local index exists, and
fall back to remote Gaia TAP only if the local index is missing.

## Preflight

Confirm the raw files exist and the NAS is writable:

```bash
cd /home/clive/dev/NIROSETI_SPHEREx
find /mnt/niroseti/spherex_cache/gaia/raw_download/gaia_dr3 -maxdepth 1 -name 'GaiaSource_*.csv.gz' | wc -l
test -w /mnt/niroseti/spherex_cache && echo writable
```

The full DR3 source download should contain thousands of `GaiaSource_*.csv.gz`
files. The full-build wrapper refuses to start if it finds fewer than `3000`.

Check Python dependencies:

```bash
.venv/bin/spherex-mine doctor
.venv/bin/python - <<'PY'
import duckdb, pyarrow
print("duckdb", duckdb.__version__)
print("pyarrow", pyarrow.__version__)
PY
```

## Build The Full Lite Index

Use the wrapper so logs, status JSONL, PID tracking, and a final DuckDB smoke
check are produced:

```bash
tmux new-session -d -s gaia-lite-build \
  'cd /home/clive/dev/NIROSETI_SPHEREx && bash run_gaia_lite_full_build.sh'
```

Defaults:

```text
CACHE_ROOT=/mnt/niroseti/spherex_cache
HPX_LEVEL=3
MAX_ROWS_PER_FILE=2500000
MAX_BUFFERED_ROWS=5000000
PYARROW_NUM_THREADS=32
OMP_NUM_THREADS=32
```

Override them when needed:

```bash
HPX_LEVEL=3 PYARROW_NUM_THREADS=24 OMP_NUM_THREADS=24 \
  bash run_gaia_lite_full_build.sh
```

Do not use `sudo` for the normal build. The same user running the miner should
own and write the cache.

## Watch The Build

Attach to the build:

```bash
tmux attach -t gaia-lite-build
```

Detach without stopping:

```text
Ctrl-b, then d
```

Watch from another shell:

```bash
cd /home/clive/dev/NIROSETI_SPHEREx
bash watch_gaia_lite_build.sh
```

Status files are append-only JSONL under:

```text
/mnt/niroseti/spherex_cache/gaia/logs/gaia_lite_build_*.status.jsonl
```

The PID file is:

```text
/mnt/niroseti/spherex_cache/gaia/logs/gaia_lite_build.pid
```

## Smoke Check

The full-build wrapper runs this automatically on success:

```bash
.venv/bin/spherex-mine smoke-local-gaia-duckdb \
  --cache-root /mnt/niroseti/spherex_cache \
  --max-sources 25
```

Manual query check with DuckDB:

```bash
.venv/bin/spherex-mine query-local-gaia \
  --engine duckdb \
  --cache-root /mnt/niroseti/spherex_cache \
  --s-region "POLYGON 210.0 20.0 210.5 20.0 210.5 20.5 210.0 20.5" \
  --g-min 11 \
  --g-max 16 \
  --max-sources 100 \
  --output /mnt/niroseti/spherex_cache/gaia/query_smoke.parquet
```

Determinism/contract check:

```bash
.venv/bin/spherex-mine compare-local-gaia \
  --cache-root /mnt/niroseti/spherex_cache \
  --s-region "POLYGON 210.0 20.0 210.5 20.0 210.5 20.5 210.0 20.5" \
  --g-min 11 \
  --g-max 16 \
  --max-sources 100
```

Expected success fields include:

```text
columns_match_remote_contract: true
inside_conservative_bounds: true
magnitude_cuts_respected: true
source_ids_unique: true
deterministic_repeated_query: true
```

## How The Miner Uses It

The campaign path calls `query_gaia_for_s_region(...)`. That function checks:

```text
/mnt/niroseti/spherex_cache/gaia/parquet/dr3_source_lite/manifest.json
```

and Parquet partitions under:

```text
hpx_level=*/hpx=*/part-*.parquet
```

If the local lite index exists and the manifest says the raw file set is
complete, the miner queries local Parquet with DuckDB. Otherwise it falls back
to remote Gaia TAP and caches per-field query results.

## Rebuild Rules

Use `--overwrite` only for intentional rebuilds. The full-build wrapper already
passes it because cold start is expected to replace the lite index.

For a small dev build:

```bash
.venv/bin/spherex-mine build-gaia-lite \
  --cache-root /mnt/niroseti/spherex_cache \
  --limit-files 10 \
  --overwrite \
  --hpx-level 3
```

Do not use a partial dev build for production campaigns unless you explicitly
want remote Gaia fallback or incomplete sky coverage.

## Science Annotations

Keep mutable science products separate from the base Gaia Parquet cache. Use:

```text
/mnt/niroseti/spherex_cache/gaia/science/
```

keyed by `source_id`, `target_id`, `image_id`, or `run_id`. Do not mutate the
base Gaia DR3 lite Parquet files for classifications, spectra, or campaign
results.
