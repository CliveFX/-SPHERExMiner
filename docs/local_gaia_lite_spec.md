# Local Gaia Lite Index Spec

## Goal

Replace slow remote Gaia TAP field queries with a local, reproducible query path over the downloaded Gaia DR3 `gaia_source` bulk files.

The first implementation should be correct, inspectable, and good enough for SPHEREx field target selection. It does not need to be a general Gaia database.

## Inputs

Raw Gaia files:

```text
/mnt/niroseti/spherex_cache/gaia/raw_download/gaia_dr3/GaiaSource_*.csv.gz
```

Manifest and expected sizes:

```text
/mnt/niroseti/spherex_cache/gaia/manifests/gaia_dr3_urls.txt
/mnt/niroseti/spherex_cache/gaia/manifests/gaia_dr3_urls.sizes.tsv
```

Existing remote query contract:

```python
query_gaia_for_s_region(
    s_region: str,
    cache_path: Path,
    max_sources: int = 500,
    g_min: float = 8.0,
    g_max: float = 19.0,
) -> pandas.DataFrame
```

Current required columns:

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

## Output Layout

Build a local lite Parquet dataset:

```text
/mnt/niroseti/spherex_cache/gaia/parquet/dr3_source_lite/
  manifest.json
  hp_level=<N>/
    hp=<cell_id>/
      part-*.parquet
```

Recommended first HEALPix-like level:

```text
hp_level = 6
```

The implementation may use a simple deterministic sky grid instead of true HEALPix if `healpy`/`astropy-healpix` is not available, but the API and directory names should leave room to swap in real HEALPix later.

## Query Semantics

Local query should:

1. Parse the SPHEREx SIA `s_region` polygon.
2. Compute a conservative RA/Dec bounding box.
3. Identify candidate spatial partitions intersecting the bounding box.
4. Read only those Parquet partitions.
5. Apply:
   - exact polygon containment if practical
   - otherwise conservative bounding-box filtering plus a clear warning/metadata flag
6. Apply magnitude cuts:

```text
phot_g_mean_mag BETWEEN g_min AND g_max
```

7. Deduplicate by `source_id`.
8. Return a deterministic distributed sample up to `max_sources`.

The returned dataframe should use the same column names as the TAP query so existing downstream code remains unchanged.

## Distributed Sampling

Avoid the old problem where `TOP N` returns a clumped source-id ordered patch.

For a field result:

1. Split the field bounding box into a 4x4 grid.
2. Within each tile sort by `phot_g_mean_mag`.
3. Take roughly `max_sources / 16` from each populated tile.
4. Fill remaining slots from the global sorted candidate list.
5. Return deterministic order.

## Builder CLI

Add:

```bash
spherex-mine build-gaia-lite \
  --cache-root /mnt/niroseti/spherex_cache \
  --limit-files <optional> \
  --overwrite
```

Requirements:

- Stream raw `.csv.gz` files; do not decompress whole files to disk.
- Read only the selected columns if possible.
- Write Parquet in batches.
- Write a `manifest.json` with:
  - build time
  - source raw directory
  - file count processed
  - row count
  - selected columns
  - partition scheme
  - package version / command args

## Query CLI

Add:

```bash
spherex-mine query-local-gaia \
  --s-region "POLYGON ..." \
  --g-min 12.5 \
  --g-max 14.0 \
  --max-sources 100 \
  --cache-root /mnt/niroseti/spherex_cache
```

It should print JSON summary and optionally write a Parquet/CSV output if `--output` is provided.

## Integration

Update `spherex_laser_miner/catalog/gaia.py`:

```text
if local lite index exists:
    query local index
else:
    use remote TAP fallback
```

Keep the remote TAP implementation intact for fallback and regression comparison.

## Validation

Add a comparison helper or CLI:

```bash
spherex-mine compare-local-gaia \
  --s-region "POLYGON ..." \
  --g-min 12.5 \
  --g-max 14.0 \
  --max-sources 100
```

Minimum validation:

- local returns rows for known SIMP field polygons
- returned columns match remote contract
- all local rows are inside the conservative field bounds
- magnitude cuts are respected
- source IDs are unique
- query is deterministic across repeated calls

## Performance Metrics

For builder:

```text
files/sec
rows/sec
MB/sec compressed read
Parquet write MB/sec
```

For query:

```text
partitions scanned
rows scanned
rows after bbox
rows after magnitude
query wall time
```

## Non-Goals For First Pass

- No mutable science labels in Gaia base Parquet.
- No full PostgreSQL import.
- No XP spectra import.
- No all-Gaia crossmatch tables.
- No cloud-scale distributed query service.

Science annotations should live separately under:

```text
/mnt/niroseti/spherex_cache/gaia/science/
```

keyed by `source_id`, `target_id`, `image_id`, or `run_id`.
