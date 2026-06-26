# 2MASS Cache Cold Start

2MASS is the near-infrared catalog companion to Gaia. The first step is a raw
Point Source Catalog download on the NAS; the next step is converting the raw
pipe-delimited gzip shards into HEALPix Parquet shards.

## NAS Downloader Container

The Synology-friendly downloader container lives on the NAS, parallel to the
Gaia downloader:

```text
/mnt/niroseti/spherex_cache/2mass/downloader_container
```

It writes raw data under:

```text
/mnt/niroseti/spherex_cache/2mass/raw_download/psc/
```

The status page is bound to:

```text
http://<nas-ip>:18767/
http://<nas-ip>:18767/status.json
```

The Compose file defaults to status-only mode:

```yaml
environment:
  START_DOWNLOAD: "0"
```

Set `START_DOWNLOAD=1` in Synology Container Manager and restart the container
to launch the resumable `aria2c` pull.

## Raw Download Format

The official 2MASS All-Sky Point Source Catalog bulk files are gzipped ASCII
tables from:

```text
https://irsa.ipac.caltech.edu/2MASS/download/allsky/
```

The files are:

```text
psc_aaa.gz .. psc_ace.gz
psc_baa.gz .. psc_bbi.gz
```

Rows are pipe-delimited and use `\N` for nulls. The raw gzipped PSC is about
43 GiB. IRSA documents a loaded Postgres table as about 152 GiB.

## Science-Ready Layer

Do not make every miner script parse raw 2MASS rows. Build a processed catalog
layer instead:

```text
/mnt/niroseti/spherex_cache/2mass/parquet/psc_lite/
```

The builder lives in the repo:

```text
tools/build_2mass_parquet.py
```

Smoke test:

```bash
cd /home/clive/dev/NIROSETI_SPHEREx
.venv/bin/python tools/build_2mass_parquet.py \
  --cache-root /mnt/niroseti/spherex_cache \
  --dataset-name psc_lite_smoke \
  --hpx-level 5 \
  --chunk-rows 50000 \
  --max-buffered-rows 250000 \
  --max-rows-per-file 100000 \
  --limit-files 1 \
  --limit-rows 200000 \
  --overwrite
```

Full build:

```bash
cd /home/clive/dev/NIROSETI_SPHEREx
.venv/bin/python tools/build_2mass_parquet.py \
  --cache-root /mnt/niroseti/spherex_cache \
  --dataset-name psc_lite \
  --hpx-level 5 \
  --chunk-rows 500000 \
  --max-buffered-rows 500000 \
  --max-rows-per-file 500000 \
  --primary-band Ks \
  --workers 20 \
  --overwrite
```

Progress is written atomically to:

```text
/mnt/niroseti/spherex_cache/2mass/parquet/psc_lite/build_status.json
```

With `--workers > 1`, each worker processes one raw gzip shard at a time and
writes uniquely named Parquet files into the shared HEALPix partition
directories. The aggregate status JSON includes `files_done`, `files_active`,
`files_waiting`, total rows, aggregate rows/sec, and an `active_files` list.
Per-worker status files are under:

```text
/mnt/niroseti/spherex_cache/2mass/parquet/psc_lite/_worker_status/
```

The output is Hive-style coordinate HEALPix Parquet:

```text
/mnt/niroseti/spherex_cache/2mass/parquet/psc_lite/
  manifest.json
  build_status.json
  hpx_level=5/
    hpx=<nested_healpix_id>/
      part-00000.parquet
```

Initial useful 2MASS columns:

```text
pts_key
designation
ra
dec
j_m, j_cmsig
h_m, h_cmsig
k_m, k_cmsig
ph_qual
rd_flg
bl_flg
cc_flg
prox
ext_key
scan_key
healpix_nside
healpix_nested
```

The builder also writes canonical columns used by the miner front end:

```text
target_id
target_type
source_id
source_catalog
source_catalog_id
object_name
ra_deg
dec_deg
ra_reference_deg
dec_reference_deg
reference_epoch_yr
pmra_masyr
pmdec_masyr
parallax_mas
mag_primary
mag_primary_band
priority_score
target_filter_flags
```

For generic source-count caps, `mag_primary` defaults to Ks. Keep the raw
`j_m`, `h_m`, and `k_m` columns available because 2MASS does not have a single
Gaia-like broad optical magnitude.

2MASS positions are in the J2000/ICRS frame, but the observations occurred
around 1997-2001. For crossmatching high-proper-motion Gaia stars, propagate
Gaia positions back toward the 2MASS observation epoch rather than hiding that
logic inside mining scripts.

## Query API

The miner now uses a catalog flag rather than a Gaia-specific path:

```text
spherex-mine run-depth-test --catalog gaia
spherex-mine run-depth-test --catalog 2mass
spherex-mine run-depth-test --catalog all
```

The implemented 2MASS path reads the processed local Parquet cache and emits
the same fixed-target schema as the Gaia path. `--catalog all` concatenates Gaia
and 2MASS target rows before photometry.

For survey bins, prefer stratified 2MASS selection:

```text
--twomass-selection stratified
```

This spreads the requested source count across the magnitude interval. The old
brightest-first behavior is still available with `--twomass-selection brightest`
but is usually not appropriate for wide survey bins because it samples only the
bright edge.

The provider boundary is:

```python
query_catalog_cone(catalog="2mass", ra_deg=..., dec_deg=..., radius_deg=...)
```

That keeps 2MASS epoch handling, quality filtering, and HEALPix Parquet lookup
out of the photometry pipeline.

Current caveat: raw 2MASS PSC rows are static J2000/ICRS positions without
proper motion or parallax. Manual targets such as UCS still use their explicit
proper-motion fields, but ordinary 2MASS catalog rows are not HPM-corrected yet.
