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
/mnt/niroseti/spherex_cache/catalogs/2mass/parquet/
```

Initial useful columns:

```text
pts_key
designation
ra
decl
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

2MASS positions are in the J2000/ICRS frame, but the observations occurred
around 1997-2001. For crossmatching high-proper-motion Gaia stars, propagate
Gaia positions back toward the 2MASS observation epoch rather than hiding that
logic inside mining scripts.

## Planned Query API

The miner should eventually use a catalog flag rather than a Gaia-specific path:

```text
spherex-mine run-depth-test --catalog gaia
spherex-mine run-depth-test --catalog 2mass
spherex-mine run-depth-test --catalog gaia,2mass
```

The implementation should route through a catalog provider boundary:

```python
query_catalog_cone(catalog="2mass", ra_deg=..., dec_deg=..., radius_deg=...)
```

That keeps 2MASS epoch handling, quality filtering, and HEALPix Parquet lookup
out of the photometry pipeline.
