# Candidate FITS Inspector

Standalone, lazy image inspector for SPHEREx narrowband candidates.

This app is intentionally separate from the main spectra/candidate viewer because it opens FITS files, builds many target-centered cutouts, writes PNG/GIF products, and can be slow/heavy by design.

## Run

```bash
cd /home/clive/dev/NIROSETI_SPHEREx
.venv/bin/python fits_inspector_app/server.py --host 0.0.0.0 --port 8776 --cache-root /mnt/niroseti/spherex_cache
```

Then open:

```text
http://192.168.1.224:8776/
```

Deep link example:

```text
http://192.168.1.224:8776/?run=deep_grid_test_all_catalog_mid_g11_16_hpx_nside0016_nested_00000912_b0008_baseline&target=twomass_psc_20105039%2B3322233&line_nm=4502
```

## Products

Products are written under:

```text
/mnt/niroseti/spherex_cache/fits_inspector/
```

Each inspected candidate gets:

- `summary.json`
- `frames.parquet`
- `blink.html`
- `blink.gif`
- `cutouts/*.png`

Cutouts are 100x100 pixels centered on the target. The stretch is fixed across the whole target stack from pooled target-local pixels, not per-frame autostretch.
