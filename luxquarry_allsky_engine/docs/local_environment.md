# Local Environment

## Rule

Do not use the existing repository `.venv` for LuxQuarry All-Sky Engine
experiments.

The current miner environment is a working reference environment. The all-sky
engine needs a separate venv, conda env, or container because RAPIDS/CUDA
dependency choices may be disruptive.

## Current Probe

Initial local probe:

```text
python: 3.12.3
driver: NVIDIA 595.71.05
nvidia-smi CUDA: 13.2
GPUs:
  - RTX 6000 Ada, 49140 MiB
  - RTX 6000 Ada, 49140 MiB
  - RTX 6000 Ada, 46068 MiB
```

The isolated `luxquarry_allsky_engine/.venv` started intentionally blank. The
first probe confirmed GPU visibility through `nvidia-smi`, but no science or
RAPIDS packages were installed.

The initial implementation then installed the minimal CPU/reference stack needed
for manifest building:

```text
numpy
pandas
pyarrow
astropy
```

RAPIDS remains intentionally uninstalled until we choose a conda/container CUDA
setup.

The existing miner `.venv` has core science packages like NumPy, Pandas,
PyArrow, Astropy, and NVIDIA Warp, but not RAPIDS/cuDF.

## Recommended Setup Path

Prefer one of these for RAPIDS work:

1. A dedicated RAPIDS conda/miniforge environment.
2. A RAPIDS/NVIDIA CUDA container.
3. A dedicated pip venv only after confirming local CUDA toolkit/NVRTC
   compatibility.

Do not install heavy CUDA/RAPIDS packages into the current miner `.venv`.

## RAPIDS Notes

The RAPIDS install docs say pip packages are published on the NVIDIA Python
Package Index and require wheels matching the installed CUDA toolkit suffix
(`-cu12`, `-cu13`, etc.). They also note that pip installs require NVRTC for
Numba to function properly.

This machine currently exposes the CUDA driver through `nvidia-smi`, but no
`nvcc` or `/usr/local/cuda*` toolkit was visible in the initial shell probe.
That makes conda/container setup the safer next step for RAPIDS.

Reference:

- RAPIDS install guide: https://docs.rapids.ai/install/

## Commands Already Added

Environment probe:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky env-probe --out runs/env_probe.json
```

Benchmark smoke skeleton:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky benchmark-smoke \
  --campaign-id local_smoke \
  --out-dir runs/local_smoke
```

Generated run artifacts are intentionally ignored by git.

## First Manifest Smoke

Command:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky build-manifest \
  --input-root /mnt/niroseti/spherex_cache/raw/qr2/level2 \
  --out runs/manifest_smoke_v2/frame_manifest.parquet \
  --campaign-id manifest_smoke_v2 \
  --limit 10
```

Result:

```text
frame_count: 10
fits_total_bytes: 716,368,320
total_wall_sec: 0.260
discover_fits: 0.074 sec
read_headers: 0.176 sec
write_manifest: 0.007 sec
```

The manifest correctly parsed:

- planning period
- processing version
- detector directory
- detector
- exposure id
- frame-in-exposure
- image dimensions
- approximate WCS footprint

