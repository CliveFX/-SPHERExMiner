# GPU PSF Grid Photometry Notes

Date: 2026-06-23

## Summary

We prototyped GPU forced-PSF photometry in `spherex_laser_miner/photometry/psf_forced.py` and tested it with
`tools/compare_psf_photometry.py`.

The correct GPU shape is a flattened local-grid solve:

```text
target x grid_offset -> independent PSF fit candidate -> best candidate per target
```

The local grid solve is required, not optional. The GPU path now launches one candidate-fit thread per
`(target, grid_offset)`, then a reduction kernel chooses the best candidate by the configured metric.

## CUDA Codegen Finding

The first Warp kernel generated about 161k lines of CUDA and could time out or dump core during compile. The cause was:

- static `range(12)` loops being unrolled
- unnecessary backward/adjoint generation
- too much work in one kernel

The rewritten kernels use dynamic `while` loops and `@wp.kernel(enable_backward=False)`.

Current generated CUDA for the spline GPU-builder version is sane:

- about 2,318 lines
- about 109 KB
- compile time under 1 second on `cuda:0`
- no backward CUDA kernel

## Kernel-Build Modes

The comparison harness supports three PSF-kernel build modes:

```text
cpu_scipy     CPU builds per-candidate PSF kernels with SciPy/SPExPI-style shift, then uploads them.
gpu_bilinear  GPU builds per-candidate PSF kernels from the FITS PSF cube with bilinear sampling.
gpu_spline    CPU prefilters each PSF plane once, uploads spline coefficients, GPU builds all candidate kernels.
```

`cpu_scipy` is the exactness baseline for the GPU fit/reduce math, but it is not the desired production shape because it
does CPU work per candidate. `gpu_spline` is the preferred production direction.

## Accuracy Results

Compared against CPU grid photometry.

Exact GPU fit/reduce with CPU/SPExPI-built kernels:

```text
UCS one-row fractional error:       4.46e-8  = 0.00000446%
Arcturus 15-row median fraction:    1.15e-7  = 0.0000115%
```

GPU-built bilinear kernels:

```text
UCS one-row error:                  0.0517%
Arcturus median error:              0.0436%
Arcturus p90 abs error:             0.1466%
Arcturus max abs error:             0.4451%
```

GPU-built spline kernels:

```text
UCS one-row error:                  0.00158%
Arcturus median error:             -0.000179%
Arcturus median abs error:          0.000621%
Arcturus p90 abs error:             0.002798%
Arcturus max abs error:             0.004042%
Arcturus grid offset agreement:     15 / 15
```

The remaining `gpu_spline` difference is likely floating-point and implementation detail:

- SciPy prefiltering/sampling uses float64 internally.
- The GPU coefficient cube is stored as float32.
- Warp candidate fitting currently uses float32 sums.
- Operation order and boundary arithmetic differ slightly.

This is good enough to proceed with integration unless later science validation requires float64 sums or stricter
kernel-array equivalence.

## Repro Commands

UCS spline smoke:

```bash
.venv/bin/python tools/compare_psf_photometry.py \
  --run-dir /mnt/niroseti/spherex_cache/runs/ucs0972_gpu_g14_16_n1000_f80 \
  --target-id ucs_0972 \
  --max-rows 1 \
  --max-rows-per-target 1 \
  --enable-gpu \
  --allow-experimental-warp \
  --gpu-kernel-build-mode gpu_spline \
  --warp-device cuda:0 \
  --output-dir /mnt/niroseti/spherex_cache/runs/ucs0972_gpu_g14_16_n1000_f80/benchmarks/psf_correctness_gpu_grid_spline_smoke
```

Arcturus spline sample:

```bash
.venv/bin/python tools/compare_psf_photometry.py \
  --run-dir /mnt/niroseti/spherex_cache/runs/arcturus_mag_calibration_g5_16_n100_f220_cwave_gpu_w8 \
  --mag-min 11 \
  --mag-max 16 \
  --max-rows 36 \
  --max-rows-per-target 3 \
  --enable-gpu \
  --allow-experimental-warp \
  --gpu-kernel-build-mode gpu_spline \
  --warp-device cuda:0 \
  --output-dir /mnt/niroseti/spherex_cache/runs/arcturus_mag_calibration_g5_16_n100_f220_cwave_gpu_w8/benchmarks/psf_correctness_gpu_grid_spline_mag11_16
```

## Production Integration Status

The GPU spline grid path is now the preferred production PSF backend in
`run-depth-test`:

```bash
--enable-psf \
--psf-photometry-backend warp_grid \
--psf-kernel-build-mode gpu_spline \
--psf-grid-half-range-pix 1.0 \
--psf-grid-step-pix 0.5 \
--psf-grid-metric snr
```

The current target-centered campaign runner uses this path for both baseline
and injected spectra runs. It still launches relatively small per-field GPU
jobs, so GPU occupancy remains low; the future frame-scale survey engine should
batch whole frames and large target arrays more aggressively.

Remaining PSF work:

- Continue comparing PSF and aperture spectra for injected recovery.
- Add stricter science validation against SPExPI/reference cases.
- Profile PSF setup versus fit/reduction time in full campaign runs.
- Consider larger frame-batched GPU scheduling before any cloud-scale survey.
