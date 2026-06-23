# SPHEREx Miner Operator Runbook

This is the handoff document for running the current NIROSETI/SPHEREx prototype
without an agent driving each step.

The current system is a target-centered survey prototype. It is not yet the
future frame-scale all-sky scheduler. A run starts from one anchor target,
selects SPHEREx fields around that anchor, selects Gaia sources in those fields,
measures aperture and PSF photometry, assembles spectra, and optionally runs
FITS-level fake signal injection plus paired recovery scoring.

## Paths

Run commands from the repo root:

```bash
cd /home/clive/dev/NIROSETI_SPHEREx
```

Use the project virtual environment:

```bash
.venv/bin/spherex-mine doctor
```

Default cache/output root:

```text
/mnt/niroseti/spherex_cache
```

Important output directories:

```text
/mnt/niroseti/spherex_cache/runs/<run_name>/
/mnt/niroseti/spherex_cache/injection_campaigns/<campaign_name>/
/mnt/niroseti/spherex_cache/campaigns/<campaign_name>/
```

The main spectra products for a completed run are:

```text
spectra/all_measurements.parquet
spectra/target_spectra.parquet
spectra/target_summary.parquet
spectra/assembly_summary.json
```

## Run One Miner

Use this when you want one baseline spectra run around one target.

GPU aperture plus GPU PSF, current preferred path:

```bash
.venv/bin/spherex-mine run-depth-test \
  --target simp0136 \
  --run-name manual_simp_g11_16_f220 \
  --release qr2 \
  --limit-fields 220 \
  --max-gaia-sources 6000 \
  --gaia-g-min 11 \
  --gaia-g-max 16 \
  --max-field-workers 24 \
  --photometry-backend warp_calibrated \
  --warp-devices cuda:0,cuda:1,cuda:2 \
  --status-mode jsonl \
  --max-field-retries 1 \
  --enable-psf \
  --psf-photometry-backend warp_grid \
  --psf-kernel-build-mode gpu_spline \
  --psf-grid-half-range-pix 1.0 \
  --psf-grid-step-pix 0.5 \
  --psf-grid-metric snr \
  --cache-root /mnt/niroseti/spherex_cache
```

For a smaller smoke run, reduce:

```text
--limit-fields 20
--max-gaia-sources 100
```

For a deeper target-centered run, increase `--limit-fields`. A value of `500`
has been the current deep-run setting. Keep `--max-field-workers 24` unless you
are deliberately benchmarking; raising it to 72 did not materially improve
throughput in the current architecture.

## Run The Visible-Sky Injection Campaign

This wrapper runs, per target:

1. Baseline spectra run.
2. Mixed-laser injection plan.
3. FITS-level injection into copied FITS files.
4. Injected spectra run using path overrides.
5. Paired baseline/injected matched-filter classifier.
6. Recovery scoring and false-positive review manifest.

Full default campaign:

```bash
.venv/bin/python tools/run_visible_sky_injection_campaign.py
```

One target only:

```bash
.venv/bin/python tools/run_visible_sky_injection_campaign.py \
  --only-target cvj_arcturus
```

Small sanity pass:

```bash
.venv/bin/python tools/run_visible_sky_injection_campaign.py \
  --limit-targets 1 \
  --limit-fields 40 \
  --max-gaia-sources 500
```

Current deep default shape:

```bash
.venv/bin/python tools/run_visible_sky_injection_campaign.py \
  --limit-fields 500 \
  --max-gaia-sources 6000 \
  --gaia-g-min 11 \
  --gaia-g-max 16 \
  --max-field-workers 24 \
  --warp-devices cuda:0,cuda:1,cuda:2
```

Wider injection ladder for threshold/false-positive characterization:

```bash
.venv/bin/python tools/run_visible_sky_injection_campaign.py \
  --campaign-prefix cv_june_g11_16_f500_wideinj \
  --strengths-sigma 0.5,1,2,3,5,8,12 \
  --max-line-flux-uJy 50000 \
  --min-snr 1.5 \
  --limit-fields 500 \
  --max-gaia-sources 6000 \
  --gaia-g-min 11 \
  --gaia-g-max 16 \
  --max-field-workers 24 \
  --warp-devices cuda:0,cuda:1,cuda:2
```

The `--max-line-flux-uJy` cap matters. Without it, a low requested
`find_me_snr` can still become an extremely bright intrinsic line if the target
only has weak spectral response or high baseline uncertainty near that laser
line. Use the uncapped `5,8,12` ladder as a pipeline correctness test, not as
the final discriminator-training distribution.

The campaign is resumable. If the expected output for a stage already exists,
that stage is skipped. Use `--force` only when you intentionally want to rerun
existing products.

## Run In The Background

Use `tmux` for long jobs:

```bash
tmux new-session -d -s cv-june-campaign \
  '.venv/bin/python tools/run_visible_sky_injection_campaign.py'
```

Watch it:

```bash
tmux attach -t cv-june-campaign
```

Detach from tmux without stopping the job:

```text
Ctrl-b, then d
```

Check process state:

```bash
ps -eo pid,etime,%cpu,%mem,cmd | rg 'run_visible_sky_injection_campaign|spherex-mine run-depth-test'
```

Inspect recent tmux output without attaching:

```bash
tmux capture-pane -t cv-june-campaign -p -S -120 | tail -80
```

## Dashboards

Start the viewer:

```bash
tmux new-session -d -s spherex-viewer \
  '.venv/bin/spherex-mine viewer \
    --host 0.0.0.0 \
    --port 8765 \
    --cache-root /mnt/niroseti/spherex_cache \
    --run-name <run_name>'
```

Example:

```bash
tmux new-session -d -s spherex-viewer \
  '.venv/bin/spherex-mine viewer \
    --host 0.0.0.0 \
    --port 8765 \
    --cache-root /mnt/niroseti/spherex_cache \
    --run-name cv_june_g11_16_f500_cvj_denebola_gaia_3923573919068194432_injected'
```

If the viewer is already running, restart only the viewer session:

```bash
tmux kill-session -t spherex-viewer
tmux new-session -d -s spherex-viewer \
  '.venv/bin/spherex-mine viewer --host 0.0.0.0 --port 8765 --cache-root /mnt/niroseti/spherex_cache --run-name <run_name>'
```

Do not kill the tmux server; campaign and viewer sessions can share the same
server process.

Dashboard URLs:

```text
http://<host>:8765/simple-status?run=<run_name>
http://<host>:8765/spectra?run=<run_name>
http://<host>:8765/injections?run=<injected_run_name>
http://<host>:8765/injections?run=<injected_run_name>&status=candidate
```

Dashboard meanings:

- `simple-status`: low-overhead run progress and field status.
- `spectra`: target spectra browser with aperture, PSF, flags, and injection
  markers when present.
- `injections`: injection/recovery browser with recovery truth, aperture and PSF
  spectra, scorer candidates, and a separate synthetic injected-response panel.
- `/`: older field image viewer. Useful for visual context, but not the primary
  status surface.

## Manual Injection/Recovery Pipeline

Use this only when you need to run stages by hand instead of the campaign
wrapper.

Create a mixed laser injection plan from a completed baseline run:

```bash
.venv/bin/python tools/make_mixed_laser_injection_plan.py \
  --run-dir /mnt/niroseti/spherex_cache/runs/<baseline_run_name> \
  --campaign-id <campaign_id> \
  --output-root /mnt/niroseti/spherex_cache/injection_campaigns \
  --strengths-sigma 5,8,12 \
  --targets-per-cell 3 \
  --line-width-nm 1.0 \
  --min-measurements 20 \
  --max-line-flux-uJy 50000
```

Apply the plan to copied FITS files:

```bash
.venv/bin/python tools/run_injection_plan.py \
  --plan /mnt/niroseti/spherex_cache/injection_campaigns/<campaign_id>/injection_plan.json
```

Run the miner against the injected FITS path overrides:

```bash
.venv/bin/spherex-mine run-depth-test \
  --target <target_id> \
  --run-name <injected_run_name> \
  --release qr2 \
  --limit-fields 500 \
  --max-gaia-sources 6000 \
  --gaia-g-min 11 \
  --gaia-g-max 16 \
  --max-field-workers 24 \
  --photometry-backend warp_calibrated \
  --warp-devices cuda:0,cuda:1,cuda:2 \
  --status-mode jsonl \
  --max-field-retries 1 \
  --enable-psf \
  --psf-photometry-backend warp_grid \
  --psf-kernel-build-mode gpu_spline \
  --psf-grid-half-range-pix 1.0 \
  --psf-grid-step-pix 0.5 \
  --psf-grid-metric snr \
  --cache-root /mnt/niroseti/spherex_cache \
  --path-overrides /mnt/niroseti/spherex_cache/injection_campaigns/<campaign_id>/path_overrides.json
```

Run paired baseline/injected scoring:

```bash
.venv/bin/python tools/classify_paired_delta_matched_filter.py \
  --baseline-run-dir /mnt/niroseti/spherex_cache/runs/<baseline_run_name> \
  --injected-run-dir /mnt/niroseti/spherex_cache/runs/<injected_run_name> \
  --plan /mnt/niroseti/spherex_cache/injection_campaigns/<campaign_id>/injection_plan.json \
  --output-dir /mnt/niroseti/spherex_cache/runs/<injected_run_name>/classifier_paired_delta \
  --min-snr 5 \
  --ignore-flagged
```

Score recovery against truth:

```bash
.venv/bin/python tools/score_injection_recovery.py \
  --manifest /mnt/niroseti/spherex_cache/injection_campaigns/<campaign_id>/injection_manifest.json \
  --candidates /mnt/niroseti/spherex_cache/runs/<injected_run_name>/classifier_paired_delta/matched_filter_candidates.parquet \
  --output-dir /mnt/niroseti/spherex_cache/runs/<injected_run_name>/recovery_score_mixed_lasers \
  --min-snr 5 \
  --wavelength-tolerance-nm 10 \
  --require-line-family
```

## Manual Kickoff Checklist

Use this checklist when GPT/Codex is not driving the run.

1. Confirm the NAS is mounted:

   ```bash
   df -h /mnt/niroseti
   ```

2. Confirm the virtual environment and imports:

   ```bash
   .venv/bin/spherex-mine doctor
   ```

3. Confirm local Gaia exists if the run will query Gaia:

   ```bash
   find /mnt/niroseti/spherex_cache/gaia_lite -maxdepth 3 -type f | head
   ```

4. Start the dashboard:

   ```bash
   tmux new-session -d -s spherex-viewer \
     '.venv/bin/spherex-mine viewer --host 0.0.0.0 --port 8765 --cache-root /mnt/niroseti/spherex_cache --run-name <run_name>'
   ```

5. Start either one direct miner run or the campaign wrapper in tmux.

6. Watch `simple-status` first. Use spectra/injection viewers after spectra
   products are assembled.

7. For injected runs, inspect:

   ```text
   recovery_score_mixed_lasers/recovery_summary.json
   recovery_score_mixed_lasers/false_positive_candidates.parquet
   classifier_paired_delta/matched_filter_summary.json
   ```

8. Before changing code after a good stopping point:

   ```bash
   git status --short
   .venv/bin/python -m compileall -q spherex_laser_miner
   git diff --check
   ```

## Current Practical Defaults

- Science wavelength source should be `spectral_wcs_CWAVE_CBAND`.
- Current useful magnitude range for broad survey tests is Gaia `G=11..16`.
- Safe manual anchors should not be bright stars themselves. The campaign
  resolves bright sky centers to nearby Gaia `G=12..14` anchors.
- Use `--ignore-flagged` for classifiers unless deliberately auditing flags.
- Keep `--status-mode jsonl`; the old SQLite live-status path has been removed.
- Prefer compact exported spectra/candidate bundles for web sharing. Do not
  publish copied injected FITS products unless they are explicitly needed.

## Stopping And Resuming

The campaign wrapper is stage-resumable. Restarting the same command should skip
completed stages and continue at the first missing output.

To stop a foreground job, use `Ctrl-c`. For tmux jobs, attach and stop the
foreground process, or kill only the specific session:

```bash
tmux kill-session -t cv-june-campaign
```

Avoid deleting run directories unless you intentionally want to discard that
stage. Use `--force` for controlled reruns.
