# How To Run The Current System

This is the practical startup guide for the current target-centered SPHEREx
miner. It covers the viewer servers, one-off miner runs, campaign scripts,
candidate scanners, and status checks. It assumes the repo lives at:

```bash
cd /home/clive/dev/NIROSETI_SPHEREx
```

All examples use the NAS cache root:

```text
/mnt/niroseti/spherex_cache
```

## 1. Preflight

Confirm the NAS and Python environment:

```bash
df -h /mnt/niroseti
.venv/bin/spherex-mine doctor
```

Confirm CUDA/Warp can see the GPUs:

```bash
.venv/bin/python - <<'PY'
import warp as wp
wp.init()
print(wp.get_devices())
PY
```

## 2. Start Or Restart The Web Viewer

The viewer serves all current dashboards from one process. Bind to `0.0.0.0`
so other machines on the LAN can reach it.

```bash
tmux new-session -d -s spherex-viewer \
  'cd /home/clive/dev/NIROSETI_SPHEREx && .venv/bin/spherex-mine viewer \
    --host 0.0.0.0 \
    --port 8765 \
    --cache-root /mnt/niroseti/spherex_cache'
```

Restart only the viewer:

```bash
tmux kill-session -t spherex-viewer
tmux new-session -d -s spherex-viewer \
  'cd /home/clive/dev/NIROSETI_SPHEREx && .venv/bin/spherex-mine viewer \
    --host 0.0.0.0 \
    --port 8765 \
    --cache-root /mnt/niroseti/spherex_cache'
```

Useful pages:

```text
http://192.168.1.224:8765/campaign-status?campaign=<campaign>
http://192.168.1.224:8765/simple-status?run=<run_name>
http://192.168.1.224:8765/spectra?run=<run_name>
http://192.168.1.224:8765/candidate-summary?campaign=<campaign>&quality=pass
http://192.168.1.224:8765/blind-candidates?run=<run_name>&scope=raw
http://192.168.1.224:8765/injections?run=<injected_run_name>
```

## 3. Run One Depth Miner

Use this to build spectra around one manual or safe Gaia anchor. This runs GPU
aperture photometry plus GPU PSF photometry.

```bash
.venv/bin/spherex-mine run-depth-test \
  --target cvj_arcturus_gaia_1233967198192238208 \
  --run-name manual_arcturus_g11_16_f500 \
  --release qr2 \
  --limit-fields 500 \
  --max-gaia-sources 500 \
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

Smoke-test version:

```text
--limit-fields 20
--max-gaia-sources 100
```

Primary outputs:

```text
/mnt/niroseti/spherex_cache/runs/<run_name>/spectra/all_measurements.parquet
/mnt/niroseti/spherex_cache/runs/<run_name>/spectra/target_spectra.parquet
/mnt/niroseti/spherex_cache/runs/<run_name>/spectra/target_summary.parquet
/mnt/niroseti/spherex_cache/runs/<run_name>/status_summary.json
```

## 4. Run The GPU Narrowband Scorer By Hand

Use this after a run has `spectra/target_spectra.parquet`.

```bash
.venv/bin/python tools/warp_narrowband_detector.py \
  --run-dir /mnt/niroseti/spherex_cache/runs/<run_name> \
  --output-dir /mnt/niroseti/spherex_cache/runs/<run_name>/narrowband_detector_raw \
  --grid-step-nm 1.0 \
  --min-joint-rho 3.0 \
  --top-k-per-target 20 \
  --device cuda:0 \
  --quality-min-support 3 \
  --quality-max-flagged-points 3 \
  --quality-max-candidates-per-target 5 \
  --quality-max-aperture-psf-ratio 3.0 \
  --diagnostic-line-half-window-nm 80 \
  --diagnostic-line-max-rows-per-candidate 201
```

Outputs:

```text
narrowband_detector_raw/narrowband_candidates.parquet
narrowband_detector_raw/narrowband_line_scores.parquet
narrowband_detector_raw/narrowband_detector_summary.json
```

`narrowband_candidates.parquet` is the candidate table. The viewer uses
`narrowband_line_scores.parquet` for the detailed line-score plot around each
candidate. It is intentionally compact; do not write the full target x
wavelength score cube for normal campaigns.

For injection truth recovery, add:

```text
--manifest /mnt/niroseti/spherex_cache/injection_campaigns/<campaign_target>_mixed_lasers_s1_3_8/injection_manifest.json
--target-ids-file /mnt/niroseti/spherex_cache/runs/<injected_run>/blind_raw_recovery_truth_target_ids.txt
```

## 5. Run A Full Campaign

The campaign runner performs, per safe Gaia anchor:

1. Baseline depth run.
2. Raw baseline GPU narrowband scan.
3. Mixed-laser injection plan.
4. FITS-level injection into copied FITS files.
5. Injected depth run using `path_overrides.json`.
6. Raw injected GPU narrowband scan.
7. Paired-delta sanity scan.
8. Truth-target raw injected recovery scan.
9. Recovery scoring and false-positive review manifest.

Deep overnight-style command:

```bash
tmux new-session -d -s spherex-overnight-diag \
  'cd /home/clive/dev/NIROSETI_SPHEREx && .venv/bin/python tools/run_visible_sky_injection_campaign.py \
    --campaign-prefix cv_june_g11_16_f500_diag_overnight_v1 \
    --targets configs/castro_valley_june_survey_targets.yaml \
    --resolve-gaia-anchors \
    --only-target cvj_regulus \
    --only-target cvj_denebola \
    --only-target cvj_porrima \
    --only-target cvj_spica \
    --only-target cvj_arcturus \
    --only-target cvj_izar \
    --only-target cvj_alphecca \
    --only-target cvj_unukalhai \
    --only-target cvj_antarest \
    --only-target cvj_rasalhague \
    --only-target cvj_vega \
    --only-target cvj_sheltan \
    --only-target cvj_tarazed \
    --only-target cvj_deneb \
    --only-target cvj_sadr \
    --only-target cvj_enif \
    --only-target cvj_scheat \
    --only-target cvj_markab \
    --only-target cvj_fomalhaut \
    --limit-fields 500 \
    --max-gaia-sources 500 \
    --gaia-g-min 11 \
    --gaia-g-max 16 \
    --max-field-workers 24 \
    --warp-devices cuda:0,cuda:1,cuda:2 \
    --strengths-sigma 1,3,8 \
    --max-line-flux-uJy 50000 \
    --min-snr 1.5 \
    --blind-scanner narrowband_gpu \
    --blind-grid-step-nm 1.0 \
    --blind-top-k-per-target 20 \
    --narrowband-min-joint-rho 3.0 \
    --narrowband-diagnostic-line-half-window-nm 80 \
    --narrowband-diagnostic-line-max-rows-per-candidate 201 \
    --viewer-base-url http://192.168.1.224:8765 \
    2>&1 | tee /mnt/niroseti/spherex_cache/campaigns/cv_june_g11_16_f500_diag_overnight_v1/campaign_stdout.log'
```

Notes:

- `--resolve-gaia-anchors` converts bright sky centers into nearby safe Gaia
  anchors. This avoids centering on saturated bright stars.
- The command above excludes `cvj_altair`, which previously lacked measured
  parent fields in this prototype.
- `--max-gaia-sources 500` is a thin target sample for reviewable overnight
  runs. Increase it only when you are ready for larger outputs.
- The runner is stage-resumable. Re-running the same command skips completed
  stages. Use `--force` only to intentionally overwrite stage products.

Small campaign smoke test:

```bash
.venv/bin/python tools/run_visible_sky_injection_campaign.py \
  --campaign-prefix smoke_one_target \
  --targets configs/castro_valley_june_survey_targets.yaml \
  --resolve-gaia-anchors \
  --limit-targets 1 \
  --limit-fields 40 \
  --max-gaia-sources 100 \
  --max-field-workers 8 \
  --warp-devices cuda:0 \
  --blind-scanner narrowband_gpu \
  --viewer-base-url http://192.168.1.224:8765
```

## 6. Monitor Jobs

List tmux sessions:

```bash
tmux ls
```

Watch a campaign:

```bash
tmux attach -t spherex-overnight-diag
```

Detach without stopping:

```text
Ctrl-b, then d
```

Inspect output without attaching:

```bash
tmux capture-pane -t spherex-overnight-diag -p -S -120 | tail -80
tail -80 /mnt/niroseti/spherex_cache/campaigns/<campaign>/campaign_stdout.log
```

Process check:

```bash
ps -eo pid,ppid,pcpu,pmem,etime,cmd | rg 'run_visible_sky|run-depth-test|warp_narrowband|spherex-mine viewer'
```

GPU check:

```bash
nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader,nounits
```

## 7. Review Results

Campaign status:

```text
http://192.168.1.224:8765/campaign-status?campaign=<campaign>
```

Quality-pass candidates across a campaign:

```text
http://192.168.1.224:8765/candidate-summary?campaign=<campaign>&quality=pass
```

Only baseline science candidates:

```text
http://192.168.1.224:8765/candidate-summary?campaign=<campaign>&source=baseline&quality=pass
```

Only injected QA candidates:

```text
http://192.168.1.224:8765/candidate-summary?campaign=<campaign>&source=injected&quality=pass
```

Per-run raw blind browser:

```text
http://192.168.1.224:8765/blind-candidates?run=<run_name>&scope=raw
```

Per-run S/A/B tier filters:

```text
http://192.168.1.224:8765/blind-candidates?run=<run_name>&scope=raw&tier=S
http://192.168.1.224:8765/blind-candidates?run=<run_name>&scope=raw&tier=A
http://192.168.1.224:8765/blind-candidates?run=<run_name>&scope=raw&tier=B
```

The blind candidate browser line plot uses `narrowband_line_scores.parquet`
when present. If it says no saved line score rows, rerun the scorer/campaign
with:

```text
--narrowband-diagnostic-line-half-window-nm 80
--narrowband-diagnostic-line-max-rows-per-candidate 201
```

## 8. Stop Or Resume

Stop a campaign session:

```bash
tmux kill-session -t spherex-overnight-diag
```

Resume by re-running the same command. Existing stage outputs are skipped.

Avoid deleting run directories unless you intentionally want to discard that
stage. If a stage output is bad but later products depend on it, delete or move
that stage output and resume, or rerun with `--force` after confirming the blast
radius.
