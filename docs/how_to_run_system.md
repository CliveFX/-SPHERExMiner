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

Cold-starting a new machine or rebuilding the local Gaia cache is covered in
[Gaia Cache Cold Start](gaia_cache_cold_start.md).

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
3. Optional transformer ML sandbox score on baseline spectra.
4. Mixed-laser injection plan.
5. FITS-level injection into copied FITS files.
6. Injected depth run using `path_overrides.json`.
7. Raw injected GPU narrowband scan.
8. Optional transformer ML sandbox score on injected spectra.
9. Paired-delta sanity scan.
10. Truth-target raw injected recovery scan.
11. Optional transformer ML sandbox score on truth-target injected spectra.
12. Recovery scoring and false-positive review manifest.

Current full overnight magnitude sequence:

```bash
tmux new-session -d -s spherex-ml-mag-sequence \
  'cd /home/clive/dev/NIROSETI_SPHEREx && bash tools/run_tonight_ml_mag_sequence.sh'
```

This runs full baseline/injection/recovery campaigns over all named Castro
Valley June anchors in this order:

1. `G 11-16`, known-good range.
2. `G 8-11`, brighter stress range.
3. `G 5-8`, brightest stress range.

The sequence defaults to:

```text
LIMIT_FIELDS=500
MAX_GAIA_SOURCES=3000
MAX_FIELD_WORKERS=24
WARP_DEVICES=cuda:0,cuda:1,cuda:2
```

Override values without editing the script:

```bash
MAX_GAIA_SOURCES=3000 LIMIT_FIELDS=500 EXTRA_ARGS="--limit-targets 3" \
  bash tools/run_tonight_ml_mag_sequence.sh
```

Single-bin YAML-driven command:

```bash
tmux new-session -d -s spherex-g11-16 \
  'cd /home/clive/dev/NIROSETI_SPHEREx && .venv/bin/python tools/run_visible_sky_injection_campaign.py \
    --config configs/campaign_with_ml_transformer.yaml \
    --campaign-prefix cv_june_g11_16_f500_n3000_ml \
    --limit-fields 500 \
    --max-gaia-sources 3000 \
    --gaia-g-min 11 \
    --gaia-g-max 16 \
    2>&1 | tee /mnt/niroseti/spherex_cache/campaigns/cv_june_g11_16_f500_n3000_ml/campaign_stdout.log'
```

Notes:

- `--resolve-gaia-anchors` converts bright sky centers into nearby safe Gaia
  anchors. This avoids centering on saturated bright stars.
- `configs/campaign_with_ml_transformer.yaml` preserves the current campaign
  defaults. Command-line flags override config values.
- The transformer ML scorer is exploratory. Its outputs are written as separate
  `ml_narrowband_transformer` products and campaign cards, but it is not a
  science gate.
- The runner is stage-resumable. Re-running the same command skips completed
  stages. Use `--force` only to intentionally overwrite stage products.
- Targets with no measured parent fields, such as the current Altair anchor,
  are recorded as skipped and the campaign continues.
- Targets whose spectra cannot support any requested injection line are also
  recorded as skipped. This can happen in sparse bright-star bins where only a
  few usable spectra survive and none overlap the configured laser lines.
- Recovery Summary shows only targets that have finished
  `recovery_score_mixed_lasers/recovery_summary.json`; active targets appear
  later.

Campaign Status recovery cards:

- `GPU Raw Recovery`: truth-target GPU scanner matches near injected
  wavelengths before strict review cuts.
- `Quality Recovery`: raw recovery matches that also pass candidate quality
  filters.
- `Paired Recovery`: baseline-vs-injected matched-filter recovery. This is the
  cleanest check that the system can see fake signals it injected.
- `ML Truth Detect`: experimental transformer says a truth target looks
  line-like, regardless of wavelength accuracy.
- `ML Truth Recover`: experimental transformer detects and predicts the
  injected wavelength within tolerance.

Small campaign smoke test:

```bash
.venv/bin/python tools/run_visible_sky_injection_campaign.py \
  --config configs/campaign_with_ml_transformer.yaml \
  --campaign-prefix smoke_full_pipeline_g11_16_f120_n150 \
  --limit-targets 1 \
  --limit-fields 120 \
  --max-gaia-sources 150 \
  --gaia-g-min 11 \
  --gaia-g-max 16
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
