# GPU Worker Dispatch

LuxQuarry V2 is moving away from one-shot target/campaign commands and toward
persistent frame workers. A worker owns one GPU, initializes RAPIDS/Warp once,
keeps detector calibration maps resident on that GPU, and processes a disjoint
slice of a frame manifest.

## Worker Contract

Each worker receives the same inputs:

```text
frame_manifest.parquet
projected_targets.parquet
worker_index
worker_count
device
output_dir
```

Partitioning is deterministic:

```text
frame_ordinal % worker_count == worker_index
```

This means local processes, Kubernetes Jobs, or many nodes can consume the same
manifest without a live scheduler in the hot path. A failed worker can be
restarted with the same `worker_index` and `worker_count`.

## Hot Path

The persistent worker uses this frame-level GPU path:

```text
read FITS IMAGE/VARIANCE/FLAGS
optionally stage FITS onto local SSD/NVMe
load/reuse resident SAPM + CWAVE + CBAND maps on GPU
launch one Warp kernel per frame
emit device columns through DLPack to CuPy/cuDF
optionally assemble cuDF tables once per shard batch
queue or write independent cuDF parquet shard
atomically rewrite run_status.json
```

The kernel performs:

- image unit conversion to uJy using SAPM
- variance scaling
- CWAVE/CBAND bilinear sampling
- aperture flux
- annulus background
- uncertainty
- aperture flag summary

## Dispatch Plan

Generate a local multi-GPU dispatch plan:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky plan-gpu-dispatch \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/dispatch_smoke10 \
  --run-id dispatch_smoke10 \
  --plan-out runs/dispatch_smoke10/dispatch_plan.json \
  --devices cuda:0,cuda:1,cuda:2 \
  --shard-batch-frames 5 \
  --prefetch-frames 2 \
  --async-shard-writes \
  --batch-table-assembly \
  --materialize-worker-inputs \
  --status-interval-frames 25 \
  --local-cache-dir /tmp/luxquarry_stage_smoke \
  --limit-frames 10
```

This writes:

```text
dispatch_plan.json
dispatch_plan.sh
```

The shell file launches one persistent worker per listed GPU. The JSON file is
the portable contract for an EKS Job generator.

`--materialize-worker-inputs` writes each worker's manifest and projected-target
slice before the shell plan is emitted:

```text
worker_inputs/w0000/frame_manifest.parquet
worker_inputs/w0000/projected_targets.parquet
worker_inputs/w0001/frame_manifest.parquet
worker_inputs/w0001/projected_targets.parquet
```

In materialized mode, the generated worker command uses `--worker-index 0` and
`--worker-count 1` because partitioning has already happened. The plan still
records the original logical worker index for aggregation. This is the preferred
shape for large multi-node runs where repeatedly reading the full projected
target parquet would waste startup I/O.

## Local One-Command Runner

For workstation runs, use `run-local-dispatch` to execute the same contract
without manually invoking the generated shell:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky run-local-dispatch \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/local_dispatch_smoke2 \
  --run-id local_dispatch_smoke2 \
  --devices cuda:0 \
  --limit-frames 2 \
  --shard-batch-frames 2 \
  --prefetch-frames 1 \
  --status-interval-frames 2 \
  --status-snapshot-interval-sec 1.0 \
  --local-cache-dir /tmp/luxquarry_stage_smoke \
  --async-shard-writes \
  --batch-table-assembly \
  --finalize-device cuda:0
```

This command:

```text
builds dispatch_plan.json
materializes worker inputs by default
launches one persistent worker process per planned worker
captures worker stdout/stderr under worker_logs/
waits for all workers
runs finalize-dispatch-run
writes local_dispatch_summary.json
```

It uses the same dispatch plan and worker argv as the shell/Kubernetes path, so
local testing stays aligned with scale-out execution.

While workers are active, the local runner also refreshes
`dispatch_status.json` from per-worker `run_status.json` files. Tune this with
`--status-snapshot-interval-sec`; use `0` to write only the final snapshot after
workers exit.

Use `--resume` to skip workers that already have a complete `run_summary.json`:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky run-local-dispatch \
  --manifest runs/manifest_smoke_v2/frame_manifest.parquet \
  --projected-targets runs/projected_targets_smoke_current/frame_targets_projected.parquet \
  --out-dir runs/local_dispatch_smoke2 \
  --run-id local_dispatch_smoke2 \
  --devices cuda:0 \
  --limit-frames 2 \
  --resume \
  --finalize-device cuda:0
```

Resume mode still rebuilds the plan and finalizes the run, but complete workers
are not relaunched. Missing, incomplete, or failed workers are launched normally.

## Status Snapshots

Workers atomically rewrite their own `run_status.json`. To avoid dashboard code
scanning every worker directory, write one aggregate snapshot:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky dispatch-status \
  --plan runs/local_dispatch_smoke2/dispatch_plan.json
```

This writes `dispatch_status.json` next to the dispatch plan unless `--out` is
provided. It reads only small JSON files and writes via
`dispatch_status.json.tmp -> dispatch_status.json`.

Snapshot fields include:

```text
worker_count
active_workers
complete_workers
missing_workers
errored_workers
completed_frames
frame_count
progress_fraction
measurement_rows
ok_measurement_rows
queued_shard_writes
workers[]
```

The command is safe to run every few hundred milliseconds from a dashboard or
control-plane loop; it does not touch measurement shards or parquet data.

## Kubernetes Job Manifest

The same dispatch plan can be converted into Kubernetes Jobs:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky write-k8s-jobs \
  --plan runs/dispatch_smoke10_materialized2/dispatch_plan.json \
  --out-dir runs/dispatch_smoke10_materialized2/k8s \
  --image luxquarry-allsky:local \
  --namespace luxquarry \
  --container-executable luxquarry-allsky \
  --working-dir /workspace/luxquarry_allsky_engine \
  --pvc-name luxquarry-data \
  --mount-path /workspace \
  --env LUXQUARRY_MODE=smoke
```

This writes:

```text
<run_id>.worker-jobs.yaml
k8s_jobs_summary.json
```

The YAML intentionally uses JSON-shaped YAML documents so validation does not
require another dependency. Each Job preserves the worker argument vector from
the dispatch plan, requests `nvidia.com/gpu`, and optionally mounts a PVC. In
materialized mode each Job points at a per-worker manifest/target parquet and
runs as `--worker-index 0 --worker-count 1`.

This is still the baseline measurement dispatch layer. It must be followed by
spectra assembly, scoring, injection/recovery jobs, and viewer index builds for
a complete campaign.

## Campaign Stage Contract

After dispatch collection and spectra assembly, write a campaign contract:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky write-campaign-contract \
  --campaign-id dispatch_smoke10_materialized2_contract \
  --out runs/dispatch_smoke10_materialized2/campaign_contract.json \
  --baseline-plan runs/dispatch_smoke10_materialized2/dispatch_plan.json \
  --baseline-spectra-dir runs/dispatch_smoke10_materialized2/spectra_fast
```

The contract is deliberately simple JSON. It lists expected artifacts for:

```text
baseline_dispatch
baseline_spectra_assembly
baseline_blind_scoring
injected_dispatch
injected_spectra_assembly
injected_blind_scoring
truth_target_recovery
viewer_indexes
```

Each stage is marked `complete`, `missing`, or `blocked`. This gives the
next-gen runner and future dashboard one place to answer, "is this campaign
science complete?" without accidentally ignoring injected recovery.

For normal post-worker operation, prefer the combined finalizer:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky finalize-dispatch-run \
  --plan runs/dispatch_smoke10_materialized2/dispatch_plan.json \
  --spectra-out-dir runs/dispatch_smoke10_materialized2/spectra_finalize \
  --spectra-run-id dispatch_smoke10_materialized2_finalize \
  --campaign-id dispatch_smoke10_materialized2_finalize \
  --campaign-contract-out runs/dispatch_smoke10_materialized2/campaign_contract_finalize.json \
  --device cuda:0 \
  --score-baseline
```

This runs `collect-dispatch-run`, `assemble-spectra`, and
`write-campaign-contract` in one command. With `--score-baseline`, it also runs
the simple cuDF target-zscore scorer and writes `baseline_candidates.parquet`
and `baseline_candidate_summary.json`. It fails closed if any worker or shard is
incomplete; use `--allow-incomplete` only for diagnostic partial-run products.

The baseline scorer is only the first candidate product. It is useful as a fast
wiring test because it preserves target, wavelength, flag, detector, and FITS
provenance through a RAPIDS postprocess stage. It is not the final narrowband
matched filter, and it does not replace injected dispatch, injected scoring, or
truth-target recovery.

Standalone candidate scoring:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky score-spectra-candidates \
  --spectra runs/local_dispatch_smoke2/spectra/local_dispatch_smoke2.spectra_measurements.parquet \
  --out-dir runs/local_dispatch_smoke2/candidates \
  --run-id local_dispatch_smoke2 \
  --device cuda:0
```

If an injected run has already produced spectra, the finalizer can score both
baseline and injected spectra and write manifest-based recovery products:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky finalize-dispatch-run \
  --plan runs/baseline_dispatch/dispatch_plan.json \
  --spectra-out-dir runs/baseline_dispatch/spectra \
  --spectra-run-id baseline_dispatch \
  --campaign-id injection_recovery_contract \
  --campaign-contract-out runs/baseline_dispatch/campaign_contract.json \
  --injected-plan runs/injected_dispatch/dispatch_plan.json \
  --injected-spectra-dir runs/injected_dispatch/spectra \
  --injection-truth /mnt/niroseti/spherex_cache/injection_campaigns/<campaign>/injection_manifest.json \
  --candidate-dir runs/baseline_dispatch/candidates \
  --score-baseline \
  --score-injected \
  --recover-injections \
  --device cuda:0
```

That writes:

```text
baseline_candidates.parquet
baseline_candidate_summary.json
injected_candidates.parquet
injected_candidate_summary.json
injection_recovery.parquet
false_positive_candidates.parquet
recovery_by_strength.parquet
recovery_by_line.parquet
truth_recovery_summary.json
false_positive_summary.json
```

The matching post-worker Kubernetes artifact is:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky write-k8s-postprocess-job \
  --plan runs/dispatch_smoke10_materialized2/dispatch_plan.json \
  --out-dir runs/dispatch_smoke10_materialized2/k8s \
  --image luxquarry-allsky:local \
  --namespace luxquarry \
  --container-executable luxquarry-allsky \
  --working-dir /workspace/luxquarry_allsky_engine \
  --pvc-name luxquarry-data \
  --mount-path /workspace \
  --campaign-id dispatch_smoke10_materialized2_finalize \
  --score-baseline
```

For injected/recovery campaigns, pass the same finalizer flags to
`write-k8s-postprocess-job`: `--injected-plan`, `--injected-spectra-dir`,
`--injection-truth`, `--candidate-dir`, `--score-injected`, and
`--recover-injections`. The generated Kubernetes job remains a single
post-worker `finalize-dispatch-run`, so local and EKS postprocess behavior stay
aligned.

When using `--working-dir /workspace/luxquarry_allsky_engine`, pass paths
relative to that directory (`runs/...`). Repo-root-prefixed paths will be wrong
inside the container.

After the workers finish, collect the run:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky collect-dispatch-run \
  --plan runs/dispatch_smoke10/dispatch_plan.json
```

This writes:

```text
aggregate_summary.json
measurement_shard_manifest.parquet
```

The collector does not combine measurement data. It validates worker summaries,
checks shard file existence, sums frame/row counts, and writes a parquet shard
manifest for downstream spectra assembly.

Then assemble target-ordered spectra:

```bash
cd luxquarry_allsky_engine
.venv/bin/luxquarry-allsky assemble-spectra \
  --shard-manifest runs/dispatch_smoke10/measurement_shard_manifest.parquet \
  --out-dir runs/dispatch_smoke10/spectra \
  --run-id dispatch_smoke10 \
  --device cuda:0
```

This is a RAPIDS/cuDF post-processing stage. It preserves ragged spectra as one
measurement row per wavelength sample, sorted by `catalog`, `target_id`,
`cwave_um`, `frame_group_id`, and `image_id`. It also writes a compact
`target_summary.parquet` with measurement counts, wavelength range, ok fraction,
and basic flux statistics.

## Injection and Recovery Contract

Frame-first survey mode still needs the same science gates as the current
campaign miner. A complete run should emit these data products:

```text
baseline measurement_shard_manifest.parquet
baseline spectra_measurements.parquet
baseline candidate tables
injected measurement_shard_manifest.parquet
injected spectra_measurements.parquet
injected candidate tables
truth recovery summary for injected targets
false-positive/candidate review indexes
```

The injected run should reuse the same target/materialized-input contract as the
baseline run. That keeps recovery comparisons honest: baseline, injected, raw
blind scoring, quality-gated blind scoring, and truth-target recovery all refer
to the same target IDs and frame provenance. The current next-gen layer can
score already-assembled injected spectra and join those candidates to an
existing injection manifest. It still does not generate or dispatch injected
FITS products itself, and the simple z-score scorer is not the final narrowband
matched filter.

## Current Benchmark

Smoke dataset:

```text
10 frames
5,000 projected rows
2,770 GPU measurement rows
2,766 ok measurements
```

Single persistent worker, one shard per frame:

```text
total wall: 2.39 sec
kernel: ~0.03 sec/frame
FITS read: ~0.11-0.13 sec/frame
table assembly: ~0.011 sec/frame after first frame
write shard: ~0.010 sec/frame after first frame
```

Single persistent worker, batched shards and FITS prefetch:

```text
command flags: --shard-batch-frames 5 --prefetch-frames 2
total wall: 1.16 sec
shards: 2
rows: 2,770
ok rows: 2,766
matched CPU ok rows: 2,766
flux p95 abs delta vs CPU: 0.025 uJy
```

The worker also supports `--status-interval-frames N` to avoid rewriting the
status JSON every frame on large runs. Use a small number for interactive local
smokes and a larger value for all-sky batch jobs.

The worker supports `--local-cache-dir PATH` for local FITS staging. A cache hit
does not copy the FITS again; it validates by file size and then reads from the
local path. Measurement rows retain both `fits_path` and `local_fits_path` so
the original source and staged read path are explicit.

The worker supports `--async-shard-writes` for queued cuDF parquet writes. The
worker still waits for every queued shard before writing the final summary, so a
completed run means all listed shards are durable. Status exposes
`queued_shard_writes`, and frame timings separate `shard_submit_wall_sec` from
actual shard `write_wall_sec`.

The worker supports `--batch-table-assembly` to defer cuDF table construction
until shard flush. In that mode, each frame keeps target metadata in pandas and
kernel results as CuPy device columns. The shard writer performs one
`cudf.from_pandas` plus one set of CuPy concatenations per shard batch instead
of one cuDF table build per frame.

Three-worker local dispatch over three GPUs:

```text
workers: 3
frame split: 4 / 3 / 3
shards: 10
rows: 2,770
ok rows: 2,766
failed frames: 0
```

Collected three-worker run:

```text
complete: true
complete_workers: 3
completed_frames: 10
measurement_rows: 2,770
ok_measurement_rows: 2,766
shard_count: 10
missing_shards: 0
collect_wall_sec: ~0.008 sec
```

Materialized three-worker smoke:

```text
input slices: 4 / 3 / 3 frames
projected target slices: 2,000 / 1,500 / 1,500 rows
complete: true
measurement_rows: 2,770
ok_measurement_rows: 2,766
worker_max_wall_sec: 0.811
shard_count: 3
missing_shards: 0
```

Spectra assembly over that materialized run:

```text
input_measurement_rows: 2,770
spectra_measurement_rows: 2,770
target_count: 720
total_wall_sec: 1.25
target_summary_wall_sec: 0.15
```

For this tiny smoke test, process startup dominates. The point of the worker
contract is long frame queues where startup and calibration upload are amortized.

## Current Bottleneck

The aperture kernel is no longer the dominant stage for this smoke. FITS reads
are. Prefetching overlaps much of that read time with GPU/table work, but it
does not eliminate storage bandwidth pressure. The next performance work should
focus on:

1. local NVMe staging
2. reducing FITS extension read overhead
3. avoiding per-frame metadata table construction where possible
4. writing larger shard batches tuned to downstream spectra assembly
5. evaluating KvikIO/local NVMe paths for staged FITS and output shards
6. larger benchmarks where async write overlap matters more than the tiny smoke

RAPIDS should remain the table/shard engine, while Warp/CUDA owns the aperture
kernel.

## Precision

The current hot path uses FP32 image/variance payloads and FP32 Warp kernels.
RTX 6000 Ada cards support FP64 functionally, but their FP64 throughput is far
below FP32. On the smoke benchmark, FP32 GPU photometry matches the CPU baseline
for 2,766 ok measurements with p95 aperture flux delta of about 0.025 uJy.

If we need numerical audits, add a small FP64 audit mode for selected frames and
targets. Do not make FP64 the default mining path unless it changes a science
decision.
