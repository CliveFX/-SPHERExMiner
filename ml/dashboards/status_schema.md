# ML Training Dashboard Status Schema

Training jobs should write small status files that the web viewer can poll
cheaply.

## `training_status.json`

```json
{
  "run_name": "science_embedding_smoke_001",
  "model_type": "science_embedding",
  "status": "running",
  "dataset_name": "cvj_smoke_v1",
  "model_version": "science_embedding_v0",
  "epoch": 3,
  "step": 1200,
  "examples_per_sec": 8421.5,
  "elapsed_sec": 391.2,
  "eta_sec": 920.0,
  "latest_train_loss": 0.82,
  "latest_validation_loss": 0.91,
  "best_checkpoint": "checkpoints/best.pt",
  "updated_at": "2026-06-24T00:00:00Z"
}
```

## `training_metrics.jsonl`

One JSON object per line:

```json
{"step": 100, "epoch": 1, "train_loss": 1.42, "validation_loss": 1.55, "examples_per_sec": 8012.3}
```

Narrowband jobs should also emit recovery and false-positive metrics:

```json
{"step": 100, "recovery_fraction": 0.87, "false_positives_per_1000_targets": 1.4}
```
