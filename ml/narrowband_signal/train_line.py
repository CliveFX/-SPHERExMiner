#!/usr/bin/env python3
"""Train a ragged narrowband line/no-line model.

This is the supervised counterpart to the science embedding model. It consumes
the ML dataset builder's `narrowband_*` tables and learns two targets:

- objectness: whether the target has an injected narrowband line;
- line wavelength: injected wavelength in nm, trained only on positives.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.common.ragged_spectra import FEATURE_COLUMNS, append_jsonl, make_point_features, now_status, write_status


@dataclass
class LineExample:
    dataset_name: str
    run_name: str
    split_id: str
    target_id: str
    is_positive: float
    line_nm: float
    features: np.ndarray


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover
        raise SystemExit("PyTorch is required for narrowband line training.") from exc
    return torch, nn, F


def main() -> None:
    torch, nn, F = _require_torch()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, help="Prepared cache from build_feature_cache.py. Avoids parquet/groupby feature construction.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/niroseti/spherex_cache/ml_runs"))
    parser.add_argument("--model-version", default="narrowband_line_v0")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test", "all"])
    parser.add_argument("--quality-category", action="append", choices=["good", "review", "bad"], default=["good", "review"])
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--max-points", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--line-min-nm", type=float, default=700.0)
    parser.add_argument("--line-max-nm", type=float, default=5000.0)
    parser.add_argument("--line-loss-weight", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "training_status.json"
    metrics_path = run_dir / "training_metrics.jsonl"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    if args.feature_cache_dir:
        examples = _load_cached_examples(args.feature_cache_dir, max_targets=args.max_targets)
    else:
        examples = _load_examples(
            args.dataset_dir,
            split_id=args.split,
            quality_categories=set(args.quality_category or []),
            max_targets=args.max_targets,
            line_min_nm=args.line_min_nm,
            line_max_nm=args.line_max_nm,
        )
    positives = sum(1 for example in examples if example.is_positive > 0.5)
    negatives = len(examples) - positives
    if positives < 2 or negatives < 2:
        raise SystemExit(f"Need both positive and negative examples; got positives={positives} negatives={negatives}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = RaggedLineModel(
        torch=torch,
        nn=nn,
        input_dim=len(FEATURE_COLUMNS),
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
    ).to(device)
    pos_weight = torch.tensor([negatives / max(positives, 1)], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    write_status(
        status_path,
        now_status(
            run_name=args.run_name,
            model_type="narrowband_line",
            status="running",
            dataset_name=args.dataset_dir.name,
            model_version=args.model_version,
            started=started,
            examples=len(examples),
            positive_examples=positives,
            negative_examples=negatives,
            device=str(device),
        ),
    )

    best_score = float("inf")
    last_metric: dict[str, float | int | str] = {}
    step = 0
    for epoch in range(1, args.epochs + 1):
        random.shuffle(examples)
        for batch_examples in _batches(examples, args.batch_size):
            step += 1
            x, mask, y, line_norm = _collate(
                batch_examples,
                args.max_points,
                args.line_min_nm,
                args.line_max_nm,
                torch,
                device,
            )
            object_logits, pred_line_norm = model(x, mask)
            object_loss = F.binary_cross_entropy_with_logits(object_logits, y, pos_weight=pos_weight)
            positive_mask = y > 0.5
            if bool(positive_mask.any()):
                line_loss = F.smooth_l1_loss(pred_line_norm[positive_mask], line_norm[positive_mask])
            else:
                line_loss = object_loss.detach() * 0.0
            loss = object_loss + float(args.line_loss_weight) * line_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            metric = _metrics(
                object_logits.detach(),
                pred_line_norm.detach(),
                y.detach(),
                line_norm.detach(),
                args.line_min_nm,
                args.line_max_nm,
                torch,
            )
            metric.update(
                {
                    "step": step,
                    "epoch": epoch,
                    "train_loss": float(loss.detach().cpu()),
                    "object_loss": float(object_loss.detach().cpu()),
                    "line_loss": float(line_loss.detach().cpu()),
                    "device": str(device),
                }
            )
            append_jsonl(metrics_path, metric)
            last_metric = metric
            rank_score = metric["train_loss"] + 0.001 * metric["positive_line_mae_nm"]
            if rank_score < best_score:
                best_score = rank_score
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_version": args.model_version,
                        "feature_columns": FEATURE_COLUMNS,
                        "embedding_dim": args.embedding_dim,
                        "line_min_nm": args.line_min_nm,
                        "line_max_nm": args.line_max_nm,
                        "epoch": epoch,
                        "step": step,
                        "rank_score": best_score,
                    },
                    checkpoint_dir / "best.pt",
                )
            write_status(
                status_path,
                now_status(
                    run_name=args.run_name,
                    model_type="narrowband_line",
                    status="running",
                    dataset_name=args.dataset_dir.name,
                    model_version=args.model_version,
                    started=started,
                    epoch=epoch,
                    step=step,
                    examples=len(examples),
                    positive_examples=positives,
                    negative_examples=negatives,
                    latest_train_loss=metric["train_loss"],
                    injected_detected_fraction=metric["injected_detected_fraction"],
                    baseline_false_alarm_fraction=metric["baseline_false_alarm_fraction"],
                    positive_line_mae_nm=metric["positive_line_mae_nm"],
                    best_checkpoint=str(checkpoint_dir / "best.pt"),
                    device=str(device),
                ),
            )

    model_card = {
        "run_name": args.run_name,
        "model_type": "narrowband_line",
        "model_version": args.model_version,
        "dataset_dir": str(args.dataset_dir),
        "feature_cache_dir": str(args.feature_cache_dir) if args.feature_cache_dir else None,
        "feature_columns": FEATURE_COLUMNS,
        "example_count": len(examples),
        "positive_example_count": positives,
        "negative_example_count": negatives,
        "checkpoint_path": str(checkpoint_dir / "best.pt"),
        "status_path": str(status_path),
        "metrics_path": str(metrics_path),
        "line_min_nm": args.line_min_nm,
        "line_max_nm": args.line_max_nm,
        "last_metric": last_metric,
    }
    (run_dir / "model_card.json").write_text(json.dumps(model_card, indent=2, sort_keys=True), encoding="utf-8")
    write_status(
        status_path,
        now_status(
            run_name=args.run_name,
            model_type="narrowband_line",
            status="done",
            dataset_name=args.dataset_dir.name,
            model_version=args.model_version,
            started=started,
            best_checkpoint=str(checkpoint_dir / "best.pt"),
            latest_train_loss=last_metric.get("train_loss"),
            injected_detected_fraction=last_metric.get("injected_detected_fraction"),
            baseline_false_alarm_fraction=last_metric.get("baseline_false_alarm_fraction"),
            positive_line_mae_nm=last_metric.get("positive_line_mae_nm"),
            examples=len(examples),
            positive_examples=positives,
            negative_examples=negatives,
            device=str(device),
        ),
    )
    print(json.dumps(model_card, indent=2, sort_keys=True), flush=True)


class RaggedLineModel:
    def __new__(cls, *, torch, nn, input_dim: int, hidden_dim: int, embedding_dim: int):
        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.point = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                )
                self.embed = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, embedding_dim),
                    nn.GELU(),
                )
                self.object_head = nn.Linear(embedding_dim, 1)
                self.line_head = nn.Sequential(nn.Linear(embedding_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))

            def forward(self, x, mask):
                h = self.point(x)
                mask_f = mask.unsqueeze(-1).to(h.dtype)
                mean = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
                h_masked = h.masked_fill(~mask.unsqueeze(-1), -1e9)
                max_pool = h_masked.max(dim=1).values
                emb = self.embed(torch.cat([mean, max_pool], dim=-1))
                return self.object_head(emb).squeeze(-1), self.line_head(emb).squeeze(-1)

        return _Model()


def _load_examples(
    dataset_dir: Path,
    *,
    split_id: str,
    quality_categories: set[str],
    max_targets: int | None,
    line_min_nm: float,
    line_max_nm: float,
) -> list[LineExample]:
    targets = pd.read_parquet(dataset_dir / "narrowband_targets.parquet")
    points = pd.read_parquet(dataset_dir / "narrowband_points.parquet")
    truth_path = dataset_dir / "injection_truth.parquet"
    truth = pd.read_parquet(truth_path) if truth_path.exists() else pd.DataFrame()
    if split_id != "all":
        targets = targets[targets["split_id"].astype(str).eq(split_id)].copy()
    if quality_categories and "spectrum_quality_category" in targets:
        targets = targets[targets["spectrum_quality_category"].astype(str).isin(quality_categories)].copy()
    if max_targets and len(targets) > max_targets:
        injected = targets[targets.get("is_injected_target", False).fillna(False).astype(bool)].copy()
        rest = targets[~targets["target_id"].astype(str).isin(set(injected["target_id"].astype(str)))].copy()
        rest = rest.sort_values(["spectrum_quality_score", "n_usable_measurements"], ascending=[False, False], na_position="last")
        targets = pd.concat([injected, rest.head(max(0, max_targets - len(injected)))], ignore_index=True)

    truth_map: dict[str, float] = {}
    if not truth.empty:
        truth = truth.dropna(subset=["target_id", "injected_line_nm"]).copy()
        truth = truth[(truth["injected_line_nm"] >= line_min_nm) & (truth["injected_line_nm"] <= line_max_nm)]
        for key, rows in truth.groupby(truth["run_name"].astype(str) + "::" + truth["target_id"].astype(str), sort=False):
            truth_map[str(key)] = float(pd.to_numeric(rows["injected_line_nm"], errors="coerce").median())

    targets["_example_key"] = targets["run_name"].astype(str) + "::" + targets["target_id"].astype(str)
    keep = set(targets["_example_key"].astype(str))
    points["_example_key"] = points["run_name"].astype(str) + "::" + points["target_id"].astype(str)
    points = points[points["_example_key"].isin(keep)].copy()
    meta = targets.set_index("_example_key").to_dict(orient="index")

    examples: list[LineExample] = []
    for example_key, rows in points.groupby("_example_key", sort=False):
        row = meta.get(str(example_key))
        if not row:
            continue
        features = make_point_features(rows)
        if len(features) < 8:
            continue
        line_nm = truth_map.get(str(example_key), float("nan"))
        positive = float(math.isfinite(line_nm))
        examples.append(
            LineExample(
                dataset_name=str(row.get("dataset_name") or ""),
                run_name=str(row.get("run_name") or ""),
                split_id=str(row.get("split_id") or ""),
                target_id=str(row.get("target_id") or ""),
                is_positive=positive,
                line_nm=line_nm if positive else 0.0,
                features=features,
            )
        )
    return examples


def _load_cached_examples(cache_dir: Path, *, max_targets: int | None) -> list[LineExample]:
    manifest_path = cache_dir / "cache_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing feature cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples: list[LineExample] = []
    for shard in manifest.get("shards", []):
        if shard.get("status") != "ok":
            continue
        path = Path(str(shard.get("path")))
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        x = data["x"]
        mask = data["mask"].astype(bool)
        y = data["y"]
        line_norm = data["line_norm"]
        target_ids = data["target_ids"]
        run_name = str(data["run_name"][0]) if "run_name" in data else path.stem
        for index in range(len(y)):
            features = x[index][mask[index]].astype(np.float32, copy=True)
            if len(features) < 8:
                continue
            line_nm = 0.0
            if float(y[index]) > 0.5:
                line_nm = float(manifest["line_min_nm"]) + float(line_norm[index]) * (
                    float(manifest["line_max_nm"]) - float(manifest["line_min_nm"])
                )
            examples.append(
                LineExample(
                    dataset_name=str(Path(str(manifest.get("dataset_dir", ""))).name),
                    run_name=run_name,
                    split_id=str(manifest.get("split", "")),
                    target_id=str(target_ids[index]),
                    is_positive=float(y[index]),
                    line_nm=line_nm,
                    features=features,
                )
            )
            if max_targets and len(examples) >= max_targets:
                return examples
    return examples


def _collate(examples, max_points: int, line_min_nm: float, line_max_nm: float, torch, device):
    views = [example.features[:max_points] if len(example.features) > max_points else example.features for example in examples]
    width = views[0].shape[1]
    max_len = max(len(view) for view in views)
    x = np.zeros((len(views), max_len, width), dtype=np.float32)
    mask = np.zeros((len(views), max_len), dtype=bool)
    y = np.zeros(len(views), dtype=np.float32)
    line_norm = np.zeros(len(views), dtype=np.float32)
    span = max(line_max_nm - line_min_nm, 1.0)
    for i, (example, view) in enumerate(zip(examples, views, strict=True)):
        x[i, : len(view)] = view
        mask[i, : len(view)] = True
        y[i] = example.is_positive
        if example.is_positive > 0.5:
            line_norm[i] = np.clip((example.line_nm - line_min_nm) / span, 0.0, 1.0)
    return torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device), torch.from_numpy(y).to(device), torch.from_numpy(line_norm).to(device)


def _metrics(logits, pred_line_norm, labels, line_norm, line_min_nm: float, line_max_nm: float, torch):
    probs = torch.sigmoid(logits)
    pred = probs >= 0.5
    actual = labels >= 0.5
    positive_count = int(actual.sum().detach().cpu())
    negative_count = int((~actual).sum().detach().cpu())
    detected = int((pred & actual).sum().detach().cpu())
    missed = int((~pred & actual).sum().detach().cpu())
    false_alarms = int((pred & ~actual).sum().detach().cpu())
    correct_negative = int((~pred & ~actual).sum().detach().cpu())
    span = line_max_nm - line_min_nm
    if positive_count:
        line_mae = torch.mean(torch.abs(pred_line_norm[actual] - line_norm[actual])).detach().cpu().item() * span
    else:
        line_mae = 0.0
    return {
        "injected_total": positive_count,
        "baseline_total": negative_count,
        "injected_detected": detected,
        "injected_missed": missed,
        "baseline_correct": correct_negative,
        "baseline_false_alarms": false_alarms,
        "injected_detected_fraction": detected / max(positive_count, 1),
        "baseline_false_alarm_fraction": false_alarms / max(negative_count, 1),
        "positive_line_mae_nm": float(line_mae),
        "mean_object_probability": float(probs.mean().detach().cpu()),
    }


def _batches(items, batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


if __name__ == "__main__":
    main()
