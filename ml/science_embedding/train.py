#!/usr/bin/env python3
"""Train a ragged self-supervised science embedding prototype."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.common.ragged_spectra import (
    FEATURE_COLUMNS,
    append_jsonl,
    load_ragged_examples,
    now_status,
    random_view,
    write_status,
)


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - depends on optional local install
        raise SystemExit(
            "PyTorch is required for ML training in this prototype. "
            "Install a CUDA-enabled torch build in the project venv, then rerun."
        ) from exc
    return torch, nn, F


def main() -> None:
    torch, nn, F = _require_torch()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/niroseti/spherex_cache/ml_runs"))
    parser.add_argument("--model-version", default="science_embedding_v0")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-points", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=0.15)
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

    dataset_name = args.dataset_dir.name
    examples = load_ragged_examples(
        args.dataset_dir,
        point_table="science_points.parquet",
        target_table="science_targets.parquet",
        split_id=args.split,
        quality_categories={"good", "review"},
        max_targets=args.max_targets,
    )
    if len(examples) < 2:
        raise SystemExit("Need at least two spectra examples for contrastive training")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = RaggedSpectrumEncoder(
        torch=torch,
        nn=nn,
        input_dim=len(FEATURE_COLUMNS),
        hidden_dim=128,
        embedding_dim=args.embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)

    write_status(
        status_path,
        now_status(
            run_name=args.run_name,
            model_type="science_embedding",
            status="running",
            dataset_name=dataset_name,
            model_version=args.model_version,
            started=started,
            examples=len(examples),
            device=str(device),
        ),
    )

    best_loss = float("inf")
    step = 0
    last_metric: dict[str, float | int | str] = {}
    for epoch in range(1, args.epochs + 1):
        random.shuffle(examples)
        for batch_examples in _batches(examples, args.batch_size):
            step += 1
            view_a, mask_a = _collate_views(batch_examples, args.max_points, rng, torch, device)
            view_b, mask_b = _collate_views(batch_examples, args.max_points, rng, torch, device)
            emb_a = model(view_a, mask_a)
            emb_b = model(view_b, mask_b)
            loss, retrieval_top1 = contrastive_loss(emb_a, emb_b, args.temperature, torch, F)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            metric = {
                "step": step,
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu()),
                "contrastive_retrieval_top1": float(retrieval_top1),
                "examples_per_sec": float(len(batch_examples) / max(time.perf_counter() - started, 1e-6) * step),
                "device": str(device),
            }
            append_jsonl(metrics_path, metric)
            last_metric = metric
            if metric["train_loss"] < best_loss:
                best_loss = metric["train_loss"]
                checkpoint_path = checkpoint_dir / "best.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_version": args.model_version,
                        "feature_columns": FEATURE_COLUMNS,
                        "embedding_dim": args.embedding_dim,
                        "epoch": epoch,
                        "step": step,
                        "loss": best_loss,
                    },
                    checkpoint_path,
                )
            write_status(
                status_path,
                now_status(
                    run_name=args.run_name,
                    model_type="science_embedding",
                    status="running",
                    dataset_name=dataset_name,
                    model_version=args.model_version,
                    started=started,
                    epoch=epoch,
                    step=step,
                    examples=len(examples),
                    latest_train_loss=metric["train_loss"],
                    contrastive_retrieval_top1=metric["contrastive_retrieval_top1"],
                    best_checkpoint=str(checkpoint_dir / "best.pt"),
                    device=str(device),
                ),
            )

    model_card = {
        "run_name": args.run_name,
        "model_type": "science_embedding",
        "model_version": args.model_version,
        "dataset_dir": str(args.dataset_dir),
        "feature_columns": FEATURE_COLUMNS,
        "embedding_dim": args.embedding_dim,
        "example_count": len(examples),
        "best_loss": best_loss,
        "status_path": str(status_path),
        "metrics_path": str(metrics_path),
        "checkpoint_path": str(checkpoint_dir / "best.pt"),
    }
    (run_dir / "model_card.json").write_text(json.dumps(model_card, indent=2, sort_keys=True), encoding="utf-8")
    write_status(
        status_path,
        now_status(
            run_name=args.run_name,
            model_type="science_embedding",
            status="done",
            dataset_name=dataset_name,
            model_version=args.model_version,
            started=started,
            best_checkpoint=str(checkpoint_dir / "best.pt"),
            latest_train_loss=best_loss,
            epoch=last_metric.get("epoch"),
            step=last_metric.get("step"),
            contrastive_retrieval_top1=last_metric.get("contrastive_retrieval_top1"),
            examples=len(examples),
            device=str(device),
        ),
    )
    print(json.dumps(model_card, indent=2, sort_keys=True), flush=True)


class RaggedSpectrumEncoder:
    def __new__(cls, *, torch, nn, input_dim: int, hidden_dim: int, embedding_dim: int):
        class _Encoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.point = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                )
                self.out = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, embedding_dim),
                )

            def forward(self, x, mask):
                h = self.point(x)
                mask_f = mask.unsqueeze(-1).to(h.dtype)
                mean = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
                h_masked = h.masked_fill(~mask.unsqueeze(-1), -1e9)
                max_pool = h_masked.max(dim=1).values
                emb = self.out(torch.cat([mean, max_pool], dim=-1))
                return torch.nn.functional.normalize(emb, dim=-1)

        return _Encoder()


def contrastive_loss(emb_a, emb_b, temperature: float, torch, F):
    logits = emb_a @ emb_b.T / temperature
    labels = torch.arange(len(emb_a), device=emb_a.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    top1 = (logits.argmax(dim=1) == labels).float().mean().detach().cpu().item()
    return loss, top1


def _collate_views(examples, max_points: int, rng: np.random.Generator, torch, device):
    views = [random_view(example.features, max_points=max_points, keep_fraction=0.72, rng=rng) for example in examples]
    width = views[0].shape[1]
    max_len = max(len(view) for view in views)
    x = np.zeros((len(views), max_len, width), dtype=np.float32)
    mask = np.zeros((len(views), max_len), dtype=bool)
    for i, view in enumerate(views):
        x[i, : len(view)] = view
        mask[i, : len(view)] = True
    return torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device)


def _batches(items, batch_size: int):
    for index in range(0, len(items), batch_size):
        batch = items[index : index + batch_size]
        if len(batch) >= 2:
            yield batch


if __name__ == "__main__":
    main()
