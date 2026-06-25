#!/usr/bin/env python3
"""Train a metadata-aware science embedding.

This v2 model keeps the ragged spectrum contrastive objective from v0, but
also feeds Gaia/quality metadata into the encoder and trains auxiliary heads
to preserve those coordinates in the learned embedding.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
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

from ml.common.ragged_spectra import FEATURE_COLUMNS, append_jsonl, make_point_features, now_status, random_view, write_status
from ml.science_embedding.train import contrastive_loss, _batches, _require_torch


META_COLUMNS = [
    "phot_g_mean_mag",
    "bp_rp",
    "parallax_mas",
    "spectrum_quality_score",
    "flag_fraction",
    "smoothness_score",
    "aperture_psf_corr",
    "median_abs_aperture_snr",
    "n_usable_measurements",
]


@dataclass(frozen=True)
class HybridExample:
    dataset_name: str
    run_name: str
    run_kind: str
    split_id: str
    target_id: str
    source_id: str | None
    features: np.ndarray
    meta: np.ndarray
    meta_target: np.ndarray


def main() -> None:
    torch, nn, F = _require_torch()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/niroseti/spherex_cache/ml_runs"))
    parser.add_argument("--model-version", default="science_embedding_v2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-points", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=0.12)
    parser.add_argument("--metadata-loss-weight", type=float, default=0.35)
    parser.add_argument("--prep-workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
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

    examples, meta_stats = load_hybrid_examples(
        args.dataset_dir,
        split_id=args.split,
        max_targets=args.max_targets,
        quality_categories={"good", "review"},
        prep_workers=args.prep_workers,
    )
    if len(examples) < 2:
        raise SystemExit("Need at least two spectra examples for contrastive training")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = HybridSpectrumEncoder(
        torch=torch,
        nn=nn,
        input_dim=len(FEATURE_COLUMNS),
        meta_dim=len(_meta_feature_names()),
        hidden_dim=160,
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
            dataset_name=args.dataset_dir.name,
            model_version=args.model_version,
            started=started,
            examples=len(examples),
            prep_workers=args.prep_workers,
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
            view_a, mask_a, meta, meta_target = _collate_hybrid(batch_examples, args.max_points, rng, torch, device)
            view_b, mask_b, _meta_b, _target_b = _collate_hybrid(batch_examples, args.max_points, rng, torch, device)
            emb_a, pred_a = model(view_a, mask_a, meta)
            emb_b, pred_b = model(view_b, mask_b, meta)
            loss_contrastive, retrieval_top1 = contrastive_loss(emb_a, emb_b, args.temperature, torch, F)
            loss_meta = 0.5 * (F.mse_loss(pred_a, meta_target) + F.mse_loss(pred_b, meta_target))
            loss = loss_contrastive + args.metadata_loss_weight * loss_meta

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            metric = {
                "step": step,
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu()),
                "contrastive_loss": float(loss_contrastive.detach().cpu()),
                "metadata_loss": float(loss_meta.detach().cpu()),
                "contrastive_retrieval_top1": float(retrieval_top1),
                "metadata_loss_weight": args.metadata_loss_weight,
                "device": str(device),
            }
            append_jsonl(metrics_path, metric)
            last_metric = metric
            if metric["train_loss"] < best_loss:
                best_loss = metric["train_loss"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_version": args.model_version,
                        "feature_columns": FEATURE_COLUMNS,
                        "metadata_columns": META_COLUMNS,
                        "metadata_feature_names": _meta_feature_names(),
                        "metadata_stats": meta_stats,
                        "embedding_dim": args.embedding_dim,
                        "epoch": epoch,
                        "step": step,
                        "loss": best_loss,
                    },
                    checkpoint_dir / "best.pt",
                )
            write_status(
                status_path,
                now_status(
                    run_name=args.run_name,
                    model_type="science_embedding",
                    status="running",
                    dataset_name=args.dataset_dir.name,
                    model_version=args.model_version,
                    started=started,
                    epoch=epoch,
                    step=step,
                    examples=len(examples),
                    latest_train_loss=metric["train_loss"],
                    contrastive_loss=metric["contrastive_loss"],
                    metadata_loss=metric["metadata_loss"],
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
        "metadata_columns": META_COLUMNS,
        "metadata_feature_names": _meta_feature_names(),
        "metadata_loss_weight": args.metadata_loss_weight,
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
            dataset_name=args.dataset_dir.name,
            model_version=args.model_version,
            started=started,
            best_checkpoint=str(checkpoint_dir / "best.pt"),
            latest_train_loss=best_loss,
            epoch=last_metric.get("epoch"),
            step=last_metric.get("step"),
            contrastive_loss=last_metric.get("contrastive_loss"),
            metadata_loss=last_metric.get("metadata_loss"),
            contrastive_retrieval_top1=last_metric.get("contrastive_retrieval_top1"),
            examples=len(examples),
            device=str(device),
        ),
    )
    print(json.dumps(model_card, indent=2, sort_keys=True), flush=True)


class HybridSpectrumEncoder:
    def __new__(cls, *, torch, nn, input_dim: int, meta_dim: int, hidden_dim: int, embedding_dim: int):
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
                self.meta = nn.Sequential(
                    nn.Linear(meta_dim, 64),
                    nn.GELU(),
                    nn.LayerNorm(64),
                    nn.Linear(64, 64),
                    nn.GELU(),
                )
                self.out = nn.Sequential(
                    nn.Linear(hidden_dim * 2 + 64, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, embedding_dim),
                )
                self.metadata_head = nn.Sequential(
                    nn.Linear(embedding_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, len(META_COLUMNS)),
                )

            def forward(self, x, mask, meta):
                h = self.point(x)
                mask_f = mask.unsqueeze(-1).to(h.dtype)
                mean = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
                h_masked = h.masked_fill(~mask.unsqueeze(-1), -1e9)
                max_pool = h_masked.max(dim=1).values
                meta_h = self.meta(meta)
                raw = self.out(torch.cat([mean, max_pool, meta_h], dim=-1))
                emb = torch.nn.functional.normalize(raw, dim=-1)
                return emb, self.metadata_head(emb)

        return _Encoder()


def load_hybrid_examples(
    dataset_dir: Path,
    *,
    split_id: str,
    max_targets: int | None,
    quality_categories: set[str] | None,
    prep_workers: int = 1,
    min_points: int = 8,
) -> tuple[list[HybridExample], dict[str, dict[str, float]]]:
    targets = pd.read_parquet(dataset_dir / "science_targets.parquet")
    points = pd.read_parquet(dataset_dir / "science_points.parquet")
    if split_id != "all":
        targets = targets[targets["split_id"].astype(str).eq(split_id)].copy()
    if quality_categories and "spectrum_quality_category" in targets:
        targets = targets[targets["spectrum_quality_category"].astype(str).isin(quality_categories)].copy()
    if max_targets is not None and len(targets) > max_targets:
        targets = targets.sort_values(
            ["spectrum_quality_score", "n_usable_measurements"],
            ascending=[False, False],
            na_position="last",
            kind="mergesort",
        ).head(max_targets)
    targets = targets.copy()
    targets["_example_key"] = targets["run_name"].astype(str) + "::" + targets["target_id"].astype(str)
    keep = set(targets["_example_key"])
    points = points.copy()
    points["_example_key"] = points["run_name"].astype(str) + "::" + points["target_id"].astype(str)
    points = points[points["_example_key"].isin(keep)].copy()

    meta_stats = _metadata_stats(targets)
    meta_by_key = targets.set_index("_example_key").to_dict(orient="index")
    grouped = points.groupby("_example_key", dropna=False, sort=False)
    tasks = ((str(example_key), rows.copy(deep=False)) for example_key, rows in grouped)
    examples: list[HybridExample] = []
    if prep_workers > 1:
        for example in _threaded_hybrid_examples(tasks, meta_by_key, meta_stats, split_id, min_points, prep_workers):
            if example is not None:
                examples.append(example)
    else:
        for item in tasks:
            example = _make_hybrid_example(item, meta_by_key, meta_stats, split_id, min_points)
            if example is not None:
                examples.append(example)
    return examples, meta_stats


def _threaded_hybrid_examples(
    tasks,
    meta_by_key: dict[str, dict[str, object]],
    meta_stats: dict[str, dict[str, float]],
    split_id: str,
    min_points: int,
    prep_workers: int,
):
    max_pending = max(prep_workers * 4, prep_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=prep_workers) as pool:
        pending: set[concurrent.futures.Future] = set()
        task_iter = iter(tasks)
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < max_pending:
                try:
                    item = next(task_iter)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(pool.submit(_make_hybrid_example, item, meta_by_key, meta_stats, split_id, min_points))
            if not pending:
                continue
            done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                yield future.result()


def _make_hybrid_example(
    item: tuple[str, pd.DataFrame],
    meta_by_key: dict[str, dict[str, object]],
    meta_stats: dict[str, dict[str, float]],
    split_id: str,
    min_points: int,
) -> HybridExample | None:
    example_key, rows = item
    meta = meta_by_key.get(str(example_key))
    if not meta:
        return None
    features = make_point_features(rows)
    if len(features) < min_points:
        return None
    meta_target = _metadata_vector(meta, meta_stats)
    meta_input = np.concatenate([meta_target, _metadata_missing_vector(meta)]).astype(np.float32)
    source_id = meta.get("source_id")
    return HybridExample(
        dataset_name=str(meta.get("dataset_name") or ""),
        run_name=str(meta.get("run_name") or rows["run_name"].iloc[0]),
        run_kind=str(meta.get("run_kind") or rows.get("run_kind", pd.Series(["unknown"])).iloc[0]),
        split_id=str(meta.get("split_id") or split_id),
        target_id=str(meta.get("target_id")),
        source_id=None if source_id is None or pd.isna(source_id) else str(source_id),
        features=features,
        meta=meta_input,
        meta_target=meta_target,
    )


def _metadata_stats(targets: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for col in META_COLUMNS:
        values = _metadata_series(targets, col)
        med = float(values.median(skipna=True))
        std = float(values.std(skipna=True))
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        if not np.isfinite(med):
            med = 0.0
        stats[col] = {"median": med, "std": std}
    return stats


def _metadata_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    if col == "parallax_mas":
        values = pd.to_numeric(df[col], errors="coerce")
        return np.log10(values.clip(lower=0.05))
    if col == "median_abs_aperture_snr":
        values = pd.to_numeric(df[col], errors="coerce")
        return np.log10(values.clip(lower=1e-3))
    if col == "n_usable_measurements":
        values = pd.to_numeric(df[col], errors="coerce")
        return np.log10(values.clip(lower=1.0))
    return pd.to_numeric(df[col], errors="coerce")


def _metadata_vector(meta: dict[str, object], stats: dict[str, dict[str, float]]) -> np.ndarray:
    values: list[float] = []
    one = pd.DataFrame([meta])
    for col in META_COLUMNS:
        raw = float(_metadata_series(one, col).iloc[0])
        med = stats[col]["median"]
        std = stats[col]["std"]
        if not np.isfinite(raw):
            raw = med
        values.append(float(np.clip((raw - med) / max(std, 1e-6), -6.0, 6.0)))
    return np.asarray(values, dtype=np.float32)


def _metadata_missing_vector(meta: dict[str, object]) -> np.ndarray:
    missing = []
    for col in META_COLUMNS:
        value = meta.get(col)
        missing.append(1.0 if value is None or pd.isna(value) else 0.0)
    return np.asarray(missing, dtype=np.float32)


def _meta_feature_names() -> list[str]:
    return [f"{col}_norm" for col in META_COLUMNS] + [f"{col}_missing" for col in META_COLUMNS]


def _collate_hybrid(examples, max_points: int, rng: np.random.Generator, torch, device):
    views = [random_view(example.features, max_points=max_points, keep_fraction=0.78, rng=rng) for example in examples]
    width = views[0].shape[1]
    max_len = max(len(view) for view in views)
    x = np.zeros((len(views), max_len, width), dtype=np.float32)
    mask = np.zeros((len(views), max_len), dtype=bool)
    meta = np.stack([example.meta for example in examples]).astype(np.float32)
    meta_target = np.stack([example.meta_target for example in examples]).astype(np.float32)
    for i, view in enumerate(views):
        x[i, : len(view)] = view
        mask[i, : len(view)] = True
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(mask).to(device),
        torch.from_numpy(meta).to(device),
        torch.from_numpy(meta_target).to(device),
    )


if __name__ == "__main__":
    main()
