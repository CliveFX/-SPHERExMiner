#!/usr/bin/env python3
"""Export metadata-aware science embedding vectors from a v2 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.common.ragged_spectra import now_status, write_status
from ml.science_embedding.train import _require_torch
from ml.science_embedding.train_v2 import (
    HybridSpectrumEncoder,
    META_COLUMNS,
    _collate_hybrid,
    load_hybrid_examples,
)


def main() -> None:
    torch, nn, _F = _require_torch()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", default="all", choices=["all", "train", "validation", "test"])
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-points", type=int, default=320)
    parser.add_argument("--prep-workers", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "export_status.json"
    embeddings_path = args.output_dir / "embeddings.parquet"
    model_card_path = args.output_dir / "embedding_export_card.json"

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    embedding_dim = int(checkpoint.get("embedding_dim") or 96)
    meta_feature_names = list(checkpoint.get("metadata_feature_names") or [])
    if not meta_feature_names:
        raise SystemExit(f"{args.checkpoint} does not look like a v2 metadata-aware checkpoint")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = HybridSpectrumEncoder(
        torch=torch,
        nn=nn,
        input_dim=len(checkpoint.get("feature_columns") or []),
        meta_dim=len(meta_feature_names),
        hidden_dim=160,
        embedding_dim=embedding_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    write_status(
        status_path,
        now_status(
            run_name=args.run_name,
            model_type="science_embedding_export",
            status="loading_examples",
            dataset_name=args.dataset_dir.name,
            model_version=str(checkpoint.get("model_version") or "unknown"),
            started=started,
            device=str(device),
        ),
    )
    examples, _meta_stats = load_hybrid_examples(
        args.dataset_dir,
        split_id=args.split,
        max_targets=args.max_targets,
        quality_categories={"good", "review"},
        prep_workers=args.prep_workers,
    )

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(args.seed)
    with torch.no_grad():
        for batch_index, batch in enumerate(_batches(examples, args.batch_size), start=1):
            x, mask, meta, meta_target = _collate_hybrid(batch, args.max_points, rng, torch, device)
            emb, pred = model(x, mask, meta)
            emb_np = emb.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            target_np = meta_target.detach().cpu().numpy()
            for example, vector, pred_meta, target_meta in zip(batch, emb_np, pred_np, target_np, strict=True):
                row: dict[str, object] = {
                    "dataset_name": example.dataset_name,
                    "run_name": example.run_name,
                    "run_kind": example.run_kind,
                    "split_id": example.split_id,
                    "target_id": example.target_id,
                    "source_id": example.source_id,
                    "n_input_points": int(len(example.features)),
                    "embedding_family": "metadata_aware",
                }
                for i, value in enumerate(vector):
                    row[f"shape_embedding_{i:03d}"] = float(value)
                for col, pred_value, target_value in zip(META_COLUMNS, pred_meta, target_meta, strict=True):
                    row[f"pred_{col}_norm"] = float(pred_value)
                    row[f"target_{col}_norm"] = float(target_value)
                rows.append(row)
            if batch_index % 10 == 0:
                write_status(
                    status_path,
                    now_status(
                        run_name=args.run_name,
                        model_type="science_embedding_export",
                        status="running",
                        dataset_name=args.dataset_dir.name,
                        model_version=str(checkpoint.get("model_version") or "unknown"),
                        started=started,
                        examples=len(examples),
                        exported=len(rows),
                        device=str(device),
                    ),
                )

    embeddings = pd.DataFrame(rows)
    targets = pd.read_parquet(args.dataset_dir / "science_targets.parquet")
    join_cols = [
        col
        for col in (
            "run_name",
            "target_id",
            "object_name",
            "phot_g_mean_mag",
            "phot_bp_mean_mag",
            "phot_rp_mean_mag",
            "bp_rp",
            "parallax_mas",
            "ruwe",
            "spectrum_quality_score",
            "spectrum_quality_category",
            "n_measurements",
            "n_usable_measurements",
            "flag_fraction",
            "smoothness_score",
            "aperture_psf_corr",
            "median_abs_aperture_snr",
            "campaign",
            "mag_bin",
            "grid_tile_id",
            "grid_nside",
            "grid_order",
            "grid_hpx",
            "grid_batch_index",
        )
        if col in targets.columns
    ]
    if join_cols:
        embeddings = embeddings.merge(targets[join_cols], on=["run_name", "target_id"], how="left")
    embeddings.to_parquet(embeddings_path, index=False)

    card = {
        "run_name": args.run_name,
        "dataset_dir": str(args.dataset_dir),
        "checkpoint": str(args.checkpoint),
        "embedding_dim": embedding_dim,
        "metadata_columns": META_COLUMNS,
        "split": args.split,
        "example_count": int(len(examples)),
        "output": str(embeddings_path),
        "elapsed_sec": time.perf_counter() - started,
        "device": str(device),
    }
    model_card_path.write_text(json.dumps(card, indent=2, sort_keys=True), encoding="utf-8")
    write_status(
        status_path,
        now_status(
            run_name=args.run_name,
            model_type="science_embedding_export",
            status="done",
            dataset_name=args.dataset_dir.name,
            model_version=str(checkpoint.get("model_version") or "unknown"),
            started=started,
            examples=len(examples),
            exported=len(rows),
            output=str(embeddings_path),
            device=str(device),
        ),
    )
    print(json.dumps(card, indent=2, sort_keys=True), flush=True)


def _batches(items, batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


if __name__ == "__main__":
    main()
