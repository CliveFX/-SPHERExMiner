#!/usr/bin/env python3
"""Export science embedding vectors from a trained ragged encoder checkpoint."""

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

from ml.common.ragged_spectra import FEATURE_COLUMNS, load_ragged_examples, now_status, write_status
from ml.science_embedding.train import RaggedSpectrumEncoder, _require_torch


def main() -> None:
    torch, nn, _F = _require_torch()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", default="all", choices=["all", "train", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-points", type=int, default=160)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "export_status.json"
    embeddings_path = args.output_dir / "embeddings.parquet"
    model_card_path = args.output_dir / "embedding_export_card.json"

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    embedding_dim = int(checkpoint.get("embedding_dim") or 64)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = RaggedSpectrumEncoder(
        torch=torch,
        nn=nn,
        input_dim=len(FEATURE_COLUMNS),
        hidden_dim=128,
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
    examples = load_ragged_examples(
        args.dataset_dir,
        point_table="science_points.parquet",
        target_table="science_targets.parquet",
        split_id=args.split,
        quality_categories={"good", "review"},
    )

    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(_batches(examples, args.batch_size), start=1):
            x, mask = _collate(batch, args.max_points, torch, device)
            emb = model(x, mask).detach().cpu().numpy()
            for example, vector in zip(batch, emb, strict=True):
                row: dict[str, object] = {
                    "dataset_name": example.dataset_name,
                    "run_name": example.run_name,
                    "run_kind": example.run_kind,
                    "split_id": example.split_id,
                    "target_id": example.target_id,
                    "source_id": example.source_id,
                    "n_input_points": int(len(example.features)),
                }
                for i, value in enumerate(vector):
                    row[f"shape_embedding_{i:03d}"] = float(value)
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
            "spectrum_quality_score",
            "spectrum_quality_category",
            "n_measurements",
            "n_usable_measurements",
            "flag_fraction",
            "smoothness_score",
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


def _collate(examples, max_points: int, torch, device):
    views = [example.features[:max_points] if len(example.features) > max_points else example.features for example in examples]
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
        yield items[index : index + batch_size]


if __name__ == "__main__":
    main()
