#!/usr/bin/env python3
"""Build a small UMAP/clustering artifact for browser visualization."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=30000)
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "umap_status.json"
    _write_status(status_path, {"status": "reading", "elapsed_sec": 0.0})

    df = pd.read_parquet(args.embeddings)
    embedding_cols = sorted([col for col in df.columns if col.startswith("shape_embedding_")])
    if not embedding_cols:
        raise SystemExit(f"No shape_embedding_* columns in {args.embeddings}")

    rng = np.random.default_rng(args.seed)
    if len(df) > args.sample_size:
        sample_idx = rng.choice(len(df), size=args.sample_size, replace=False)
        sample_idx.sort()
        work = df.iloc[sample_idx].copy()
    else:
        work = df.copy()
    matrix = work[embedding_cols].to_numpy(dtype=np.float32)
    matrix = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)

    _write_status(status_path, {"status": "clustering", "rows": int(len(work)), "elapsed_sec": time.perf_counter() - started})
    cluster_count = min(args.clusters, max(2, len(work) // 20))
    cluster_model = MiniBatchKMeans(n_clusters=cluster_count, random_state=args.seed, batch_size=4096, n_init="auto")
    clusters = cluster_model.fit_predict(matrix)

    _write_status(status_path, {"status": "projecting", "rows": int(len(work)), "elapsed_sec": time.perf_counter() - started})
    try:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=args.neighbors,
            min_dist=args.min_dist,
            metric="cosine",
            random_state=args.seed,
            low_memory=True,
        )
        projection = reducer.fit_transform(matrix)
        projection_method = "umap"
    except Exception:
        projection = PCA(n_components=2, random_state=args.seed).fit_transform(StandardScaler().fit_transform(matrix))
        projection_method = "pca_fallback"

    out = work[_display_cols(work)].copy()
    out.insert(0, "projection_method", projection_method)
    out.insert(1, "sampled", len(df) > len(work))
    out.insert(2, "embedding_row_count", int(len(df)))
    out["umap_x"] = projection[:, 0].astype(float)
    out["umap_y"] = projection[:, 1].astype(float)
    out["cluster_id"] = clusters.astype(int)
    out["cluster_size"] = pd.Series(clusters).map(pd.Series(clusters).value_counts()).to_numpy(dtype=int)

    projection_path = args.output_dir / "umap_projection.parquet"
    summary_path = args.output_dir / "umap_summary.json"
    out.to_parquet(projection_path, index=False)
    summary = {
        "status": "done",
        "embeddings": str(args.embeddings),
        "projection_path": str(projection_path),
        "row_count": int(len(out)),
        "embedding_row_count": int(len(df)),
        "sample_size": int(args.sample_size),
        "sampled": bool(len(df) > len(work)),
        "cluster_count": int(cluster_count),
        "projection_method": projection_method,
        "elapsed_sec": time.perf_counter() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_status(status_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _display_cols(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in (
            "run_name",
            "run_kind",
            "target_id",
            "source_id",
            "object_name",
            "split_id",
            "n_input_points",
            "phot_g_mean_mag",
            "bp_rp",
            "parallax_mas",
            "spectrum_quality_score",
            "spectrum_quality_category",
            "n_measurements",
            "flag_fraction",
            "smoothness_score",
            "campaign",
            "mag_bin",
            "grid_tile_id",
            "grid_nside",
            "grid_order",
            "grid_hpx",
            "grid_batch_index",
        )
        if col in df.columns
    ]


def _write_status(path: Path, payload: dict[str, object]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


if __name__ == "__main__":
    main()
