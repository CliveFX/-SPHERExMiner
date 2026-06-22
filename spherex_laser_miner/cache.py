from __future__ import annotations

import hashlib
from pathlib import Path


def ensure_cache_dirs(cache_root: Path) -> None:
    for relative in (
        "external/docs",
        "external/source",
        "raw/qr2/level2",
        "manifests",
        "manual_targets",
        "derived/measurements",
        "derived/qa",
        "runs",
    ):
        (cache_root / relative).mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def cache_path_for_access_url(cache_root: Path, access_url: str) -> Path:
    marker = "/spherex/qr2/level2/"
    if marker in access_url:
        suffix = access_url.split(marker, 1)[1]
        return cache_root / "raw" / "qr2" / "level2" / suffix
    return cache_root / "raw" / "qr2" / "level2" / Path(access_url).name
