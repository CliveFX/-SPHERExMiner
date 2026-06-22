from __future__ import annotations

from pathlib import Path

import requests


def download_file(url: str, local_path: Path, redownload: bool = False) -> tuple[Path, str]:
    if local_path.exists() and local_path.stat().st_size > 0 and not redownload:
        return local_path, "cached"

    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    tmp_path.replace(local_path)
    return local_path, "downloaded"
