from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import quote, urlparse


def is_s3_uri(value: str) -> bool:
    return value.startswith("s3://")


def stage_input_file(source: str | Path, cache_dir: Path, *, s3_region: str = "us-east-1") -> tuple[Path, int]:
    source_text = str(source)
    if is_s3_uri(source_text):
        return stage_s3_uri(source_text, cache_dir, s3_region=s3_region)
    return stage_local_file(Path(source_text), cache_dir)


def stage_local_file(source_path: Path, cache_dir: Path) -> tuple[Path, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    size = source_path.stat().st_size
    digest = hashlib.sha1(str(source_path.resolve()).encode("utf-8")).hexdigest()[:16]
    dest = cache_dir / f"{digest}_{source_path.name}"
    if dest.exists() and dest.stat().st_size == size:
        return dest, 0
    tmp = cache_dir / f".{source_path.name}.{os.getpid()}.tmp"
    shutil.copyfile(source_path, tmp)
    tmp.replace(dest)
    return dest, int(size)


def stage_s3_uri(uri: str, cache_dir: Path, *, s3_region: str = "us-east-1") -> tuple[Path, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    https_url = s3_uri_to_https_url(uri, region=s3_region)
    parsed = urlparse(uri)
    name = Path(parsed.path).name or "object"
    digest = hashlib.sha1(uri.encode("utf-8")).hexdigest()[:16]
    dest = cache_dir / f"{digest}_{name}"
    expected_size = _content_length(https_url)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        return dest, 0
    tmp = cache_dir / f".{name}.{os.getpid()}.tmp"
    bytes_written = _download_https(https_url, tmp)
    if expected_size is not None and bytes_written != expected_size:
        tmp.unlink(missing_ok=True)
        raise IOError(f"Downloaded {bytes_written} bytes for {uri}, expected {expected_size}")
    tmp.replace(dest)
    return dest, int(bytes_written)


def s3_uri_to_https_url(uri: str, *, region: str = "us-east-1") -> str:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid S3 URI: {uri}")
    key = quote(parsed.path.lstrip("/"))
    if region == "us-east-1":
        return f"https://{parsed.netloc}.s3.amazonaws.com/{key}"
    return f"https://{parsed.netloc}.s3.{region}.amazonaws.com/{key}"


def _content_length(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.headers.get("Content-Length")
    except Exception:
        return None
    return int(raw) if raw is not None else None


def _download_https(url: str, dest: Path) -> int:
    bytes_written = 0
    with urllib.request.urlopen(url, timeout=120) as response, dest.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            bytes_written += len(chunk)
    return bytes_written
