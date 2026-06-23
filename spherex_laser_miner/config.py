from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


DEFAULT_CACHE_ROOT = Path("/mnt/niroseti/spherex_cache")


class MinerConfig(BaseModel):
    cache_root: Path = Field(default_factory=lambda: Path(os.getenv("SPHEREX_CACHE_ROOT", DEFAULT_CACHE_ROOT)))
    run_name: str = Field(default_factory=lambda: os.getenv("SPHEREX_RUN_NAME", "smoke_simp_field"))
    release: str = Field(default_factory=lambda: os.getenv("SPHEREX_RELEASE", "qr2"))
    filter_profile: str = Field(default_factory=lambda: os.getenv("FILTER_PROFILE", "broad_debug"))
    photometry_backend: Literal["cpu_numpy", "warp_calibrated"] = "cpu_numpy"
    psf_photometry_backend: Literal["cpu_single", "warp_grid"] = "cpu_single"
    psf_kernel_build_mode: Literal["cpu_scipy", "gpu_bilinear", "gpu_spline"] = "gpu_spline"
    psf_grid_half_range_pix: float = 1.0
    psf_grid_step_pix: float = 0.5
    psf_grid_metric: Literal["snr", "chi2"] = "snr"
    psf_kernel_radius_native: int = 5
    status_mode: Literal["jsonl", "off"] = "jsonl"
    warp_devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2")
    enable_psf_photometry: bool = False
    enable_diagnostic_aperture: bool = False
    aperture_radius_pix: float = 2.0
    annulus_inner_pix: float = 4.0
    annulus_outer_pix: float = 6.0
    edge_margin_pix: float = 8.0
    fatal_flag_bits: tuple[int, ...] = (0, 1, 2, 4, 6, 7, 9, 10, 11, 14, 15, 17, 19, 22, 24, 26, 27, 28, 29)

    @property
    def docs_dir(self) -> Path:
        return self.cache_root / "external" / "docs"

    @property
    def spexpi_dir(self) -> Path:
        return self.cache_root / "external" / "source" / "spexpi"

    @property
    def raw_level2_dir(self) -> Path:
        return self.cache_root / "raw" / self.release / "level2"

    @property
    def manual_targets_path(self) -> Path:
        return Path("configs/manual_targets.yaml")

    @property
    def smoke_run_dir(self) -> Path:
        return self.cache_root / "runs" / self.run_name


def load_config(cache_root: Path | None = None) -> MinerConfig:
    cfg = MinerConfig()
    if cache_root is not None:
        cfg.cache_root = cache_root
    return cfg
