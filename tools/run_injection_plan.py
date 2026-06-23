from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.inject_fake_line import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    DEFAULT_RELEASE,
    InjectionFrame,
    _choose_line_flux_uJy,
    _inject_one_frame,
    _load_target_rows,
    _required_float,
    _response_for_row,
    _safe_id,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def _campaign_fits_path(campaign_root: Path, image_id: str) -> Path:
    return campaign_root / "fits" / f"{_safe_id(image_id)}.fits"


def _prepare_copy(source_path: Path, output_path: Path, overwrite: bool) -> None:
    if output_path.exists():
        if not overwrite:
            return
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)


def _run_one_injection(
    injection: dict[str, Any],
    baseline_run_dir: Path,
    campaign_root: Path,
    cache_root: Path,
    release: str,
    kernel_radius_native: int,
    overwrite: bool,
    dry_run: bool,
    allow_approx_wavelengths: bool,
) -> dict[str, Any]:
    target_id = str(injection["target_id"])
    line_nm = float(injection["injected_line_nm"])
    line_um = line_nm / 1000.0
    line_width_nm = float(injection.get("line_width_nm", 1.0))
    line_width_um = line_width_nm / 1000.0
    min_response = float(injection.get("min_response", 1e-3))
    max_frames = injection.get("max_frames")

    rows = _load_target_rows(
        baseline_run_dir,
        target_id,
        allow_approx_wavelengths=allow_approx_wavelengths,
    )
    responses = np.array(
        [
            _response_for_row(
                _required_float(row, "cwave_um"),
                _required_float(row, "cband_um"),
                line_um,
                line_width_um,
            )
            for _, row in rows.iterrows()
        ],
        dtype=float,
    )
    line_flux_uJy, intensity = _choose_line_flux_uJy(
        rows=rows,
        responses=responses,
        line_flux_uJy=injection.get("line_flux_uJy"),
        find_me_snr=injection.get("find_me_snr"),
        min_response=min_response,
    )
    rows = rows.assign(line_response=responses)
    rows = rows[rows["line_response"] >= min_response].copy()
    if max_frames:
        rows = rows.sort_values("line_response", ascending=False).head(int(max_frames)).sort_values("cwave_um")

    frames: list[InjectionFrame] = []
    path_overrides: dict[str, str] = {}
    for frame_index, (_, row) in enumerate(rows.iterrows(), start=1):
        original_path = Path(str(row["input_file_path"]))
        image_id = str(row.get("image_id") or original_path.stem)
        output_path = _campaign_fits_path(campaign_root, image_id)
        detector = int(row["detector"])
        x_pix = _required_float(row, "x_pix")
        y_pix = _required_float(row, "y_pix")
        cwave_um = _required_float(row, "cwave_um")
        cband_um = _required_float(row, "cband_um")
        response = _required_float(row, "line_response")
        frame_flux_uJy = float(line_flux_uJy * response)

        print(
            f"{injection['injection_id']} [{frame_index}/{len(rows)}] {image_id} "
            f"response={response:.4g} inject={frame_flux_uJy:.3f}uJy",
            flush=True,
        )
        try:
            if not dry_run:
                _prepare_copy(original_path, output_path, overwrite=overwrite)
                source_for_update = output_path
            else:
                source_for_update = original_path
            injected_path, sector, psf_sum, psf_shape, sapm_path = _inject_one_frame(
                source_path=source_for_update,
                output_path=output_path,
                detector=detector,
                x_pix=x_pix,
                y_pix=y_pix,
                frame_flux_uJy=frame_flux_uJy,
                cache_root=cache_root,
                release=release,
                kernel_radius_native=kernel_radius_native,
                injection_id=str(injection["injection_id"]),
                dry_run=dry_run,
                overwrite=True,
            )
        except Exception as exc:
            frames.append(
                InjectionFrame(
                    image_id=image_id,
                    original_path=str(original_path),
                    injected_path=None,
                    detector=detector,
                    x_pix=x_pix,
                    y_pix=y_pix,
                    cwave_um=cwave_um,
                    cband_um=cband_um,
                    line_response=response,
                    injected_flux_uJy=frame_flux_uJy,
                    psf_sector=None,
                    psf_kernel_sum=None,
                    psf_kernel_shape=None,
                    sapm_path=None,
                    status="skipped",
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
            print(f"  skipped: {type(exc).__name__}: {exc}", flush=True)
            continue

        if injected_path:
            path_overrides[str(original_path)] = injected_path
        frames.append(
            InjectionFrame(
                image_id=image_id,
                original_path=str(original_path),
                injected_path=injected_path,
                detector=detector,
                x_pix=x_pix,
                y_pix=y_pix,
                cwave_um=cwave_um,
                cband_um=cband_um,
                line_response=response,
                injected_flux_uJy=frame_flux_uJy,
                psf_sector=sector,
                psf_kernel_sum=psf_sum,
                psf_kernel_shape=psf_shape,
                sapm_path=sapm_path,
                status="planned" if dry_run else "injected",
            )
        )

    return {
        **injection,
        "line_flux_uJy": line_flux_uJy,
        "intensity": intensity,
        "response_model": "gaussian_fwhm_sqrt_cband2_plus_line_width2",
        "frames_requested": int(len(rows)),
        "frames_written": int(sum(frame.status == "injected" for frame in frames)),
        "frames_planned": int(sum(frame.status == "planned" for frame in frames)),
        "frames_skipped": int(sum(frame.status == "skipped" for frame in frames)),
        "frames": [asdict(frame) for frame in frames],
        "path_overrides": path_overrides,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a fake-line injection plan into copied FITS files.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, help="Override plan campaign_root.")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--kernel-radius-native", type=int, default=5)
    parser.add_argument("--limit-injections", type=int)
    parser.add_argument("--strength-sigma", type=float, help="Run only one find_me_snr slice from a multi-strength plan.")
    parser.add_argument(
        "--allow-accumulate-strengths",
        action="store_true",
        help="Allow multiple strengths for the same target/line to accumulate in one FITS campaign.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-approx-wavelengths",
        action="store_true",
        help="Allow old MEF WCS-WAVE spectra. Not valid for science-grade injection/recovery.",
    )
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    baseline_run_dir = Path(str(plan["baseline_run_dir"]))
    campaign_root = args.campaign_root or Path(str(plan["campaign_root"]))
    if args.campaign_root is None and args.strength_sigma is not None:
        campaign_root = campaign_root.with_name(f"{campaign_root.name}_s{float(args.strength_sigma):g}")
    campaign_root = campaign_root.resolve()
    injections = list(plan.get("injections") or [])
    if args.strength_sigma is not None:
        injections = [
            item for item in injections if abs(float(item.get("find_me_snr", float("nan"))) - float(args.strength_sigma)) < 1e-9
        ]
    if args.limit_injections:
        injections = injections[: int(args.limit_injections)]
    if not injections:
        raise SystemExit("Plan has no injections to run")
    if not args.allow_accumulate_strengths:
        seen_groups: set[tuple[str, str, float]] = set()
        duplicates: list[tuple[str, str, float]] = []
        for item in injections:
            key = (
                str(item["target_id"]),
                str(item.get("line_family")),
                float(item["injected_line_nm"]),
            )
            if key in seen_groups:
                duplicates.append(key)
            seen_groups.add(key)
        if duplicates:
            raise SystemExit(
                "Plan contains multiple strengths for the same target/line. "
                "Run one strength slice with --strength-sigma or pass --allow-accumulate-strengths."
            )

    results: list[dict[str, Any]] = []
    merged_overrides: dict[str, str] = {}
    for index, injection in enumerate(injections, start=1):
        print(f"\n=== injection {index}/{len(injections)} {injection['injection_id']} ===", flush=True)
        result = _run_one_injection(
            injection=injection,
            baseline_run_dir=baseline_run_dir,
            campaign_root=campaign_root,
            cache_root=args.cache_root,
            release=args.release,
            kernel_radius_native=args.kernel_radius_native,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            allow_approx_wavelengths=args.allow_approx_wavelengths,
        )
        results.append(result)
        for original_path, injected_path in result["path_overrides"].items():
            merged_overrides[original_path] = injected_path

    manifest = {
        "manifest_version": 1,
        "plan_path": str(args.plan.resolve()),
        "campaign_id": plan.get("campaign_id"),
        "baseline_run_dir": str(baseline_run_dir),
        "campaign_root": str(campaign_root),
        "dry_run": bool(args.dry_run),
        "cache_root": str(args.cache_root),
        "release": args.release,
        "kernel_radius_native": int(args.kernel_radius_native),
        "injection_count": len(results),
        "frame_write_count": int(sum(item["frames_written"] for item in results)),
        "frame_skip_count": int(sum(item["frames_skipped"] for item in results)),
        "path_override_count": len(merged_overrides),
        "path_overrides_path": str(campaign_root / "path_overrides.json"),
        "injections": results,
    }
    campaign_root.mkdir(parents=True, exist_ok=True)
    manifest_path = campaign_root / "injection_manifest.json"
    overrides_path = campaign_root / "path_overrides.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    overrides_path.write_text(json.dumps(merged_overrides, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "path_overrides_path": str(overrides_path),
                "injection_count": len(results),
                "frame_write_count": manifest["frame_write_count"],
                "path_override_count": len(merged_overrides),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
