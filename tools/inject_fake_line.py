from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.units import Unit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spherex_laser_miner.calibration import load_sapm  # noqa: E402
from spherex_laser_miner.photometry.psf import _extract_native_psf  # noqa: E402
from tools.wavelength_guard import assert_science_wavelengths  # noqa: E402


DEFAULT_CACHE_ROOT = Path("/mnt/niroseti/spherex_cache")
DEFAULT_RELEASE = "qr2"


@dataclass(frozen=True)
class InjectionFrame:
    image_id: str
    original_path: str
    injected_path: str | None
    detector: int
    x_pix: float
    y_pix: float
    cwave_um: float
    cband_um: float
    line_response: float
    injected_flux_uJy: float
    psf_sector: int | None
    psf_kernel_sum: float | None
    psf_kernel_shape: list[int] | None
    sapm_path: str | None
    status: str
    reason: str | None = None


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.+-]+", "_", value.strip())
    return safe.strip("_") or "injection"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _required_float(row: pd.Series, column: str) -> float:
    value = _finite_float(row.get(column))
    if not math.isfinite(value):
        raise ValueError(f"missing finite {column}")
    return value


def _response_for_row(cwave_um: float, cband_um: float, line_um: float, line_width_um: float) -> float:
    fwhm_um = math.sqrt(max(cband_um, 0.0) ** 2 + max(line_width_um, 0.0) ** 2)
    if fwhm_um <= 0.0:
        return 0.0
    sigma_um = fwhm_um / 2.354820045
    return math.exp(-0.5 * ((cwave_um - line_um) / sigma_um) ** 2)


def _choose_line_flux_uJy(
    rows: pd.DataFrame,
    responses: np.ndarray,
    line_flux_uJy: float | None,
    find_me_snr: float | None,
    min_response: float,
) -> tuple[float, dict[str, Any]]:
    if line_flux_uJy is not None:
        return float(line_flux_uJy), {"mode": "line_flux_uJy", "line_flux_uJy": float(line_flux_uJy)}
    if find_me_snr is None:
        raise ValueError("Provide either --line-flux-uJy or --find-me-snr")

    unc = pd.to_numeric(rows.get("aperture_flux_unc_uJy"), errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(unc) & (unc > 0.0) & np.isfinite(responses) & (responses >= min_response)
    if not np.any(good):
        raise ValueError("Cannot estimate find-me flux: no finite aperture uncertainties above response cut")
    needed_peak = unc[good] * float(find_me_snr) / responses[good]
    needed_peak = needed_peak[np.isfinite(needed_peak) & (needed_peak > 0.0)]
    if needed_peak.size == 0:
        raise ValueError("Cannot estimate find-me flux: response/uncertainty values were unusable")
    chosen = float(np.nanmedian(needed_peak))
    return chosen, {
        "mode": "find_me_snr",
        "find_me_snr": float(find_me_snr),
        "line_flux_uJy": chosen,
        "supporting_rows": int(needed_peak.size),
        "median_unc_uJy_used": float(np.nanmedian(unc[good])),
        "max_response": float(np.nanmax(responses[good])),
    }


def _image_hdu(hdul: fits.HDUList) -> fits.ImageHDU | fits.PrimaryHDU:
    if "IMAGE" in hdul:
        return hdul["IMAGE"]
    if hdul[0].data is not None:
        return hdul[0]
    raise ValueError("no IMAGE HDU and primary HDU has no data")


def _sapm_scale_uJy_per_image_unit(
    image_header: fits.Header,
    sapm_data: np.ndarray,
    sapm_header: fits.Header,
) -> np.ndarray:
    bunit = Unit(image_header.get("BUNIT", "MJy/sr"))
    sapm_unit = Unit(sapm_header.get("BUNIT", "arcsec2"))
    return ((1.0 * bunit).to(u.uJy / u.arcsec**2) * (sapm_data * sapm_unit)).value


def _inject_one_frame(
    source_path: Path,
    output_path: Path,
    detector: int,
    x_pix: float,
    y_pix: float,
    frame_flux_uJy: float,
    cache_root: Path,
    release: str,
    kernel_radius_native: int,
    injection_id: str,
    dry_run: bool,
    overwrite: bool,
) -> tuple[str | None, int | None, float | None, list[int] | None, str | None]:
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if output_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"{output_path} exists; pass --overwrite")

    with fits.open(source_path, memmap=False) as hdul:
        image_hdu = _image_hdu(hdul)
        image = np.asarray(image_hdu.data, dtype=np.float64)
        if image.ndim != 2:
            raise ValueError(f"IMAGE is not 2D: shape={image.shape}")
        if "PSF" not in hdul:
            raise ValueError("missing PSF extension")
        psf_kernel, sector = _extract_native_psf(
            hdul["PSF"].data,
            image.shape,
            x_pix,
            y_pix,
            kernel_radius_native,
        )
        psf_sum = float(np.sum(psf_kernel))

        sapm_data, sapm_header, sapm_path = load_sapm(cache_root, release, detector)
        scale = _sapm_scale_uJy_per_image_unit(image_hdu.header, sapm_data, sapm_header)
        if scale.shape != image.shape:
            raise ValueError(f"SAPM/image shape mismatch: sapm={scale.shape} image={image.shape}")

        ph, pw = psf_kernel.shape
        x0 = int(round(x_pix))
        y0 = int(round(y_pix))
        lx = pw // 2
        ly = ph // 2
        x1 = max(0, x0 - lx)
        x2 = min(image.shape[1], x0 + (pw - lx))
        y1 = max(0, y0 - ly)
        y2 = min(image.shape[0], y0 + (ph - ly))
        if x1 >= x2 or y1 >= y2:
            raise ValueError("PSF kernel falls fully outside image")

        px1 = lx - (x0 - x1)
        py1 = ly - (y0 - y1)
        psf_cut = psf_kernel[py1 : py1 + (y2 - y1), px1 : px1 + (x2 - x1)]
        scale_cut = scale[y1:y2, x1:x2]
        good_scale = np.isfinite(scale_cut) & (scale_cut > 0.0)
        if not np.any(good_scale):
            raise ValueError("no finite positive SAPM conversion pixels under PSF")

        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.resolve() != output_path.resolve():
                shutil.copy2(source_path, output_path)
            with fits.open(output_path, mode="update", memmap=False) as out_hdul:
                out_image_hdu = _image_hdu(out_hdul)
                out_image = np.asarray(out_image_hdu.data, dtype=np.float64)
                delta_uJy = frame_flux_uJy * psf_cut
                delta_image_units = np.zeros_like(delta_uJy, dtype=np.float64)
                delta_image_units[good_scale] = delta_uJy[good_scale] / scale_cut[good_scale]
                out_image[y1:y2, x1:x2] += delta_image_units
                out_image_hdu.data = out_image.astype(out_image_hdu.data.dtype, copy=False)
                out_image_hdu.header["HIERARCH NIRO INJECTED"] = (True, "Fake line signal injected")
                out_image_hdu.header["HIERARCH NIRO INJID"] = injection_id
                out_image_hdu.header["HIERARCH NIRO INJ_UJY"] = (float(frame_flux_uJy), "Injected frame flux density")
                out_image_hdu.header["HIERARCH NIRO INJ_XPIX"] = float(x_pix)
                out_image_hdu.header["HIERARCH NIRO INJ_YPIX"] = float(y_pix)
                out_hdul.flush()

    return (
        str(output_path) if not dry_run else None,
        int(sector),
        psf_sum,
        [int(psf_kernel.shape[0]), int(psf_kernel.shape[1])],
        str(sapm_path),
    )


def _load_target_rows(run_dir: Path, target_id: str, *, allow_approx_wavelengths: bool = False) -> pd.DataFrame:
    spectra_path = run_dir / "spectra" / "target_spectra.parquet"
    if not spectra_path.exists():
        raise FileNotFoundError(f"Missing spectra parquet: {spectra_path}")
    rows = pd.read_parquet(spectra_path)
    if "target_id" not in rows.columns:
        raise ValueError(f"{spectra_path} has no target_id column")
    assert_science_wavelengths(rows, spectra_path, allow_approx=allow_approx_wavelengths)
    rows = rows[rows["target_id"].astype(str).eq(str(target_id))].copy()
    if rows.empty:
        raise ValueError(f"Target {target_id!r} not found in {spectra_path}")
    required = ["input_file_path", "x_pix", "y_pix", "detector", "cwave_um", "cband_um"]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"Target spectra missing required columns: {', '.join(missing)}")
    rows = rows.dropna(subset=required).copy()
    rows["input_file_path"] = rows["input_file_path"].astype(str)
    rows = rows.drop_duplicates(subset=["input_file_path"]).sort_values("cwave_um")
    if rows.empty:
        raise ValueError(f"Target {target_id!r} has no usable FITS rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inject a fake narrowband point-source line into copied SPHEREx FITS files for one "
            "target from an existing run."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing spectra/target_spectra.parquet.",
    )
    parser.add_argument("--target-id", required=True, help="Target ID in target_spectra.parquet.")
    parser.add_argument("--line-nm", type=float, required=True, help="Line center wavelength in nm.")
    parser.add_argument("--line-width-nm", type=float, default=1.0, help="Intrinsic line FWHM in nm.")
    parser.add_argument("--line-flux-uJy", type=float, help="Peak line flux density at response=1.")
    parser.add_argument("--find-me-snr", type=float, help="Set line flux from median aperture uncertainty / response.")
    parser.add_argument(
        "--min-response",
        type=float,
        default=1e-3,
        help="Skip frames below this spectral response.",
    )
    parser.add_argument("--max-frames", type=int, help="Limit frames for smoke tests.")
    parser.add_argument(
        "--allow-approx-wavelengths",
        action="store_true",
        help="Allow old MEF WCS-WAVE spectra. Not valid for science-grade injection/recovery.",
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--kernel-radius-native", type=int, default=5)
    parser.add_argument("--output-root", type=Path, help="Directory for injected FITS and manifests.")
    parser.add_argument("--injection-id", help="Stable ID used in output names and FITS headers.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.line_nm <= 0.0:
        raise SystemExit("--line-nm must be positive")
    if args.line_width_nm < 0.0:
        raise SystemExit("--line-width-nm must be non-negative")
    if args.line_flux_uJy is not None and args.find_me_snr is not None:
        raise SystemExit("Use only one of --line-flux-uJy or --find-me-snr")
    if args.line_flux_uJy is None and args.find_me_snr is None:
        raise SystemExit("Provide --line-flux-uJy or --find-me-snr")

    run_dir = args.run_dir.resolve()
    line_um = args.line_nm / 1000.0
    line_width_um = args.line_width_nm / 1000.0
    injection_id = args.injection_id or _safe_id(f"{args.target_id}_{args.line_nm:g}nm")
    output_root = args.output_root or (run_dir / "injections" / injection_id)
    output_root = output_root.resolve()

    print(f"Loading target rows for {args.target_id} from {run_dir}", flush=True)
    rows = _load_target_rows(run_dir, args.target_id, allow_approx_wavelengths=args.allow_approx_wavelengths)
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
        rows,
        responses,
        args.line_flux_uJy,
        args.find_me_snr,
        args.min_response,
    )

    rows = rows.assign(line_response=responses)
    rows = rows[rows["line_response"] >= args.min_response].copy()
    if args.max_frames:
        rows = rows.sort_values("line_response", ascending=False).head(args.max_frames).sort_values("cwave_um")
    if rows.empty:
        raise SystemExit("No frames survived --min-response")

    print(
        json.dumps(
            {
                "injection_id": injection_id,
                "target_id": args.target_id,
                "line_nm": args.line_nm,
                "line_width_nm": args.line_width_nm,
                "line_flux_uJy": line_flux_uJy,
                "frames_to_process": int(len(rows)),
                "output_root": str(output_root),
                "dry_run": bool(args.dry_run),
            },
            indent=2,
        ),
        flush=True,
    )

    frames: list[InjectionFrame] = []
    path_overrides: dict[str, str] = {}
    fits_root = output_root / "fits"
    total = int(len(rows))
    for index, (_, row) in enumerate(rows.iterrows(), start=1):
        source_path = Path(str(row["input_file_path"]))
        image_id = str(row.get("image_id") or source_path.stem)
        detector = int(row["detector"])
        x_pix = _required_float(row, "x_pix")
        y_pix = _required_float(row, "y_pix")
        cwave_um = _required_float(row, "cwave_um")
        cband_um = _required_float(row, "cband_um")
        response = _required_float(row, "line_response")
        frame_flux_uJy = float(line_flux_uJy * response)
        output_path = fits_root / f"{_safe_id(image_id)}.fits"

        print(
            f"[{index}/{total}] {image_id} D{detector} "
            f"cwave={cwave_um:.6f}um response={response:.4g} inject={frame_flux_uJy:.3f}uJy",
            flush=True,
        )
        try:
            injected_path, sector, psf_sum, psf_shape, sapm_path = _inject_one_frame(
                source_path=source_path,
                output_path=output_path,
                detector=detector,
                x_pix=x_pix,
                y_pix=y_pix,
                frame_flux_uJy=frame_flux_uJy,
                cache_root=args.cache_root,
                release=args.release,
                kernel_radius_native=args.kernel_radius_native,
                injection_id=injection_id,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            frames.append(
                InjectionFrame(
                    image_id=image_id,
                    original_path=str(source_path),
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
            path_overrides[str(source_path)] = injected_path
        frames.append(
            InjectionFrame(
                image_id=image_id,
                original_path=str(source_path),
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
                status="planned" if args.dry_run else "injected",
            )
        )

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "injection_id": injection_id,
        "run_dir": str(run_dir),
        "target_id": args.target_id,
        "line": {
            "line_nm": float(args.line_nm),
            "line_um": float(line_um),
            "line_width_nm": float(args.line_width_nm),
            "line_width_um": float(line_width_um),
            "response_model": "gaussian_fwhm_sqrt_cband2_plus_line_width2",
        },
        "intensity": intensity,
        "cache_root": str(args.cache_root),
        "release": args.release,
        "kernel_radius_native": int(args.kernel_radius_native),
        "min_response": float(args.min_response),
        "dry_run": bool(args.dry_run),
        "frames_requested": total,
        "frames_written": int(sum(frame.status == "injected" for frame in frames)),
        "frames_planned": int(sum(frame.status == "planned" for frame in frames)),
        "frames_skipped": int(sum(frame.status == "skipped" for frame in frames)),
        "path_overrides_path": str(output_root / "path_overrides.json"),
        "frames": [asdict(frame) for frame in frames],
    }
    manifest_path = output_root / "injection_manifest.json"
    overrides_path = output_root / "path_overrides.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    overrides_path.write_text(json.dumps(path_overrides, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "path_overrides_path": str(overrides_path),
                "frames_written": manifest["frames_written"],
                "frames_planned": manifest["frames_planned"],
                "frames_skipped": manifest["frames_skipped"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
