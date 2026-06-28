#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.io import fits
from PIL import Image, ImageDraw


DEFAULT_CACHE_ROOT = Path("/mnt/niroseti/spherex_cache")
CUTOUT_SIZE = 100
FATAL_FLAG_BITS = (0, 1, 2, 4, 6, 7, 9, 10, 11, 14, 15, 17, 19, 22, 24, 26, 27, 28, 29)
FATAL_FLAG_MASK = sum(1 << bit for bit in FATAL_FLAG_BITS)
SOURCE_FLAG_BIT = 21
SOURCE_FLAG_VALUE = 1 << SOURCE_FLAG_BIT


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone FITS cutout inspector for SPHEREx candidates.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8776)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), _make_handler(args.cache_root.resolve()))
    print(f"FITS inspector serving http://{args.host}:{args.port} cache_root={args.cache_root}", flush=True)
    server.serve_forever()


def _make_handler(cache_root: Path):
    runs_root = cache_root / "runs"
    product_root = cache_root / "fits_inspector"
    product_root.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            params = _parse_query_preserve_plus(parsed.query)
            try:
                if path == "/":
                    self._send_html(_index_html())
                elif path == "/api/inspect":
                    run_dir = _requested_run_dir(params, runs_root)
                    target_id = (params.get("target") or params.get("target_id") or [""])[0]
                    line_nm = _query_float(params, "line_nm")
                    force = _query_bool(params, "force")
                    self._send_json(build_inspection(run_dir, target_id, line_nm, product_root, force=force))
                elif path == "/api/runs":
                    self._send_json(_runs(runs_root))
                elif path == "/api/candidates":
                    run_dir = _requested_run_dir(params, runs_root)
                    self._send_json(_candidate_rows(run_dir, params))
                elif path.startswith("/product/"):
                    self._send_product(product_root, path.removeprefix("/product/"))
                else:
                    self.send_error(404)
            except Exception as exc:
                self._send_json({"error": type(exc).__name__, "message": str(exc)}, status=500)

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}", flush=True)

        def _send_json(self, data: object, status: int = 200) -> None:
            body = json.dumps(_clean_json(data), indent=2, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, text: str) -> None:
            body = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_product(self, root: Path, rel_path: str) -> None:
            product = (root / urllib.parse.unquote(rel_path)).resolve()
            try:
                product.relative_to(root.resolve())
            except ValueError:
                self.send_error(403)
                return
            if not product.exists() or not product.is_file():
                self.send_error(404)
                return
            body = product.read_bytes()
            ctype = {
                ".png": "image/png",
                ".gif": "image/gif",
                ".json": "application/json",
                ".html": "text/html; charset=utf-8",
                ".parquet": "application/octet-stream",
            }.get(product.suffix.lower(), "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def build_inspection(run_dir: Path, target_id: str, line_nm: float | None, product_root: Path, *, force: bool = False) -> dict[str, object]:
    if not target_id:
        raise ValueError("target is required")
    spectra_path = run_dir / "spectra" / "target_spectra.parquet"
    if not spectra_path.exists():
        raise FileNotFoundError(f"No spectra parquet in {run_dir}")

    rows = _read_target_rows(spectra_path, target_id)
    if rows.empty:
        raise ValueError(f"No spectrum rows for target={target_id} in run={run_dir.name}")
    rows = rows.reset_index(drop=True)

    candidate = _best_candidate(run_dir, target_id, line_nm)
    candidate_line_nm = line_nm or _maybe_float(candidate.get("peak_line_nm") or candidate.get("candidate_line_nm")) if candidate else line_nm
    line_token = f"{candidate_line_nm:.1f}nm" if candidate_line_nm is not None and math.isfinite(candidate_line_nm) else "all"
    out_dir = product_root / _safe_name(run_dir.name) / _safe_name(target_id) / _safe_name(line_token)
    summary_path = out_dir / "summary.json"
    frames_path = out_dir / "frames.parquet"
    if summary_path.exists() and frames_path.exists() and not force:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["cached"] = True
        return summary

    started = time.time()
    cutout_dir = out_dir / "cutouts"
    cutout_dir.mkdir(parents=True, exist_ok=True)

    support_image_ids = _support_image_ids(candidate)
    stretch = _fixed_target_stretch(rows)
    frame_records: list[dict[str, object]] = []
    gif_images: list[Image.Image] = []

    for idx, row in rows.iterrows():
        record = _frame_record(row, idx, support_image_ids, candidate_line_nm)
        try:
            product = _build_cutout_png(row, cutout_dir, idx, target_id, stretch, record["is_support"])
            record.update(product)
            if record["png_path"]:
                gif_images.append(Image.open(record["png_path"]).convert("P"))
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["png_url"] = None
            record["png_path"] = None
        frame_records.append(record)
    _mark_driver_frames(frame_records)

    gif_path = out_dir / "blink.gif"
    if gif_images:
        gif_images[0].save(gif_path, save_all=True, append_images=gif_images[1:], duration=180, loop=0)

    frames_df = pd.DataFrame(frame_records)
    frames_df.to_parquet(frames_path, index=False)
    blink_path = _write_blink_html(out_dir, run_dir.name, target_id, candidate_line_nm, frame_records)
    peak_frame = next((row for row in frame_records if row.get("is_peak_flux_frame")), None)
    response_driver = next((row for row in frame_records if row.get("is_response_driver_frame")), None)

    summary = {
        "run": run_dir.name,
        "target_id": target_id,
        "candidate_line_nm": candidate_line_nm,
        "candidate": candidate,
        "frame_count": len(frame_records),
        "support_frame_count": int(sum(bool(row.get("is_support")) for row in frame_records)),
        "error_count": int(sum(bool(row.get("error")) for row in frame_records)),
        "cutout_size": CUTOUT_SIZE,
        "stretch": stretch,
        "spectrum": _spectrum_summary(rows),
        "wide_background": _wide_background_summary(frame_records),
        "peak_flux_frame": peak_frame,
        "response_driver_frame": response_driver,
        "product_root": str(out_dir),
        "frames_parquet": str(frames_path),
        "frames_url": _product_url(product_root, frames_path),
        "summary_url": _product_url(product_root, summary_path),
        "gif_url": _product_url(product_root, gif_path) if gif_path.exists() else None,
        "blink_url": _product_url(product_root, blink_path),
        "peak_flux_blink_url": (_product_url(product_root, blink_path) + f"?frame={int(peak_frame['index'])}") if peak_frame else None,
        "response_driver_blink_url": (_product_url(product_root, blink_path) + f"?frame={int(response_driver['index'])}") if response_driver else None,
        "frames": frame_records,
        "build_sec": time.time() - started,
        "cached": False,
    }
    summary_path.write_text(json.dumps(_clean_json(summary), indent=2, allow_nan=False), encoding="utf-8")
    return summary


def _read_target_rows(path: Path, target_id: str) -> pd.DataFrame:
    columns = _existing_columns(
        path,
        [
            "target_id",
            "target_type",
            "object_name",
            "source_id",
            "ra_reference_deg",
            "dec_reference_deg",
            "reference_epoch_yr",
            "cwave_um",
            "cband_um",
            "detector",
            "observation_id",
            "obs_mid_time",
            "image_id",
            "x_pix",
            "y_pix",
            "edge_distance_pix",
            "aperture_flux_uJy",
            "aperture_flux_unc_uJy",
            "psf_flux_uJy",
            "psf_flux_unc_uJy",
            "psf_fit_status",
            "psf_chi2",
            "centroid_dx_pix",
            "centroid_dy_pix",
            "fatal_flag_present",
            "flags_summary",
            "input_file_path",
            "original_input_file_path",
        ],
    )
    try:
        rows = pd.read_parquet(path, columns=columns, filters=[("target_id", "=", target_id)])
    except Exception:
        df = pd.read_parquet(path, columns=columns)
        rows = df[df["target_id"].astype(str).eq(target_id)]
    if "cwave_um" in rows:
        rows = rows.sort_values("cwave_um", na_position="last", kind="mergesort")
    return rows


def _best_candidate(run_dir: Path, target_id: str, line_nm: float | None) -> dict[str, object] | None:
    candidates_path = run_dir / "narrowband_detector_raw" / "narrowband_candidates.parquet"
    if not candidates_path.exists():
        return None
    try:
        df = pd.read_parquet(candidates_path, filters=[("target_id", "=", target_id)])
    except Exception:
        df = pd.read_parquet(candidates_path)
        df = df[df["target_id"].astype(str).eq(target_id)]
    if df.empty:
        return None
    if line_nm is not None:
        wave_col = "peak_line_nm" if "peak_line_nm" in df else "candidate_line_nm"
        if wave_col in df:
            waves = pd.to_numeric(df[wave_col], errors="coerce")
            close = df[(waves - float(line_nm)).abs().le(5.0)]
            if not close.empty:
                df = close
    sort_col = "rank_score" if "rank_score" in df else "joint_rho" if "joint_rho" in df else None
    if sort_col:
        df = df.sort_values(sort_col, ascending=False, na_position="last")
    return df.iloc[0].to_dict()


def _support_image_ids(candidate: dict[str, object] | None) -> set[str]:
    if not candidate:
        return set()
    raw = str(candidate.get("frame_ids") or candidate.get("best_frame_ids") or "")
    return {part.strip() for part in raw.split(",") if part.strip()}


def _fixed_target_stretch(rows: pd.DataFrame) -> dict[str, float | str]:
    samples: list[np.ndarray] = []
    for _, row in rows.head(80).iterrows():
        path = Path(str(row.get("input_file_path") or ""))
        if not path.exists():
            continue
        try:
            with fits.open(path, memmap=True) as hdul:
                image = np.asarray(hdul["IMAGE"].data, dtype=np.float32)
                cut = _cutout_array(image, _maybe_float(row.get("x_pix")), _maybe_float(row.get("y_pix")), CUTOUT_SIZE)
        except Exception:
            continue
        finite = cut[np.isfinite(cut)]
        if finite.size:
            samples.append(finite[:: max(1, finite.size // 2000)])
    if not samples:
        return {"mode": "fallback", "vmin": 0.0, "vmax": 1.0, "median": 0.0, "mad": 1.0}
    pooled = np.concatenate(samples)
    median = float(np.nanmedian(pooled))
    mad = float(np.nanmedian(np.abs(pooled - median)))
    sigma = 1.4826 * mad if mad > 0 else float(np.nanstd(pooled) or 1.0)
    vmin = median - 3.0 * sigma
    vmax = median + 12.0 * sigma
    if not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanpercentile(pooled, 1)), float(np.nanpercentile(pooled, 99))
    return {"mode": "pooled_target_cutout_median_mad", "vmin": float(vmin), "vmax": float(vmax), "median": median, "mad": mad}


def _build_cutout_png(row: pd.Series, cutout_dir: Path, idx: int, target_id: str, stretch: dict[str, object], is_support: bool) -> dict[str, object]:
    path = Path(str(row.get("input_file_path") or ""))
    if not path.exists():
        raise FileNotFoundError(str(path))
    x = _maybe_float(row.get("x_pix"))
    y = _maybe_float(row.get("y_pix"))
    with fits.open(path, memmap=True) as hdul:
        image = np.asarray(hdul["IMAGE"].data, dtype=np.float32)
        flags = np.asarray(hdul["FLAGS"].data) if "FLAGS" in hdul else None
        cut = _cutout_array(image, x, y, CUTOUT_SIZE)
        flag_cut = _cutout_array(flags, x, y, CUTOUT_SIZE) if flags is not None else None
    arr = _scale_to_uint8(cut, float(stretch["vmin"]), float(stretch["vmax"]))
    rgb = Image.fromarray(arr, mode="L").convert("RGB")
    draw = ImageDraw.Draw(rgb)
    cx = cy = CUTOUT_SIZE // 2
    color = (255, 191, 64) if is_support else (54, 231, 255)
    draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), outline=color, width=1)
    draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), outline=color, width=1)
    draw.ellipse((cx - 16, cy - 16, cx + 16, cy + 16), outline=(120, 120, 120), width=1)
    nonzero_flag_pixels = 0
    source_flag_pixels = 0
    fatal_flag_pixels = 0
    if flag_cut is not None:
        flag_arr = np.nan_to_num(np.asarray(flag_cut), nan=0.0, posinf=0.0, neginf=0.0).astype(np.uint32, copy=False)
        nonzero_flag_pixels = int(np.count_nonzero(flag_arr))
        source_flag_pixels = int(np.count_nonzero((flag_arr & SOURCE_FLAG_VALUE) != 0))
        bad = (flag_arr & FATAL_FLAG_MASK) != 0
        fatal_flag_pixels = int(np.count_nonzero(bad))
        ys, xs = np.where(bad)
        for px, py in zip(xs[:: max(1, len(xs) // 120)], ys[:: max(1, len(ys) // 120)]):
            draw.point((int(px), int(py)), fill=(255, 70, 70))
    label = f"{idx:03d} {float(row.get('cwave_um') or 0):.4f}um D{row.get('detector') or ''}"
    draw.rectangle((1, 1, 99, 12), fill=(0, 0, 0))
    draw.text((3, 2), label, fill=(255, 255, 255))
    if is_support:
        draw.rectangle((0, 0, CUTOUT_SIZE - 1, CUTOUT_SIZE - 1), outline=(255, 191, 64), width=2)
    out = cutout_dir / f"{idx:04d}_{_safe_name(str(row.get('image_id') or idx))}.png"
    rgb.save(out)
    finite = cut[np.isfinite(cut)]
    return {
        "png_path": str(out),
        "png_url": _product_url(cutout_dir.parents[3], out) if len(cutout_dir.parents) >= 4 else str(out),
        "cutout_median": float(np.nanmedian(finite)) if finite.size else None,
        "cutout_mad": float(np.nanmedian(np.abs(finite - np.nanmedian(finite)))) if finite.size else None,
        "cutout_p95": float(np.nanpercentile(finite, 95)) if finite.size else None,
        "target_pixel_value": _pixel_value(cut, CUTOUT_SIZE // 2, CUTOUT_SIZE // 2),
        "nonzero_flag_pixel_count_cutout": nonzero_flag_pixels,
        "source_flag_pixel_count_cutout": source_flag_pixels,
        "fatal_flag_pixel_count_cutout": fatal_flag_pixels,
    }


def _frame_record(row: pd.Series, idx: int, support_image_ids: set[str], line_nm: float | None) -> dict[str, object]:
    image_id = str(row.get("image_id") or "")
    response = _response_to_line(row, line_nm)
    aperture_flux = _maybe_float(row.get("aperture_flux_uJy"))
    psf_flux = _maybe_float(row.get("psf_flux_uJy"))
    return {
        "index": int(idx),
        "is_support": image_id in support_image_ids,
        "is_response_support": bool(response is not None and response >= 0.01),
        "response_to_line": response,
        "weighted_aperture_flux": response * aperture_flux if response is not None and aperture_flux is not None else None,
        "weighted_psf_flux": response * psf_flux if response is not None and psf_flux is not None else None,
        "image_id": image_id,
        "fits_path": str(row.get("input_file_path") or ""),
        "original_fits_path": str(row.get("original_input_file_path") or ""),
        "detector": _maybe_float(row.get("detector")),
        "observation_id": row.get("observation_id"),
        "obs_mid_time": row.get("obs_mid_time"),
        "cwave_um": _maybe_float(row.get("cwave_um")),
        "cband_um": _maybe_float(row.get("cband_um")),
        "x_pix": _maybe_float(row.get("x_pix")),
        "y_pix": _maybe_float(row.get("y_pix")),
        "edge_distance_pix": _maybe_float(row.get("edge_distance_pix")),
        "aperture_flux_uJy": aperture_flux,
        "aperture_flux_unc_uJy": _maybe_float(row.get("aperture_flux_unc_uJy")),
        "psf_flux_uJy": psf_flux,
        "psf_flux_unc_uJy": _maybe_float(row.get("psf_flux_unc_uJy")),
        "psf_fit_status": row.get("psf_fit_status"),
        "psf_chi2": _maybe_float(row.get("psf_chi2")),
        "centroid_dx_pix": _maybe_float(row.get("centroid_dx_pix")),
        "centroid_dy_pix": _maybe_float(row.get("centroid_dy_pix")),
        "fatal_flag_present": bool(row.get("fatal_flag_present")) if pd.notna(row.get("fatal_flag_present")) else False,
        "flags_summary": _maybe_float(row.get("flags_summary")),
    }


def _response_to_line(row: pd.Series, line_nm: float | None) -> float | None:
    if line_nm is None:
        return None
    cwave = _maybe_float(row.get("cwave_um"))
    cband = _maybe_float(row.get("cband_um"))
    if cwave is None:
        return None
    line_um = float(line_nm) / 1000.0
    band = cband if cband is not None and cband > 0 else 0.04
    sigma = max(band / 2.355, 1e-6)
    return float(math.exp(-0.5 * ((cwave - line_um) / sigma) ** 2))


def _mark_driver_frames(records: list[dict[str, object]]) -> None:
    for row in records:
        row["is_peak_flux_frame"] = False
        row["is_response_driver_frame"] = False
    ap_rows = [row for row in records if row.get("aperture_flux_uJy") is not None]
    if ap_rows:
        max(ap_rows, key=lambda row: float(row.get("aperture_flux_uJy") or -np.inf))["is_peak_flux_frame"] = True
    weighted_rows = [row for row in records if row.get("weighted_aperture_flux") is not None]
    if weighted_rows:
        max(weighted_rows, key=lambda row: float(row.get("weighted_aperture_flux") or -np.inf))["is_response_driver_frame"] = True


def _write_blink_html(out_dir: Path, run: str, target_id: str, line_nm: float | None, frames: list[dict[str, object]]) -> Path:
    path = out_dir / "blink.html"
    rows = [row for row in frames if row.get("png_url")]
    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(target_id)} blink</title>
<style>
body{{margin:0;background:#050914;color:#e7f6ff;font:13px system-ui,sans-serif}}
main{{display:grid;grid-template-columns:360px 1fr;gap:12px;padding:12px}}
section{{border:1px solid #1f3a5f;background:#0b1424;border-radius:6px;padding:10px}}
img{{image-rendering:pixelated;width:640px;height:640px;border:1px solid #36e7ff;background:#000}}
table{{width:100%;border-collapse:collapse;font-size:12px}}td,th{{border-bottom:1px solid #1f3a5f;padding:4px;text-align:left}}
.support{{color:#ffbf40;font-weight:700}}button{{background:#07101e;color:#e7f6ff;border:1px solid #1f3a5f;border-radius:4px;padding:8px}}
</style></head><body>
<main><section><h2>{html.escape(target_id)}</h2><div>{html.escape(run)}<br>{line_nm or ""} nm</div>
<button onclick="toggle()">Play/Pause</button><button onclick="step(-1)">Prev</button><button onclick="step(1)">Next</button><button onclick="jumpPeak()">Peak</button>
<div class="small">Keys: space play/pause, left/right step, home/end, p peak flux, d response driver, s next response-support frame.</div>
<div id="meta"></div><table><tbody id="rows"></tbody></table></section>
<section><img id="frame"></section></main>
<script>
const frames={json.dumps(_clean_json(rows))};
let i=initialFrame(), playing=true;
function initialFrame(){{
  const p=new URLSearchParams(location.search);
  const n=Number(p.get('frame') || location.hash.replace('#frame=',''));
  if(Number.isFinite(n)){{
    const idx=frames.findIndex(f=>Number(f.index)===n);
    if(idx>=0)return idx;
  }}
  const peak=frames.findIndex(f=>f.is_peak_flux_frame);
  return peak>=0 ? peak : 0;
}}
function render(){{
  if(!frames.length)return;
  const f=frames[i%frames.length];
  document.getElementById('frame').src=f.png_url;
  history.replaceState(null,'', location.pathname + '?frame=' + encodeURIComponent(f.index));
  const label=f.is_peak_flux_frame?'PEAK FLUX':(f.is_response_driver_frame?'RESPONSE DRIVER':(f.is_response_support?'RESPONSE SUPPORT':(f.is_support?'CANDIDATE FRAME':'context')));
  document.getElementById('meta').innerHTML=`<h3>${{i+1}}/${{frames.length}}</h3><div class="${{f.is_support||f.is_response_support?'support':''}}">${{label}}</div><div>${{f.image_id}}</div><div>${{Number(f.cwave_um).toFixed(4)}} um D${{f.detector}}</div><div>response=${{fmt(f.response_to_line)}} weighted_ap=${{fmt(f.weighted_aperture_flux)}} ap=${{fmt(f.aperture_flux_uJy)}} psf=${{fmt(f.psf_flux_uJy)}}</div>`;
  document.getElementById('rows').innerHTML=frames.map((r,idx)=>`<tr onclick="i=${{idx}};render()" class="${{r.is_support||r.is_response_support?'support':''}}"><td>${{idx+1}}</td><td>${{Number(r.cwave_um).toFixed(4)}}</td><td>${{r.detector}}</td><td>${{r.is_peak_flux_frame?'peak':(r.is_response_driver_frame?'driver':(r.is_response_support?'resp':(r.is_support?'cand':'')))}}</td><td>${{fmt(r.response_to_line)}}</td></tr>`).join('');
}}
function step(d){{i=(i+d+frames.length)%frames.length;render();}}
function toggle(){{playing=!playing;}}
function jumpPeak(){{ const j=frames.findIndex(f=>f.is_peak_flux_frame); if(j>=0){{i=j;render();}} }}
function jumpDriver(){{ const j=frames.findIndex(f=>f.is_response_driver_frame); if(j>=0){{i=j;render();}} }}
function nextSupport(){{ for(let k=1;k<=frames.length;k++){{ const j=(i+k)%frames.length; if(frames[j].is_response_support){{i=j;render();return;}} }} }}
function fmt(v){{ const n=Number(v); if(!Number.isFinite(n))return ''; if(Math.abs(n)>=1000)return n.toExponential(3); if(Math.abs(n)>=100)return n.toFixed(1); if(Math.abs(n)>=10)return n.toFixed(2); return n.toFixed(4); }}
window.addEventListener('keydown', ev=>{{ if(ev.target && ['INPUT','SELECT','TEXTAREA'].includes(ev.target.tagName))return; if(ev.key===' '){{ev.preventDefault();toggle();}} else if(ev.key==='ArrowRight')step(1); else if(ev.key==='ArrowLeft')step(-1); else if(ev.key==='Home'){{i=0;render();}} else if(ev.key==='End'){{i=frames.length-1;render();}} else if(ev.key.toLowerCase()==='p')jumpPeak(); else if(ev.key.toLowerCase()==='d')jumpDriver(); else if(ev.key.toLowerCase()==='s')nextSupport(); }});
setInterval(()=>{{if(playing)step(1)}},220);
render();
</script></body></html>"""
    path.write_text(body, encoding="utf-8")
    return path


def _cutout_array(image: np.ndarray | None, x: float | None, y: float | None, size: int) -> np.ndarray:
    if image is None or x is None or y is None:
        return np.full((size, size), np.nan, dtype=np.float32)
    half = size // 2
    cx, cy = int(round(x)), int(round(y))
    out = np.full((size, size), np.nan, dtype=np.float32)
    y0, y1 = max(0, cy - half), min(image.shape[0], cy + half)
    x0, x1 = max(0, cx - half), min(image.shape[1], cx + half)
    oy0, ox0 = y0 - (cy - half), x0 - (cx - half)
    if y1 > y0 and x1 > x0:
        out[oy0 : oy0 + (y1 - y0), ox0 : ox0 + (x1 - x0)] = image[y0:y1, x0:x1]
    return out


def _scale_to_uint8(image: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    scaled = (arr - vmin) / max(vmax - vmin, 1e-9)
    scaled = np.clip(scaled, 0, 1)
    scaled[~np.isfinite(scaled)] = 0
    return (scaled * 255).astype(np.uint8)


def _pixel_value(cutout: np.ndarray, x: int, y: int) -> float | None:
    try:
        value = float(cutout[y, x])
    except Exception:
        return None
    return value if np.isfinite(value) else None


def _spectrum_summary(rows: pd.DataFrame) -> dict[str, object]:
    cwave = pd.to_numeric(rows.get("cwave_um"), errors="coerce")
    ap = pd.to_numeric(rows.get("aperture_flux_uJy"), errors="coerce")
    psf = pd.to_numeric(rows.get("psf_flux_uJy"), errors="coerce")
    return {
        "measurement_count": int(len(rows)),
        "wavelength_min_um": float(cwave.min()) if cwave.notna().any() else None,
        "wavelength_max_um": float(cwave.max()) if cwave.notna().any() else None,
        "aperture_flux_median_uJy": float(ap.median()) if ap.notna().any() else None,
        "psf_flux_median_uJy": float(psf.median()) if psf.notna().any() else None,
        "fatal_flag_count": int(pd.Series(rows.get("fatal_flag_present", [])).fillna(False).astype(bool).sum()),
    }


def _wide_background_summary(records: list[dict[str, object]]) -> dict[str, object]:
    med = pd.to_numeric(pd.Series([row.get("cutout_median") for row in records]), errors="coerce")
    p95 = pd.to_numeric(pd.Series([row.get("cutout_p95") for row in records]), errors="coerce")
    return {
        "cutout_median_median": float(med.median()) if med.notna().any() else None,
        "cutout_median_p10": float(med.quantile(0.1)) if med.notna().any() else None,
        "cutout_median_p90": float(med.quantile(0.9)) if med.notna().any() else None,
        "cutout_p95_median": float(p95.median()) if p95.notna().any() else None,
    }


def _candidate_rows(run_dir: Path, params: dict[str, list[str]]) -> dict[str, object]:
    path = run_dir / "narrowband_detector_raw" / "narrowband_candidates.parquet"
    if not path.exists():
        return {"rows": [], "total": 0}
    q = str((params.get("q") or [""])[0]).lower()
    limit = min(500, max(1, _query_int(params, "limit", 100)))
    df = pd.read_parquet(path)
    if q:
        mask = df["target_id"].astype(str).str.lower().str.contains(q, regex=False)
        if "joint_candidate_id" in df:
            mask |= df["joint_candidate_id"].astype(str).str.lower().str.contains(q, regex=False)
        df = df[mask]
    sort_col = "rank_score" if "rank_score" in df else "joint_rho" if "joint_rho" in df else None
    if sort_col:
        df = df.sort_values(sort_col, ascending=False, na_position="last")
    return {"rows": df.head(limit).to_dict(orient="records"), "total": int(len(df))}


def _runs(runs_root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(runs_root.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)[:500]:
        if not path.is_dir():
            continue
        rows.append(
            {
                "name": path.name,
                "has_spectra": (path / "spectra" / "target_spectra.parquet").exists(),
                "has_candidates": (path / "narrowband_detector_raw" / "narrowband_candidates.parquet").exists(),
            }
        )
    return rows


def _requested_run_dir(params: dict[str, list[str]], runs_root: Path) -> Path:
    name = Path(str((params.get("run") or [""])[0])).name
    if not name:
        raise ValueError("run is required")
    run_dir = (runs_root / name).resolve()
    try:
        run_dir.relative_to(runs_root.resolve())
    except ValueError:
        raise ValueError("invalid run")
    if not run_dir.exists():
        raise FileNotFoundError(str(run_dir))
    return run_dir


def _product_url(root: Path, path: Path) -> str:
    return "/product/" + urllib.parse.quote(str(path.resolve().relative_to(root.resolve())))


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value))[:180]


def _existing_columns(path: Path, columns: list[str]) -> list[str]:
    try:
        names = set(pq.ParquetFile(path).schema.names)
    except Exception:
        return columns
    return [col for col in columns if col in names]


def _query_int(params: dict[str, list[str]], name: str, default: int) -> int:
    try:
        return int((params.get(name) or [default])[0])
    except Exception:
        return default


def _query_float(params: dict[str, list[str]], name: str) -> float | None:
    return _maybe_float((params.get(name) or [""])[0])


def _query_bool(params: dict[str, list[str]], name: str) -> bool:
    return str((params.get(name) or [""])[0]).lower() in {"1", "true", "yes", "on"}


def _maybe_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _parse_query_preserve_plus(query: str) -> dict[str, list[str]]:
    params: dict[str, list[str]] = {}
    if not query:
        return params
    for part in query.split("&"):
        if not part:
            continue
        key, sep, value = part.partition("=")
        if not sep:
            value = ""
        params.setdefault(urllib.parse.unquote(key), []).append(urllib.parse.unquote(value))
    return params


def _clean_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [_clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value) if not isinstance(value, (str, bytes, dict, list, tuple)) else False:
        return None
    return value


def _index_html() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>LuxQuarry FITS Inspector</title>
  <style>
    :root { --bg:#050914; --panel:#0b1424; --line:#1f3a5f; --text:#e7f6ff; --muted:#86a4bf; --cyan:#36e7ff; --pink:#ff4fd8; --amber:#ffb84a; --green:#25f38c; --red:#ff5c7c; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--text); font:13px system-ui,sans-serif; background:linear-gradient(90deg,rgba(54,231,255,.055) 1px,transparent 1px),linear-gradient(rgba(255,79,216,.04) 1px,transparent 1px),var(--bg); background-size:42px 42px; }
    header { display:flex; justify-content:space-between; align-items:center; padding:12px 16px; border-bottom:1px solid var(--line); background:rgba(5,9,20,.94); }
    h1 { margin:0; font-size:18px; color:var(--cyan); }
    main { display:grid; grid-template-columns:440px minmax(800px,1fr); gap:12px; padding:12px; }
    section { background:rgba(11,20,36,.94); border:1px solid var(--line); border-radius:6px; padding:10px; min-width:0; }
    label { display:block; color:var(--muted); font-size:12px; margin:8px 0 4px; }
    input,select,button { width:100%; padding:8px; color:var(--text); background:#07101e; border:1px solid var(--line); border-radius:4px; }
    button { cursor:pointer; }
    button:hover { border-color:var(--cyan); }
    a { color:#93e8ff; }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .cards { display:grid; grid-template-columns:repeat(5,minmax(120px,1fr)); gap:8px; margin-bottom:10px; }
    .card { border:1px solid var(--line); background:#091527; border-radius:4px; padding:8px; }
    .k { color:var(--muted); font-size:10px; text-transform:uppercase; }
    .v { font-size:15px; margin-top:3px; overflow-wrap:anywhere; }
    .support { color:var(--amber); font-weight:700; }
    .driver { color:var(--pink); font-weight:800; }
    .small { color:var(--muted); font-size:12px; }
    .mono { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
    .scroll { overflow:auto; max-height:calc(100vh - 250px); border:1px solid var(--line); border-radius:4px; }
    table { width:100%; border-collapse:collapse; font-size:12px; }
    th,td { border-bottom:1px solid var(--line); padding:5px 6px; text-align:left; white-space:nowrap; }
    th { position:sticky; top:0; background:#0b182b; color:#8eeaff; z-index:1; }
    tr:hover { background:rgba(54,231,255,.08); }
    #blink { width:520px; height:520px; image-rendering:pixelated; border:1px solid var(--cyan); background:#000; }
    #thumbs { display:grid; grid-template-columns:repeat(auto-fill, minmax(100px,1fr)); gap:6px; margin-top:10px; max-height:360px; overflow:auto; }
    #thumbs img { width:100px; height:100px; image-rendering:pixelated; border:1px solid #24415f; }
    #thumbs img.support { border-color:var(--amber); border-width:2px; }
  </style>
</head>
<body>
<header><h1>LuxQuarry FITS Inspector</h1><div class="small">standalone heavy cutout builder</div></header>
<main>
  <section>
    <label>Run</label><input id="run" placeholder="run name">
    <label>Target</label><input id="target" placeholder="twomass_psc_...">
    <div class="row">
      <div><label>Line nm</label><input id="line" placeholder="4502"></div>
      <div><label>Force rebuild</label><select id="force"><option value="0">no</option><option value="1">yes</option></select></div>
    </div>
    <button onclick="inspect()" style="margin-top:10px">Build / Load Inspection</button>
    <div id="status" class="mono small" style="margin-top:10px"></div>
    <h3>Candidates</h3>
    <label>Search current run</label><input id="candQ" oninput="scheduleCandidates()" placeholder="target or candidate">
    <div id="candidates" class="scroll"></div>
  </section>
  <section>
    <div id="cards" class="cards"></div>
    <div class="row">
      <section style="padding:8px">
        <h3>Blinker</h3>
        <img id="blink">
        <div class="row"><button onclick="step(-1)">Prev</button><button onclick="toggle()">Play/Pause</button></div>
        <div class="row" style="margin-top:8px"><button onclick="jumpPeak()">Peak Flux</button><button onclick="jumpDriver()">Line Driver</button></div>
        <div class="small">Keys: space play/pause, left/right step, home/end, p peak flux, d line driver, s next response-support frame.</div>
        <div id="frameMeta" class="small mono"></div>
      </section>
      <section style="padding:8px">
        <h3>Products</h3>
        <div id="products" class="small"></div>
        <div id="thumbs"></div>
      </section>
    </div>
    <h3>All FITS Rows</h3>
    <div id="frames" class="scroll"></div>
  </section>
</main>
<script>
const params = new URLSearchParams(location.search);
let result = null, idx = 0, playing = true, timer = null, candidateTimer = null;
function rawParam(name){ const q=location.search.replace(/^\\?/,''); for(const part of q.split('&')){ if(!part)continue; const pieces=part.split('='); const key=decodeURIComponent(pieces.shift()||''); if(key===name)return decodeURIComponent(pieces.join('=')||''); } return ''; }
function set(id,v){ document.getElementById(id).value = v || ''; }
function val(id){ return document.getElementById(id).value.trim(); }
function esc(v){ return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function fmt(v,d=3){ const n=Number(v); if(!Number.isFinite(n))return v??''; if(Math.abs(n)>=1000)return n.toExponential(3); if(Math.abs(n)>=100)return n.toFixed(1); if(Math.abs(n)>=10)return n.toFixed(2); return n.toFixed(d); }
async function getJSON(url){ const r=await fetch(url,{cache:'no-store'}); if(!r.ok) throw new Error(await r.text()); return await r.json(); }
function qs(){ const p=new URLSearchParams(); p.set('run',val('run')); p.set('target',val('target')); if(val('line'))p.set('line_nm',val('line')); if(val('force')==='1')p.set('force','1'); return '?' + p.toString(); }
async function inspect(){
  document.getElementById('status').textContent = 'building/loading inspection...';
  result = await getJSON('/api/inspect' + qs());
  idx = 0;
  render();
  history.replaceState(null,'','/' + qs());
}
function render(){
  const s=result||{};
  document.getElementById('status').textContent = JSON.stringify({cached:s.cached, build_sec:s.build_sec, product_root:s.product_root}, null, 2);
  document.getElementById('cards').innerHTML = [
    ['Frames',s.frame_count],['Support',s.support_frame_count],['Errors',s.error_count],['Line nm',s.candidate_line_nm],['Quality',s.candidate?.quality_tier||'']
  ].map(([k,v])=>`<div class="card"><div class="k">${esc(k)}</div><div class="v">${esc(fmt(v))}</div></div>`).join('');
  document.getElementById('products').innerHTML = `<p><a href="${esc(s.gif_url)}" target="_blank">animated gif</a></p><p><a href="${esc(s.blink_url)}" target="_blank">standalone blink html</a></p><p><a href="${esc(s.peak_flux_blink_url || s.blink_url)}" target="_blank">open at peak flux frame</a></p><p><a href="${esc(s.response_driver_blink_url || s.blink_url)}" target="_blank">open at line-driver frame</a></p><p><a href="${esc(s.summary_url)}" target="_blank">summary json</a></p><pre>${esc(JSON.stringify({stretch:s.stretch,spectrum:s.spectrum,wide_background:s.wide_background,peak_flux_frame:s.peak_flux_frame,response_driver_frame:s.response_driver_frame}, null, 2))}</pre>`;
  const frames=s.frames||[];
  document.getElementById('thumbs').innerHTML = frames.filter(f=>f.png_url).map((f,i)=>`<img onclick="idx=${i};drawFrame()" class="${f.is_peak_flux_frame||f.is_response_driver_frame?'driver':(f.is_support||f.is_response_support?'support':'')}" src="${esc(f.png_url)}" title="${esc(f.image_id)}">`).join('');
  document.getElementById('frames').innerHTML = table(frames, ['index','is_peak_flux_frame','is_response_driver_frame','is_response_support','response_to_line','weighted_aperture_flux','cwave_um','detector','obs_mid_time','aperture_flux_uJy','psf_flux_uJy','fatal_flag_present','flags_summary','fatal_flag_pixel_count_cutout','source_flag_pixel_count_cutout','nonzero_flag_pixel_count_cutout','image_id','fits_path','error']);
  jumpDriver();
}
function table(rows, cols){ if(!rows.length)return '<div class="small">No rows</div>'; return '<table><thead><tr>'+cols.map(c=>`<th>${esc(c)}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>`<tr class="${r.is_peak_flux_frame||r.is_response_driver_frame?'driver':(r.is_support||r.is_response_support?'support':'')}" onclick="idx=${Number(r.index)||0};drawFrame()">`+cols.map(c=>`<td title="${esc(r[c])}">${esc(fmt(r[c]))}</td>`).join('')+'</tr>').join('')+'</tbody></table>'; }
function drawFrame(){ const frames=result?.frames||[]; if(!frames.length)return; idx=(idx+frames.length)%frames.length; const f=frames[idx]; document.getElementById('blink').src=f.png_url||''; history.replaceState(null,'','/' + qs() + '&frame=' + encodeURIComponent(f.index)); document.getElementById('frameMeta').textContent=JSON.stringify(f,null,2); }
function step(d){ idx += d; drawFrame(); }
function toggle(){ playing=!playing; }
function jumpPeak(){ const frames=result?.frames||[]; const j=frames.findIndex(f=>f.is_peak_flux_frame); if(j>=0){idx=j; drawFrame();} }
function jumpDriver(){ const frames=result?.frames||[]; const j=frames.findIndex(f=>f.is_response_driver_frame); if(j>=0){idx=j; drawFrame();} else drawFrame(); }
function nextSupport(){ const frames=result?.frames||[]; for(let k=1;k<=frames.length;k++){ const j=(idx+k)%frames.length; if(frames[j].is_response_support){idx=j;drawFrame();return;} } }
setInterval(()=>{ if(playing && result) step(1); }, 240);
window.addEventListener('keydown', ev=>{ if(ev.target && ['INPUT','SELECT','TEXTAREA'].includes(ev.target.tagName))return; if(ev.key===' '){ev.preventDefault();toggle();} else if(ev.key==='ArrowRight')step(1); else if(ev.key==='ArrowLeft')step(-1); else if(ev.key==='Home'){idx=0;drawFrame();} else if(ev.key==='End'){idx=(result?.frames||[]).length-1;drawFrame();} else if(ev.key.toLowerCase()==='p')jumpPeak(); else if(ev.key.toLowerCase()==='d')jumpDriver(); else if(ev.key.toLowerCase()==='s')nextSupport(); });
function scheduleCandidates(){ clearTimeout(candidateTimer); candidateTimer=setTimeout(loadCandidates,250); }
async function loadCandidates(){ if(!val('run'))return; const p=new URLSearchParams(); p.set('run',val('run')); p.set('q',val('candQ')); const data=await getJSON('/api/candidates?' + p.toString()); document.getElementById('candidates').innerHTML=table(data.rows||[], ['target_id','peak_line_nm','candidate_line_nm','quality_tier','rank_score','support_count','detectors']); }
set('run', rawParam('run')); set('target', rawParam('target') || rawParam('target_id')); set('line', rawParam('line_nm')); if(val('run')) loadCandidates(); if(val('run') && val('target')) inspect().catch(e=>document.getElementById('status').textContent=e.stack||String(e));
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
