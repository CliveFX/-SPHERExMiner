from __future__ import annotations

import argparse
import html
import io
import json
import math
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


DEFAULT_MANIFEST = Path(
    "/mnt/niroseti/spherex_cache/injections/smoke_ucs0972_1064nm_one_frame/injection_manifest.json"
)


def _escape(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _query_float(params: dict[str, list[str]], name: str, default: float) -> float:
    return _safe_float((params.get(name) or [default])[0], default)


def _query_int(params: dict[str, list[str]], name: str, default: int) -> int:
    try:
        return int(float((params.get(name) or [default])[0]))
    except Exception:
        return default


class BlinkState:
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.lock = threading.Lock()
        self._manifest_mtime = 0.0
        self._manifest: dict[str, Any] | None = None

    def manifest(self) -> dict[str, Any]:
        with self.lock:
            mtime = self.manifest_path.stat().st_mtime
            if self._manifest is None or mtime != self._manifest_mtime:
                self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                self._manifest_mtime = mtime
            return self._manifest

    def frames(self) -> list[dict[str, Any]]:
        frames = self.manifest().get("frames", [])
        return [frame for frame in frames if frame.get("injected_path")]

    def frame(self, index: int) -> dict[str, Any]:
        frames = self.frames()
        if not frames:
            raise FileNotFoundError("manifest has no frames with injected_path")
        return frames[max(0, min(index, len(frames) - 1))]


def _read_image(path: Path) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul["IMAGE"] if "IMAGE" in hdul else hdul[0]
        return np.asarray(hdu.data, dtype=float)


def _crop_pair(
    original_path: Path,
    injected_path: Path,
    x_pix: float,
    y_pix: float,
    crop_size: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    original = _read_image(original_path)
    injected = _read_image(injected_path)
    if original.shape != injected.shape:
        raise ValueError(f"shape mismatch: original={original.shape} injected={injected.shape}")

    radius = max(8, int(crop_size) // 2)
    x0 = int(round(x_pix))
    y0 = int(round(y_pix))
    x1 = max(0, x0 - radius)
    x2 = min(original.shape[1], x0 + radius)
    y1 = max(0, y0 - radius)
    y2 = min(original.shape[0], y0 + radius)
    if x1 >= x2 or y1 >= y2:
        raise ValueError("requested crop is outside image")
    return original[y1:y2, x1:x2], injected[y1:y2, x1:x2], (x1, x2, y1, y2)


def _limits(values: np.ndarray, low_pct: float, high_pct: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [low_pct, high_pct])
    if not math.isfinite(float(vmin)) or not math.isfinite(float(vmax)) or vmin == vmax:
        med = float(np.nanmedian(finite))
        spread = float(np.nanstd(finite)) or 1.0
        return med - spread, med + spread
    return float(vmin), float(vmax)


def _render_png(
    frame: dict[str, Any],
    mode: str,
    crop_size: int,
    low_pct: float,
    high_pct: float,
    diff_scale: float,
    overlay: bool,
) -> bytes:
    original_path = Path(str(frame["original_path"]))
    injected_path = Path(str(frame["injected_path"]))
    x_pix = _safe_float(frame.get("x_pix"))
    y_pix = _safe_float(frame.get("y_pix"))
    original, injected, extent = _crop_pair(original_path, injected_path, x_pix, y_pix, crop_size)
    x1, x2, y1, y2 = extent

    if mode == "injected":
        image = injected
        cmap = "gray"
        vmin, vmax = _limits(np.concatenate([original.ravel(), injected.ravel()]), low_pct, high_pct)
        title = "injected"
    elif mode == "diff":
        image = injected - original
        cmap = "seismic"
        finite = image[np.isfinite(image)]
        if finite.size:
            base = float(np.percentile(np.abs(finite), 99.5)) or 1.0
        else:
            base = 1.0
        vmax = base / max(diff_scale, 0.01)
        vmin = -vmax
        title = "injected - original"
    else:
        image = original
        cmap = "gray"
        vmin, vmax = _limits(np.concatenate([original.ravel(), injected.ravel()]), low_pct, high_pct)
        title = "original"

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=120)
    fig.patch.set_facecolor("#020617")
    ax.set_facecolor("#020617")
    ax.imshow(
        image,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[x1, x2, y1, y2],
        interpolation="nearest",
    )
    if overlay:
        radius = 5.0 if mode == "diff" else 7.0
        ax.add_patch(
            plt.Circle(
                (x_pix, y_pix),
                radius,
                fill=False,
                edgecolor="#38f8ff" if mode != "diff" else "#facc15",
                linewidth=1.0,
                alpha=0.95,
            )
        )
        ax.plot([x_pix - 13, x_pix - 7], [y_pix, y_pix], color="#facc15", lw=0.8)
        ax.plot([x_pix + 7, x_pix + 13], [y_pix, y_pix], color="#facc15", lw=0.8)
        ax.plot([x_pix, x_pix], [y_pix - 13, y_pix - 7], color="#facc15", lw=0.8)
        ax.plot([x_pix, x_pix], [y_pix + 7, y_pix + 13], color="#facc15", lw=0.8)
    ax.set_title(title, color="#dbeafe", fontsize=10)
    ax.tick_params(colors="#8aa4bf", labelsize=7, length=2)
    for spine in ax.spines.values():
        spine.set_color("#1f7895")
        spine.set_linewidth(0.7)
    fig.tight_layout(pad=0.4)
    out = io.BytesIO()
    fig.savefig(out, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out.getvalue()


def _page(manifest_path: Path) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SPHEREx Injection Blink Comparator</title>
  <style>
    :root {{
      --bg:#040b12; --panel:#091827; --panel2:#0d2236; --line:#1d4e68;
      --text:#e5f4ff; --muted:#8ba7bd; --cyan:#38f8ff; --pink:#ff4fd8;
      --amber:#facc15; --green:#72ff9e;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; color:var(--text); font:13px/1.35 system-ui,-apple-system,Segoe UI,sans-serif;
      background:
        linear-gradient(90deg, rgba(56,248,255,.055) 1px, transparent 1px),
        linear-gradient(rgba(255,79,216,.035) 1px, transparent 1px),
        var(--bg);
      background-size: 38px 38px;
    }}
    header {{
      display:flex; align-items:center; justify-content:space-between; gap:16px;
      padding:12px 16px; border-bottom:1px solid var(--line); background:rgba(4,11,18,.94);
      position:sticky; top:0; z-index:2;
    }}
    h1 {{ margin:0; color:var(--cyan); font-size:18px; letter-spacing:0; text-shadow:0 0 14px rgba(56,248,255,.35); }}
    main {{ display:grid; grid-template-columns:340px minmax(680px,1fr); gap:12px; padding:12px; }}
    section {{ border:1px solid var(--line); background:rgba(9,24,39,.94); border-radius:6px; padding:12px; min-width:0; }}
    .controls {{ align-self:start; position:sticky; top:58px; }}
    label {{ display:grid; grid-template-columns:1fr auto; gap:10px; color:var(--muted); font-size:12px; margin:10px 0 4px; }}
    input, select, button {{
      width:100%; border:1px solid var(--line); background:#06111d; color:var(--text);
      border-radius:4px; padding:8px;
    }}
    input[type=range] {{ padding:0; accent-color:var(--cyan); }}
    input[type=checkbox] {{ width:auto; }}
    button {{ cursor:pointer; }}
    button:hover {{ border-color:var(--cyan); color:white; }}
    .row {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
    .stageGrid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; align-items:start; }}
    .stageGrid.single {{ grid-template-columns:1fr; }}
    .plate {{
      border:1px solid #17617e; background:#020617; border-radius:6px; overflow:hidden;
      min-height:360px; box-shadow:0 0 30px rgba(0,0,0,.22), inset 0 0 0 1px rgba(56,248,255,.04);
    }}
    .plate img {{ display:block; width:100%; height:auto; }}
    .blinkPlate {{ position:relative; }}
    .blinkPlate img {{ transition:opacity 28ms linear; }}
    .blinkPlate img.blinkTop {{ position:absolute; inset:0; opacity:0; }}
    .blinkPlate.showInjected img.blinkTop {{ opacity:1; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,minmax(120px,1fr)); gap:8px; margin-bottom:10px; }}
    .metric {{ background:rgba(6,17,29,.92); border:1px solid var(--line); border-radius:5px; padding:8px; overflow-wrap:anywhere; }}
    .k {{ color:var(--muted); font-size:10px; text-transform:uppercase; }}
    .v {{ margin-top:3px; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
    .hint {{ color:var(--muted); font-size:12px; margin-top:10px; }}
    code {{ color:var(--green); overflow-wrap:anywhere; }}
    @media (max-width: 980px) {{ main {{ grid-template-columns:1fr; }} .controls {{ position:static; }} .stageGrid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>SPHEREx Injection Blink Comparator</h1>
    <div><code>{_escape(manifest_path)}</code></div>
  </header>
  <main>
    <section class="controls">
      <label>Frame <span id="frameCount"></span></label>
      <select id="frameSelect"></select>

      <div class="row">
        <button id="prevBtn">Prev</button>
        <button id="nextBtn">Next</button>
      </div>

      <label>Mode</label>
      <select id="mode">
        <option value="blink">Blink original/injected</option>
        <option value="side">Side by side</option>
        <option value="diff">Difference only</option>
        <option value="original">Original only</option>
        <option value="injected">Injected only</option>
      </select>

      <label>Blink interval <span id="intervalValue"></span></label>
      <input id="intervalMs" type="range" min="120" max="1600" step="20" value="420">

      <label>Crop size <span id="cropValue"></span></label>
      <input id="cropSize" type="range" min="32" max="320" step="8" value="96">

      <div class="row">
        <div>
          <label>Low pct <span id="lowValue"></span></label>
          <input id="lowPct" type="range" min="0" max="20" step="0.2" value="1">
        </div>
        <div>
          <label>High pct <span id="highValue"></span></label>
          <input id="highPct" type="range" min="80" max="100" step="0.05" value="99.7">
        </div>
      </div>

      <label>Diff boost <span id="diffValue"></span></label>
      <input id="diffScale" type="range" min="0.2" max="12" step="0.1" value="2.5">

      <label><span>Overlay reticle</span><input id="overlay" type="checkbox" checked></label>
      <label><span>Autoplay blink</span><input id="play" type="checkbox" checked></label>

      <div class="hint">
        For entertainment: the injected source will usually be hard to see in a normal image stretch.
        Use blink for plate-comparison vibes, then switch to difference and boost it to see the PSF-shaped residual.
      </div>
    </section>

    <section>
      <div class="metrics">
        <div class="metric"><div class="k">Target</div><div class="v" id="targetId"></div></div>
        <div class="metric"><div class="k">Line</div><div class="v" id="lineInfo"></div></div>
        <div class="metric"><div class="k">Frame Flux</div><div class="v" id="frameFlux"></div></div>
        <div class="metric"><div class="k">Response</div><div class="v" id="response"></div></div>
      </div>
      <div id="stage" class="stageGrid"></div>
      <div class="hint" id="pathInfo"></div>
    </section>
  </main>
  <script>
    const manifestUrl = '/api/manifest';
    const state = {{ manifest: null, frames: [], frameIndex: 0, blinkOnInjected: false, timer: null }};
    const $ = id => document.getElementById(id);

    function imageUrl(frameIndex, mode) {{
      const p = new URLSearchParams();
      p.set('idx', frameIndex);
      p.set('mode', mode);
      p.set('crop', $('cropSize').value);
      p.set('low', $('lowPct').value);
      p.set('high', $('highPct').value);
      p.set('diffScale', $('diffScale').value);
      p.set('overlay', $('overlay').checked ? '1' : '0');
      return '/api/image?' + p.toString();
    }}

    function fmt(n, digits=3) {{
      const x = Number(n);
      if (!Number.isFinite(x)) return '';
      if (Math.abs(x) >= 1000 || Math.abs(x) < 0.01) return x.toExponential(2);
      return x.toFixed(digits);
    }}

    function selectedFrame() {{ return state.frames[state.frameIndex] || {{}}; }}

    function updateMetrics() {{
      const m = state.manifest || {{}};
      const f = selectedFrame();
      $('targetId').textContent = m.target_id || '';
      $('lineInfo').textContent = m.line ? `${{fmt(m.line.line_nm, 1)}} nm` : '';
      $('frameFlux').textContent = f.injected_flux_uJy ? `${{fmt(f.injected_flux_uJy, 2)}} uJy` : '';
      $('response').textContent = f.line_response ? fmt(f.line_response, 4) : '';
      $('pathInfo').innerHTML = `image_id: <code>${{f.image_id || ''}}</code><br>original: <code>${{f.original_path || ''}}</code><br>injected: <code>${{f.injected_path || ''}}</code>`;
      $('frameCount').textContent = `${{state.frameIndex + 1}} / ${{state.frames.length}}`;
    }}

    function render() {{
      $('intervalValue').textContent = $('intervalMs').value + ' ms';
      $('cropValue').textContent = $('cropSize').value + ' px';
      $('lowValue').textContent = $('lowPct').value;
      $('highValue').textContent = $('highPct').value;
      $('diffValue').textContent = $('diffScale').value + 'x';
      updateMetrics();
      const mode = $('mode').value;
      const stage = $('stage');
      stage.className = 'stageGrid';
      if (mode === 'side') {{
        stage.innerHTML = `<div class="plate"><img src="${{imageUrl(state.frameIndex, 'original')}}"></div><div class="plate"><img src="${{imageUrl(state.frameIndex, 'injected')}}"></div>`;
      }} else if (mode === 'blink') {{
        stage.className = 'stageGrid single';
        stage.innerHTML = `<div id="blinkPlate" class="plate blinkPlate ${{state.blinkOnInjected ? 'showInjected' : ''}}"><img src="${{imageUrl(state.frameIndex, 'original')}}" alt="original"><img class="blinkTop" src="${{imageUrl(state.frameIndex, 'injected')}}" alt="injected"></div>`;
      }} else {{
        stage.className = 'stageGrid single';
        stage.innerHTML = `<div class="plate"><img src="${{imageUrl(state.frameIndex, mode)}}"></div>`;
      }}
    }}

    function tickBlink() {{
      if (!$('play').checked || $('mode').value !== 'blink') return;
      state.blinkOnInjected = !state.blinkOnInjected;
      const plate = $('blinkPlate');
      if (plate) plate.classList.toggle('showInjected', state.blinkOnInjected);
    }}

    function resetTimer() {{
      if (state.timer) clearInterval(state.timer);
      state.timer = setInterval(() => {{
        tickBlink();
      }}, Number($('intervalMs').value));
    }}

    function setFrame(index) {{
      state.frameIndex = Math.max(0, Math.min(index, state.frames.length - 1));
      $('frameSelect').value = String(state.frameIndex);
      render();
    }}

    async function init() {{
      state.manifest = await fetch(manifestUrl).then(r => r.json());
      state.frames = (state.manifest.frames || []).filter(f => f.injected_path);
      $('frameSelect').innerHTML = state.frames.map((f, i) => `<option value="${{i}}">${{i + 1}} | ${{f.image_id}} | ${{fmt(f.cwave_um, 4)}} um</option>`).join('');
      $('frameSelect').onchange = () => setFrame(Number($('frameSelect').value));
      $('prevBtn').onclick = () => setFrame(state.frameIndex - 1);
      $('nextBtn').onclick = () => setFrame(state.frameIndex + 1);
      for (const id of ['mode','intervalMs','cropSize','lowPct','highPct','diffScale','overlay','play']) {{
        $(id).addEventListener('input', () => {{ render(); resetTimer(); }});
        $(id).addEventListener('change', () => {{ render(); resetTimer(); }});
      }}
      render();
      resetTimer();
    }}
    init().catch(err => {{ document.body.innerHTML = '<pre style="color:#ffb4b4;padding:20px">' + err.stack + '</pre>'; }});
  </script>
</body>
</html>"""


class BlinkHandler(BaseHTTPRequestHandler):
    state: BlinkState

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, value: object) -> None:
        self._send(json.dumps(value, indent=2).encode("utf-8"), "application/json; charset=utf-8")

    def _send_error(self, status: int, message: str) -> None:
        self._send(json.dumps({"error": message}).encode("utf-8"), "application/json", status=status)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path in {"/", "/index.html"}:
                self._send(_page(self.state.manifest_path).encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/manifest":
                self._send_json(self.state.manifest())
            elif parsed.path == "/api/image":
                idx = _query_int(params, "idx", 0)
                mode = (params.get("mode") or ["original"])[0]
                crop = _query_int(params, "crop", 96)
                low = _query_float(params, "low", 1.0)
                high = _query_float(params, "high", 99.7)
                diff_scale = _query_float(params, "diffScale", 2.5)
                overlay = (params.get("overlay") or ["1"])[0] != "0"
                body = _render_png(
                    self.state.frame(idx),
                    mode=mode,
                    crop_size=crop,
                    low_pct=low,
                    high_pct=high,
                    diff_scale=diff_scale,
                    overlay=overlay,
                )
                self._send(body, "image/png")
            else:
                self._send_error(404, "not found")
        except Exception as exc:
            self._send_error(500, f"{type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Blink original/injected SPHEREx FITS cutouts.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18779)
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Missing manifest: {args.manifest}")

    state = BlinkState(args.manifest.resolve())
    handler = type("ConfiguredBlinkHandler", (BlinkHandler,), {"state": state})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        json.dumps(
            {
                "url": f"http://{args.host}:{args.port}/",
                "manifest": str(args.manifest.resolve()),
                "frames": len(state.frames()),
            },
            indent=2,
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
