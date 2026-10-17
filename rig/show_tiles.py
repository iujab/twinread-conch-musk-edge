#!/usr/bin/env python
"""Show held-out CRC test tiles with a frame-ID marker beside each, for a camera to film (SIMULATION-FREE real screen capture).

Layout (see marker.py): 1280x800 page, 672 px tile (3x upscaled 224 px) at TILE_XY, marker at MARKER_XY to its left.
Modes: browser page at http://127.0.0.1:1778/rig (default) or --window (OpenCV fullscreen). Localhost only; no other network use.
Writes rig_session.json = {session_id, session_marker, seed, n, x_sha256, map: {frame_id: test_index}}. Research prototype, not a diagnostic device.
"""
import argparse
import hashlib
import json
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import marker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PORT = 1778


def render_page(tile: np.ndarray, frame_id: int, session: int = 0) -> np.ndarray:
    """BGR page (PAGE_H x PAGE_W). `tile` is an HxWx3 uint8 RGB tile; stored as given (channel order is the caller's)."""
    page = np.full((marker.PAGE_H, marker.PAGE_W, 3), 255, np.uint8)
    t = cv2.resize(np.ascontiguousarray(tile), (marker.TILE_PX, marker.TILE_PX), interpolation=cv2.INTER_CUBIC)
    x, y = marker.TILE_XY
    f = marker.FRAME_PX
    page[y - f:y + marker.TILE_PX + f, x - f:x + marker.TILE_PX + f] = 0
    page[y:y + marker.TILE_PX, x:x + marker.TILE_PX] = t
    m = cv2.resize(marker.encode_marker(frame_id), (marker.MARKER_PX, marker.MARKER_PX), interpolation=cv2.INTER_NEAREST)
    mx, my = marker.MARKER_XY
    page[my:my + marker.MARKER_PX, mx:mx + marker.MARKER_PX] = m
    sm = cv2.resize(marker.encode_marker(session % 2048), (marker.MARKER_PX, marker.MARKER_PX), interpolation=cv2.INTER_NEAREST)
    sx, sy = marker.SESSION_XY
    page[sy:sy + marker.MARKER_PX, sx:sx + marker.MARKER_PX] = sm
    return page


def blank_page():
    return np.full((marker.PAGE_H, marker.PAGE_W, 3), 255, np.uint8)


def load_plan(n, seed, x_path, y_path):
    x = np.load(x_path, mmap_mode="r")
    if n > 2048:
        raise SystemExit("at most 2048 tiles per session (11-bit frame ids)")
    rng = np.random.RandomState(seed)
    n = min(n, len(x))
    idx = np.sort(rng.choice(len(x), n, replace=False)) if n < len(x) else np.arange(len(x))
    rng.shuffle(idx)
    return x, [int(i) for i in idx]


def tile_to_bgr(tile_rgb):
    return cv2.cvtColor(np.ascontiguousarray(tile_rgb), cv2.COLOR_RGB2BGR)


HTML = """<!doctype html><meta charset=utf-8><title>TwinRead rig</title>
<body style="margin:0;background:#fff;overflow:hidden"><img id=p style="display:block;margin:0 auto;image-rendering:auto">
<script>
const N=%(n)d, DWELL=%(dwell)d, GAP=%(gap)d;let k=0;const p=document.getElementById('p');
function fit(){p.style.width=(%(w)d/devicePixelRatio)+'px'}fit();addEventListener('resize',fit);
function show(){p.src='/page/'+k+'.png?'+Date.now();k=(k+1)%%N;setTimeout(()=>{p.src='/blank.png';setTimeout(show,GAP)},DWELL)}
show();
</script>"""


def make_handler(x, plan, dwell_ms, gap_ms, session=0):
    cache = {}
    blank = cv2.imencode(".png", blank_page())[1].tobytes()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/rig", "/"):
                self._send((HTML % dict(n=len(plan), dwell=dwell_ms, gap=gap_ms, w=marker.PAGE_W)).encode(), "text/html")
            elif path == "/blank.png":
                self._send(blank, "image/png")
            elif path.startswith("/page/") and path.endswith(".png"):
                try:
                    k = int(path[6:-4])
                    assert 0 <= k < len(plan)
                except Exception:
                    self.send_error(404)
                    return
                if k not in cache:
                    cache[k] = cv2.imencode(".png", render_page(tile_to_bgr(x[plan[k]]), k, session))[1].tobytes()
                self._send(cache[k], "image/png")
            else:
                self.send_error(404)

    return H


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=600, help="number of held-out test tiles to cycle (<=2048)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dwell", type=float, default=1.5, help="seconds per tile")
    ap.add_argument("--gap", type=float, default=0.3, help="seconds of blank page between tiles (marker-less frames get dropped)")
    ap.add_argument("--x", default=str(ROOT / "data" / "crc_test_x.npy"))
    ap.add_argument("--y", default=str(ROOT / "data" / "crc_test_y.npy"))
    ap.add_argument("--session", default="rig_session.json")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--window", action="store_true", help="fullscreen OpenCV window instead of the browser page (q quits)")
    a = ap.parse_args(argv)
    x, plan = load_plan(a.n, a.seed, a.x, a.y)
    sid = secrets.token_hex(8)
    smark = int(sid, 16) % 2048
    h = hashlib.sha256()
    with open(a.x, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    Path(a.session).write_text(json.dumps(dict(
        session_id=sid, session_marker=smark, seed=a.seed, n=len(plan), x_sha256=h.hexdigest(),
        map={str(k): i for k, i in enumerate(plan)})))
    print(f"session {sid} (marker {smark}): {len(plan)} tiles -> {a.session}")
    if a.window:
        cv2.namedWindow("rig", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("rig", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        k = 0
        while True:
            cv2.imshow("rig", render_page(tile_to_bgr(x[plan[k]]), k, smark))
            if cv2.waitKey(int(a.dwell * 1000)) & 0xFF == ord("q"):
                break
            cv2.imshow("rig", blank_page())
            if cv2.waitKey(int(a.gap * 1000)) & 0xFF == ord("q"):
                break
            k = (k + 1) % len(plan)
        cv2.destroyAllWindows()
        return 0
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(x, plan, int(a.dwell * 1000), int(a.gap * 1000), smark))
    print(f"open http://127.0.0.1:{a.port}/rig fullscreen (F11); Ctrl-C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
