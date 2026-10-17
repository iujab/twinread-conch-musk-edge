#!/usr/bin/env python
"""TwinRead device app (Arduino UNO Q, offline). Research prototype - not a diagnostic device.

loop: frame -> crop -> infer -> send_status -> JSON log line -> UI state.
  GET /          static UI (ui/)           GET /state     JSON state          GET /frame.jpg   latest frame with crop box
Binds 127.0.0.1 only; makes no network calls. Imports: numpy, onnxruntime (via twinread_runtime), cv2, Pillow, stdlib.

usage: uno_q_app.py --bundle DIR --mode camera|replay|sim [--device /dev/video0] [--port 8080] [--threads 4]
                    [--replay-dir DIR] [--crop center|marker] [--interval 1.0]
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import platform
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge  # noqa: E402
import twinread_runtime as rt  # noqa: E402

UI_DIR = Path(__file__).resolve().parent / "ui"
DT_MODEL = Path("/proc/device-tree/model")
FRAME_LABELS = {"camera": "Live frame", "replay": "Replay frame", "sim": "Simulated frame"}


def host_info(dt_model: Path = DT_MODEL) -> dict:
    """Which machine runs the app. The UNO Q is an aarch64 board whose device-tree model names Arduino; anything else
    (e.g. the x86 desktop used for screenshots) is labelled as not the UNO Q, so its latency is never mistaken for
    on-board latency."""
    machine = platform.machine() or "unknown"
    try:
        model = dt_model.read_text(errors="replace").strip("\x00\n ")
    except OSError:
        model = ""
    is_uno_q = machine in ("aarch64", "arm64") and "arduino" in model.lower()
    return {"machine": machine, "model": model, "is_uno_q": is_uno_q}
BANNER = "Research prototype - not a diagnostic device"
HOST = "127.0.0.1"


def synth_frame(i: int, h=480, w=640) -> np.ndarray:
    """Procedural H&E-like RGB frame for sim/replay-without-files. Every 4th frame is blank glass."""
    rng = np.random.default_rng(1000 + i)
    if i % 4 == 3:
        return np.full((h, w, 3), 250, np.uint8)
    base = np.array([[225, 150, 190], [150, 90, 160], [235, 190, 205]][i % 3], np.float32)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 6 + 2 * (i % 3))
    noise = noise / (noise.std() + 1e-6)
    img = base[None, None, :] + noise[..., None] * np.array([22, 28, 18], np.float32) \
        + rng.normal(0, 4, (h, w, 3)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


class App:
    def __init__(self, bundle, mode="sim", device="/dev/video0", threads=4, replay_dir=None, crop="center", interval=1.0):
        self.model = rt.EdgeModel(Path(bundle), threads=threads)
        self.requested_mode, self.mode, self.mode_reason = mode, mode, None
        self.device, self.crop_mode, self.interval = device, crop, interval
        self.cap = None
        self.replay_files = []
        if replay_dir:
            self.replay_files = sorted(p for p in Path(replay_dir).iterdir()
                                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        self.i = 0
        self.host = host_info()
        self.lock = threading.Lock()
        self.jpg = b""
        self._state = {"status": "idle", "mode": self.mode, "call": None}
        self.pair = str(self.model.cfg.get("pair", "A+B"))
        self.readers = (self.pair.split("+") + ["B"])[:2] if "+" in self.pair else ["Reader A", "Reader B"]
        if mode == "camera":
            self._open_camera()
        self._set_banner()
        self._publish(status="idle")
        bridge.send_status("idle")

    # -- sources -------------------------------------------------------------------------------------------------
    def _open_camera(self):
        try:
            cap = cv2.VideoCapture(self.device)
            ok = cap.isOpened() and cap.read()[0]
        except Exception as e:  # noqa: BLE001
            cap, ok = None, False
            self.mode_reason = f"camera error: {e}"
        if ok:
            self.cap = cap
        else:
            if cap is not None:
                cap.release()
            self._fallback(self.mode_reason or f"camera {self.device} not available")

    def _fallback(self, reason):
        self.cap = None
        self.mode = "replay"
        self.mode_reason = f"camera fallback: {reason}; replaying " + \
            (f"{len(self.replay_files)} recorded frames" if self.replay_files else "synthetic frames")
        print(json.dumps({"event": "mode_fallback", "mode": "replay", "reason": self.mode_reason}), flush=True)

    def _set_banner(self):
        self.mode_banner = {"camera": "Camera filming a screen (demo)", "sim": "SIMULATION",
                            "replay": "REPLAY (stored/synthetic frames, not live)"}[self.mode]

    def _next_frame(self) -> np.ndarray:
        """RGB uint8 [H,W,3]."""
        if self.mode == "camera" and self.cap is not None:
            ok, bgr = self.cap.read()
            if ok and bgr is not None:
                return np.ascontiguousarray(bgr[..., ::-1])
            self._fallback("camera stopped delivering frames")
            self._set_banner()
        if self.mode == "replay" and self.replay_files:
            bgr = cv2.imread(str(self.replay_files[self.i % len(self.replay_files)]))
            if bgr is not None:
                return np.ascontiguousarray(bgr[..., ::-1])
        return synth_frame(self.i)

    # -- loop ----------------------------------------------------------------------------------------------------
    def step(self):
        try:
            self._publish(status="busy", keep=True)
            bridge.send_status("busy")
            t0 = time.perf_counter()
            frame = self._next_frame()
            tile = rt.crop_from_frame(frame, self.crop_mode)
            r = self.model.infer(tile[None])[0]
            ms = (time.perf_counter() - t0) * 1000
            status = "no_tissue" if r.no_tissue else ("review" if r.review else "agree")
            led = bridge.send_status(status)
            tf = rt.tissue_fraction(tile)
            h, w = frame.shape[:2]
            x0, y0, x1, y1 = rt.center_box(h, w)
            vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (0, 220, 255), 2)
            ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with self.lock:
                self.jpg = buf.tobytes() if ok else b""
            call = {"label": r.label, "prob": r.prob, "probsA": r.probsA, "probsB": r.probsB, "probsF": r.probsF,
                    "labelA": self.model.classes[int(np.argmax(r.probsA))],
                    "labelB": self.model.classes[int(np.argmax(r.probsB))], "review_score": r.review_score}
            self._publish(status=status, call=call, reasons=r.reasons, tissue_fraction=tf, latency_ms=ms, led=led,
                          crop_box=[x0 / w, y0 / h, x1 / w, y1 / h], n=self.i)
            print(json.dumps({"event": "tile", "n": self.i, "mode": self.mode, "status": status, "label": r.label,
                              "prob": round(r.prob, 4), "reasons": r.reasons, "tissue_fraction": round(tf, 3),
                              "latency_ms": round(ms, 1), "led": led}), flush=True)
        except Exception as e:  # noqa: BLE001 - keep serving the UI
            bridge.send_status("error")
            self._publish(status="error", error=str(e), keep=True)
            print(json.dumps({"event": "error", "error": str(e)}), flush=True)
        self.i += 1

    def _publish(self, status, keep=False, **kw):
        with self.lock:
            st = dict(self._state) if keep else {}
            st.update(status=status, mode=self.mode, mode_reason=self.mode_reason, requested_mode=self.requested_mode,
                      banner=BANNER, mode_banner=self.mode_banner if hasattr(self, "mode_banner") else "",
                      bundle_note=("simulation with test bundle" if self.mode == "sim" else ""),
                      pair=self.pair, readers=self.readers, classes=self.model.classes, thresholds=self.model.thr,
                      distributable=self.model.cfg.get("distributable"), host=self.host,
                      host_label=(f"host: {self.host['model'] or 'UNO Q'}" if self.host["is_uno_q"] else
                                  f"host: {self.host['machine']}, not the UNO Q (latency is this host's)"),
                      frame_label=FRAME_LABELS.get(self.mode, "Frame"), **kw)
            st.setdefault("call", None)
            self._state = st

    def state(self) -> dict:
        with self.lock:
            return dict(self._state)

    def frame_jpg(self) -> bytes:
        with self.lock:
            return self.jpg

    def run_forever(self, stop: threading.Event):
        while not stop.is_set():
            t = time.perf_counter()
            self.step()
            stop.wait(max(0.0, self.interval - (time.perf_counter() - t)))


def make_server(app: App, port: int) -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/state":
                return self._send(200, json.dumps(app.state()).encode(), "application/json")
            if path == "/frame.jpg":
                b = app.frame_jpg()
                return self._send(200 if b else 404, b or b"no frame yet", "image/jpeg" if b else "text/plain")
            name = "index.html" if path in ("/", "") else path.lstrip("/")
            f = (UI_DIR / name).resolve()
            if UI_DIR.resolve() in f.parents and f.is_file():
                return self._send(200, f.read_bytes(), mimetypes.guess_type(str(f))[0] or "application/octet-stream")
            self._send(404, b"not found", "text/plain")

    return ThreadingHTTPServer((HOST, port), H)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--mode", choices=["camera", "replay", "sim"], default="camera")
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--replay-dir", default=None)
    ap.add_argument("--crop", choices=["center", "marker"], default="center")
    ap.add_argument("--interval", type=float, default=1.0)
    a = ap.parse_args(argv)
    app = App(a.bundle, a.mode, a.device, a.threads, a.replay_dir, a.crop, a.interval)
    srv = make_server(app, a.port)
    stop = threading.Event()
    threading.Thread(target=app.run_forever, args=(stop,), daemon=True).start()
    print(json.dumps({"event": "serving", "url": f"http://{HOST}:{a.port}/", "mode": app.mode,
                      "reason": app.mode_reason}), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        srv.server_close()
        bridge.send_status("idle")


if __name__ == "__main__":
    main()
