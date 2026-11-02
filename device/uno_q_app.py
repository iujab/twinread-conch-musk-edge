#!/usr/bin/env python
"""TwinRead device app and dashboard (Arduino UNO Q, offline).

loop: picture -> crop -> infer -> send_status -> JSON log line -> UI state.
  GET /          dashboard (ui/)     GET /state        JSON state
  GET /frame.jpg latest picture      GET /stream.mjpg  live video, both with the read region drawn on
  GET /slide/N.jpg demo thumbnail    POST /control     start or pause the model and the camera, switch source, move through slides,
  GET /explain/picture.jpg the tile an explanation is about       and start or stop an explanation (see explainer.py)
Binds 127.0.0.1 only; makes no network calls. Imports: numpy, onnxruntime (via twinread_runtime), cv2, Pillow, stdlib.

usage: uno_q_app.py --bundle DIR --mode camera|demo|replay|sim [--standby] [--device auto|/dev/videoN]
                    [--camera-loss wait|replay] [--demo-dir DIR] [--port 8080] [--threads 4] [--replay-dir DIR]
                    [--crop center|marker] [--interval 1.0] [--explain-* see --help]

--standby starts with the model unloaded and the camera closed, serving only the dashboard, at about 0% CPU.
--camera-loss wait (the kiosk) keeps retrying and says the camera is missing; it never reads anything but the camera.
--camera-loss replay (the default) falls back to labelled stored or synthetic frames when the camera is missing.

Pauses are real, because heat matters on this board. Pausing the model drops the ONNX session and frees its memory; pausing
the camera closes the V4L2 device and shows the last frame; demo slides also close the camera. Nothing a dashboard button
starts can take the dashboard down: a model or camera that fails stays off and the page says why. Pauses and the source are
not remembered across a restart.
"""
from __future__ import annotations

import argparse
import collections
import ipaddress
import json
import mimetypes
import os
import platform
import re
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge  # noqa: E402
import camera  # noqa: E402
import demo  # noqa: E402
import explain_prompt as ep  # noqa: E402
import explainer  # noqa: E402
import twinread_runtime as rt  # noqa: E402

UI_DIR = Path(__file__).resolve().parent / "ui"
DEFAULT_DEMO_DIR = Path(__file__).resolve().parents[1] / "data" / "demo_slides"
DT_MODEL = Path("/proc/device-tree/model")
FRAME_LABELS = {"camera": "Live", "demo": "Demo slide", "replay": "Replay frame", "sim": "Simulated frame"}
SOURCE_NOTES = {"replay": "Replaying stored frames, not live", "sim": "Simulated frames, not live"}
READ_REGION_BGR = (0, 196, 244)  # the yellow box, #f4c400
STALE_S = 3.0        # a camera frame older than this is not "live" any more
PREVIEW_MAX_W = 960  # the on-screen feed; inference always uses the full-resolution frame
PREVIEW_IDLE_FPS = 1.0
DEMO_PREVIEW = 720   # a 224 px slide is enlarged to this for the screen; the model reads the original
LOG_EVERY = 20       # with a live camera: log every status change, otherwise only every Nth read
VITALS_EVERY_S = 2.0
IDLE_TICK_S = 5.0    # how often the loop wakes while nothing is running; it also retries a failed LED call
STREAM_RESEND_S = 5.0  # a held picture is resent this often; that write is what notices a closed browser
MAX_CONTROL_BYTES = 1024
HANDLER_TIMEOUT_S = 30.0   # a request that stalls this long, or a viewer that stops taking video this long, is dropped
MAX_STREAMS = 4            # each live video costs the board 60 to 80 CPU points of one core
MAX_CONNECTIONS = 32
IDLE_STANDBY_MIN = 10.0    # with no viewer for this long, the model and the camera go off
RESULT_GONE = "That result has gone from the page. Press Explain again for the one on screen now."
DEMO_INTERVAL_RANGE = (1.0, 60.0)
DEMO_INTERVAL_DEFAULT = 4.0
CONTROL_KEYS = {"model_paused", "feed_paused", "source", "demo_step", "demo_index", "demo_interval"}
EXPLAIN_KEYS = {"explain", "explain_decision", "explain_stop", "explain_close"}
RECENT_DECISIONS = 8  # at 2 reads a second this reaches back about 4 s, longer than any page's polling, so a click explains what its viewer saw
DECIDED = ("agree", "review", "no_tissue")
EXPLAIN_WHERE = {"Live": "Live camera", "Frozen frame": "Frozen camera frame"}
CAMERA_NEVER_OPENED = {"connected": False, "device": None, "name": None, "width": None, "height": None, "fps": None,
                       "reason": None, "open": False, "paused": True, "opening": False}


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
    return {"machine": machine, "model": model, "is_uno_q": is_uno_q, "hostname": platform.node()}


def system_vitals() -> dict:
    """What the dashboard shows about the machine. Anything unreadable is simply left out."""
    out = {}
    try:
        out["uptime_s"] = float(Path("/proc/uptime").read_text().split()[0])
        out["load"] = round(os.getloadavg()[0], 2)
        mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        out["mem_available_mb"] = int(mem["MemAvailable"].split()[0]) // 1024
    except (OSError, ValueError, KeyError):
        pass
    temps = []
    for p in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            temps.append(int(p.read_text()) / 1000)
        except (OSError, ValueError):
            pass
    if temps:
        out["temp_c"] = round(max(temps), 1)
    return out


HOST = "127.0.0.1"
# The device's two heads share ONE student backbone (distilled into the teachers' aligned spaces); the UI names them as
# heads.
HEAD_LABELS = {"CM": ["Head A (CONCH-space)", "Head B (MUSK-space)"]}
GENERIC_HEADS = ["Head A (space A)", "Head B (space B)"]


def head_labels(pair: str) -> list:
    return list(HEAD_LABELS.get(str(pair).upper(), GENERIC_HEADS))


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


def preview_jpeg(bgr: np.ndarray, quality: int = 80) -> bytes:
    """Downscaled copy of the frame with the region the model reads drawn in yellow. Never writes into bgr."""
    h, w = bgr.shape[:2]
    if w > PREVIEW_MAX_W:
        vis = cv2.resize(bgr, (PREVIEW_MAX_W, int(round(h * PREVIEW_MAX_W / w))), interpolation=cv2.INTER_AREA)
    else:
        vis = bgr.copy()
    vh, vw = vis.shape[:2]
    x0, y0, x1, y1 = rt.center_box(vh, vw)
    cv2.rectangle(vis, (x0 + 1, y0 + 1), (x1 - 2, y1 - 2), READ_REGION_BGR, 3)
    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def host_allowed(header, short_name) -> bool:
    """DNS rebinding sends a browser here under the attacker's own name. Only the names this machine is really reached by pass: an IP
    literal, localhost, its short hostname, or that name under a tailnet (<short>.<anything>.ts.net)."""
    if not header:
        return False
    host = header.strip().lower()
    if host.startswith("["):
        host = host[1:].split("]", 1)[0]
    elif host.count(":") == 1:
        host = host.split(":", 1)[0]
    host = host.rstrip(".")
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    if host in ("localhost", short_name):
        return True
    labels = host.split(".")
    return len(labels) >= 4 and labels[0] == short_name and labels[-2:] == ["ts", "net"]


def _rgb(bgr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(bgr[..., ::-1])


def _is_whole(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def validate_control(cmd) -> dict:
    """A dashboard command, checked whole: one bad field rejects all of it, so nothing is half applied."""
    if not isinstance(cmd, dict) or not cmd:
        raise ValueError("expected a JSON object with at least one setting")
    unknown = set(cmd) - CONTROL_KEYS - EXPLAIN_KEYS
    if unknown:
        raise ValueError(f"unknown settings: {', '.join(sorted(unknown))}")
    if set(cmd) & EXPLAIN_KEYS:  # only the audience and which decision: nothing a viewer types can reach the language model
        if set(cmd) - EXPLAIN_KEYS or len({"explain", "explain_stop", "explain_close"} & set(cmd)) != 1:
            raise ValueError("send one explain command on its own")
        if "explain" in cmd and cmd["explain"] not in ep.AUDIENCES:
            raise ValueError(f"explain must be one of: {', '.join(ep.AUDIENCES)}")
        if "explain_decision" in cmd and ("explain" not in cmd or not _is_whole(cmd["explain_decision"])):
            raise ValueError("explain_decision must be a whole number, sent with explain")
        for k in ("explain_stop", "explain_close"):
            if k in cmd and cmd[k] is not True:
                raise ValueError(f"{k} must be true")
    for k in ("model_paused", "feed_paused"):
        if k in cmd and not isinstance(cmd[k], bool):
            raise ValueError(f"{k} must be true or false")
    if "source" in cmd and cmd["source"] not in ("camera", "demo"):
        raise ValueError("source must be camera or demo")
    if "demo_step" in cmd and (not _is_whole(cmd["demo_step"]) or cmd["demo_step"] not in (-1, 1)):
        raise ValueError("demo_step must be -1 or 1")
    if "demo_index" in cmd and not _is_whole(cmd["demo_index"]):
        raise ValueError("demo_index must be a whole number")
    if "demo_step" in cmd and "demo_index" in cmd:
        raise ValueError("send demo_step or demo_index, not both")
    if "demo_interval" in cmd:
        v, (lo, hi) = cmd["demo_interval"], DEMO_INTERVAL_RANGE
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
            raise ValueError(f"demo_interval must be {lo:g} to {hi:g} seconds")
    return cmd


class App:
    def __init__(self, bundle, mode="sim", device=camera.AUTO, threads=4, replay_dir=None, crop="center", interval=1.0,
                 camera_loss="replay", cam_size=(1280, 720), preview_fps=15.0, source=None, demo_dir=None, standby=False, explain=None,
                 idle_standby_s=IDLE_STANDBY_MIN * 60):
        self.bundle, self.threads = Path(bundle), threads
        self.requested_mode, self.mode, self.mode_reason = mode, mode, None
        self.device, self.crop_mode, self.interval = device, crop, interval
        self.camera_loss, self.cam_size, self.preview_fps = camera_loss, cam_size, preview_fps
        self.cam, self._source, self.camera_started = None, source, False
        self.demo_dir = demo_dir
        self.deck, self.deck_error = demo.load_deck(demo_dir)
        self.replay_files = []
        if replay_dir:
            self.replay_files = sorted(p for p in Path(replay_dir).iterdir()
                                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        self.i = 0
        self.reads = 0
        self.host = host_info()
        self.lock = threading.Lock()
        self.jpg = b""
        self._jpg_seq = 0
        self._jpg_cond = threading.Condition(self.lock)
        self.stream_clients = 0
        self.idle_standby_s, self._last_viewer = idle_standby_s, time.monotonic()
        self._closing = threading.Event()
        self._wake = threading.Event()
        self._preview_on = threading.Event()  # set while the live feed runs: the preview thread sleeps on it otherwise
        self._preview_started = False
        self._led_state, self._led_ok, self._led_t = None, False, 0.0
        self._last_key = None
        # Controls. _ctl guards everything a dashboard command can change while the loop is choosing a picture. /state
        # never takes it, so the page stays live while a command is loading the model.
        self._ctl = threading.RLock()
        self.feed_paused = standby
        self._held = None            # (bgr, id): the frozen picture
        self._read_id = None         # the picture last read, so a held picture is read once and not over and over
        self._gen = 0                # bumped by commands: a read that spans a command is dropped, not published
        self._last_read = None
        self.demo_i, self.demo_interval, self._demo_next, self._demo_gen = 0, DEMO_INTERVAL_DEFAULT, 0.0, 0
        self._demo_jpgs = {}
        self._demo_waited = False
        # The decisions an explanation can be about, newest last. Only held decisions live here: no read, no entry, and they go when the
        # picture does. No thread and no process exists for the explainer until someone presses the button.
        self._decisions, self._decision_id = collections.deque(maxlen=RECENT_DECISIONS), 0
        self.explainer = explainer.Explainer(**{"on_change": self._wake.set, **(explain or {})})
        self._vitals, self._vitals_t = {}, 0.0
        self._view = {"status": "idle", "call": None}
        # The model. Nothing here may raise: with a broken bundle the dashboard still has to come up and say so.
        self.model, self.model_error, self.model_started = None, None, False
        self.model_paused = True
        self.pair, self.heads, self.classes, self.thr, self.distributable = "unknown", head_labels(""), [], {}, None
        self._prepare_model()
        if not standby and self._load_model():
            self.model_paused = False

        if mode == "demo" and self.deck is None:  # stay up on the closed camera and say why
            self.mode, self.mode_reason, self.feed_paused = "camera", self.deck_error, True
        if self.mode == "demo":
            self._demo_show()
        self._sync_camera()
        self._show(self._idle_status())
        self._led("idle")

    # -- the model -----------------------------------------------------------------------------------------------
    def _prepare_model(self) -> bool:
        """Read the bundle's config. Failure is reported on the page, never raised."""
        if self.model is not None:
            return True
        try:
            self.model = rt.EdgeModel(self.bundle, threads=self.threads, load=False)
        except Exception as e:  # noqa: BLE001 - the dashboard must outlive any bundle problem
            self._model_failed(e)
            return False
        self.pair = str(self.model.cfg.get("pair", "A+B"))
        self.heads = head_labels(self.pair)
        self.classes, self.thr = self.model.classes, self.model.thr
        self.distributable = self.model.cfg.get("distributable")
        return True

    def _load_model(self) -> bool:
        """True once the session is ready. On failure the model stays off and the page says why."""
        if not self._prepare_model():
            return False
        try:
            self.model.load()
        except Exception as e:  # noqa: BLE001 - includes a missing or broken onnxruntime and running out of memory
            self._model_failed(e)
            return False
        self.model_error, self.model_started = None, True
        return True

    def _model_failed(self, e):
        self.model_error = f"The model did not load: {e}. Fix that, then start the model again."
        print(json.dumps({"event": "model_failed", "error": str(e)}), flush=True)

    # -- sources -------------------------------------------------------------------------------------------------
    def _open_camera(self):
        w, h = self.cam_size
        self.cam = self._source or camera.CameraSource(self.device, w, h)
        if self.camera_loss != "wait" and not self.cam.open_once():
            reason = self.cam.status()["reason"]
            self.cam = None
            self._fallback(reason or f"camera {self.device} not available")
            return
        self.cam.resume()
        self.cam.start()

    def _sync_camera(self):
        """Make the real camera match what is wanted: the device is open only while the live feed is running."""
        if self.mode == "camera" and not self.feed_paused:
            if self.cam is None:
                self._open_camera()
            else:
                self.cam.resume()
                self.cam.start()
            if self.cam is not None:
                self.camera_started = True
                self._preview_on.set()
                if not self._preview_started:
                    self._preview_started = True
                    threading.Thread(target=self._preview_loop, name="preview", daemon=True).start()
        else:
            self._preview_on.clear()
            if self.cam is not None:
                self.cam.pause()

    def _fallback(self, reason):
        self._preview_on.clear()
        if self.cam is not None:
            self.cam.stop()
            self.cam = None
        self.mode = "replay"
        self.mode_reason = f"camera fallback: {reason}; replaying " + \
            (f"{len(self.replay_files)} recorded frames" if self.replay_files else "synthetic frames")
        print(json.dumps({"event": "mode_fallback", "mode": "replay", "reason": self.mode_reason}), flush=True)

    def _demo_show(self):
        with self.lock:
            self._decisions.clear()  # a new slide: a decision about the old one must not look like it is about this one
        if self.demo_i not in self._demo_jpgs:  # enlarged and encoded once per slide, so going round the deck stays cheap
            big = cv2.resize(self.deck.slides[self.demo_i]["bgr"], (DEMO_PREVIEW, DEMO_PREVIEW), interpolation=cv2.INTER_CUBIC)
            self._demo_jpgs[self.demo_i] = preview_jpeg(big, 85)
        self._set_jpg(self._demo_jpgs[self.demo_i])
        self._demo_gen += 1
        self._demo_next = time.monotonic() + self.demo_interval

    def _advance(self):
        """Whatever is due by the clock: the next slide, or giving up on a lost camera under the replay policy."""
        if self.mode == "demo":
            answering = self.explainer.running
            if answering or self._demo_waited:  # the deck waits for an answer, then gives the next slide its full time
                self._demo_next = time.monotonic() + self.demo_interval
                self._demo_waited = answering
            elif not self.feed_paused and time.monotonic() >= self._demo_next:
                self.demo_i = (self.demo_i + 1) % len(self.deck)
                self._demo_show()
                self._show(self._idle_status())
        elif self.mode == "camera" and not self.feed_paused and self.camera_loss != "wait" and self.cam is not None:
            bgr, _, age = self.cam.latest()
            st = self.cam.status()
            if (bgr is None or age > STALE_S) and not st["opening"]:
                self._fallback(st["reason"] or "camera stopped delivering frames")

    def _picture(self):
        """(BGR picture or None, id of this picture or None). No id means a new picture on every step."""
        if self.mode == "demo":
            return self.deck.slides[self.demo_i]["bgr"], ("demo", self._demo_gen)
        if self.feed_paused and self._held is not None:
            return self._held
        if self.mode == "camera":
            if self.feed_paused or self.cam is None:
                return None, None
            bgr, seq, age = self.cam.latest()
            return (bgr, ("camera", seq)) if bgr is not None and age <= STALE_S else (None, None)
        bgr = None
        if self.mode == "replay" and self.replay_files:
            bgr = cv2.imread(str(self.replay_files[self.i % len(self.replay_files)]))
        if bgr is None:
            bgr = synth_frame(self.i)[..., ::-1]
        self._set_jpg(preview_jpeg(bgr, 85))
        if self.feed_paused:
            self._held = (bgr, (self.mode, self.i))
            return self._held
        return bgr, None

    def _set_jpg(self, b: bytes):
        with self._jpg_cond:
            self.jpg = b
            if b:
                self._jpg_seq += 1
                self._jpg_cond.notify_all()

    def _preview_loop(self):
        """Encode the on-screen feed apart from inference, at full rate only while someone is watching."""
        last_seq = -1
        while not self._closing.is_set():
            self._preview_on.wait()
            if self._closing.is_set():
                return
            t = time.perf_counter()
            cam = self.cam
            bgr, seq, _ = cam.latest() if cam is not None else (None, -1, 0)
            if bgr is not None and seq != last_seq:
                last_seq = seq
                jpg = preview_jpeg(bgr)
                with self._ctl:  # checked under the lock: a pause must keep the frozen frame on screen
                    if self._preview_on.is_set():
                        self._set_jpg(jpg)
            rate = self.preview_fps if self.stream_clients else PREVIEW_IDLE_FPS
            self._closing.wait(max(0.0, 1.0 / rate - (time.perf_counter() - t)))

    def _led(self, state) -> bool:
        """Tell the MCU only about changes (a call costs ~8 ms), and retry a failed call every 5 s."""
        now = time.monotonic()
        if state != self._led_state or (not self._led_ok and now - self._led_t > 5.0):
            self._led_ok = bridge.send_status(state)
            self._led_state, self._led_t = state, now
        return self._led_ok

    # -- dashboard commands --------------------------------------------------------------------------------------
    def control(self, cmd) -> dict:
        """Apply one dashboard command. ValueError (nothing changed) if it is malformed or cannot be done right now."""
        cmd = validate_control(cmd)
        if set(cmd) & EXPLAIN_KEYS:  # never takes _ctl: a click must answer even while a model is loading
            self._explain_command(cmd)
            return self.state()
        with self._ctl:
            source = cmd.get("source", self.mode)
            if source == "demo" and self.deck is None:
                self.deck, self.deck_error = demo.load_deck(self.demo_dir)  # they may have been built since the start
                if self.deck is None:
                    raise ValueError(self.deck_error)
            stepping = "demo_step" in cmd or "demo_index" in cmd
            if stepping and source != "demo":
                raise ValueError("switch to demo slides first")
            if "demo_index" in cmd and not 0 <= cmd["demo_index"] < len(self.deck):
                raise ValueError(f"demo_index must be 0 to {len(self.deck) - 1}")
            # Loading the model is the one step that can still fail, so it comes before anything changes.
            start_model = cmd.get("model_paused") is False and self.model_paused
            if start_model:
                self._show("model_loading")
                if not self._load_model():
                    self._show(self._idle_status())
                    raise ValueError(self.model_error)

            changed = False
            if source != self.mode:
                self._switch_source(source, cmd.get("feed_paused", False))
                changed = True
            elif "feed_paused" in cmd and cmd["feed_paused"] != self.feed_paused:
                self._set_feed_paused(cmd["feed_paused"])
                changed = self.mode != "demo"  # pausing slides changes no picture, so the decision on it still stands
            if start_model:
                self.model_paused, changed = False, True
            elif cmd.get("model_paused") is True and not self.model_paused:
                self.model_paused, changed = True, True
                self.model.unload()
            if stepping:
                self.demo_i = cmd.get("demo_index", (self.demo_i + cmd.get("demo_step", 0)) % len(self.deck))
                self._demo_show()
                self.explainer.dismiss()  # the reader moved on by hand. Everything else leaves an answer to finish about its snapshot
                changed = True
            if "demo_interval" in cmd:
                self.demo_interval = float(cmd["demo_interval"])
                self._demo_next = time.monotonic() + self.demo_interval
            if changed:  # no decision may outlive the picture or the model it came from
                self._gen += 1
                self._read_id = None
            if changed or self._view["call"] is None:
                self._show(self._idle_status(), **self._last_mention())
        self._wake.set()
        return self.state()

    def _switch_source(self, source, paused):
        self.mode, self.feed_paused, self._held, self.mode_reason, self._last_read = source, paused, None, None, None
        with self.lock:
            self._decisions.clear()
        if source == "demo":
            self._demo_show()
        else:
            self._set_jpg(b"")
            self.camera_loss = "wait"  # chosen on purpose: from here on, losing it must not switch source by itself
        self._sync_camera()

    def _set_feed_paused(self, paused):
        if paused and self.mode == "camera":
            bgr, seq, age = self.cam.latest() if self.cam is not None else (None, 0, float("inf"))
            if bgr is not None and age <= STALE_S:
                self._held = (bgr, ("camera", seq))
                self._set_jpg(preview_jpeg(bgr))  # the screen shows exactly the frame the model reads
            else:
                self._set_jpg(b"")
        elif not paused:
            self._held = None
            if self.mode == "camera":
                self._set_jpg(b"")
        self.feed_paused = paused
        if self.mode == "demo" and not paused:
            self._demo_next = time.monotonic() + self.demo_interval
        self._sync_camera()

    # -- loop ----------------------------------------------------------------------------------------------------
    def _idle_status(self) -> str:
        """The status while there is no decision to show. "reading" means the loop has something to read."""
        if self.model_paused and self.model_error:
            return "model_error"
        if self.mode == "camera" and not self.feed_paused:
            bgr, _, age = self.cam.latest() if self.cam is not None else (None, 0, float("inf"))
            if bgr is None or age > STALE_S:
                return "camera_starting" if self.cam is not None and self.cam.status()["opening"] else "no_camera"
        if self.model_paused:
            return "standby" if self.feed_paused else "model_paused"
        if self.mode == "camera" and self.feed_paused and self._held is None:
            return "camera_off"
        return "reading"

    def _last_mention(self) -> dict:
        return {"last": self._last_read} if self.model_paused and self._last_read else {}

    def _show(self, status, **decision):
        """Replace what the page shows. There is no way to keep an old decision by accident: a call must be passed in."""
        with self.lock:
            self._view = {"status": status, "call": None, **decision}

    # -- explaining a decision ---------------------------------------------------------------------------------------------
    def _remember_decision(self, status, call, reasons, tissue_fraction, tile, mode):
        """Called under self.lock with the decision being published. The frozen copy of the picture is only encoded on a click."""
        self._decision_id += 1
        where = f"Demo slide {self.demo_i + 1} of {len(self.deck)}" if mode == "demo" else EXPLAIN_WHERE.get(self._frame_label(), self._frame_label())
        self._decisions.append({"id": self._decision_id, "tile": tile, "where": where, "t": time.monotonic(),
                                "snapshot": {"status": status, "call": call, "reasons": list(reasons), "tissue_fraction": tissue_fraction}})

    def _explain_target(self, view):
        """(the decision the button would explain, whether it is the current or the last one, why not if there is none). No lock taken."""
        latest = self._decisions[-1] if self._decisions else None
        status = view["status"]
        if latest is not None and status in DECIDED and view.get("call") is not None:
            return latest, "current", None
        if latest is not None and self.model_paused:  # the model is off, and the app still holds what it last said about this picture
            return latest, "last", None
        if status in ("reading", "model_loading"):
            why = "The model is reading this picture. Explain opens when it has a result."
        elif status == "model_error":
            why = "The model did not load, so there is nothing to explain."
        elif status == "error":
            why = "This picture could not be read, so there is nothing to explain."
        elif self.model_paused:
            why = "Nothing to explain yet. Start the model and give it a picture."
        else:
            why = "The model has no picture to read, so there is nothing to explain."
        return None, None, why

    def _explain_command(self, cmd):
        print(json.dumps({"event": "explain_command", **cmd}), flush=True)
        if cmd.get("explain_stop"):
            self.explainer.stop()
            return
        if cmd.get("explain_close"):
            self.explainer.dismiss()
            return
        if self.explainer.running:  # a click on a job already running changes nothing, whatever it was looking at
            return
        with self.lock:
            target, _, why = self._explain_target(self._view)
            wanted = cmd.get("explain_decision")
            if wanted is not None:  # what the viewer saw when they pressed, and only that: never a newer one they did not see
                target, why = next((d for d in self._decisions if d["id"] == wanted), None), RESULT_GONE
        if target is None:
            raise ValueError(why)
        ok, buf = cv2.imencode(".jpg", _rgb(target["tile"]), [cv2.IMWRITE_JPEG_QUALITY, 80])
        call = target["snapshot"]["call"]
        about = {"decision": target["id"], "status": target["snapshot"]["status"], "label": call["label"], "prob": call["prob"],
                 "where": target["where"], "t": target["t"]}
        self.explainer.start(target["snapshot"], cmd["explain"], about=about, picture=buf.tobytes() if ok else b"")
        self._wake.set()

    def _explain_state(self, view) -> dict:
        with self.lock:
            target, kind, why = self._explain_target(view)
            newest = self._decisions[-1]["id"] if self._decisions else None
        x = self.explainer.state()
        job = x["job"]
        if job is not None:  # the page says when the view has moved on from what this explanation is about
            job["current"] = newest is not None and newest == job["about"]["decision"]
            job["about"] = {k: v for k, v in job["about"].items() if k != "t"} | {"age_s": int(time.monotonic() - job["about"]["t"])}
        running = job is not None and job["phase"] in explainer.ACTIVE
        x.update(available=target is not None and not running, target=kind, decision=target["id"] if target else None,
                 why_not="An answer is being written. Stop it to start another." if running else why,
                 hint=self.explainer.hint(self._vitals.get("temp_c")))
        return x

    # -- nobody watching --------------------------------------------------------------------------------------------------
    def viewer_seen(self):
        self._last_viewer = time.monotonic()

    def _idle_standby(self) -> bool:
        """With no /state poll, command or open video for a while, go to plain standby, as at boot: the owner's rule is model unloaded and
        camera off whenever nobody needs them. A kiosk page polls, so it counts as a viewer. Nothing that was showing is kept."""
        if not self.idle_standby_s:
            return False
        now = time.monotonic()
        if self.stream_clients:
            self._last_viewer = now
            return False
        idle = now - self._last_viewer
        if idle < self.idle_standby_s:
            return False
        with self._ctl:
            if self.feed_paused and self.model_paused and not self.explainer.running:
                return False
            was = {"mode": self.mode, "feed_paused": self.feed_paused, "model_paused": self.model_paused, "explaining": self.explainer.running}
            self.explainer.dismiss()
            if self.mode == "demo":
                self._switch_source("camera", True)
            elif not self.feed_paused:
                self._set_feed_paused(True)
            self._held, self._last_read, self._read_id = None, None, None
            self._set_jpg(b"")
            self.model_paused = True
            if self.model is not None:
                self.model.unload()
            with self.lock:
                self._decisions.clear()
            self._gen += 1
            self._show(self._idle_status())
        self._led("idle")
        print(json.dumps({"event": "idle_standby", "no_viewer_for_s": round(idle), "was": was}), flush=True)
        return True

    def _log_once(self, key, **fields):
        if key != self._last_key:
            print(json.dumps(fields), flush=True)
        self._last_key = key

    def step(self):
        try:
            with self._ctl:
                self._advance()
                bgr, pic = self._picture()
                status = self._idle_status()
                gen, mode = self._gen, self.mode
                truth = self.deck.slides[self.demo_i]["truth"] if mode == "demo" else None
                if status != "reading":
                    if status == "no_camera":
                        self._set_jpg(b"")  # a stale frame must not stay on screen as if it were current
                    self._show(status, **self._last_mention())
                elif bgr is None or (pic is not None and pic == self._read_id):
                    return  # a frame arrived just after the picture was picked, or: same picture, same decision
            if status != "reading":
                self._led("idle")
                reason = self.cam.status()["reason"] if status == "no_camera" and self.cam is not None else None
                self._log_once((status, reason), event=status, reason=reason)
                return
            if mode not in ("camera", "demo"):
                self._led("busy")
            t0 = time.perf_counter()
            tile = rt.crop_from_frame(_rgb(bgr), self.crop_mode)
            r = self.model.infer(tile[None])[0]
            ms = (time.perf_counter() - t0) * 1000
            decided = "no_tissue" if r.no_tissue else ("review" if r.review else "agree")
            if gen != self._gen:  # a command arrived during the read: this decision is about a picture no longer current
                return
            led = self._led(decided)
            tf = rt.tissue_fraction(tile)
            h, w = bgr.shape[:2]
            x0, y0, x1, y1 = rt.center_box(h, w)
            call = {"label": r.label, "prob": r.prob, "probsA": r.probsA, "probsB": r.probsB, "probsF": r.probsF,
                    "labelA": self.model.classes[int(np.argmax(r.probsA))],
                    "labelB": self.model.classes[int(np.argmax(r.probsB))], "review_score": r.review_score}
            extra = {"truth": truth, "truth_match": r.label == truth} if truth else {}
            with self._ctl:
                if gen != self._gen:
                    return
                self.reads += 1
                self._read_id = pic
                self._last_read = {"status": decided, "label": r.label, "prob": r.prob}
                with self.lock:
                    self._remember_decision(decided, call, r.reasons, tf, tile, mode)
                self._show(decided, call=call, reasons=r.reasons, tissue_fraction=tf, latency_ms=ms,
                           crop_box=[x0 / w, y0 / h, x1 / w, y1 / h], n=self.i, **extra)
            key = (decided, r.label)
            if mode != "camera" or key != self._last_key or self.i % LOG_EVERY == 0:
                print(json.dumps({"event": "tile", "n": self.i, "mode": mode, "status": decided, "label": r.label,
                                  "prob": round(r.prob, 4), "reasons": r.reasons, "tissue_fraction": round(tf, 3),
                                  "latency_ms": round(ms, 1), "led": led, **extra}), flush=True)
            self._last_key = key
        except rt.ModelNotLoaded:  # paused between the check and the read: the next step shows that
            pass
        except Exception as e:  # noqa: BLE001 - keep serving the UI
            self._led("error")
            self._show("error", error=str(e))
            print(json.dumps({"event": "error", "error": str(e)}), flush=True)
        finally:
            self.i += 1

    def _next_wait(self) -> float:
        """How long the loop may sleep. Nothing is due while a picture is held, so a paused board costs about 0% CPU, and
        slides wait for their own time. A command wakes the loop at once."""
        with self._ctl:
            if self.feed_paused:
                return IDLE_TICK_S
            if self.mode == "demo":
                return min(IDLE_TICK_S, max(0.0, self._demo_next - time.monotonic()))
            return self.interval

    def run_forever(self, stop: threading.Event):
        while not stop.is_set():
            t = time.perf_counter()
            try:
                self._idle_standby()
            except Exception as e:  # noqa: BLE001 - keep serving the UI
                print(json.dumps({"event": "idle_standby_failed", "error": repr(e)}), flush=True)
            self.step()
            self._wake.wait(max(0.0, self._next_wait() - (time.perf_counter() - t)))
            self._wake.clear()

    # -- what the page sees ----------------------------------------------------------------------------------------
    def _demo_state(self):
        if self.deck is None:
            return None
        slide = self.deck.slides[self.demo_i]
        return {"index": self.demo_i, "total": len(self.deck), "interval": self.demo_interval, "file": slide["file"],
                "truth": slide["truth"], "running": self.mode == "demo" and not self.feed_paused,
                "held_for_explain": self.mode == "demo" and not self.feed_paused and self.explainer.running,
                "truths": [s["truth"] for s in self.deck.slides]}

    def _frame_label(self) -> str:
        if self.mode == "camera" and self.feed_paused:
            return "Frozen frame" if self._held is not None else "Camera off"
        return FRAME_LABELS.get(self.mode, "Frame")

    def state(self) -> dict:
        """Read fresh on every request, so a board nobody is watching does no work for the page."""
        with self.lock:
            st = dict(self._view)
            has_picture = bool(self.jpg)
        now = time.monotonic()
        if now - self._vitals_t >= VITALS_EVERY_S:
            self._vitals, self._vitals_t = system_vitals(), now
        source = self.cam  # a command may clear it at any moment
        cam = source.status() if source is not None else dict(CAMERA_NEVER_OPENED)
        st.update(mode=self.mode, mode_reason=self.mode_reason, requested_mode=self.requested_mode,
                  source_note=SOURCE_NOTES.get(self.mode, ""), pair=self.pair, heads=self.heads, classes=self.classes,
                  thresholds=self.thr, distributable=self.distributable, host=self.host, camera=cam,
                  host_label=("" if self.host["is_uno_q"] else
                              f"host: {self.host['machine']}, not the UNO Q (latency is this host's)"),
                  frame_label=self._frame_label(), has_picture=has_picture, model_error=self.model_error,
                  led=self._led_ok, led_state=self._led_state,
                  controls={"source": self.mode, "model_paused": self.model_paused, "feed_paused": self.feed_paused,
                            "model_loaded": self.model is not None and self.model.loaded, "camera_open": cam["open"],
                            "model_started": self.model_started, "camera_started": self.camera_started,
                            "demo_available": self.deck is not None, "demo_error": self.deck_error},
                  demo=self._demo_state(), system=self._vitals, reads=self.reads)
        st["explain"] = self._explain_state(st)
        return st

    def frame_jpg(self) -> bytes:
        with self.lock:
            return self.jpg

    def wait_jpg(self, after: int, timeout: float = STREAM_RESEND_S):
        """Next frame newer than sequence number `after`: (jpg, seq), or (None, after) on timeout."""
        with self._jpg_cond:
            self._jpg_cond.wait_for(lambda: self._jpg_seq != after and self.jpg, timeout)
            if self._jpg_seq != after and self.jpg:
                return self.jpg, self._jpg_seq
            return None, after

    def _clients(self, delta: int):
        with self.lock:
            self.stream_clients += delta

    def stream_slot(self) -> bool:
        with self.lock:
            if self.stream_clients >= MAX_STREAMS:
                return False
            self.stream_clients += 1
            return True

    def close(self):
        self._closing.set()
        self._wake.set()
        self._preview_on.set()
        self.explainer.shutdown()
        if self.cam is not None:
            self.cam.stop()


class _Server(ThreadingHTTPServer):
    """One thread a connection, up to a limit: past it a new connection is closed at once, and the page's next poll tries again."""

    def __init__(self, *a, **kw):
        self._slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        super().__init__(*a, **kw)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        super().process_request(request, client_address)

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def make_server(app: App, port: int) -> ThreadingHTTPServer:
    short_name = platform.node().split(".")[0].lower()

    class H(BaseHTTPRequestHandler):
        timeout = HANDLER_TIMEOUT_S  # on the socket: a stalled body ends here, and so does video to a viewer who stopped taking it

        def log_message(self, *a):
            pass

        def _refused(self) -> bool:
            if host_allowed(self.headers.get("Host"), short_name):
                return False
            self._send(403, b"this host name is not allowed", "text/plain")
            return True

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _stream(self):
            if not app.stream_slot():
                return self._send(503, b"too many viewers", "text/plain")
            seq = -1
            part = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            try:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(part)
                while not app._closing.is_set():
                    jpg, seq = app.wait_jpg(seq)
                    if jpg is None:  # nothing new (a held frame or slide): resend, which also notices a closed browser
                        jpg = app.frame_jpg()
                        if not jpg:
                            continue
                    # The boundary and the next part's headers go right after the image: Chromium only paints a part once
                    # the next part's headers have arrived, and a held frame may have no successor for seconds. So there is
                    # no Content-Length (the next image's size is not known yet); parsers find the end by the boundary.
                    self.wfile.write(jpg + b"\r\n" + part)
                    self.wfile.flush()
            except OSError:  # the browser went away
                pass
            finally:
                app._clients(-1)

        def do_GET(self):
            if self._refused():
                return
            url = urlsplit(self.path)
            path = url.path
            if path == "/state":
                app.viewer_seen()
                return self._json(200, app.state())
            if path == "/frame.jpg":
                b = app.frame_jpg()
                return self._send(200 if b else 404, b or b"no picture yet", "image/jpeg" if b else "text/plain")
            if path == "/stream.mjpg":
                return self._stream()
            if path == "/explain/picture.jpg":
                wanted = parse_qs(url.query).get("job", [None])[0]  # a page still on an older job must not be shown a newer job's tile
                b = app.explainer.picture(int(wanted) if wanted and wanted.isdigit() else None)
                return self._send(200 if b else 404, b or b"no explanation open", "image/jpeg" if b else "text/plain")
            m = re.fullmatch(r"/slide/(\d{1,4})\.jpg", path)
            if m:
                if app.deck is None or int(m[1]) >= len(app.deck):
                    return self._send(404, b"no such slide", "text/plain")
                return self._send(200, app.deck.thumb(int(m[1])), "image/jpeg")
            name = "index.html" if path in ("/", "") else path.lstrip("/")
            f = (UI_DIR / name).resolve()
            if UI_DIR.resolve() in f.parents and f.is_file():
                ctype = "font/woff2" if f.suffix == ".woff2" else mimetypes.guess_type(str(f))[0]
                return self._send(200, f.read_bytes(), ctype or "application/octet-stream")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self._refused():
                return
            if self.path.split("?")[0] != "/control":
                return self._send(404, b"not found", "text/plain")
            app.viewer_seen()
            # A web page elsewhere can only send JSON here after a preflight, which this server does not answer, so
            # requiring JSON stops another site from driving the dashboard through a viewer's browser.
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._json(415, {"error": "send application/json"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = 0
            if not 0 < n <= MAX_CONTROL_BYTES:
                return self._json(400, {"error": f"body must be 1 to {MAX_CONTROL_BYTES} bytes"})
            try:
                body = self.rfile.read(n)
            except OSError:  # a body that never finished: the timeout ends it here and nothing was applied
                self.close_connection = True
                return None
            try:
                return self._json(200, app.control(json.loads(body)))
            except ValueError as e:  # includes malformed JSON
                return self._json(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001 - the page should say what failed, not lose its connection
                return self._json(500, {"error": f"That command failed: {e}"})

    return _Server((HOST, port), H)


def _terminate(*_):
    raise KeyboardInterrupt  # run main()'s cleanup when systemd stops the service


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--mode", choices=["camera", "demo", "replay", "sim"], default="camera")
    ap.add_argument("--standby", action="store_true", help="start with the model unloaded and the camera closed")
    ap.add_argument("--device", default=camera.AUTO, help="auto = first USB camera, or a path such as /dev/video2")
    ap.add_argument("--camera-loss", choices=["replay", "wait"], default="replay")
    ap.add_argument("--demo-dir", default=str(DEFAULT_DEMO_DIR))
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--replay-dir", default=None)
    ap.add_argument("--crop", choices=["center", "marker"], default="center")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--idle-standby-min", type=float, default=IDLE_STANDBY_MIN,
                    help="minutes with no viewer before the model and the camera go off; 0 never")
    ap.add_argument("--explain-binary", default=str(explainer.DEFAULT_BINARY), help="llama-server for the Explain button")
    ap.add_argument("--explain-model", default=str(explainer.DEFAULT_MODEL), help="the explainer's GGUF")
    ap.add_argument("--explain-port", type=int, default=explainer.DEFAULT_PORT, help="loopback port for llama-server")
    ap.add_argument("--explain-threads", type=int, default=explainer.DEFAULT_THREADS)
    ap.add_argument("--explain-idle", type=float, default=explainer.DEFAULT_IDLE_S,
                    help="seconds after the last answer before llama-server exits and a finished panel closes")
    ap.add_argument("--explain-start-max-c", type=float, default=explainer.START_MAX_C, help="no answer starts above this temperature")
    ap.add_argument("--explain-stop-max-c", type=float, default=explainer.STOP_MAX_C, help="an answer stops above this temperature")
    a = ap.parse_args(argv)
    signal.signal(signal.SIGTERM, _terminate)
    app = App(a.bundle, a.mode, a.device, a.threads, a.replay_dir, a.crop, a.interval, a.camera_loss,
              (a.cam_width, a.cam_height), demo_dir=a.demo_dir, standby=a.standby, idle_standby_s=a.idle_standby_min * 60,
              explain={"binary": a.explain_binary, "model": a.explain_model, "port": a.explain_port, "threads": a.explain_threads,
                       "idle_s": a.explain_idle, "start_max_c": a.explain_start_max_c, "stop_max_c": a.explain_stop_max_c})
    srv = make_server(app, a.port)
    stop = threading.Event()
    threading.Thread(target=app.run_forever, args=(stop,), daemon=True).start()
    print(json.dumps({"event": "serving", "url": f"http://{HOST}:{a.port}/", "mode": app.mode, "standby": a.standby,
                      "reason": app.mode_reason, "model_error": app.model_error,
                      "demo_slides": len(app.deck) if app.deck else 0}), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        app.close()
        srv.server_close()
        bridge.send_status("idle")


if __name__ == "__main__":
    main()
