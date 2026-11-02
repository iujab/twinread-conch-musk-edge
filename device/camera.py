"""USB camera source for the device app.

Finds the UVC camera by itself, keeps only the newest frame so a decision is never about a stale picture, and survives
unplug and replug. The capture thread dequeues every frame (grab) but decodes only at decode_hz (retrieve), which keeps the
V4L2 queue empty without paying for a JPEG decode of frames nobody will look at.

pause() really closes the V4L2 device: the sensor goes off, USB stops polling, and the capture thread sleeps on an event
until resume() reopens it. Only the capture thread touches an open device, so it is never released in the middle of a read.
"""
from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import cv2

SYSFS_V4L = Path("/sys/class/video4linux")
AUTO = "auto"


def find_usb_cameras(sysfs=SYSFS_V4L) -> list:
    """UVC capture nodes, lowest number first. The UNO Q's /dev/video0,1 are Qualcomm Venus codecs, not cameras, and every
    UVC camera also exposes a metadata node (index 1) that never delivers frames, so both are skipped."""
    found = []
    try:
        nodes = list(Path(sysfs).iterdir())
    except OSError:
        return found
    for node in nodes:
        m = re.fullmatch(r"video(\d+)", node.name)
        if not m:
            continue
        try:
            driver = (node / "device" / "driver").resolve().name
            index = int((node / "index").read_text().strip())
            name = (node / "name").read_text().strip()
        except (OSError, ValueError):
            continue
        if driver == "uvcvideo" and index == 0:
            found.append({"path": f"/dev/{node.name}", "name": name, "number": int(m.group(1))})
    return sorted(found, key=lambda c: c["number"])


def open_v4l2(path, width, height, fps):
    """Open a V4L2 device asking for MJPG: USB 2 webcams only reach 720p at full frame rate in MJPG. The camera may pick a
    nearby mode; CameraSource reports what it actually delivers."""
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class CameraSource:
    def __init__(self, device=AUTO, width=1280, height=720, fps=30, decode_hz=15.0, retry_s=1.0,
                 finder=find_usb_cameras, opener=open_v4l2):
        self.device, self.width, self.height, self.fps = device, width, height, fps
        self.decode_hz, self.retry_s = decode_hz, retry_s
        self._finder, self._opener = finder, opener
        self._lock = threading.Lock()
        self._cap = None
        self._frame = None
        self._seq = 0
        self._t = 0.0
        self._status = {"connected": False, "device": None, "name": None, "width": None, "height": None, "fps": None,
                        "reason": "camera not opened yet"}
        self._paused = False
        self._opening = False            # the first attempt since start() or resume() has not finished yet
        self._stop = threading.Event()
        self._run_ok = threading.Event()  # cleared while paused: the capture thread sleeps on this with the device closed
        self._run_ok.set()
        self._parked = threading.Event()  # set by the capture thread once it has closed the device and gone to sleep
        self._retry = threading.Event()   # ends the wait between attempts to find a camera
        self._thread = None

    # -- opening ---------------------------------------------------------------------------------------------------
    def open_once(self) -> bool:
        """Try every candidate once. A camera counts as open only after it has delivered a real frame."""
        try:
            return self._open_first_that_works()
        except Exception as e:  # noqa: BLE001 - a broken camera stack is reported on the page; it must not end the thread
            self._mark_lost(f"could not look for a camera: {e}")
            return False
        finally:
            with self._lock:
                self._opening = False

    def _open_first_that_works(self) -> bool:
        cams = self._finder() if self.device == AUTO else [{"path": self.device, "name": str(self.device)}]
        if not cams:
            self._mark_lost("no USB camera found")
            return False
        reason = None
        for c in cams:
            cap, ok, bgr = None, False, None
            try:
                cap = self._opener(c["path"], self.width, self.height, self.fps)
                if cap is not None and cap.isOpened():
                    ok, bgr = cap.read()
            except Exception as e:  # noqa: BLE001 - a bad device must not stop the search
                reason = f"{c['path']}: {e}"
            if ok and bgr is not None:
                h, w = bgr.shape[:2]
                with self._lock:
                    self._cap = cap
                    self._publish_locked(bgr)
                    self._status = {"connected": True, "device": c["path"], "name": c["name"], "width": int(w),
                                    "height": int(h), "fps": None, "reason": None}
                return True
            if cap is not None:
                cap.release()
            reason = reason or f"{c['path']} delivered no frame"
        self._mark_lost(reason)
        return False

    def _publish_locked(self, bgr):
        self._frame = bgr
        self._seq += 1
        self._t = time.monotonic()

    def _mark_lost(self, reason):
        with self._lock:
            cap, self._cap, self._frame = self._cap, None, None
            self._status = {**self._status, "connected": False, "fps": None, "reason": reason}
        if cap is not None:
            cap.release()

    # -- capture thread --------------------------------------------------------------------------------------------
    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            with self._lock:
                self._opening = self._cap is None and not self._paused
            self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._run_ok.set()
        self._retry.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._mark_lost("camera stopped")

    def pause(self, timeout: float = 3.0) -> bool:
        """Close the device and put the capture thread to sleep. True once it is closed; False if the thread is stuck in
        a blocked read, in which case it closes the device as soon as that read returns."""
        with self._lock:
            self._paused, self._opening = True, False
        self._run_ok.clear()
        self._retry.set()
        if self._thread is None or not self._thread.is_alive():
            self._mark_lost("camera paused")
            return True
        return self._parked.wait(timeout)

    def resume(self):
        """Reopen the device, from the capture thread. status() reports opening until the first attempt has finished."""
        with self._lock:
            if not self._paused:
                return
            self._paused, self._opening = False, True
            self._status = {**self._status, "reason": None}
        self._retry.clear()
        self._run_ok.set()

    def _run(self):
        while not self._stop.is_set():
            try:
                self._capture()
            except Exception as e:  # noqa: BLE001 - say so and keep trying: only stop() ends this thread
                self._mark_lost(f"camera error: {e}")
                self._retry.wait(self.retry_s)
                self._retry.clear()

    def _capture(self):
        last_decode, grabbed, t_fps = 0.0, 0, time.monotonic()
        min_gap = 1.0 / self.decode_hz
        while not self._stop.is_set():
            if not self._run_ok.is_set():
                self._mark_lost("camera paused")
                self._parked.set()
                self._run_ok.wait()
                self._parked.clear()
                continue
            with self._lock:
                cap = self._cap
            if cap is None:
                if not self.open_once():
                    self._retry.wait(self.retry_s)
                    self._retry.clear()
                grabbed, t_fps = 0, time.monotonic()
                continue
            try:
                ok = bool(cap.grab())
            except Exception:  # noqa: BLE001 - treated as an unplugged camera
                ok = False
            if not ok:
                self._mark_lost("camera stopped delivering frames")
                continue
            grabbed += 1
            now = time.monotonic()
            if now - t_fps >= 2.0:
                fps = round(grabbed / (now - t_fps), 1)
                grabbed, t_fps = 0, now
                with self._lock:
                    self._status = {**self._status, "fps": fps}
            if now - last_decode < min_gap:
                continue
            last_decode = now
            try:
                ok, bgr = cap.retrieve()
            except Exception:  # noqa: BLE001
                ok, bgr = False, None
            if not ok or bgr is None:
                self._mark_lost("camera delivered an unreadable frame")
                continue
            with self._lock:
                self._publish_locked(bgr)

    # -- readers ---------------------------------------------------------------------------------------------------
    def latest(self):
        """(BGR frame or None, sequence number, age in seconds). Consumers must not write into the frame."""
        with self._lock:
            return self._frame, self._seq, (time.monotonic() - self._t if self._frame is not None else float("inf"))

    def status(self) -> dict:
        with self._lock:
            return {**self._status, "open": self._cap is not None, "paused": self._paused, "opening": self._opening}
