#!/usr/bin/env python
"""Evidence that the TwinRead kiosk really runs on this board: session, camera, display, decisions, resources.

Run on the board with the device venv: device/.venv/bin/python scripts/uno_q/kiosk_check.py --out results/uno_q/onboard/kiosk
Writes check.json, what the camera sees (camera_frame.jpg) and what X is displaying (screen.png). It only looks: in standby it
says so and turns nothing on, so start the camera and the model from the dashboard first if you want them checked.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "device"))
import camera  # noqa: E402

URL = "http://127.0.0.1:8080"


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()


def get(path: str, timeout=3.0) -> bytes:
    return urllib.request.urlopen(URL + path, timeout=timeout).read()


def session() -> dict:
    out = {}
    for sid in sh("loginctl list-sessions --no-legend | awk '$3==\"arduino\"{print $1}'").split():
        d = dict(line.split("=", 1) for line in sh(f"loginctl show-session {sid} -p Type -p Class -p Desktop -p Service -p State").splitlines())
        if d.get("Class") == "user" and d.get("Type") == "x11":
            out = d
    return out


def display() -> dict:
    conns = {p.name: p.joinpath("status").read_text().strip() for p in Path("/sys/class/drm").glob("card*-*")}
    env = dict(os.environ, DISPLAY=":0", XAUTHORITY=str(Path.home() / ".Xauthority"))
    try:
        xr = subprocess.run(["xrandr", "--current"], capture_output=True, text=True, env=env, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        xr = ""
    return {"connectors": conns, "xrandr": xr.strip().splitlines()[:4], "chromium_procs": int(sh("pgrep -c chromium") or 0)}


def screenshot(path: Path) -> dict:
    os.environ.update(DISPLAY=":0", XAUTHORITY=str(Path.home() / ".Xauthority"))
    from PIL import ImageGrab
    img = ImageGrab.grab(xdisplay=":0").convert("RGB")
    img.save(path)
    a = np.asarray(img)
    return {"file": path.name, "width": img.width, "height": img.height, "mean": round(float(a.mean()), 1), "std": round(float(a.std()), 1)}


def camera_check(out: Path, seconds: float) -> dict:
    """Two frames apart in time. A live sensor always differs (noise, at least); a frozen or fake source does not."""
    res = {"usb_cameras": camera.find_usb_cameras()}
    a = cv2.imdecode(np.frombuffer(get("/frame.jpg"), np.uint8), cv2.IMREAD_COLOR)
    time.sleep(seconds)
    raw = get("/frame.jpg")
    b = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    (out / "camera_frame.jpg").write_bytes(raw)
    res.update(frame_shape=list(b.shape), frame_mean=round(float(b.mean()), 1), frame_std=round(float(b.std()), 1),
               mean_abs_diff_between_frames=round(float(np.abs(a.astype(np.int16) - b).mean()), 3))
    return res


def stream_rate(seconds=5.0) -> dict:
    """Frames per second a viewer of /stream.mjpg really gets."""
    r = urllib.request.urlopen(URL + "/stream.mjpg", timeout=5)
    t0, n, first = time.perf_counter(), 0, None
    while time.perf_counter() - t0 < seconds:
        if b"--frame" in r.read(4096):
            n += 1
            first = first or time.perf_counter() - t0
    r.close()
    return {"jpeg_frames": n, "fps": round(n / seconds, 1), "first_frame_s": round(first or -1, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/uno_q/onboard/kiosk"))
    ap.add_argument("--seconds", type=float, default=30.0, help="how long to sample decisions")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"research_prototype": True, "machine": platform.machine(), "simulated": False,
           "dt_model": Path("/proc/device-tree/model").read_text(errors="replace").strip("\x00\n "),
           "uptime_s": float(Path("/proc/uptime").read_text().split()[0]),
           "wifi": {"radio": sh("nmcli radio wifi"), "state": sh("nmcli -g GENERAL.STATE dev show wlan0")},
           "service": dict(line.split("=", 1) for line in sh("systemctl show twinread -p ActiveState -p NRestarts -p IPAddressDeny").splitlines()),
           "session": session(), "display": display(), "usb": sh("lsusb").splitlines()}
    st = json.loads(get("/state"))
    rep["camera_state"] = st.get("camera")
    rep["controls"] = st.get("controls")
    if st["controls"]["camera_open"]:
        rep["camera_check"] = camera_check(out, 2.0)
        rep["stream"] = stream_rate()
    try:
        rep["screen"] = screenshot(out / "screen.png")
    except Exception as e:  # noqa: BLE001 - a missing screen is a finding, not a crash
        rep["screen"] = {"error": str(e)}
    samples, t0 = [], time.time()
    while time.time() - t0 < a.seconds:
        s = json.loads(get("/state"))
        samples.append({"status": s["status"], "label": (s.get("call") or {}).get("label"), "n": s.get("reads"),
                        "latency_ms": s.get("latency_ms"), "led": s.get("led"), "fps": (s.get("camera") or {}).get("fps")})
        time.sleep(0.5)
    counts = {}
    for s in samples:
        k = f"{s['status']}:{s['label']}"
        counts[k] = counts.get(k, 0) + 1
    lat = [s["latency_ms"] for s in samples if s["latency_ms"]]
    fps = [s["fps"] for s in samples if s["fps"]]
    ns = [s["n"] for s in samples if s["n"] is not None]
    rep["decisions"] = {"seconds": a.seconds, "samples": len(samples), "counts": counts,
                        "reads_done": (ns[-1] - ns[0]) if ns else 0, "led_true": sum(1 for s in samples if s["led"]),
                        "latency_ms_p50": round(float(np.median(lat)), 1) if lat else None,
                        "camera_fps_median": round(float(np.median(fps)), 1) if fps else None}
    top = sh("top -bn2 -d 2 | awk '/^top -/{n++} n==2' | sed -n '1,14p'").splitlines()
    rep["resources"] = {"load": Path("/proc/loadavg").read_text().strip(), "mem": sh("free -m | sed -n 2p"), "top": top,
                        "app_rss_mb": round(int(sh("ps -o rss= -p $(systemctl show twinread -p MainPID --value)") or 0) / 1024, 1),
                        "temps_c": sorted({round(int(p.read_text()) / 1000, 1) for p in Path("/sys/class/thermal").glob("thermal_zone*/temp")}, reverse=True)[:3]}
    (out / "check.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep.get(k) for k in ("uptime_s", "wifi", "service", "session", "display", "camera_state", "camera_check", "stream", "screen", "decisions")}, indent=1))
    print("\n".join(top[:12]))
    print("rss", rep["resources"]["app_rss_mb"], "temps", rep["resources"]["temps_c"], "load", rep["resources"]["load"])


if __name__ == "__main__":
    main()
