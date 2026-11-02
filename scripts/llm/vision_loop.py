#!/usr/bin/env python3
"""Keep the TwinRead vision model loaded next to the LLM: either idle, or reading one stored image in a loop at N Hz.

Runs under device/.venv/bin/python. Uses EdgeModel on the bundle exactly as the app's read path does (crop_from_frame then
infer). Never opens a camera and never starts the service. Appends one JSON line per read, stamped with CLOCK_MONOTONIC so
bench_on_board.py can split read latency by what the LLM was doing at that moment.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "device"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(REPO / "bundles/cm"))
    ap.add_argument("--image", default=str(REPO / "data/replay_crc/crc_test_2621_TUM.png"))
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--hz", type=float, default=2.0)
    ap.add_argument("--mode", choices=["idle", "loop"], default="loop")
    ap.add_argument("--log", required=True)
    a = ap.parse_args()

    stop = {"now": False}

    def on_signal(*_):
        stop["now"] = True
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, on_signal)

    import numpy as np
    from PIL import Image

    import twinread_runtime as rt

    t0 = time.monotonic()
    model = rt.EdgeModel(a.bundle, threads=a.threads)
    frame = np.asarray(Image.open(a.image).convert("RGB"))
    with open(a.log, "a", buffering=1) as log:
        log.write(json.dumps({"event": "ready", "t": time.monotonic(), "load_s": time.monotonic() - t0,
                              "frame_shape": list(frame.shape), "threads": a.threads, "mode": a.mode}) + "\n")
        if a.mode == "idle":
            while not stop["now"]:
                time.sleep(0.2)
            return
        period = 1.0 / a.hz
        next_t = time.monotonic()
        n = 0
        while not stop["now"]:
            t1 = time.monotonic()
            r = model.infer(rt.crop_from_frame(frame)[None])[0]
            t2 = time.monotonic()
            n += 1
            log.write(json.dumps({"t": t1, "ms": (t2 - t1) * 1000, "label": r.label, "late_ms": (t1 - next_t) * 1000}) + "\n")
            next_t += period
            wait = next_t - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            else:
                next_t = time.monotonic()  # fell behind: do not try to catch up, as the app's own loop does not


if __name__ == "__main__":
    main()
