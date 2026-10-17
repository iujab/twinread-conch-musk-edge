#!/usr/bin/env python
"""Record timestamped JPEG frames from a USB camera (MJPEG) on the UNO Q. Only cv2 + numpy + stdlib.

usage: record_frames.py --device /dev/video0 --seconds 60 --out frames/ [--fps 5] [--width 1280 --height 720]
Writes frame_<unix_ms>.jpg and index.csv (file,t_unix,t_rel_s). Exit 2 if the camera is missing/unopenable,
3 if it drops out mid-run (frames saved so far are kept).
"""
import argparse
import os
import sys
import time

import cv2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--out", default="frames")
    ap.add_argument("--fps", type=float, default=5.0, help="saved frames per second")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--quality", type=int, default=92)
    a = ap.parse_args(argv)
    dev = int(a.device) if a.device.isdigit() else a.device
    if isinstance(dev, str) and not os.path.exists(dev):
        print(f"error: camera device {a.device} not found", file=sys.stderr)
        return 2
    cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        print(f"error: camera {a.device} could not be opened", file=sys.stderr)
        return 2
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    last = 0.0
    n = fails = 0
    code = 0
    with open(os.path.join(a.out, "index.csv"), "w") as idx:
        idx.write("file,t_unix,t_rel_s\n")
        while time.time() - t0 < a.seconds:
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if fails >= 10:
                    print("error: camera stopped delivering frames (unplugged?)", file=sys.stderr)
                    code = 3
                    break
                time.sleep(0.1)
                continue
            fails = 0
            now = time.time()
            if now - last < 1.0 / a.fps:
                continue
            last = now
            name = f"frame_{int(now * 1000)}.jpg"
            cv2.imwrite(os.path.join(a.out, name), frame, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            idx.write(f"{name},{now:.3f},{now - t0:.3f}\n")
            idx.flush()
            n += 1
    cap.release()
    print(f"saved {n} frames to {a.out}")
    if n == 0 and code == 0:
        print("error: camera delivered no frames", file=sys.stderr)
        code = 3
    return code


if __name__ == "__main__":
    sys.exit(main())
