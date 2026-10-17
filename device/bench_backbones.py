"""Latency harness for candidate backbones. Imports only numpy + onnxruntime (+ stdlib): runs on the UNO Q.
Research prototype, not a diagnostic device. Rows from a non-aarch64 host are flagged simulated=True."""
import argparse
import glob
import json
import os
import platform
import re
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort


def rss_mb():
    try:
        with open("/proc/self/status") as f:
            m = re.search(r"VmRSS:\s+(\d+) kB", f.read())
        return round(int(m.group(1)) / 1024, 1)
    except Exception:
        return float("nan")


def peak_rss_mb():
    try:
        with open("/proc/self/status") as f:
            m = re.search(r"VmHWM:\s+(\d+) kB", f.read())
        return round(int(m.group(1)) / 1024, 1)
    except Exception:
        return float("nan")


def bench_one(path, threads, runs, warmup=3):
    """Runs in a fresh child process: rss_mb = peak RSS minus the pre-session baseline (after importing onnxruntime)."""
    base = rss_mb()
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    x = np.random.RandomState(0).rand(1, 3, 224, 224).astype(np.float32)
    for _ in range(warmup):
        s.run(None, {name: x})
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        s.run(None, {name: x})
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.percentile(ts, 50)), float(np.percentile(ts, 90)), round(peak_rss_mb() - base, 1), base


def bench_isolated(path, threads, runs):
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", path, str(threads), str(runs)],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        p50, p90, rss, base = bench_one(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
        print(json.dumps([p50, p90, rss, base]))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="device/backbones")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--out", default="results/uno_q/backbone_latency.json")
    a = ap.parse_args()
    simulated = bool(a.simulate or platform.machine() != "aarch64")
    host = {"machine": platform.machine(), "system": platform.system(), "release": platform.release(),
            "python": platform.python_version(), "onnxruntime": ort.__version__, "cpu_count": os.cpu_count()}
    rows = []
    for p in sorted(glob.glob(os.path.join(a.dir, "*_*.onnx"))):
        m = re.match(r"(.+)_(fp32|int8)\.onnx$", os.path.basename(p))
        if not m:
            continue
        for th in a.threads:
            p50, p90, rss, base = bench_isolated(p, th, a.runs)
            rows.append({"backbone": m.group(1), "precision": m.group(2), "threads": th,
                         "p50_ms": round(p50, 2), "p90_ms": round(p90, 2), "rss_mb": rss, "baseline_rss_mb": base,
                         "size_mb": round(os.path.getsize(p) / 1e6, 2)})
            print(rows[-1], flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"host": host, "simulated": simulated, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
