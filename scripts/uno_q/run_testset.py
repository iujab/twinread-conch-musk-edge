"""Run a TwinRead int8 bundle over CRC-VAL-HE-7K with device/twinread_runtime.EdgeModel and save per-tile outputs.

    # on the UNO Q (repo at ~/twinread-conch-musk-edge, venv from device/install_offline.sh):
    OMP_NUM_THREADS=4 device/.venv/bin/python -I scripts/uno_q/run_testset.py --repo . --bundle bundles/cm \
        --x data/crc_test_x.npy --out board_cm          # -> board_cm.npz + board_cm.json
    # the same on the x86 host gives host_cm.npz for the tile-by-tile comparison

Same inference loop as scripts/int8_flag_fidelity.py (EdgeModel.infer in batches of 64), but numpy + onnxruntime only,
so it runs on the UNO Q. Metrics are computed on the host from the saved npz. Research prototype, not a diagnostic device.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

REASONS = ("disagree", "js", "review_head", "no_tissue")


def temps():
    out = {}
    for z in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            out[(z / "type").read_text().strip()] = int((z / "temp").read_text()) / 1000
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--x", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, str(a.repo / "device"))
    import onnxruntime as ort
    import twinread_runtime as rt

    x = np.load(a.x, mmap_mode="r")
    n = min(len(x), a.limit) if a.limit else len(x)
    m = rt.EdgeModel(a.bundle, threads=a.threads)
    t_before = temps()
    res, batch_s = [], []
    t0 = time.time()
    for s in range(0, n, a.batch):
        xb = np.ascontiguousarray(x[s:min(s + a.batch, n)])
        tb = time.perf_counter()
        res.extend(m.infer(xb))
        batch_s.append(time.perf_counter() - tb)
        if (s // a.batch) % 20 == 0:
            print(json.dumps({"done": len(res), "of": n, "elapsed_s": round(time.time() - t0, 1)}), flush=True)
    secs = time.time() - t0
    cls = m.classes
    np.savez_compressed(
        a.out.with_suffix(".npz"),
        pred=np.array([cls.index(r.label) for r in res], np.int16),
        review=np.array([r.review for r in res], bool),
        reasons=np.array([[k in r.reasons for k in REASONS] for r in res], bool),
        review_score=np.array([r.review_score for r in res], np.float64),
        probsA=np.array([r.probsA for r in res], np.float64),
        probsB=np.array([r.probsB for r in res], np.float64),
        probsF=np.array([r.probsF for r in res], np.float64),
    )
    meta = {
        "bundle": str(a.bundle), "config": m.cfg, "n": n, "threads": a.threads, "batch": a.batch,
        "seconds": round(secs, 2), "ms_per_tile": round(1000 * secs / n, 2), "tiles_per_s": round(n / secs, 2),
        "batch_seconds_p50": float(np.percentile(batch_s, 50)), "batch_seconds_max": float(max(batch_s)),
        "host": {"machine": platform.machine(), "system": platform.system(), "release": platform.release(),
                 "node": platform.node(), "python": platform.python_version(), "onnxruntime": ort.__version__,
                 "numpy": np.__version__, "cpu_count": os.cpu_count()},
        "temps_c_before": t_before, "temps_c_after": temps(),
        "reasons_order": REASONS,
        "note": "EdgeModel.infer timing includes the tissue check and probes, not camera capture or cropping.",
    }
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("n", "seconds", "ms_per_tile", "tiles_per_s")}), flush=True)


if __name__ == "__main__":
    main()
