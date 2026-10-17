"""Run the K2 kernels under NVIDIA compute-sanitizer (memcheck, racecheck, synccheck) and record the summaries in
results/kernels/k2_sanitizer.json.

Workload (``--child``): odd frame sizes (37x53, 61x83, 7x300, 300x5), 1x1 and 8x8 frames, 224x224 and 480x640, batch
1..5, identity params (severity 0) and severity 1 with sigma forced to 4.0 (the largest blur radius, 12) on image 0,
plus the Philox test hook. Random uint8 content.
  flock $B/.gpu.lock python bench/sanitize_k2.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1]
OUT = P / "results" / "kernels" / "k2_sanitizer.json"
SHAPES = [(5, 37, 53), (2, 61, 83), (1, 1, 1), (3, 8, 8), (2, 7, 300), (2, 300, 5), (2, 224, 224), (1, 480, 640)]


def child() -> None:
    sys.path.insert(0, str(P / "src"))
    import numpy as np
    import torch

    from twinread.kernels import CameraAugParams, camera_aug, load_k2_ext, sample_camera_params
    ext = load_k2_ext()
    assert ext is not None, "K2 extension unavailable"
    rng = np.random.default_rng(0)
    for (B, H, W) in SHAPES:
        x = torch.from_numpy(rng.integers(0, 256, (B, H, W, 3), dtype=np.uint8)).cuda()
        for sev in (0.0, 1.0):
            p = sample_camera_params(B, sev, seed=H + W, device="cuda")
            if sev > 0:
                d = p.to_dicts()
                d[0]["blur_sigma"] = 4.0
                p = CameraAugParams.from_dicts(d, "cuda")
            camera_aug(x, p, seed=1, impl="cuda")
            torch.cuda.synchronize()
    ext.philox_bits(torch.zeros(10, 4, dtype=torch.int64, device="cuda"), 1, 2)
    torch.cuda.synchronize()
    print("workload done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args()
    if a.child:
        child()
        return
    res = {"date": dt.datetime.now().isoformat(timespec="seconds"), "shapes_BHW": SHAPES,
           "git": subprocess.run(["git", "-C", str(P), "describe", "--always", "--dirty", "--exclude", "*"],
                                 capture_output=True, text=True).stdout.strip(), "tools": {}}
    for tool in ("memcheck", "racecheck", "synccheck"):
        pr = subprocess.run(["/usr/local/cuda/bin/compute-sanitizer", "--tool", tool, "--error-exitcode", "9",
                             sys.executable, __file__, "--child"], capture_output=True, text=True, timeout=1800)
        summary = [ln.strip("= ").strip() for ln in pr.stdout.splitlines() if "SUMMARY" in ln]
        n = re.findall(r"(\d+) (?:errors|hazards)", " ".join(summary))
        res["tools"][tool] = {"returncode": pr.returncode, "summary": summary,
                              "workload_completed": "workload done" in pr.stdout,
                              "clean": pr.returncode == 0 and "workload done" in pr.stdout and all(v == "0" for v in n)}
        print(tool, res["tools"][tool], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
