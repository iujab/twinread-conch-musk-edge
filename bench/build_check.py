"""Compile-check the K1+K2 extension for other GPU architectures and record the result in
results/kernels/builds.json.

For each arch this runs, in a fresh process:
  TORCH_CUDA_ARCH_LIST=<arch> python -c "from twinread.kernels import load_ext; m = load_ext('<arch>'); \
      assert m is not None and hasattr(m, 'camera_aug')"
(the second condition fails if only the K1 fallback module built), then lists the SASS targets and kernel symbols
embedded in the built .so with cuobjdump. COMPILE ONLY: no record here was executed (``"executed": false``); an
sm_80/sm_90 binary cannot run on the local sm_120 GPU, and sm_120 execution is covered by the test suite, not by
this script. Delta: 8.0 = A100, 8.6 = A40, 9.0 = GH200 (DeltaAI).
  python bench/build_check.py --clean --arch 8.0 --arch 9.0 --arch 12.0
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / "src"))
OUT = P / "results" / "kernels" / "builds.json"


def git_rev() -> str:
    return subprocess.run(["git", "-C", str(P), "describe", "--always", "--dirty", "--exclude", "*"],
                          capture_output=True, text=True).stdout.strip()


def check(arch: str, clean: bool) -> dict:
    from twinread.kernels import _arch_tag
    tag = _arch_tag(arch)
    if clean:  # a true compile time, not a cached load
        shutil.rmtree(P / "build" / f"ext_{tag}", ignore_errors=True)
    env = dict(os.environ, TORCH_CUDA_ARCH_LIST=arch, PYTHONPATH=str(P / "src"), MAX_JOBS="4",
               PATH="/usr/local/cuda/bin:" + os.environ.get("PATH", ""))
    if clean:
        env["CCACHE_DISABLE"] = "1"  # /usr/bin/ccache wraps gcc/nvcc on this machine
    cmd = [sys.executable, "-c", f"from twinread.kernels import load_ext; m = load_ext('{arch}'); "
           "assert m is not None and hasattr(m, 'camera_aug')"]
    t = time.time()
    pr = subprocess.run(cmd, cwd=str(P), env=env, capture_output=True, text=True, timeout=900)
    secs = time.time() - t
    so = P / "build" / f"ext_{tag}" / f"twinread_ext_{tag}.so"
    rec = {"arch": arch, "command": f"TORCH_CUDA_ARCH_LIST=\"{arch}\" python -c \"{cmd[2]}\"",
           "ok": pr.returncode == 0 and so.exists(), "returncode": pr.returncode, "seconds": round(secs, 1),
           "clean_build": clean, "executed": False, "git": git_rev(),
           "note": "compile-only: built and SASS/symbols listed; no kernel from this binary was launched",
           "so": str(so.relative_to(P)) if so.exists() else None,
           "stderr_tail": pr.stderr.strip().splitlines()[-5:] if pr.returncode else []}
    if so.exists():
        lst = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "--list-elf", str(so)], capture_output=True, text=True)
        rec["embedded_sass"] = sorted(set(re.findall(r"sm_\d+", lst.stdout)))
        sym = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "--dump-elf-symbols", str(so)], capture_output=True,
                             text=True).stdout
        rec["kernels"] = [k for k in ("k1_hpass_dual", "k1_vpass_vec4", "k1_hpass", "k1_vpass", "k2_fused", "k2_jpeg",
                                      "k2_philox") if re.search(rf"\d{k}E", sym)]  # mangled: <len><name>E
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", action="append", default=None)
    ap.add_argument("--clean", action="store_true", help="delete build/ext_<arch> first (fresh compile)")
    a = ap.parse_args()
    archs = a.arch or ["8.0"]
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    nvcc = subprocess.run(["/usr/local/cuda/bin/nvcc", "--version"], capture_output=True, text=True).stdout
    d["toolchain"] = {"nvcc": nvcc.strip().splitlines()[-1], "date": dt.datetime.now().isoformat(timespec="seconds"),
                      "flags": "extra_cuda_cflags=['-O3'] (no --use_fast_math), extra_cflags=['-O3']",
                      "git": git_rev(), "sources": "bindings.cpp, k1_dual_preprocess.cu, k2_camera_aug.cu",
                      "executed": False}
    d.setdefault("builds", {})
    for arch in archs:
        rec = check(arch, a.clean)
        d["builds"][arch] = rec
        print(json.dumps(rec), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
