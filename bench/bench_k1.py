"""K1 benchmark: dual-reader preprocessing (CONCH 448/448 + MUSK 384/384 inputs from the same raw frame).

Throughput in tiles/s (one "tile" = one raw input turned into BOTH readers' normalized tensors) for
  (a) PIL + the official transforms, per image, in one process (the organizer model-server style);
  (b) a torch DataLoader running (a) in 4 and 8 worker processes;
  (c) torch GPU ops: the exact fixed-point path (``impl="torch"``) and the common ``F.interpolate(bicubic,
      antialias=True)`` path (NOT PIL-exact; its error is reported);
  (d) Triton (``impl="triton"``); (e) the CUDA K1 extension (``impl="cuda"``), also including the H2D copy of the
      uint8 batch from pinned host memory.
Inputs: batch 256 of real 224-px CRC_VAL tiles, and 256 640x480 frames built as mosaics of those tiles (real tissue
content; no camera). GPU timing: CUDA events, warmup 3, median of 10; achieved GB/s vs a cudaMemcpy D2D baseline.

  python bench/bench_k1.py --part cpu                                   # CPU baselines, no GPU
  flock $B/.gpu.lock python bench/bench_k1.py --part gpu                # GPU paths (< 1 min)
  flock $B/.gpu.lock python bench/bench_k1.py --part parity             # bit-exactness + embedding cosines
  python bench/bench_k1.py --part gpu --batch 64 --out results/delta/kernels   # NCSA J1: short run on the A100
Parts merge into <out>/k1_bench.json and <out>/k1_parity.json (default out: results/kernels). Research prototype.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / "src"))
sys.path.insert(0, str(P / "tools"))

from twinread.kernels import dual_preprocess, make_plan, resized_geometry  # noqa: E402
from twinread.readers import _compose, spec_of  # noqa: E402

OUT = P / "results" / "kernels"
A, B = spec_of("conch"), spec_of("musk")
BATCH = 256


# ------------------------------------------------------------------------------------------------ inputs
def load_inputs(n: int = BATCH) -> tuple[dict[str, np.ndarray], dict]:
    try:
        from make_crc_fixture import load_crc_val
        tiles, labels = load_crc_val(n, seed=1)
        src = f"1aurent/NCT-CRC-HE CRC_VAL_HE_7K, {n} tiles class-balanced (seed 1), HF cache"
    except Exception as e:  # noqa: BLE001
        t16 = np.load(P / "tests" / "fixtures" / "crc16.npy")
        tiles = np.concatenate([t16] * (n // 16 + 1))[:n]
        src = f"tests/fixtures/crc16.npy repeated to {n} (HF cache unavailable: {e})"
    frames = []
    for i in range(n):
        t = [tiles[(i + k) % n] for k in range(9)]
        m = np.concatenate([np.concatenate(t[r * 3:r * 3 + 3], 1) for r in range(3)], 0)  # 672x672
        frames.append(m[96:576, 16:656])
    frames = np.ascontiguousarray(np.stack(frames))
    meta = {"tiles224": {"source": src, "shape": list(tiles.shape)},
            "frames480x640": {"source": "3x3 mosaics of the tiles above cropped to 480x640 (real tissue, no camera)",
                              "shape": list(frames.shape)}}
    return {"tiles224": np.ascontiguousarray(tiles), "frames480x640": frames}, meta


def official_transforms():
    from twinread.compat import patch_conch
    patch_conch()
    from conch.open_clip_custom.transform import image_transform
    return image_transform(A.size, mean=A.mean, std=A.std), _compose(B)


# ------------------------------------------------------------------------------------------------ env
def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({e})"


def env_block(gpu: bool) -> dict:
    import PIL
    import torchvision
    e = {
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "torch": torch.__version__, "torch_cuda_runtime": torch.version.cuda, "torchvision": torchvision.__version__,
        "pillow": PIL.__version__, "numpy": np.__version__, "python": platform.python_version(),
        "cpu": next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), "?"),
        "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "loadavg_1m": os.getloadavg()[0],
        "git": _sh(["git", "-C", str(P), "describe", "--always", "--dirty", "--exclude", "*"]),
    }
    if gpu:
        import triton
        e.update({
            "gpu": torch.cuda.get_device_name(0), "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "driver": _sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
            "nvcc": (_sh(["/usr/local/cuda/bin/nvcc", "--version"]) or "unavailable").splitlines()[-1],
            "triton": triton.__version__,
        })
    return e


# ------------------------------------------------------------------------------------------------ CPU part
class _DS(torch.utils.data.Dataset):
    def __init__(self, arr, reps, tfa, tfb):
        self.arr, self.reps, self.tfa, self.tfb = arr, reps, tfa, tfb

    def __len__(self):
        return len(self.arr) * self.reps

    def __getitem__(self, i):
        from PIL import Image
        im = Image.fromarray(self.arr[i % len(self.arr)])
        return self.tfa(im), self.tfb(im)


def bench_cpu(inputs: dict, repeats: int = 1) -> dict:
    """CPU baselines; with repeats > 1 every method is re-measured and the BEST rate is kept (the machine is shared
    with other jobs, so best-of-N is the conservative baseline for K1's speedup). All runs are recorded."""
    runs = [_bench_cpu_once(inputs) for _ in range(repeats)]
    res = {}
    for name in inputs:
        res[name] = {}
        for m in runs[0][name]:
            all_r = [r[name][m] for r in runs]
            best = max(all_r, key=lambda v: v["tiles_per_s"])
            res[name][m] = dict(best, all_tiles_per_s=[v["tiles_per_s"] for v in all_r],
                                loadavg_1m_per_run=[v["loadavg_1m"] for v in all_r], kept="best of runs")
    return res


def _bench_cpu_once(inputs: dict) -> dict:
    from PIL import Image
    tfa, tfb = official_transforms()
    res = {}
    for name, arr in inputs.items():
        r = {}
        for im in arr[:8]:  # warmup
            pim = Image.fromarray(im)
            tfa(pim), tfb(pim)
        runs = []
        for _ in range(3):
            t = time.perf_counter()
            for im in arr:
                pim = Image.fromarray(im)
                tfa(pim), tfb(pim)
            runs.append(time.perf_counter() - t)
        s = float(np.median(runs))
        r["pil_single_process"] = {"n": len(arr), "seconds_median3": s, "tiles_per_s": len(arr) / s,
                                   "torch_threads": torch.get_num_threads(), "loadavg_1m": os.getloadavg()[0]}
        reps, bs = (8, 64) if name == "tiles224" else (2, 32)
        for nw in (4, 8):
            ds = _DS(arr, reps, tfa, tfb)
            dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=nw, persistent_workers=True)
            for _ in dl:  # warmup epoch: worker start-up excluded from the timed epoch
                pass
            t = time.perf_counter()
            n = 0
            for xa, xb in dl:
                n += xa.shape[0]
            s = time.perf_counter() - t
            del dl
            r[f"dataloader_{nw}w"] = {"n": n, "batch_size": bs, "seconds": s, "tiles_per_s": n / s,
                                      "loadavg_1m": os.getloadavg()[0]}
        res[name] = r
        print(name, {k: round(v["tiles_per_s"], 1) for k, v in r.items()}, flush=True)
    return res


# ------------------------------------------------------------------------------------------------ GPU part
def cuda_time(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def interp_dual(x: torch.Tensor, chunk: int = 32):
    """The usual GPU shortcut: float bicubic antialias interpolate + crop + normalize (no uint8 rounding/clamp).
    Chunked by 32 images (float temporaries of 256 frames do not fit next to other jobs on an 8 GB card)."""
    import torch.nn.functional as F
    n, H, W = x.shape[:3]
    outs = [torch.empty((n, 3, s.crop, s.crop), dtype=torch.float16, device=x.device) for s in (A, B)]
    for i in range(0, n, chunk):
        xf = x[i:i + chunk].permute(0, 3, 1, 2).float()
        for o, spec in zip(outs, (A, B)):
            nh, nw, top, left = resized_geometry(H, W, spec)
            y = F.interpolate(xf, size=(nh, nw), mode="bicubic", antialias=True, align_corners=False)
            y = y[:, :, top:top + spec.crop, left:left + spec.crop]
            m = torch.tensor(spec.mean, device=x.device).view(1, 3, 1, 1)
            sd = torch.tensor(spec.std, device=x.device).view(1, 3, 1, 1)
            o[i:i + chunk] = ((y / 255.0 - m) / sd).half()
    return tuple(outs)


def kernel_breakdown(fn) -> dict:
    from torch.profiler import ProfilerActivity, profile
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
    out = {}
    for ev in prof.key_averages():
        if ev.device_time_total > 0:
            k = ev.key.replace("(anonymous namespace)::", "").split("(")[0][:60]
            out[k] = out.get(k, 0.0) + round(ev.device_time_total / 10 / 1000, 4)  # ms per dual_preprocess call
    return out


def bench_gpu(inputs: dict) -> dict:
    dev = "cuda"
    buf = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    dst = torch.empty_like(buf)
    mc_ms = cuda_time(lambda: dst.copy_(buf))
    memcpy = 2 * buf.numel() / mc_ms / 1e6
    del buf, dst
    res = {"memcpy_d2d": {"bytes": 256 * 1024 * 1024, "ms": mc_ms, "GBps_read_plus_write": memcpy}}
    for name, arr in inputs.items():
        x = torch.from_numpy(arr).to(dev)
        n, H, W, _ = x.shape
        pa, pb = make_plan(H, W, A, x.device), make_plan(H, W, B, x.device)
        in_b = x.numel()
        out_b = n * 3 * (A.crop ** 2 + B.crop ** 2) * 2
        tmp_b = n * 3 * (pa.rows * pa.crop + pb.rows * pb.crop)
        floor_ms = (in_b + 2 * tmp_b + out_b) / (memcpy * 1e6)
        r = {"bytes": {"input_uint8": in_b, "outputs_fp16": out_b, "intermediates_uint8": tmp_b},
             "two_pass_dram_floor_ms": floor_ms,
             "note_floor": "(input + 2*intermediate + outputs) / memcpy rate; the CUDA hpass reads the input once"}
        pinned = torch.from_numpy(arr).pin_memory()
        meths = {
            "torch_exact": lambda: dual_preprocess(x, A, B, impl="torch"),
            "torch_interpolate": lambda: interp_dual(x),
            "triton": lambda: dual_preprocess(x, A, B, impl="triton"),
            "cuda_k1": lambda: dual_preprocess(x, A, B, impl="cuda"),
            "cuda_k1_incl_h2d": lambda: dual_preprocess(pinned.to(dev, non_blocking=True), A, B, impl="cuda"),
        }
        for m, fn in meths.items():
            try:  # one failing path (e.g. Triton on another GPU/toolchain) must not lose the others' numbers
                ms = cuda_time(fn)
            except Exception as e:  # noqa: BLE001
                r[m] = {"error": f"{type(e).__name__}: {e}"}
                print(name, m, "FAILED:", r[m]["error"], flush=True)
                continue
            finally:
                torch.cuda.empty_cache()
            r[m] = {"ms": ms, "tiles_per_s": n / ms * 1e3, "eff_GBps": (in_b + out_b) / ms / 1e6,
                    "eff_frac_of_memcpy": (in_b + out_b) / ms / 1e6 / memcpy}
        for m in ("cuda_k1", "triton"):
            if "error" not in r[m]:
                r[m]["kernels_ms"] = kernel_breakdown(meths[m])
        # accuracy of the interpolate shortcut on the first 16 inputs (the exact paths: see k1_parity.json)
        from PIL import Image
        tfa, tfb = official_transforms()
        ia, ib = interp_dual(x[:16])
        acc = {}
        for key, o, tf, spec in (("conch", ia, tfa, A), ("musk", ib, tfb, B)):
            ref = torch.stack([tf(Image.fromarray(a)) for a in arr[:16]])
            lv = (o.float().cpu() - ref).abs() * 255 * torch.tensor(spec.std).view(1, 3, 1, 1)
            acc[key] = {"max_levels": lv.max().item(), "mean_levels": lv.mean().item(),
                        "frac_bit_exact_vs_official_half": (o.cpu() == ref.half()).float().mean().item()}
        r["torch_interpolate"]["accuracy_first16"] = acc
        res[name] = r
        print(name, {k: round(v["tiles_per_s"]) for k, v in r.items() if isinstance(v, dict) and "tiles_per_s" in v},
              flush=True)
        del x, pinned
        torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------------------------------------------ parity part
def bench_parity(inputs: dict) -> dict:
    from PIL import Image
    from twinread.readers import load_reader
    tfa, tfb = official_transforms()
    rng = np.random.default_rng(11)
    sets = {"crc_tiles256": inputs["tiles224"], "crc_mosaic_frames480x640_first32": inputs["frames480x640"][:32],
            "random_480x640": rng.integers(0, 256, (4, 480, 640, 3), dtype=np.uint8),
            "random_640x480": rng.integers(0, 256, (4, 640, 480, 3), dtype=np.uint8),
            "random_720x1280": rng.integers(0, 256, (4, 720, 1280, 3), dtype=np.uint8),
            "random_225x301": rng.integers(0, 256, (4, 225, 301, 3), dtype=np.uint8)}
    res = {"unit": "levels = |out - official| * 255 * std (one uint8 level / std = 1.0)", "sets": {}}
    for name, arr in sets.items():
        refs = {"conch": torch.stack([tfa(Image.fromarray(a)) for a in arr]),
                "musk": torch.stack([tfb(Image.fromarray(a)) for a in arr])}
        x = torch.from_numpy(arr).cuda()
        row = {"n": len(arr), "hw": list(arr.shape[1:3])}
        for impl in ("cuda", "triton", "torch", "interpolate"):
            oa, ob = interp_dual(x) if impl == "interpolate" else dual_preprocess(x, A, B, impl=impl)
            row[impl] = {}
            for key, o, spec in (("conch", oa, A), ("musk", ob, B)):
                o = o.cpu()
                lv = (o.float() - refs[key]).abs() * 255 * torch.tensor(spec.std).view(1, 3, 1, 1)
                row[impl][key] = {"frac_bit_exact_vs_official_half": (o == refs[key].half()).float().mean().item(),
                                  "max_levels": lv.max().item(), "mean_levels": lv.mean().item()}
        res["sets"][name] = row
        print(name, {i: (round(row[i]["conch"]["frac_bit_exact_vs_official_half"], 6),
                         round(row[i]["musk"]["frac_bit_exact_vs_official_half"], 6))
                     for i in ("cuda", "triton", "torch", "interpolate")}, flush=True)
        del x
    # embeddings: 256 real tiles, official transform vs K1 vs the interpolate shortcut
    tiles = inputs["tiles224"]
    x = torch.from_numpy(tiles).cuda()
    k1 = dict(zip(("conch", "musk"), [t.cpu() for t in dual_preprocess(x, A, B, impl="cuda")]))
    it = dict(zip(("conch", "musk"), [t.cpu() for t in interp_dual(x)]))
    del x
    emb = {}
    for name in ("conch", "musk"):
        r = load_reader(name)
        try:
            e_ref, e_k1, e_it = [], [], []
            for s in range(0, len(tiles), 16):
                ref = torch.stack([r.preprocess_pil(Image.fromarray(a)) for a in tiles[s:s + 16]])
                e_ref.append(r.encode_aligned(ref))
                e_k1.append(r.encode_aligned(k1[name][s:s + 16].float()))
                e_it.append(r.encode_aligned(it[name][s:s + 16].float()))
        finally:
            r.close()
        e_ref, e_k1, e_it = map(np.concatenate, (e_ref, e_k1, e_it))
        cos = lambda u, v: (u * v).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1)  # noqa: E731
        c1, c2 = cos(e_k1, e_ref), cos(e_it, e_ref)
        emb[name] = {"n": len(tiles), "dtype": "fp32" if name == "conch" else "fp16",
                     "cuda_k1_vs_official": {"min": float(c1.min()), "mean": float(c1.mean())},
                     "interpolate_vs_official": {"min": float(c2.min()), "mean": float(c2.mean())}}
        print(name, emb[name], flush=True)
    res["embedding_cosine_crc_tiles256"] = emb
    return res


# ------------------------------------------------------------------------------------------------ main
def _merge(path: Path, key: str, value) -> dict:
    d = json.loads(path.read_text()) if path.exists() else {}
    d[key] = value
    path.write_text(json.dumps(d, indent=2))
    return d


def summarize(d: dict) -> dict:
    s = {}
    for name in ("tiles224", "frames480x640"):
        c, g = d.get("cpu", {}).get(name), d.get("gpu", {}).get(name)
        if not (c and g):
            continue
        k = g["cuda_k1"]["tiles_per_s"]
        s[name] = {"cuda_k1_tiles_per_s": k,
                   "speedup_vs_pil_single_process": k / c["pil_single_process"]["tiles_per_s"],
                   "speedup_vs_dataloader_8w": k / c["dataloader_8w"]["tiles_per_s"],
                   "speedup_vs_triton": k / g["triton"]["tiles_per_s"],
                   "speedup_vs_torch_exact": k / g["torch_exact"]["tiles_per_s"],
                   "incl_h2d_speedup_vs_dataloader_8w": g["cuda_k1_incl_h2d"]["tiles_per_s"]
                   / c["dataloader_8w"]["tiles_per_s"],
                   "cuda_k1_ms_vs_two_pass_floor": g["cuda_k1"]["ms"] / g["two_pass_dram_floor_ms"]}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["cpu", "gpu", "parity", "all"], default="all")
    ap.add_argument("--cpu-repeats", type=int, default=3, help="CPU baselines: keep the best of N runs")
    ap.add_argument("--out", type=Path, default=OUT, help="output dir (default results/kernels)")
    ap.add_argument("--batch", type=int, default=BATCH, help="inputs per set (default 256; NCSA J1 uses 64)")
    a = ap.parse_args()
    torch.set_num_threads(4)
    out = a.out if a.out.is_absolute() else P / a.out
    out.mkdir(parents=True, exist_ok=True)
    inputs, meta = load_inputs(a.batch)
    bench_path = out / "k1_bench.json"
    parts = ["cpu", "gpu", "parity"] if a.part == "all" else [a.part]
    for part in parts:
        if part == "parity":
            _merge(out / "k1_parity.json", "env", env_block(True))
            _merge(out / "k1_parity.json", "result", bench_parity(inputs))
            continue
        res = bench_cpu(inputs, a.cpu_repeats) if part == "cpu" else bench_gpu(inputs)
        d = _merge(bench_path, part, res)
        d = _merge(bench_path, "env" if part == "gpu" else "env_cpu", env_block(part == "gpu"))
        _merge(bench_path, "inputs", meta)
        _merge(bench_path, "config", {
            "batch": a.batch, "readers": {"A": f"conch {A.size}/{A.crop} bicubic", "B": f"musk {B.size}/{B.crop} bicubic"},
            "tiles_per_s": "raw inputs per second turned into BOTH readers' normalized tensors",
            "gpu_timing": "CUDA events, warmup 3, median of 10; input already on the GPU unless *_incl_h2d",
            "cpu_timing": "time.perf_counter; PIL single process: median of 3 passes over 256 inputs; DataLoader: "
                          "second epoch (persistent workers) over 2048 tiles (bs 64) or 512 frames (bs 32); whole CPU "
                          "part repeated --cpu-repeats times on a shared 16-core machine, best rate kept",
            "eff_GBps": "(uint8 input bytes + fp16 output bytes) / time",
        })
        _merge(bench_path, "summary", summarize(json.loads(bench_path.read_text())))
    print("wrote", bench_path)


if __name__ == "__main__":
    main()
