"""K2 benchmark, parity and figure: SIMULATED camera-of-screen augmentation (warp, HED jitter, display gamma + colour
profile, moire, glare, defocus, white balance, sensor noise, JPEG-like 8x8 DCT quantization) on real 224-px
CRC-VAL-HE-7K tiles. No camera is involved anywhere in this script.

Throughput, images/s at 224 px, batch 256, params ``sample_camera_params(256, severity, seed=0)`` (severity 1.0
unless stated):
  (a) numpy reference per image, one process (``ref.camera_ops.ref_camera_aug``, float64, 1 BLAS thread);
  (b) the same in a 4-process pool (fork, 1 BLAS thread each; pickling the tiles is included);
  (c) torch GPU ops (``impl="torch"``, float32, same Philox noise stream emulated with int64 ops, chunks of 64);
  (c2) the same torch ops with ``torch.randn`` noise (not the kernel's stream): what plain torch would cost;
  (d) the CUDA K2 kernel (``impl="cuda"``: k2_fused + k2_jpeg, plus one host sync to size the blur tile).
GPU timing: CUDA events, warmup 3, median of 10. CPU: best of N runs (the machine is shared), all runs recorded.

  python bench/bench_k2.py --part cpu
  flock $B/.gpu.lock python bench/bench_k2.py --part gpu
  flock $B/.gpu.lock python bench/bench_k2.py --part parity
  flock $B/.gpu.lock python bench/bench_k2.py --part figure
Parts merge into results/kernels/k2_bench.json and k2_parity.json; the figure is figures/k2_severity_grid.png
(its params: results/kernels/k2_figure.json). Research prototype, not a diagnostic device.
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

P = Path(__file__).resolve().parents[1]
for _p in (P / "src", P / "tools", P / "bench"):
    sys.path.insert(0, str(_p))

from bench_k1 import cuda_time, env_block, kernel_breakdown  # noqa: E402
from twinread.k2_torch import camera_aug_torch  # noqa: E402
from twinread.kernels import CameraAugParams, _camera_aug_impl, camera_aug, load_k2_ext, sample_camera_params  # noqa: E402
from twinread.ref import camera_ops as co  # noqa: E402

OUT = P / "results" / "kernels"
FIG = P / "figures" / "k2_severity_grid.png"
BATCH, PSEED, NSEED = 256, 0, 1234
CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]  # 1aurent/NCT-CRC-HE label names
OP_KEYS = {"warp": ["warp"], "hed": ["hed_alpha", "hed_beta"], "display": ["gamma", "profile"],
           "moire": ["moire_amp", "moire_period", "moire_angle"], "glare": ["glare_amp", "glare_angle"],
           "defocus": ["blur_sigma"], "white_balance": ["wb_gain"], "noise": ["noise_std"], "jpeg": ["jpeg_quality"]}


def load_tiles(n: int, seed: int = 1) -> tuple[np.ndarray, str]:
    try:
        from make_crc_fixture import load_crc_val
        tiles, _ = load_crc_val(n, seed=seed)
        return np.ascontiguousarray(tiles), f"1aurent/NCT-CRC-HE CRC_VAL_HE_7K, {n} tiles class-balanced (seed {seed})"
    except Exception as e:  # noqa: BLE001
        t16 = np.load(P / "tests" / "fixtures" / "crc16.npy")
        return np.concatenate([t16] * (n // 16 + 1))[:n], f"tests/fixtures/crc16.npy repeated to {n} ({e})"


def _dicts(p: CameraAugParams, seed: int) -> list[dict]:
    return [dict(d, seed=seed, index=i) for i, d in enumerate(p.to_dicts())]


def _merge(path: Path, key: str, val) -> None:
    d = json.loads(path.read_text()) if path.exists() else {}
    d[key] = val
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=2))
    print("wrote", path, key, flush=True)


# ------------------------------------------------------------------------------------------------ CPU
def _worker_init() -> None:
    global _LIMIT  # noqa: PLW0603 - keep the limiter alive in the worker
    _LIMIT = threadpool_limits(1)


def _ref_one(args):
    img, d = args
    return co.ref_camera_aug(img, d)


def bench_cpu(tiles: np.ndarray, repeats: int) -> dict:
    p = sample_camera_params(BATCH, 1.0, PSEED, device="cpu")
    ds = _dicts(p, NSEED)
    a_runs, b_runs, loads = [], [], []
    for _ in range(repeats):
        with threadpool_limits(1):
            for i in range(4):
                co.ref_camera_aug(tiles[i], ds[i])
            t = time.perf_counter()
            for img, d in zip(tiles, ds):
                co.ref_camera_aug(img, d)
            a_runs.append(BATCH / (time.perf_counter() - t))
        with mp.get_context("fork").Pool(4, initializer=_worker_init) as pool:
            pool.map(_ref_one, list(zip(tiles[:8], ds[:8])), chunksize=2)  # worker start-up excluded
            t = time.perf_counter()
            pool.map(_ref_one, list(zip(tiles, ds)), chunksize=8)
            b_runs.append(BATCH / (time.perf_counter() - t))
        loads.append(os.getloadavg()[0])
        print("cpu run", round(a_runs[-1], 1), round(b_runs[-1], 1), flush=True)
    return {"severity": 1.0, "batch": BATCH,
            "a_numpy_single_process": {"images_per_s": max(a_runs), "all_runs": a_runs, "kept": "best of runs",
                                       "blas_threads": 1},
            "b_numpy_pool_4proc": {"images_per_s": max(b_runs), "all_runs": b_runs, "kept": "best of runs",
                                   "blas_threads_per_worker": 1, "chunksize": 8},
            "loadavg_1m_per_run": loads}


# ------------------------------------------------------------------------------------------------ GPU
def bench_gpu(tiles: np.ndarray) -> dict:
    x = torch.from_numpy(tiles).cuda()
    res: dict = {"batch": BATCH, "hw": list(tiles.shape[1:3])}
    for sev in (1.0, 0.5, 0.0):
        p = sample_camera_params(BATCH, sev, PSEED, device="cuda")
        r = {"blur_radius_max": co.blur_radius(float(p.blur_sigma.max()))}
        Pk = p.packed(BATCH, "cuda")
        meths = (("c_torch_gpu", lambda: camera_aug(x, p, NSEED, impl="torch")),
                 ("c2_torch_gpu_randn_noise", lambda: camera_aug_torch(x, Pk, NSEED, noise_rng="randn")),
                 ("d_cuda_k2", lambda: camera_aug(x, p, NSEED, impl="cuda")))
        for name, fn in meths:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            ms = cuda_time(fn)
            r[name] = {"ms": ms, "images_per_s": BATCH / ms * 1e3,
                       "peak_extra_MB": (torch.cuda.max_memory_allocated() - base) / 2 ** 20}
        r["speedup_cuda_vs_torch"] = r["c_torch_gpu"]["ms"] / r["d_cuda_k2"]["ms"]
        r["speedup_cuda_vs_torch_randn"] = r["c2_torch_gpu_randn_noise"]["ms"] / r["d_cuda_k2"]["ms"]
        res[f"severity_{sev}"] = r
        print(sev, {k: round(v["images_per_s"]) for k, v in r.items() if isinstance(v, dict)}, flush=True)
    p = sample_camera_params(BATCH, 1.0, PSEED, device="cuda")
    res["severity_1.0"]["d_cuda_k2"]["kernels_ms"] = kernel_breakdown(lambda: camera_aug(x, p, NSEED, impl="cuda"))
    ts = []
    for k in range(20):
        t = time.perf_counter()
        sample_camera_params(BATCH, 1.0, k, device="cuda")
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    res["sample_camera_params_ms_median"] = float(np.median(ts))
    vals = BATCH * tiles.shape[1] * tiles.shape[2] * 3
    res["bytes_note"] = ("k2_fused reads uint8 and writes float32; k2_jpeg reads float32 and writes uint8: "
                         f"{vals * 10 / 1e6:.0f} MB of DRAM traffic per batch at minimum")
    return res


# ------------------------------------------------------------------------------------------------ parity
def _err(out: np.ndarray, ref: np.ndarray) -> dict:
    d = np.abs(out.astype(np.int32) - ref.astype(np.int32))
    return {"mean_levels": float(d.mean()), "max_levels": int(d.max()), "n_values": int(d.size),
            "n_diff": int((d > 0).sum()), "frac_gt1": float((d > 1).mean())}


def _refs(arr: np.ndarray, dicts: list[dict], seed: int) -> np.ndarray:
    return np.stack([co.ref_camera_aug(im, dict(d, seed=seed, index=i)) for i, (im, d) in enumerate(zip(arr, dicts))])


def _explain(arr: np.ndarray, out: np.ndarray, inter: np.ndarray, dicts: list[dict]) -> dict:
    """camera_ops.explain_differences summed over a batch: the max error in blocks whose JPEG rounding provably
    matches the reference, and how many blocks are flip-possible."""
    rs = []
    for i, (im, d) in enumerate(zip(arr, dicts)):
        d = dict(d, seed=NSEED, index=i)
        rs.append(co.explain_differences(out[i], co.ref_camera_aug(im, d), inter[i], co.ref_pre_jpeg(im, d), d))
    return {"max_err_provably_same_rounding": max(r["max_err_provably_same_rounding"] for r in rs),
            "max_err_flip_possible": max(r["max_err_flip_possible"] for r in rs),
            "n_flip_possible_blocks": sum(r["n_flip_possible"] for r in rs), "n_blocks": sum(r["n_blocks"] for r in rs),
            "max_pre_jpeg_diff_levels": max(r["max_pre_jpeg_diff_levels"] for r in rs)}


def _op_dicts(op: str, n: int) -> list[dict]:  # same construction as tests/test_k2.py::_op_dicts
    out = []
    for i in range(n):
        d = co.identity_params()
        s = co.sample_params(1.0, seed=100 + i)
        for k in OP_KEYS[op]:
            d[k] = s[k]
        out.append(d)
    if op == "jpeg":
        for i, q in enumerate([10, 23, 37, 50, 64, 79, 95, 100][:n]):
            out[i]["jpeg_quality"] = float(q)
    if op == "defocus":
        out[-1]["blur_sigma"] = 4.0
    if op == "noise":
        out[-1]["noise_std"] = 0.05
    if op == "warp":
        out[-1]["warp"] = [0.05, -0.08, 0.06, 0.07, -0.04, -0.05, 0.06, -0.06]
    return out


def parity(tiles64: np.ndarray) -> dict:
    from PIL import Image
    res: dict = {"unit": "uint8 levels", "reference": "twinread.ref.camera_ops (numpy float64)"}
    arr8 = np.ascontiguousarray(np.load(P / "tests" / "fixtures" / "crc16.npy")[:8])
    x8 = torch.from_numpy(arr8).cuda()
    res["per_op"] = {}
    for op in OP_KEYS:
        d = _op_dicts(op, 8)
        ref = _refs(arr8, d, 7)
        r = {"op_effect_ref_vs_input_mae": float(np.abs(ref.astype(int) - arr8).mean())}
        for impl in ("cuda", "torch"):
            out = camera_aug(x8, CameraAugParams.from_dicts(d, "cuda"), 7, impl=impl).cpu().numpy()
            r[impl] = _err(out, ref)
        res["per_op"][op] = r
        print(op, r["cuda"]["max_levels"], r["torch"]["max_levels"], flush=True)
    x64 = torch.from_numpy(tiles64).cuda()
    comp, agg = [], {"cuda": [], "torch": [], "torch_vs_cuda": []}
    for sev in (0.25, 0.5, 0.75, 1.0):
        for k in range(4):
            p = sample_camera_params(len(tiles64), sev, 1000 + k, device="cuda")
            ref = _refs(tiles64, p.to_dicts(), NSEED)
            oc, ic = (t.cpu().numpy() for t in _camera_aug_impl(x64, p, NSEED, impl="cuda", return_float=True))
            ot, it = (t.cpu().numpy() for t in _camera_aug_impl(x64, p, NSEED, impl="torch", return_float=True))
            row = {"severity": sev, "param_seed": 1000 + k, "n_images": len(tiles64),
                   "cuda": _err(oc, ref), "torch": _err(ot, ref), "torch_vs_cuda": _err(ot, oc),
                   "effect_ref_vs_input_mae": float(np.abs(ref.astype(int) - tiles64).mean()),
                   "cuda_by_block": _explain(tiles64, oc, ic, p.to_dicts()),
                   "torch_by_block": _explain(tiles64, ot, it, p.to_dicts())}
            comp.append(row)
            for key in agg:
                agg[key].append(row[key])
        print("composition", sev, [(r["cuda"]["mean_levels"], r["cuda"]["max_levels"]) for r in comp[-4:]], flush=True)
    res["composition_runs"] = comp
    res["composition_summary"] = {
        key: {"n_runs": len(v), "mean_of_mean_levels": float(np.mean([e["mean_levels"] for e in v])),
              "worst_mean_levels": max(e["mean_levels"] for e in v), "worst_max_levels": max(e["max_levels"] for e in v),
              "runs_with_max_gt8": sum(e["max_levels"] > 8 for e in v),
              "frac_values_gt1_overall": float(np.mean([e["frac_gt1"] for e in v]))} for key, v in agg.items()}
    for impl in ("cuda", "torch"):
        bb = [r[f"{impl}_by_block"] for r in comp]
        res["composition_summary"][impl]["by_block"] = {
            "max_err_provably_same_rounding": max(b["max_err_provably_same_rounding"] for b in bb),
            "max_err_flip_possible": max(b["max_err_flip_possible"] for b in bb),
            "frac_blocks_flip_possible": sum(b["n_flip_possible_blocks"] for b in bb) / sum(b["n_blocks"] for b in bb),
            "max_pre_jpeg_diff_levels": max(b["max_pre_jpeg_diff_levels"] for b in bb),
            "rule": "camera_ops.explain_differences: block margin >= 8 * max|pre-JPEG diff| + DCT_F32_ERR"}
    # non-square camera-sized frames (3x3 mosaics of real tiles cropped to 480x640)
    frames = []
    for i in range(4):
        t = [tiles64[(i * 9 + k) % len(tiles64)] for k in range(9)]
        m = np.concatenate([np.concatenate(t[r * 3:r * 3 + 3], 1) for r in range(3)], 0)
        frames.append(m[96:576, 16:656])
    frames = np.ascontiguousarray(np.stack(frames))
    p = sample_camera_params(4, 1.0, 77, device="cuda")
    oc = camera_aug(torch.from_numpy(frames).cuda(), p, NSEED, impl="cuda").cpu().numpy()
    res["frames_480x640_severity1"] = _err(oc, _refs(frames, p.to_dicts(), NSEED))
    # uniform tiles (blank glass / background): finite and in range
    uni = {}
    for level in (0, 255):
        arr = np.full((8, 224, 224, 3), level, np.uint8)
        p = sample_camera_params(8, 1.0, level + 1, device="cuda")
        r = {}
        for impl in ("cuda", "torch"):
            out, inter = _camera_aug_impl(torch.from_numpy(arr).cuda(), p, 3, impl=impl, return_float=True)
            r[impl] = {"finite": bool(torch.isfinite(inter).all()), "inter_min": float(inter.min()),
                       "inter_max": float(inter.max()), **_err(out.cpu().numpy(), _refs(arr, p.to_dicts(), 3))}
        uni[str(level)] = r
    res["uniform_tiles_severity1"] = uni
    # JPEG model vs libjpeg (PIL, 4:4:4) and quality 100
    jp = {}
    for q in (100, 95, 75, 50, 25, 10):
        d = [dict(co.identity_params(), jpeg_quality=float(q)) for _ in range(8)]
        mine = camera_aug(x8, CameraAugParams.from_dicts(d, "cuda"), 0, impl="cuda").cpu().numpy().astype(int)
        pil = []
        for im in arr8:
            b = io.BytesIO()
            Image.fromarray(im).save(b, "JPEG", quality=q, subsampling=0)
            pil.append(np.asarray(Image.open(b)))
        pil = np.stack(pil).astype(int)
        jp[str(q)] = {"k2_vs_input_mae": float(np.abs(mine - arr8).mean()), "k2_vs_input_max": int(np.abs(mine - arr8).max()),
                      "k2_vs_input_frac_within1": float((np.abs(mine - arr8) <= 1).mean()),
                      "libjpeg444_vs_input_mae": float(np.abs(pil - arr8).mean()),
                      "k2_vs_libjpeg444_mae": float(np.abs(mine - pil).mean())}
    res["jpeg_vs_libjpeg444"] = {"note": "PIL JPEG (libjpeg integer DCT, quality=q, subsampling=0) on the 8 fixture "
                                         "tiles; K2 = CUDA kernel with only JPEG on", "by_quality": jp}
    return res


# ------------------------------------------------------------------------------------------------ figure
def figure() -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from make_crc_fixture import load_crc_val
    tiles16, labels = load_crc_val(16, seed=0)
    assert np.array_equal(tiles16, np.load(P / "tests" / "fixtures" / "crc16.npy"))
    pick = [3, 4, 5, 6, 7, 8]
    tiles = np.ascontiguousarray(tiles16[pick])
    names = [CLASSES[int(labels[i])] for i in pick]
    sevs = [0.0, 0.33, 0.67, 1.0]
    impl = "cuda" if load_k2_ext() is not None else "torch"
    x = torch.from_numpy(tiles).cuda()
    rows, rec = [], []
    for s in sevs:
        p = sample_camera_params(6, s, seed=7, device="cuda")
        rows.append(camera_aug(x, p, seed=7, impl=impl).cpu().numpy())
        rec.append({"severity": s, "params": p.to_dicts()})
    fig, axs = plt.subplots(4, 6, figsize=(10.0, 7.6), dpi=150)
    for r, (s, row) in enumerate(zip(sevs, rows)):
        for c in range(6):
            ax = axs[r, c]
            ax.imshow(row[c])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(names[c], fontsize=10)
            if c == 0:
                ax.set_ylabel("input (severity 0)" if s == 0 else f"severity {s:.2f}", fontsize=9.5)
            if s > 0:
                d = rec[r]["params"][c]
                ax.text(0.02, 0.03, f"q{int(d['jpeg_quality'])} σ{d['blur_sigma']:.1f}", transform=ax.transAxes,
                        fontsize=7, color="white", bbox=dict(facecolor="black", alpha=0.55, pad=1.2, lw=0))
    fig.suptitle("K2: simulated camera-of-screen augmentation (CUDA kernel output on real CRC-VAL tiles)",
                 fontsize=12, y=0.985)
    fig.text(0.5, 0.012, "SIMULATION of a USB camera filming a screen: warp, HED stain jitter, display gamma + colour "
             "profile, moiré, glare, defocus, white balance, sensor noise, JPEG (4:4:4).\n"
             "Corner label: JPEG quality and defocus σ (px). Same random draws per column, scaled by severity. "
             "Research prototype, not a diagnostic device.", ha="center", fontsize=7.5)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.925, bottom=0.07, wspace=0.03, hspace=0.04)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(FIG).size
    meta = {"figure": str(FIG.relative_to(P)), "size_px": [w, h], "impl": impl, "tiles": "CRC_VAL_HE_7K fixture "
            "tests/fixtures/crc16.npy indices " + str(pick), "classes": names, "param_seed": 7, "noise_seed": 7,
            "rows": rec}
    print("figure", w, h, impl, flush=True)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["cpu", "gpu", "parity", "figure"], required=True)
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(4)
    if a.part == "cpu":
        tiles, src = load_tiles(BATCH)
        _merge(OUT / "k2_bench.json", "cpu", {"env": env_block(False), "inputs": src, **bench_cpu(tiles, a.repeats)})
    elif a.part == "gpu":
        assert load_k2_ext() is not None, "K2 CUDA extension unavailable"
        tiles, src = load_tiles(BATCH)
        _merge(OUT / "k2_bench.json", "gpu", {"env": env_block(True), "inputs": src, **bench_gpu(tiles)})
    elif a.part == "parity":
        assert load_k2_ext() is not None, "K2 CUDA extension unavailable"
        tiles, src = load_tiles(64, seed=2)
        _merge(OUT / "k2_parity.json", "parity", {"env": env_block(True), "inputs_composition": src, **parity(tiles)})
    else:
        _merge(OUT / "k2_figure.json", "figure", {"env": env_block(True), **figure()})


if __name__ == "__main__":
    main()
