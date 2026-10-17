"""End-to-end dual-reader embedding throughput: CONCH + MUSK aligned embeddings of the same tiles, impl="reference"
(official PIL transforms on the CPU, reader by reader) vs impl="k1" (one dual_preprocess launch per batch feeds both
readers). Both go through the library entry point ``twinread.embed.embed_arrays`` exactly as ``embed_dataset`` calls
it (batch 16, fresh cache dir, both readers resident, load time excluded).

Where the time goes (same tiles, same batch):
  * preprocess_reference_cpu - both readers' official PIL transforms, one process (what impl=reference does);
  * preprocess_k1_gpu        - uint8 H2D copy + one dual_preprocess launch per batch (what impl=k1 does);
  * forward_only             - both models' encode_aligned on inputs already on the GPU (the floor for any
                               preprocessing scheme); per reader as well;
  * reference_dataloader_4w  - an overlapped CPU baseline: a 4-worker DataLoader runs the official transforms while
                               the GPU encodes (not what the library does; reported so K1 is not compared only
                               against the serial path).
CONCH runs in fp32 and MUSK in fp16, as in the readers (organizer adapter semantics). Shared GPU and 16-core CPU:
every method is repeated (median kept, all runs recorded). Research prototype, not a diagnostic device.

  flock $B/.gpu.lock python bench/bench_k1_e2e.py            # ~3 min -> results/kernels/k1_embed_e2e.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / "src"))
sys.path.insert(0, str(P / "bench"))

from bench_k1 import env_block  # noqa: E402
from twinread import embed  # noqa: E402
from twinread.data import load_split  # noqa: E402
from twinread.kernels import dual_preprocess  # noqa: E402
from twinread.readers import load_reader  # noqa: E402

OUT = P / "results" / "kernels" / "k1_embed_e2e.json"
BATCH = 16


def _sync():
    torch.cuda.synchronize()


def _timed(fn, repeats: int) -> dict:
    ts = []
    for _ in range(repeats):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append(time.perf_counter() - t0)
    return {"seconds_median": float(np.median(ts)), "seconds_all": [round(t, 4) for t in ts]}


def _rate(d: dict, n: int) -> dict:
    return {**d, "tiles_per_s": n / d["seconds_median"]}


class _DS(torch.utils.data.Dataset):
    def __init__(self, arr, tfa, tfb):
        self.arr, self.tfa, self.tfb = arr, tfa, tfb

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, i):
        from PIL import Image
        im = Image.fromarray(self.arr[i])
        return self.tfa(im), self.tfb(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        sys.exit("needs CUDA")
    free0 = torch.cuda.mem_get_info()[0]
    xt_all, yt_all = load_split(P / "data", "crc", "test")
    idx = np.sort(np.random.default_rng(0).choice(len(xt_all), a.n, replace=False))
    x = np.ascontiguousarray(xt_all[idx])
    del xt_all
    n = len(x)

    t0 = time.perf_counter()
    conch = load_reader("conch", "cuda")
    t_conch = time.perf_counter() - t0
    t0 = time.perf_counter()
    musk = load_reader("musk", "cuda")
    t_musk = time.perf_counter() - t0
    mem_models = torch.cuda.memory_allocated()
    readers = [conch, musk]
    res: dict = {}
    work = Path(tempfile.mkdtemp(prefix="k1e2e_", dir=P / "build"))
    try:
        # warmup (cudnn/cublas autotune, extension load, PIL)
        for impl in ("reference", "k1"):
            embed.embed_arrays(readers, x[:32], "aligned", [work / f"w_{impl}_a.npy", work / f"w_{impl}_b.npy"],
                               batch=BATCH, impl=impl, device="cuda")
        backend = embed._k1_backend("cuda")

        # ---- end to end through the library, alternating order; fresh output paths each run (no cache hits)
        e2e = {"reference": [], "k1": []}
        outs = {}
        for rep in range(a.repeats):
            for impl in (("reference", "k1") if rep % 2 == 0 else ("k1", "reference")):
                ps = [work / f"{impl}_{rep}_conch.npy", work / f"{impl}_{rep}_musk.npy"]
                _sync()
                t0 = time.perf_counter()
                outs[impl] = embed.embed_arrays(readers, x, "aligned", ps, batch=BATCH, impl=impl, device="cuda")
                _sync()
                e2e[impl].append(time.perf_counter() - t0)
        for impl, ts in e2e.items():
            res[f"e2e_{impl}"] = _rate({"seconds_median": float(np.median(ts)),
                                        "seconds_all": [round(t, 4) for t in ts]}, n)
        cos = {}
        for j, r in enumerate(("conch", "musk")):
            a_, b_ = outs["k1"][j], outs["reference"][j]
            c = (a_ * b_).sum(1) / np.linalg.norm(a_, axis=1) / np.linalg.norm(b_, axis=1)
            cos[r] = {"min": float(c.min()), "mean": float(c.mean())}
        res["k1_vs_reference_cosine_aligned"] = cos

        # ---- breakdown
        def prep_ref():
            for i in range(0, n, BATCH):
                for r in readers:
                    embed._prep_reference(r, x[i:i + BATCH])
        res["preprocess_reference_cpu"] = _rate(_timed(prep_ref, a.repeats), n)

        def prep_k1():
            for i in range(0, n, BATCH):
                xt = torch.from_numpy(np.ascontiguousarray(x[i:i + BATCH])).to("cuda")
                dual_preprocess(xt, conch.spec, musk.spec, impl="cuda")
        res["preprocess_k1_gpu"] = _rate(_timed(prep_k1, a.repeats), n)

        staged = []  # fp16 on the GPU (~1 GB for 512 tiles); cast per batch exactly as the k1 path does
        for i in range(0, n, BATCH):
            staged.append(dual_preprocess(torch.from_numpy(np.ascontiguousarray(x[i:i + BATCH])).cuda(), conch.spec,
                                          musk.spec, impl="cuda"))

        def fwd(which):
            def f():
                for ta, tb in staged:
                    if which in ("both", "conch"):
                        conch.encode_aligned(ta.float())
                    if which in ("both", "musk"):
                        musk.encode_aligned(tb.float())
            return f
        for which in ("both", "conch", "musk"):
            res[f"forward_only_{which}"] = _rate(_timed(fwd(which), a.repeats), n)
        del staged
        torch.cuda.empty_cache()

        # ---- overlapped CPU baseline: 4 DataLoader workers run the official transforms while the GPU encodes
        dl = torch.utils.data.DataLoader(_DS(x, conch.preprocess_pil, musk.preprocess_pil), batch_size=BATCH,
                                         num_workers=4, pin_memory=True, persistent_workers=True)

        def dl_run():
            for ta, tb in dl:
                conch.encode_aligned(ta.cuda(non_blocking=True))
                musk.encode_aligned(tb.cuda(non_blocking=True))
        dl_run()  # spin up the workers
        res["reference_dataloader_4w"] = _rate(_timed(dl_run, a.repeats), n)
        del dl
    finally:
        for r in readers:
            r.close()
        import shutil
        shutil.rmtree(work, ignore_errors=True)

    ref, k1, fwd_b = (res["e2e_reference"]["seconds_median"], res["e2e_k1"]["seconds_median"],
                      res["forward_only_both"]["seconds_median"])
    pr, pk = res["preprocess_reference_cpu"]["seconds_median"], res["preprocess_k1_gpu"]["seconds_median"]
    summary = {
        "e2e_tiles_per_s": {"reference": n / ref, "k1": n / k1},
        "e2e_speedup_k1_vs_reference": ref / k1,
        "e2e_speedup_k1_vs_dataloader_4w": res["reference_dataloader_4w"]["seconds_median"] / k1,
        "forward_only_tiles_per_s": n / fwd_b,
        "share_of_reference_e2e": {"preprocess_cpu": pr / ref, "model_forward": fwd_b / ref},
        "share_of_k1_e2e": {"preprocess_k1": pk / k1, "model_forward": fwd_b / k1},
        "preprocess_speedup_k1_vs_pil": pr / pk,
        "max_e2e_speedup_if_preprocessing_were_free": ref / fwd_b,
        "k1_backend": backend,
    }
    doc = {
        "what": "CONCH + MUSK aligned embeddings of the same tiles; one 'tile' = both readers' embeddings",
        "inputs": {"source": "data/crc_test_x.npy (CRC-VAL-HE-7K, test split)", "n": n,
                   "selection": "np.random.default_rng(0).choice(7180, n, replace=False), sorted",
                   "class_counts": np.bincount(yt_all[idx], minlength=9).tolist(), "shape": list(x.shape)},
        "config": {"batch": BATCH, "repeats": a.repeats, "kind": "aligned",
                   "readers": {"conch": "fp32, 448/448 bicubic", "musk": "fp16, 384/384 bicubic"},
                   "e2e": "twinread.embed.embed_arrays([conch, musk], x, 'aligned', ...) wall time incl. fp16 .npy "
                          "writes; both readers resident; model load excluded; fresh output paths (no cache hits)",
                   "timing": "time.perf_counter around torch.cuda.synchronize(); median of repeats, all runs kept",
                   "dataloader_baseline": "4 workers, pin_memory, persistent workers (second pass timed)"},
        "model_load_seconds": {"conch": round(t_conch, 2), "musk": round(t_musk, 2)},
        "gpu_mem": {"free_at_start_bytes": int(free0), "allocated_after_both_models_bytes": int(mem_models)},
        "results": res, "summary": summary,
        "note": ("Shared GPU/CPU (other agents); estimate. K1 removes the CPU preprocessing from the serial path; the "
                 "model forward (CONCH fp32 ViT-B/16 at 448 px + MUSK fp16 ViT-L at 384 px) dominates, which bounds "
                 "the end-to-end gain (see max_e2e_speedup_if_preprocessing_were_free). Research prototype."),
        "env": env_block(True),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2))
    print(json.dumps(summary, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
