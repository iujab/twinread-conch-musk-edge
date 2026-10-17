"""Resumable, content-keyed embedding cache.

Layout: ``<data>/emb/<dataset>/<reader>_<kind>_<split>.npy`` (float16) + ``.json`` sidecar holding the cache key.
The key hashes reader name, kind, full PreprocessSpec, preprocessing impl, array shape and a digest of the first and
last 64 tiles, so a changed reader/kind/preprocessing/input silently never reuses a stale file.
Work is done in chunks of 2,048 tiles written to ``<out>.parts/``; once the time budget is used the call raises
``NeedsRerun`` (CLI exit code 3) so each locked GPU job stays short and fair.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import PIL.Image
import torch

from twinread.readers import READER_NAMES, Reader, PreprocessSpec, load_reader, spec_of

PAIRS = {"CM": ("conch", "musk"), "OPEN": ("pathgenclip", None)}
CHUNK = 2048
DEFAULT_BUDGET_S = 540.0
EXIT_RERUN = 3
ROOT = Path(__file__).resolve().parents[2]


class NeedsRerun(Exception):
    """Time budget used up with chunks remaining; rerun the same command (exit code 3)."""


N_STRIDED = 256


def _sample_idx(n: int) -> np.ndarray:
    """First 64, last 64 and >= 256 evenly spaced tile indices (all tiles when n is small)."""
    idx = np.concatenate([np.arange(min(64, n)), np.linspace(0, n - 1, min(n, N_STRIDED)).astype(np.int64),
                          np.arange(max(0, n - 64), n)]) if n else np.zeros(0, np.int64)
    return np.unique(idx)


def cache_key(reader_name: str, kind: str, spec: PreprocessSpec, impl: str, x: np.ndarray) -> str:
    h = hashlib.sha256()
    edge = hashlib.sha256(np.ascontiguousarray(x[_sample_idx(len(x))]).tobytes()).hexdigest()
    h.update(repr((reader_name, kind, dataclasses.astuple(spec), impl, tuple(x.shape), edge)).encode())
    return h.hexdigest()


def _prep_reference(reader: Reader, xb: np.ndarray) -> torch.Tensor:
    return torch.stack([reader.preprocess_pil(PIL.Image.fromarray(np.asarray(t))) for t in xb])


def _prep_k1(reader: Reader, xb: np.ndarray) -> torch.Tensor:
    try:
        from twinread.kernels import dual_preprocess
    except Exception as e:  # module missing (not built yet) or extension failed to build
        raise RuntimeError(f"impl='k1' needs twinread.kernels.dual_preprocess (unavailable: {e!r}); "
                           "use impl='reference'") from e
    # NOTE: single-reader call; the shared one-upload pass for a pair is dual_preprocess(x, specA, specB).
    a, _ = dual_preprocess(torch.from_numpy(np.ascontiguousarray(xb)).cuda(), reader.spec, reader.spec, impl="k1")
    return a


def _encode(reader: Reader, kind: str, xb: np.ndarray, impl: str, bs: int) -> np.ndarray:
    fn = reader.encode_aligned if kind == "aligned" else reader.encode_probe
    prep = _prep_k1 if impl == "k1" else _prep_reference
    outs = []
    i = 0
    while i < len(xb):
        try:
            outs.append(fn(prep(reader, xb[i:i + bs])))
            i += bs
        except torch.cuda.OutOfMemoryError:
            if bs == 1:
                raise
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
            warnings.warn(f"CUDA OOM; halving batch to {bs}")
    return np.concatenate(outs).astype(np.float32)


def embed_array(reader: Reader, x: np.ndarray, kind: str, out_path: Path, batch: int = 16,
                impl: str = "reference", chunk: int = CHUNK, budget_s: float | None = None) -> np.ndarray:
    """Embed uint8 tiles ``x`` [N,H,W,3]; saved float16, returned float32. Cached and resumable."""
    if kind not in ("aligned", "probe"):
        raise ValueError(kind)
    if impl not in ("reference", "k1"):
        raise ValueError(impl)
    out_path = Path(out_path)
    side = out_path.with_suffix(".json")
    key = cache_key(reader.name, kind, reader.spec, impl, x)
    if out_path.exists() and side.exists():
        try:
            if json.loads(side.read_text()).get("key") == key:
                return np.load(out_path).astype(np.float32)
        except (ValueError, OSError):
            pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dim = reader.dim_aligned if kind == "aligned" else reader.dim_probe
    if len(x) == 0:
        res = np.zeros((0, dim), np.float32)
    else:
        parts = out_path.parent / (out_path.name + ".parts")
        kf = parts / "key.txt"
        if parts.exists() and (not kf.exists() or kf.read_text() != key):
            shutil.rmtree(parts)  # parts from a different reader/spec/impl/data: discard
        parts.mkdir(exist_ok=True)
        kf.write_text(key)
        t0, n_ch = time.time(), -(-len(x) // chunk)
        for c in range(n_ch):
            pf = parts / f"{c:05d}.part.npy"
            if pf.exists():
                continue
            if budget_s is not None and c > 0 and time.time() - t0 >= budget_s:
                raise NeedsRerun(f"{out_path.name}: {c}/{n_ch} chunks done")
            e = _encode(reader, kind, x[c * chunk:(c + 1) * chunk], impl, batch).astype(np.float16)
            np.save(str(pf) + ".tmp.npy", e)
            Path(str(pf) + ".tmp.npy").replace(pf)
        res = np.concatenate([np.load(parts / f"{c:05d}.part.npy") for c in range(n_ch)]).astype(np.float32)
        shutil.rmtree(parts)
    tmp = out_path.with_name(out_path.stem + ".tmp.npy")
    np.save(tmp, res.astype(np.float16))
    tmp.replace(out_path)
    side.write_text(json.dumps({"key": key, "reader": reader.name, "kind": kind, "impl": impl,
                                "shape": list(res.shape), "n_tiles": int(len(x)),
                                "spec": dataclasses.asdict(reader.spec)}))
    return res.astype(np.float16).astype(np.float32)


def resolve_readers(pair: str | None, readers: list | None) -> list:
    if readers:
        for r in readers:
            if r not in READER_NAMES:
                raise KeyError(f"unknown reader {r!r}; choose from {READER_NAMES}")
        return list(readers)
    if pair not in PAIRS:
        raise KeyError(f"unknown pair {pair!r}; choose from {tuple(PAIRS)}")
    return [r for r in PAIRS[pair] if r]


def embed_dataset(pair: str | None, dataset: str, kinds=("aligned", "probe"), impl: str = "reference",
                  readers: list | None = None, splits=("train", "val", "test"), data_dir: Path | None = None,
                  budget_s: float = DEFAULT_BUDGET_S, batch: int = 16, device: str = "cuda",
                  limit: int | None = None) -> dict:
    """Embed every (reader, kind, split). Returns a summary; ``done`` False means rerun (exit code 3)."""
    from twinread.data import load_split
    data_dir = Path(data_dir) if data_dir else ROOT / "data"
    out_dir = data_dir / "emb" / dataset
    names = resolve_readers(pair, readers)
    t0 = time.time()
    summary = {"dataset": dataset, "readers": names, "impl": impl, "files": {}, "done": True}
    xs = {s: load_split(data_dir, dataset, s) for s in splits}
    for name in names:
        reader = load_reader(name, device)
        try:
            for kind in kinds:
                for s in splits:
                    x = xs[s][0][:limit] if limit else xs[s][0]
                    suffix = f"_{s}" if not limit else f"_{s}_n{limit}"
                    p = out_dir / f"{name}_{kind}{suffix}.npy"
                    left = budget_s - (time.time() - t0)
                    ts = time.time()
                    try:
                        e = embed_array(reader, x, kind, p, batch=batch, impl=impl, budget_s=max(left, 0))
                    except NeedsRerun as ex:
                        summary["done"] = False
                        summary["stopped_at"] = str(ex)
                        return summary
                    summary["files"][p.name] = {"n": int(len(e)), "seconds": round(time.time() - ts, 2)}
        finally:
            reader.close()
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Embed a dataset with a reader pair (resumable; exit 3 = rerun).")
    ap.add_argument("--pair", choices=list(PAIRS))
    ap.add_argument("--readers", nargs="+")
    ap.add_argument("--dataset", choices=["crc", "cam"], required=True)
    ap.add_argument("--impl", choices=["reference", "k1"], default="reference")
    ap.add_argument("--kinds", nargs="+", default=["aligned", "probe"])
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET_S)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, help="smoke runs: first N tiles per split")
    a = ap.parse_args(argv)
    if not (a.pair or a.readers):
        ap.error("give --pair or --readers")
    s = embed_dataset(a.pair, a.dataset, tuple(a.kinds), a.impl, a.readers, tuple(a.splits), None, a.budget,
                      a.batch, a.device, a.limit)
    print(json.dumps(s, indent=1))
    return 0 if s["done"] else EXIT_RERUN


if __name__ == "__main__":
    sys.exit(main())
