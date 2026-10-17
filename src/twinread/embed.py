"""Resumable, content-keyed embedding cache.

Layout: ``<data>/emb/<dataset>/<reader>_<kind>_<split>.npy`` (float16) + ``.json`` sidecar holding the cache key.
The key hashes reader name, kind, full PreprocessSpec, preprocessing impl, array shape and a digest of the first and
last 64 tiles, so a changed reader/kind/preprocessing/input silently never reuses a stale file.
Work is done in chunks of 2,048 tiles written to ``<out>.parts/``; once the time budget is used the call raises
``NeedsRerun`` (CLI exit code 3) so each locked GPU job stays short and fair.

``impl="k1"`` (R12/R23): readers are embedded in pairs (the two readers of ``--pair``, or consecutive ``--readers``).
Both readers of a pair stay resident and every encode batch is preprocessed ONCE for both of them by a single
``kernels.dual_preprocess`` launch (uint8 upload once, both readers' fp16 inputs out), so the raw tiles are read
once per pair. A lone reader uses ``dual_preprocess(x, spec, spec)`` and keeps the first output. If the pair does not
fit in GPU memory it falls back, with a warning, to one reader at a time. ``impl="reference"`` keeps the official
PIL transforms on the CPU and loads one reader at a time. The cache key includes ``impl``, so k1 and reference caches
never mix (same file names: switching impl recomputes); the k1 sidecar also records the K1 backend that ran
(cuda | triton | torch, all bit-exact to ``official(img).half()``) and the readers preprocessed together.
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


def _k1_dual():
    """``twinread.kernels.dual_preprocess`` or a clear RuntimeError when the kernels module cannot be imported."""
    try:
        from twinread.kernels import dual_preprocess
    except Exception as e:  # module missing (not built yet) or extension failed to build
        raise RuntimeError(f"impl='k1' needs twinread.kernels.dual_preprocess (unavailable: {e!r}); "
                           "use impl='reference'") from e
    return dual_preprocess


def _k1_backend(device: str) -> str:
    """The K1 implementation ``dual_preprocess(impl="cuda")`` resolves to for inputs on ``device`` (warns once on a
    downgrade: CPU input or missing extension -> triton / torch)."""
    from twinread.kernels import resolve_impl
    return resolve_impl("cuda", torch.empty((1, 1, 1, 3), dtype=torch.uint8, device=device))


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _encode_group(readers: list, kind: str, xb: np.ndarray, impl: str, bs: int, device: str) -> list:
    """Embeddings of tiles ``xb`` for every reader in ``readers`` (float32, one array per reader).

    reference: each reader's official PIL transform on the CPU, reader by reader. k1: one ``dual_preprocess`` launch
    per batch of ``bs`` tiles produces the inputs of BOTH readers (a lone reader passes its spec twice); the fp16
    outputs are cast to float32 (exact) so every reader sees the same dtype as on the reference path."""
    fns = [r.encode_aligned if kind == "aligned" else r.encode_probe for r in readers]
    dual = _k1_dual() if impl == "k1" else None
    if dual is not None and len(readers) > 2:
        raise ValueError("impl='k1' preprocesses at most two readers per launch")
    outs = [[] for _ in readers]
    i = 0
    while i < len(xb):
        try:
            if dual is None:
                res = [fn(_prep_reference(r, xb[i:i + bs])) for r, fn in zip(readers, fns)]
            else:
                specs = [r.spec for r in readers] * (2 if len(readers) == 1 else 1)
                xt = torch.from_numpy(np.ascontiguousarray(xb[i:i + bs])).to(device)
                ins = dual(xt, specs[0], specs[1], impl="cuda")
                del xt
                res = [fn(t.float()) for fn, t in zip(fns, ins)]
                del ins
        except torch.cuda.OutOfMemoryError:
            if bs == 1:
                raise
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
            warnings.warn(f"CUDA OOM; halving batch to {bs}")
            continue
        for o, r in zip(outs, res):
            o.append(r)
        i += bs
    return [np.concatenate(o).astype(np.float32) for o in outs]


def _encode(reader: Reader, kind: str, xb: np.ndarray, impl: str, bs: int, device: str | None = None) -> np.ndarray:
    return _encode_group([reader], kind, xb, impl, bs, device or _default_device())[0]


def embed_arrays(readers: list, x: np.ndarray, kind: str, out_paths: list, batch: int = 16,
                 impl: str = "reference", chunk: int = CHUNK, budget_s: float | None = None,
                 device: str | None = None) -> list:
    """Embed uint8 tiles ``x`` [N,H,W,3] with one or more readers in one pass over the chunks; each saved float16
    to its ``out_paths`` entry, returned float32. Cached per reader and resumable per (reader, chunk).

    With ``impl="k1"`` (at most two readers) each encode batch is preprocessed once for all readers by one
    ``dual_preprocess`` launch on ``device`` (default cuda when available)."""
    if kind not in ("aligned", "probe"):
        raise ValueError(kind)
    if impl not in ("reference", "k1"):
        raise ValueError(impl)
    readers, out_paths = list(readers), [Path(p) for p in out_paths]
    if len(readers) != len(out_paths) or not readers:
        raise ValueError("need one out_path per reader")
    if impl == "k1":
        _k1_dual()  # fail fast with a clear error before any work
        if len(readers) > 2:
            raise ValueError("impl='k1' preprocesses at most two readers per launch")
    device = device or _default_device()
    keys = [cache_key(r.name, kind, r.spec, impl, x) for r in readers]
    res: list = [None] * len(readers)
    for j, (op, key) in enumerate(zip(out_paths, keys)):
        side = op.with_suffix(".json")
        if op.exists() and side.exists():
            try:
                if json.loads(side.read_text()).get("key") == key:
                    res[j] = np.load(op).astype(np.float32)
            except (ValueError, OSError):
                pass
    todo = [j for j in range(len(readers)) if res[j] is None]
    if not todo:
        return res
    for j in todo:
        out_paths[j].parent.mkdir(parents=True, exist_ok=True)
    if len(x) == 0:
        for j in todo:
            r = readers[j]
            res[j] = np.zeros((0, r.dim_aligned if kind == "aligned" else r.dim_probe), np.float32)
    else:
        parts = {j: out_paths[j].parent / (out_paths[j].name + ".parts") for j in todo}
        for j in todo:
            kf = parts[j] / "key.txt"
            if parts[j].exists() and (not kf.exists() or kf.read_text() != keys[j]):
                shutil.rmtree(parts[j])  # parts from a different reader/spec/impl/data: discard
            parts[j].mkdir(exist_ok=True)
            kf.write_text(keys[j])
        t0, n_ch = time.time(), -(-len(x) // chunk)
        for c in range(n_ch):
            need = [j for j in todo if not (parts[j] / f"{c:05d}.part.npy").exists()]
            if not need:
                continue
            if budget_s is not None and c > 0 and time.time() - t0 >= budget_s:
                names = ",".join(out_paths[j].name for j in need)
                raise NeedsRerun(f"{names}: {c}/{n_ch} chunks done")
            es = _encode_group([readers[j] for j in need], kind, x[c * chunk:(c + 1) * chunk], impl, batch, device)
            for j, e in zip(need, es):
                pf = parts[j] / f"{c:05d}.part.npy"
                np.save(str(pf) + ".tmp.npy", e.astype(np.float16))
                Path(str(pf) + ".tmp.npy").replace(pf)
        for j in todo:
            res[j] = np.concatenate([np.load(parts[j] / f"{c:05d}.part.npy") for c in range(n_ch)]).astype(np.float32)
            shutil.rmtree(parts[j])
    extra = {"k1_backend": _k1_backend(device), "k1_group": [r.name for r in readers]} if impl == "k1" else {}
    for j in todo:
        r, op = readers[j], out_paths[j]
        tmp = op.with_name(op.stem + ".tmp.npy")
        np.save(tmp, res[j].astype(np.float16))
        tmp.replace(op)
        op.with_suffix(".json").write_text(json.dumps({
            "key": keys[j], "reader": r.name, "kind": kind, "impl": impl, "shape": list(res[j].shape),
            "n_tiles": int(len(x)), "spec": dataclasses.asdict(r.spec), **extra}))
        res[j] = res[j].astype(np.float16).astype(np.float32)
    return res


def embed_array(reader: Reader, x: np.ndarray, kind: str, out_path: Path, batch: int = 16,
                impl: str = "reference", chunk: int = CHUNK, budget_s: float | None = None,
                device: str | None = None) -> np.ndarray:
    """Embed uint8 tiles ``x`` [N,H,W,3]; saved float16, returned float32. Cached and resumable."""
    return embed_arrays([reader], x, kind, [out_path], batch, impl, chunk, budget_s, device)[0]


def resolve_readers(pair: str | None, readers: list | None) -> list:
    if readers:
        for r in readers:
            if r not in READER_NAMES:
                raise KeyError(f"unknown reader {r!r}; choose from {READER_NAMES}")
        return list(readers)
    if pair not in PAIRS:
        raise KeyError(f"unknown pair {pair!r}; choose from {tuple(PAIRS)}")
    return [r for r in PAIRS[pair] if r]


def reader_groups(names: list, impl: str) -> list:
    """k1: consecutive pairs (both readers of a pair share each dual_preprocess launch); reference: one by one."""
    if impl == "k1":
        return [list(names[i:i + 2]) for i in range(0, len(names), 2)]
    return [[n] for n in names]


def _load_group(group: list, device: str) -> list | None:
    """Load every reader of ``group`` (resident together); None (all closed) on CUDA OOM while loading."""
    loaded = []
    try:
        for n in group:
            loaded.append(load_reader(n, device))
        return loaded
    except torch.cuda.OutOfMemoryError:
        for r in loaded:
            r.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None


def embed_dataset(pair: str | None, dataset: str, kinds=("aligned", "probe"), impl: str = "reference",
                  readers: list | None = None, splits=("train", "val", "test"), data_dir: Path | None = None,
                  budget_s: float = DEFAULT_BUDGET_S, batch: int = 16, device: str = "cuda",
                  limit: int | None = None) -> dict:
    """Embed every (reader, kind, split). Returns a summary; ``done`` False means rerun (exit code 3).

    impl="k1": the readers are embedded in pairs from one ``dual_preprocess`` launch per encode batch (see module
    docstring); a pair that does not fit in GPU memory falls back to one reader at a time (warned)."""
    from twinread.data import load_split
    data_dir = Path(data_dir) if data_dir else ROOT / "data"
    out_dir = data_dir / "emb" / dataset
    names = resolve_readers(pair, readers)
    t0 = time.time()
    summary = {"dataset": dataset, "readers": names, "impl": impl, "groups": [], "files": {}, "done": True}
    xs = {s: load_split(data_dir, dataset, s) for s in splits}
    queue = reader_groups(names, impl)
    while queue:
        group = queue.pop(0)
        loaded = _load_group(group, device)
        if loaded is None:
            if len(group) == 1:
                raise torch.cuda.OutOfMemoryError(f"CUDA OOM loading reader {group[0]}")
            warnings.warn(f"CUDA OOM loading readers {group} together; embedding them one at a time "
                          "(K1 then preprocesses each reader in its own launch)")
            queue[:0] = [[n] for n in group]
            continue
        summary["groups"].append(list(group))
        try:
            for kind in kinds:
                for s in splits:
                    x = xs[s][0][:limit] if limit else xs[s][0]
                    suffix = f"_{s}" if not limit else f"_{s}_n{limit}"
                    ps = [out_dir / f"{name}_{kind}{suffix}.npy" for name in group]
                    left = budget_s - (time.time() - t0)
                    ts = time.time()
                    try:
                        es = embed_arrays(loaded, x, kind, ps, batch=batch, impl=impl, budget_s=max(left, 0),
                                          device=device)
                    except NeedsRerun as ex:
                        summary["done"] = False
                        summary["stopped_at"] = str(ex)
                        return summary
                    for p, e in zip(ps, es):
                        summary["files"][p.name] = {"n": int(len(e)), "seconds": round(time.time() - ts, 2),
                                                    "group": list(group)}
        finally:
            for r in loaded:
                r.close()
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
