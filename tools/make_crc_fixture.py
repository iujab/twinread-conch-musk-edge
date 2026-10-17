"""Write tests/fixtures/crc16.npy: 16 real NCT-CRC-HE CRC_VAL_HE_7K tiles (uint8 [16,224,224,3]).

Reads the 1aurent/NCT-CRC-HE CRC_VAL shards from the local HF cache (pyarrow), takes tiles round-robin over the
9 tissue classes with a fixed seed, so the fixture covers every class.
  python -I tools/make_crc_fixture.py tests/fixtures/crc16.npy [--n 16] [--seed 0]
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os

import numpy as np


def crc_val_shards() -> list[str]:
    root = os.path.expanduser("~/.cache/huggingface/hub/datasets--1aurent--NCT-CRC-HE/snapshots")
    fs = sorted(glob.glob(os.path.join(root, "*", "data", "CRC_VAL_HE_7K-*.parquet")))
    if not fs:
        raise FileNotFoundError("CRC_VAL_HE_7K parquet shards not in the HF cache")
    return fs


def load_crc_val(n: int, seed: int = 0, balanced: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """n tiles (uint8 [n,224,224,3]) and their labels from CRC_VAL_HE_7K."""
    import pyarrow.parquet as pq
    from PIL import Image
    labels, refs = [], []
    for f in crc_val_shards():
        lab = pq.read_table(f, columns=["label"]).column("label").to_pylist()
        labels += lab
        refs += [(f, i) for i in range(len(lab))]
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    if balanced:
        pools = [rng.permutation(np.flatnonzero(labels == c)).tolist() for c in sorted(set(labels.tolist()))]
        pick = []
        while len(pick) < n:
            for p in pools:
                if p and len(pick) < n:
                    pick.append(p.pop())
    else:
        pick = rng.choice(len(labels), size=n, replace=False).tolist()
    by_file: dict[str, list[int]] = {}
    for j in pick:
        by_file.setdefault(refs[j][0], []).append(refs[j][1])
    imgs = {}
    for f, rows in by_file.items():
        col = pq.read_table(f, columns=["image"]).column("image")
        for r in rows:
            b = col[r].as_py()["bytes"]
            imgs[(f, r)] = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8)
    arr = np.stack([imgs[refs[j]] for j in pick])
    return arr, labels[pick]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    arr, lab = load_crc_val(a.n, a.seed)
    assert arr.shape == (a.n, 224, 224, 3) and arr.dtype == np.uint8, arr.shape
    np.save(a.out, arr)
    print(json.dumps({"out": a.out, "shape": list(arr.shape), "labels": lab.tolist()}))


if __name__ == "__main__":
    main()
