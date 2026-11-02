#!/usr/bin/env python
"""Write the device app's preset demo slides from the CRC test set, into data/demo_slides (git-ignored, like all data).

Slides are drawn at random per class with a fixed seed, and never by looking at model output, so a run through the deck
shows ordinary behaviour, including wrong calls and review flags. Needs numpy and cv2 only, so it runs on the board.

usage: make_demo_slides.py [--data data] [--out data/demo_slides] [--per-class 2] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CLASSES = ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")  # order of crc_test_y.npy and the bundles
ROOT = Path(__file__).resolve().parents[2]


def pick(y: np.ndarray, per_class: int, seed: int) -> list:
    """Tile indices: one round through the classes per slide-per-class, each round in its own random order."""
    rng = np.random.default_rng(seed)
    chosen = {k: rng.choice(np.flatnonzero(y == k), per_class, replace=False) for k in range(len(CLASSES))}
    return [int(chosen[k][r]) for r in range(per_class) for k in rng.permutation(len(CLASSES))]


def write_deck(x, y, out: Path, per_class: int, seed: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    slides = []
    for n, tile in enumerate(pick(y, per_class, seed)):
        truth = CLASSES[int(y[tile])]
        name = f"{n:02d}_{truth}_{tile:04d}.png"
        cv2.imwrite(str(out / name), np.asarray(x[tile])[..., ::-1])  # the arrays are RGB
        slides.append({"file": name, "truth": truth, "tile": tile})
    manifest = {"source": "CRC-VAL-HE-7K test set (data/crc_test_x.npy)", "seed": seed, "per_class": per_class,
                "selection": "random per class with a fixed seed, not filtered by model output", "slides": slides}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "demo_slides")
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    m = write_deck(np.load(a.data / "crc_test_x.npy", mmap_mode="r"), np.load(a.data / "crc_test_y.npy"), a.out,
                   a.per_class, a.seed)
    print(f"{len(m['slides'])} slides in {a.out}")


if __name__ == "__main__":
    main()
