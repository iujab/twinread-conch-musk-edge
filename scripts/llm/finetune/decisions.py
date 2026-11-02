#!/usr/bin/env python
"""Real TwinRead decisions for the explainer dataset: the int8 bundles/cm through device/twinread_runtime.py on the host CPU.

Every record has exactly the shape uno_q_app.py publishes in /state (status, call, reasons, tissue_fraction, thresholds).
Sources:
  * CRC-VAL-HE-7K tiles (data/crc_test_x.npy), read as the app reads a 224 px tile;
  * no_tissue frames sent through crop_from_frame + infer, the app's own path. No real camera capture of a blank scene
    exists on this PC and the camera stays closed, so these frames are constructed: white glass, a grey screen, dark frames,
    uniform pink, real tiles dimmed to near black, and real tiles covering a small part of a white frame.

Writes data/llm/finetune/decisions.jsonl (numeric outputs of a CC-BY-NC-ND bundle: never distribute) and
results/llm/finetune/natural_rates.json. Truth labels are stored only for bookkeeping and are never given to the LLM.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "device"))
import explain_prompt as ep  # noqa: E402
import twinread_runtime as rt  # noqa: E402

CLASS_NAMES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]


def record(model, r, tile_rgb, source, truth=None, family=None):
    snap = ep.snapshot_from_result(r, model.classes, tissue_fraction=rt.tissue_fraction(tile_rgb), thresholds=model.thr)
    return {"source": source, "family": family, "truth_not_for_llm": truth, "situation": ep.situation(snap), **snap}


def blank_frames(x, seed):
    """Yield (family, source_id, uint8 frame). Frames go through crop_from_frame, so sizes differ from 224 on purpose."""
    rng = np.random.default_rng(seed)
    hw = (480, 640)

    def noisy(level, sigma, color=None):
        base = np.full((*hw, 3), level, np.float32) if color is None else np.broadcast_to(np.array(color, np.float32), (*hw, 3))
        return np.clip(base + rng.normal(0, sigma, (*hw, 3)), 0, 255).astype(np.uint8)
    for i in range(70):
        yield "white_glass", f"white_{i}", noisy(rng.uniform(236, 254), rng.uniform(0, 3))
    for i in range(70):
        yield "grey_screen", f"grey_{i}", noisy(rng.uniform(190, 235), rng.uniform(1, 4))
    for i in range(70):
        yield "dark", f"dark_{i}", noisy(rng.uniform(0, 22), rng.uniform(0, 3))
    for i in range(40):
        yield "uniform_pink", f"pink_{i}", noisy(0, rng.uniform(0, 2), color=[rng.uniform(200, 235), rng.uniform(120, 170), rng.uniform(170, 210)])
    # Real stained tiles, dimmed until almost nothing passes the tissue test.
    idx = rng.choice(len(x), 70, replace=False)
    for i, k in enumerate(idx):
        yield "dimmed_real_tile", f"dim_{i}_of_{int(k)}", (np.asarray(x[k], np.float32) * rng.uniform(0.04, 0.11)).astype(np.uint8)
    # Real tiles covering 1 to 9 % of a white frame: the tile is mostly out of the square.
    idx = rng.choice(len(x), 120, replace=False)
    for i, k in enumerate(idx):
        side = int(224 * np.sqrt(rng.uniform(0.01, 0.09)))
        f = noisy(245, 1.5)[:224, :224]
        y0, x0 = rng.integers(0, 224 - side, 2)
        f[y0:y0 + side, x0:x0 + side] = np.asarray(x[k])[:side, :side]
        yield "partial_real_tile", f"part_{i}_of_{int(k)}", f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default=str(REPO / "data/crc_test_x.npy"))
    ap.add_argument("--labels", default=str(REPO / "data/crc_test_y.npy"))
    ap.add_argument("--bundle", default=str(REPO / "bundles/cm"))
    ap.add_argument("--out", default=str(REPO / "data/llm/finetune/decisions.jsonl"))
    ap.add_argument("--rates", default=str(REPO / "results/llm/finetune/natural_rates.json"))
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    x = np.load(a.tiles, mmap_mode="r")
    y = np.load(a.labels, mmap_mode="r")
    m = rt.EdgeModel(a.bundle, threads=4)
    t0 = time.time()
    rows = []
    for i in range(0, len(x), a.batch):
        batch = np.asarray(x[i:i + a.batch])
        for j, r in enumerate(m.infer(batch)):
            rows.append(record(m, r, batch[j], f"crc_test_{i + j}", truth=CLASS_NAMES[int(y[i + j])], family="crc_val_7k"))
    natural = collections.Counter(r["situation"] for r in rows)
    print("natural", dict(natural), f"{time.time() - t0:.0f} s", flush=True)

    extra = collections.Counter()
    for fam, sid, frame in blank_frames(x, a.seed):
        tile = rt.crop_from_frame(frame, "center")
        r = m.infer(tile[None])[0]
        rec = record(m, r, tile, sid, family=fam)
        rows.append(rec)
        extra[(fam, rec["situation"])] += 1
    print("constructed", {f"{k[0]}/{k[1]}": v for k, v in sorted(extra.items())}, flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    nat = [r for r in rows if r["family"] == "crc_val_7k"]
    n = len(nat)
    reasons = collections.Counter(tuple(r["reasons"]) for r in nat)
    fused_vs_heads = collections.Counter()
    for r in nat:
        c = r["call"]
        if c["labelA"] == c["labelB"] and c["labelA"] != c["label"]:
            fused_vs_heads["both_heads_same_but_fused_differs"] += 1
        if c["labelA"] != c["labelB"] and c["label"] not in (c["labelA"], c["labelB"]):
            fused_vs_heads["heads_differ_and_fused_is_neither"] += 1
    rates = {
        "source": "bundles/cm int8 through device/twinread_runtime.py EdgeModel on host CPU, CRC-VAL-HE-7K",
        "n_tiles": n,
        "situation_counts": dict(natural),
        "situation_rates": {k: round(v / n, 5) for k, v in natural.items()},
        "reason_combinations": {"+".join(k) or "(none)": v for k, v in reasons.most_common()},
        "fused_special_cases": dict(fused_vs_heads),
        "thresholds": m.thr,
        "constructed_no_tissue_frames": {f"{k[0]}/{k[1]}": v for k, v in sorted(extra.items())},
        "note": "Constructed frames are not natural data and are not part of these rates. They were built on this PC, not captured by the board camera.",
    }
    Path(a.rates).parent.mkdir(parents=True, exist_ok=True)
    Path(a.rates).write_text(json.dumps(rates, indent=1))
    print(json.dumps(rates, indent=1))


if __name__ == "__main__":
    main()
