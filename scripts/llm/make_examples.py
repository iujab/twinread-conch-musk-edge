#!/usr/bin/env python
"""Pick three real TwinRead results and build the explanation prompts used by every benchmark.

Runs the real EdgeModel on real NCT-CRC test tiles (data/crc_test_x.npy), so the numbers the LLM is asked to explain are
real model outputs: (1) a confident call both heads agree on, (2) heads agree but the review head flags it,
(3) the two heads disagree. Writes results/llm/example_results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "device"))
import twinread_runtime as rt  # noqa: E402

SYSTEM = (
    "You explain one result from TwinRead, a research prototype that classifies a colorectal H&E tissue tile. It is not a "
    "diagnostic device. Two model heads, reader A and reader B, each give a class and a confidence from 0 to 1. The fused "
    "value is the combined call. Classes: ADI adipose tissue; BACK background; DEB debris; LYM lymphocytes; MUC mucus; "
    "MUS smooth muscle; NORM normal colon mucosa; STR cancer-associated stroma; TUM colorectal adenocarcinoma epithelium. "
    "A tile is flagged for review when the heads disagree or when the review score is high. Write one short paragraph of "
    "about 100 words for a doctor. Say what the tile was called, how confident each head was, whether they agree, and "
    "whether a pathologist should look at it and why. Use only the numbers given. Do not give a diagnosis or treatment."
)


def compact(model, r, tile_id):
    kA, kB = int(np.argmax(r.probsA)), int(np.argmax(r.probsB))
    return {"tile": tile_id, "fused": {"class": r.label, "confidence": round(r.prob, 3)},
            "reader_A": {"class": model.classes[kA], "confidence": round(r.probsA[kA], 3)},
            "reader_B": {"class": model.classes[kB], "confidence": round(r.probsB[kB], 3)},
            "flagged_for_review": bool(r.review), "review_score": round(r.review_score, 3), "reasons": list(r.reasons)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default=str(REPO / "data/crc_test_x.npy"))
    ap.add_argument("--labels", default=str(REPO / "data/crc_test_y.npy"))
    ap.add_argument("--bundle", default=str(REPO / "bundles/cm"))
    ap.add_argument("--scan", type=int, default=7180, help="number of tiles to scan, stride-sampled over the test set")
    ap.add_argument("--out", default=str(REPO / "results/llm/example_results.json"))
    a = ap.parse_args()
    x = np.load(a.tiles, mmap_mode="r")
    y = np.load(a.labels, mmap_mode="r") if Path(a.labels).exists() else None
    m = rt.EdgeModel(a.bundle, threads=4)
    print("tiles", x.shape, x.dtype)
    agree = flagged = disagree = None
    n = min(a.scan, len(x))
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    for start in range(0, n, 64):
        ids = idx[start:start + 64]
        for i, r in zip(ids, m.infer(np.asarray(x[ids]))):
            kA, kB = int(np.argmax(r.probsA)), int(np.argmax(r.probsB))
            truth = str(y[i]) if y is not None else None
            rec = (int(i), r, truth)
            if agree is None and not r.review and r.label == "TUM" and kA == kB and r.prob > 0.9:
                agree = rec
            if flagged is None and r.review and kA == kB and r.reasons == ["review_head"] and r.label == "MUS":
                flagged = rec
            if disagree is None and kA != kB and "disagree" in r.reasons and not r.no_tissue:
                disagree = rec
        if agree and flagged and disagree:
            break
    out = {"system": SYSTEM, "source": "EdgeModel on bundles/cm, NCT-CRC CRC-VAL-HE-7K tiles", "examples": []}
    for name, rec in (("agree_confident", agree), ("flagged_review_head", flagged), ("heads_disagree", disagree)):
        i, r, truth = rec
        res = compact(m, r, f"crc_test_{i}")
        out["examples"].append({"name": name, "index": i, "truth_label_not_shown_to_llm": truth, "result": res,
                                "user": "Result: " + json.dumps(res, separators=(",", ":"))})
        print(name, truth, json.dumps(res))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
