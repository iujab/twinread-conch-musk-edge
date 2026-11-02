#!/usr/bin/env python
"""How often the device's call is right under each status it can show, on all CRC-VAL-HE-7K test tiles.

Reads the int8 bundles/cm decisions cached by scripts/llm/finetune (run through device/twinread_runtime.py EdgeModel on the
PC) and writes results/uno_q/status_accuracy.json. Plain accuracy of the fused call, not the README's balanced accuracy.

usage: status_accuracy.py [--decisions data/llm/finetune/decisions.jsonl] [--out results/uno_q/status_accuracy.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarise(rows: list) -> dict:
    natural = [r for r in rows if r["family"] == "crc_val_7k"]
    def group(keep):
        hits = [r["call"]["label"] == r["truth_not_for_llm"] for r in natural if keep(r)]
        return {"tiles": len(hits), "share_of_all_tiles": round(len(hits) / len(natural), 4),
                "right": sum(hits), "wrong": len(hits) - sum(hits),
                "accuracy": round(sum(hits) / len(hits), 4) if hits else None}
    confident = group(lambda r: r["status"] == "agree")
    review = group(lambda r: r["status"] == "review")
    wrong = confident["wrong"] + review["wrong"]
    return {"source": "bundles/cm int8 through EdgeModel on the PC, CRC-VAL-HE-7K; not measured on camera frames",
            "tiles": len(natural),
            "confident (status agree)": confident,
            "needs_review (status review)": review,
            "needs_review, heads named different classes": group(
                lambda r: r["status"] == "review" and r["situation"] == "disagree"),
            "no_tissue_tiles": sum(r["status"] == "no_tissue" for r in natural),
            "wrong_calls": wrong,
            "share_of_wrong_calls_flagged_needs_review": round(review["wrong"] / wrong, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", default="data/llm/finetune/decisions.jsonl")
    ap.add_argument("--out", default="results/uno_q/status_accuracy.json")
    a = ap.parse_args()
    rows = [json.loads(line) for line in Path(a.decisions).read_text().splitlines()]
    out = summarise(rows)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
