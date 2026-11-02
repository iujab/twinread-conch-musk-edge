#!/usr/bin/env python
"""Split decisions into train, validation and held-out BY TILE, before anything is composed. Writes results/llm/finetune/split.json.

Held-out is scenario-balanced: 50 tiles per situation, audience alternating, so the arms are compared on 200 prompts. A separate
small edge set holds tiles where the combined call differs from what the heads named; they are rare and are reported on their own.
A frame family (uniform pink) that did not trigger no_tissue is dropped and recorded. Natural-mix rates come from natural_rates.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SEED = 20261102
HELD_PER_SITUATION = 50
VAL_PER_SITUATION = {"agree": 60, "review": 60, "disagree": 15, "no_tissue": 40}
EDGE_FUSED_DIFFERS, EDGE_NEITHER = 12, 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", default=str(REPO / "data/llm/finetune/decisions.jsonl"))
    ap.add_argument("--out", default=str(REPO / "results/llm/finetune/split.json"))
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.decisions)]
    dropped = [r["source"] for r in rows if r["family"] == "uniform_pink"]
    rows = [r for r in rows if r["family"] != "uniform_pink"]
    rng = random.Random(SEED)

    def special(r):
        if r["situation"] == "no_tissue":
            return None
        c = r["call"]
        if c["labelA"] == c["labelB"] and c["labelA"] != c["label"]:
            return "fused_differs"
        if c["labelA"] != c["labelB"] and c["label"] not in (c["labelA"], c["labelB"]):
            return "fused_neither"
        return None

    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["situation"], special(r))].append(r["source"])
    for k in sorted(groups, key=str):
        groups[k].sort()
        rng.shuffle(groups[k])

    edge = []
    for sit in ("agree", "review", "disagree"):
        pool = groups[(sit, "fused_differs")]
        n = min(EDGE_FUSED_DIFFERS // 2, len(pool) // 2)
        edge += [pool.pop() for _ in range(n)]
    pool = groups[("disagree", "fused_neither")]
    edge += [pool.pop() for _ in range(min(EDGE_NEITHER, len(pool) // 2))]

    # Natural no_tissue tiles are few: a third goes to held-out, and constructed frames fill the rest, one family at a time.
    def take(sit, n, by_family=False):
        pool = groups[(sit, None)]
        if not by_family:
            out = pool[:n]
            del pool[:n]
            return out
        fam = {r["source"]: r["family"] for r in rows}
        byf = collections.defaultdict(list)
        for s in pool:
            byf[fam[s]].append(s)
        out = []
        fams = sorted(byf)
        i = 0
        natural = byf.get("crc_val_7k", [])
        for s in natural[:max(1, len(natural) // 3)]:
            out.append(s)
            byf["crc_val_7k"].remove(s)
        while len(out) < n:
            f = fams[i % len(fams)]
            if f != "crc_val_7k" and byf[f]:
                out.append(byf[f].pop())
            i += 1
        for s in out:
            pool.remove(s)
        return out

    held, val = [], []
    for sit in ("agree", "review", "disagree", "no_tissue"):
        held += take(sit, HELD_PER_SITUATION, by_family=sit == "no_tissue")
        val += take(sit, VAL_PER_SITUATION[sit], by_family=sit == "no_tissue")
    used = set(held) | set(val) | set(edge)
    train = sorted(s for s in (r["source"] for r in rows) if s not in used)
    assert not (set(train) & set(held)) and not (set(val) & set(held)) and not (set(edge) & set(held))
    by_sit = {r["source"]: r["situation"] for r in rows}
    summ = {name: dict(collections.Counter(by_sit[s] for s in part)) for name, part in
            [("train", train), ("val", val), ("held_out", held), ("edge", edge)]}
    out = {"seed": SEED, "rule": "split by tile before composing; held-out scenario-balanced", "counts": summ, "dropped_uniform_pink": dropped,
           "dropped_note": "Uniform pink frames were read as agree, not no_tissue: std over all pixels counts the channel differences. Not explanation data; see report.",
           "held_out": held, "val": val, "edge": edge, "train": train}
    Path(a.out).write_text(json.dumps(out))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
