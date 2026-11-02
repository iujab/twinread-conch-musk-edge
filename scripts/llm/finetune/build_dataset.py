#!/usr/bin/env python
"""Build the fine-tune dataset: train and val tiles only, targets from composer.py only. Every target must pass explain_check.

Situation-balanced by repeating rare situations with new phrasings, since only the phrasing varies for a given tile. The fine-tune
never sees a held-out or edge tile, and no target comes from any LLM. Numeric results are from bundles/cm (CC-BY-NC-ND): do not
distribute this data or the weights trained on it without the owner's decision. Writes data/llm/finetune/{train,val}.jsonl and
results/llm/finetune/dataset_stats.json.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "device"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import composer  # noqa: E402
import explain_check as ec  # noqa: E402
import explain_prompt as ep  # noqa: E402

SEED = 20261102


def build(sources, dec, per_situation, rng):
    bysit = collections.defaultdict(list)
    for s in sources:
        bysit[dec[s]["situation"]].append(s)
    rows = []
    for sit, srcs in sorted(bysit.items()):
        srcs = sorted(srcs)
        rng.shuffle(srcs)
        for i in range(per_situation):
            r = dec[srcs[i % len(srcs)]]
            snap = {k: r[k] for k in ("status", "call", "reasons", "tissue_fraction", "thresholds")}
            aud = ep.AUDIENCES[i % 2]
            facts = ep.facts_from_snapshot(snap)
            text = composer.compose(facts, aud, rng)
            assert ec.check(text, facts)["passed"], text
            rows.append({"source": r["source"], "situation": sit, "audience": aud,
                         "messages": ep.build_messages(snap, aud) + [{"role": "assistant", "content": text}]})
    rng.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-situation", type=int, default=2000)
    ap.add_argument("--val-per-situation", type=int, default=100)
    ap.add_argument("--out", default=str(REPO / "data/llm/finetune"))
    a = ap.parse_args()
    split = json.loads((REPO / "results/llm/finetune/split.json").read_text())
    dec = {r["source"]: r for r in map(json.loads, open(REPO / "data/llm/finetune/decisions.jsonl"))}
    rng = random.Random(SEED)
    train = build(split["train"], dec, a.per_situation, rng)
    val = build(split["val"], dec, a.val_per_situation, rng)
    held = set(split["held_out"]) | set(split["edge"])
    assert not held & {r["source"] for r in train + val}
    out = Path(a.out)
    for name, rows in (("train", train), ("val", val)):
        with (out / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    def distinct(rows):
        return len({r["messages"][1]["content"] for r in rows})
    def distinct_text(rows):
        return len({r["messages"][2]["content"] for r in rows})
    held_turns = set()
    for s in split["held_out"]:
        r = dec[s]
        f = ep.facts_from_snapshot(r)
        held_turns |= {ep.user_turn(f, aud) for aud in ep.AUDIENCES}
    train_turns = {r["messages"][1]["content"] for r in train}
    words = sorted(len(r["messages"][2]["content"].split()) for r in train)
    stats = {
        "seed": SEED, "train_rows": len(train), "val_rows": len(val),
        "train_by_situation": dict(collections.Counter(r["situation"] for r in train)),
        "train_tiles": len({r["source"] for r in train}), "train_tiles_by_situation": {k: len({r["source"] for r in train if r["situation"] == k}) for k in ep.SITUATIONS},
        "distinct_user_turns_train": distinct(train), "distinct_targets_train": distinct_text(train),
        "target_words_min_median_max": [words[0], words[len(words) // 2], words[-1]],
        "held_out_user_turns_both_audiences": len(held_turns),
        "held_out_user_turns_also_in_train": len(held_turns & train_turns),
        "prompt_version": ep.PROMPT_VERSION, "system_sha256": hashlib.sha256(ep.SYSTEM.encode()).hexdigest(),
        "explain_prompt_sha256": hashlib.sha256((REPO / "device/explain_prompt.py").read_bytes()).hexdigest(),
        "explain_check_sha256": hashlib.sha256((REPO / "device/explain_check.py").read_bytes()).hexdigest(),
        "composer_sha256": hashlib.sha256((Path(__file__).resolve().parent / "composer.py").read_bytes()).hexdigest(),
    }
    (REPO / "results/llm/finetune/dataset_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
