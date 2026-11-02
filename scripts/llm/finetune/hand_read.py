#!/usr/bin/env python
"""Hand-read sample. `make`: print items in random order with no arm name or checker verdict, and keep the key separately.
`score LABELS.json`: join my labels {index: {"correct": bool, "note": str}} with the key and the checker. Writes results/llm/finetune/hand_read.json."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "device"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load  # noqa: E402
from generate import load_decisions  # noqa: E402

RES = REPO / "results/llm/finetune"
KEY = REPO / "data/llm/finetune/hand_read_key.json"
ARMS = ["base_seed1", "finetuned_seed1", "composer_seed1"]
PER_ARM_PER_SITUATION = 5

if sys.argv[1] == "make":
    arms = {a: {r["id"]: r for r in map(json.loads, open(RES / f"heldout/{a}.jsonl"))} for a in ARMS}
    rng = random.Random(20261103)
    picks = []
    for a in ARMS:
        for sit in ("agree", "review", "disagree", "no_tissue"):
            ids = sorted(i for i, r in arms[a].items() if r["situation"] == sit)
            picks += [(a, i) for i in rng.sample(ids, PER_ARM_PER_SITUATION)]
    rng.shuffle(picks)
    KEY.write_text(json.dumps(picks))
    for n, (a, i) in enumerate(picks):
        r = arms[a][i]
        print(f"[{n}] {r['user_turn']}\n=> {r['text']}\n")
else:
    labels = json.loads(Path(sys.argv[2]).read_text())
    picks = json.loads(KEY.read_text())
    dec = load_decisions()
    arms = {a: load(RES / f"heldout/{a}.jsonl", dec) for a in ARMS}
    rows = []
    for n, (a, i) in enumerate(picks):
        r = arms[a][i]
        lab = labels[str(n)]
        rows.append({"n": n, "arm": a, "id": i, "situation": r["situation"], "audience": r["audience"], "text": r["text"], "hand_correct": lab["correct"], "note": lab.get("note", ""),
                     "checker_passed": r["passed"], "failed_checks": [k for k, v in r["checks"].items() if not v]})
    summary = {}
    for a in ARMS:
        q = [x for x in rows if x["arm"] == a]
        summary[a] = {"n": len(q), "hand_correct": sum(x["hand_correct"] for x in q), "checker_passed": sum(x["checker_passed"] for x in q),
                      "checker_pass_hand_wrong": sum(x["checker_passed"] and not x["hand_correct"] for x in q),
                      "checker_fail_hand_right": sum((not x["checker_passed"]) and x["hand_correct"] for x in q)}
    (RES / "hand_read.json").write_text(json.dumps({"method": "Read in random order without arm names or checker verdicts; the writing style can still reveal the arm.", "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
