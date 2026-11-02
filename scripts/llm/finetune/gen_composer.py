#!/usr/bin/env python
"""The deterministic control arm: composer.py on a split, in the same JSONL shape as generate.py. Seed = base seed + item position."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "device"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import composer  # noqa: E402
import explain_check as ec  # noqa: E402
import explain_prompt as ep  # noqa: E402
from generate import items  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="held_out")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", required=True)
a = ap.parse_args()
t0 = time.perf_counter()
rows = []
for it in items(a.split):
    facts = ep.facts_from_snapshot(it["snapshot"])
    text = composer.compose(facts, it["audience"], random.Random(a.seed + it["position"]))
    chk = ec.check(text, facts)
    rows.append({"id": it["id"], "source": it["source"], "situation": it["situation"], "family": it["family"], "audience": it["audience"], "seed": a.seed + it["position"],
                 "text": text, "finish_reason": "composer", "usage": {"completion_tokens": 0}, "timings": None, "passed": chk["passed"], "invented_claim": chk["invented_claim"],
                 "checks": chk["checks"], "details": chk["details"], "user_turn": ep.user_turn(facts, it["audience"])})
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"{len(rows)} answers in {(time.perf_counter() - t0) * 1000:.0f} ms -> {a.out}")
