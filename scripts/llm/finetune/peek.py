#!/usr/bin/env python
"""Summarise a generate.py output: pass rate, each check, and failures to read."""
from __future__ import annotations

import argparse
import collections
import json

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--show", type=int, default=0)
ap.add_argument("--fails", type=int, default=0)
a = ap.parse_args()
for path in a.files:
    rows = [json.loads(line) for line in open(path)]
    n = len(rows)
    per = collections.defaultdict(list)
    fails = collections.Counter()
    for r in rows:
        per[r["situation"]].append(r["passed"])
        for k, v in r["checks"].items():
            fails[k] += not v
    words = sorted(r["details"]["words"] for r in rows)
    gen = sum(r["usage"]["completion_tokens"] for r in rows) / n
    print(f"{path}: n={n} pass={sum(r['passed'] for r in rows) / n:.3f} invented={sum(r['invented_claim'] for r in rows) / n:.3f} "
          f"words median={words[n // 2]} min={words[0]} max={words[-1]} gen_tokens={gen:.0f}")
    print("  by situation", {k: f"{sum(v) / len(v):.2f}" for k, v in sorted(per.items())})
    print("  failed checks", dict(fails.most_common()))
    for r in rows[:a.show]:
        print(f"--- {r['id']} {r['situation']} {r['audience']} passed={r['passed']}\n{r['user_turn']}\n=> {r['text']}")
    shown = 0
    for r in rows:
        if not r["passed"] and shown < a.fails:
            shown += 1
            bad = [k for k, v in r["checks"].items() if not v]
            print(f"--- FAIL {r['id']} {r['situation']} {r['audience']} {bad} {r['details']}\n{r['user_turn']}\n=> {r['text']}")
