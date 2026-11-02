#!/usr/bin/env python
"""Unblind the judge's answers and summarise them: ratings per arm, paired tests, best picks, and how often the judge's accuracy verdict
disagrees with explain_check, per arm. Writes results/llm/finetune/judge/judge_summary.json."""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "device"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import binom_two_sided, load, signflip  # noqa: E402
from generate import load_decisions  # noqa: E402

RES = REPO / "results/llm/finetune"
OUT = RES / "judge"
key = json.loads((OUT / "judge_key.json").read_text())
rows = [json.loads(line) for line in open(OUT / "judge_answers.jsonl")]
errors = [r["id"] for r in rows if "error" in r["answer"]]
rows = [r for r in rows if "error" not in r["answer"]]
dec = load_decisions()
arms = sorted(set(next(iter(key.values())).values()))
checked = {a: load(RES / f"heldout/{a}.jsonl", dec) for a in arms}

per = {a: {"readability": {}, "usefulness": {}, "accurate": {}} for a in arms}
best = collections.Counter()
best_by = collections.defaultdict(collections.Counter)
problems = collections.defaultdict(list)
for r in rows:
    for letter, arm in key[r["id"]].items():
        ans = r["answer"][letter]
        per[arm]["readability"][r["id"]] = float(ans["readability"])
        per[arm]["usefulness"][r["id"]] = float(ans["usefulness"])
        per[arm]["accurate"][r["id"]] = bool(ans["accurate"])
        if ans.get("problem"):
            problems[arm].append({"id": r["id"], "problem": ans["problem"]})
    b = key[r["id"]][r["answer"]["best"]]
    best[b] += 1
    best_by[r["audience"]][b] += 1
    best_by[r["situation"]][b] += 1

n = len(rows)
rng = random.Random(20261102)
out = {"items_rated": n, "errors": errors, "blinding": "Per-item random order, letters A/B/C, no arm names; judge not told a template or fine-tune exists. Sonnet via claude -p.",
       "arms": {}, "best_counts": dict(best), "best_by_group": {k: dict(v) for k, v in best_by.items()}, "pairwise": {}}
for a in arms:
    p = per[a]
    ids = sorted(p["accurate"])
    disagreements = [{"id": i, "checker_pass": checked[a][i]["passed"], "judge_accurate": p["accurate"][i]} for i in ids if checked[a][i]["passed"] != p["accurate"][i]]
    out["arms"][a] = {"readability_mean": round(sum(p["readability"].values()) / n, 3), "usefulness_mean": round(sum(p["usefulness"].values()) / n, 3),
                      "judge_accurate": sum(p["accurate"].values()), "judge_accurate_rate": round(sum(p["accurate"].values()) / n, 4),
                      "checker_pass_on_these_items": sum(checked[a][i]["passed"] for i in ids),
                      "checker_vs_judge_disagree": len(disagreements), "checker_pass_judge_says_wrong": sum(d["checker_pass"] for d in disagreements),
                      "checker_fail_judge_says_right": sum(not d["checker_pass"] for d in disagreements), "disagreement_rows": disagreements}
for i, a in enumerate(arms):
    for b in arms[i + 1:]:
        res = {}
        for metric in ("readability", "usefulness"):
            diffs = [per[b][metric][k] - per[a][metric][k] for k in sorted(per[a][metric])]
            res[metric] = {"mean_b_minus_a": round(sum(diffs) / len(diffs), 3), "b_higher": sum(d > 0 for d in diffs), "a_higher": sum(d < 0 for d in diffs), "signflip_p": signflip(diffs, rng)}
        a_ok, b_ok = per[a]["accurate"], per[b]["accurate"]
        a_only = sum(a_ok[k] and not b_ok[k] for k in a_ok)
        b_only = sum(b_ok[k] and not a_ok[k] for k in a_ok)
        res["judge_accurate_mcnemar"] = {"a_only": a_only, "b_only": b_only, "p_exact": round(binom_two_sided(a_only, a_only + b_only), 6)}
        out["pairwise"][f"{a}__vs__{b}"] = res
out["problem_notes"] = {k: v for k, v in problems.items()}
(OUT / "judge_summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "problem_notes"}, indent=1))
