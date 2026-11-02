#!/usr/bin/env python
"""Analyses added after seeing the registered result. None of this can change the ship rule, and everything here is labelled exploratory.

1. Per-situation paired tests over all seeds, since the registered test pools situations and no_tissue prompts repeat.
2. The registered test without no_tissue, whose frames are constructed and whose prompts are few.
3. How much of the base/finetuned gap sits in prompts that also appear in training, per situation.
4. Re-scoring with the checker as corrected after the run: two false failures found by reading the failures.
   results/llm/finetune/registered/explain_check.py is the frozen file that produced evaluation.json.
Writes results/llm/finetune/exploratory.json.
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "device"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explain_prompt as ep  # noqa: E402
from evaluate import binom_two_sided, signflip, wilson  # noqa: E402
from generate import load_decisions  # noqa: E402

RES = REPO / "results/llm/finetune"
SITUATIONS = ("agree", "review", "disagree", "no_tissue")
SEEDS = ("1", "1001", "2001")


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


frozen = module(RES / "registered/explain_check.py", "explain_check_registered")
fixed = module(REPO / "device/explain_check.py", "explain_check_fixed")
frozen_prompt = module(RES / "registered/explain_prompt.py", "explain_prompt_registered")
dec = load_decisions()


def score(arm, checker):
    out = {}
    for r in map(json.loads, open(RES / f"heldout/{arm}.jsonl")):
        # Always the facts the answer was generated from. The corrected prompt only moves tissue 9.5 to 9.99 % from 10 to 9.
        facts = frozen_prompt.facts_from_snapshot(dec[r["source"]])
        c = checker.check(r["text"], facts)
        out[r["id"]] = {"situation": r["situation"], "audience": r["audience"], "passed": c["passed"], "invented": c["invented_claim"], "checks": c["checks"], "forbidden": c["details"]["forbidden"],
                        "unknown_numbers": c["details"]["unknown_numbers"], "missing_numbers": c["details"]["missing_numbers"], "head_issues": c["details"]["head_issues"],
                        "user_turn": r["user_turn"], "text": r["text"]}
    return out


def mcn(a, b):
    a_only = sum(a[i]["passed"] and not b[i]["passed"] for i in a)
    b_only = sum(b[i]["passed"] and not a[i]["passed"] for i in a)
    return {"n": len(a), "base_passed": sum(a[i]["passed"] for i in a), "finetuned_passed": sum(b[i]["passed"] for i in a), "base_only": a_only, "finetuned_only": b_only,
            "p_exact": round(binom_two_sided(a_only, a_only + b_only), 6)}


out = {"label": "Exploratory: added after the registered result was known."}
rng = random.Random(20261102)
natural = json.loads((RES / "natural_rates.json").read_text())["situation_rates"]
train_turns = {json.loads(line)["messages"][1]["content"] for line in open(REPO / "data/llm/finetune/train.jsonl")}
base = {s: score(f"base_seed{s}", frozen) for s in SEEDS}
ft = {s: score(f"finetuned_seed{s}", frozen) for s in SEEDS}

per = {}
for sit in SITUATIONS:
    ids = [i for i, r in base["1"].items() if r["situation"] == sit]
    diffs = [sum(ft[s][i]["passed"] - base[s][i]["passed"] for s in SEEDS) / len(SEEDS) for i in ids]
    per[sit] = {"n_items": len(ids), "base_pass_by_seed": [round(sum(base[s][i]["passed"] for i in ids) / len(ids), 3) for s in SEEDS],
                "finetuned_pass_by_seed": [round(sum(ft[s][i]["passed"] for i in ids) / len(ids), 3) for s in SEEDS],
                "mean_difference_all_seeds": round(sum(diffs) / len(diffs), 4), "signflip_p_all_seeds": signflip(diffs, rng),
                "distinct_prompts": len({base["1"][i]["user_turn"] for i in ids}), "prompts_also_in_training": sum(base["1"][i]["user_turn"] in train_turns for i in ids)}
out["per_situation_registered_checker"] = per

no_nt = [i for i, r in base["1"].items() if r["situation"] != "no_tissue"]
out["without_no_tissue_seed1"] = mcn({i: base["1"][i] for i in no_nt}, {i: ft["1"][i] for i in no_nt})
diffs = [sum(ft[s][i]["passed"] - base[s][i]["passed"] for s in SEEDS) / len(SEEDS) for i in no_nt]
out["without_no_tissue_all_seeds"] = {"mean_difference": round(sum(diffs) / len(diffs), 4), "signflip_p": signflip(diffs, rng)}
seen_ids = [i for i, r in base["1"].items() if r["user_turn"] in train_turns]
out["prompts_also_in_training_seed1"] = mcn({i: base["1"][i] for i in seen_ids}, {i: ft["1"][i] for i in seen_ids})

# Re-score everything with the corrected checker.
re = {}
for arm in [f"base_seed{s}" for s in SEEDS] + [f"finetuned_seed{s}" for s in SEEDS] + ["composer_seed1", "realbase_seed1"]:
    old, new = score(arm, frozen), score(arm, fixed)
    n = len(old)
    bys = {s: round(sum(new[i]["passed"] for i in new if new[i]["situation"] == s) / sum(new[i]["situation"] == s for i in new), 3) for s in SITUATIONS}
    re[arm] = {"registered_passed": sum(r["passed"] for r in old.values()), "corrected_passed": sum(r["passed"] for r in new.values()), "n": n, "corrected_pass_rate": round(sum(r["passed"] for r in new.values()) / n, 4),
               "corrected_ci95": wilson(sum(r["passed"] for r in new.values()), n), "registered_invented": sum(r["invented"] for r in old.values()), "corrected_invented": sum(r["invented"] for r in new.values()),
               "corrected_by_situation": bys, "corrected_natural_mix": round(sum(natural[s] * bys[s] for s in SITUATIONS), 4), "passed_changed_items": sorted(i for i in old if old[i]["passed"] != new[i]["passed"])}
out["corrected_checker"] = re

# What kinds of error each arm makes, over all its runs, with the corrected checker.
kinds = {}
for name, arms_ in (("base", [f"base_seed{s}" for s in SEEDS]), ("finetuned", [f"finetuned_seed{s}" for s in SEEDS]), ("composer", ["composer_seed1"])):
    rows = [r for a in arms_ for r in score(a, fixed).values()]
    kinds[name] = {"answers": len(rows), "wrong_number": sum(bool(r["unknown_numbers"]) for r in rows), "number_left_out": sum(bool(r["missing_numbers"]) for r in rows),
                   "number_or_class_on_wrong_head": sum(bool(r["head_issues"]) for r in rows), "says_what_image_shows": sum("image_claim" in r["forbidden"] for r in rows),
                   "flag_status_wrong_or_missing": sum(not r["checks"]["status_correct"] for r in rows), "class_wrong": sum(not (r["checks"]["class_correct"] and r["checks"]["no_wrong_class"]) for r in rows),
                   "wrong_number_examples": [{"text": r["text"]} for r in rows if r["unknown_numbers"]][:4]}
out["error_kinds_all_runs_corrected_checker"] = kinds
out["corrected_checker_tests"] = {f"seed{s}": mcn(score(f"base_seed{s}", fixed), score(f"finetuned_seed{s}", fixed)) for s in SEEDS}
(RES / "exploratory.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
