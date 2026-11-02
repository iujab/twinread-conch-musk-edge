#!/usr/bin/env python
"""The pre-registered comparison. Reads arm outputs from results/llm/finetune/heldout/ and writes results/llm/finetune/evaluation.json.

Scores are recomputed here from the stored text with the current explain_check.py, whose sha256 is compared with the pre-registered one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "device"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explain_check as ec  # noqa: E402
import explain_prompt as ep  # noqa: E402
from generate import load_decisions  # noqa: E402

RES = REPO / "results/llm/finetune"
SITUATIONS = ("agree", "review", "disagree", "no_tissue")


def binom_two_sided(k, n):
    """Exact two-sided sign test at p=0.5."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson(k, n, z=1.959964):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def load(path, dec):
    rows = {}
    for r in map(json.loads, open(path)):
        facts = ep.facts_from_snapshot(dec[r["source"]])
        c = ec.check(r["text"], facts)
        rows[r["id"]] = {**r, "passed": c["passed"], "invented_claim": c["invented_claim"], "checks": c["checks"], "details": c["details"]}
    return rows


def summary(rows, natural):
    n = len(rows)
    out = {"n": n, "passed": sum(r["passed"] for r in rows.values()), "pass_rate": round(sum(r["passed"] for r in rows.values()) / n, 4),
           "pass_ci95": wilson(sum(r["passed"] for r in rows.values()), n), "invented_claims": sum(r["invented_claim"] for r in rows.values()),
           "invented_rate": round(sum(r["invented_claim"] for r in rows.values()) / n, 4),
           "words_median": sorted(r["details"]["words"] for r in rows.values())[n // 2],
           "words_min_max": [min(r["details"]["words"] for r in rows.values()), max(r["details"]["words"] for r in rows.values())]}
    out["by_situation"] = {}
    for s in SITUATIONS:
        rs = [r for r in rows.values() if r["situation"] == s]
        if rs:
            out["by_situation"][s] = {"n": len(rs), "pass_rate": round(sum(r["passed"] for r in rs) / len(rs), 4), "invented_rate": round(sum(r["invented_claim"] for r in rs) / len(rs), 4)}
    out["by_audience"] = {}
    for a in ep.AUDIENCES:
        rs = [r for r in rows.values() if r["audience"] == a]
        if rs:
            out["by_audience"][a] = {"n": len(rs), "pass_rate": round(sum(r["passed"] for r in rs) / len(rs), 4)}
    keys = next(iter(rows.values()))["checks"].keys()
    out["failures_by_check"] = {k: sum(not r["checks"][k] for r in rows.values()) for k in keys}
    cats = {}
    for r in rows.values():
        for k in r["details"]["forbidden"]:
            cats[k] = cats.get(k, 0) + 1
    out["forbidden_by_category"] = cats
    if natural and all(s in out["by_situation"] for s in natural):
        out["natural_mix_pass_rate"] = round(sum(natural[s] * out["by_situation"][s]["pass_rate"] for s in natural), 4)
        out["natural_mix_invented_rate"] = round(sum(natural[s] * out["by_situation"][s]["invented_rate"] for s in natural), 4)
    return out


def mcnemar(a, b):
    """a and b are dicts of the same ids. b_only counts items where b passes and a fails."""
    ids = sorted(a)
    assert ids == sorted(b)
    a_only = sum(a[i]["passed"] and not b[i]["passed"] for i in ids)
    b_only = sum(b[i]["passed"] and not a[i]["passed"] for i in ids)
    return {"n": len(ids), "first_only": a_only, "second_only": b_only, "both": sum(a[i]["passed"] and b[i]["passed"] for i in ids),
            "neither": sum(not a[i]["passed"] and not b[i]["passed"] for i in ids), "p_exact_two_sided": round(binom_two_sided(a_only, a_only + b_only), 6)}


def signflip(diffs, rng, n=20000):
    """Two-sided sign-flip permutation test of the mean of paired differences."""
    diffs = [d for d in diffs if d != 0]
    if not diffs:
        return 1.0
    obs = abs(sum(diffs))
    hits = sum(abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= obs - 1e-12 for _ in range(n))
    return round((hits + 1) / (n + 1), 5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(RES / "heldout"))
    ap.add_argument("--out", default=str(RES / "evaluation.json"))
    a = ap.parse_args()
    d = Path(a.dir)
    dec = load_decisions()
    natural = json.loads((RES / "natural_rates.json").read_text())["situation_rates"]
    reg = json.loads((RES / "preregistered_hashes.json").read_text())
    now = {"device/explain_check.py": hashlib.sha256((REPO / "device/explain_check.py").read_bytes()).hexdigest(),
           "device/explain_prompt.py": hashlib.sha256((REPO / "device/explain_prompt.py").read_bytes()).hexdigest()}
    arms = {}
    for path in sorted(d.glob("*.jsonl")):
        arms[path.stem] = load(path, dec)
    out = {"files": {k: len(v) for k, v in arms.items()}, "checker_and_prompt_unchanged_since_registration": all(now[k] == reg[k] for k in now), "arms": {k: summary(v, natural) for k, v in arms.items()}}

    base, ft = arms["base_seed1"], arms["finetuned_seed1"]
    prim = mcnemar(base, ft)
    better = out["arms"]["finetuned_seed1"]["passed"] > out["arms"]["base_seed1"]["passed"]
    inv_ok = out["arms"]["finetuned_seed1"]["invented_claims"] <= out["arms"]["base_seed1"]["invented_claims"]
    out["primary"] = {"metric": "share of held-out answers passing every check", "base": out["arms"]["base_seed1"]["pass_rate"], "finetuned": out["arms"]["finetuned_seed1"]["pass_rate"],
                      "mcnemar_base_vs_finetuned": prim, "finetuned_higher": better, "significant_at_0.05": prim["p_exact_two_sided"] < 0.05,
                      "invented_not_raised": inv_ok, "ship_finetuned_by_rule": bool(better and prim["p_exact_two_sided"] < 0.05 and inv_ok)}

    # Secondary: other seeds; every seed pooled per item with a sign-flip test.
    seeds = sorted({k.rsplit("_seed", 1)[1] for k in arms if k.startswith("base_seed")})
    sec = {}
    for s in seeds:
        if f"finetuned_seed{s}" in arms:
            sec[f"seed{s}"] = {"mcnemar": mcnemar(arms[f"base_seed{s}"], arms[f"finetuned_seed{s}"]), "base": out["arms"][f"base_seed{s}"]["pass_rate"], "finetuned": out["arms"][f"finetuned_seed{s}"]["pass_rate"]}
    usable = [s for s in seeds if f"finetuned_seed{s}" in arms]
    rng = random.Random(20261102)
    diffs = [sum(arms[f"finetuned_seed{s}"][i]["passed"] - arms[f"base_seed{s}"][i]["passed"] for s in usable) / len(usable) for i in sorted(base)]
    sec["all_seeds_signflip_p"] = signflip(diffs, rng)
    sec["all_seeds_mean_difference"] = round(sum(diffs) / len(diffs), 4)
    if "realbase_seed1" in arms:
        sec["realbase_vs_t2file_same_prompts"] = mcnemar(arms["realbase_seed1"], base)
        sec["realbase_vs_finetuned"] = mcnemar(arms["realbase_seed1"], ft)

    # Secondary: only prompts whose user turn never appeared in training.
    train_turns = {json.loads(line)["messages"][1]["content"] for line in open(REPO / "data/llm/finetune/train.jsonl")}
    novel = [i for i in sorted(base) if base[i]["user_turn"] not in train_turns]
    sec["novel_prompts"] = {"n": len(novel), "of": len(base), "base_pass": sum(base[i]["passed"] for i in novel), "finetuned_pass": sum(ft[i]["passed"] for i in novel),
                            "mcnemar": mcnemar({i: base[i] for i in novel}, {i: ft[i] for i in novel})}
    distinct = {}
    for i in base:
        distinct.setdefault(base[i]["situation"], set()).add(base[i]["user_turn"])
    sec["distinct_prompts_by_situation"] = {k: len(v) for k, v in distinct.items()}
    out["secondary"] = sec
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"primary": out["primary"], "secondary": sec}, indent=1))
    for k, v in out["arms"].items():
        print(k, v["pass_rate"], v["invented_rate"], v.get("natural_mix_pass_rate"), v["by_situation"], v["failures_by_check"], v["forbidden_by_category"])


if __name__ == "__main__":
    main()
