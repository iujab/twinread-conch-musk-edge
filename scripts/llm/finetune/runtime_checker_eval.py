#!/usr/bin/env python
"""Score the stored answers with the runtime checker and with the registered one, writing results/llm/finetune/runtime_checker_eval.json.

This is exploratory. It does not change the pre-registered result, which used results/llm/finetune/registered/explain_check.py. It runs
no model: the text comes from the held-out files and each row's facts are rebuilt from data/llm/finetune/decisions.jsonl.
"""
from __future__ import annotations

import collections
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "device"))
import explain_check as new  # noqa: E402
import explain_prompt as ep  # noqa: E402

spec = importlib.util.spec_from_file_location("registered_check", ROOT / "results/llm/finetune/registered/explain_check.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def main():
    by_source = {}
    for line in (ROOT / "data/llm/finetune/decisions.jsonl").read_text().splitlines():
        r = json.loads(line)
        by_source[r["source"]] = r
    out, examples = {}, {}
    for path in sorted((ROOT / "results/llm/finetune/heldout").glob("*.jsonl")):
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        tally = collections.Counter()
        fails = collections.Counter()
        for r in rows:
            facts = ep.facts_from_snapshot(by_source[r["source"]])
            a, b = new.check(r["text"], facts), old.check(r["text"], facts)
            tally["n"] += 1
            tally["registered_passed_as_stored"] += bool(r["passed"])
            tally["registered_rescored_on_fixed_facts"] += b["passed"]
            tally["runtime_passed"] += a["passed"]
            tally["passed_registered_rejected_now"] += b["passed"] and not a["passed"]
            if b["passed"] and not a["passed"]:
                fails.update(k for k, v in a["checks"].items() if not v)
                examples.setdefault(path.stem, []).append(
                    {"source": r["source"], "failed": [k for k, v in a["checks"].items() if not v],
                     "outside": a["details"].get("words_outside_vocabulary"), "missing": a["details"].get("reasons_missing"),
                     "combined": a["details"].get("combined_stated"), "text": r["text"]})
        out[path.stem] = {**tally, "newly_failed_checks": dict(fails)}
    result = {"note": "Exploratory rescoring of stored answers; the pre-registered result is unchanged. The runtime checker's word list "
                      "comes from the fine-tune's training text, so the base arms fail it largely by construction: this file shows the "
                      "false-rejection rate for the shipped model and is NOT a comparison between arms. registered_rescored_on_fixed_facts "
                      "differs from the stored count where the 9.96% tissue fix changed the facts after registration.", "arms": out,
              "examples_newly_rejected": {k: v[:6] for k, v in examples.items()}}
    (ROOT / "results/llm/finetune/runtime_checker_eval.json").write_text(json.dumps(result, indent=1) + "\n")
    for k, v in out.items():
        print(k, v)
    for k in examples:
        for e in examples[k][:4]:
            print(k, e["failed"], e["outside"], e["missing"], e["combined"], "|", e["text"][:170])


if __name__ == "__main__":
    main()
