#!/usr/bin/env python
"""Re-score every prompt-development run on the validation split with the frozen checker. Writes results/llm/finetune/prompt_dev_log.json."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "device"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load  # noqa: E402
from generate import load_decisions  # noqa: E402

NOTES = {"base_v1_rp11": "facts as Key: value lines; base echoes them as a list", "base_v2_rp11": "+ one flagged example in the system prompt, repeat_penalty 1.1", "base_v2_rp10": "same, repeat_penalty 1.0",
         "base_v3_rp11": "pattern with <placeholders>, repeat_penalty 1.1", "base_v3_rp10": "same, repeat_penalty 1.0", "base_v4_rp11": "two examples, repeat_penalty 1.1", "base_v4_rp10": "same, repeat_penalty 1.0",
         "base_v5_rp11": "facts as short sentences, repeat_penalty 1.1", "base_v5_rp10": "same, repeat_penalty 1.0", "base_v6_rp10_t0.3": "+ keep agree and flag, temperature 0.3", "base_v6_rp10_t0.1": "same, temperature 0.1",
         "base_v7_val_t0.1": "+ never say what the tile shows, temperature 0.1", "base_v7_val_t0.2": "same, temperature 0.2", "base_v7_val_t0.3": "same, temperature 0.3", "base_v8_val": "agree and flag in one sentence, temperature 0.3: frozen as PROMPT_VERSION 1",
         "smoke_val": "150-step smoke fine-tune, older prompt", "ft1_val": "1000-step fine-tune, frozen prompt: the registered candidate"}
dec = load_decisions()
out = {}
for k, note in NOTES.items():
    p = REPO / f"data/llm/finetune/dev/{k}.jsonl"
    if not p.exists():
        continue
    rows = load(p, dec)
    n = len(rows)
    fails = {}
    for r in rows.values():
        for c, v in r["checks"].items():
            if not v:
                fails[c] = fails.get(c, 0) + 1
    out[k] = {"note": note, "n": n, "pass_rate": round(sum(r["passed"] for r in rows.values()) / n, 3), "invented_rate": round(sum(r["invented_claim"] for r in rows.values()) / n, 3), "failures_by_check": fails}
(REPO / "results/llm/finetune/prompt_dev_log.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(f"{k:22} n={v['n']:3} pass={v['pass_rate']:.3f} invented={v['invented_rate']:.3f}  {v['note']}")
