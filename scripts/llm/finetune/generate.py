#!/usr/bin/env python
"""Generate one arm's answers on a split through a host llama-server, using device/explain_prompt.py and its sampling settings.

Each item has its own seed (base seed + item position), so repeated identical prompts are real replicates. Output is resumable JSONL.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "device"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explain_check as ec  # noqa: E402
import explain_prompt as ep  # noqa: E402
from serve import Server  # noqa: E402


def load_decisions():
    return {r["source"]: r for r in map(json.loads, open(REPO / "data/llm/finetune/decisions.jsonl"))}


def items(split_name, limit_per_situation=None):
    """The fixed item list of a split. Audience alternates within each situation."""
    split = json.loads((REPO / "results/llm/finetune/split.json").read_text())
    dec = load_decisions()
    seen = {}
    out = []
    for src in split[split_name]:
        r = dec[src]
        k = seen.get(r["situation"], 0)
        seen[r["situation"]] = k + 1
        if limit_per_situation and k >= limit_per_situation:
            continue
        out.append({"id": f"{split_name}_{len(out):04d}", "source": src, "situation": r["situation"], "family": r["family"],
                    "audience": ep.AUDIENCES[k % 2], "snapshot": {x: r[x] for x in ("status", "call", "reasons", "tissue_fraction", "thresholds")},
                    "position": len(out)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit-per-situation", type=int)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=1791)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--override", default="{}", help="JSON of sampling overrides, for tuning on val only")
    ap.add_argument("--log")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in open(out)}
    todo = [it for it in items(a.split, a.limit_per_situation) if it["id"] not in done]
    if not todo:
        print("nothing to do", out)
        return
    over = json.loads(a.override)
    t0 = time.time()
    with Server(a.model, port=a.port, threads=a.threads, log=a.log) as srv, out.open("a") as f:
        srv.chat(ep.prewarm_request())
        for it in todo:
            body = ep.chat_request(it["snapshot"], it["audience"], seed=a.seed + it["position"], stream=False)
            body.update(over)
            r = srv.chat(body)
            c = r["choices"][0]
            text = c["message"].get("content") or ""
            facts = ep.facts_from_snapshot(it["snapshot"])
            chk = ec.check(text, facts)
            f.write(json.dumps({"id": it["id"], "source": it["source"], "situation": it["situation"], "family": it["family"],
                                "audience": it["audience"], "seed": a.seed + it["position"], "text": text,
                                "finish_reason": c.get("finish_reason"), "usage": r.get("usage"), "timings": r.get("timings"),
                                "passed": chk["passed"], "invented_claim": chk["invented_claim"], "checks": chk["checks"],
                                "details": chk["details"], "user_turn": body["messages"][1]["content"]}) + "\n")
            f.flush()
    print(f"{len(todo)} answers in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
