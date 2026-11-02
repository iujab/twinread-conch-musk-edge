#!/usr/bin/env python
"""Blind judge: claude -p --model sonnet rates three arms' answers per item, for evaluation only. Its output never enters a training set.

Blinding: every item gets its own random order and the texts are labelled A, B, C. The judge is told nothing about which arm wrote what,
that a template exists, or that a fine-tune exists. The mapping is written to judge_key.json before any call, in a file the prompt never
sees. Sample: 15 held-out items per situation, drawn with the split seed. Answers go to results/llm/finetune/judge/.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import random
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "device"))
import explain_prompt as ep  # noqa: E402

RES = REPO / "results/llm/finetune"
OUT = RES / "judge"
SEED = 20261102

PROMPT = """You are checking short explanations written for a tissue-tile classifier called TwinRead. It reads one small H&E image tile and has two heads, Head A and Head B, plus a combined call. Each explanation should state the model's reading of this ONE tile, in one paragraph, for the audience named. It must add nothing beyond the facts: no stage, grade, prognosis, treatment, patient details, claims about the whole slide, or claims about what the image shows. Disclaimers are not wanted.

Audience: {audience}

The facts each explanation had to convey:
{facts}

Candidate explanations:
{candidates}

Rate each candidate.
- accurate: true only if every statement agrees with the facts and nothing was invented or left out that the facts require (class, confidences, whether the heads agree, whether it is flagged and why).
- readability: 1 to 5, how clearly and naturally it reads for this audience (5 is excellent).
- usefulness: 1 to 5, how well it helps this audience understand this result (5 is excellent).
Then name the single best candidate.

Reply with JSON only, no other text, in this shape:
{{"A": {{"accurate": true, "readability": 3, "usefulness": 3, "problem": "short note or empty"}}, "B": {{...}}, "C": {{...}}, "best": "A"}}"""

AUD = {"doctor": "a doctor (histology terms are fine)", "user": "someone with no medical training (plain words are needed)"}


def call(prompt, timeout=240):
    r = subprocess.run(["claude", "-p", prompt, "--model", "sonnet", "--output-format", "json"], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout,
                       cwd="/tmp")
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:300] or r.stdout[:300])
    text = json.loads(r.stdout)["result"]
    return json.loads(re.search(r"\{.*\}", text, re.S).group(0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs=3, default=["base_seed1", "finetuned_seed1", "composer_seed1"])
    ap.add_argument("--per-situation", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    arms = {k: {r["id"]: r for r in map(json.loads, open(RES / f"heldout/{k}.jsonl"))} for k in a.arms}
    first = arms[a.arms[0]]
    rng = random.Random(SEED)
    bysit = collections.defaultdict(list)
    for i in sorted(first):
        bysit[first[i]["situation"]].append(i)
    chosen = []
    for s in sorted(bysit):
        chosen += rng.sample(bysit[s], a.per_situation)
    key_path = OUT / "judge_key.json"
    if key_path.exists():
        key = json.loads(key_path.read_text())
    else:
        key = {}
        for i in chosen:
            order = list(a.arms)
            rng.shuffle(order)
            key[i] = dict(zip("ABC", order))
        key_path.write_text(json.dumps(key, indent=1))
    ans_path = OUT / "judge_answers.jsonl"
    done = {json.loads(line)["id"] for line in open(ans_path)} if ans_path.exists() else set()

    def work(i):
        r = first[i]
        facts = "\n".join(line for line in r["user_turn"].split("\n") if not line.startswith(("Audience", "Facts:")))
        cands = "\n\n".join(f"{letter}: {arms[arm][i]['text']}" for letter, arm in key[i].items())
        prompt = PROMPT.format(audience=AUD[r["audience"]], facts=facts, candidates=cands)
        for attempt in range(3):
            try:
                return i, call(prompt), prompt
            except Exception as e:  # noqa: BLE001 - retried, then reported
                err = str(e)
        return i, {"error": err}, prompt

    todo = [i for i in key if i not in done]
    with cf.ThreadPoolExecutor(a.workers) as ex, ans_path.open("a") as f:
        for i, ans, prompt in ex.map(work, todo):
            f.write(json.dumps({"id": i, "situation": first[i]["situation"], "audience": first[i]["audience"], "answer": ans}) + "\n")
            f.flush()
            print(i, "ok" if "error" not in ans else ans, flush=True)
    if todo:
        (OUT / "prompt_template.txt").write_text(PROMPT)


if __name__ == "__main__":
    main()
