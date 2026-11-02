#!/usr/bin/env python
"""Check by token ID that the training render equals what the pinned llama-server renders for the same messages.

Compares: HF apply_chat_template(enable_thinking=False) ids, the server's /apply-template text re-tokenized by /tokenize, and the
prompt_tokens count the server reports for a real /v1/chat/completions call. Also compares training text = prompt + answer + <|im_end|>
and records the prompt token counts. Writes results/llm/finetune/render_check.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "device"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explain_prompt as ep  # noqa: E402
from generate import items  # noqa: E402
from serve import Server  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(REPO / "data/llm/gen/Qwen3-0.6B-Q4_0-embQ8.gguf"))
ap.add_argument("--port", type=int, default=1795)
ap.add_argument("--n-per-situation", type=int, default=8)
ap.add_argument("--out", default=str(REPO / "results/llm/finetune/render_check.json"))
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(str(REPO / "data/llm/finetune/base"))
its = items("val", a.n_per_situation)
rows = []
with Server(a.model, port=a.port, threads=2) as srv:
    for it in its:
        msgs = ep.build_messages(it["snapshot"], it["audience"])
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, add_special_tokens=False).input_ids
        rendered = srv.post("/apply-template", {"messages": msgs})["prompt"]
        sids = srv.post("/tokenize", {"content": rendered, "add_special": False, "parse_special": True})["tokens"]
        r = srv.chat({"messages": msgs, "max_tokens": 1, "temperature": 0, "cache_prompt": False})
        rows.append({"id": it["id"], "same_text": rendered == text, "same_ids": ids == sids, "hf_tokens": len(ids),
                     "server_prompt_tokens": r["usage"]["prompt_tokens"], "count_matches": r["usage"]["prompt_tokens"] == len(ids)})
    sys_ids = tok(ep.SYSTEM, add_special_tokens=False).input_ids
    prefix = tok.apply_chat_template([{"role": "system", "content": ep.SYSTEM}], tokenize=False)
    train = tok.apply_chat_template(msgs + [{"role": "assistant", "content": "ANSWER"}], tokenize=False, enable_thinking=False)
    gen = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
ok = {"all_same_text": all(r["same_text"] for r in rows), "all_same_ids": all(r["same_ids"] for r in rows), "all_counts_match": all(r["count_matches"] for r in rows),
      "training_text_is_prompt_plus_answer_plus_eos": train == gen + "ANSWER<|im_end|>\n", "n": len(rows), "system_tokens": len(sys_ids),
      "cached_prefix_tokens": len(tok(prefix, add_special_tokens=False).input_ids),
      "prompt_tokens_min_max": [min(r["hf_tokens"] for r in rows), max(r["hf_tokens"] for r in rows)],
      "system_sha256": hashlib.sha256(ep.SYSTEM.encode()).hexdigest(), "prompt_version": ep.PROMPT_VERSION,
      "generation_prompt_tail": gen[-40:], "model": Path(a.model).name, "rows": rows}
out = Path(a.out)
out.write_text(json.dumps(ok, indent=1))
print(json.dumps({k: v for k, v in ok.items() if k != "rows"}, indent=1))
