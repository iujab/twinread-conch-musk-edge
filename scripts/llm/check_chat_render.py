#!/usr/bin/env python
"""Check what a fine-tune must feed the model: the assistant turn as rendered for training must start with exactly the prefix
the server renders at inference, for both candidates. Uses only the tokenizer files. Writes results/llm/chat_render_check.json.
"""
import json
from pathlib import Path

from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
MSG = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USR"}]
ANS = "ANSWER"
out = {}
for key, kw in (("qwen3", {"enable_thinking": False}), ("qwen25", {})):
    tok = AutoTokenizer.from_pretrained(str(REPO / f"data/llm/conversion_check/{key}/meta"))
    gen_prompt = tok.apply_chat_template(MSG, tokenize=False, add_generation_prompt=True, **kw)
    full = tok.apply_chat_template(MSG + [{"role": "assistant", "content": ANS}], tokenize=False, add_generation_prompt=False, **kw)
    out[key] = {"generation_prompt": gen_prompt, "training_render": full,
                "training_equals_prompt_plus_answer_plus_eos": full == gen_prompt + ANS + "<|im_end|>\n",
                "eos_token": tok.eos_token, "eos_id": tok.eos_token_id, "pad_token": tok.pad_token, "pad_id": tok.pad_token_id,
                "think_ids": {t: tok.convert_tokens_to_ids(t) for t in ("<think>", "</think>")} if key == "qwen3" else None,
                "tokenizer_class": type(tok).__name__, "vocab_size": len(tok)}
    print(key, json.dumps(out[key], indent=0))
(REPO / "results/llm/chat_render_check.json").write_text(json.dumps(out, indent=1))
