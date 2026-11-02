#!/usr/bin/env python3
"""Compare the chat templates and token ids in the kept GGUFs with the base repos. Writes results/llm/template_compare.json."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ins = json.loads((REPO / "results/llm/gguf_inspect.json").read_text())
out = {}
for gguf, key in (("Qwen3-0.6B-Q4_0-embQ8.gguf", "qwen3"), ("Qwen2.5-0.5B-Instruct-Q4_K_M.gguf", "qwen25")):
    base = json.loads((REPO / f"data/llm/conversion_check/{key}/meta/tokenizer_config.json").read_text())
    g = ins[gguf]
    same = g["chat_template"] == base.get("chat_template")
    out[gguf] = {"chat_template_identical_to_base_repo": same, "gguf_template_chars": len(g["chat_template"] or ""),
                 "base_template_chars": len(base.get("chat_template") or ""),
                 "eos": g["kv"].get("tokenizer.ggml.eos_token_id"), "pad": g["kv"].get("tokenizer.ggml.padding_token_id"),
                 "bos": g["kv"].get("tokenizer.ggml.bos_token_id"), "params_total": g["params_total"], "params_by_type": g["params_by_type"],
                 "tensor_type_counts": g["tensor_type_counts"], "token_embd": g["key_tensors"].get("token_embd.weight"),
                 "separate_output_tensor": g["separate_output_tensor"]}
    if not same:
        a, b = g["chat_template"], base.get("chat_template") or ""
        i = next((n for n, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        out[gguf]["first_difference_at"] = i
        out[gguf]["gguf_context"] = a[max(0, i - 60):i + 120]
        out[gguf]["base_context"] = b[max(0, i - 60):i + 120]
(REPO / "results/llm/template_compare.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1)[:2600])
