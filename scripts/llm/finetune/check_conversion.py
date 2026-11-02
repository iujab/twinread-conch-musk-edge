#!/usr/bin/env python
"""Verify real weights through the conversion path: T2's file (from unsloth's GGUF) against ours (from HF safetensors), and PyTorch against GGUF.

1. Same tensor names, shapes and quantization types, tied embeddings, chat template, and how many tensors are byte-identical.
2. Greedy text from both GGUFs on the same prompts.
3. PyTorch merged model against its GGUF: greedy continuation agreement token by token on the same prompts.
Writes results/llm/finetune/conversion_check_real.json. Run with the model names as in gguf_built.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "device"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "build/llama.cpp-v0.8.0/gguf-py"))
import explain_prompt as ep  # noqa: E402
from generate import items  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from serve import Server  # noqa: E402


def compare_tensors(p_a, p_b):
    ra, rb = GGUFReader(p_a, "r"), GGUFReader(p_b, "r")
    ta, tb = {t.name: t for t in ra.tensors}, {t.name: t for t in rb.tensors}
    same_names = set(ta) == set(tb)
    same_type = sum(ta[k].tensor_type == tb[k].tensor_type for k in ta if k in tb)
    same_shape = sum(list(ta[k].shape) == list(tb[k].shape) for k in ta if k in tb)
    same_bytes, diff = 0, []
    for k in ta:
        if k in tb and bytes(ta[k].data) == bytes(tb[k].data):
            same_bytes += 1
        else:
            diff.append(k)
    tmpl = [r.fields["tokenizer.chat_template"].contents() for r in (ra, rb)]
    return {"tensors": [len(ta), len(tb)], "same_names": same_names, "same_type": same_type, "same_shape": same_shape, "byte_identical": same_bytes,
            "not_identical_examples": diff[:6], "no_output_tensor": ["output.weight" not in ta, "output.weight" not in tb],
            "chat_template_identical": tmpl[0] == tmpl[1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t2", default=str(REPO / "data/llm/gen/Qwen3-0.6B-Q4_0-embQ8.gguf"))
    ap.add_argument("--ours", default=str(REPO / "data/llm/finetune/realbase-Q4_0-embQ8.gguf"))
    ap.add_argument("--merged", default=str(REPO / "data/llm/finetune/convert/realbase/merged"))
    ap.add_argument("--n", type=int, default=6, help="prompts per situation")
    ap.add_argument("--tokens", type=int, default=60)
    ap.add_argument("--port", type=int, default=1796)
    ap.add_argument("--key", default="realbase")
    a = ap.parse_args()
    out = {"tensors_t2_vs_ours": compare_tensors(a.t2, a.ours)}
    its = items("val", a.n)
    msgs = [ep.build_messages(it["snapshot"], it["audience"]) for it in its]

    def greedy(path):
        res = []
        with Server(path, port=a.port, threads=4) as srv:
            for m in msgs:
                r = srv.chat({"messages": m, "max_tokens": a.tokens, "temperature": 0.0, "seed": 1, "repeat_penalty": 1.0, "cache_prompt": False})
                res.append(r["choices"][0]["message"]["content"])
        return res
    g_t2, g_ours = greedy(a.t2), greedy(a.ours)
    out["greedy_text_identical_t2_vs_ours"] = sum(x == y for x, y in zip(g_t2, g_ours))
    out["n_prompts"] = len(msgs)
    out["example_t2"], out["example_ours"] = g_t2[0], g_ours[0]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(a.merged)
    model = AutoModelForCausalLM.from_pretrained(a.merged, dtype=torch.float32).eval()

    def along(texts):
        agree, total, full = 0, 0, 0
        for m, g in zip(msgs, texts):
            ids = tok(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False).input_ids
            target = tok(g, add_special_tokens=False).input_ids
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([ids + target]), logits_to_keep=len(target) + 1).logits[0]
            pred = logits[:-1].argmax(-1).tolist()
            match = [p == t for p, t in zip(pred, target)]
            agree += sum(match)
            total += len(target)
            full += all(match)
        return {"agree": agree, "total": total, "rate": round(agree / total, 4), "whole_sequences_identical": full}
    out["pytorch_vs_gguf_next_token_agreement_along_gguf_greedy"] = along(g_ours)
    out["pytorch_vs_t2_file_next_token_agreement_along_its_greedy"] = along(g_t2)
    p = REPO / "results/llm/finetune/conversion_check_real.json"
    store = json.loads(p.read_text()) if p.exists() else {}
    store[a.key] = out
    p.write_text(json.dumps(store, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
