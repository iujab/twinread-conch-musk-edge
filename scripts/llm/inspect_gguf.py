#!/usr/bin/env python
"""Read GGUF headers only (no inference): tensor quantization mix, tied output or not, chat template, special tokens.

Run with -I and pass paths as arguments, since the files are untrusted downloads. Writes results/llm/gguf_inspect.json.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--gguf-py", required=True, help="gguf-py directory of the pinned llama.cpp checkout")
ap.add_argument("--out", default="results/llm/gguf_inspect.json")
a = ap.parse_args()
sys.path.insert(0, a.gguf_py)
from gguf import GGUFReader  # noqa: E402


def field_value(r, key):
    f = r.fields.get(key)
    if f is None:
        return None
    try:
        return f.contents()
    except Exception:
        return None


def macs(t):
    n = 1
    for d in t.shape:
        n *= int(d)
    return n


out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
for path in a.files:
    r = GGUFReader(path, "r")
    mix = collections.Counter()
    params = collections.Counter()
    key_tensors = {}
    for t in r.tensors:
        q = t.tensor_type.name
        mix[q] += 1
        params[q] += macs(t)
        if t.name in ("token_embd.weight", "output.weight") or t.name.startswith("blk.0."):
            key_tensors[t.name] = {"type": q, "shape": [int(x) for x in t.shape], "bytes": int(t.n_bytes)}
    arch = field_value(r, "general.architecture")
    kv = {k: field_value(r, k) for k in (
        "general.architecture", "general.name", "general.file_type", "general.quantization_version", f"{arch}.block_count",
        f"{arch}.embedding_length", f"{arch}.feed_forward_length", f"{arch}.attention.head_count",
        f"{arch}.attention.head_count_kv", f"{arch}.context_length", "tokenizer.ggml.model", "tokenizer.ggml.pre",
        "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id", "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.add_bos_token")}
    toks = r.fields.get("tokenizer.ggml.tokens")
    tmpl = field_value(r, "tokenizer.chat_template")
    has_output = any(t.name == "output.weight" for t in r.tensors)
    out[Path(path).name] = {"bytes": Path(path).stat().st_size, "kv": kv, "n_vocab": len(toks.data) if toks else None,
                            "n_tensors": len(r.tensors), "tensor_type_counts": dict(mix), "params_by_type": dict(params),
                            "params_total": sum(params.values()), "separate_output_tensor": has_output,
                            "key_tensors": key_tensors, "chat_template": tmpl}
    print(Path(path).name, "params", sum(params.values()), dict(mix), "separate output tensor:", has_output)
    print("  ", {k: v for k, v in kv.items() if v is not None})
    print("  ", {k: (v["type"], v["shape"]) for k, v in key_tensors.items() if "blk" not in k})
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(out, indent=1))
