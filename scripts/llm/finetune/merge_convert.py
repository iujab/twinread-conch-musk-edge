#!/usr/bin/env python
"""Merge a LoRA adapter into Qwen3-0.6B on the CPU, keep embeddings tied, convert with the pinned llama.cpp, quantize, inspect.

  merged-f16.gguf -> <name>-Q4_0-embQ8.gguf, with --token-embedding-type q8_0 (device/llm/README.md). Also runs the same path with no
  adapter (--no-adapter), which puts the real base weights through conversion so the result can be compared with the file T2 made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
REPO = Path(__file__).resolve().parents[3]
LLAMA = REPO / "build/llama.cpp-v0.8.0"
PY = sys.executable


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter")
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--base", default=str(REPO / "data/llm/finetune/base"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--workdir", default=str(REPO / "data/llm/finetune/convert"))
    a = ap.parse_args()
    work = Path(a.workdir) / a.name
    merged = work / "merged"
    work.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    if not (merged / "model.safetensors").exists():
        model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float32)
        if not a.no_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
        model = model.to(torch.bfloat16)
        assert model.get_output_embeddings().weight is model.get_input_embeddings().weight or \
            model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr(), "embeddings are not tied"
        model.save_pretrained(str(merged), safe_serialization=True)
        AutoTokenizer.from_pretrained(a.base).save_pretrained(str(merged))
        for fn in ("generation_config.json",):
            src = Path(a.base) / fn
            if src.exists():
                (merged / fn).write_bytes(src.read_bytes())
        del model

    from safetensors import safe_open
    with safe_open(str(merged / "model.safetensors"), framework="pt") as f:
        keys = list(f.keys())
    assert not any(k.startswith("lm_head") for k in keys), "merged file holds a separate lm_head"

    f16 = work / "merged-f16.gguf"
    if not f16.exists():
        subprocess.run([PY, str(LLAMA / "convert_hf_to_gguf.py"), str(merged), "--outfile", str(f16), "--outtype", "f16"], check=True,
                       stdout=open(work / "convert.log", "w"), stderr=subprocess.STDOUT)
    out = Path(a.workdir).parent / f"{a.name}-Q4_0-embQ8.gguf"
    subprocess.run([str(LLAMA / "build-host/bin/llama-quantize"), "--token-embedding-type", "q8_0", str(f16), str(out), "Q4_0"], check=True,
                   stdout=open(work / "quantize.log", "w"), stderr=subprocess.STDOUT)
    res = REPO / "results/llm/finetune"
    subprocess.run([PY, "-I", str(REPO / "scripts/llm/inspect_gguf.py"), str(out), "--gguf-py", str(LLAMA / "gguf-py"), "--out",
                    str(res / "gguf_inspect.json")], check=True)
    info = {"name": a.name, "path": str(out.relative_to(REPO)), "bytes": out.stat().st_size, "sha256": sha256(out)}
    log = res / "gguf_built.json"
    built = json.loads(log.read_text()) if log.exists() else {}
    built[a.name] = info
    log.write_text(json.dumps(built, indent=1))
    print(json.dumps(info, indent=1))
    f16.unlink()  # 1.2 GB that the quantized file makes redundant


if __name__ == "__main__":
    main()
