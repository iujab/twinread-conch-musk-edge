#!/usr/bin/env python
"""Prove the HF -> GGUF -> quantize path at the pinned llama.cpp commit without downloading full weights.

The real weights are over the download budget, so this builds a tiny, randomly initialised model with the same architecture
class and the real tokenizer and chat template of the base repo, saves it the way a merged fine-tune would be saved, then runs
the pinned convert_hf_to_gguf.py and llama-quantize on it and checks what came out: architecture, tied or separate output
tensor, chat template identical to the base repo's, special token ids. It proves the toolchain, not the answer quality.

Writes results/llm/conversion_check.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LLAMA = REPO / "build/llama.cpp-v0.8.0"
BASES = {
    "qwen3": {"repo": "Qwen/Qwen3-0.6B", "over": {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
                                                   "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16}},
    "qwen25": {"repo": "Qwen/Qwen2.5-0.5B-Instruct", "over": {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
                                                               "num_attention_heads": 4, "num_key_value_heads": 2}},
}
META = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]


def run(key: str, spec: dict, ftype: str, embed: str, work: Path) -> dict:
    from huggingface_hub import hf_hub_download
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    meta_dir = work / key / "meta"
    for f in META:
        try:
            hf_hub_download(spec["repo"], f, local_dir=str(meta_dir), token=False)
        except Exception as e:
            print(f"  {spec['repo']}/{f}: {type(e).__name__}")
    base_cfg = json.loads((meta_dir / "config.json").read_text())
    tok_cfg = json.loads((meta_dir / "tokenizer_config.json").read_text())
    gen_cfg = json.loads((meta_dir / "generation_config.json").read_text())

    cfg = AutoConfig.from_pretrained(str(meta_dir))
    for k, v in spec["over"].items():
        setattr(cfg, k, v)
    if getattr(cfg, "layer_types", None):  # transformers validates this list against num_hidden_layers
        cfg.layer_types = cfg.layer_types[:cfg.num_hidden_layers]
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    tiny = work / key / "tiny"
    shutil.rmtree(tiny, ignore_errors=True)
    model.save_pretrained(str(tiny), safe_serialization=True)
    for f in META[2:]:
        if (meta_dir / f).exists():
            shutil.copy(meta_dir / f, tiny / f)
    saved_keys = []
    from safetensors import safe_open
    with safe_open(str(tiny / "model.safetensors"), "pt") as st:
        saved_keys = list(st.keys())

    f16 = work / key / "tiny-f16.gguf"
    # -I keeps the reader of downloaded files isolated; only the pinned checkout is added to the import path.
    boot = ("import runpy,sys;sys.path.insert(0,sys.argv[1]);c=sys.argv[1]+'/convert_hf_to_gguf.py';sys.argv=[c]+sys.argv[2:];"
            "runpy.run_path(c,run_name='__main__')")
    conv = subprocess.run([sys.executable, "-I", "-c", boot, str(LLAMA), str(tiny), "--outfile", str(f16), "--outtype", "f16"],
                          capture_output=True, text=True, cwd=str(work))
    if conv.returncode:
        return {"repo": spec["repo"], "ok": False, "stage": "convert", "stderr": conv.stderr[-1500:]}
    out = work / key / f"tiny-{ftype}{'-emb' + embed if embed else ''}.gguf"
    q = [str(LLAMA / "build-host/bin/llama-quantize")]
    if embed:
        q += ["--token-embedding-type", embed]
    qr = subprocess.run(q + [str(f16), str(out), ftype, "4"], capture_output=True, text=True)
    if qr.returncode:
        return {"repo": spec["repo"], "ok": False, "stage": "quantize", "stderr": qr.stderr[-1500:]}

    sys.path.insert(0, str(LLAMA / "gguf-py"))
    from gguf import GGUFReader
    r = GGUFReader(str(out), "r")
    names = {t.name: t.tensor_type.name for t in r.tensors}

    def field(key_):
        f = r.fields.get(key_)
        return f.contents() if f is not None else None
    tmpl = field("tokenizer.chat_template")
    base_tmpl = tok_cfg.get("chat_template")
    if base_tmpl is None and (meta_dir / "chat_template.jinja").exists():
        base_tmpl = (meta_dir / "chat_template.jinja").read_text()
    return {
        "repo": spec["repo"], "ok": True, "tie_word_embeddings_base": base_cfg.get("tie_word_embeddings"),
        "safetensors_has_lm_head": "lm_head.weight" in saved_keys, "gguf_has_separate_output": "output.weight" in names,
        "token_embd_type": names.get("token_embd.weight"), "architecture": field("general.architecture"),
        "chat_template_identical_to_base_repo": tmpl == base_tmpl, "eos_token_id": field("tokenizer.ggml.eos_token_id"),
        "pad_token_id": field("tokenizer.ggml.padding_token_id"), "bos_token_id": field("tokenizer.ggml.bos_token_id"),
        "base_generation_config": gen_cfg, "base_tokenizer_eos": tok_cfg.get("eos_token"), "base_tokenizer_pad": tok_cfg.get("pad_token"),
        "base_config": {k: base_cfg.get(k) for k in ("architectures", "hidden_size", "num_hidden_layers", "num_attention_heads",
                                                       "num_key_value_heads", "intermediate_size", "vocab_size", "torch_dtype", "dtype",
                                                       "max_position_embeddings", "tie_word_embeddings")},
        "tiny_output": str(out.relative_to(REPO)), "quantize_cmd": " ".join(q + ["IN-f16.gguf", "OUT.gguf", ftype]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="qwen3,qwen25")
    ap.add_argument("--ftype", default="Q4_K_M")
    ap.add_argument("--embed", default="")
    ap.add_argument("--out", default=str(REPO / "results/llm/conversion_check.json"))
    a = ap.parse_args()
    work = REPO / "data/llm/conversion_check"
    work.mkdir(parents=True, exist_ok=True)
    res = {"llama_cpp_commit": subprocess.check_output(["git", "-C", str(LLAMA), "rev-parse", "HEAD"], text=True).strip(),
           "ftype": a.ftype, "embed": a.embed, "weights": "tiny random, same architecture class, real tokenizer", "checks": {}}
    for key in a.which.split(","):
        print(key, flush=True)
        res["checks"][key] = run(key, BASES[key], a.ftype, a.embed, work)
        print(json.dumps({k: v for k, v in res["checks"][key].items() if k not in ("base_generation_config", "base_config")}, indent=0)[:1400], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
