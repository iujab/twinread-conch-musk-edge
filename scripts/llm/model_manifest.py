#!/usr/bin/env python3
"""Record exact file names, sizes, hashes and provenance of the kept models. Writes results/llm/models.json."""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = {
    "primary": {"path": "data/llm/gen/Qwen3-0.6B-Q4_0-embQ8.gguf", "base_hf": "Qwen/Qwen3-0.6B", "source_repo": "unsloth/Qwen3-0.6B-GGUF",
                "source_file": "Qwen3-0.6B-Q4_0.gguf",
                "made_with": "build/llama.cpp-v0.8.0/build-host/bin/llama-quantize --allow-requantize --token-embedding-type q8_0 "
                             "Qwen3-0.6B-Q4_0.gguf Qwen3-0.6B-Q4_0-embQ8.gguf Q4_0 8 (scripts/llm/make_quant_variants.sh)"},
    "fallback": {"path": "data/llm/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf", "base_hf": "Qwen/Qwen2.5-0.5B-Instruct", "source_repo": "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
                 "source_file": "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf", "made_with": "unchanged download"},
    "primary_source": {"path": "data/llm/Qwen3-0.6B-Q4_0.gguf", "base_hf": "Qwen/Qwen3-0.6B", "source_repo": "unsloth/Qwen3-0.6B-GGUF",
                       "source_file": "Qwen3-0.6B-Q4_0.gguf", "made_with": "unchanged download"},
}
for k, m in MODELS.items():
    p = REPO / m["path"]
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    m["bytes"], m["sha256"] = p.stat().st_size, h.hexdigest()
    print(k, m["bytes"], m["sha256"])
(REPO / "results/llm/models.json").write_text(json.dumps(MODELS, indent=1))
