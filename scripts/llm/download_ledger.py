#!/usr/bin/env python3
"""Tally what was fetched from the internet for this work, from the real file sizes on disk where the file still exists.

Files deleted later stay in the ledger with their size from results/llm/hf_candidates.json. Writes results/llm/download_ledger.json.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
cand = json.loads((REPO / "results/llm/hf_candidates.json").read_text())
sizes = {(r["repo"], f["file"]): f["bytes"] for r in cand["gguf_repos"] if r.get("exists") for f in r["gguf"]}
models = [("bartowski/Qwen2.5-0.5B-Instruct-GGUF", "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"), ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf"),
          ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_0.gguf"), ("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_0.gguf")]
rows = [{"item": f"{r}/{f}", "bytes": sizes[(r, f)]} for r, f in models]
meta = sum(p.stat().st_size for p in (REPO / "data/llm/conversion_check").glob("*/meta/*") if p.is_file())
rows.append({"item": "tokenizer and config files of Qwen/Qwen3-0.6B and Qwen/Qwen2.5-0.5B-Instruct", "bytes": meta})
tar = REPO / "build/official-b11960/llama-b11960-bin-ubuntu-arm64.tar.gz"
rows.append({"item": "official llama.cpp b11960 arm64 tarball", "bytes": tar.stat().st_size})
rows.append({"item": "6 x 12 MB range reads of GGUF headers, not stored", "bytes": 72_000_000})
total = sum(r["bytes"] for r in rows)
out = {"rows": rows, "total_bytes": total, "total_gb": round(total / 1e9, 3), "limit_gb": 2.0}
(REPO / "results/llm/download_ledger.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
