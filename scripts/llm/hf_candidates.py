#!/usr/bin/env python
"""List small Qwen GGUF candidates on Hugging Face with no token, so names and sizes are verified rather than assumed.

Writes results/llm/hf_candidates.json: per repo gated flag, licence and every .gguf with its exact size and sha256.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

from huggingface_hub import HfApi

REPOS = [
    "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
    "unsloth/Qwen2.5-0.5B-Instruct-GGUF",
    "Qwen/Qwen3-0.6B-GGUF",
    "unsloth/Qwen3-0.6B-GGUF",
    "bartowski/Qwen_Qwen3-0.6B-GGUF",
    "ggml-org/Qwen3-0.6B-GGUF",
    "unsloth/Qwen3.5-0.8B-GGUF",
    "ggml-org/Qwen3.5-0.8B-GGUF",
    "bartowski/Qwen_Qwen3.5-0.8B-GGUF",
    "Qwen/Qwen2-0.5B-Instruct-GGUF",
    "Qwen/Qwen1.5-0.5B-Chat-GGUF",
]
BASE = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B", "Qwen/Qwen2-0.5B-Instruct"]


def repo_info(api: HfApi, rid: str) -> dict:
    try:
        info = api.model_info(rid, files_metadata=True)
    except Exception as e:  # missing or private: record it, because "verify what actually exists" is the point
        return {"repo": rid, "exists": False, "error": type(e).__name__}
    files = []
    weights_bytes = 0
    for s in info.siblings:
        if s.rfilename.endswith(".gguf"):
            sha = s.lfs.sha256 if s.lfs else None
            m = re.search(r"([qQ]\d+(?:_\d+)?(?:_[kK])?(?:_[A-Za-z0-9]+)*|[Ff]\d+|BF16|bf16)(?=\.gguf$)", s.rfilename)
            files.append({"file": s.rfilename, "bytes": s.size, "sha256": sha, "quant": m.group(1) if m else None})
        if s.rfilename.endswith((".safetensors", ".bin")) and s.size:
            weights_bytes += s.size
    tags = info.tags or []
    lic = next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None)
    return {"repo": rid, "exists": True, "gated": info.gated, "license": lic, "sha": info.sha,
            "last_modified": str(info.last_modified), "downloads": info.downloads,
            "safetensors_bytes": weights_bytes, "gguf": sorted(files, key=lambda f: f["bytes"] or 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/llm/hf_candidates.json")
    a = ap.parse_args()
    api = HfApi(token=False)
    out = {"checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "token": None,
           "gguf_repos": [repo_info(api, r) for r in REPOS], "base_repos": [repo_info(api, r) for r in BASE]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for r in out["gguf_repos"] + out["base_repos"]:
        if not r["exists"]:
            print(f"MISSING {r['repo']} ({r['error']})")
            continue
        print(f"{r['repo']} gated={r['gated']} license={r['license']} safetensors={r['safetensors_bytes']/1e6:.0f}MB")
        for f in r["gguf"]:
            print(f"    {f['bytes']/1e6:8.1f} MB  {f['file']}")


if __name__ == "__main__":
    main()
