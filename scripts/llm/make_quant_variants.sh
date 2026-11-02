#!/usr/bin/env bash
# Make quantization variants locally with the pinned llama-quantize, so speed can be measured without more downloads.
# --allow-requantize re-encodes already-quantized tensors. Anything requantized from a lower-precision source is labelled
# "speedonly": its speed and memory are those of the real format, but its answer quality is not representative.
# A variant that only changes token_embd (source Q6_K, nearly lossless) stays a fair quality candidate.
#
# Usage: make_quant_variants.sh SOURCE.gguf OUTPUT.gguf FTYPE [EMBED_TYPE]
#   e.g. make_quant_variants.sh data/llm/Qwen3-0.6B-Q4_0.gguf data/llm/gen/Qwen3-0.6B-Q4_0-embQ8.gguf Q4_0 q8_0
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
Q="$REPO/build/llama.cpp-v0.8.0/build-host/bin/llama-quantize"
SRC="$1"; OUT="$2"; FTYPE="$3"; EMBED="${4:-}"
mkdir -p "$(dirname "$OUT")"
ARGS=(--allow-requantize)
[ -n "$EMBED" ] && ARGS+=(--token-embedding-type "$EMBED")
nice -n 5 "$Q" "${ARGS[@]}" "$SRC" "$OUT" "$FTYPE" 8 > "$OUT.log" 2>&1 || { tail -20 "$OUT.log" >&2; exit 1; }
ls -l "$OUT"
