#!/usr/bin/env bash
# OPEN pair (PathGen-CLIP + chosen partner) end-to-end analysis. Embeddings must already be cached under data/emb/
# (produced by `python -m twinread.embed --impl reference` under the GPU lock); this script only verifies them.
set -euo pipefail
P="$(cd "$(dirname "$0")/.." && pwd)"
B="${BENCH:-$(cd "$P/.." && pwd)}"   # bench root; set BENCH for a clone elsewhere
PY=${PY:-$B/.venv/bin/python}
export PYTHONPATH="$P/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$P"
for ds in cam crc; do for r in pathgenclip quiltnet_b16 biomedclip; do
  n=$(ls data/emb/$ds/${r}_{aligned,probe}_{train,val,test}.npy 2>/dev/null | wc -l)
  [ "$n" = 6 ] || { echo "missing embeddings: $r $ds ($n/6)"; exit 3; }
done; done
mkdir -p results/open
[ -f results/open/partner.json ] || $PY scripts/choose_partner.py results/open/partner.json
PARTNER=$($PY -c "import json;print(json.load(open('results/open/partner.json'))['partner'])")
# analysis is CPU-only (text banks load the reader on CPU), so no GPU lock is taken
for ds in crc cam; do
  [ -f results/open/OPEN/$ds/review.json ] || [ -f results/open/$ds/review.json ] || \
  CUDA_VISIBLE_DEVICES="" $PY -c "
from pathlib import Path
from twinread.analyze import run_pair_analysis
run_pair_analysis('OPEN', '$ds', Path('results/open'), partner='$PARTNER')"
done
