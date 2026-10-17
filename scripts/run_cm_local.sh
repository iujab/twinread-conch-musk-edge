#!/usr/bin/env bash
# CM pair (CONCH + MUSK) local-subset dev run. CONCH embeddings come from a separate driver; this embeds MUSK
# (looping on exit code 3 = time budget used) and then runs the CPU-only pair analysis.
# Usage: scripts/run_cm_local.sh embed <cam|crc> | analyze
set -uo pipefail
P="$(cd "$(dirname "$0")/.." && pwd)"
B=/home/nanashi-mumei/dev/ashbyhackathonbench
PY=$B/.venv/bin/python
export PYTHONPATH="$P/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$P"
case "$1" in
embed)
  ds=$2; run=0
  # aligned first (complete for all splits; the student distils tA/tB from it), then probe. impl=reference: K1 not merged.
  for kind in aligned probe; do
    while :; do
      run=$((run+1))
      flock $B/.gpu.lock $PY -m twinread.embed --readers musk --dataset $ds --kinds $kind \
        --splits train val test --budget 480 --batch 8 --impl reference > /dev/null
      rc=$?; echo "$(date +%T) reader=musk dataset=$ds kind=$kind run=$run rc=$rc"
      [ $rc = 3 ] || break
    done
    [ $rc = 0 ] || exit $rc
  done ;;
analyze)
  for ds in cam crc; do
    [ -d results/cm_local/CM/$ds ] && [ -f results/cm_local/CM/$ds/review.json ] && continue
    $PY -c "
from pathlib import Path
from twinread.analyze import run_pair_analysis
run_pair_analysis('CM', '$ds', results_dir=Path('results/cm_local'))"
  done ;;
esac
