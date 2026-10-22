#!/usr/bin/env bash
# CM pair (CONCH + MUSK) local-subset dev run: embed the readers (looping on exit code 3 = time budget used, each run
# under the GPU lock), then run the CPU-only pair analysis.
# Usage: scripts/run_cm_local.sh embed <cam|crc> [reader ...]   (readers default: conch musk)
#        scripts/run_cm_local.sh analyze
# Env overrides (tests): PY (python), LOCK (lock prefix, default "flock $B/.gpu.lock"), RESULTS (analysis results dir),
# IMPL (default reference: the cached local embeddings are reference-impl; k1 would recompute them, keys differ).
set -uo pipefail
P="$(cd "$(dirname "$0")/.." && pwd)"
B="${BENCH:-$(cd "$P/.." && pwd)}"   # bench root (venv + shared GPU lock); set BENCH for a clone elsewhere
PY=${PY:-$B/.venv/bin/python}
LOCK=${LOCK-flock $B/.gpu.lock}
RESULTS=${RESULTS:-results/cm_local}
IMPL=${IMPL:-reference}
export PYTHONPATH="$P/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$P"
case "${1:-}" in
embed)
  ds=$2; shift 2
  readers=("$@"); [ ${#readers[@]} -gt 0 ] || readers=(conch musk)
  run=0
  # aligned first (complete for all splits; the student distils tA/tB from it), then probe.
  # MUSK gets --budget 300 and batch 8: its probe (ms_aug) chunks are slow and a 480 s budget overran the 10-min lock.
  for kind in aligned probe; do
    for r in "${readers[@]}"; do
      budget=480; batch=16
      if [ "$r" = musk ]; then budget=300; batch=8; fi
      while :; do
        run=$((run+1))
        $LOCK $PY -m twinread.embed --readers "$r" --dataset "$ds" --kinds $kind \
          --splits train val test --budget $budget --batch $batch --impl "$IMPL" > /dev/null
        rc=$?; echo "$(date +%T) reader=$r dataset=$ds kind=$kind run=$run rc=$rc"
        [ $rc = 3 ] || break
      done
      [ $rc = 0 ] || exit $rc
    done
  done ;;
analyze)
  set -e
  for ds in cam crc; do
    [ -d "$RESULTS/CM/$ds" ] && [ -f "$RESULTS/CM/$ds/review.json" ] && continue
    $PY -c "
from pathlib import Path
from twinread.analyze import run_pair_analysis
run_pair_analysis('CM', '$ds', results_dir=Path('$RESULTS'))"
  done ;;
*)
  echo "usage: $0 embed <cam|crc> [reader ...] | analyze" >&2; exit 2 ;;
esac
