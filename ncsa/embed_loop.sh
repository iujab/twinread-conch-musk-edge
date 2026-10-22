#!/bin/bash
# Sourced by j2_embed.slurm and j4_spare.slurm (Delta only; run_demo.sh has its own local loop). Defines
#   run_embed <dataset> <impl> <reader selector...>     e.g.  run_embed crc k1 --pair CM
#                                                             run_embed crc k1 --readers pathgenclip quiltnet_b16
# Exit code 3 from twinread.embed means "time budget used, rerun me". We loop INSIDE the job (not resubmission),
# passing the remaining walltime minus a 300 s margin as --budget, and stop when the budget is < 300 s or two
# consecutive exit-3 runs add no cache parts. twinread.embed checks that budget before every 2,048-tile chunk, so one
# call normally runs to the end of the job with the readers loaded once. (The local 300 s per-call MUSK cap is not
# used on Delta: every call reloads CONCH and MUSK. EMBED_CALL_CAP=<seconds> re-enables a cap for debugging.)
# Needs T0 (job start epoch) and WALL (walltime seconds, default 5400). EMBED_BATCH = encode batch (default 64 for the
# A100 40 GB; a CUDA OOM halves it inside twinread.embed).
_emb_count() { find data/emb -name '*.npy' 2>/dev/null | wc -l; }
run_embed() {
  local ds=$1 impl=$2 rc=3 stalls=0 before after budget call_budget
  shift 2
  local wall=${WALL:-5400} margin=300 batch=${EMBED_BATCH:-64} cap=${EMBED_CALL_CAP:-}
  before=$(_emb_count)
  while [ $rc -eq 3 ]; do
    budget=$(( wall - ( $(date +%s) - T0 ) - margin ))
    if [ $budget -lt 300 ]; then echo "budget ${budget}s < 300s: stopping; rerun in a new, user-approved job"; break; fi
    call_budget=$budget
    if [ -n "$cap" ] && [ $call_budget -gt "$cap" ]; then call_budget=$cap; fi
    python -m twinread.embed "$@" --dataset "$ds" --impl "$impl" --batch "$batch" --budget "$call_budget"
    rc=$?
    [ $rc -eq 3 ] || break
    after=$(_emb_count)
    if [ "$after" -le "$before" ]; then stalls=$((stalls+1)); else stalls=0; fi
    before=$after
    if [ $stalls -ge 2 ]; then echo "two exit-3 runs without new cache parts: stopping"; break; fi
  done
  return $rc
}
