#!/bin/bash
# Sourced by j2_embed.sbatch and j4_spare.sbatch. Defines run_embed <pair> <dataset> <impl>.
# Exit code 3 from twinread.embed means "time budget used, rerun me". We loop INSIDE the job (not resubmission), passing
# the remaining walltime minus a 300 s margin as --budget, and stop when the budget is < 300 s or two consecutive
# exit-3 runs add no cache parts. Needs T0 (job start epoch) and WALL (walltime seconds, default 5400).
_emb_count() { find data/emb -name '*.npy' 2>/dev/null | wc -l; }
run_embed() {
  local pair=$1 ds=$2 impl=$3 rc=3 stalls=0 before after budget
  local wall=${WALL:-5400} margin=300
  before=$(_emb_count)
  while [ $rc -eq 3 ]; do
    budget=$(( wall - ( $(date +%s) - T0 ) - margin ))
    if [ $budget -lt 300 ]; then echo "budget ${budget}s < 300s: stopping; rerun in a new, user-approved job"; break; fi
    python -m twinread.embed --pair "$pair" --dataset "$ds" --impl "$impl" --budget "$budget"
    rc=$?
    [ $rc -eq 3 ] || break
    after=$(_emb_count)
    if [ "$after" -le "$before" ]; then stalls=$((stalls+1)); else stalls=0; fi
    before=$after
    if [ $stalls -ge 2 ]; then echo "two exit-3 runs without new cache parts: stopping"; break; fi
  done
  return $rc
}
