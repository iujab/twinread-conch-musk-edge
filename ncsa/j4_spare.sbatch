#!/bin/bash
#SBATCH --job-name=j4_spare
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --output=logs/j4_spare_%j.out
# J4: spare job, only if J1-J3 leave a gap. Purpose: set by the user at submission via ACTION.
# No --reservation. Never resubmits itself. Submit only with the user's explicit OK.
# Record: results/ncsa/<job>_${SLURM_JOB_ID}.json is written by ncsa/job_record.py.
# ACTION is an allow-list of named, fixed commands. Anything else is rejected. Use: sbatch --export=ALL,ACTION=embed_open
set -uo pipefail
ACTION=${ACTION:-}
case "$ACTION" in
  embed_open|distill_seed2|kernel_build) ;;
  *) echo "unknown ACTION '$ACTION'; allowed: embed_open | distill_seed2 | kernel_build" >&2; exit 2 ;;
esac
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}   # hub cache = $HF_HOME/hub, same as download_weights.py
export TORCH_CUDA_ARCH_LIST=8.0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TWINREAD_HOME=${TWINREAD_HOME:-$HOME/twinread}
export PYTHONPATH=$TWINREAD_HOME/src:${PYTHONPATH:-}
cd "$TWINREAD_HOME"
mkdir -p logs results/ncsa
T0=$(date +%s)
IMPL=k1
[ "$(cat results/ncsa/j1_build_status.txt 2>/dev/null)" = "ok" ] || IMPL=reference
rc=0
case "$ACTION" in
  embed_open)
    source ncsa/embed_loop.sh
    run_embed OPEN crc "$IMPL"; rc=$? ;;
  distill_seed2)
    AUG=k2; [ "$IMPL" = k1 ] || AUG=torch
    TARGETS=${TARGETS:-results/CM/crc}   # analysis outputs, as in J3
    if [ -f "$TARGETS/student_targets.npz" ] && [ -f "$TARGETS/probes_aligned.npz" ]; then
      python scripts/train_student.py --pair CM --dataset crc --targets-dir "$TARGETS" --backbone mobilenetv3_large_100 \
        --epochs 15 --aug "$AUG" --seed 2
      rc=$?
    else
      echo "missing $TARGETS/student_targets.npz or probes_aligned.npz; run the CPU analysis job first (README 6b)" >&2; rc=2
    fi ;;
  kernel_build)
    python -c 'import twinread.kernels as k; k.load_ext("8.0"); print("extension build OK")'
    rc=$? ;;
esac
python ncsa/job_record.py j4_spare "$T0" "$rc" "action=$ACTION impl=$IMPL"
exit $rc
