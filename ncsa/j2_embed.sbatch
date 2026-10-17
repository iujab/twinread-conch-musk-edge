#!/bin/bash
#SBATCH --job-name=j2_embed
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --output=logs/j2_embed_%j.out
# J2: embed full CRC 100k + 7k with CONCH+MUSK (pair CM). Purpose: cache embeddings for J3.
# No --reservation. Never resubmits itself. Submit only with the user's explicit OK.
# Record: results/ncsa/<job>_${SLURM_JOB_ID}.json is written by ncsa/job_record.py.
set -uo pipefail
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
echo "impl=$IMPL"
source ncsa/embed_loop.sh
run_embed CM crc "$IMPL"
rc=$?
python ncsa/job_record.py j2_embed "$T0" "$rc" "impl=$IMPL"
exit $rc
