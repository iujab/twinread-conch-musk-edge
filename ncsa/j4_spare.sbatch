#!/bin/bash
#SBATCH --job-name=j4_spare
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --output=logs/j4_spare_%j.out
# J4: spare job, only if J1-J3 leave a gap (for example an OPEN-pair repeat). Purpose: set by the user at submission.
# No --reservation. Never resubmits itself. Submit only with the user's explicit OK.
# Record: results/ncsa/<job>_${SLURM_JOB_ID}.json is written by ncsa/job_record.py.
set -uo pipefail
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_CUDA_ARCH_LIST=8.0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TWINREAD_HOME=${TWINREAD_HOME:-$HOME/twinread}
export PYTHONPATH=$TWINREAD_HOME/src:${PYTHONPATH:-}
cd "$TWINREAD_HOME"
mkdir -p logs results/ncsa
T0=$(date +%s)
# Edit CMD with the user before submitting. Default: OPEN pair embeddings, same flags as J2.
CMD=${CMD:-"python -m twinread.embed --pair OPEN --dataset crc --impl reference"}
echo "running: $CMD"
bash -c "$CMD"
rc=$?
python ncsa/job_record.py j4_spare "$T0" "$rc" "cmd=$CMD"
exit $rc
