#!/bin/bash
#SBATCH --job-name=j3_distill
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --output=logs/j3_distill_%j.out
# J3: distill the CM student on the cached CRC embeddings. Purpose: student checkpoint + metrics.
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
BACKBONE=${BACKBONE:-mobilenetv3_large_100}   # choice made by the user before submission; override with --export
python scripts/train_student.py --pair CM --dataset crc --backbone "$BACKBONE" --epochs 15 --aug k2
rc=$?
python ncsa/job_record.py j3_distill "$T0" "$rc" "backbone=$BACKBONE"
exit $rc
