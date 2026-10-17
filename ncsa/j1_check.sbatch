#!/bin/bash
#SBATCH --job-name=j1_check
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=logs/j1_check_%j.out
# J1: environment, GPU, CUDA toolkit probe, kernel build check. Purpose: decide k1 vs reference for J2.
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
rc=0
{
  echo "== nvidia-smi"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  echo "== python"; python -c "import torch,transformers;print(torch.__version__,torch.version.cuda,transformers.__version__,torch.cuda.is_available())"
  echo "== CUDA toolkit probe (module name unverified)"; module avail cuda 2>&1 | head -30
  which nvcc && nvcc --version
} > results/ncsa/j1_toolkit_${SLURM_JOB_ID}.txt 2>&1 || true
cat results/ncsa/j1_toolkit_${SLURM_JOB_ID}.txt
# If nvcc is missing, load a cuda module named in the probe above and re-run with the user's OK; do not guess here.
python -c 'import twinread.kernels as k; k.load_ext("8.0"); print("K1/K2 extension build OK")'
build_rc=$?
if [ $build_rc -eq 0 ]; then echo ok > results/ncsa/j1_build_status.txt; else echo failed > results/ncsa/j1_build_status.txt; rc=$build_rc; fi
python - <<'PY'
import torch
print("gpu smoke:", (torch.randn(2048, 2048, device="cuda") @ torch.randn(2048, 2048, device="cuda")).sum().item())
PY
python ncsa/job_record.py j1_check "$T0" "$rc" "build=$(cat results/ncsa/j1_build_status.txt)"
exit $rc
