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
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}   # hub cache = $HF_HOME/hub, same as download_weights.py
export TORCH_CUDA_ARCH_LIST=8.0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TWINREAD_HOME=${TWINREAD_HOME:-$HOME/twinread}
export PYTHONPATH=$TWINREAD_HOME/src:${PYTHONPATH:-}
cd "$TWINREAD_HOME"
mkdir -p logs results/ncsa
T0=$(date +%s)
BACKBONE=${BACKBONE:-mobilenetv3_large_100}   # confirm with the user; override via --export=ALL,BACKBONE=...
die() { echo "J3 PRECHECK FAILED: $*" >&2; python ncsa/job_record.py j3_distill "$T0" 2 "precheck: $*"; exit 2; }

# 1. J2 outputs must exist (CONCH and MUSK embeddings, train split).
for r in conch musk; do
  ls data/emb/crc/${r}_*_train.npy >/dev/null 2>&1 || die "missing data/emb/crc/${r}_*_train.npy; J2 did not finish"
done
ls data/emb/crc/*.parts >/dev/null 2>&1 && die "unfinished data/emb/crc/*.parts; J2 is incomplete"
[ -d data/student_targets ] || echo "note: data/student_targets absent; train_student.py derives targets from data/emb"

# 2. Augmentation kernels: k2 only if J1 built the extension, else the torch path.
AUG=k2
[ "$(cat results/ncsa/j1_build_status.txt 2>/dev/null)" = "ok" ] || AUG=torch
echo "aug=$AUG"

# 3. Backbone weights must be in the offline HF cache (download_weights.py --backbones on the login node).
python - "$BACKBONE" <<'PY' || die "backbone weights for $BACKBONE not in the offline HF cache; run download_weights.py --backbones $BACKBONE on the login node"
import sys, timm
timm.create_model(sys.argv[1], pretrained=True)
PY

python scripts/train_student.py --pair CM --dataset crc --backbone "$BACKBONE" --epochs 15 --aug "$AUG"
rc=$?
python ncsa/job_record.py j3_distill "$T0" "$rc" "backbone=$BACKBONE aug=$AUG"
exit $rc
