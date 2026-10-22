#!/bin/bash
#SBATCH --job-name=j3_distill
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --output=logs/j3_distill_%j.out
# J3: distill the CM student. Purpose: student checkpoint + metrics in results/delta/student/cm_crc (never the
# committed results/student/cm_crc). Inputs: the CRC tiles in data/ and the analysis outputs
# $TARGETS/{student_targets.npz,probes_aligned.npz} from ncsa/analyze_cpu.slurm after J2 (results/delta/CM/crc,
# private home only). Targets come only from $TARGETS, not from data/emb.
# Submit from ~/twinread (the --output path is relative; setup_login.sh creates logs/).
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
BACKBONE=${BACKBONE:-mobilenetv4_conv_small}  # = train_student.py default, every local student and the latency harness
TARGETS=${TARGETS:-results/delta/CM/crc}       # run_pair_analysis output dir (the CPU analysis job)
OUT=results/delta/student/cm_crc               # Delta results never overwrite the committed local-subset ones
die() { echo "J3 PRECHECK FAILED: $*" >&2; python ncsa/job_record.py j3_distill "$T0" 2 "precheck: $*"; exit 2; }

# 1. Analysis outputs must exist: J2 embeds -> analyze_cpu.slurm (CPU) -> J3.
[ -f "$TARGETS/student_targets.npz" ] || die "missing $TARGETS/student_targets.npz; run the CPU analysis job after J2 (README step 6b)"
[ -f "$TARGETS/probes_aligned.npz" ] || die "missing $TARGETS/probes_aligned.npz; run the CPU analysis job after J2 (README step 6b)"
[ -f "$TARGETS/preregistration.json" ] || echo "note: no $TARGETS/preregistration.json; test metrics will be skipped"
for f in crc_train_x.npy crc_train_y.npy crc_test_x.npy crc_test_y.npy; do
  [ -f "data/$f" ] || die "missing data/$f (the student trains on the tiles)"
done

# 2. Augmentation kernels: k2 only if J1 built the extension, else the torch path.
AUG=k2
[ "$(cat results/ncsa/j1_build_status.txt 2>/dev/null)" = "ok" ] || AUG=torch
echo "aug=$AUG"

# 3. Backbone weights must be in the offline HF cache (download_weights.py --backbones on the login node).
python - "$BACKBONE" <<'PY' || die "backbone weights for $BACKBONE not in the offline HF cache; run download_weights.py --backbones $BACKBONE on the login node"
import sys, timm
timm.create_model(sys.argv[1], pretrained=True)
PY

python scripts/train_student.py --pair CM --dataset crc --targets-dir "$TARGETS" --backbone "$BACKBONE" --epochs 15 --aug "$AUG" \
  --out "$OUT"
rc=$?
python ncsa/job_record.py j3_distill "$T0" "$rc" "backbone=$BACKBONE aug=$AUG targets=$TARGETS out=$OUT"
exit $rc
