#!/bin/bash
#SBATCH --job-name=j1_check
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=logs/j1_check_%j.out
# J1: environment, GPU, CUDA toolkit probe, sm_80 kernel build (asserted), GPU correctness smoke of K1/K2 on the A100
# (pytest -m gpu tests/test_k1.py tests/test_k2.py, incl. K1's embedding-cosine test, which loads CONCH and MUSK offline)
# and a short K1 benchmark. Purpose: decide k1 vs reference for J2 (and k2 vs torch for J3).
# results/ncsa/j1_build_status.txt = ok only if the build AND the kernel tests pass; anything else -> fallbacks.
# Submit from ~/twinread (the --output path is relative; setup_login.sh creates logs/). Expected ~10-15 min.
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
mkdir -p logs results/ncsa results/delta/kernels
T0=$(date +%s)
JID=${SLURM_JOB_ID:-local}
KD=results/delta/kernels
rc=0
{
  echo "== nvidia-smi"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  echo "== python"; python -c "import torch,transformers;print(torch.__version__,torch.version.cuda,transformers.__version__,torch.cuda.is_available())"
  echo "== CUDA toolkit probe (module name unverified)"; module avail cuda 2>&1 | head -30
  which nvcc && nvcc --version
} > results/ncsa/j1_toolkit_${JID}.txt 2>&1 || true
cat results/ncsa/j1_toolkit_${JID}.txt
# If nvcc is missing, load a cuda module named in the probe above and re-run with the user's OK; do not guess here.

# 1. Build. load_ext() catches every exception and returns None, so the result is asserted, not assumed.
python -c 'import twinread.kernels as k; m = k.load_ext("8.0"); assert m is not None and hasattr(m, "camera_aug"), k.EXT_ERRORS; print("K1/K2 extension build OK")' \
  > "$KD/j1_build_${JID}.txt" 2>&1
build_rc=$?
cat "$KD/j1_build_${JID}.txt"
if [ $build_rc -eq 0 ]; then BUILD=ok; else BUILD=failed; rc=$build_rc; fi

# 2. Correctness smoke on the A100: the GPU K1/K2 tests (CPU-only tests deselected). Weights must be cached here.
TESTS=skipped; BENCH=skipped
if [ "$BUILD" = ok ]; then
  python -c "import pytest" 2>/dev/null || echo "pytest missing: run setup_login.sh (it installs pytest --user)"
  TWINREAD_GPU_TESTS=1 TWINREAD_REQUIRE_WEIGHTS=1 python -m pytest -m gpu tests/test_k1.py tests/test_k2.py -q -rs \
    -p no:cacheprovider --junitxml="$KD/j1_pytest_${JID}.xml" > "$KD/j1_pytest_${JID}.txt" 2>&1
  t_rc=$?
  tail -25 "$KD/j1_pytest_${JID}.txt"
  if [ $t_rc -eq 0 ]; then TESTS=passed; else TESTS=failed; rc=$t_rc; fi
  # 3. Short K1 benchmark (GPU part only, 64 inputs per set) into results/delta/kernels/k1_bench.json.
  python bench/bench_k1.py --part gpu --batch 64 --out "$KD" > "$KD/j1_bench_${JID}.txt" 2>&1
  b_rc=$?
  tail -5 "$KD/j1_bench_${JID}.txt"
  if [ $b_rc -eq 0 ]; then BENCH=ok; else BENCH=failed; fi   # informational: does not change the k1/reference choice
fi
if [ "$BUILD" = ok ] && [ "$TESTS" = passed ]; then STATUS=ok; else STATUS=failed; fi
echo "$STATUS" > results/ncsa/j1_build_status.txt
ERR=""
[ "$BUILD" = failed ] && ERR=" errors=$(tail -c 600 "$KD/j1_build_${JID}.txt" | tr '\n' ' ')"
python - <<'PY'
import torch
print("gpu smoke:", (torch.randn(2048, 2048, device="cuda") @ torch.randn(2048, 2048, device="cuda")).sum().item())
PY
python ncsa/job_record.py j1_check "$T0" "$rc" "build=$BUILD kernel_tests=$TESTS bench=$BENCH status=$STATUS$ERR"
exit $rc
