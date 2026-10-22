#!/bin/bash
#SBATCH --job-name=analyze_cpu
#SBATCH --account=bfqi-delta-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=logs/analyze_cpu_%j.out
# CPU only, between J2 and J3: CM pair analysis (probes, fusion, review flag, preregistration, student targets) on
# the J2 embeddings in data/emb/crc. Writes results/delta/CM/crc/ incl. student_targets.npz + probes_aligned.npz,
# which J3 trains on (--targets-dir results/delta/CM/crc). This job is the ONLY supported full-data analysis route:
# it reads Delta's embeddings, labels and text banks together and never touches the committed local-subset results.
# Submit from ~/twinread (the --output path is relative; setup_login.sh creates logs/).
# No --reservation. Never resubmits itself. Submit only with the user's explicit OK.
# Record: results/ncsa/<job>_${SLURM_JOB_ID}.json is written by ncsa/job_record.py.
set -uo pipefail
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}   # text banks: CONCH/MUSK text towers on CPU if not cached
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export TWINREAD_HOME=${TWINREAD_HOME:-$HOME/twinread}
export PYTHONPATH=$TWINREAD_HOME/src:${PYTHONPATH:-}
cd "$TWINREAD_HOME"
mkdir -p logs results/ncsa
T0=$(date +%s)
OUT=results/delta/CM/crc
die() { echo "ANALYZE PRECHECK FAILED: $*" >&2; python ncsa/job_record.py analyze_cpu "$T0" 2 "precheck: $*"; exit 2; }

# 1. J2 must be complete: every reader x kind x split, no unfinished chunk parts.
for r in conch musk; do for k in aligned probe; do for s in train val test; do
  [ -f "data/emb/crc/${r}_${k}_${s}.npy" ] || die "missing data/emb/crc/${r}_${k}_${s}.npy; J2 did not finish"
done; done; done
ls -d data/emb/crc/*.parts >/dev/null 2>&1 && die "unfinished data/emb/crc/*.parts; J2 is incomplete"
# 2. run_pair_analysis refuses to overwrite results (the preregistration is written once).
[ -f "$OUT/review.json" ] && die "$OUT already analysed; J3 can start"

python -c 'from pathlib import Path; from twinread.analyze import run_pair_analysis; run_pair_analysis("CM", "crc", results_dir=Path("results/delta"))'
rc=$?
for f in student_targets.npz probes_aligned.npz; do
  [ -f "$OUT/$f" ] || { echo "missing $OUT/$f after the analysis" >&2; [ $rc -eq 0 ] && rc=1; }
done
python ncsa/job_record.py analyze_cpu "$T0" "$rc" "pair=CM dataset=crc out=$OUT"
exit $rc
