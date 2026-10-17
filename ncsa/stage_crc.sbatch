#!/bin/bash
#SBATCH --job-name=stage_crc
#SBATCH --account=bfqi-delta-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/stage_crc_%j.out
# CPU only: convert the shared NCT-CRC-HE-100K class folders to crc_train_{x,y}.npy. Never resubmits itself.
# Submit only with the user's explicit OK.
set -uo pipefail
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TWINREAD_HOME=${TWINREAD_HOME:-$HOME/twinread}
export PYTHONPATH=$TWINREAD_HOME/src:${PYTHONPATH:-}
cd "$TWINREAD_HOME" && mkdir -p logs data
python -c 'from pathlib import Path; from twinread.data import crc_from_folder; print(crc_from_folder(Path("/projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples"), Path("data"), "train"))'
