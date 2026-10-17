#!/bin/bash
# Stage NCT-CRC-HE for the Delta jobs. Run on the login node after the user's explicit OK. Executes only the
# read-only check plus, if the shared folder is readable, the train conversion (CPU, writes under $OUT).
set -euo pipefail
SRC=/projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples
TWINREAD_HOME=${TWINREAD_HOME:-$HOME/twinread}
OUT=${OUT:-$TWINREAD_HOME/data}
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
export PYTHONPATH=$TWINREAD_HOME/src:${PYTHONPATH:-}

if [ -d "$SRC" ] && [ -r "$SRC" ] && [ -n "$(ls "$SRC" 2>/dev/null | head -1)" ]; then
  echo "shared CRC-100K readable at $SRC; building crc_train_{x,y}.npy in $OUT"
  python -c "from pathlib import Path; from twinread.data import crc_from_folder; print(crc_from_folder(Path('$SRC'), Path('$OUT'), 'train'))"
else
  echo "NOT readable: $SRC"
  echo "Fallback for the user to approve (login node, has internet): fetch_crc pulls all 100k from HF parquet:"
  echo "  python -c \"from pathlib import Path; from twinread.data import fetch_crc; fetch_crc(Path('$OUT'), n_train=None)\""
fi

echo
echo "CRC-VAL-HE-7K (test) is NOT on Delta. Nothing below is run automatically. Options for the user to approve:"
echo "  (a) HF download on the login node: the same fetch_crc call above also writes crc_test_{x,y}.npy"
echo "      from the CRC_VAL_HE_7K shards (skip the train rebuild if train came from the shared folder)."
echo "  (b) rsync from the laptop/this machine (run THERE):"
echo "      rsync -av data/crc_test_x.npy data/crc_test_y.npy <user>@login.delta.ncsa.illinois.edu:$OUT/"
