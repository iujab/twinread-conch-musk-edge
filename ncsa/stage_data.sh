#!/bin/bash
# PRINT-ONLY. Checks readability of the shared CRC folder and prints the commands for the user to approve one by one.
# It never converts or downloads anything itself.
SRC=/projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples
TWINREAD_HOME=${TWINREAD_HOME:-$HOME/twinread}
OUT=${OUT:-$TWINREAD_HOME/data}

if [ -d "$SRC" ] && [ -r "$SRC" ] && [ -n "$(ls "$SRC" 2>/dev/null | head -1)" ]; then
  echo "shared CRC-100K readable at $SRC"
  echo "Train conversion (100k TIFs, CPU job; ask the user, then submit):"
  echo "  cd $TWINREAD_HOME && mkdir -p logs && sbatch ncsa/stage_crc.sbatch     # --time=01:00:00, runs crc_from_folder(train)"
else
  echo "NOT readable: $SRC"
  echo "Fallback for the user to approve (login node, has internet), fetch_crc pulls all 100k from HF parquet:"
  echo "  module load pytorch-conda/2.8 && unset PYTHONNOUSERSITE && cd $TWINREAD_HOME && PYTHONPATH=src \\"
  echo "    python -c \"from pathlib import Path; from twinread.data import fetch_crc; fetch_crc(Path('$OUT'), n_train=None)\""
fi

echo
echo "CRC-VAL-HE-7K (test) is NOT on Delta. Nothing is run automatically. Options for the user to approve:"
echo "  (a) HF download on the login node: the fetch_crc call above also writes crc_test_{x,y}.npy from the CRC_VAL_HE_7K shards."
echo "  (b) rsync from the laptop/this machine (run THERE):"
echo "      rsync -av data/crc_test_x.npy data/crc_test_y.npy <user>@login.delta.ncsa.illinois.edu:$OUT/"
