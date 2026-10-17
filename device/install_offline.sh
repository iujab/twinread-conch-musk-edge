#!/usr/bin/env bash
# Install the TwinRead device app on the UNO Q (Debian aarch64). Stage wheels once on a machine WITH internet, copy the
# folder to the board; the board itself never needs the network. Research prototype, not a diagnostic device.
#   1. on the board:         python3 --version              (Debian trixie: Python 3.13.x)
#   2. stage (online host):  PYVER=3.13 ./install_offline.sh stage     -> wheels/ (aarch64 wheels for that Python)
#   3. install (on board):   ./install_offline.sh install              -> .venv installed with --no-index from wheels/
# PYVER accepts "3.13" or the board's whole `python3 --version` output ("Python 3.13.5"); default 3.13 (trixie).
# Current onnxruntime / numpy / pillow publish aarch64 wheels only as manylinux_2_28 (glibc >= 2.28; trixie has 2.41),
# so manylinux2014_aarch64 alone finds no onnxruntime 1.31.0. Both tags are passed (pure/abi3 deps such as protobuf
# still ship manylinux2014). Pins: versions checked on PyPI (2026-10-17) to have cp313 manylinux_2_28_aarch64 wheels.
# `install` refuses wheels staged for a different Python than the board's. Unverified on a board.
set -euo pipefail
cd "$(dirname "$0")"
PINS=(onnxruntime==1.31.0 numpy==2.5.4 pillow==13.0.0 opencv-python-headless==5.0.0.93)
PLATFORMS=(--platform manylinux_2_28_aarch64 --platform manylinux2014_aarch64)
WHEELS=${WHEELS:-wheels}   # wheel folder (relative to device/)

pyver() {  # "Python 3.13.5" | "3.13.5" | "3.13" -> "3.13"
  local v=${1#Python }
  v=${v// /}
  echo "$v" | cut -d. -f1,2
}

case "${1:-install}" in
  stage)
    PV=$(pyver "${PYVER:-3.13}")
    [[ "$PV" =~ ^3\.[0-9]+$ ]] || { echo "PYVER '$PYVER' is not a Python version (e.g. PYVER=3.13)" >&2; exit 2; }
    echo "staging aarch64 wheels for Python $PV (set PYVER to the board's python3 --version if it differs)"
    mkdir -p "$WHEELS"
    python3 -m pip download --only-binary=:all: "${PLATFORMS[@]}" --implementation cp --python-version "$PV" \
      -d "$WHEELS" "${PINS[@]}"
    echo "$PV" > "$WHEELS/PYVER" ;;
  install)
    have=$(pyver "$(python3 --version 2>&1)")
    want=$(cat "$WHEELS/PYVER" 2>/dev/null || echo "unknown")
    if [ "$have" != "$want" ]; then
      echo "$WHEELS/ was staged for Python $want but this board runs Python $have: re-stage with PYVER=$have" >&2
      exit 2
    fi
    python3 -m venv .venv && .venv/bin/pip install --no-index --find-links "$WHEELS" "${PINS[@]}"
    echo "run: .venv/bin/python uno_q_app.py --bundle /path/to/bundle --mode camera --device /dev/video0 --port 8080" ;;
  *) echo "usage: [PYVER=3.13] $0 stage|install"; exit 2 ;;
esac
