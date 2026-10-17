#!/usr/bin/env bash
# Install the TwinRead device app on the UNO Q (Debian aarch64). Stage wheels once on a machine WITH internet, copy the
# folder to the board; the board itself never needs the network.
#   stage (online host):  ./install_offline.sh stage     -> wheels/ (aarch64: numpy, onnxruntime, opencv-python-headless, pillow)
#   install (on board):   ./install_offline.sh install   -> .venv installed with --no-index from wheels/
# Set PYVER to the board's Python (default 3.11). Unverified on a board.
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-install}" in
  stage) pip download --only-binary=:all: --platform manylinux2014_aarch64 --python-version "${PYVER:-3.11}" -d wheels \
           numpy onnxruntime opencv-python-headless pillow ;;
  install) python3 -m venv .venv && .venv/bin/pip install --no-index --find-links wheels numpy onnxruntime opencv-python-headless pillow
           echo "run: .venv/bin/python uno_q_app.py --bundle /path/to/bundle --mode camera --device /dev/video0 --port 8080" ;;
  *) echo "usage: $0 stage|install"; exit 2 ;;
esac
