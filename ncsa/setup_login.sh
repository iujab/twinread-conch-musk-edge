#!/bin/bash
# Run ON THE DELTA LOGIN NODE (it has internet; compute nodes do not). Run only after the user's explicit OK.
# Installs into the user site (~/.local). Never touches system Python.
set -euo pipefail
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE

pip install --user "open_clip_torch>=2.24" onnx onnxruntime
# CONCH and MUSK: --no-deps so their pins cannot replace Delta's torch/transformers.
pip install --user --no-deps "git+https://github.com/mahmoodlab/CONCH.git"
pip install --user --no-deps "git+https://github.com/lilab-stanford/MUSK.git"
pip install --user fairscale sentencepiece

python - <<'PY'
import torch, transformers, open_clip
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__, "(twinread.compat shims are no-ops when < 5)")
print("open_clip", open_clip.__version__)
for m in ("conch", "musk", "fairscale", "sentencepiece", "onnx", "onnxruntime"):
    try:
        __import__(m); print(m, "ok")
    except Exception as e:
        print(m, "FAILED:", e)
PY
