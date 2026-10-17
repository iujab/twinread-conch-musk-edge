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
# MUSK vendors torchscale (musk/torchscale): do NOT pip-install torchscale. musk.modeling needs einops, safetensors, timm.
python -c "import einops" 2>/dev/null || pip install --user einops
python -c "import safetensors" 2>/dev/null || pip install --user safetensors

python - <<'PY'
import sys
import torch, transformers, open_clip, timm
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__, "(twinread.compat shims are no-ops when < 5)")
print("open_clip", open_clip.__version__)
print("timm", timm.__version__)
bad = []
for m in ("conch", "fairscale", "sentencepiece", "onnx", "onnxruntime", "einops", "safetensors", "musk.modeling"):
    try:
        __import__(m); print(m, "ok")
    except Exception as e:
        bad.append(m); print(m, "FAILED:", repr(e))
try:
    import musk.modeling  # noqa: F401  registers musk_* with timm
    timm.create_model("musk_large_patch16_384", pretrained=False, num_classes=0)
    print("timm.create_model musk_large_patch16_384 ok (no weights)")
except Exception as e:
    bad.append("musk_large_patch16_384"); print("musk_large_patch16_384 FAILED:", repr(e))
sys.exit(1 if bad else 0)
PY
