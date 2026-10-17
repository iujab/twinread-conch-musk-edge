import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "device"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _gpu_ok():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if _gpu_ok():
        return
    skip = pytest.mark.skip(reason="CUDA unavailable")
    for it in items:
        if "gpu" in it.keywords:
            it.add_marker(skip)
