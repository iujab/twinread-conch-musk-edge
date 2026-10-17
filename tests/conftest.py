"""Shared test setup.

GPU tests (``@pytest.mark.gpu``) run ONLY when ``TWINREAD_GPU_TESTS=1`` and CUDA is visible, so the bench's default
command (``pytest <slug>/tests -q``) never loads CONCH/MUSK onto the shared GPU without the lock. Run them with
    flock $B/.gpu.lock env TWINREAD_GPU_TESTS=1 PATH=/usr/local/cuda/bin:$PATH $B/.venv/bin/python -m pytest $P/tests -q -m gpu
Tests that need reader weights call the ``require_weights`` fixture: they skip when the weights are not in the local
HF cache, or fail when ``TWINREAD_REQUIRE_WEIGHTS=1`` (NCSA J1, where a missing cache must be caught before J2).
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "device"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

GPU_ENV = "TWINREAD_GPU_TESTS"
REQUIRE_WEIGHTS_ENV = "TWINREAD_REQUIRE_WEIGHTS"
# reader -> (HF repo, files that must ALL be cached; a tuple entry means any one of them)
WEIGHT_FILES = {
    "conch": ("MahmoodLab/CONCH", ["pytorch_model.bin"]),
    "musk": ("xiangjx/musk", ["model.safetensors", "tokenizer.spm"]),
    "pathgenclip": ("jamessyx/pathgenclip-vit-large-patch14-hf",
                    ["config.json", ("model.safetensors", "pytorch_model.bin")]),
    "quiltnet_b16": ("wisdomik/QuiltNet-B-16", ["open_clip_config.json", "open_clip_pytorch_model.bin"]),
    "biomedclip": ("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                   ["open_clip_config.json", "open_clip_pytorch_model.bin"]),
}


def _gpu_ok():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def gpu_tests_enabled() -> bool:
    return os.environ.get(GPU_ENV) == "1"


def pytest_collection_modifyitems(config, items):
    if gpu_tests_enabled() and _gpu_ok():
        return
    reason = (f"GPU tests run only with {GPU_ENV}=1 under the GPU lock" if not gpu_tests_enabled()
              else "CUDA unavailable")
    skip = pytest.mark.skip(reason=reason)
    for it in items:
        if "gpu" in it.keywords:
            it.add_marker(skip)


def missing_weights(name: str) -> list:
    """Files of reader ``name`` that are not in the local HF hub cache (empty list = all cached)."""
    from huggingface_hub import try_to_load_from_cache
    repo, files = WEIGHT_FILES[name]
    missing = []
    for f in files:
        alts = f if isinstance(f, tuple) else (f,)
        if not any(isinstance(try_to_load_from_cache(repo, a), str) for a in alts):
            missing.append(f"{repo}/{'|'.join(alts)}")
    return missing


@pytest.fixture
def require_weights():
    """``require_weights("conch", "musk")``: skip (or fail with TWINREAD_REQUIRE_WEIGHTS=1) unless cached."""
    def _require(*names):
        miss = [m for n in names for m in missing_weights(n)]
        if miss:
            msg = f"reader weights not in the HF cache: {', '.join(miss)}"
            if os.environ.get(REQUIRE_WEIGHTS_ENV) == "1":
                pytest.fail(msg)
            pytest.skip(msg)
    return _require
