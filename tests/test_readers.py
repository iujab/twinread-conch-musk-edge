import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torchvision import transforms as T

from twinread.readers import READER_NAMES, load_reader, spec_of

FIX = Path(__file__).parent / "fixtures" / "tiles4.npy"
BENCH = Path(__file__).resolve().parents[2]
DIMS = {  # name -> (dim_aligned, dim_probe)
    "conch": (512, 512), "musk": (1024, 2048), "pathgenclip": (768, 1024),
    "quiltnet_b16": (512, 768), "biomedclip": (512, 768),
}


def test_spec_of_no_weights():
    assert spec_of("conch").size == 448
    assert spec_of("musk").mean == (0.5, 0.5, 0.5)
    assert set(READER_NAMES) == set(DIMS)
    with pytest.raises(KeyError):
        spec_of("nope")


def _tiles_pil():
    from PIL import Image
    return [Image.fromarray(a) for a in np.load(FIX)]


@pytest.mark.gpu
@pytest.mark.parametrize("name", READER_NAMES)
def test_spec_table_matches_loaded_transform(name, require_weights):
    require_weights(name)
    spec = spec_of(name)
    r = load_reader(name)
    try:
        ts = r.preprocess_pil.transforms
        rs = [t for t in ts if isinstance(t, T.Resize)]
        cc = [t for t in ts if isinstance(t, T.CenterCrop)]
        nm = [t for t in ts if isinstance(t, T.Normalize)]
        assert rs and cc and nm
        assert isinstance(rs[0].size, int), "Resize must be shortest-edge (int), not a squash"
        assert rs[0].size == spec.size
        csz = cc[0].size
        csz = csz[0] if isinstance(csz, (list, tuple)) else csz
        assert csz == spec.crop
        assert tuple(round(float(x), 6) for x in nm[0].mean) == tuple(round(x, 6) for x in spec.mean)
        assert tuple(round(float(x), 6) for x in nm[0].std) == tuple(round(x, 6) for x in spec.std)
        assert rs[0].interpolation.value == spec.resample
    finally:
        r.close()


@pytest.mark.gpu
@pytest.mark.parametrize("name", READER_NAMES)
def test_reader_dims(name, require_weights):
    require_weights(name)
    r = load_reader(name)
    try:
        batch = torch.stack([r.preprocess_pil(im) for im in _tiles_pil()])
        a, p = r.encode_aligned(batch), r.encode_probe(batch)
        da, dp = DIMS[name]
        assert (r.dim_aligned, r.dim_probe) == (da, dp)
        assert a.shape == (4, da) and p.shape == (4, dp)
        assert a.dtype == np.float32 and p.dtype == np.float32
        assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-3)
        t = r.encode_texts(["colorectal adenocarcinoma", "normal colonic mucosa"])
        assert t.shape == (2, da) and np.allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-3)
    finally:
        r.close()


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["conch", "musk"])
def test_matches_organizer_adapter(name, require_weights):
    require_weights(name)
    if not (BENCH / "ashby-hackathon-2026" / "models.py").exists():
        pytest.skip("organizer adapter ashby-hackathon-2026/models.py not present next to the project")
    import transformers
    from twinread.compat import XLMRSpTokenizer, patch_conch
    patch_conch()
    sys.path.insert(0, str(BENCH / "ashby-hackathon-2026"))
    import models as org
    old = getattr(transformers, "XLMRobertaTokenizer", None)
    transformers.XLMRobertaTokenizer = XLMRSpTokenizer
    try:
        ad = (org.load_conch if name == "conch" else org.load_musk)("cuda")
        batch = torch.stack([ad.preprocess(im) for im in _tiles_pil()])
        ref = ad.encode_images(batch)
        ref_t = ad.encode_texts(["colorectal adenocarcinoma", "normal colonic mucosa"])
        del ad
        torch.cuda.empty_cache()
    finally:
        if old is not None:
            transformers.XLMRobertaTokenizer = old
    r = load_reader(name)
    try:
        mine = r.encode_aligned(batch)
        mine_t = r.encode_texts(["colorectal adenocarcinoma", "normal colonic mucosa"])
    finally:
        r.close()
    cos = lambda x, y: (x * y).sum(1) / np.linalg.norm(x, axis=1) / np.linalg.norm(y, axis=1)
    assert cos(mine, ref).min() >= 0.9999
    assert cos(mine_t, ref_t).min() >= 0.9999


@pytest.mark.gpu
def test_close_frees_gpu_memory(require_weights):
    require_weights("quiltnet_b16")
    before = torch.cuda.memory_allocated()
    r = load_reader("quiltnet_b16")
    assert torch.cuda.memory_allocated() > before
    r.close()
    assert torch.cuda.memory_allocated() <= before + 1e6


@pytest.mark.gpu
@pytest.mark.parametrize("name", READER_NAMES)
def test_spec_numerically_matches_official_transform(name, require_weights):
    require_weights(name)
    """Pin spec_of to each model's loaded transform on a non-square image (300x500)."""
    from PIL import Image
    from twinread.readers import _compose
    rng = np.random.default_rng(0)
    im = Image.fromarray(rng.integers(0, 256, (300, 500, 3), dtype=np.uint8))
    r = load_reader(name)
    try:
        official = r.preprocess_pil(im)
        mine = _compose(spec_of(name))(im)
        assert official.shape == mine.shape == (3, spec_of(name).crop, spec_of(name).crop)
        assert (official - mine).abs().max().item() < 1e-2
        if name == "pathgenclip":
            from transformers import CLIPProcessor
            ref = CLIPProcessor.from_pretrained("jamessyx/pathgenclip-vit-large-patch14-hf")(
                images=im, return_tensors="pt")["pixel_values"][0]
            assert ref.shape == official.shape
            # HF resizes in uint8 (rounding) while torchvision resizes float: a few quantisation levels at most
            d = (ref - official).abs()
            assert d.max().item() < 0.05 and d.mean().item() < 2e-3
    finally:
        r.close()
