"""K1 fused dual-preprocess: parity with each reader's official PIL transform (Review Focus 1 and 3).

Error unit: one uint8 level divided by the channel std, i.e. |k1 - ref| * 255 * std. The official transforms resize
PIL images (PIL ImagingResample: bicubic a=-0.5, uint8 intermediate), so a correct K1 is bit-exact up to the fp16
cast of the normalized value.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from twinread.readers import PreprocessSpec, _compose, spec_of

FIX = Path(__file__).parent / "fixtures"
CONCH, MUSK = spec_of("conch"), spec_of("musk")
# off-table specs that exercise the other code paths: crop % 4 != 0 (scalar vertical kernel) and PIL bilinear
ODD = PreprocessSpec(230, 225, "bicubic", spec_of("conch").mean, spec_of("conch").std)
BIL = PreprocessSpec(256, 224, "bilinear", (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def _crc16() -> np.ndarray:
    a = np.load(FIX / "crc16.npy")
    assert a.shape == (16, 224, 224, 3) and a.dtype == np.uint8
    return a


def _mosaics16(tiles: np.ndarray) -> np.ndarray:
    """16 non-square 400x448 inputs of real tissue: 2x2 mosaics of the CRC tiles, rows 24..424."""
    out = []
    for i in range(16):
        t = [tiles[(i + k) % 16] for k in range(4)]
        m = np.concatenate([np.concatenate(t[:2], 1), np.concatenate(t[2:], 1)], 0)
        out.append(m[24:424])
    return np.stack(out)


def _official(name: str):
    spec = spec_of(name)
    if name == "conch":  # CONCH's own image_transform (what create_model_from_pretrained returns), no weights
        from twinread.compat import patch_conch
        patch_conch()
        from conch.open_clip_custom.transform import image_transform
        return image_transform(spec.size, mean=spec.mean, std=spec.std)
    return _compose(spec)  # MUSK README transform (Resize 384 bicubic antialias, CenterCrop, ToTensor, Normalize)


def _ref(arr: np.ndarray, name: str) -> torch.Tensor:
    tf = _official(name)
    return torch.stack([tf(Image.fromarray(a)) for a in arr])


def _levels(out: torch.Tensor, ref: torch.Tensor, spec) -> float:
    std = torch.tensor(spec.std, dtype=torch.float32).view(1, 3, 1, 1)
    return ((out.float().cpu() - ref.float()).abs() * 255.0 * std).max().item()


@pytest.fixture(scope="module")
def ext():
    from twinread.kernels import load_ext
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    m = load_ext()
    if m is None:
        pytest.skip("twinread CUDA extension unavailable")
    return m


def _check_pair(arr, a, b, ctx=""):
    n = len(arr)
    assert a.shape == (n, 3, CONCH.crop, CONCH.crop) and a.dtype == torch.float16, ctx
    assert b.shape == (n, 3, MUSK.crop, MUSK.crop) and b.dtype == torch.float16, ctx
    assert torch.isfinite(a).all() and torch.isfinite(b).all(), ctx
    la, lb = _levels(a, _ref(arr, "conch"), CONCH), _levels(b, _ref(arr, "musk"), MUSK)
    assert la <= 1.0, f"{ctx} conch max error {la:.3f} levels"
    assert lb <= 1.0, f"{ctx} musk max error {lb:.3f} levels"
    return la, lb


# ----------------------------------------------------------------------------------------------- GPU (CUDA K1)
@pytest.mark.gpu
def test_k1_matches_pil_square(ext):
    from twinread.kernels import dual_preprocess
    arr = _crc16()
    a, b = dual_preprocess(torch.from_numpy(arr).cuda(), CONCH, MUSK, impl="cuda")
    assert a.is_cuda and b.is_cuda
    _check_pair(arr, a, b, "crc16")


@pytest.mark.gpu
@pytest.mark.parametrize("hw", [(480, 640), (720, 1280), (640, 480), (512, 512), (225, 301)])
def test_k1_nonsquare_matches_reference(ext, hw):
    """Camera frames (landscape and portrait), a tile larger than 224 and an odd size: shorter side -> size,
    center crop, same result as the reference transform."""
    from twinread.kernels import dual_preprocess
    rng = np.random.default_rng(hw[0] * 7 + hw[1])
    arr = rng.integers(0, 256, (2, *hw, 3), dtype=np.uint8)
    a, b = dual_preprocess(torch.from_numpy(arr).cuda(), CONCH, MUSK, impl="cuda")
    _check_pair(arr, a, b, f"{hw}")


@pytest.mark.gpu
@pytest.mark.parametrize("impl", ["cuda", "triton", "torch"])
def test_k1_bit_exact_vs_official_half(ext, impl):
    """Stronger than the 1-level tolerance: every element equals official(img).half() exactly (real tiles and a
    non-square frame)."""
    from twinread.kernels import dual_preprocess
    if impl == "triton":
        pytest.importorskip("triton")
    rng = np.random.default_rng(2)
    for arr in (_crc16()[:6], rng.integers(0, 256, (2, 480, 640, 3), dtype=np.uint8)):
        a, b = dual_preprocess(torch.from_numpy(arr).cuda(), CONCH, MUSK, impl=impl)
        assert torch.equal(a.cpu(), _ref(arr, "conch").half()), impl
        assert torch.equal(b.cpu(), _ref(arr, "musk").half()), impl


@pytest.mark.gpu
def test_k1_fallback_kernels_exact(ext, monkeypatch):
    """The CUDA scalar kernels (used for frames wider than ~16k px and for crop % 4 != 0) and the bilinear filter are
    exact too: forced scalar for CONCH/MUSK, and the auto path for odd-crop / bilinear specs."""
    from twinread import kernels
    rng = np.random.default_rng(3)
    sets = (_crc16()[:4], rng.integers(0, 256, (2, 480, 640, 3), dtype=np.uint8))
    for arr in sets:
        x = torch.from_numpy(arr).cuda()
        a, b = kernels.dual_preprocess(x, ODD, BIL, impl="cuda")  # fused hpass + scalar vpass
        assert torch.equal(a.cpu(), torch.stack([_compose(ODD)(Image.fromarray(im)) for im in arr]).half())
        assert torch.equal(b.cpu(), torch.stack([_compose(BIL)(Image.fromarray(im)) for im in arr]).half())
        monkeypatch.setattr(kernels, "_FORCE_SCALAR", True)
        try:
            a, b = kernels.dual_preprocess(x, CONCH, MUSK, impl="cuda")
        finally:
            monkeypatch.setattr(kernels, "_FORCE_SCALAR", False)
        assert torch.equal(a.cpu(), _ref(arr, "conch").half())
        assert torch.equal(b.cpu(), _ref(arr, "musk").half())


@pytest.mark.gpu
@pytest.mark.parametrize("n", [0, 1, 3, 33])
def test_k1_batch_tails(ext, n):
    from twinread.kernels import dual_preprocess
    base = _crc16()
    arr = np.concatenate([base, base[::-1], base[:1] // 2 + 7])[:n] if n else base[:0]
    x = torch.from_numpy(np.ascontiguousarray(arr)).cuda()
    a, b = dual_preprocess(x, CONCH, MUSK, impl="cuda")
    assert a.shape == (n, 3, 448, 448) and b.shape == (n, 3, 384, 384)
    assert a.dtype == b.dtype == torch.float16 and a.is_cuda and b.is_cuda
    if n == 0:
        return
    # every batch element equals the same image processed alone (no cross-batch indexing errors)
    for i in sorted({0, n // 2, n - 1}):
        ai, bi = dual_preprocess(x[i:i + 1], CONCH, MUSK, impl="cuda")
        assert torch.equal(ai[0], a[i]) and torch.equal(bi[0], b[i])
    _check_pair(arr, a, b, f"B={n}")


@pytest.mark.gpu
def test_k1_embedding_cosine(ext):
    """CONCH (fp32) and MUSK (fp16) aligned embeddings from K1 vs the reader's official transform: min cos >= 0.999
    on 32 real-tissue inputs (16 CRC tiles + 16 non-square 400x448 mosaics)."""
    from twinread.kernels import dual_preprocess
    from twinread.readers import load_reader
    tiles = _crc16()
    groups = [tiles, _mosaics16(tiles)]
    k1 = {"conch": [], "musk": []}
    for g in groups:
        a, b = dual_preprocess(torch.from_numpy(g).cuda(), CONCH, MUSK, impl="cuda")
        k1["conch"].append(a.cpu())
        k1["musk"].append(b.cpu())
    for name in ("conch", "musk"):
        r = load_reader(name)
        try:
            mine, ref = [], []
            for g, kb in zip(groups, k1[name]):
                rb = torch.stack([r.preprocess_pil(Image.fromarray(im)) for im in g])
                for s in range(0, len(g), 16):
                    mine.append(r.encode_aligned(kb[s:s + 16].float()))
                    ref.append(r.encode_aligned(rb[s:s + 16]))
            mine, ref = np.concatenate(mine), np.concatenate(ref)
        finally:
            r.close()
        assert mine.shape[0] == 32
        cos = (mine * ref).sum(1) / np.linalg.norm(mine, axis=1) / np.linalg.norm(ref, axis=1)
        assert cos.min() >= 0.999, f"{name} min cosine {cos.min():.6f}"


@pytest.mark.gpu
def test_triton_and_torch_paths_agree(ext):
    from twinread.kernels import dual_preprocess
    pytest.importorskip("triton")
    rng = np.random.default_rng(1)
    for arr in (_crc16(), rng.integers(0, 256, (3, 480, 640, 3), dtype=np.uint8)):
        x = torch.from_numpy(arr).cuda()
        ca, cb = dual_preprocess(x, CONCH, MUSK, impl="cuda")
        for impl in ("triton", "torch"):
            a, b = dual_preprocess(x, CONCH, MUSK, impl=impl)
            assert a.is_cuda and a.dtype == torch.float16 and a.shape == ca.shape and b.shape == cb.shape
            assert _levels(a, ca.float().cpu(), CONCH) <= 1.0, impl
            assert _levels(b, cb.float().cpu(), MUSK) <= 1.0, impl


# ----------------------------------------------------------------------------------------------- CPU-safe
def test_fallback_when_ext_missing(monkeypatch):
    from twinread import kernels
    monkeypatch.setattr(kernels, "load_ext", lambda *a, **k: None)
    monkeypatch.setattr(kernels, "_WARNED", set())
    arr = _crc16()[:2]
    x = torch.from_numpy(arr)  # CPU tensor: neither the extension nor Triton can run, torch must
    with pytest.warns(RuntimeWarning, match="falling back"):
        a, b = kernels.dual_preprocess(x, CONCH, MUSK, impl="cuda")
    ta, tb = kernels.dual_preprocess(x, CONCH, MUSK, impl="torch")
    assert torch.equal(a, ta) and torch.equal(b, tb)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # warns once: a second call is silent
        kernels.dual_preprocess(x, CONCH, MUSK, impl="cuda")
    _check_pair(arr, a, b, "torch-cpu")


@pytest.mark.parametrize("hw", [(224, 224), (480, 640), (301, 225), (97, 513)])
def test_torch_resample_bit_exact_vs_pil(hw):
    """The shared coefficients + fixed-point math reproduce PIL Image.resize + center crop exactly (uint8)."""
    from twinread.kernels import resize_crop_uint8
    rng = np.random.default_rng(hw[0] + hw[1])
    arr = rng.integers(0, 256, (1, *hw, 3), dtype=np.uint8)
    for spec in (CONCH, MUSK, ODD, BIL):
        tf = (_official("conch") if spec is CONCH else _compose(spec)).transforms[:2]  # Resize + CenterCrop on PIL
        ref = np.asarray(tf[1](tf[0](Image.fromarray(arr[0]))))
        mine = resize_crop_uint8(torch.from_numpy(arr), spec)[0].numpy()
        assert mine.shape == ref.shape == (spec.crop, spec.crop, 3)
        assert np.array_equal(mine, ref), f"{hw} {spec.size}: {np.abs(mine.astype(int) - ref).max()} levels"


@pytest.mark.parametrize("hw", [(1, 1), (1, 2000), (2000, 3), (7, 5), (448, 448), (720, 1280), (4000, 300)])
def test_plan_reads_stay_in_bounds(hw):
    """Every tap the kernels read lies inside the input frame and the intermediate (no out-of-bounds reads)."""
    from twinread.kernels import make_plan
    H, W = hw
    for spec in (CONCH, MUSK):
        p = make_plan(H, W, spec, "cpu")
        hb, vb = p.hb.numpy(), p.vb.numpy()
        assert hb.shape == (spec.crop, 2) and vb.shape == (spec.crop, 2)
        assert (hb[:, 0] >= 0).all() and (hb[:, 1] >= 1).all() and (hb.sum(1) <= W).all()
        assert p.row0 >= 0 and p.row0 + p.rows <= H
        assert (vb[:, 0] >= 0).all() and (vb[:, 1] >= 1).all() and (vb.sum(1) <= p.rows).all()
        assert p.hk.shape == (spec.crop, p.kh) and p.vk.shape == (spec.crop, p.kv)


def test_rejects_bad_input():
    from twinread.kernels import dual_preprocess
    with pytest.raises(ValueError):
        dual_preprocess(torch.zeros(1, 8, 8, 3, dtype=torch.float32), CONCH, MUSK, impl="torch")
    with pytest.raises(ValueError):
        dual_preprocess(torch.zeros(1, 8, 8, 4, dtype=torch.uint8), CONCH, MUSK, impl="torch")
    with pytest.raises(ValueError):
        dual_preprocess(torch.zeros(8, 8, 3, dtype=torch.uint8), CONCH, MUSK, impl="torch")
