"""K2 camera-of-screen augmentation (simulation): CUDA kernel and torch fallback vs the numpy references in
``twinread.ref.camera_ops`` (Review Focus 2 and 3).

Error unit: uint8 levels. Tolerances, and why each one exists:
* exact (``torch.equal``): identity params, determinism, batch independence, Philox bits, IJG tables;
* <= 1 level, one op at a time: the kernel computes in float32, the reference in float64, so a value can land on
  the other side of the final uint8 rounding boundary;
* composition (brief): mean <= 2 levels, max <= 8 levels on the fixture tiles. The max is NOT a universal bound:
  when a DCT coefficient lies within float32 error of a JPEG quantization boundary, float32 and float64 round it
  differently and the decoded block differs by up to Q * |basis| (bench/bench_k2.py --part parity found 11 levels
  in 2 of 16 runs of 64 tiles). So every composition, odd-size, batch-tail and uniform test also runs the
  decomposed check ``_Decomposed`` (ruling R29; ``camera_ops.decompose_differences``):
  (i)  pre-JPEG: |implementation float32 JPEG input - ref_pre_jpeg| <= EPS_PRE_JPEG[impl] levels at every value.
       CUDA 0.02 (worst measured 0.0079 over 16 runs x 64 tiles). torch 0.2 (worst measured 0.098, at a near-black
       input pixel [6, 0, 0]; elsewhere <= 0.008): grid_sample unnormalizes the warp coordinates in its own float32
       order (source coordinates differ by ~1e-5 px from the reference's) and the HED log(max(v, 1e-6)) that follows
       has slope 1/v, so where an input channel is ~0 that difference is amplified ~100x. Both are >= 2x the worst
       measured value and catch any regression of a level or more.
  (ii) JPEG: the output equals ``camera_ops.jpeg`` applied to the implementation's OWN float32 JPEG input within 1
       level in every 8x8 block, except blocks with a coefficient within JPEG_TIE_TOL (1e-4 levels) of a rounding
       boundary on that input; the exempt fraction must stay <= 0.1 % of the test's blocks.
Research prototype, not a diagnostic device.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

FIX = Path(__file__).parent / "fixtures"

OP_KEYS = {
    "warp": ["warp"],
    "hed": ["hed_alpha", "hed_beta"],
    "display": ["gamma", "profile"],
    "moire": ["moire_amp", "moire_period", "moire_angle"],
    "glare": ["glare_amp", "glare_angle"],
    "defocus": ["blur_sigma"],
    "white_balance": ["wb_gain"],
    "noise": ["noise_std"],
    "jpeg": ["jpeg_quality"],
}


def _tiles(n: int = 8) -> np.ndarray:
    a = np.load(FIX / "crc16.npy")
    assert a.shape == (16, 224, 224, 3) and a.dtype == np.uint8
    return np.ascontiguousarray(a[:n])


def _ref_batch(arr: np.ndarray, dicts: list[dict], seed: int) -> np.ndarray:
    from twinread.ref.camera_ops import ref_camera_aug
    return np.stack([ref_camera_aug(im, dict(d, seed=seed, index=i)) for i, (im, d) in enumerate(zip(arr, dicts))])


EPS_PRE_JPEG = {"cuda": 0.02, "torch": 0.2}  # levels; justification in the module docstring
MAX_EXEMPT_FRAC = 1e-3


class _Decomposed:
    """The decomposed check (module docstring), accumulated over one test; ``finish`` asserts the exempt fraction."""

    def __init__(self):
        self.exempt = self.blocks = 0
        self.pre_max: dict[str, float] = {}

    def check(self, arr: np.ndarray, inter: torch.Tensor, out: torch.Tensor, dicts: list[dict], seed: int,
              impl: str) -> None:
        from twinread.ref.camera_ops import decompose_differences, ref_pre_jpeg
        inter, out = inter.cpu().numpy(), out.cpu().numpy()
        for i, (im, d) in enumerate(zip(arr, dicts)):
            d = dict(d, seed=seed, index=i)
            r = decompose_differences(out[i], inter[i], ref_pre_jpeg(im, d), d)
            assert r["pre_jpeg_max_diff_levels"] <= EPS_PRE_JPEG[impl], (impl, i, r)  # (i)
            assert r["jpeg_max_err"] <= 1, (impl, i, r)  # (ii), non-exempt blocks
            self.exempt += r["n_exempt"]
            self.blocks += r["n_blocks"]
            self.pre_max[impl] = max(self.pre_max.get(impl, 0.0), r["pre_jpeg_max_diff_levels"])

    def finish(self) -> None:
        assert self.blocks > 0
        assert self.exempt / self.blocks <= MAX_EXEMPT_FRAC, f"exempt {self.exempt}/{self.blocks} blocks"


def _err(out: torch.Tensor, ref: np.ndarray) -> tuple[float, float]:
    d = np.abs(out.cpu().numpy().astype(np.int32) - ref.astype(np.int32))
    return float(d.mean()), int(d.max())


def _op_dicts(op: str, n: int) -> list[dict]:
    """Identity params except ``op``, which takes severity-1 draws (plus fixed extremes for coverage)."""
    from twinread.ref.camera_ops import identity_params, sample_params
    out = []
    for i in range(n):
        d = identity_params()
        s = sample_params(1.0, seed=100 + i)
        for k in OP_KEYS[op]:
            d[k] = s[k]
        out.append(d)
    if op == "jpeg":  # the whole IJG quality range, including the q < 50 branch and q = 100
        for i, q in enumerate([10, 23, 37, 50, 64, 79, 95, 100][:n]):
            out[i]["jpeg_quality"] = float(q)
    if op == "defocus":
        out[-1]["blur_sigma"] = 4.0  # the largest supported sigma (radius 12)
    if op == "noise":
        out[-1]["noise_std"] = 0.05
    if op == "warp":
        out[-1]["warp"] = [0.05, -0.08, 0.06, 0.07, -0.04, -0.05, 0.06, -0.06]
    return out


@pytest.fixture(scope="module")
def ext():
    from twinread.kernels import load_k2_ext
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    m = load_k2_ext()
    if m is None:
        pytest.skip("twinread K2 CUDA extension unavailable")
    return m


# ----------------------------------------------------------------------------------------------- GPU (CUDA K2)
@pytest.mark.gpu
def test_identity_params_are_identity(ext):
    from twinread.kernels import CameraAugParams, camera_aug, sample_camera_params
    rng = np.random.default_rng(0)
    for arr in (_tiles(8), rng.integers(0, 256, (3, 61, 83, 3), dtype=np.uint8)):
        x = torch.from_numpy(arr).cuda()
        p = sample_camera_params(len(arr), 0.0, seed=3, device="cuda")
        for impl in ("cuda", "torch"):
            assert torch.equal(camera_aug(x, p, seed=11, impl=impl), x), impl
        assert torch.equal(camera_aug(x, CameraAugParams.identity(len(arr), "cuda"), seed=11), x)
    # JPEG at quality 100 is not skipped: every table entry is 1, so only coefficient rounding (+-0.5 per
    # coefficient, mixed by YCbCr -> RGB) remains. Measured on these tiles: mean 0.28 levels, 99.6 % of values
    # within 1 level, max 3; the brief's "<= 1 level" holds for the mean and for >= 99 % of values, not the max.
    from twinread.ref.camera_ops import identity_params
    arr = _tiles(8)
    x = torch.from_numpy(arr).cuda()
    d = [dict(identity_params(), jpeg_quality=100.0) for _ in range(8)]
    out = camera_aug(x, CameraAugParams.from_dicts(d, "cuda"), seed=0)
    mean, mx = _err(out, arr)
    within1 = (np.abs(out.cpu().numpy().astype(int) - arr) <= 1).mean()
    assert mean <= 1.0 and within1 >= 0.99 and mx <= 3, (mean, within1, mx)
    assert _err(out, _ref_batch(arr, d, 0))[1] <= 1


@pytest.mark.gpu
@pytest.mark.parametrize("op", list(OP_KEYS))
def test_each_op_matches_numpy(ext, op):
    from twinread.kernels import CameraAugParams, camera_aug
    arr = _tiles(8)
    dicts = _op_dicts(op, 8)
    out = camera_aug(torch.from_numpy(arr).cuda(), CameraAugParams.from_dicts(dicts, "cuda"), seed=7, impl="cuda")
    assert out.dtype == torch.uint8 and out.shape == arr.shape and out.is_cuda
    ref = _ref_batch(arr, dicts, 7)
    mean, mx = _err(out, ref)
    assert mx <= 1, f"{op}: max {mx} levels (mean {mean:.4f})"
    assert (ref != arr).mean() > 0.05, f"{op}: the op changed almost nothing; the test would be vacuous"


@pytest.mark.gpu
@pytest.mark.parametrize("severity", [0.5, 1.0])
def test_composition_matches_numpy(ext, severity):
    from twinread.kernels import _camera_aug_impl, sample_camera_params
    arr = _tiles(8)
    p = sample_camera_params(8, severity, seed=21, device="cuda")
    dc = _Decomposed()
    for impl in ("cuda", "torch"):
        out, inter = _camera_aug_impl(torch.from_numpy(arr).cuda(), p, 5, impl=impl, return_float=True)
        mean, mx = _err(out, _ref_batch(arr, p.to_dicts(), 5))
        assert mean <= 2.0 and mx <= 8, f"{impl} severity {severity}: mean {mean:.4f} max {mx}"
        dc.check(arr, inter, out, p.to_dicts(), 5, impl)
    dc.finish()


@pytest.mark.gpu
def test_composition_differences_are_jpeg_rounding_flips(ext):
    """Wider than the brief's fixture check: 16 tiles x 3 severities x 2 param seeds (96 images), both impls. Mean
    <= 2 levels against the full reference; no max is asserted against it. Instead the decomposed check: pre-JPEG
    within EPS_PRE_JPEG everywhere, and the output within 1 level of the reference JPEG of the implementation's own
    input in every block except the <= 0.1 % within JPEG_TIE_TOL of a rounding boundary."""
    from twinread.kernels import _camera_aug_impl, sample_camera_params
    arr = _tiles(16)
    x = torch.from_numpy(arr).cuda()
    dc = _Decomposed()
    for sev in (0.5, 0.75, 1.0):
        for ps in (1000, 1002):
            p = sample_camera_params(16, sev, seed=ps, device="cuda")
            for impl in ("cuda", "torch"):
                out, inter = _camera_aug_impl(x, p, 1234, impl=impl, return_float=True)
                mean, _ = _err(out, _ref_batch(arr, p.to_dicts(), 1234))
                assert mean <= 2.0, (impl, sev, ps, mean)
                dc.check(arr, inter, out, p.to_dicts(), 1234, impl)
    dc.finish()


@pytest.mark.gpu
@pytest.mark.parametrize("hw", [(37, 53), (61, 83), (480, 640), (232, 224)])
def test_k2_odd_sizes_match_numpy(ext, hw):
    """H, W not multiples of the 8x8 JPEG block or the 32x16 kernel tile (edge padding inside the block)."""
    from twinread.kernels import _camera_aug_impl, sample_camera_params
    rng = np.random.default_rng(hw[0] * 3 + hw[1])
    t = _tiles(16)
    reps = (-(-hw[0] // 224), -(-hw[1] // 224))
    arr = np.stack([np.tile(t[i], (*reps, 1))[:hw[0], :hw[1]] for i in rng.choice(16, 2, replace=False)])
    arr = np.ascontiguousarray(arr)
    p = sample_camera_params(2, 1.0, seed=hw[1], device="cuda")
    out, inter = _camera_aug_impl(torch.from_numpy(arr).cuda(), p, 9, impl="cuda", return_float=True)
    assert out.shape == arr.shape
    mean, mx = _err(out, _ref_batch(arr, p.to_dicts(), 9))
    assert mean <= 2.0 and mx <= 8, f"{hw}: mean {mean:.4f} max {mx}"
    dc = _Decomposed()
    dc.check(arr, inter, out, p.to_dicts(), 9, "cuda")
    dc.finish()


@pytest.mark.gpu
def test_deterministic_given_seed(ext):
    from twinread.kernels import camera_aug, sample_camera_params
    x = torch.from_numpy(_tiles(8)).cuda()
    p = sample_camera_params(8, 1.0, seed=4, device="cuda")
    a = camera_aug(x, p, seed=17)
    assert torch.equal(a, camera_aug(x, p, seed=17))
    assert not torch.equal(a, camera_aug(x, p, seed=18))  # noise stream keyed by seed
    p2 = sample_camera_params(8, 1.0, seed=4, device="cuda")
    for f in p.__dataclass_fields__:
        assert torch.equal(getattr(p, f), getattr(p2, f)), f
    assert not torch.equal(a, camera_aug(x, sample_camera_params(8, 1.0, seed=5, device="cuda"), seed=17))


@pytest.mark.gpu
@pytest.mark.parametrize("n", [0, 1, 5])
def test_k2_batch_tails(ext, n):
    from twinread.kernels import CameraAugParams, camera_aug, sample_camera_params
    arr = _tiles(8)[:n]
    x = torch.from_numpy(np.ascontiguousarray(arr)).cuda()
    p = sample_camera_params(n, 1.0, seed=31, device="cuda")
    for impl in ("cuda", "torch"):
        out = camera_aug(x, p, seed=2, impl=impl)
        assert out.shape == (n, 224, 224, 3) and out.dtype == torch.uint8 and out.is_cuda, impl
    if n == 0:
        return
    from twinread.kernels import _camera_aug_impl
    out, inter = _camera_aug_impl(x, p, 2, impl="cuda", return_float=True)
    mean, mx = _err(out, _ref_batch(arr, p.to_dicts(), 2))
    assert mean <= 2.0 and mx <= 8, (n, mean, mx)
    dc = _Decomposed()
    dc.check(arr, inter, out, p.to_dicts(), 2, "cuda")
    dc.finish()
    # with the noise off (its stream is keyed by the image index), every element equals the image processed alone
    d = p.to_dicts()
    for e in d:
        e["noise_std"] = 0.0
    q = CameraAugParams.from_dicts(d, "cuda")
    full = camera_aug(x, q, seed=2)
    for i in range(n):
        one = camera_aug(x[i:i + 1], CameraAugParams.from_dicts(d[i:i + 1], "cuda"), seed=2)
        assert torch.equal(one[0], full[i]), i


@pytest.mark.gpu
@pytest.mark.parametrize("level", [0, 255])
def test_k2_uniform_input_finite(ext, level):
    """Blank glass / background: all-0 and all-255 tiles stay finite and in range through every op (Focus 2)."""
    from twinread.kernels import _camera_aug_impl, sample_camera_params
    arr = np.full((6, 224, 224, 3), level, dtype=np.uint8)
    x = torch.from_numpy(arr).cuda()
    dc = _Decomposed()
    for sev in (0.3, 1.0):
        p = sample_camera_params(6, sev, seed=level + 1, device="cuda")
        ref = _ref_batch(arr, p.to_dicts(), 3)
        for impl in ("cuda", "torch"):
            out, inter = _camera_aug_impl(x, p, seed=3, impl=impl, return_float=True)
            assert torch.isfinite(inter).all(), impl
            assert inter.min() >= 0.0 and inter.max() <= 1.0, impl
            assert out.dtype == torch.uint8 and out.shape == x.shape
            mean, mx = _err(out, ref)
            assert mean <= 2.0 and mx <= 8, (impl, level, sev, mean, mx)
            dc.check(arr, inter, out, p.to_dicts(), 3, impl)
    dc.finish()


@pytest.mark.gpu
def test_philox_matches_curand(ext):
    """The numpy Philox4x32-10 used by the reference noise equals curand_Philox4x32_10 bit for bit."""
    from twinread.ref.camera_ops import philox4x32_10
    rng = np.random.default_rng(1)
    ctr = rng.integers(0, 2 ** 32, (4096, 4), dtype=np.uint64).astype(np.uint32)
    for key in ((0, 0), (123456789, 987654321), (2 ** 32 - 1, 5)):
        dev = ext.philox_bits(torch.from_numpy(ctr.astype(np.int64)).cuda(), key[0], key[1]).cpu().numpy()
        assert np.array_equal(dev.astype(np.uint32), philox4x32_10(ctr, key)), key


@pytest.mark.gpu
def test_torch_path_matches_cuda(ext):
    from twinread.kernels import _camera_aug_impl, sample_camera_params
    arr = _tiles(8)
    x = torch.from_numpy(arr).cuda()
    dc = _Decomposed()
    for sev in (0.5, 1.0):
        p = sample_camera_params(8, sev, seed=12, device="cuda")
        a, ia = _camera_aug_impl(x, p, 1, impl="cuda", return_float=True)
        b, ib = _camera_aug_impl(x, p, 1, impl="torch", return_float=True)
        assert b.is_cuda and b.dtype == torch.uint8
        mean, mx = _err(b, a.cpu().numpy())
        assert mean <= 2.0 and mx <= 8, (sev, mean, mx)
        dc.check(arr, ia, a, p.to_dicts(), 1, "cuda")
        dc.check(arr, ib, b, p.to_dicts(), 1, "torch")
    dc.finish()


@pytest.mark.gpu
def test_k2_build_failure_leaves_k1_usable(ext, monkeypatch):
    """A K2 translation unit that does not compile: load_ext falls back to the K1-only module, K1 stays exact on
    the GPU and camera_aug falls back to torch with a warning. Separate build root (build/k2_build_failure_test,
    cached between runs), so the real extension's build directory is untouched."""
    from twinread import kernels
    from twinread.readers import spec_of
    root = kernels.PROJECT / "build" / "k2_build_failure_test"
    bad = root / "k2_broken.cu"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("#error deliberately broken K2 translation unit (test_k2_build_failure_leaves_k1_usable)\n")
    monkeypatch.setattr(kernels, "PROJECT", root)
    monkeypatch.setattr(kernels, "SOURCES", kernels.SOURCES_K1 + [bad])
    monkeypatch.setattr(kernels, "_EXT", {})
    monkeypatch.setattr(kernels, "EXT_ERRORS", {})
    monkeypatch.setattr(kernels, "_WARNED", set())
    tag = kernels._arch_tag(kernels.default_arch())
    m = kernels.load_ext()
    assert m is not None and hasattr(m, "dual_preprocess") and not hasattr(m, "camera_aug")
    assert tag in kernels.EXT_ERRORS and f"{tag}_k1" not in kernels.EXT_ERRORS  # K1+K2 failed, K1 alone built
    assert kernels.load_k2_ext() is None
    x = torch.from_numpy(_tiles(4)).cuda()
    a, b = kernels.dual_preprocess(x, spec_of("conch"), spec_of("musk"), impl="cuda")
    ta, tb = kernels.dual_preprocess(x, spec_of("conch"), spec_of("musk"), impl="torch")
    assert torch.equal(a, ta) and torch.equal(b, tb)
    p = kernels.sample_camera_params(4, 0.8, seed=2, device="cuda")
    with pytest.warns(RuntimeWarning, match="K2 CUDA extension unavailable; falling back to torch"):
        y = kernels.camera_aug(x, p, seed=4, impl="cuda")
    assert torch.equal(y, kernels.camera_aug(x, p, seed=4, impl="torch"))


# ----------------------------------------------------------------------------------------------- CPU-safe
def test_dct_matches_scipy():
    from scipy.fft import dctn, idctn

    from twinread.ref.camera_ops import dct2, idct2
    rng = np.random.default_rng(0)
    blocks = rng.uniform(-128, 128, (50, 8, 8))
    np.testing.assert_allclose(dct2(blocks), dctn(blocks, type=2, axes=(1, 2), norm="ortho"), rtol=0, atol=1e-6)
    np.testing.assert_allclose(idct2(blocks), idctn(blocks, type=2, axes=(1, 2), norm="ortho"), rtol=0, atol=1e-6)
    np.testing.assert_allclose(idct2(dct2(blocks)), blocks, rtol=0, atol=1e-9)


def test_ijg_quant_tables():
    from twinread.ref.camera_ops import CHROMA_Q, LUMA_Q, quant_table
    assert np.array_equal(quant_table(50, chroma=False), LUMA_Q) and np.array_equal(quant_table(50, True), CHROMA_Q)
    assert (quant_table(100, False) == 1).all() and (quant_table(100, True) == 1).all()
    # IJG jpeg_quality_scaling + jpeg_add_quant_table(force_baseline): q=10 -> scale 500, q=75 -> scale 50
    assert quant_table(10, False)[0, 0] == (16 * 500 + 50) // 100 == 80
    assert quant_table(75, False)[0, 0] == (16 * 50 + 50) // 100 == 8
    assert quant_table(10, True).max() == 255 and quant_table(1, False).max() == 255


def test_philox_known_answers():
    """Random123 kat_vectors for philox4x32_10."""
    from twinread.ref.camera_ops import philox4x32_10
    kat = [((0, 0, 0, 0), (0, 0), (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)),
           ((0xffffffff,) * 4, (0xffffffff, 0xffffffff), (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)),
           ((0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344), (0xa4093822, 0x299f31d0),
            (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1))]
    for ctr, key, want in kat:
        got = philox4x32_10(np.array([ctr], dtype=np.uint32), key)[0]
        assert tuple(int(v) for v in got) == want, [hex(int(v)) for v in got]


def test_sample_params_matches_sampler():
    """Ruling R6: sample_params(severity, seed) is image 0 of sample_camera_params(B, severity, seed) for any B."""
    from twinread.kernels import sample_camera_params
    from twinread.ref.camera_ops import sample_params
    for sev, seed in ((0.0, 1), (0.4, 2), (1.0, 3)):
        want = sample_params(sev, seed)
        for B in (1, 5):
            got = sample_camera_params(B, sev, seed, device="cpu").to_dicts()[0]
            assert set(got) | {"seed", "index"} == set(want), set(want) ^ set(got)
            for k, v in got.items():
                assert np.array_equal(np.asarray(v, np.float32), np.asarray(want[k], np.float32)), (sev, B, k)


def test_severity_zero_is_identity_params():
    from twinread.kernels import CameraAugParams, sample_camera_params
    p, ident = sample_camera_params(4, 0.0, seed=9, device="cpu"), CameraAugParams.identity(4, "cpu")
    for f in p.__dataclass_fields__:
        assert torch.equal(getattr(p, f), getattr(ident, f)), f
    q = sample_camera_params(64, 1.0, seed=9, device="cpu")
    assert q.jpeg_quality.min() >= 10 and q.jpeg_quality.max() <= 95
    assert (q.blur_sigma > 0).all() and (q.blur_sigma <= 4.0).all()
    assert all(getattr(q, f).dtype == torch.float32 for f in q.__dataclass_fields__)


def test_ref_ops_identity_and_range():
    from twinread.ref import camera_ops as co
    img = _tiles(2)[1].astype(np.float64) / 255.0
    ident = co.identity_params()
    for name, op in co.OPS:
        assert np.array_equal(op(img, ident), img), name
        p = dict(ident, **{k: co.sample_params(1.0, 4)[k] for k in OP_KEYS[name]})
        y = op(img, p)
        assert y.shape == img.shape and np.isfinite(y).all() and y.min() >= 0 and y.max() <= 1, name
    assert np.array_equal(co.ref_camera_aug(_tiles(2)[1], ident), _tiles(2)[1])


def test_torch_path_cpu_matches_numpy():
    from twinread.kernels import camera_aug, sample_camera_params
    arr = _tiles(3)
    p = sample_camera_params(3, 1.0, seed=8, device="cpu")
    out = camera_aug(torch.from_numpy(arr), p, seed=6, impl="torch")
    assert out.dtype == torch.uint8 and not out.is_cuda
    mean, mx = _err(out, _ref_batch(arr, p.to_dicts(), 6))
    assert mean <= 2.0 and mx <= 8, (mean, mx)
    from twinread.kernels import _camera_aug_impl
    out2, inter = _camera_aug_impl(torch.from_numpy(arr), p, 6, impl="torch", return_float=True)
    assert torch.equal(out2, out)
    dc = _Decomposed()
    dc.check(arr, inter, out2, p.to_dicts(), 6, "torch")
    dc.finish()


def test_k2_fallback_when_ext_missing(monkeypatch):
    import warnings

    from twinread import kernels
    monkeypatch.setattr(kernels, "load_k2_ext", lambda *a, **k: None)
    monkeypatch.setattr(kernels, "_WARNED", set())
    x = torch.from_numpy(_tiles(2))
    p = kernels.sample_camera_params(2, 0.7, seed=1, device="cpu")
    with pytest.warns(RuntimeWarning, match="falling back"):
        a = kernels.camera_aug(x, p, seed=3, impl="cuda")
    assert torch.equal(a, kernels.camera_aug(x, p, seed=3, impl="torch"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kernels.camera_aug(x, p, seed=3, impl="cuda")


def test_camera_aug_rejects_bad_input():
    from twinread.kernels import camera_aug, sample_camera_params
    p = sample_camera_params(1, 0.5, seed=1, device="cpu")
    with pytest.raises(ValueError):
        camera_aug(torch.zeros(1, 8, 8, 3), p, seed=0, impl="torch")
    with pytest.raises(ValueError):
        camera_aug(torch.zeros(1, 8, 8, 4, dtype=torch.uint8), p, seed=0, impl="torch")
    with pytest.raises(ValueError):  # params for 1 image, batch of 2
        camera_aug(torch.zeros(2, 8, 8, 3, dtype=torch.uint8), p, seed=0, impl="torch")
    with pytest.raises(ValueError):
        sample_camera_params(1, -0.1, seed=0, device="cpu")
