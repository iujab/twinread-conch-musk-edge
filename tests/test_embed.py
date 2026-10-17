import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from twinread import embed
from twinread.readers import PreprocessSpec, Reader


def make_fake(size=8, name="fake"):
    spec = PreprocessSpec(size, size, "bicubic", (0.5,) * 3, (0.5,) * 3)
    calls = SimpleNamespace(n=0)

    def pre(im):
        import torchvision.transforms.functional as F
        return F.to_tensor(im.resize((spec.size, spec.size)))

    def enc(b):
        calls.n += 1
        return b.float().mean(dim=(2, 3)).numpy()  # [B,3]

    r = Reader(name, spec, 3, 3, pre, enc, lambda b: enc(b) * 2, lambda t: np.zeros((len(t), 3), np.float32))
    return r, calls


def tiles(n=37, hw=10, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (n, hw, hw, 3), dtype=np.uint8)


def test_embed_array_shapes_and_dtype(tmp_path):
    r, _ = make_fake()
    x = tiles()
    out = embed.embed_array(r, x, "aligned", tmp_path / "a.npy", batch=4)
    assert out.shape == (37, 3) and out.dtype == np.float32
    assert np.load(tmp_path / "a.npy").dtype == np.float16
    assert (tmp_path / "a.json").exists()
    assert embed.embed_array(r, x[:0], "probe", tmp_path / "e.npy").shape[0] == 0


def test_cache_hit_skips_compute(tmp_path):
    r, c = make_fake()
    x = tiles()
    a = embed.embed_array(r, x, "aligned", tmp_path / "a.npy", batch=8)
    n = c.n
    b = embed.embed_array(r, x, "aligned", tmp_path / "a.npy", batch=8)
    assert c.n == n and np.array_equal(a, b)


def test_cache_invalidated_on_spec_change(tmp_path):
    r, c = make_fake(size=8)
    x = tiles()
    embed.embed_array(r, x, "aligned", tmp_path / "a.npy", batch=8)
    n = c.n
    r2 = dataclasses.replace(r, spec=dataclasses.replace(r.spec, size=12))
    embed.embed_array(r2, x, "aligned", tmp_path / "a.npy", batch=8)
    assert c.n > n


def test_cache_invalidated_on_kind_impl_data(tmp_path):
    r, c = make_fake()
    x = tiles()
    k = embed.cache_key
    base = k("fake", "aligned", r.spec, "reference", x)
    assert base != k("fake", "probe", r.spec, "reference", x)
    assert base != k("fake", "aligned", r.spec, "k1", x)
    assert base != k("other", "aligned", r.spec, "reference", x)
    x2 = x.copy(); x2[0, 0, 0, 0] ^= 1
    assert base != k("fake", "aligned", r.spec, "reference", x2)
    assert base != k("fake", "aligned", r.spec, "reference", x[:-1])
    embed.embed_array(r, x, "aligned", tmp_path / "a.npy", batch=8)
    n = c.n
    embed.embed_array(r, x2, "aligned", tmp_path / "a.npy", batch=8)
    assert c.n > n


def test_resume_from_parts(tmp_path):
    r, c = make_fake()
    x = tiles(50)
    ref = embed.embed_array(r, x, "aligned", tmp_path / "ref.npy", batch=7, chunk=16)
    out = tmp_path / "o.npy"
    with pytest.raises(embed.NeedsRerun):
        embed.embed_array(r, x, "aligned", out, batch=7, chunk=16, budget_s=0)
    assert not out.exists() and list(out.parent.glob("o.npy.parts/*.part*"))
    n = c.n
    res = embed.embed_array(r, x, "aligned", out, batch=7, chunk=16)
    assert np.array_equal(res, ref)
    assert c.n - n < ref.shape[0] / 7 * 1.0 + 1  # did not redo the first chunk(s) fully
    assert not (tmp_path / "o.npy.parts").exists()


def test_k1_unavailable_raises(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "twinread.kernels", None)
    r, _ = make_fake()
    with pytest.raises(RuntimeError, match="k1"):
        embed.embed_array(r, tiles(4), "aligned", tmp_path / "k.npy", impl="k1")


def test_resolve_readers():
    assert embed.resolve_readers("CM", None) == ["conch", "musk"]
    assert embed.resolve_readers("OPEN", None) == ["pathgenclip"]
    assert embed.resolve_readers(None, ["pathgenclip", "quiltnet_b16"]) == ["pathgenclip", "quiltnet_b16"]
    with pytest.raises(KeyError):
        embed.resolve_readers(None, ["nope"])


def test_cache_invalidated_on_middle_tile_change(tmp_path):
    r, c = make_fake()
    x = tiles(1200, hw=4)
    embed.embed_array(r, x, "aligned", tmp_path / "a.npy", batch=64)
    n = c.n
    embed.embed_array(r, x, "aligned", tmp_path / "a.npy", batch=64)
    assert c.n == n
    i = int(embed._sample_idx(len(x))[len(embed._sample_idx(len(x))) // 2])
    assert 64 <= i < len(x) - 64
    x2 = x.copy(); x2[i, 0, 0, 0] ^= 1
    embed.embed_array(r, x2, "aligned", tmp_path / "a.npy", batch=64)
    assert c.n > n


def test_cache_invalidated_on_middle_permutation(tmp_path):
    r, c = make_fake()
    x = tiles(1200, hw=4)
    embed.embed_array(r, x, "aligned", tmp_path / "a.npy", batch=64)
    n = c.n
    x2 = x.copy(); x2[64:-64] = x[64:-64][::-1]
    embed.embed_array(r, x2, "aligned", tmp_path / "a.npy", batch=64)
    assert c.n > n


# ------------------------------------------------------------------------------------------- Task 8b: pair-level K1
def make_official(size, crop, name):
    """Fake reader whose preprocess is the official-style transform (Resize shorter side, CenterCrop, Normalize), so
    the K1 path (bit-exact to official(img).half()) and the reference path agree up to the fp16 cast."""
    from twinread.readers import _compose
    spec = PreprocessSpec(size, crop, "bicubic", (0.5, 0.4, 0.3), (0.2, 0.25, 0.3))
    calls = SimpleNamespace(n=0, dtypes=set())

    def enc(b):
        calls.n += 1
        calls.dtypes.add(b.dtype)
        b = b.float()
        return torch.cat([b.mean(dim=(2, 3)), b.std(dim=(2, 3))], 1).numpy()  # [B,6]

    r = Reader(name, spec, 6, 6, _compose(spec), enc, lambda b: enc(b) * 2, lambda t: np.zeros((len(t), 6), np.float32))
    return r, calls


def _count_dual(monkeypatch):
    from twinread import kernels
    real, seen = kernels.dual_preprocess, []

    def dual(x, a, b, impl="cuda"):
        seen.append((int(x.shape[0]), a, b, impl))
        return real(x, a, b, impl=impl)
    monkeypatch.setattr(kernels, "dual_preprocess", dual)
    return seen


def test_k1_pair_one_dual_call_per_batch(tmp_path, monkeypatch):
    from twinread import kernels
    monkeypatch.setattr(kernels, "_WARNED", set())
    seen = _count_dual(monkeypatch)
    rA, cA = make_official(8, 8, "ra")
    rB, cB = make_official(7, 6, "rb")
    x = tiles(10, hw=12)
    pa, pb = tmp_path / "ra_aligned.npy", tmp_path / "rb_aligned.npy"
    with pytest.warns(RuntimeWarning):  # CPU input: K1 runs its torch path (warned once)
        eA, eB = embed.embed_arrays([rA, rB], x, "aligned", [pa, pb], batch=4, impl="k1", device="cpu")
    # one dual_preprocess launch per encode batch, carrying BOTH readers' specs
    assert [s[0] for s in seen] == [4, 4, 2]
    assert all(s[1] == rA.spec and s[2] == rB.spec for s in seen)
    assert cA.n == cB.n == 3 and cA.dtypes == cB.dtypes == {torch.float32}
    assert eA.shape == eB.shape == (10, 6)
    # equals the reference path up to the fp16 input cast
    rA2, _ = make_official(8, 8, "ra")
    rB2, _ = make_official(7, 6, "rb")
    ref = embed.embed_arrays([rA2, rB2], x, "aligned", [tmp_path / "a_ref.npy", tmp_path / "b_ref.npy"], batch=4)
    assert np.allclose(eA, ref[0], atol=2e-2) and np.allclose(eB, ref[1], atol=2e-2)
    # sidecars record impl=k1 (and the K1 backend that ran); keys differ from the reference caches
    sa, sr = json.loads(pa.with_suffix(".json").read_text()), json.loads((tmp_path / "a_ref.json").read_text())
    assert sa["impl"] == "k1" and sa["k1_backend"] == "torch" and sa["k1_group"] == ["ra", "rb"]
    assert sr["impl"] == "reference" and sa["key"] != sr["key"]


def test_k1_pair_cache_hit_and_reference_not_reused(tmp_path, monkeypatch):
    rA, cA = make_official(8, 8, "ra")
    rB, cB = make_official(7, 6, "rb")
    x = tiles(6, hw=12)
    paths = [tmp_path / "ra.npy", tmp_path / "rb.npy"]
    embed.embed_arrays([rA, rB], x, "aligned", paths, batch=8)  # reference cache at the same paths
    n = cA.n
    seen = _count_dual(monkeypatch)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        embed.embed_arrays([rA, rB], x, "aligned", paths, batch=8, impl="k1", device="cpu")
    assert cA.n > n and len(seen) == 1  # reference cache was not reused for k1
    n, m = cA.n, len(seen)
    embed.embed_arrays([rA, rB], x, "aligned", paths, batch=8, impl="k1", device="cpu")
    assert cA.n == n and len(seen) == m  # k1 cache hit


def test_k1_pair_resume_only_missing_reader(tmp_path, monkeypatch):
    """A chunk part present for reader A but not B is not recomputed for A."""
    rA, cA = make_official(8, 8, "ra")
    rB, cB = make_official(7, 6, "rb")
    x = tiles(20, hw=12)
    paths = [tmp_path / "ra.npy", tmp_path / "rb.npy"]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(embed.NeedsRerun):
            embed.embed_arrays([rA, rB], x, "aligned", paths, batch=4, impl="k1", device="cpu", chunk=8, budget_s=0)
        pb = tmp_path / "rb.npy.parts" / "00000.part.npy"
        pb.unlink()  # B lost its first chunk
        a0, b0 = cA.n, cB.n
        eA, eB = embed.embed_arrays([rA, rB], x, "aligned", paths, batch=4, impl="k1", device="cpu", chunk=8)
    assert cA.n - a0 == 3 and cB.n - b0 == 5  # A: chunks 1-2 (2+1 batches); B: chunks 0-2 (2+2+1)
    assert eA.shape == eB.shape == (20, 6)


def test_k1_single_reader_path(tmp_path):
    r, c = make_official(8, 8, "solo")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        e = embed.embed_array(r, tiles(5, hw=12), "probe", tmp_path / "s.npy", batch=2, impl="k1", device="cpu")
    assert e.shape == (5, 6) and c.n == 3
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["impl"] == "k1" and s["k1_group"] == ["solo"]


def test_k1_cpu_falls_back_with_warning(tmp_path, monkeypatch):
    """Extension unavailable: K1 warns that it falls back (and the sidecar records the backend that ran)."""
    from twinread import kernels
    monkeypatch.setattr(kernels, "load_ext", lambda *a, **k: None)
    monkeypatch.setattr(kernels, "_WARNED", set())
    rA, _ = make_official(8, 8, "ra")
    rB, _ = make_official(7, 6, "rb")
    with pytest.warns(RuntimeWarning, match="falling back to torch"):
        embed.embed_arrays([rA, rB], tiles(3, hw=12), "aligned", [tmp_path / "a.npy", tmp_path / "b.npy"],
                           impl="k1", device="cpu")
    assert json.loads((tmp_path / "a.json").read_text())["k1_backend"] == "torch"


def _fake_dataset(monkeypatch, n=6, hw=12):
    from twinread import data as tdata
    xs = {s: (tiles(n, hw=hw, seed=i), np.arange(n) % 3) for i, s in enumerate(("train", "val", "test"))}
    monkeypatch.setattr(tdata, "load_split", lambda d, ds, s: xs[s])
    return xs


def test_embed_dataset_k1_loads_pair_together(tmp_path, monkeypatch):
    _fake_dataset(monkeypatch)
    log = []
    fakes = {"conch": make_official(8, 8, "conch")[0], "musk": make_official(7, 6, "musk")[0]}

    def load(name, device="cuda"):
        log.append(("load", name, device))
        r = fakes[name]
        return dataclasses.replace(r, _close=lambda n=name: log.append(("close", n)))
    monkeypatch.setattr(embed, "load_reader", load)
    seen = _count_dual(monkeypatch)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s = embed.embed_dataset("CM", "crc", kinds=("aligned",), impl="k1", splits=("val", "test"),
                                data_dir=tmp_path, device="cpu", batch=4)
    assert s["done"] and s["groups"] == [["conch", "musk"]]
    assert log[:2] == [("load", "conch", "cpu"), ("load", "musk", "cpu")]  # both resident, then one pass
    assert all(a == fakes["conch"].spec and b == fakes["musk"].spec for _, a, b, _ in seen)
    assert len(seen) == 4  # 2 splits x ceil(6/4) batches, both readers per call
    for f in ("conch_aligned_val", "musk_aligned_val", "conch_aligned_test", "musk_aligned_test"):
        assert json.loads((tmp_path / "emb" / "crc" / f"{f}.json").read_text())["impl"] == "k1"


def test_embed_dataset_reference_one_reader_at_a_time(tmp_path, monkeypatch):
    _fake_dataset(monkeypatch)
    log = []

    def load(name, device="cuda"):
        log.append(("load", name))
        r = make_official(8, 8, name)[0]
        return dataclasses.replace(r, _close=lambda n=name: log.append(("close", n)))
    monkeypatch.setattr(embed, "load_reader", load)
    s = embed.embed_dataset("CM", "crc", kinds=("aligned",), splits=("val",), data_dir=tmp_path, device="cpu")
    assert s["done"] and s["groups"] == [["conch"], ["musk"]]
    assert log == [("load", "conch"), ("close", "conch"), ("load", "musk"), ("close", "musk")]


def test_embed_dataset_k1_pair_oom_on_load_falls_back_to_solo(tmp_path, monkeypatch):
    _fake_dataset(monkeypatch)
    log, state = [], {"musk_fail": 1}

    def load(name, device="cuda"):
        if name == "musk" and state["musk_fail"]:
            state["musk_fail"] -= 1
            raise torch.cuda.OutOfMemoryError("fake OOM")
        log.append(("load", name))
        r = make_official(8, 8, name)[0]
        return dataclasses.replace(r, _close=lambda n=name: log.append(("close", n)))
    monkeypatch.setattr(embed, "load_reader", load)
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = embed.embed_dataset("CM", "crc", kinds=("aligned",), impl="k1", splits=("val",), data_dir=tmp_path,
                                device="cpu")
    assert s["done"] and s["groups"] == [["conch"], ["musk"]]
    assert any("one at a time" in str(x.message) for x in w)
    assert log[:2] == [("load", "conch"), ("close", "conch")]


# ------------------------------------------------------------------------------------------- GPU: real CONCH + MUSK
@pytest.mark.gpu
def test_k1_pair_matches_reference_conch_musk(tmp_path):
    """CONCH+MUSK on 16 real CRC tiles through embed_dataset: k1 (one dual launch per batch) vs reference, per reader
    and kind, cosine >= 0.999 (K1 is bit-exact to official(img).half(), so ~1.0 is expected)."""
    from pathlib import Path
    from twinread import kernels
    if kernels.load_ext() is None:
        pytest.skip("twinread CUDA extension unavailable")
    x = np.load(Path(__file__).parent / "fixtures" / "crc16.npy")
    np.save(tmp_path / "crc_test_x.npy", x)
    np.save(tmp_path / "crc_test_y.npy", np.arange(16) % 9)
    out = {}
    for impl in ("k1", "reference"):
        s = embed.embed_dataset("CM", "crc", kinds=("aligned", "probe"), impl=impl, splits=("test",),
                                data_dir=tmp_path, batch=8)
        assert s["done"]
        assert s["groups"] == ([["conch", "musk"]] if impl == "k1" else [["conch"], ["musk"]])
        for r in ("conch", "musk"):
            for k in ("aligned", "probe"):
                p = tmp_path / "emb" / "crc" / f"{r}_{k}_test.npy"
                side = json.loads(p.with_suffix(".json").read_text())
                assert side["impl"] == impl
                if impl == "k1":
                    assert side["k1_backend"] == "cuda" and side["k1_group"] == ["conch", "musk"]
                out[impl, r, k] = np.load(p).astype(np.float32)
    for r in ("conch", "musk"):
        for k in ("aligned", "probe"):
            a, b = out["k1", r, k], out["reference", r, k]
            cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
            assert a.shape == b.shape and a.shape[0] == 16
            assert cos.min() >= 0.999, f"{r} {k}: min cosine {cos.min():.6f}"
