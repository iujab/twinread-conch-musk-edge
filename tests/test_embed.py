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
