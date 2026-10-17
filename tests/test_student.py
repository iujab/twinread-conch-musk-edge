"""Task 10: TwinStudent, distillation loss, torch camera augmentation, device flag thresholds, trainer (CPU, small)."""
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from twinread import probes as probes_mod
from twinread import review, student

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------------- fixtures
def make_fixture(tmp_path, dimA=16, dimB=24, n_classes=2, seed=0, prereg=True):
    """16 cam-layout tiles (96 px, hospitals -> train 8 / val 4 / test 4) + student_targets.npz + probes_aligned.npz."""
    rng = np.random.default_rng(seed)
    data = tmp_path / "data"
    tgt_dir = tmp_path / "targets"
    data.mkdir()
    tgt_dir.mkdir()
    h = np.array([0, 0, 0, 3, 3, 3, 4, 4, 1, 1, 1, 1, 2, 2, 2, 2])
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0])
    x = rng.integers(0, 256, (16, 96, 96, 3), dtype=np.uint8)
    np.save(data / "cam_x.npy", x)
    np.save(data / "cam_y.npy", y)
    np.save(data / "cam_h.npy", h)
    split_of = {"train": (0, 3, 4), "val": (1,), "test": (2,)}
    tgt = {}
    for s, hs in split_of.items():
        idx = np.flatnonzero(np.isin(h, hs))
        n = len(idx)
        tA = rng.normal(size=(n, dimA)).astype(np.float32)
        tB = rng.normal(size=(n, dimB)).astype(np.float32)
        tA[:, 0] += 3 * y[idx]  # make the classes separable for the probes
        tB[:, 0] += 3 * y[idx]
        tgt[f"tA_{s}"] = tA / np.linalg.norm(tA, axis=1, keepdims=True)
        tgt[f"tB_{s}"] = tB / np.linalg.norm(tB, axis=1, keepdims=True)
        p = np.clip(np.eye(n_classes)[y[idx]] * 0.8 + 0.1, 1e-8, 1)
        tgt[f"fused_logits_{s}"] = np.log(p / p.sum(1, keepdims=True)).astype(np.float32)
        tgt[f"review_{s}"] = (np.arange(n) % 2).astype(np.int8)
        tgt[f"y_{s}"] = y[idx].astype(np.int64)
    np.savez(tgt_dir / "student_targets.npz", **tgt)
    arrs = {}
    for r in "AB":
        pr = probes_mod.fit_probe(tgt[f"t{r}_train"], tgt["y_train"], tgt[f"t{r}_val"], tgt["y_val"])
        arrs.update(pr.to_arrays(r))
    np.savez(tgt_dir / "probes_aligned.npz", **arrs)
    if prereg:
        (tgt_dir / "preregistration.json").write_text(json.dumps({"pair": "OPEN", "dataset": "cam"}))
    return data, tgt_dir


def _perfect_out(tA, tB, fused, review_logit):
    return {"hA": tA.clone(), "hB": tB.clone(), "logits": fused.clone(), "review_logit": review_logit}


# ------------------------------------------------------------------------------------------------- model
def test_backbone_list_matches_plan():
    assert student.BACKBONES == ("mobilenetv4_conv_small", "efficientnet_b0", "mobileone_s0", "regnetx_002",
                                 "resnet18", "lcnet_050")


def test_forward_shapes_and_norms():
    torch.manual_seed(0)
    m = student.TwinStudent("lcnet_050", dimA=32, dimB=48, n_classes=5, pretrained=False).eval()
    x = torch.rand(3, 3, 224, 224)
    with torch.no_grad():
        out = m(x)
    assert set(out) == {"hA", "hB", "logits", "review_logit"}
    assert out["hA"].shape == (3, 32) and out["hB"].shape == (3, 48)
    assert out["logits"].shape == (3, 5) and out["review_logit"].shape == (3,)
    assert torch.allclose(out["hA"].norm(dim=1), torch.ones(3), atol=1e-5)
    assert torch.allclose(out["hB"].norm(dim=1), torch.ones(3), atol=1e-5)
    for v in out.values():
        assert torch.isfinite(v).all()


def test_normalization_is_inside_the_module():
    """ONNX takes [0,1] input: the module must apply ImageNet mean/std itself."""
    m = student.TwinStudent("lcnet_050", dimA=8, dimB=8, n_classes=3, pretrained=False)
    assert torch.allclose(m.mean.flatten(), torch.tensor(student.IMAGENET_MEAN))
    assert torch.allclose(m.std.flatten(), torch.tensor(student.IMAGENET_STD))
    seen = {}
    m.trunk.register_forward_pre_hook(lambda mod, args: seen.setdefault("x", args[0].detach().clone()))
    x = torch.full((1, 3, 224, 224), 0.5)
    with torch.no_grad():
        m.eval()(x)
    expect = (0.5 - torch.tensor(student.IMAGENET_MEAN)) / torch.tensor(student.IMAGENET_STD)
    assert torch.allclose(seen["x"][0, :, 0, 0], expect, atol=1e-6)


def test_batch_of_one_forward():
    m = student.TwinStudent("lcnet_050", dimA=8, dimB=8, n_classes=3, pretrained=False).eval()
    with torch.no_grad():
        out = m(torch.rand(1, 3, 224, 224))
    assert out["review_logit"].shape == (1,)


# ------------------------------------------------------------------------------------------------- losses
def test_distill_loss_zero_when_perfect():
    g = torch.Generator().manual_seed(0)
    tA = F.normalize(torch.randn(8, 16, generator=g), dim=1)
    tB = F.normalize(torch.randn(8, 24, generator=g), dim=1)
    fused = torch.log_softmax(torch.randn(8, 4, generator=g), dim=1)
    y = torch.randint(0, 4, (8,), generator=g)
    rev_y = torch.randint(0, 2, (8,), generator=g)
    out = _perfect_out(tA, tB, fused, torch.randn(8, generator=g))
    total, parts = student.distill_loss(out, tA, tB, fused, y, rev_y)
    assert abs(parts["cos"]) < 1e-6 and abs(parts["cos_A"]) < 1e-6 and abs(parts["cos_B"]) < 1e-6
    assert abs(parts["kd"]) < 1e-6
    assert set(parts) >= {"cos", "rel", "kd", "ce", "rev", "total"}
    assert torch.isfinite(total)
    assert abs(parts["total"] - float(total)) < 1e-5


def test_distill_loss_default_weights_and_formula():
    g = torch.Generator().manual_seed(1)
    tA = F.normalize(torch.randn(6, 8, generator=g), dim=1)
    tB = F.normalize(torch.randn(6, 8, generator=g), dim=1)
    hA = F.normalize(torch.randn(6, 8, generator=g), dim=1)
    hB = F.normalize(torch.randn(6, 8, generator=g), dim=1)
    logits, fused = torch.randn(6, 3, generator=g), torch.log_softmax(torch.randn(6, 3, generator=g), 1)
    y, rev_y, rl = torch.randint(0, 3, (6,), generator=g), torch.randint(0, 2, (6,), generator=g), torch.randn(6, generator=g)
    out = {"hA": hA, "hB": hB, "logits": logits, "review_logit": rl}
    T = 2.0
    total, parts = student.distill_loss(out, tA, tB, fused, y, rev_y, T=T)
    assert student.DEFAULT_WEIGHTS == {"cos": 1.0, "rel": 0.5, "kd": 1.0, "ce": 0.5, "rev": 0.5}
    p_t = torch.softmax(fused / T, 1)
    kd = T * T * (p_t * (p_t.log() - torch.log_softmax(logits / T, 1))).sum(1).mean()
    assert abs(parts["kd"] - float(kd)) < 1e-5
    assert abs(parts["ce"] - float(F.cross_entropy(logits, y))) < 1e-5
    assert abs(parts["rev"] - float(F.binary_cross_entropy_with_logits(rl, rev_y.float()))) < 1e-5
    cosA = float((1 - (hA * tA).sum(1)).mean())
    cosB = float((1 - (hB * tB).sum(1)).mean())
    assert abs(parts["cos"] - (cosA + cosB) / 2) < 1e-5
    w = student.DEFAULT_WEIGHTS
    expect = sum(w[k] * parts[k] for k in ("cos", "rel", "kd", "ce", "rev"))
    assert abs(float(total) - expect) < 1e-4
    total0, _ = student.distill_loss(out, tA, tB, fused, y, rev_y, T=T, w={"rel": 0.0, "kd": 0.0, "ce": 0.0, "rev": 0.0})
    assert abs(float(total0) - parts["cos"]) < 1e-5


def test_info_nce_matches_explicit_formula():
    g = torch.Generator().manual_seed(2)
    h = F.normalize(torch.randn(10, 12, generator=g), dim=1)
    t = F.normalize(torch.randn(10, 12, generator=g), dim=1)
    tau = 0.07
    s = (h @ t.T) / tau
    n = len(h)
    expect = (sum(torch.logsumexp(s[i], 0) - s[i, i] for i in range(n)) +
              sum(torch.logsumexp(s[:, j], 0) - s[j, j] for j in range(n))) / (2 * n)
    got = student.info_nce(h, t, tau, impl="torch")
    assert abs(float(got) - float(expect)) < 1e-5
    assert abs(float(student.info_nce(t, h, tau, impl="torch")) - float(got)) < 1e-5  # symmetric
    assert float(student.info_nce(h[:1], t[:1], tau, impl="torch")) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.gpu
def test_info_nce_bigbatch_agrees_with_torch():
    if student.bigbatch_clip_loss() is None:
        pytest.skip("BigBatch not importable")
    g = torch.Generator().manual_seed(3)
    for n, d in ((128, 768), (37, 512), (128, 1024)):
        h = F.normalize(torch.randn(n, d, generator=g), dim=1).cuda().requires_grad_(True)
        t = F.normalize(torch.randn(n, d, generator=g), dim=1).cuda()
        a = student.info_nce(h, t, 0.07, impl="torch")
        (ga,) = torch.autograd.grad(a, h)
        b = student.info_nce(h, t, 0.07, impl="bigbatch")
        (gb,) = torch.autograd.grad(b, h)
        assert abs(float(a) - float(b)) < 1e-4 * max(1.0, abs(float(a)))
        assert torch.allclose(ga, gb, atol=1e-5, rtol=1e-3)


def test_distill_loss_gradients_flow():
    torch.manual_seed(0)
    m = student.TwinStudent("lcnet_050", dimA=16, dimB=24, n_classes=3, pretrained=False).train()
    x = torch.rand(4, 3, 224, 224)
    tA = F.normalize(torch.randn(4, 16), dim=1)
    tB = F.normalize(torch.randn(4, 24), dim=1)
    fused = torch.log_softmax(torch.randn(4, 3), 1)
    y, rev_y = torch.tensor([0, 1, 2, 0]), torch.tensor([0, 1, 1, 0])
    total, _ = student.distill_loss(m(x), tA, tB, fused, y, rev_y)
    total.backward()
    named = dict(m.named_parameters())
    for prefix in ("trunk.", "head_A.", "head_B.", "cls.", "review."):
        grads = [p.grad for k, p in named.items() if k.startswith(prefix)]
        assert grads and all(gr is not None for gr in grads), prefix
        assert any(float(gr.abs().sum()) > 0 for gr in grads), prefix
        assert all(torch.isfinite(gr).all() for gr in grads), prefix


# ------------------------------------------------------------------------------------------------- augmentation
def _tiles(b, h=64, w=64, seed=0):
    return torch.from_numpy(np.random.default_rng(seed).integers(0, 256, (b, h, w, 3), dtype=np.uint8))


def test_torch_aug_identity_at_zero_and_deterministic():
    x = _tiles(5)
    assert torch.equal(student.torch_camera_aug(x, 0.0, seed=1), x)
    a = student.torch_camera_aug(x, 0.8, seed=7)
    b = student.torch_camera_aug(x, 0.8, seed=7)
    c = student.torch_camera_aug(x, 0.8, seed=8)
    assert a.dtype == torch.uint8 and a.shape == x.shape
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert not torch.equal(a, x)
    d1 = (student.torch_camera_aug(x, 0.3, seed=7).float() - x.float()).abs().mean()
    d2 = (student.torch_camera_aug(x, 1.0, seed=7).float() - x.float()).abs().mean()
    assert 0 < d1 < d2  # stronger severity -> larger change


def test_torch_aug_batch_tails_and_nonsquare():
    for b in (0, 1, 5):
        out = student.torch_camera_aug(_tiles(b, 40, 52), 1.0, seed=3)
        assert out.shape == (b, 40, 52, 3) and out.dtype == torch.uint8


def test_torch_aug_uniform_input_stays_sane():
    """Review Focus 2: blank glass / near-uniform frames stay finite and in range (no NaN cast to 0)."""
    white = torch.full((4, 64, 64, 3), 255, dtype=torch.uint8)
    black = torch.zeros((4, 64, 64, 3), dtype=torch.uint8)
    f_white = student.torch_camera_aug(white, 1.0, seed=0, as_float=True)
    f_black = student.torch_camera_aug(black, 1.0, seed=0, as_float=True)
    for f in (f_white, f_black):
        assert torch.isfinite(f).all() and f.min() >= 0 and f.max() <= 1
    assert float(f_white.mean()) > 0.6 and float(f_black.mean()) < 0.4


def test_k2_falls_back_to_torch_with_one_warning(monkeypatch):
    monkeypatch.setitem(sys.modules, "twinread.kernels", None)  # import fails -> fallback
    monkeypatch.setattr(student, "_K2_STATE", {})
    x = _tiles(3)
    with pytest.warns(UserWarning, match="K2"):
        a = student.augment(x, "k2", 0.6, seed=11)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # second call: no new warning
        b = student.augment(x, "k2", 0.6, seed=11)
    ref = student.torch_camera_aug(x, 0.6, seed=11)
    assert torch.equal(a, ref) and torch.equal(b, ref)
    assert student.aug_impl_used("k2") == "torch"
    assert student.aug_impl_used("none") == "none"
    assert torch.equal(student.augment(x, "none", 1.0, seed=11), x)


def test_to_input_resizes_and_scales():
    x = torch.full((2, 96, 96, 3), 255, dtype=torch.uint8)
    z = student.to_input(x)
    assert z.shape == (2, 3, 224, 224) and z.dtype == torch.float32
    assert torch.allclose(z, torch.ones_like(z), atol=1e-5)
    z2 = student.to_input(_tiles(1, 224, 300))  # non-square: center crop then resize
    assert z2.shape == (1, 3, 224, 224)


# ------------------------------------------------------------------------------------------------- device rule
def _rand_probs(n, k, rng, conc=1.0):
    return rng.dirichlet(np.full(k, conc), size=n)


def test_probe_probs_match_sklearn_probe():
    rng = np.random.default_rng(0)
    for k in (2, 4):
        z = rng.normal(size=(60, 6))
        y = rng.integers(0, k, 60)
        y[:k] = np.arange(k)
        z[:, 0] += y
        pr = probes_mod.fit_probe(z, y, z, y)
        arrs = pr.to_arrays("A")
        got = student.probe_probs(z, arrs, "A")
        assert np.allclose(got, pr.predict_proba(z), atol=1e-4)


def test_device_flags_match_review_signals():
    rng = np.random.default_rng(1)
    pA, pB = _rand_probs(200, 5, rng), _rand_probs(200, 5, rng)
    rl = rng.normal(size=200)
    fl = student.device_flags(pA, pB, rl, {"js": 0.3, "review": 0.8})
    sig = review.signals(pA, pB, pA)
    assert np.array_equal(fl["disagree"], sig["s1_disagree"].astype(bool))
    assert np.allclose(fl["js"], sig["s2_js"])
    assert np.array_equal(fl["flag"], fl["disagree"] | (fl["js"] > 0.3) | (1 / (1 + np.exp(-rl)) > 0.8))


def test_choose_thresholds_flags_about_ten_percent():
    rng = np.random.default_rng(2)
    n = 1000
    base = _rand_probs(n, 4, rng, conc=0.3)
    pA = base
    pB = np.where(rng.uniform(size=(n, 1)) < 0.04, _rand_probs(n, 4, rng, 0.3), 0.9 * base + 0.1 * _rand_probs(n, 4, rng))
    rl = rng.normal(size=n)
    teacher = rng.uniform(size=n) < 0.1
    thr = student.choose_thresholds(pA, pB, rl, teacher, budget=0.10)
    assert set(thr) >= {"js", "review"} and 0 <= thr["js"] <= 1 and 0 <= thr["review"] <= 1
    fl = student.device_flags(pA, pB, rl, thr)
    assert abs(fl["flag"].mean() - 0.10) <= 0.006
    assert fl["disagree"].mean() < 0.10


def test_choose_thresholds_when_disagreement_exceeds_budget():
    rng = np.random.default_rng(3)
    pA, pB = _rand_probs(300, 3, rng), _rand_probs(300, 3, rng)  # independent -> ~2/3 disagree
    thr = student.choose_thresholds(pA, pB, rng.normal(size=300), None, budget=0.10)
    assert thr["js"] == 1.0 and thr["review"] == 1.0
    fl = student.device_flags(pA, pB, rng.normal(size=300), thr)
    assert np.array_equal(fl["flag"], fl["disagree"])


# ------------------------------------------------------------------------------------------------- trainer
def test_train_student_one_step_cpu(tmp_path):
    data, tgt = make_fixture(tmp_path)
    out = tmp_path / "out"
    m = student.train_student("OPEN", "cam", "lcnet_050", epochs=1, batch=4, aug="none", out_dir=out,
                              targets_dir=tgt, data_dir=data, pretrained=False, device="cpu")
    assert m["done"] is True
    met = json.loads((out / "metrics.json").read_text())
    for s in ("val", "test"):
        for sev in ("clean", "sev0.5", "sev1.0"):
            blk = met[s][sev]
            for k in ("bal_acc", "cos_A", "cos_B", "review_auroc", "device_flag"):
                assert k in blk, (s, sev, k)
    assert "argmax" in met["device_flag_definition"]
    assert met["throughput"]["train_images_per_s"] > 0
    assert met["aug_requested"] == "none" and met["aug_used"] == "none"
    thr = json.loads((out / "thresholds.json").read_text())
    assert set(thr) == {"js", "review"} and all(isinstance(v, float) for v in thr.values())
    assert (out / "best.pt").exists()
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["backbone"] == "lcnet_050" and ck["config"]["dimA"] == 16 and ck["config"]["dimB"] == 24
    mm = student.load_student(out / "best.pt", map_location="cpu")
    assert isinstance(mm, student.TwinStudent)
    assert (out / ".gitignore").read_text().strip() == "*.pt"


def test_train_student_skips_test_without_preregistration(tmp_path):
    data, tgt = make_fixture(tmp_path, prereg=False)
    out = tmp_path / "out"
    student.train_student("OPEN", "cam", "lcnet_050", epochs=1, batch=8, aug="torch", out_dir=out,
                          targets_dir=tgt, data_dir=data, pretrained=False, device="cpu")
    met = json.loads((out / "metrics.json").read_text())
    assert "test" not in met and "preregistration" in met["test_skipped"]
    assert met["aug_used"] == "torch"


def test_train_student_rejects_misaligned_targets(tmp_path):
    data, tgt = make_fixture(tmp_path)
    y = np.load(data / "cam_y.npy")
    np.save(data / "cam_y.npy", 1 - y)  # labels no longer match y_train in the targets file
    with pytest.raises(ValueError, match="align"):
        student.train_student("OPEN", "cam", "lcnet_050", epochs=1, batch=4, aug="none", out_dir=tmp_path / "o",
                              targets_dir=tgt, data_dir=data, pretrained=False, device="cpu")


def test_train_student_resumes_after_budget(tmp_path):
    data, tgt = make_fixture(tmp_path)
    out = tmp_path / "out"
    kw = dict(targets_dir=tgt, data_dir=data, pretrained=False, device="cpu")
    r1 = student.train_student("OPEN", "cam", "lcnet_050", epochs=2, batch=4, aug="none", out_dir=out, budget_s=0.0, **kw)
    assert r1["done"] is False and r1["epochs_completed"] == 1
    assert not (out / "metrics.json").exists() and (out / "last.pt").exists()
    r2 = student.train_student("OPEN", "cam", "lcnet_050", epochs=2, batch=4, aug="none", out_dir=out, **kw)
    assert r2["done"] is True and r2["epochs_completed"] == 2 and r2["resumed_from_epoch"] == 1
    met = json.loads((out / "metrics.json").read_text())
    assert [h["epoch"] for h in met["history"]] == [1, 2]
    # a different config never reuses the stale checkpoint
    r3 = student.train_student("OPEN", "cam", "lcnet_050", epochs=1, batch=4, aug="none", out_dir=out, seed=5, **kw)
    assert r3["resumed_from_epoch"] == 0


def _load_cli():
    spec = importlib.util.spec_from_file_location("train_student_cli", ROOT / "scripts" / "train_student.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_exit_codes_and_args(tmp_path):
    data, tgt = make_fixture(tmp_path)
    cli = _load_cli()
    common = ["--pair", "OPEN", "--dataset", "cam", "--backbone", "lcnet_050", "--batch", "4", "--aug", "none",
              "--targets-dir", str(tgt), "--data-dir", str(data), "--device", "cpu", "--no-pretrained", "--seed", "2"]
    a = cli.parse_args(common + ["--epochs", "1", "--out", str(tmp_path / "o")])
    assert a.seed == 2 and a.batch == 4 and a.aug == "none" and a.budget is None
    assert cli.parse_args(["--pair", "CM", "--dataset", "crc", "--backbone", "x"]).batch == 128
    assert cli.main(common + ["--epochs", "2", "--out", str(tmp_path / "o"), "--budget", "0"]) == student.EXIT_RERUN
    assert cli.main(common + ["--epochs", "2", "--out", str(tmp_path / "o")]) == 0
    assert json.loads((tmp_path / "o" / "metrics.json").read_text())["seed"] == 2


def test_resolve_targets_dir(tmp_path):
    for rel in ("OPEN/crc", "open/OPEN/crc", "open/crc"):
        d = tmp_path / "res" / rel
        d.mkdir(parents=True)
        (d / "student_targets.npz").write_bytes(b"")
        assert student.resolve_targets_dir("OPEN", "crc", tmp_path / "res") == d
        (d / "student_targets.npz").unlink()
    with pytest.raises(FileNotFoundError, match="student_targets.npz"):
        student.resolve_targets_dir("OPEN", "crc", tmp_path / "res")


def test_cli_throughput_writes_json(tmp_path):
    data, tgt = make_fixture(tmp_path)
    cli = _load_cli()
    out = tmp_path / "tp"
    rc = cli.main(["--pair", "OPEN", "--dataset", "cam", "--backbone", "lcnet_050", "--batch", "4", "--targets-dir",
                   str(tgt), "--data-dir", str(data), "--device", "cpu", "--no-pretrained", "--out", str(out),
                   "--throughput", "--throughput-steps", "1", "--throughput-warmup", "0"])
    assert rc == 0
    tp = json.loads((out / "throughput.json").read_text())
    assert set(tp["results"]) == {"none", "torch", "k2"}
    for v in tp["results"].values():
        assert v["aug_images_per_s"] > 0 and v["train_step_images_per_s"] > 0
    assert tp["results"]["k2"]["impl_used"] in ("k2", "torch")


def test_cli_loss_weights_override(tmp_path):
    cli = _load_cli()
    a = cli.parse_args(["--pair", "OPEN", "--dataset", "crc", "--loss-weights", '{"rel": 0.0}'])
    assert a.loss_weights == {"rel": 0.0}
    assert cli.parse_args(["--pair", "OPEN", "--dataset", "crc"]).loss_weights is None
    with pytest.raises(SystemExit):
        cli.parse_args(["--pair", "OPEN", "--dataset", "crc", "--loss-weights", '{"bogus": 1}'])
    data, tgt = make_fixture(tmp_path)
    out = tmp_path / "o"
    assert cli.main(["--pair", "OPEN", "--dataset", "cam", "--backbone", "lcnet_050", "--batch", "4", "--aug", "none",
                     "--epochs", "1", "--targets-dir", str(tgt), "--data-dir", str(data), "--device", "cpu",
                     "--no-pretrained", "--out", str(out), "--loss-weights", '{"rel": 0.0}']) == 0
    assert json.loads((out / "metrics.json").read_text())["loss_weights"]["rel"] == 0.0
