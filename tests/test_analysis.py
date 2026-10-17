import json
import math
import zlib

import numpy as np
import pytest

from twinread import analyze, blindspots, fusion, prompts, review, stats


def test_js_divergence_identical_is_zero():
    p = np.array([[0.2, 0.8], [0.5, 0.5]])
    assert np.allclose(review.js_divergence(p, p), 0.0)


def test_js_divergence_disjoint_is_one():
    assert np.allclose(review.js_divergence(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])), 1.0)


def test_risk_coverage_perfect_ranking():
    n, e = 100, 10
    correct = np.ones(n, bool)
    correct[:e] = False
    score = np.where(correct, 0.0, 1.0)
    rc = review.risk_coverage(score, correct)
    analytic = sum(max(0, k - (n - e)) / k for k in range(1, n + 1)) / n
    assert math.isclose(rc["aurc"], analytic)
    assert rc["risk"][n - e - 1] == 0.0 and math.isclose(rc["coverage"][-1], 1.0)


def test_errors_caught_budget():
    correct = np.ones(100, bool)
    correct[:10] = False
    score = np.where(correct, 0.1, 0.9) + np.random.default_rng(0).uniform(0, 0.01, 100)
    r = review.errors_caught(score, correct)
    assert r[0.10]["errors_caught"] == 1.0 and r[0.10]["acc_cleared"] == 1.0
    assert r[0.05]["errors_caught"] == 0.5


def test_threshold_for_budget_flags_about_budget():
    s = np.random.default_rng(1).uniform(size=1000)
    t = review.threshold_for_budget(s, 0.10)
    assert abs((s > t).mean() - 0.10) < 0.005


def test_mcnemar_known_values():
    a = np.array([1] * 10 + [0] * 2 + [1] * 20, bool)
    b = np.array([0] * 10 + [1] * 2 + [1] * 20, bool)
    r = stats.mcnemar_exact(a, b)
    assert (r["b"], r["c"]) == (10, 2) and abs(r["p"] - 0.0386) < 1e-3
    assert stats.mcnemar_exact(a, a)["p"] == 1.0


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    pred = np.where(rng.uniform(size=300) < 0.8, y, rng.integers(0, 3, 300))
    lo, hi = stats.bootstrap_ci(stats.balanced_accuracy, y, pred, n=200)
    assert lo <= stats.balanced_accuracy(y, pred) <= hi and hi - lo > 0


def test_blindspot_table_counts():
    y = np.array([0, 0, 0, 0, 1, 1])
    pa = np.array([0, 0, 1, 1, 1, 0])
    pb = np.array([0, 1, 0, 1, 1, 0])
    t = blindspots.blindspot_table(pa, pb, y, ["a", "b"]).set_index("class")
    assert t.loc["a", ["both_right", "A_only", "B_only", "both_wrong", "both_wrong_same_label"]].tolist() == [1, 1, 1, 1, 1]
    assert t.loc["b", ["both_right", "both_wrong", "both_wrong_same_label"]].tolist() == [1, 1, 1]
    assert t["n"].tolist() == [4, 2]


def test_confident_shared_errors():
    y = np.array([0, 0, 0])
    pA = np.array([[0.1, 0.9], [0.4, 0.6], [0.9, 0.1]])
    pB = np.array([[0.05, 0.95], [0.3, 0.7], [0.1, 0.9]])
    assert blindspots.confident_shared_errors(pA, pB, y).tolist() == [0]


def test_fuse_average_rows_sum_to_one():
    rng = np.random.default_rng(0)
    pA, pB = rng.dirichlet(np.ones(4), 20), rng.dirichlet(np.ones(4), 20)
    f = fusion.fuse_average(pA, pB)
    assert np.allclose(f.sum(1), 1) and np.allclose(f, (pA + pB) / 2)


def _blobs(n, rng, d=8, k=3, sep=3.0):
    y = rng.integers(0, k, n)
    mu = np.random.default_rng(99).normal(size=(k, d)) * sep
    return (mu[y] + rng.normal(size=(n, d))).astype(np.float32), y


def test_fit_concat_lr_beats_chance_on_separable():
    rng = np.random.default_rng(0)
    zA, y = _blobs(300, rng)
    zB = zA + rng.normal(size=zA.shape).astype(np.float32)
    zAv, yv = _blobs(100, rng)
    zBv = zAv + rng.normal(size=zAv.shape).astype(np.float32)
    f = fusion.fit_concat_lr(zA, zB, y, zAv, zBv, yv)
    p = f.predict_proba(zAv, zBv)
    assert p.shape == (100, 3) and np.allclose(p.sum(1), 1) and (p.argmax(1) == yv).mean() > 0.9
    g = fusion.fit_gated_mlp(zA, zB, y, zAv, zBv, yv, epochs=5)
    assert (g.predict_proba(zAv, zBv).argmax(1) == yv).mean() > 0.8


def test_text_bank_with_fake_reader_and_zeroshot():
    class Fake:
        def encode_texts(self, texts):
            out = []
            for t in texts:
                r = np.random.default_rng(zlib.crc32(t.encode())).normal(size=16)
                out.append(r / np.linalg.norm(r))
            return np.array(out, np.float32)

    bank = prompts.text_bank(Fake(), "cam")
    classes, syn, templ = prompts.load_prompts("cam")
    assert bank.shape == (2, 16) and np.allclose(np.linalg.norm(bank, axis=1), 1, atol=1e-5)
    assert len(classes) == 2 and all(len(syn[c]) >= 3 for c in classes) and all("{}" in t for t in templ)
    p = prompts.zeroshot_probs(bank, bank)
    assert np.allclose(p.sum(1), 1) and (p.argmax(1) == [0, 1]).all()


def test_crc_prompts_cover_classes():
    classes, syn, _ = prompts.load_prompts("crc")
    assert classes == ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
    assert all(3 <= len(syn[c]) <= 5 for c in classes)
    assert any("blank glass" in s for s in syn["BACK"]) and any("background" in s for s in syn["BACK"])


def test_preregistration_refuses_overwrite(tmp_path):
    p = analyze.write_preregistration("CM", "cam", tmp_path)
    doc = json.loads(p.read_text())
    assert set(doc["prompts_sha256"]) == {"crc.json", "cam.json", "templates.json"}
    assert doc["primary_flag"] == "s4" and doc["budgets"] == [0.05, 0.1, 0.2]
    before = p.read_bytes()
    with pytest.raises(FileExistsError):
        analyze.write_preregistration("CM", "cam", tmp_path)
    assert p.read_bytes() == before


def _make_fixture(tmp_path, d=12):
    rng = np.random.default_rng(0)
    emb = tmp_path / "data" / "emb" / "cam"
    emb.mkdir(parents=True)
    labels = {}
    sizes = {"train": 400, "val": 150, "test": 200}
    for s, n in sizes.items():
        y = rng.integers(0, 2, n)
        labels[s] = y
        for r in ("conch", "musk"):
            mu = np.random.default_rng(zlib.crc32(r.encode()) % 1000).normal(size=(2, d)) * 1.2
            probe = mu[y] + rng.normal(size=(n, d))
            al = np.random.default_rng(5).normal(size=(d, 8))
            al = probe @ al
            al /= np.linalg.norm(al, axis=1, keepdims=True)
            np.save(emb / f"{r}_probe_{s}.npy", probe.astype(np.float16))
            np.save(emb / f"{r}_aligned_{s}.npy", al.astype(np.float16))
    bank = np.random.default_rng(3).normal(size=(2, 8)).astype(np.float32)
    bank /= np.linalg.norm(bank, axis=1, keepdims=True)
    return tmp_path / "data", labels, {"conch": bank, "musk": bank}


def test_run_pair_analysis_test_evaluated_once(tmp_path, monkeypatch):
    data_dir, labels, banks = _make_fixture(tmp_path)
    calls = []
    real = analyze._test_metrics

    def counting(ctx):
        calls.append((tmp_path / "res" / "CM" / "cam" / "preregistration.json").exists())
        return real(ctx)

    monkeypatch.setattr(analyze, "_test_metrics", counting)
    res = analyze.run_pair_analysis("CM", "cam", tmp_path / "res", data_dir=data_dir, labels=labels, banks=banks,
                                    epochs=3, n_boot=30)
    assert calls == [True]
    out = tmp_path / "res" / "CM" / "cam"
    for f in ("readers.json", "fusion.json", "review.json", "blindspots.csv", "preregistration.json",
              "student_targets.npz", "probes_aligned.npz"):
        assert (out / f).exists(), f
    pre = json.loads((out / "preregistration.json").read_text())
    assert pre["primary_fusion"] == res["primary"] and pre["primary_fusion"] in analyze.CANDIDATES
    t = np.load(out / "student_targets.npz")
    for s, n in (("train", 400), ("val", 150), ("test", 200)):
        assert t[f"tA_{s}"].shape == (n, 8) and t[f"fused_logits_{s}"].shape == (n, 2)
        assert set(np.unique(t[f"review_{s}"])) <= {0, 1} and (t[f"y_{s}"] == labels[s]).all()
    pa = np.load(out / "probes_aligned.npz")
    assert {"A_coef", "A_intercept", "A_mean", "A_scale", "B_coef"} <= set(pa.files)
    assert res["fusion"]["candidates"][res["primary"]]["bal_acc"] > 0.6
    with pytest.raises(FileExistsError):  # rerun must not clobber the pre-registration
        analyze.run_pair_analysis("CM", "cam", tmp_path / "res", data_dir=data_dir, labels=labels, banks=banks, epochs=1)
    assert calls == [True]
