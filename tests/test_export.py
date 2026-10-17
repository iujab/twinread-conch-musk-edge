"""Task 11: ONNX export, int8 quantisation, edge bundle (cpu, offline, random-init lcnet_050)."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import twinread_runtime as rt
from twinread import export
from twinread.student import TwinStudent

C, DA, DB = 4, 16, 24
CLASSES = ["a", "b", "c", "d"]


@pytest.fixture(scope="module")
def fp32(tmp_path_factory):
    torch.manual_seed(0)
    m = TwinStudent("lcnet_050", DA, DB, C, pretrained=False, hidden=32).eval()
    return m, export.export_onnx(m, tmp_path_factory.mktemp("onnx") / "m.onnx")


def _x(n, seed=0):
    return np.random.default_rng(seed).random((n, 3, 224, 224), dtype=np.float32)


def test_export_and_int8_run(fp32, tmp_path):
    m, p = fp32
    import onnxruntime as ort
    q = export.quantize_int8(p, _x(32), tmp_path / "q.onnx")
    for path in (p, q):
        s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        assert [i.name for i in s.get_inputs()] == ["x"]
        hA, hB, lg, rl = s.run(None, {"x": _x(3, 1)})
        assert hA.shape == (3, DA) and hB.shape == (3, DB) and lg.shape == (3, C) and rl.shape == (3,)
        assert np.isfinite(hA).all()
    with torch.no_grad():
        ref = m(torch.from_numpy(_x(3, 1)))["hA"].numpy()
    s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    assert np.allclose(s.run(None, {"x": _x(3, 1)})[0], ref, atol=1e-4)
    d = export.drift_report(p, q, _x(8, 2))
    assert set(d) >= {"cos_hA_mean", "cos_hB_mean", "label_agreement", "review_agreement"}
    assert -1 <= d["cos_hA_mean"] <= 1


def _bundle(tmp_path, pair="OPEN"):
    rng = np.random.default_rng(0)
    pr = {}
    for r, dd in (("A", DA), ("B", DB)):
        pr.update({f"{r}_coef": rng.normal(size=(C, dd)).astype("f4"), f"{r}_intercept": rng.normal(size=C).astype("f4"),
                   f"{r}_mean": rng.normal(size=dd).astype("f4"), f"{r}_scale": (rng.random(dd) + .5).astype("f4"),
                   f"{r}_classes": np.arange(C)})
    return export.build_bundle(pair, _FP32[1], {"A": rng.normal(size=(C, DA)).astype("f4"),
                                                "B": rng.normal(size=(C, DB)).astype("f4")},
                               pr, CLASSES, {"js": 0.1, "review": 0.5, "extra": 1}, tmp_path / pair.lower())


_FP32 = [None, None]


@pytest.fixture(autouse=True)
def _grab(fp32):
    _FP32[0], _FP32[1] = fp32


def test_bundle_checksums(tmp_path):
    b = _bundle(tmp_path)
    assert export.verify_bundle(b) and rt.verify_bundle_dir(b)
    assert sorted(p.name for p in b.iterdir()) == ["SHA256SUMS", "config.json", "model.onnx", "probes.npz",
                                                  "prompt_bank.npz"]
    cfg = json.loads((b / "config.json").read_text())
    assert cfg["thresholds"] == {"js": 0.1, "review": 0.5} and cfg["classes"] == CLASSES
    m = rt.EdgeModel(b, threads=1)
    res = m.infer(np.random.default_rng(1).integers(0, 255, (2, 224, 224, 3), dtype=np.uint8))
    assert len(res) == 2 and res[0].label in CLASSES
    raw = bytearray((b / "model.onnx").read_bytes())
    raw[len(raw) // 2] ^= 1
    (b / "model.onnx").write_bytes(bytes(raw))
    assert not export.verify_bundle(b)


def test_cm_bundle_marked_not_distributable(tmp_path):
    cfg = json.loads((_bundle(tmp_path, "CM") / "config.json").read_text())
    assert cfg["distributable"] is False and cfg["pair"] == "CM" and "not" in cfg["licence_note"].lower()
    cfg = json.loads((_bundle(tmp_path, "OPEN") / "config.json").read_text())
    assert cfg["distributable"] is True


def test_probe_key_check(tmp_path):
    with pytest.raises(KeyError):
        export.build_bundle("OPEN", _FP32[1], {"A": np.zeros((C, DA), "f4"), "B": np.zeros((C, DB), "f4")},
                            {"A_coef": np.zeros((C, DA))}, CLASSES, {"js": .1, "review": .5}, tmp_path / "x")


def test_export_leaves_model_in_eval(fp32):
    assert not fp32[0].training


OPEN = Path(__file__).resolve().parents[1] / "bundles" / "open"


@pytest.mark.skipif(not (OPEN / "model.onnx").exists(), reason="OPEN bundle not built (ONNX is git-ignored)")
def test_real_open_bundle_runs_on_device_runtime():
    assert export.verify_bundle(OPEN)
    m = rt.EdgeModel(OPEN, threads=2)
    cfg = json.loads((OPEN / "config.json").read_text())
    assert cfg["distributable"] is True and cfg["pair"] == "OPEN" and len(cfg["classes"]) == 9
    x = np.load(Path(__file__).resolve().parents[1] / "data" / "crc_test_x.npy", mmap_mode="r")
    y = np.load(Path(__file__).resolve().parents[1] / "data" / "crc_test_y.npy")
    idx = np.random.default_rng(0).choice(len(x), 64, replace=False)
    res = m.infer(np.ascontiguousarray(x[idx]))
    acc = np.mean([r.label == cfg["classes"][y[i]] for r, i in zip(res, idx)])
    assert acc > 0.8 and all(np.isfinite(r.review_score) for r in res)


def test_bundle_rebuild_is_byte_identical(tmp_path):
    import time
    a = _bundle(tmp_path / "a")
    time.sleep(2.1)  # zip timestamps have 2 s resolution
    b = _bundle(tmp_path / "b")
    assert (a / "SHA256SUMS").read_text() == (b / "SHA256SUMS").read_text()
    z = np.load(a / "probes.npz", allow_pickle=False)
    assert z["A_coef"].shape == (C, DA) and z["A_classes"].dtype.kind == "i"
