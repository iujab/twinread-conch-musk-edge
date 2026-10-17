import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_bench_runs_simulated(tmp_path):
    from export_backbones import export_backbone
    calib = np.random.RandomState(0).randint(0, 255, (8, 224, 224, 3), dtype=np.uint8)
    d = tmp_path / "bb"
    d.mkdir()
    export_backbone("lcnet_050", d, calib, pretrained=False)
    assert (d / "lcnet_050_fp32.onnx").exists() and (d / "lcnet_050_int8.onnx").exists()
    out = tmp_path / "lat.json"
    r = subprocess.run([sys.executable, str(ROOT / "device" / "bench_backbones.py"), "--dir", str(d),
                        "--threads", "1", "--runs", "3", "--simulate", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    j = json.loads(out.read_text())
    assert j["simulated"] is True and "host" in j
    assert {(x["backbone"], x["precision"]) for x in j["rows"]} == {("lcnet_050", "fp32"), ("lcnet_050", "int8")}
    for row in j["rows"]:
        for k in ("backbone", "precision", "threads", "p50_ms", "p90_ms", "rss_mb", "size_mb"):
            assert k in row
        assert row["p50_ms"] > 0 and row["p90_ms"] >= row["p50_ms"]


def test_bench_imports_no_torch():
    src = (ROOT / "device" / "bench_backbones.py").read_text()
    assert "import torch" not in src and "import timm" not in src


def _tiny_onnx(path):
    """A real 224-px ONNX model in milliseconds: x [1,3,224,224] -> ReduceMean."""
    import onnx
    from onnx import TensorProto, helper
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 224, 224])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 1, 1])
    g = helper.make_graph([helper.make_node("ReduceMean", ["x"], ["y"], keepdims=1)], "tiny", [x], [y])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(path))


def test_bench_one_failing_child_keeps_other_rows(tmp_path):
    """I5: a child that fails (here: a corrupt model) becomes an error row; the good rows are kept; exit code 1."""
    d = tmp_path / "bb"
    d.mkdir()
    _tiny_onnx(d / "good_fp32.onnx")
    (d / "bad_int8.onnx").write_bytes(b"not an onnx model")
    out = tmp_path / "lat.json"
    r = subprocess.run([sys.executable, str(ROOT / "device" / "bench_backbones.py"), "--dir", str(d),
                        "--threads", "1", "--runs", "3", "--simulate", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "1 of 2 rows failed" in r.stderr
    j = json.loads(out.read_text())
    assert j["complete"] is True and j["n_errors"] == 1 and j["simulated"] is True
    rows = {x["backbone"]: x for x in j["rows"]}
    assert "error" in rows["bad"] and "p50_ms" not in rows["bad"]
    assert "error" not in rows["good"] and rows["good"]["p50_ms"] > 0


def test_bench_writes_json_after_every_row(tmp_path, monkeypatch):
    """I5: an interrupted run (e.g. the board loses power) keeps every row measured so far."""
    import bench_backbones as bb
    d = tmp_path / "bb"
    d.mkdir()
    for n in ("a_fp32", "b_fp32", "c_fp32"):
        (d / f"{n}.onnx").write_bytes(b"x")
    calls = []

    def fake(path, threads, runs, timeout=None):
        calls.append(path)
        if len(calls) == 2:
            raise subprocess.CalledProcessError(-9, ["child"], stderr="Killed")
        if len(calls) == 3:
            raise KeyboardInterrupt
        return 1.0, 2.0, 3.0, 4.0
    monkeypatch.setattr(bb, "bench_isolated", fake)
    out = tmp_path / "lat.json"
    monkeypatch.setattr(sys, "argv", ["bench_backbones.py", "--dir", str(d), "--threads", "4", "--out", str(out)])
    with pytest.raises(KeyboardInterrupt):
        bb.main()
    j = json.loads(out.read_text())
    assert j["complete"] is False and [r["backbone"] for r in j["rows"]] == ["a", "b"]
    assert j["rows"][0]["p50_ms"] == 1.0 and "child exited -9" in j["rows"][1]["error"]
