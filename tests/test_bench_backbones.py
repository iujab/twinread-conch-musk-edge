import json
import subprocess
import sys
from pathlib import Path

import numpy as np

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
