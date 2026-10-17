"""Task 14: device runtime, app and bridge (cpu, offline). Fixture bundle = tiny random model in the R24 format."""
import hashlib
import json
import shutil
import socket
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
import twinread_runtime as rt  # noqa: E402  (device/ is on sys.path via conftest)

from make_device_bundle import C, CLASSES, THR, build  # noqa: E402


@pytest.fixture(scope="session")
def bundle(tmp_path_factory):
    return build(tmp_path_factory.mktemp("bundle"))


def test_verify_and_infer(bundle):
    assert rt.verify_bundle_dir(bundle)


def test_edge_model_infers_bundle(bundle):
    m = rt.EdgeModel(bundle, threads=1)
    x = np.random.default_rng(1).integers(0, 255, (3, 224, 224, 3), dtype=np.uint8)
    res = m.infer(x)
    assert len(res) == 3
    for r in res:
        assert r.label in CLASSES
        assert abs(sum(r.probsA) - 1) < 1e-4 and abs(sum(r.probsB) - 1) < 1e-4 and abs(sum(r.probsF) - 1) < 1e-4
        assert len(r.probsF) == C and 0 <= r.prob <= 1 and isinstance(r.review, bool)
        assert np.isfinite(r.review_score)
    assert m.infer(np.zeros((0, 224, 224, 3), np.uint8)) == []


def test_refuses_tampered_bundle(bundle, tmp_path):
    d = tmp_path / "b"
    shutil.copytree(bundle, d)
    assert rt.verify_bundle_dir(d)
    with open(d / "probes.npz", "ab") as f:
        f.write(b"x")
    assert not rt.verify_bundle_dir(d)
    with pytest.raises(ValueError):
        rt.EdgeModel(d)
    d2 = tmp_path / "b2"
    shutil.copytree(bundle, d2)
    (d2 / "SHA256SUMS").unlink()
    assert not rt.verify_bundle_dir(d2)
    d3 = tmp_path / "b3"
    shutil.copytree(bundle, d3)
    (d3 / "extra.bin").write_bytes(b"1")  # file not listed in SHA256SUMS
    assert not rt.verify_bundle_dir(d3)
    assert not rt.verify_bundle_dir(tmp_path / "missing")


def test_device_rule_matches_review_signals():
    sys.path.insert(0, str(ROOT / "src"))
    try:
        import twinread.review as rv
    except ImportError:  # lane worktree may lack review.py; read the main-tree copy
        import importlib.util
        spec = importlib.util.spec_from_file_location("rv", ROOT.parents[1] / "src/twinread/review.py")
        rv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rv)
    rng = np.random.default_rng(3)
    pA = rng.dirichlet(np.ones(4) * 0.5, 200)
    pB = rng.dirichlet(np.ones(4) * 0.5, 200)
    pF = rng.dirichlet(np.ones(4), 200)
    sig = rv.signals(pA, pB, pF)
    thr = {"js": 0.3, "review": 0.99}
    n_flag = 0
    for i in range(200):
        rev, score, reasons = rt.device_review_rule(pA[i], pB[i], pF[i], -10.0, thr)  # review head silent
        want_dis = bool(sig["s1_disagree"][i])
        want_js = bool(sig["s2_js"][i] > thr["js"])
        assert ("disagree" in reasons) == want_dis
        assert ("js" in reasons) == want_js
        assert abs(rt.js_divergence(pA[i], pB[i]) - sig["s2_js"][i]) < 1e-9
        assert rev == (want_dis or want_js)
        n_flag += rev
    assert 0 < n_flag < 200
    rev, _, reasons = rt.device_review_rule(pA[0], pA[0], pA[0], 5.0, thr)
    assert rev and reasons == ["review_head"]
    rev, _, reasons = rt.device_review_rule(pA[0], pA[0], pA[0], -5.0, thr, no_tissue=True)
    assert rev and reasons == ["no_tissue"]


@pytest.mark.parametrize("v", [0, 255])
def test_blank_frame_is_no_tissue(bundle, v):
    m = rt.EdgeModel(bundle, threads=1)
    x = np.full((1, 224, 224, 3), v, np.uint8)
    r = m.infer(x)[0]
    assert r.no_tissue and r.review and "no_tissue" in r.reasons
    assert rt.tissue_fraction(x) < 0.1 if v == 255 else True
    assert np.isfinite(r.probsF).all()


def test_tissue_fraction_rule():
    x = np.full((1, 224, 224, 3), 255, np.uint8)
    assert rt.tissue_fraction(x) == 0.0
    x[:, :, :112] = (200, 100, 150)  # half the tile is stained
    assert abs(rt.tissue_fraction(x) - 0.5) < 1e-6
    # black: OD = -log(1/256)*3 = 16.6 > 0.15 counts as "tissue" by the OD rule; no_tissue must still catch it
    assert rt.tissue_fraction(np.zeros((1, 224, 224, 3), np.uint8)) == 1.0


def test_black_frame_flagged_by_uniformity(bundle):
    r = rt.EdgeModel(bundle, threads=1).infer(np.zeros((1, 224, 224, 3), np.uint8))[0]
    assert r.no_tissue


@pytest.mark.parametrize("shape", [(480, 640), (720, 1280), (224, 224), (300, 200)])
def test_crop_from_frame_shapes(shape):
    h, w = shape
    f = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    f[h // 2 - 40:h // 2 + 40, w // 2 - 40:w // 2 + 40] = (10, 200, 30)
    out = rt.crop_from_frame(f)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8
    assert tuple(out[112, 112]) == (10, 200, 30)
    if shape == (224, 224):
        assert (out == f).all()


def test_crop_marker_mode_falls_back_to_center():
    f = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    assert (rt.crop_from_frame(f, "marker") == rt.crop_from_frame(f, "center")).all()


def test_bridge_absent_noop():
    import bridge
    for s in ("idle", "busy", "agree", "review", "no_tissue", "error"):
        assert bridge.send_status(s) in (True, False)
    assert bridge.send_status("bogus") is False


def test_bridge_socket_missing_returns_false(monkeypatch, tmp_path):
    import bridge
    monkeypatch.setattr(bridge, "ROUTER_SOCK", str(tmp_path / "nope.sock"))
    monkeypatch.setattr(bridge, "_app_utils_bridge", lambda: None)
    assert bridge.send_status("agree") is False


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_camera_missing_falls_back_to_replay(bundle):
    import uno_q_app
    app = uno_q_app.App(bundle, mode="camera", device="/dev/video_does_not_exist", threads=1)
    assert app.mode == "replay"
    assert app.mode_reason and "camera" in app.mode_reason.lower()
    app.step()
    st = app.state()
    assert st["mode"] == "replay" and st["call"] is not None


def test_app_binds_localhost_only_and_serves(bundle):
    import uno_q_app
    port = _free_port()
    app = uno_q_app.App(bundle, mode="sim", device=None, threads=1)
    srv = uno_q_app.make_server(app, port)
    assert srv.server_address[0] == "127.0.0.1"
    import threading
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        app.step()
        base = f"http://127.0.0.1:{port}"
        st = json.loads(urllib.request.urlopen(base + "/state", timeout=5).read())
        assert st["mode"] == "sim" and st["banner"].startswith("Research prototype")
        assert st["status"] in ("agree", "review", "no_tissue")
        jpg = urllib.request.urlopen(base + "/frame.jpg", timeout=5).read()
        assert jpg[:2] == b"\xff\xd8"
        html = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        assert "not a diagnostic device" in html
    finally:
        srv.shutdown()
        srv.server_close()


def test_device_modules_import_only_allowed():
    import ast
    ok = {"numpy", "onnxruntime", "cv2", "PIL"} | set(sys.stdlib_module_names)
    local = {"twinread_runtime", "bridge", "uno_q_app"}
    for f in ("twinread_runtime.py", "uno_q_app.py", "bridge.py"):
        tree = ast.parse((ROOT / "device" / f).read_text())
        for n in ast.walk(tree):
            names = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module] if isinstance(n, ast.ImportFrom) and n.module else []
            for nm in names:
                top = nm.split(".")[0]
                assert top in ok | local | {"arduino", "rig", "match_captures"}, (f, nm)
