"""Task 14: device runtime, app and bridge (cpu, offline). Fixture bundle = tiny random model in the R24 format."""
import hashlib
import json
import re
import shutil
import socket
import sys
import time
import urllib.error
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
    import twinread.review as rv  # src/ is on sys.path via conftest
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
    assert rt.tissue_fraction(np.zeros((1, 224, 224, 3), np.uint8)) == 0.0  # black is not tissue


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
        assert st["mode"] == "sim" and st["status"] in ("agree", "review", "no_tissue")
        jpg = urllib.request.urlopen(base + "/frame.jpg", timeout=5).read()
        assert jpg[:2] == b"\xff\xd8"
        html = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        assert "<title>TwinRead</title>" in html
    finally:
        srv.shutdown()
        srv.server_close()


def test_device_modules_import_only_allowed():
    import ast
    ok = {"numpy", "onnxruntime", "cv2", "PIL"} | set(sys.stdlib_module_names)
    local = {"twinread_runtime", "bridge", "uno_q_app", "camera", "demo", "explainer", "explain_prompt", "explain_check", "explain_vocab"}
    for f in ("twinread_runtime.py", "uno_q_app.py", "bridge.py", "camera.py", "demo.py", "explainer.py", "explain_check.py", "explain_vocab.py"):
        tree = ast.parse((ROOT / "device" / f).read_text())
        for n in ast.walk(tree):
            names = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module] if isinstance(n, ast.ImportFrom) and n.module else []
            for nm in names:
                top = nm.split(".")[0]
                assert top in ok | local | {"arduino", "rig", "match_captures"}, (f, nm)


def test_half_black_half_white_is_no_tissue(bundle):
    x = np.full((1, 224, 224, 3), 255, np.uint8)
    x[:, :, :112] = 0
    assert rt.tissue_fraction(x) == 0.0
    r = rt.EdgeModel(bundle, threads=1).infer(x)[0]
    assert r.no_tissue and r.review


def test_dim_noisy_white_screen_is_no_tissue(bundle):
    rng = np.random.default_rng(0)
    x = np.clip(rng.normal(225, 4, (1, 224, 224, 3)), 0, 255).astype(np.uint8)
    assert rt.tissue_fraction(x) < 0.1
    assert rt.EdgeModel(bundle, threads=1).infer(x)[0].no_tissue


def test_real_crc_tiles_are_tissue(bundle):
    p = ROOT / "data" / "crc_test_x.npy"
    if not p.exists():
        pytest.skip("data/crc_test_x.npy not present")
    x = np.load(p, mmap_mode="r")
    y = np.load(ROOT / "data" / "crc_test_y.npy")
    idx = np.sort(np.random.default_rng(0).choice(np.flatnonzero(y != 0), 60, replace=False))  # not ADI: fat can be near-empty
    tiles = np.asarray(x[idx])
    assert min(rt.tissue_fraction(t[None]) for t in tiles) >= 0.1
    res = rt.EdgeModel(bundle, threads=1).infer(tiles)
    assert not any(r.no_tissue for r in res)


@pytest.mark.parametrize("k", [2, 3])
def test_probe_parity_with_sklearn(tmp_path, bundle, k):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    shutil.copytree(bundle, tmp_path / "b")
    d = tmp_path / "b"
    rng = np.random.default_rng(k)
    pr = {}
    ref = {}
    for r, dd in (("A", 16), ("B", 24)):
        y = rng.integers(0, k, 300)
        X = rng.normal(size=(300, dd)) + 2.0 * np.eye(k)[y] @ rng.normal(size=(k, dd))  # separable-ish
        sc = StandardScaler().fit(X)
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
        if k == 2:  # R24 binary expansion: softmax over [0, w.x + b] == sigmoid
            coef = np.vstack([np.zeros(dd), lr.coef_[0]])
            icpt = np.array([0.0, lr.intercept_[0]])
        else:
            coef, icpt = lr.coef_, lr.intercept_
        pr.update({f"{r}_coef": coef.astype("f8"), f"{r}_intercept": icpt.astype("f8"), f"{r}_mean": sc.mean_,
                   f"{r}_scale": sc.scale_, f"{r}_classes": lr.classes_})
        ref[r] = (X, sc, lr)
    np.savez(d / "probes.npz", **pr)
    from make_device_bundle import write_sums
    write_sums(d)
    m = rt.EdgeModel(d, threads=1)
    for r in ("A", "B"):
        X, sc, lr = ref[r]
        got = m._probe(r, X.astype(np.float64))
        assert got.shape == (300, k)
        assert np.abs(got - lr.predict_proba(sc.transform(X))).max() < 1e-6


@pytest.mark.parametrize("mode,label", [("replay", "Replay frame"), ("sim", "Simulated frame")])
def test_state_labels_host_and_frame_source(bundle, mode, label):
    """Task 18 fix (review 6b): a screenshot must say which host produced the latency and never call a replayed or
    simulated frame 'live'. Off the board the host label says so explicitly."""
    import uno_q_app
    app = uno_q_app.App(bundle, mode=mode, threads=1)
    app.step()
    st = app.state()
    assert st["frame_label"] == label
    host = st["host"]
    assert host["machine"] and isinstance(host["is_uno_q"], bool)
    if not host["is_uno_q"]:
        assert "not the UNO Q" in st["host_label"] and host["machine"] in st["host_label"]
    ui = (ROOT / "device" / "ui" / "app.js").read_text()
    assert "frame_label" in ui and "host_label" in ui


def test_uno_q_detection_reads_device_tree(tmp_path, monkeypatch):
    import uno_q_app
    model = tmp_path / "model"
    model.write_text("Arduino UNO Q\x00")
    monkeypatch.setattr(uno_q_app.platform, "machine", lambda: "aarch64")
    assert uno_q_app.host_info(model)["is_uno_q"] is True
    monkeypatch.setattr(uno_q_app.platform, "machine", lambda: "x86_64")
    assert uno_q_app.host_info(model)["is_uno_q"] is False
    assert uno_q_app.host_info(tmp_path / "missing")["is_uno_q"] is False


# ---- final review fixes: bridge msgpack encoder (struct), offline installer (I4) ---------------------------------
def test_bridge_mp_request_large_msgid():
    """bridge.py used struct without importing it: msgid >= 128 raised NameError."""
    import bridge
    small = bridge._mp_request(1, "set_state", "agree")
    assert small[:3] == b"\x94\x00\x01" and small.endswith(b"\x91\xa5agree")
    big = bridge._mp_request(300, "set_state", "review")
    assert big[:5] == b"\x94\x00\xcd\x01\x2c" and big.endswith(b"\x91\xa6review")


INSTALL = Path(__file__).resolve().parents[1] / "device" / "install_offline.sh"


def _fake_python3(tmp_path, version="Python 3.13.5"):
    bin_ = tmp_path / "bin"
    bin_.mkdir(parents=True, exist_ok=True)
    f = bin_ / "python3"
    f.write_text("#!/bin/bash\n"
                 f'if [ "$1" = --version ]; then echo "{version}"; exit 0; fi\n'
                 f'echo "$@" >> {tmp_path / "calls.txt"}\n')
    f.chmod(0o755)
    return bin_


def test_install_offline_stages_manylinux_2_28_pinned(tmp_path):
    import subprocess
    bin_ = _fake_python3(tmp_path)
    wheels = tmp_path / "wheels"
    env = {"PATH": f"{bin_}:/usr/bin:/bin", "WHEELS": str(wheels)}
    r = subprocess.run(["bash", str(INSTALL), "stage"], env={**env, "PYVER": "Python 3.13.5"}, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr
    args = (tmp_path / "calls.txt").read_text().split()
    assert args[:3] == ["-m", "pip", "download"]
    plats = [args[i + 1] for i, a in enumerate(args) if a == "--platform"]
    assert plats[0] == "manylinux_2_28_aarch64" and "manylinux2014_aarch64" in plats
    assert args[args.index("--python-version") + 1] == "3.13" and "--only-binary=:all:" in args
    for pin in ("onnxruntime==1.31.0", "numpy==2.5.4", "pillow==13.0.0", "opencv-python-headless==5.0.0.93"):
        assert pin in args, pin
    assert (wheels / "PYVER").read_text().strip() == "3.13"
    # on the board: wheels staged for 3.13 but the board runs 3.11 -> refuse before creating a venv
    bin2 = _fake_python3(tmp_path / "board", "Python 3.11.2")
    r = subprocess.run(["bash", str(INSTALL), "install"], env={"PATH": f"{bin2}:/usr/bin:/bin", "WHEELS": str(wheels)},
                       capture_output=True, text=True)
    assert r.returncode == 2 and "re-stage with PYVER=3.11" in r.stderr
    r = subprocess.run(["bash", str(INSTALL), "stage"], env={**env, "PYVER": "banana"}, capture_output=True, text=True)
    assert r.returncode == 2
    assert subprocess.run(["bash", "-n", str(INSTALL)]).returncode == 0


@pytest.mark.parametrize("pair,heads", [("CM", ["Head A (CONCH-space)", "Head B (MUSK-space)"]),
                                        ("OPEN", ["Head A (space A)", "Head B (space B)"]),
                                        ("test", ["Head A (space A)", "Head B (space B)"])])
def test_ui_names_student_heads_not_readers(pair, heads):
    """I7: the device's two 'readers' are two heads on one shared student backbone; the UI must not call them readers."""
    import uno_q_app
    assert uno_q_app.head_labels(pair) == heads
    ui = (ROOT / "device" / "ui" / "app.js").read_text() + (ROOT / "device" / "ui" / "index.html").read_text()
    assert "reader" not in ui.lower() and "Both heads" in ui
    assert "Reader A" not in (ROOT / "device" / "uno_q_app.py").read_text()


def test_state_carries_head_labels(bundle):
    import uno_q_app
    app = uno_q_app.App(bundle, mode="sim", threads=1)
    st = app.state()
    assert st["heads"] == ["Head A (space A)", "Head B (space B)"] and "readers" not in st


def test_int8_flag_fidelity_metrics():
    """scripts/int8_flag_fidelity.py: agreement/kappa/recall of a device flag vs the teacher flag, on a toy case."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import int8_flag_fidelity as iff
    flag = np.array([1, 1, 0, 0, 0, 1, 0, 0], bool)
    teach = np.array([1, 0, 1, 0, 0, 1, 0, 0], bool)
    pred, y = np.array([0, 1, 2, 3, 4, 5, 6, 7]), np.array([9, 1, 2, 3, 4, 9, 9, 7])  # errors at 0, 5, 6
    m = iff.flag_fidelity(flag, teach, pred, y)
    assert m["agreement"] == 6 / 8 and m["teacher_recall"] == 2 / 3 and m["precision_vs_teacher"] == 2 / 3
    po, pe = 6 / 8, (3 / 8) * (3 / 8) + (5 / 8) * (5 / 8)
    assert m["kappa"] == pytest.approx((po - pe) / (1 - pe))
    assert m["errors_caught"] == 2 / 3 and m["acc_cleared"] == 4 / 5 and m["fraction"] == 3 / 8


# ---- live camera and kiosk: USB camera discovery, newest-frame capture, unplug/replug, MJPEG feed, boot files ---------
class FakeCap:
    """Stands in for cv2.VideoCapture: delivers numbered frames until unplugged."""

    def __init__(self, shape=(480, 640, 3)):
        self.alive, self.shape, self.n, self.released, self.grabs = True, shape, 0, False, 0

    def isOpened(self):
        return True

    def grab(self):
        self.grabs += 1
        time.sleep(0.002)
        return self.alive

    def retrieve(self):
        self.n += 1
        return True, np.full(self.shape, self.n % 250, np.uint8)

    def read(self):
        return self.grab(), self.retrieve()[1]

    def release(self):
        self.released = True


def _until(cond, timeout=5.0):
    t = time.monotonic()
    while time.monotonic() - t < timeout:
        if cond():
            return True
        time.sleep(0.005)
    return False


def _source(caps, finder=None):
    """CameraSource whose opener hands out `caps` in turn."""
    import camera
    it = iter(caps)
    return camera.CameraSource(camera.AUTO, finder=finder or (lambda: [{"path": "/dev/video2", "name": "fake"}]),
                               opener=lambda *a: next(it), retry_s=0.01, decode_hz=200.0)


def _v4l_node(sysfs, n, driver, index, name="cam"):
    node = sysfs / f"video{n}"
    (node / "device").mkdir(parents=True)
    drv = sysfs.parent / "drivers" / driver
    drv.mkdir(parents=True, exist_ok=True)
    (node / "device" / "driver").symlink_to(drv)
    (node / "index").write_text(f"{index}\n")
    (node / "name").write_text(f"{name}\n")


def test_find_usb_cameras_skips_codecs_and_metadata_nodes(tmp_path):
    import camera
    sysfs = tmp_path / "video4linux"
    _v4l_node(sysfs, 0, "qcom-venus", 0, "qcom-venus-decoder")
    _v4l_node(sysfs, 1, "qcom-venus", 0, "qcom-venus-encoder")
    _v4l_node(sysfs, 10, "uvcvideo", 0, "second")
    _v4l_node(sysfs, 2, "uvcvideo", 0, "first")
    _v4l_node(sysfs, 3, "uvcvideo", 1, "first metadata")
    assert [(c["path"], c["name"]) for c in camera.find_usb_cameras(sysfs)] == [("/dev/video2", "first"),
                                                                               ("/dev/video10", "second")]
    assert camera.find_usb_cameras(tmp_path / "missing") == []


def test_camera_source_keeps_newest_frame_and_survives_replug():
    first, second = FakeCap(), FakeCap((720, 1280, 3))
    src = _source([first, second]).start()
    try:
        assert _until(lambda: src.latest()[1] >= 5)
        st = src.status()
        assert st["connected"] and (st["width"], st["height"]) == (640, 480) and st["device"] == "/dev/video2"
        assert src.latest()[2] < 1.0
        first.alive = False  # unplugged
        assert _until(lambda: first.released)
        assert _until(lambda: src.status()["width"] == 1280 and src.status()["connected"])  # plugged back in
        assert _until(lambda: src.latest()[0] is not None and src.latest()[0].shape[0] == 720)
    finally:
        src.stop()
    assert second.released


def test_no_camera_found_gives_a_plain_reason():
    src = _source([], finder=lambda: [])
    assert src.open_once() is False
    assert src.status()["connected"] is False and "no USB camera" in src.status()["reason"]
    assert src.latest()[0] is None


def test_wait_mode_reads_nothing_until_a_camera_appears(bundle):
    import uno_q_app
    cams = []
    src = _source([FakeCap()], finder=lambda: list(cams))
    app = uno_q_app.App(bundle, mode="camera", threads=1, camera_loss="wait", source=src)
    try:
        assert _until(lambda: not src.status()["opening"])
        app.step()
        st = app.state()
        assert st["status"] == "no_camera" and st["call"] is None and st["mode"] == "camera"
        assert st["camera"]["connected"] is False and app.frame_jpg() == b""
        cams.append({"path": "/dev/video2", "name": "fake"})
        assert _until(lambda: src.status()["connected"])
        app.step()
        st = app.state()
        assert st["status"] in ("agree", "review", "no_tissue") and st["call"] is not None
        assert st["frame_label"] == "Live" and st["camera"]["name"] == "fake" and st["mode"] == "camera"
    finally:
        app.close()


def test_stale_camera_frame_produces_no_decision(bundle, monkeypatch):
    import uno_q_app
    app = uno_q_app.App(bundle, mode="camera", threads=1, camera_loss="wait", source=_source([FakeCap()]))
    try:
        assert _until(lambda: app.cam.latest()[0] is not None)
        monkeypatch.setattr(uno_q_app, "STALE_S", -1.0)  # every frame now counts as too old
        app.step()
        assert app.state()["status"] == "no_camera" and app.state()["call"] is None
    finally:
        app.close()


def test_camera_lost_falls_back_only_in_replay_policy(bundle):
    import uno_q_app
    cap = FakeCap()
    app = uno_q_app.App(bundle, mode="camera", threads=1, camera_loss="replay", source=_source([cap]))
    assert app.mode == "camera" and app.cam is not None
    cap.alive = False
    assert _until(lambda: not app.cam.status()["connected"])
    app.step()
    assert app.mode == "replay" and app.state()["frame_label"] == "Replay frame" and "camera" in app.mode_reason
    app.close()


def test_stream_serves_live_jpegs_and_counts_watchers(bundle):
    import threading
    import uno_q_app
    port = _free_port()
    app = uno_q_app.App(bundle, mode="camera", threads=1, camera_loss="wait", source=_source([FakeCap()]), preview_fps=60)
    srv = uno_q_app.make_server(app, port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        r = urllib.request.urlopen(f"http://127.0.0.1:{port}/stream.mjpg", timeout=5)
        assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace")
        buf = b""
        while buf.count(b"\xff\xd8\xff") < 3:
            buf += r.read(4096)
        assert b"--frame\r\nContent-Type: image/jpeg" in buf
        assert app.stream_clients == 1
        r.close()
        assert _until(lambda: app.stream_clients == 0)
    finally:
        app.close()
        srv.shutdown()
        srv.server_close()


def test_preview_marks_the_read_region_without_touching_the_frame():
    import cv2
    import uno_q_app
    f = np.full((720, 1280, 3), 40, np.uint8)
    before = f.copy()
    img = cv2.imdecode(np.frombuffer(uno_q_app.preview_jpeg(f), np.uint8), cv2.IMREAD_COLOR)
    assert (f == before).all()
    assert img.shape[1] == uno_q_app.PREVIEW_MAX_W
    x0, y0, _, _ = rt.center_box(*img.shape[:2])
    b, g, r = img[img.shape[0] // 2, x0 + 2].astype(int)
    assert r > 200 and g > 170 and b < 90  # yellow on the square's edge


def test_led_is_only_told_about_changes(bundle, monkeypatch):
    import bridge
    import uno_q_app
    calls = []
    monkeypatch.setattr(bridge, "send_status", lambda s: (calls.append(s), True)[1])
    app = uno_q_app.App(bundle, mode="camera", threads=1, camera_loss="wait", source=_source([FakeCap()]))
    try:
        assert _until(lambda: app.cam.latest()[0] is not None)
        for _ in range(5):
            app.step()
        assert len(calls) == 2 and calls[0] == "idle" and "busy" not in calls  # uniform frames: one decision state
    finally:
        app.close()


def test_bridge_reads_a_reply_split_into_chunks(monkeypatch, tmp_path):
    """On the board the router sometimes splits its 5-byte reply; a single recv() reported the LED as failed."""
    import threading
    import bridge
    path = str(tmp_path / "r.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(1)

    def serve():
        conn, _ = srv.accept()
        conn.recv(256)
        conn.sendall(b"\x94\x01")
        time.sleep(0.05)
        conn.sendall(b"\x01\xc0\xc0")
        conn.close()
    threading.Thread(target=serve, daemon=True).start()
    monkeypatch.setattr(bridge, "ROUTER_SOCK", path)
    monkeypatch.setattr(bridge, "_app_utils_bridge", lambda: None)
    assert bridge.send_status("agree") is True
    srv.close()


def test_ui_is_self_contained(bundle):
    import re
    import threading
    import uno_q_app
    ui = ROOT / "device" / "ui"
    text = "".join((ui / n).read_text() for n in ("index.html", "app.js", "style.css"))
    assert not re.search(r"https?://|//fonts\.|url\(", text)  # must work with no network, and it loads nothing else
    assert "/stream.mjpg" in text and "no_camera" in text
    assert sorted(p.name for p in ui.iterdir()) == ["app.js", "index.html", "style.css"]
    port = _free_port()
    app = uno_q_app.App(bundle, mode="sim", threads=1)
    srv = uno_q_app.make_server(app, port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        for f in ("", "app.js", "style.css"):
            assert urllib.request.urlopen(f"http://127.0.0.1:{port}/{f}", timeout=5).status == 200
    finally:
        srv.shutdown()
        srv.server_close()


def test_boot_files_start_offline_and_without_login():
    import subprocess
    deploy = ROOT / "device" / "deploy"
    for sh in ("install.sh", "kiosk.sh"):
        assert subprocess.run(["bash", "-n", str(deploy / sh)]).returncode == 0, sh
    unit = (deploy / "twinread.service").read_text()
    assert "network" not in unit.split("[Service]")[0].replace("no network dependency", "")
    assert "IPAddressDeny=any" in unit and "IPAddressAllow=localhost" in unit and "Restart=always" in unit
    assert "ExecStartPre=-/usr/bin/timeout 10 /usr/local/bin/arduino-reset" in unit  # the LED sketch hangs on a cold boot without it
    assert "--camera-loss wait" in unit and "--device auto" in unit and "WantedBy=multi-user.target" in unit
    install = (deploy / "install.sh").read_text()
    assert "tailscale serve --bg --tcp 8080 tcp://127.0.0.1:8080" in install  # tailnet only
    # Standby is the boot default with or without the optional file, and the unit does not depend on the file existing.
    assert "Environment=TWINREAD_ARGS=--standby" in unit and "EnvironmentFile=-/etc/default/twinread" in unit
    assert unit.index("Environment=TWINREAD_ARGS") < unit.index("EnvironmentFile=-")
    assert unit.split("ExecStart=")[1].splitlines()[0].rstrip().endswith("$TWINREAD_ARGS")
    assert '[ -e "$DEFAULTS" ] ||' in install and "TWINREAD_ARGS=--standby" in install  # only if missing
    assert "make_demo_slides.py" in install and "demo_slides/manifest.json" in install and "crc_test_x.npy" in install
    assert "has_display" in (deploy / "kiosk.sh").read_text()  # no display, no browser: it costs CPU and heat for nobody
    conf = (deploy / "50-twinread-kiosk.conf").read_text()
    assert "autologin-user=@USER@" in conf and "autologin-user-timeout=0" in conf
    assert "autologin-session=twinread-kiosk" in conf
    assert "Exec=/usr/local/bin/twinread-kiosk" in (deploy / "twinread-kiosk.desktop").read_text()


# ---- the dashboard: real pauses, standby, demo slides, /control, and a dashboard that outlives every failure -----------
def _post(port, body, ctype="application/json"):
    """(HTTP status, parsed JSON) for a POST to /control."""
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/control", data=data, method="POST",
                                 headers={"Content-Type": ctype} if ctype else {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(port, path):
    return urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5).read()


@pytest.fixture
def serve():
    """Serve an App on a free port. Closes the app and the server afterwards."""
    import threading
    import uno_q_app
    started = []

    def start(app):
        port = _free_port()
        srv = uno_q_app.make_server(app, port)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        started.append((app, srv))
        return port
    yield start
    for app, srv in started:
        app.close()
        srv.shutdown()
        srv.server_close()


@pytest.fixture
def deck_dir(tmp_path):
    """A real deck written by make_demo_slides: 9 slides of random texture, one per class."""
    mds = _make_demo_slides()
    y = np.repeat(np.arange(len(mds.CLASSES)), 3)
    x = _toy_tiles(len(y))
    mds.write_deck(x, y, tmp_path / "deck", per_class=1, seed=7)
    return tmp_path / "deck"


def _toy_tiles(n):
    return np.random.default_rng(5).integers(0, 255, (n, 224, 224, 3), dtype=np.uint8)


def _make_demo_slides():
    sys.path.insert(0, str(ROOT / "scripts" / "uno_q"))
    import make_demo_slides
    return make_demo_slides


def _count_reads(app):
    """Wrap the model so a test can count real inferences."""
    calls = []
    infer = app.model.infer

    def counted(x):
        calls.append(1)
        return infer(x)
    app.model.infer = counted
    return calls


def _standby_app(bundle, source=None, **kw):
    import uno_q_app
    return uno_q_app.App(bundle, mode="camera", threads=1, camera_loss="wait", source=source, standby=True, **kw)


# -- EdgeModel load / unload
def test_unload_frees_the_session_and_reload_restores_it(bundle):
    import os
    import weakref
    m = rt.EdgeModel(bundle, threads=3)
    x = np.random.default_rng(1).integers(0, 255, (1, 224, 224, 3), dtype=np.uint8)
    before = m.infer(x)[0]
    gone = weakref.ref(m.sess)
    threads_loaded = len(os.listdir("/proc/self/task"))
    m.unload()
    assert m.sess is None and not m.loaded and gone() is None  # nothing keeps the session alive
    assert len(os.listdir("/proc/self/task")) < threads_loaded  # its thread pool stopped with it
    assert m.classes == CLASSES and m.thr == THR and m.cfg["classes"] == CLASSES and m.probes  # config stays available
    with pytest.raises(rt.ModelNotLoaded):
        m.infer(x)
    m.load()
    assert m.loaded and m.infer(x)[0].probsF == before.probsF  # the reloaded model is the same model
    sess = m.sess
    m.load()
    assert m.sess is sess  # loading twice keeps the session


def test_load_false_reads_the_config_and_opens_no_session(bundle):
    m = rt.EdgeModel(bundle, threads=1, load=False)
    assert not m.loaded and m.classes == CLASSES and m.thr == THR
    m.unload()  # harmless when already unloaded
    with pytest.raises(rt.ModelNotLoaded):
        m.infer(np.zeros((1, 224, 224, 3), np.uint8))


def test_reload_verifies_the_bundle_again(bundle, tmp_path):
    d = tmp_path / "b"
    shutil.copytree(bundle, d)
    m = rt.EdgeModel(d, threads=1, load=False)
    with open(d / "model.onnx", "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError):
        m.load()
    assert not m.loaded


# -- CameraSource pause / resume
def test_camera_pause_closes_the_device_and_resume_reopens_it():
    first, second = FakeCap(), FakeCap()
    src = _source([first, second]).start()
    try:
        assert _until(lambda: src.latest()[1] >= 3)
        assert src.status()["open"] and not src.status()["paused"]
        assert src.pause() is True  # on return the device really is closed
        assert first.released and src._cap is None
        st = src.status()
        assert not st["open"] and st["paused"] and not st["connected"] and st["reason"] == "camera paused"
        assert src.latest()[0] is None  # no stale frame can be taken for a current one after a resume
        grabs = first.grabs
        time.sleep(0.05)
        assert first.grabs == grabs and src._thread.is_alive()  # the thread sleeps: no polling of a closed device
        assert src._parked.is_set()
        src.resume()
        assert src.status()["opening"] or src.status()["open"]
        assert _until(lambda: src.status()["open"] and src.status()["connected"])
        assert not src.status()["paused"] and not src.status()["opening"]
        assert _until(lambda: src.latest()[0] is not None) and second.grabs > 0 and not second.released
    finally:
        src.stop()
    assert second.released


def test_camera_paused_before_start_never_opens():
    opened = []
    import camera
    src = camera.CameraSource(camera.AUTO, finder=lambda: [{"path": "/dev/video2", "name": "fake"}],
                              opener=lambda *a: opened.append(a) or FakeCap(), retry_s=0.01)
    assert src.pause() is True
    src.start()
    assert _until(lambda: src._parked.is_set())
    assert opened == [] and not src.status()["open"] and not src.status()["opening"]
    src.stop()


def test_camera_stop_ends_a_paused_thread():
    src = _source([FakeCap()]).start()
    assert _until(lambda: src.status()["open"])
    src.pause()
    thread = src._thread
    src.stop()
    assert not thread.is_alive()


def test_camera_keeps_trying_after_an_unexpected_failure():
    """A failure inside the capture thread becomes a reason on the page, not a dead thread."""
    calls = []

    def finder():
        calls.append(1)
        raise RuntimeError("video4linux is broken")
    src = _source([], finder=finder).start()
    try:
        assert _until(lambda: len(calls) >= 3 and "video4linux is broken" in (src.status()["reason"] or ""))
        assert src._thread.is_alive() and not src.status()["opening"]
    finally:
        src.stop()


# -- pausing through the app
def test_pause_model_unloads_it_and_shows_no_decision(bundle):
    import uno_q_app
    app = uno_q_app.App(bundle, mode="sim", threads=1)
    app.step()
    assert app.state()["call"] is not None and app.state()["controls"]["model_loaded"]
    st = app.control({"model_paused": True})
    assert st["status"] == "model_paused" and st["call"] is None and "reasons" not in st and "latency_ms" not in st
    assert app.model.sess is None and st["controls"]["model_loaded"] is False
    assert st["last"]["label"] in CLASSES  # the last decision is only ever mentioned as last
    reads = app.reads
    app.step()
    st = app.state()
    assert st["call"] is None and app.reads == reads and st["status"] == "model_paused"
    st = app.control({"model_paused": False})
    assert app.model.loaded and st["status"] == "reading" and st["call"] is None
    app.step()
    assert app.state()["call"] is not None and app.reads == reads + 1


def test_a_read_in_progress_cannot_publish_after_a_pause(bundle, monkeypatch):
    """A command that arrives during a read wins: the finished read is dropped, not shown."""
    import uno_q_app
    app = uno_q_app.App(bundle, mode="sim", threads=1)
    infer = app.model.infer

    def pause_during_read(x):
        r = infer(x)
        app.control({"model_paused": True})
        return r
    monkeypatch.setattr(app.model, "infer", pause_during_read)
    app.step()
    assert app.state()["status"] == "model_paused" and app.state()["call"] is None and app.reads == 0


def test_step_survives_being_paused_between_check_and_read(bundle):
    import uno_q_app
    app = uno_q_app.App(bundle, mode="sim", threads=1)
    app.model.unload()  # as if the pause landed right after the loop looked
    app.step()
    assert app.state()["status"] != "error"


def test_pause_camera_holds_the_frame_closes_the_device_and_reads_it_once(bundle):
    import uno_q_app
    first, second = FakeCap(), FakeCap()
    app = uno_q_app.App(bundle, mode="camera", threads=1, camera_loss="wait", source=_source([first, second]))
    calls = _count_reads(app)
    try:
        assert _until(lambda: app.cam.latest()[0] is not None)
        app.step()
        assert app.state()["call"] is not None and app.state()["frame_label"] == "Live"
        st = app.control({"feed_paused": True})
        assert first.released and st["controls"]["camera_open"] is False  # really closed
        assert st["frame_label"] == "Frozen frame" and st["has_picture"] and st["call"] is None  # read once more, below
        assert app.frame_jpg()[:2] == b"\xff\xd8"
        n = len(calls)
        for _ in range(6):
            app.step()
        assert len(calls) == n + 1  # a held frame is read once, not repeatedly
        st = app.state()
        assert st["call"] is not None and st["frame_label"] == "Frozen frame"
        held_jpg = app.frame_jpg()
        time.sleep(0.05)
        assert app.frame_jpg() == held_jpg  # stays on screen
        assert uno_q_app.IDLE_TICK_S == app._next_wait()  # nothing is due while a frame is held

        st = app.control({"feed_paused": False})
        assert st["call"] is None and not st["has_picture"]  # the old frame must not pass as live while it reopens
        assert _until(lambda: app.cam.status()["open"]) and _until(lambda: app.cam.latest()[0] is not None)
        app.step()
        assert app.state()["call"] is not None and app.state()["frame_label"] == "Live"
        assert _until(lambda: app.frame_jpg()[:2] == b"\xff\xd8")
    finally:
        app.close()


def test_preview_thread_sleeps_while_the_camera_is_closed(bundle):
    import uno_q_app
    app = uno_q_app.App(bundle, mode="camera", threads=1, camera_loss="wait", source=_source([FakeCap(), FakeCap()]))
    try:
        assert _until(lambda: app.cam.latest()[0] is not None)
        assert app._preview_on.is_set()
        app.control({"feed_paused": True})
        assert not app._preview_on.is_set()
        seq = app._jpg_seq
        time.sleep(0.05)
        assert app._jpg_seq == seq  # the frozen frame is not replaced by a late encode
    finally:
        app.close()


def test_both_paused_is_standby_with_the_last_frame_on_screen(bundle):
    import uno_q_app
    app = uno_q_app.App(bundle, mode="camera", threads=1, camera_loss="wait", source=_source([FakeCap(), FakeCap()]))
    try:
        assert _until(lambda: app.cam.latest()[0] is not None)
        app.step()
        st = app.control({"feed_paused": True, "model_paused": True})
        assert st["status"] == "standby" and st["call"] is None and st["has_picture"]
        assert st["controls"]["model_loaded"] is False and st["controls"]["camera_open"] is False
    finally:
        app.close()


# -- standby
def test_standby_starts_with_nothing_loaded_and_no_camera_open(bundle):
    import threading
    import uno_q_app
    opened = []
    import camera
    src = camera.CameraSource(camera.AUTO, finder=lambda: [{"path": "/dev/video2", "name": "fake"}],
                              opener=lambda *a: opened.append(a) or FakeCap(), retry_s=0.01)
    app = _standby_app(bundle, source=src)
    try:
        st = app.state()
        assert st["status"] == "standby" and st["call"] is None and st["model_error"] is None
        assert app.model.sess is None and st["controls"]["model_loaded"] is False
        assert app.cam is None and st["controls"]["camera_open"] is False and st["camera"]["open"] is False
        assert st["controls"]["model_paused"] and st["controls"]["feed_paused"]
        assert not st["has_picture"] and st["frame_label"] == "Camera off" and st["heads"] and st["classes"] == CLASSES
        assert opened == [] and not {"preview", "camera"} & {t.name for t in threading.enumerate()}
    finally:
        app.close()


def test_standby_loop_does_no_busy_work(bundle, monkeypatch):
    """The whole point of standby: the loop sleeps, and a board nobody is viewing does no work for the page."""
    import threading
    import uno_q_app
    app = _standby_app(bundle)
    steps, vitals = [], []
    real_step, real_vitals = app.step, uno_q_app.system_vitals
    monkeypatch.setattr(app, "step", lambda: (steps.append(1), real_step())[1])
    monkeypatch.setattr(uno_q_app, "system_vitals", lambda: (vitals.append(1), real_vitals())[1])
    stop = threading.Event()
    t = threading.Thread(target=app.run_forever, args=(stop,), daemon=True)
    t.start()
    time.sleep(0.4)
    stop.set()
    app.close()
    t.join(2)
    assert not t.is_alive() and len(steps) == 1 and vitals == []
    assert app.state()["status"] == "standby"


def test_each_button_starts_its_own_thing_from_standby_and_turns_it_off_again(bundle):
    app = _standby_app(bundle, source=_source([FakeCap(), FakeCap()]))
    try:
        st = app.control({"model_paused": False})  # model only: loaded, but nothing to read and the camera stays closed
        assert st["status"] == "camera_off" and st["controls"]["model_loaded"] and not st["controls"]["camera_open"]
        assert st["controls"]["model_started"] and not st["controls"]["camera_started"] and app.cam is None
        app.step()
        assert app.state()["status"] == "camera_off" and app.reads == 0

        st = app.control({"model_paused": True})  # the same button turns it off again
        assert st["status"] == "standby" and not st["controls"]["model_loaded"]

        st = app.control({"feed_paused": False})  # camera only: open, no model
        assert st["controls"]["camera_started"] and not st["controls"]["model_loaded"] and st["status"] in ("camera_starting", "model_paused")
        assert _until(lambda: app.cam.status()["open"]) and _until(lambda: app.cam.latest()[0] is not None)
        app.step()
        assert app.state()["status"] == "model_paused" and app.state()["call"] is None and app.reads == 0

        st = app.control({"feed_paused": True})
        assert st["status"] == "standby" and not st["controls"]["camera_open"]
    finally:
        app.close()


def test_start_reading_opens_the_camera_and_loads_the_model_with_one_command(bundle, serve):
    app = _standby_app(bundle, source=_source([FakeCap(), FakeCap()]))
    port = serve(app)
    code, st = _post(port, {"feed_paused": False, "model_paused": False})  # what the one button sends
    assert code == 200 and st["controls"]["model_loaded"] and st["controls"]["camera_started"] and st["controls"]["model_started"]
    assert not st["controls"]["model_paused"] and not st["controls"]["feed_paused"]
    assert _until(lambda: app.cam.status()["open"]) and _until(lambda: app.cam.latest()[0] is not None)
    app.step()
    assert app.state()["call"] is not None and app.reads == 1


def test_standby_offers_one_start_button_and_the_bar_keeps_its_two():
    ui = ROOT / "device" / "ui"
    html, js = (ui / "index.html").read_text(), (ui / "app.js").read_text()
    assert 'id="start-reading" class="btn primary"' in html and ">Start reading<" in html and ">Run demo slides<" in html
    assert 'on("start-reading", () => send({feed_paused: false, model_paused: false}' in js  # one POST /control
    for gone in ("start-feed", "start-model"):
        assert gone not in html + js, gone
    assert 'id="feed"' in html and 'id="model"' in html and 'on("feed", toggleFeed)' in js and 'on("model", toggleModel)' in js
    assert '"Start camera"' in js and '"Start model"' in js  # the bar's own labels stay


def test_main_starts_in_standby_end_to_end(bundle):
    """The real entry point, as a process: with --standby the page is up and nothing is on."""
    import subprocess
    port = _free_port()
    p = subprocess.Popen([sys.executable, str(ROOT / "device" / "uno_q_app.py"), "--bundle", str(bundle), "--mode", "camera",
                          "--device", "/dev/video_does_not_exist", "--camera-loss", "wait", "--threads", "1", "--port", str(port),
                          "--standby"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        assert _until(lambda: _answers(port), 30.0)
        st = json.loads(_get(port, "/state"))
        assert st["status"] == "standby" and not st["controls"]["model_loaded"] and not st["controls"]["camera_open"]
        assert b"TwinRead" in _get(port, "/")
    finally:
        p.terminate()
        p.wait(10)
    assert p.returncode in (0, -15)


def _answers(port):
    try:
        _get(port, "/state")
        return True
    except OSError:
        return False


# -- the dashboard outlives every failure: the page stays up and says what failed
def _dashboard_is_up(port):
    assert b"TwinRead" in _get(port, "/")
    return json.loads(_get(port, "/state"))


@pytest.mark.parametrize("standby", [True, False])
def test_missing_bundle_never_takes_the_dashboard_down(tmp_path, serve, standby):
    import uno_q_app
    app = uno_q_app.App(tmp_path / "no_such_bundle", mode="camera", threads=1, camera_loss="wait",
                        source=_source([], finder=lambda: []), standby=standby)
    port = serve(app)
    st = _dashboard_is_up(port)
    assert st["status"] == "model_error" and "did not load" in st["model_error"] and st["controls"]["model_paused"]
    assert not st["controls"]["model_loaded"] and st["classes"] == [] and st["heads"]
    app.step()
    code, body = _post(port, {"model_paused": False})
    assert code == 400 and "failed checksum" in body["error"]
    assert _dashboard_is_up(port)["status"] == "model_error"


def test_bundle_damaged_after_the_start_is_reported_and_recoverable(bundle, tmp_path, serve):
    d = tmp_path / "b"
    shutil.copytree(bundle, d)
    app = _standby_app(d, source=_source([], finder=lambda: []))
    port = serve(app)
    good = (d / "model.onnx").read_bytes()
    (d / "model.onnx").write_bytes(good + b"x")
    code, body = _post(port, {"model_paused": False})
    assert code == 400 and "did not load" in body["error"] and "failed checksum" in body["error"]
    st = _dashboard_is_up(port)
    assert st["status"] == "model_error" and st["model_error"] == body["error"] and not st["controls"]["model_loaded"]
    (d / "model.onnx").write_bytes(good)
    code, st = _post(port, {"model_paused": False})
    assert code == 200 and st["model_error"] is None and st["controls"]["model_loaded"]


@pytest.mark.parametrize("standby", [True, False])
def test_a_model_that_cannot_load_never_takes_the_dashboard_down(bundle, deck_dir, monkeypatch, serve, standby):
    import uno_q_app

    def no_memory(self):
        raise RuntimeError("cannot allocate memory")
    monkeypatch.setattr(rt.EdgeModel, "load", no_memory)
    app = uno_q_app.App(bundle, mode="sim", threads=1, standby=standby, demo_dir=deck_dir)
    port = serve(app)
    st = _dashboard_is_up(port)
    if standby:
        assert st["status"] == "standby"  # nothing has tried to load yet
    else:
        assert st["status"] == "model_error" and "cannot allocate memory" in st["model_error"]
    code, body = _post(port, {"model_paused": False, "source": "demo"})  # refused whole: nothing half applied either
    assert code == 400 and "cannot allocate memory" in body["error"]
    st = _dashboard_is_up(port)
    assert st["mode"] == "sim" and st["controls"]["model_paused"] and "cannot allocate memory" in st["model_error"]
    app.step()
    assert app.state()["call"] is None


def test_a_camera_that_cannot_open_never_takes_the_dashboard_down(bundle, serve):
    import camera

    def broken(*a):
        raise OSError("device is busy")
    src = camera.CameraSource(camera.AUTO, finder=lambda: [{"path": "/dev/video2", "name": "fake"}], opener=broken, retry_s=0.01)
    app = _standby_app(bundle, source=src)
    port = serve(app)
    code, st = _post(port, {"feed_paused": False})
    assert code == 200 and st["controls"]["camera_started"]
    assert _until(lambda: "device is busy" in (src.status()["reason"] or "") and not src.status()["opening"])
    app.step()
    st = _dashboard_is_up(port)
    assert st["status"] == "no_camera" and "device is busy" in st["camera"]["reason"] and not st["controls"]["camera_open"]
    code, st = _post(port, {"feed_paused": True})  # the same button stops it looking
    assert code == 200 and st["status"] == "standby"


def test_a_failed_read_says_so_and_the_next_one_recovers(bundle, monkeypatch, serve):
    import uno_q_app
    app = uno_q_app.App(bundle, mode="sim", threads=1)
    port = serve(app)
    real = app.model.infer

    def boom(x):
        raise RuntimeError("the read failed")
    monkeypatch.setattr(app.model, "infer", boom)
    app.step()
    st = _dashboard_is_up(port)
    assert st["status"] == "error" and st["error"] == "the read failed" and st["call"] is None
    monkeypatch.setattr(app.model, "infer", real)
    app.step()
    assert _dashboard_is_up(port)["call"] is not None


def test_a_frame_arriving_mid_step_is_not_an_error(bundle):
    """_picture() found nothing, then a frame arrived before _idle_status() looked: wait for the next step, say no error."""
    import uno_q_app
    app = uno_q_app.App(bundle, mode="camera", threads=1, camera_loss="wait", source=_source([FakeCap()]))
    try:
        assert _until(lambda: app.cam.latest()[0] is not None)
        app._picture = lambda: (None, None)
        app.step()
        assert app.state()["status"] != "error" and app.reads == 0
    finally:
        app.close()


def test_an_unexpected_command_failure_is_reported_and_the_dashboard_stays_up(bundle, deck_dir, monkeypatch, serve):
    app = _demo_app(bundle, deck_dir)
    port = serve(app)

    def boom(*a):
        raise RuntimeError("cannot start new thread")
    monkeypatch.setattr(app, "_sync_camera", boom)
    code, body = _post(port, {"feed_paused": False})
    assert code == 500 and "cannot start new thread" in body["error"]
    monkeypatch.undo()
    assert _dashboard_is_up(port)["status"] == "standby"


def test_mode_demo_without_slides_stays_up_on_a_closed_camera(bundle, tmp_path, serve):
    import uno_q_app
    app = uno_q_app.App(bundle, mode="demo", threads=1, camera_loss="wait", demo_dir=tmp_path / "none")
    port = serve(app)
    st = _dashboard_is_up(port)
    assert st["mode"] == "camera" and st["controls"]["feed_paused"] and "make_demo_slides.py" in st["mode_reason"]
    assert not st["controls"]["camera_open"]


def test_unit_restarts_a_crashed_app_and_nothing_can_stop_it_for_good():
    unit = (ROOT / "device" / "deploy" / "twinread.service").read_text()
    assert "Restart=always" in unit and "RestartSec=2" in unit and "StartLimitIntervalSec=0" in unit


# -- demo slides
def _demo_app(bundle, deck_dir, **kw):
    import uno_q_app
    return uno_q_app.App(bundle, mode="camera", threads=1, camera_loss="wait", source=_source([FakeCap(), FakeCap()]),
                         demo_dir=deck_dir, standby=True, **kw)


def test_demo_reads_each_slide_once_auto_advances_and_wraps(bundle, deck_dir):
    app = _demo_app(bundle, deck_dir)
    try:
        st = app.control({"source": "demo", "model_paused": False, "feed_paused": False})
        n = st["demo"]["total"]
        assert n == 9 and st["demo"]["running"] and st["mode"] == "demo" and st["frame_label"] == "Demo slide"
        assert st["controls"]["camera_open"] is False and app.frame_jpg()[:2] == b"\xff\xd8"
        calls = _count_reads(app)
        for _ in range(5):
            app.step()
        assert len(calls) == 1 and app.state()["demo"]["index"] == 0  # the slide stays until its time, and is read once
        for want in list(range(1, n)) + [0]:  # round the deck, and back to the first
            app._demo_next = 0.0
            app.step()
            st = app.state()
            assert st["demo"]["index"] == want and st["call"] is not None
        assert len(calls) == n + 1
    finally:
        app.close()


def test_demo_hold_stops_the_slides_and_keeps_the_decision(bundle, deck_dir):
    import uno_q_app
    app = _demo_app(bundle, deck_dir)
    try:
        app.control({"source": "demo", "model_paused": False, "feed_paused": False})
        app.step()
        call = app.state()["call"]
        calls = _count_reads(app)
        st = app.control({"feed_paused": True})
        assert st["call"] == call and not st["demo"]["running"]  # the picture did not change, so its decision still stands
        app._demo_next = 0.0
        app.step()
        app.step()
        assert app.state()["demo"]["index"] == 0 and calls == [] and app._next_wait() == uno_q_app.IDLE_TICK_S
        st = app.control({"feed_paused": False})
        assert st["demo"]["running"] and st["call"] == call and st["demo"]["index"] == 0
        assert 0 < app._next_wait() <= st["demo"]["interval"]
    finally:
        app.close()


def test_demo_jump_and_step_with_wrap_around(bundle, deck_dir):
    app = _demo_app(bundle, deck_dir)
    try:
        app.control({"source": "demo", "model_paused": False, "feed_paused": True})
        app.step()
        st = app.control({"demo_index": 4})
        assert st["demo"]["index"] == 4 and st["call"] is None  # no decision from the slide before
        app.step()
        assert app.state()["call"] is not None and app.state()["demo"]["index"] == 4
        assert app.control({"demo_index": 0})["demo"]["index"] == 0
        assert app.control({"demo_step": -1})["demo"]["index"] == 8  # wraps backwards
        assert app.control({"demo_step": 1})["demo"]["index"] == 0   # and forwards
    finally:
        app.close()


def test_demo_shows_the_true_label_and_whether_it_matches(bundle, deck_dir):
    app = _demo_app(bundle, deck_dir)
    try:
        st = app.control({"source": "demo", "model_paused": False, "feed_paused": True})
        assert st["demo"]["truth"] == app.deck.slides[0]["truth"] and st["demo"]["truths"][0] == st["demo"]["truth"]
        app.step()
        label = app.state()["call"]["label"]
        app.deck.slides[1]["truth"] = label
        app.deck.slides[2]["truth"] = "NOT" + label
        app.control({"demo_index": 1})
        app.step()
        st = app.state()
        assert st["truth"] == st["call"]["label"] and st["truth_match"] is True
        app.control({"demo_index": 2})
        app.step()
        st = app.state()
        assert st["truth"] == "NOT" + st["call"]["label"] and st["truth_match"] is False
    finally:
        app.close()


def test_demo_with_the_model_paused_still_moves_and_decides_nothing(bundle, deck_dir):
    app = _demo_app(bundle, deck_dir)
    try:
        st = app.control({"source": "demo"})
        assert st["status"] == "model_paused" and st["demo"]["running"] and st["call"] is None
        assert "truth_match" not in st and st["demo"]["truth"]  # the true label is known; there is no call to compare it to
        app._demo_next = 0.0
        app.step()
        st = app.state()
        assert st["demo"]["index"] == 1 and st["call"] is None and st["status"] == "model_paused" and app.reads == 0
    finally:
        app.close()


def test_demo_closes_the_camera_and_going_back_reopens_it(bundle, deck_dir):
    first, second = FakeCap(), FakeCap()
    import uno_q_app
    app = uno_q_app.App(bundle, mode="camera", threads=1, camera_loss="wait", source=_source([first, second]), demo_dir=deck_dir)
    try:
        assert _until(lambda: app.cam.latest()[0] is not None)
        st = app.control({"source": "demo"})
        assert first.released and st["controls"]["camera_open"] is False and st["mode"] == "demo"
        st = app.control({"source": "camera"})
        assert st["mode"] == "camera" and st["controls"]["feed_paused"] is False and not st["has_picture"]
        assert _until(lambda: app.cam.status()["open"]) and second.grabs >= 0
        assert _until(lambda: app.cam.latest()[0] is not None)
        app.step()
        assert app.state()["frame_label"] == "Live" and "truth" not in app.state()
    finally:
        app.close()


def test_missing_deck_is_refused_with_a_clear_reason_and_changes_nothing(bundle, tmp_path, serve):
    app = _standby_app(bundle, source=_source([], finder=lambda: []), demo_dir=tmp_path / "none")
    port = serve(app)
    before = _dashboard_is_up(port)
    assert before["controls"]["demo_available"] is False and "make_demo_slides.py" in before["controls"]["demo_error"]
    code, body = _post(port, {"source": "demo", "model_paused": False})
    assert code == 400 and "No demo slides" in body["error"] and "make_demo_slides.py" in body["error"]
    st = _dashboard_is_up(port)
    assert st["mode"] == "camera" and st["controls"]["model_paused"] and not st["controls"]["model_loaded"]
    assert _post(port, {"demo_step": 1})[0] == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(port, "/slide/0.jpg")
    assert e.value.code == 404


def test_slides_built_after_the_start_are_picked_up(bundle, tmp_path, deck_dir):
    target = tmp_path / "later"
    app = _standby_app(bundle, source=_source([], finder=lambda: []), demo_dir=target)
    try:
        with pytest.raises(ValueError):
            app.control({"source": "demo"})
        shutil.copytree(deck_dir, target)
        assert app.control({"source": "demo"})["demo"]["total"] == 9
    finally:
        app.close()


@pytest.mark.parametrize("manifest", ["not json", "[1, 2]", '{"slides": []}', '{"slides": [{"file": "gone.png", "truth": "TUM"}]}'])
def test_unusable_deck_gives_a_reason(tmp_path, manifest):
    import demo
    (tmp_path / "manifest.json").write_text(manifest)
    deck, why = demo.load_deck(tmp_path)
    assert deck is None and "unusable" in why and "make_demo_slides.py" in why


def test_thumbnails_are_served(bundle, deck_dir, serve):
    import cv2
    app = _demo_app(bundle, deck_dir)
    port = serve(app)
    jpg = _get(port, "/slide/8.jpg")
    assert jpg[:2] == b"\xff\xd8"
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[0] == img.shape[1] and img.shape[0] >= 64
    for bad in ("/slide/9.jpg", "/slide/x.jpg", "/slide/-1.jpg", "/slide/99999.jpg"):
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(port, bad)
        assert e.value.code == 404, bad


def test_the_model_reads_the_original_slide_not_the_enlarged_preview(bundle, deck_dir):
    app = _demo_app(bundle, deck_dir)
    try:
        app.control({"source": "demo", "model_paused": False, "feed_paused": True})
        seen = []
        infer = app.model.infer
        app.model.infer = lambda x: (seen.append(x.copy()), infer(x))[1]
        app.step()
        assert (seen[0][0] == app.deck.slides[0]["bgr"][..., ::-1]).all()
    finally:
        app.close()


# -- make_demo_slides
def test_make_demo_slides_is_deterministic_and_keeps_labels_and_colours(tmp_path):
    import cv2
    mds = _make_demo_slides()
    y = np.repeat(np.arange(len(mds.CLASSES)), 5)
    x = _toy_tiles(len(y))
    a = mds.write_deck(x, y, tmp_path / "a", per_class=2, seed=7)
    b = mds.write_deck(x, y, tmp_path / "b", per_class=2, seed=7)
    assert a == b and len(a["slides"]) == 2 * len(mds.CLASSES)
    for s in a["slides"]:
        assert (tmp_path / "a" / s["file"]).read_bytes() == (tmp_path / "b" / s["file"]).read_bytes()
        assert s["truth"] == mds.CLASSES[y[s["tile"]]]  # the label is the tile's own
        assert (cv2.imread(str(tmp_path / "a" / s["file"]))[..., ::-1] == x[s["tile"]]).all()  # lossless, and RGB stays RGB
    assert sorted(s["truth"] for s in a["slides"]) == sorted(list(mds.CLASSES) * 2)  # every class, the same number of times
    assert mds.write_deck(x, y, tmp_path / "c", 2, seed=8)["slides"] != a["slides"]
    mds.write_deck(x, y, tmp_path / "a", per_class=1, seed=7)
    assert len(list((tmp_path / "a").glob("*.png"))) == len(mds.CLASSES)  # nothing left over from a bigger deck


def test_make_demo_slides_cannot_see_model_output():
    """Selection is random with a fixed seed: the script reads only the labels and never touches a model."""
    import ast
    src = (ROOT / "scripts" / "uno_q" / "make_demo_slides.py").read_text()
    imported = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split(".")[0])
    assert imported <= {"__future__", "argparse", "json", "pathlib", "cv2", "numpy"}, imported
    mds = _make_demo_slides()
    y = np.repeat(np.arange(len(mds.CLASSES)), 5)
    assert mds.pick(y, 2, 7) == mds.pick(y.copy(), 2, 7)  # pick() takes the labels and a seed, and nothing else


def test_demo_classes_are_the_bundles_classes():
    mds = _make_demo_slides()
    cfg = ROOT / "bundles" / "cm" / "config.json"
    if not cfg.exists():
        pytest.skip("bundles/cm not present")
    assert list(mds.CLASSES) == json.loads(cfg.read_text())["classes"]


# -- /control
def test_control_requires_json_and_validates_everything(bundle, deck_dir, serve):
    app = _demo_app(bundle, deck_dir)
    port = serve(app)
    assert _post(port, {"model_paused": True}, ctype="text/plain")[0] == 415
    assert _post(port, {"model_paused": True}, ctype=None)[0] == 415
    assert _post(port, b"not json")[0] == 400
    assert _post(port, b"[1]")[0] == 400
    assert _post(port, {})[0] == 400
    assert _post(port, b"x" * 2000)[0] == 400
    for bad in ({"bogus": 1}, {"model_paused": "yes"}, {"model_paused": 1}, {"feed_paused": None}, {"source": "usb"}, {"source": 1},
                {"demo_step": 2}, {"demo_step": 1.0}, {"demo_step": True}, {"demo_step": "1"}, {"demo_index": 1.5}, {"demo_index": True},
                {"demo_interval": 0}, {"demo_interval": 61}, {"demo_interval": "4"}, {"demo_interval": True},
                {"demo_step": 1, "demo_index": 0}):
        code, body = _post(port, bad)
        assert code == 400 and body["error"], bad
    assert _post(port, b'{"demo_interval": NaN}')[0] == 400
    with pytest.raises(urllib.error.HTTPError) as e:  # any other path
        urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/other", data=b"{}", method="POST"), timeout=5)
    assert e.value.code == 404


def test_a_bad_command_is_not_half_applied(bundle, deck_dir, serve):
    app = _demo_app(bundle, deck_dir)
    port = serve(app)
    before = app.control({"model_paused": False, "source": "demo", "demo_interval": 8})
    assert before["demo"]["interval"] == 8
    for bad in ({"model_paused": True, "bogus": 1}, {"model_paused": True, "demo_step": 5}, {"feed_paused": True, "demo_index": 99},
                {"model_paused": True, "demo_interval": 99}, {"source": "camera", "demo_step": 1}):
        assert _post(port, bad)[0] == 400, bad
        st = app.state()
        assert st["controls"]["model_loaded"] and not st["controls"]["model_paused"], bad
        assert st["mode"] == "demo" and not st["controls"]["feed_paused"] and st["demo"]["index"] == 0, bad
        assert st["demo"]["interval"] == 8, bad


def test_control_changes_pace_without_dropping_the_decision(bundle, deck_dir):
    app = _demo_app(bundle, deck_dir)
    try:
        app.control({"source": "demo", "model_paused": False})
        app.step()
        call = app.state()["call"]
        st = app.control({"demo_interval": 2})
        assert st["demo"]["interval"] == 2.0 and st["call"] == call
    finally:
        app.close()


def test_repeating_a_command_changes_nothing(bundle, deck_dir):
    app = _demo_app(bundle, deck_dir)
    try:
        app.control({"source": "demo", "model_paused": False, "feed_paused": True})
        app.step()
        calls = _count_reads(app)
        call = app.state()["call"]
        st = app.control({"feed_paused": True, "model_paused": False, "source": "demo"})
        app.step()
        assert st["call"] == call and calls == []
    finally:
        app.close()


# -- the picture on the wire
def test_a_held_frame_paints_straight_away(bundle, deck_dir, serve):
    """Chromium paints a part only when the NEXT part's headers arrive, and a held picture has no next part for seconds. So
    the boundary and those headers are sent right after each image, not with the next one."""
    app = _demo_app(bundle, deck_dir)
    app.control({"source": "demo", "feed_paused": True})
    port = serve(app)
    jpg = app.frame_jpg()
    r = urllib.request.urlopen(f"http://127.0.0.1:{port}/stream.mjpg", timeout=5)
    t0 = time.monotonic()
    want = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    assert r.read(len(want)) == want
    assert time.monotonic() - t0 < 1.0  # well before the resend
    r.close()


# -- no disclaimers
DISCLAIMERS = ("research prototype", "diagnostic", "not a device", "less independent", "not independent", "one student network",
               "heads_note", "banner")


def test_no_disclaimer_remains_in_the_ui_or_state(bundle, deck_dir, tmp_path):
    import uno_q_app
    ui = ROOT / "device" / "ui"
    text = "".join((ui / n).read_text() for n in ("index.html", "app.js", "style.css")).lower()
    for d in DISCLAIMERS:
        assert d not in text, d
    apps = [uno_q_app.App(bundle, mode="sim", threads=1, demo_dir=deck_dir), _standby_app(bundle, demo_dir=deck_dir),
            uno_q_app.App(tmp_path / "gone", mode="sim", threads=1)]
    for app in apps:
        app.step()
        blob = json.dumps(app.state()).lower()
        for d in DISCLAIMERS:
            assert d not in blob, d
        app.close()
    assert "diagnostic" not in (ROOT / "device" / "deploy" / "twinread.service").read_text()


def test_source_labels_and_the_off_board_host_label_stay(bundle):
    import uno_q_app
    app = uno_q_app.App(bundle, mode="sim", threads=1)
    assert app.state()["frame_label"] == "Simulated frame"
    app.host = {"machine": "aarch64", "model": "Arduino UnoQ", "is_uno_q": True, "hostname": "uno-q"}
    assert app.state()["host_label"] == ""  # not shown on the board
    app.host = {"machine": "x86_64", "model": "", "is_uno_q": False, "hostname": "pc"}
    assert "not the UNO Q" in app.state()["host_label"]


def test_ui_style_rules_a_unit_test_can_see():
    """A real browser is the real check; these catch regressions the rendering pass would have to find by eye."""
    ui = ROOT / "device" / "ui"
    js, css, html = (ui / "app.js").read_text(), (ui / "style.css").read_text(), (ui / "index.html").read_text()
    assert re.search(r"^\(\(\) => \{", js, re.M) and js.rstrip().endswith("})();")  # no top-level names to shadow window.top
    assert not re.search(r"^(const|let|var|function) ", js, re.M)
    assert 'location.hostname === "127.0.0.1"' in js and "body.kiosk{cursor:none}" in css
    assert "AbortSignal.timeout" in js  # a stalled network must not leave an old decision on screen looking current
    assert "system-ui" in css and "@font-face" not in css and "blur(" not in css and "box-shadow" not in css
    assert "→" not in js + html and "·" not in js + html
    for colour in ("#f5f6f8", "#1b1f24", "#5b6572", "#1c7c3f", "#b4231a", "#2f6fdb", "#8a4fd6", "#d7dbe0", "#6b5200",  # the committed look
                   "#14171b", "#e8eaed", "#1d2126", "#9aa3ad", "#4cc17a", "#ff6b60", "#6aa0ff", "#b58cf5", "#343a42"):  # and its dark values
        assert colour in css, colour
    assert "prefers-color-scheme:dark" in css
    assert "aria-pressed" in html and ":focus-visible" in css
    assert html.count('href="') == 2 and 'rel="icon" href="data:,"' in html  # no request for a favicon


def test_ui_status_words_are_display_only():
    """The page says Confident, Needs review and No tissue. The codes underneath stay: /state, the LED and the prompt."""
    ui = ROOT / "device" / "ui"
    js, html = (ui / "app.js").read_text(), (ui / "index.html").read_text()
    for word in ("CONFIDENT", "NEEDS REVIEW", "NO TISSUE", 'agree: "Confident"', 'review: "Needs review"', 'no_tissue: "No tissue"', "Suggests "):
        assert word in js, word
    # "agree" appears only as the status code, never in anything shown.
    strings = re.findall(r'"[^"\n]*"|`[^`\n]*`', js) + [re.sub(r"<[^>]+>", " ", html)]
    shown = [t for t in strings if re.search(r"agree", t, re.I) and t not in ('"agree"', '"disagree"')]
    assert shown == []
    assert "agree" in js  # the code, as a key


def test_ui_class_names_are_the_ones_the_explanation_uses():
    import explain_prompt as ep
    js = (ROOT / "device" / "ui" / "app.js").read_text()
    for code, name in ep.CLASS_NAMES.items():
        assert f'{code}: "{name}"' in js, code


def test_ui_thresholds_are_rounded_and_the_old_note_stays_gone():
    js, html = (ROOT / "device" / "ui" / "app.js").read_text(), (ROOT / "device" / "ui" / "index.html").read_text()
    assert "thr.js.toFixed(3)" in js and "thr.review.toFixed(3)" in js
    for gone in ("banner", "headsnote", "heads_note", "ocular", "Student heads"):
        assert gone not in js + html, gone


def test_kiosk_check_still_compiles():
    import py_compile
    py_compile.compile(str(ROOT / "scripts" / "uno_q" / "kiosk_check.py"), doraise=True, cfile=None)
