"""Screen-capture rig tests (CPU, offline). Simulated capture = local degradation helper, not K2's camera_ops."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rig"))

import marker  # noqa: E402
import match_captures  # noqa: E402
import show_tiles  # noqa: E402


def sim_capture(page, rng, severity=0.7, out_hw=(480, 640)):
    """Camera-filming-a-screen SIMULATION: perspective jitter, rotation, blur, gamma, noise, JPEG q~60."""
    h, w = page.shape[:2]
    oh, ow = out_hw
    s = min(ow * 0.94 / w, oh * 0.94 / h)
    pw, ph = w * s, h * s
    ox, oy = (ow - pw) / 2, (oh - ph) / 2
    dst = np.array([[ox, oy], [ox + pw, oy], [ox + pw, oy + ph], [ox, oy + ph]], np.float32)
    jit = 0.03 * severity / 0.7 * min(pw, ph)  # a few % corner jitter
    dst = dst + rng.uniform(-jit, jit, dst.shape).astype(np.float32)
    ang = np.deg2rad(rng.uniform(-3, 3) * severity / 0.7)
    c = dst.mean(0)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
    dst = ((dst - c) @ R.T + c).astype(np.float32)
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(page, M, (ow, oh), flags=cv2.INTER_AREA, borderValue=(40, 40, 40))
    img = cv2.GaussianBlur(img, (0, 0), 1.0 + 0.5 * severity / 0.7)
    g = rng.uniform(0.8, 1.2)
    img = (255.0 * (img / 255.0) ** g)
    img = img + rng.normal(0, 3.0, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def smooth_tile(seed, n=224):
    r = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    t = np.stack([120 + 80 * np.sin(3 * xx + r.rand() * 6) * np.cos(2 * yy),
                  90 + 60 * np.cos(2 * xx + yy * 4 + r.rand() * 6),
                  150 + 70 * np.sin(xx * 2 + yy * 3 + r.rand() * 6)], -1)
    return np.clip(t, 0, 255).astype(np.uint8)


def test_marker_roundtrip_ids():
    for i in range(4096):
        m = marker.encode_marker(i)
        assert m.dtype == np.uint8 and m.shape == (120, 120, 3)
        assert marker.decode_marker(m) == i
    assert marker.decode_marker(np.full((120, 120, 3), 255, np.uint8)) is None


def test_marker_parity_rejects_single_bit_flip():
    m = marker.encode_marker(1234)
    q = marker.QUIET_FRAC
    s = 120 - 2 * int(round(120 * q))
    cell = (s - 2 * (s // 12)) // 4
    b = s // 12
    o = int(round(120 * q)) + b
    flipped = m.copy()
    cur = flipped[o + 2, o + 2, 0]
    flipped[o:o + cell, o:o + cell] = 255 - cur
    assert marker.decode_marker(flipped) is None


def test_marker_decodes_after_simulated_capture():
    rng = np.random.RandomState(0)
    tiles = [smooth_tile(k) for k in range(5)]
    ok = wrong = 0
    ids = rng.choice(4096, 200, replace=False)
    for k, fid in enumerate(ids):
        page = show_tiles.render_page(tiles[k % 5], int(fid))
        got = marker.decode_marker(sim_capture(page, rng, 0.7))
        if got == fid:
            ok += 1
        elif got is not None:
            wrong += 1
    print("decode rate", ok / 200, "wrong", wrong)
    assert wrong == 0
    assert ok >= 190


def test_match_captures_crops_tile_region():
    tile = smooth_tile(7)
    page = show_tiles.render_page(tile, 321)
    fid, crop = match_captures.crop_tile(page)
    assert fid == 321 and crop.shape == (224, 224, 3)
    assert np.abs(crop.astype(int) - tile.astype(int)).mean() <= 4
    # also after a mild shrink into a 640x480 camera frame (no other degradation)
    small = cv2.resize(page, (600, 375), interpolation=cv2.INTER_AREA)
    frame = np.full((480, 640, 3), 40, np.uint8)
    frame[50:425, 20:620] = small
    fid, crop = match_captures.crop_tile(frame)
    assert fid == 321
    assert np.abs(crop.astype(int) - tile.astype(int)).mean() <= 4


def test_match_captures_cli_builds_set(tmp_path):
    rng = np.random.RandomState(1)
    fr = tmp_path / "frames"
    fr.mkdir()
    tiles = {}
    for fid in range(3):
        t = smooth_tile(fid)
        tiles[fid] = t
        cv2.imwrite(str(fr / f"f{fid}.jpg"), sim_capture(show_tiles.render_page(t, fid), rng, 0.3))
    cv2.imwrite(str(fr / "blank.jpg"), np.full((480, 640, 3), 128, np.uint8))
    sess = {str(i): i * 10 for i in range(3)}
    (tmp_path / "rig_session.json").write_text(json.dumps(sess))
    y = np.arange(100) % 9
    np.save(tmp_path / "y.npy", y)
    out = tmp_path / "capture"
    r = subprocess.run([sys.executable, str(ROOT / "rig" / "match_captures.py"), str(fr),
                        str(tmp_path / "rig_session.json"), "--labels", str(tmp_path / "y.npy"),
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    meta = json.loads((out / "meta.json").read_text())
    assert meta["n_frames"] == 4 and meta["n_matched"] == 3 and meta["n_dropped_no_marker"] == 1
    assert np.load(out / "x.npy").shape == (3, 224, 224, 3)
    assert list(np.load(out / "idx.npy")) == [0, 10, 20]
    assert list(np.load(out / "y.npy")) == [0, 1, 2]


def test_render_page_layout_documented_and_fixed():
    p = show_tiles.render_page(smooth_tile(0, 224), 5)
    assert p.shape == (marker.PAGE_H, marker.PAGE_W, 3)


def test_record_frames_no_camera_errors(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "rig" / "record_frames.py"), "--device", "/dev/video_nonexistent",
                        "--seconds", "1", "--out", str(tmp_path / "f")], capture_output=True, text=True)
    assert r.returncode != 0 and "camera" in (r.stderr + r.stdout).lower()
