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
from sim_capture import sim_capture  # noqa: E402


def smooth_tile(seed, n=224):
    r = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    t = np.stack([120 + 80 * np.sin(3 * xx + r.rand() * 6) * np.cos(2 * yy),
                  90 + 60 * np.cos(2 * xx + yy * 4 + r.rand() * 6),
                  150 + 70 * np.sin(xx * 2 + yy * 3 + r.rand() * 6)], -1)
    return np.clip(t, 0, 255).astype(np.uint8)


def test_marker_roundtrip_ids():
    for i in range(2048):
        m = marker.encode_marker(i)
        assert m.dtype == np.uint8 and m.shape == (120, 120, 3)
        assert marker.decode_marker(m) == i
    for v in (0, 255):  # blank / all-dark blobs are never valid markers
        assert marker.decode_marker(np.full((120, 120, 3), v, np.uint8)) is None


def test_hamming_min_distance_4():
    words = [marker._codeword(v) ^ marker.MASK for v in range(2048)]
    assert len(set(words)) == 2048
    assert min(bin(marker._codeword(v)).count("1") for v in range(1, 2048)) == 4  # linear code: min weight = min distance
    w = words[1234]
    for k in range(16):
        assert marker._decode_word(w ^ (1 << k)) is None
        for j in range(k + 1, 16):
            assert marker._decode_word(w ^ (1 << k) ^ (1 << j)) is None
            for l in range(j + 1, 16):
                assert marker._decode_word(w ^ (1 << k) ^ (1 << j) ^ (1 << l)) is None
    assert marker._decode_word(0) is None and marker._decode_word(0xFFFF) is None


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
    ids = rng.choice(2048, 200, replace=False)
    for k, fid in enumerate(ids):
        page = show_tiles.render_page(tiles[k % 5], int(fid), 77)
        got = (lambda r: None if r[1] is None else r[1]["fid"])(match_captures.process_frame(sim_capture(page, rng, 0.7)))
        if got == fid:
            ok += 1
        elif got is not None:
            wrong += 1
    print("decode rate", ok / 200, "wrong", wrong)
    assert wrong == 0
    assert ok >= 190


def test_match_captures_crops_tile_region():
    tile = smooth_tile(7)
    page = show_tiles.render_page(tile, 321, 9)
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
        cv2.imwrite(str(fr / f"f{fid}.jpg"), sim_capture(show_tiles.render_page(t, fid, 5), rng, 0.3))
    cv2.imwrite(str(fr / "blank.jpg"), np.full((480, 640, 3), 128, np.uint8))
    sess = {"session_id": "ab", "session_marker": 5, "seed": 0, "n": 3, "x_sha256": "00", "map": {str(i): i * 10 for i in range(3)}}
    (tmp_path / "rig_session.json").write_text(json.dumps(sess))
    y = np.arange(100) % 9
    np.save(tmp_path / "y.npy", y)
    out = tmp_path / "capture"
    r = subprocess.run([sys.executable, str(ROOT / "rig" / "match_captures.py"), str(fr),
                        str(tmp_path / "rig_session.json"), "--labels", str(tmp_path / "y.npy"),
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    meta = json.loads((out / "meta.json").read_text())
    assert meta["n_frames"] == 4 and meta["n_matched"] == 3 and meta["rejected"]["no_marker"] == 1
    assert meta["session_id"] == "ab" and meta["session_marker"] == 5
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


def _run_match(tmp_path, frames, sess):
    (tmp_path / "s.json").write_text(json.dumps(sess))
    np.save(tmp_path / "y.npy", np.arange(100) % 9)
    return subprocess.run([sys.executable, str(ROOT / "rig" / "match_captures.py"), str(frames), str(tmp_path / "s.json"),
                           "--labels", str(tmp_path / "y.npy"), "--out", str(tmp_path / "o")], capture_output=True, text=True)


def test_session_mismatch_and_old_format_refused(tmp_path):
    fr = tmp_path / "fr"
    fr.mkdir()
    cv2.imwrite(str(fr / "a.jpg"), show_tiles.render_page(smooth_tile(1), 3, 100))
    sess = {"session_id": "x", "session_marker": 101, "map": {"3": 30}}
    r = _run_match(tmp_path, fr, sess)
    assert r.returncode == 3 and "different session" in r.stderr  # nothing matched, mismatch -> refuse
    assert not (tmp_path / "o" / "x.npy").exists()
    r = _run_match(tmp_path, fr, {"3": 30})  # old format without nonce
    assert r.returncode != 0
    sess["session_marker"] = 100
    r = _run_match(tmp_path, fr, sess)
    assert r.returncode == 0 and json.loads((tmp_path / "o" / "meta.json").read_text())["n_matched"] == 1


def test_transition_and_untrusted_frames_dropped():
    tile = smooth_tile(3)
    page = show_tiles.render_page(tile, 12, 4)
    assert match_captures.process_frame(page)[0] == "ok"
    torn = page.copy()
    torn[:, 640:] = 255  # half the tile region replaced by blank: not a clean frame
    st, r = match_captures.process_frame(torn)
    assert r is None and st in ("not_refined", "edge_check", "bad_geometry", "no_marker")
    torn2 = page.copy()
    torn2[430:, 300:980] = 255  # bottom of tile and frame missing
    assert match_captures.process_frame(torn2)[1] is None
    # one marker only (e.g. tile content mistaken for a marker) is not enough
    one = page.copy()
    one[300:500, 1050:1250] = 255
    assert match_captures.process_frame(one)[0] == "bad_geometry"
    # allow_unrefined is the explicit opt-in
    assert match_captures.process_frame(torn, allow_unrefined=True)[0] in ("ok", "uniform", "no_marker", "bad_geometry")


@pytest.mark.skipif(not (ROOT / "data" / "crc_test_x.npy").exists(), reason="needs held-out tiles")
def test_real_dark_tiles_no_false_marker():
    X = np.load(ROOT / "data" / "crc_test_x.npy", mmap_mode="r")
    sub = np.arange(0, len(X), 25)
    means = np.array([float(np.asarray(X[i]).mean()) for i in sub])
    pick = list(sub[np.argsort(means)[:6]]) + [0, 100, 2000, 5000]
    rng = np.random.RandomState(3)
    for j, ti in enumerate(pick):
        t = np.array(X[ti])
        fid, sess = int(rng.randint(2048)), int(rng.randint(2048))
        page = show_tiles.render_page(t, fid, sess)
        st, r = match_captures.process_frame(sim_capture(page, rng, 0.7))
        assert r is None or (r["fid"] == fid and r["session"] == sess), (ti, st)
        found = marker.find_markers(page)
        assert sorted(v for v, _ in found) == sorted([fid, sess]) or fid == sess
        # a tile alone (no markers/frame) never yields a pair
        assert marker.pick_pair(marker.find_markers(cv2.resize(t, (672, 672)))) is None
