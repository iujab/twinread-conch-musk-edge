#!/usr/bin/env python
"""Match recorded camera frames to the tiles shown by show_tiles.py.

usage: match_captures.py frames/ rig_session.json [--labels data/crc_test_y.npy] [--out data/capture]
Decodes the marker in each frame, maps the fixed page layout (marker.py) through the marker's perspective to crop the
tile region, resizes to 224 and writes x.npy (RGB-as-shown, uint8 [N,224,224,3] in BGR->RGB), y.npy, idx.npy, meta.json.
Frames with no decodable marker (or an id not in the session) are dropped and counted. Real camera-filming-a-screen
frames only; nothing here is simulated.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import marker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _refine_frame(frame, pred):
    """Find the tile's black-frame outer quad near the marker-predicted one; sub-pixel corners or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    thr, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (g < thr).astype(np.uint8) * 255
    cs, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    side = float(np.linalg.norm(pred[0] - pred[1]))
    best = None
    for c in sorted(cs, key=cv2.contourArea, reverse=True)[:12]:
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        for eps in (0.01, 0.02, 0.035):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4:
                q = marker._order(ap)
                if np.abs(q - pred).max() < 0.1 * side:
                    best = q
                break
        if best is not None:
            break
    if best is None:
        return None
    h, w = gray.shape[:2]
    win = int(max(2, min(10, side / 60)))
    if best.min() < win + 1 or best[:, 0].max() > w - win - 2 or best[:, 1].max() > h - win - 2:
        return best
    c = best.reshape(-1, 1, 2).copy()
    try:
        cv2.cornerSubPix(gray, c, (win, win), (-1, -1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    except cv2.error:
        return best
    out = c.reshape(4, 2)
    return out if np.abs(out - best).max() < 0.03 * side else best


def _edges_ok(gray, Hm):
    """All four black-frame edges must be present: dark ring-centre samples vs white samples just outside."""
    x, y = marker.TILE_XY
    t = marker.TILE_PX
    f = marker.FRAME_PX
    u = np.linspace(0.1, 0.9, 25) * t
    top = lambda d: np.stack([x + u, np.full_like(u, y + d)], 1)
    bot = lambda d: np.stack([x + u, np.full_like(u, y + t - d)], 1)
    lef = lambda d: np.stack([np.full_like(u, x + d), y + u], 1)
    rig = lambda d: np.stack([np.full_like(u, x + t - d), y + u], 1)
    h, w = gray.shape[:2]
    for fn in (top, bot, lef, rig):
        pr = np.concatenate([fn(-f / 2), fn(-(f + 7))]).reshape(1, -1, 2).astype(np.float32)
        pf = cv2.perspectiveTransform(pr, Hm)[0] - 0.5
        if pf.min() < 0 or pf[:, 0].max() > w - 1 or pf[:, 1].max() > h - 1:
            return False
        v = cv2.remap(gray, pf[:, 0].reshape(1, -1), pf[:, 1].reshape(1, -1), cv2.INTER_LINEAR)[0].astype(np.float32)
        r_, o_ = v[:25], v[25:]
        if np.mean(r_ < 0.75 * o_) < 0.9 or np.median(o_) < 100:
            return False
    return True


def process_frame(frame, out=224, allow_unrefined=False):
    """frame BGR -> ('ok', dict(fid, session, crop, refined)) or (reason, None).
    Reasons: no_marker, bad_geometry, not_refined, edge_check, uniform."""
    found = marker.find_markers(frame, max_n=3)
    if not found:
        return "no_marker", None
    pair = marker.pick_pair(found)
    if pair is None:
        return "bad_geometry", None
    fid, sess, qa, qb = pair
    # detected corners are in pixel-centre coords; page/tile coords are pixel-edge coords (+0.5)
    srcp = np.concatenate([marker.black_square_page_quad(), marker.black_square_page_quad(marker.SESSION_XY)])
    dstp = np.concatenate([qa, qb]).astype(np.float32) + 0.5
    H, _ = cv2.findHomography(srcp, dstp, 0)
    if H is None:
        return "bad_geometry", None
    x, y = marker.TILE_XY
    t = marker.TILE_PX
    pred = cv2.perspectiveTransform(marker.tile_frame_page_quad().reshape(1, 4, 2), H)[0] - 0.5
    ref = _refine_frame(frame, pred)
    if ref is not None:  # local homography from the tile's own frame: far more accurate than extrapolating from the markers
        H = cv2.getPerspectiveTransform(marker.tile_frame_page_quad(), ref.astype(np.float32) + 0.5)
    elif not allow_unrefined:
        return "not_refined", None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if not allow_unrefined and not _edges_ok(gray, H):
        return "edge_check", None
    corners = np.array([[[x, y], [x + t, y], [x + t, y + t], [x, y + t]]], np.float32)
    src = cv2.perspectiveTransform(corners, H)[0] - 0.5
    dst = np.array([[-0.5, -0.5], [out - 0.5, -0.5], [out - 0.5, out - 0.5], [-0.5, out - 0.5]], np.float32)
    M = cv2.getPerspectiveTransform(src.astype(np.float32), dst)
    crop = cv2.warpPerspective(frame, M, (out, out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if crop.std() < 6 and crop.mean() > 235:
        return "uniform", None
    return "ok", dict(fid=fid, session=sess, crop=crop, refined=ref is not None)


def crop_tile(frame, out=224, allow_unrefined=False):
    """frame BGR -> (frame_id, crop BGR out x out) or None (any rejection)."""
    st, r = process_frame(frame, out, allow_unrefined)
    return None if st != "ok" else (r["fid"], r["crop"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames")
    ap.add_argument("session")
    ap.add_argument("--labels", default=str(ROOT / "data" / "crc_test_y.npy"))
    ap.add_argument("--out", default=str(ROOT / "data" / "capture"))
    ap.add_argument("--allow-unrefined", action="store_true", help="keep frames whose tile frame was not located (off by default)")
    ap.add_argument("--expect-x-sha256", default=None, help="refuse if the session file's x_sha256 differs from this")
    a = ap.parse_args(argv)
    sess_file = json.loads(Path(a.session).read_text())
    if "map" not in sess_file or "session_marker" not in sess_file:
        raise SystemExit("old-format session file without session_marker/map: refusing (cannot verify the session)")
    if a.expect_x_sha256 and sess_file.get("x_sha256") != a.expect_x_sha256:
        raise SystemExit("session x_sha256 does not match --expect-x-sha256")
    session = {int(k): int(v) for k, v in sess_file["map"].items()}
    smark = int(sess_file["session_marker"])
    labels = np.load(a.labels)
    files = sorted(p for p in Path(a.frames).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    xs, idx, names = [], [], []
    rej = dict(no_marker=0, bad_geometry=0, not_refined=0, edge_check=0, uniform=0, session_mismatch=0, unknown_id=0)
    for p in files:
        img = cv2.imread(str(p))
        st, r = ("no_marker", None) if img is None else process_frame(img, allow_unrefined=a.allow_unrefined)
        if r is None:
            rej[st] += 1
            continue
        if r["session"] != smark:
            rej["session_mismatch"] += 1
            continue
        if r["fid"] not in session:
            rej["unknown_id"] += 1
            continue
        xs.append(cv2.cvtColor(r["crop"], cv2.COLOR_BGR2RGB))
        idx.append(session[r["fid"]])
        names.append(p.name)
    if files and rej["session_mismatch"] and not xs:
        print("ERROR: every decodable frame belongs to a different session; refusing", file=sys.stderr)
        return 3
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    x = np.stack(xs) if xs else np.zeros((0, 224, 224, 3), np.uint8)
    idx = np.asarray(idx, np.int64)
    np.save(out / "x.npy", x)
    np.save(out / "idx.npy", idx)
    np.save(out / "y.npy", labels[idx] if len(idx) else np.zeros((0,), labels.dtype))
    meta = dict(n_frames=len(files), n_matched=len(xs), n_dropped_no_marker=rej["no_marker"], rejected=rej,
                session_id=sess_file.get("session_id"), session_marker=smark, x_sha256=sess_file.get("x_sha256"),
                seed=sess_file.get("seed"), n_tiles=sess_file.get("n"), allow_unrefined=a.allow_unrefined,
                n_unique_tiles=int(len(set(idx.tolist()))), frames=names,
                note="real camera frames of a screen; labels from the shown held-out CRC test tiles")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "frames"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
