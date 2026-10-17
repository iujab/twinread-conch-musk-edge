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


def crop_tile(frame, out=224, return_info=False):
    """frame BGR -> (frame_id, crop BGR out x out) or None. With return_info -> (fid, crop, refined_on_frame: bool)."""
    r = marker.decode_marker_ex(frame)
    if r is None:
        return None
    fid, quad = r
    # detected corners are in pixel-centre coords; page/tile coords are pixel-edge coords (+0.5)
    H = cv2.getPerspectiveTransform(marker.black_square_page_quad(), quad.astype(np.float32) + 0.5)
    x, y = marker.TILE_XY
    t = marker.TILE_PX
    f = marker.FRAME_PX
    frame_q = marker.tile_frame_page_quad().reshape(1, 4, 2)
    pred = cv2.perspectiveTransform(frame_q, H)[0] - 0.5
    ref = _refine_frame(frame, pred)
    if ref is not None:  # local homography from the tile's own frame: far more accurate than extrapolating from the marker
        H = cv2.getPerspectiveTransform(marker.tile_frame_page_quad(), ref.astype(np.float32) + 0.5)
    corners = np.array([[[x, y], [x + t, y], [x + t, y + t], [x, y + t]]], np.float32)
    src = cv2.perspectiveTransform(corners, H)[0] - 0.5
    dst = np.array([[-0.5, -0.5], [out - 0.5, -0.5], [out - 0.5, out - 0.5], [-0.5, out - 0.5]], np.float32)
    M = cv2.getPerspectiveTransform(src.astype(np.float32), dst)
    crop = cv2.warpPerspective(frame, M, (out, out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return (fid, crop, ref is not None) if return_info else (fid, crop)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames")
    ap.add_argument("session")
    ap.add_argument("--labels", default=str(ROOT / "data" / "crc_test_y.npy"))
    ap.add_argument("--out", default=str(ROOT / "data" / "capture"))
    a = ap.parse_args(argv)
    session = {int(k): int(v) for k, v in json.loads(Path(a.session).read_text()).items()}
    labels = np.load(a.labels)
    files = sorted(p for p in Path(a.frames).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    xs, idx, names = [], [], []
    no_marker = unknown_id = n_refined = 0
    for p in files:
        img = cv2.imread(str(p))
        r = None if img is None else crop_tile(img, return_info=True)
        if r is None:
            no_marker += 1
            continue
        fid, crop, refined = r
        n_refined += int(refined)
        if fid not in session:
            unknown_id += 1
            continue
        xs.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        idx.append(session[fid])
        names.append(p.name)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    x = np.stack(xs) if xs else np.zeros((0, 224, 224, 3), np.uint8)
    idx = np.asarray(idx, np.int64)
    np.save(out / "x.npy", x)
    np.save(out / "idx.npy", idx)
    np.save(out / "y.npy", labels[idx] if len(idx) else np.zeros((0,), labels.dtype))
    meta = dict(n_frames=len(files), n_matched=len(xs), n_dropped_no_marker=no_marker, n_dropped_unknown_id=unknown_id, n_tile_frame_refined=n_refined,
                n_unique_tiles=int(len(set(idx.tolist()))), frames=names,
                note="real camera frames of a screen; labels from the shown held-out CRC test tiles")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "frames"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
