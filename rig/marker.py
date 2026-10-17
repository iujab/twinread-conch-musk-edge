"""Machine-readable frame-ID marker + the fixed page layout shared by show_tiles.py and match_captures.py.

Marker (default 120x120 px, uint8 BGR/gray-replicated):
  * 10 % white quiet zone, then a black square (96 px at size 120) whose outer 8 px are a black border,
  * inside the border a 4x4 grid of 20 px cells, cell (r, c) = bit index 4*r + c, white = 1, black = 0,
  * bits 0..11 = frame id (MSB first), bits 12..15 = parity: bit 12+j = 1 XOR (XOR of the data bits in group j) (odd parity),
    where data cell (r, c) belongs to group (c + 2*r) % 4 (horizontal, vertical and diagonal neighbours land in
    different groups, so a blurred pair of adjacent cells cannot cancel out).
Decoding also rejects any read whose cell contrast is ambiguous. Orientation is assumed upright (|rotation| < 45 deg).

Page layout (pixels of the 1280x800 page image; documented here because match_captures.py crops by it):
  marker image at MARKER_XY with side MARKER_PX (scale 1.5 over the 120 px marker), tile at TILE_XY with side TILE_PX,
  a FRAME_PX black frame just outside the tile (its outer corners give an accurate, local tile homography).
"""
import cv2
import numpy as np

QUIET_FRAC = 0.1
N_BITS = 12
BASE = 120  # canonical marker size

# page layout (see module docstring)
PAGE_W, PAGE_H = 1280, 800
MARKER_PX = 180
MARKER_XY = (60, 310)
TILE_PX = 672
TILE_XY = (300, 64)
FRAME_PX = 8  # black frame drawn just outside the tile; match_captures refines the tile corners on it


def _group(i):
    r, c = divmod(i, 4)
    return (c + 2 * r) % 4


def _bits_for(frame_id):
    if not 0 <= frame_id < (1 << N_BITS):
        raise ValueError(f"frame_id must be in [0, {1 << N_BITS})")
    data = [(frame_id >> (N_BITS - 1 - i)) & 1 for i in range(N_BITS)]
    par = [1] * 4  # odd parity, so an all-dark blob (id 0 data + even parity) can never be a valid marker
    for i, b in enumerate(data):
        par[_group(i)] ^= b
    return data + par


def encode_marker(frame_id: int, size: int = 120) -> np.ndarray:
    bits = _bits_for(int(frame_id))
    q = int(round(BASE * QUIET_FRAC))  # 12
    s = BASE - 2 * q                   # 96 black square
    b = s // 12                        # 8 border
    cell = (s - 2 * b) // 4            # 20
    img = np.full((BASE, BASE), 255, np.uint8)
    img[q:q + s, q:q + s] = 0
    for k, bit in enumerate(bits):
        r, c = divmod(k, 4)
        y, x = q + b + r * cell, q + b + c * cell
        img[y:y + cell, x:x + cell] = 255 if bit else 0
    if size != BASE:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    return np.repeat(img[:, :, None], 3, axis=2)


def _order(pts):
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = pts.sum(1), pts[:, 0] - pts[:, 1]
    return np.array([pts[np.argmin(s)], pts[np.argmax(d)], pts[np.argmax(s)], pts[np.argmin(d)]], np.float32)


def _read_candidate(gray, quad):
    """quad TL,TR,BR,BL of the black square -> frame id or None."""
    q = int(round(BASE * QUIET_FRAC))
    s = BASE - 2 * q
    b = s // 12
    cell = (s - 2 * b) // 4
    dst = np.array([[q, q], [q + s, q], [q + s, q + s], [q, q + s]], np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    w = cv2.warpPerspective(gray, M, (BASE, BASE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    w = w.astype(np.float32)
    border = np.concatenate([w[q + 2:q + b - 2, q + 8:q + s - 8].ravel(), w[q + s - b + 2:q + s - 2, q + 8:q + s - 8].ravel(),
                             w[q + 8:q + s - 8, q + 2:q + b - 2].ravel(), w[q + 8:q + s - 8, q + s - b + 2:q + s - 2].ravel()])
    dark = float(np.median(border))
    ring = np.concatenate([w[1:q - 3, :].ravel(), w[BASE - q + 3:BASE - 1, :].ravel()])
    white = float(np.median(ring))
    if white - dark < 50 or dark > 0.5 * white or border.std() > 0.3 * (white - dark) or ring.std() > 0.3 * (white - dark):
        return None
    bits = []
    for k in range(16):
        r, c = divmod(k, 4)
        cy, cx = q + b + r * cell + cell // 2, q + b + c * cell + cell // 2
        m = float(w[cy - 4:cy + 5, cx - 4:cx + 5].mean())
        n = (m - dark) / (white - dark)
        if 0.3 < n < 0.7:  # ambiguous contrast: refuse rather than guess
            return None
        bits.append(1 if n >= 0.5 else 0)
    par = [1] * 4
    for i in range(N_BITS):
        par[_group(i)] ^= bits[i]
    if par != bits[12:]:
        return None
    fid = 0
    for bit in bits[:N_BITS]:
        fid = (fid << 1) | bit
    return fid


def _refine(gray, pts):
    """Sub-pixel corner refinement (the contour quad is inset by about half a pixel per side)."""
    side = float(np.linalg.norm(pts[0] - pts[1]))
    w = int(max(2, min(8, side / 14)))
    h, wd = gray.shape[:2]
    if pts.min() < w + 1 or pts[:, 0].max() > wd - w - 2 or pts[:, 1].max() > h - w - 2:
        return pts
    c = pts.reshape(-1, 1, 2).copy()
    try:
        cv2.cornerSubPix(gray, c, (w, w), (-1, -1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    except cv2.error:
        return pts
    out = c.reshape(4, 2)
    return out if np.abs(out - pts).max() < 0.25 * side else pts


def decode_marker_ex(img: np.ndarray):
    """Return (frame_id, quad[4,2] of the black square TL,TR,BR,BL in image px) or None."""
    if img is None or img.size == 0:
        return None
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    thr, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for t in (thr, 100, 140):
        mask = (g < t).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cs, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cs = sorted(cs, key=cv2.contourArea, reverse=True)[:12]
        for c in cs:
            if cv2.contourArea(c) < 150:
                continue
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            quad = None
            for eps in (0.02, 0.035, 0.05, 0.08):
                ap = cv2.approxPolyDP(hull, eps * peri, True)
                if len(ap) == 4:
                    quad = ap
                    break
            if quad is None:
                continue
            area = cv2.contourArea(quad)
            if area < 150 or area < 0.7 * cv2.contourArea(hull):
                continue
            pts = _order(quad)
            sides = np.linalg.norm(pts - np.roll(pts, -1, 0), axis=1)
            if sides.max() > 1.6 * sides.min():
                continue
            fid = _read_candidate(gray, pts)
            if fid is not None:
                return fid, _refine(gray, pts)
    return None


def decode_marker(img: np.ndarray):
    r = decode_marker_ex(img)
    return None if r is None else r[0]


def black_square_page_quad():
    """Corners (TL,TR,BR,BL) of the marker's black square in page pixels."""
    k = MARKER_PX / BASE
    q = BASE * QUIET_FRAC * k
    s = (BASE - 2 * BASE * QUIET_FRAC) * k
    x0, y0 = MARKER_XY[0] + q, MARKER_XY[1] + q
    return np.array([[x0, y0], [x0 + s, y0], [x0 + s, y0 + s], [x0, y0 + s]], np.float32)


def tile_frame_page_quad():
    """Outer corners (TL,TR,BR,BL) of the black frame around the tile, in page pixels."""
    x, y = TILE_XY[0] - FRAME_PX, TILE_XY[1] - FRAME_PX
    t = TILE_PX + 2 * FRAME_PX
    return np.array([[x, y], [x + t, y], [x + t, y + t], [x, y + t]], np.float32)
