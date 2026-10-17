"""Camera-filming-a-screen SIMULATION (not real camera data): used by tests and the decode stress run."""
import cv2
import numpy as np


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
