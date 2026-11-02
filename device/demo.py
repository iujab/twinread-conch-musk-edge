"""Preset demo slides for the device app: real test tiles with known labels, made by scripts/uno_q/make_demo_slides.py."""
from __future__ import annotations

import json
from pathlib import Path

import cv2


class DemoDeck:
    """Slides in manifest order, decoded once, so a slide change costs no disk read while the model is running."""

    def __init__(self, folder):
        self.folder = Path(folder)
        manifest = json.loads((self.folder / "manifest.json").read_text())
        self.slides = []
        for s in manifest["slides"]:
            bgr = cv2.imread(str(self.folder / s["file"]))
            if bgr is None:
                raise ValueError(f"cannot read demo slide {s['file']}")
            self.slides.append({"file": s["file"], "truth": s["truth"], "bgr": bgr})
        if not self.slides:
            raise ValueError("the manifest lists no slides")
        self._thumbs = {}

    def __len__(self):
        return len(self.slides)

    def thumb(self, i: int) -> bytes:
        if i not in self._thumbs:
            ok, buf = cv2.imencode(".jpg", cv2.resize(self.slides[i]["bgr"], (112, 112), interpolation=cv2.INTER_AREA),
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            self._thumbs[i] = buf.tobytes() if ok else b""
        return self._thumbs[i]


BUILD = "Build them with scripts/uno_q/make_demo_slides.py."


def load_deck(folder):
    """(deck, None), or (None, a reason the dashboard can show) when there are no usable slides."""
    if folder is None:
        return None, f"No demo slides are set up for this run. {BUILD}"
    try:
        return DemoDeck(folder), None
    except FileNotFoundError:
        return None, f"No demo slides in {folder}. {BUILD}"
    except (OSError, ValueError, KeyError, TypeError) as e:
        return None, f"The demo slides in {folder} are unusable ({e}). {BUILD}"
