"""Image assets for docs/deck/build_deck.js. Writes docs/deck/assets/ (committed, so the deck builds without data/).

Needs data/crc_test_x.npy and results/cm_local/CM/crc/tile_preds.npz (both git-ignored) and the bench venv:
    ../../../.venv/bin/python docs/deck/make_assets.py        # from anywhere; paths resolve from this file
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "assets"
CRC = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
TILE_PX = 448  # 224 px tiles, doubled so they stay sharp when projected


def save_tile(arr, path):
    Image.fromarray(np.asarray(arr)).resize((TILE_PX, TILE_PX), Image.LANCZOS).save(path, quality=92)


def main():
    OUT.mkdir(exist_ok=True)
    t = np.load(ROOT / "results/cm_local/CM/crc/tile_preds.npz")
    xs = np.load(ROOT / "data/crc_test_x.npy", mmap_mode="r")
    y, a, b = t["y"], t["pA"].argmax(1), t["pB"].argmax(1)

    # Title mosaic: the tile per class that both readers get right with the highest shared confidence.
    conf = np.minimum(t["pA"].max(1), t["pB"].max(1))
    for c, name in enumerate(CRC):
        ok = np.where((y == c) & (a == c) & (b == c))[0]
        save_tile(xs[ok[np.argsort(-conf[ok])[0]]], OUT / f"class_{name}.jpg")

    # Shared blind spots: same selection as review.json, then the ones the pre-registered rule did not flag.
    idx = json.loads((ROOT / "results/cm_local/CM/crc/review.json").read_text())["confident_shared_errors"]["indices"]
    assert all(a[i] == b[i] != y[i] and conf[i] >= 0.80 for i in idx)
    rows = [{"tile": int(i), "truth": CRC[y[i]], "call": CRC[a[i]], "conch": float(t["pA"][i].max()),
             "musk": float(t["pB"][i].max()), "flagged": bool(t["flag"][i])} for i in idx]
    missed = sorted((r for r in rows if not r["flagged"]), key=lambda r: (r["truth"], -min(r["conch"], r["musk"])))
    for r in missed:
        save_tile(xs[r["tile"]], OUT / f"missed_{r['tile']}.jpg")
    pairs = Counter(f"{r['truth']}->{r['call']}" for r in rows)
    (OUT / "blindspots.json").write_text(json.dumps(
        {"n": len(rows), "flagged": sum(r["flagged"] for r in rows), "pairs": pairs, "missed": missed}, indent=1) + "\n")

    # Board photo (crop only: away from the USB dongle at the lower right) and the app screenshot (banner and threshold line off).
    Image.open(ROOT / "figures/extra/uno_q_photo.jpg").convert("RGB").crop((230, 0, 1450, 968)).save(
        OUT / "uno_q.jpg", quality=90)
    im = Image.open(ROOT / "figures/ui_device_unoq.png").convert("RGB")
    im.crop((0, 35, im.width, 895)).save(OUT / "app_ui.png")

    # Footer motif: the two readers, overlapping, in the deck's theme colours.
    w, h = 600, 400
    motif = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for x, c in ((0, (37, 115, 190, 224)), (w - h, (219, 75, 134, 209))):
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(layer).ellipse((x, 0, x + h - 1, h - 1), fill=c)
        motif = Image.alpha_composite(motif, layer)
    motif.resize((300, 200), Image.LANCZOS).save(OUT / "motif.png")
    print(f"wrote {len(list(OUT.iterdir()))} files to {OUT}; missed by the flag: {[(r['truth'], r['call']) for r in missed]}")


if __name__ == "__main__":
    main()
