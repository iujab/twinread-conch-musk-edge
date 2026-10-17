#!/usr/bin/env python
"""Large-trial SIMULATED decode stress (not real camera data): real held-out CRC tiles incl. the darkest ones,
random ids and session markers, sim_capture at severities 0.7 and 1.0. Writes results/rig/decode_stress.json."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import match_captures as mc  # noqa: E402
import show_tiles  # noqa: E402
from sim_capture import sim_capture  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def pick_tiles(X, n_dark=40, n_rand=160, stride=7):
    sub = np.arange(0, len(X), stride)
    means = np.array([float(np.asarray(X[i]).mean()) for i in sub])
    dark = sub[np.argsort(means)[:n_dark]]
    rng = np.random.RandomState(0)
    return [int(i) for i in dark] + [int(i) for i in rng.choice(len(X), n_rand, replace=False)], means.min()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "results" / "rig" / "decode_stress.json"))
    a = ap.parse_args()
    X = np.load(ROOT / "data" / "crc_test_x.npy", mmap_mode="r")
    tiles, dmin = pick_tiles(X)
    res = dict(note="SIMULATED camera-filming-a-screen (rig/sim_capture.py), real CRC test tiles; 40 darkest of a 1/7 subsample + 160 random",
               min_tile_mean_level=float(dmin), runs=[])
    for sev, hw in [(0.7, (480, 640)), (1.0, (480, 640)), (0.7, (720, 1280)), (1.0, (720, 1280))]:
        rng = np.random.RandomState(11)
        c = dict(ok=0, wrong_id=0, wrong_session=0, rejected=0)
        reasons = {}
        for k in range(a.trials):
            fid, sess = int(rng.randint(2048)), int(rng.randint(2048))
            t = np.array(X[tiles[k % len(tiles)]])
            f = sim_capture(show_tiles.render_page(t, fid, sess), rng, sev, hw)
            st, r = mc.process_frame(f)
            if r is None:
                c["rejected"] += 1
                reasons[st] = reasons.get(st, 0) + 1
            elif r["fid"] == fid and r["session"] == sess:
                c["ok"] += 1
            else:
                c["wrong_id" if r["fid"] != fid else "wrong_session"] += 1
        res["runs"].append(dict(severity=sev, frame_hw=list(hw), trials=a.trials, reject_reasons=reasons, **c,
                                accept_rate=c["ok"] / a.trials, wrong_id_rate=(c["wrong_id"] + c["wrong_session"]) / a.trials))
        print(res["runs"][-1], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
