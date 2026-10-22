"""Per-stage time of one device-app frame on the UNO Q (replay mode), using the app's own functions in App.step order.

Run from device/:  PYTHONPATH=. .venv/bin/python stage_timing.py --bundle ../bundles/cm --frame ../data/replay_crc/X.png
The staged pipeline is checked against EdgeModel.infer on the same frame. Research prototype, not a diagnostic device.
"""
import argparse
import json
import platform
import time

import cv2
import numpy as np

import bridge
import twinread_runtime as rt

STAGES = ("LED busy (Bridge RPC)", "frame read (PNG decode)", "crop to 224 px", "uint8 -> float NCHW",
          "int8 student (onnxruntime)", "probes + softmax", "tissue check", "review rule", "LED result (Bridge RPC)",
          "tissue fraction again (UI)", "UI frame (box + JPEG)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = rt.EdgeModel(a.bundle, threads=a.threads)
    rows = []
    for it in range(a.warmup + a.iters):
        t = [time.perf_counter()]
        bridge.send_status("busy"); t.append(time.perf_counter())
        frame = np.ascontiguousarray(cv2.imread(a.frame)[..., ::-1]); t.append(time.perf_counter())
        tile = rt.crop_from_frame(frame, "center"); t.append(time.perf_counter())
        xin = np.ascontiguousarray(tile[None].transpose(0, 3, 1, 2), dtype=np.float32) / 255.0
        t.append(time.perf_counter())
        hA, hB, logits, rl = m.sess.run(["hA", "hB", "logits", "review_logit"], {"x": xin}); t.append(time.perf_counter())
        pA, pB, pF = m._probe("A", hA)[0], m._probe("B", hB)[0], rt._softmax(logits.astype(np.float64))[0]
        t.append(time.perf_counter())
        nt = rt._no_tissue(tile); t.append(time.perf_counter())
        review, score, reasons = rt.device_review_rule(pA, pB, pF, float(rl[0]), m.thr, nt); t.append(time.perf_counter())
        status = "no_tissue" if nt else ("review" if review else "agree")
        bridge.send_status(status); t.append(time.perf_counter())
        tf = rt.tissue_fraction(tile); t.append(time.perf_counter())
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = rt.center_box(h, w)
        vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (0, 220, 255), 2)
        cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85]); t.append(time.perf_counter())
        if it >= a.warmup:
            rows.append(np.diff(t) * 1000)
    rows = np.array(rows)
    ref = m.infer(tile[None])[0]
    label = m.classes[int(np.argmax(pF))]
    out = {
        "bundle": a.bundle, "frame": a.frame, "frame_shape": list(frame.shape), "threads": a.threads,
        "iters": a.iters, "warmup": a.warmup,
        "host": {"machine": platform.machine(), "node": platform.node(), "release": platform.release()},
        "simulated": platform.machine() != "aarch64",
        "stages": [{"stage": s, "p50_ms": float(np.percentile(rows[:, i], 50)),
                    "p90_ms": float(np.percentile(rows[:, i], 90)), "mean_ms": float(rows[:, i].mean())}
                   for i, s in enumerate(STAGES)],
        "total_p50_ms": float(np.percentile(rows.sum(1), 50)), "total_p90_ms": float(np.percentile(rows.sum(1), 90)),
        "check": {"staged_label": label, "edgemodel_label": ref.label, "staged_review": review,
                  "edgemodel_review": ref.review, "tissue_fraction": tf, "matches_edgemodel":
                  label == ref.label and review == ref.review and abs(score - ref.review_score) < 1e-9},
        "note": "Replay mode: the frame is a 224x224 PNG, so crop is a no-op resize; a camera frame adds capture "
                "and a real resize. Research prototype, not a diagnostic device.",
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    for s in out["stages"]:
        print(f"{s['stage']:28s} p50 {s['p50_ms']:7.2f} ms  p90 {s['p90_ms']:7.2f}")
    print("total p50", round(out["total_p50_ms"], 2), "p90", round(out["total_p90_ms"], 2), out["check"])


if __name__ == "__main__":
    main()
