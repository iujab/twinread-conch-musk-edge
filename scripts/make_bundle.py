"""Build the edge bundle for a student: fp32 ONNX -> int8 (MinMax vs Entropy, lower val drift wins) -> thresholds
recalibrated on int8 val outputs -> bundle + int8_drift.json.  usage: make_bundle.py --pair OPEN --dataset crc"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from twinread import export, stats  # noqa: E402
from twinread import student as S  # noqa: E402
from twinread.data import CRC_CLASSES, load_split  # noqa: E402


def to_f(x):  # uint8 NHWC -> float NCHW [0,1]
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="OPEN")
    ap.add_argument("--dataset", default="crc")
    ap.add_argument("--student", default=None)
    ap.add_argument("--targets", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_calib", type=int, default=512)
    a = ap.parse_args()
    torch.set_num_threads(4)
    sdir = Path(a.student) if a.student else ROOT / "results" / "student" / f"{a.pair.lower()}_{a.dataset}"
    tdir = Path(a.targets) if a.targets else S.resolve_targets_dir(a.pair, a.dataset)
    out = Path(a.out) if a.out else ROOT / "bundles" / ("open" if a.pair.upper() == "OPEN" else "cm")
    with np.load(tdir / "student_targets.npz") as z:
        T = {k: z[k] for k in z.files}
    with np.load(tdir / "probes_aligned.npz") as z:
        PR = {k: z[k] for k in z.files}
    classes = [CRC_CLASSES[i] for i in PR["A_classes"]] if a.dataset == "crc" else [str(i) for i in PR["A_classes"]]
    xtr, _ = load_split(ROOT / "data", a.dataset, "train")
    xv, yv = load_split(ROOT / "data", a.dataset, "val")
    xt, yt = load_split(ROOT / "data", a.dataset, "test")
    cal = to_f(xtr[np.sort(np.random.default_rng(0).choice(len(xtr), a.n_calib, replace=False))])
    model = S.load_student(sdir / "best.pt")
    work = Path(tempfile.mkdtemp(prefix="mkbundle_"))
    fp = export.export_onnx(model, work / "fp32.onnx")
    fv, ft = to_f(xv), to_f(xt)
    o32 = {"val": export.run_onnx(fp, fv), "test": export.run_onnx(fp, ft)}
    cand = {}
    for meth in ("minmax", "entropy", "percentile99.99", "percentile99.9"):
        t0 = time.time()
        q = export.quantize_int8(fp, cal, work / f"int8_{meth}.onnx", meth)
        ov = export.run_onnx(q, fv)
        d = export.compare_outputs(o32["val"], ov)
        d["drift_score"] = 1 - 0.5 * (d["cos_hA_mean"] + d["cos_hB_mean"])
        d["quantize_s"] = round(time.time() - t0, 1)
        cand[meth] = {"path": q, "val_out": ov, "val_drift": d}
        print(meth, {k: round(v, 5) for k, v in d.items() if isinstance(v, float)}, flush=True)
    # lower val drift (no labels used): mean(1-cos) over hA,hB; ties (<1e-6) -> higher label agreement
    pick = min(cand, key=lambda m: (round(cand[m]["val_drift"]["drift_score"], 6), -cand[m]["val_drift"]["label_agreement"]))
    q, ov = cand[pick]["path"], cand[pick]["val_out"]
    ot = export.run_onnx(q, ft)

    def bal(o, y):
        return stats.balanced_accuracy(y, np.array(classes_arr)[o["logits"].argmax(1)])
    classes_arr = PR["A_classes"].astype(np.int64)
    thr_old = json.loads((sdir / "thresholds.json").read_text())
    pv = (S.probe_probs(ov["hA"], PR, "A"), S.probe_probs(ov["hB"], PR, "B"))
    thr_info = S.choose_thresholds(pv[0], pv[1], ov["review_logit"], T["review_val"], budget=S.REVIEW_BUDGET)
    thr_new = {"js": float(thr_info["js"]), "review": float(thr_info["review"])}

    def flag_rate(o, thr):
        return float(S.device_flags(S.probe_probs(o["hA"], PR, "A"), S.probe_probs(o["hB"], PR, "B"),
                                    o["review_logit"], thr)["flag"].mean())
    rep = {"pair": a.pair.upper(), "dataset": a.dataset, "backbone": model.config["backbone"], "method_picked": pick,
           "calibration": f"{a.n_calib} train tiles, seed 0, uint8 -> float [0,1] NCHW; ORT quantize_static QDQ per_channel "
                          "QUInt8 act / QInt8 weights",
           "candidates": {m: {k: v for k, v in c["val_drift"].items()} for m, c in cand.items()},
           "val_drift": cand[pick]["val_drift"], "test_drift": export.compare_outputs(o32["test"], ot),
           "bal_acc": {"val": {"fp32_onnx": bal(o32["val"], yv), "int8": bal(ov, yv)},
                       "test": {"fp32_onnx": bal(o32["test"], yt), "int8": bal(ot, yt)}},
           "thresholds": {"torch_amp_val": thr_old, "int8_recalibrated": thr_new, "budget": S.REVIEW_BUDGET},
           "flag_rate_int8": {"val_torch_thr": flag_rate(ov, thr_old), "val_recal_thr": flag_rate(ov, thr_new),
                              "test_torch_thr": flag_rate(ot, thr_old), "test_recal_thr": flag_rate(ot, thr_new)},
           "note": "label/review agreement and balanced accuracy are on the same tiles (val = held-out train groups; test "
                   "= CRC-VAL-HE-7K, evaluated after preregistration.json). fp32_onnx = exported fp32 graph, torch AMP "
                   "numbers live in metrics.json."}
    bundle = export.build_bundle(a.pair, q, {"A": np.load(ROOT / "data" / "emb" / a.dataset / "bank_pathgenclip.npy"),
                                             "B": np.load(ROOT / "data" / "emb" / a.dataset / "bank_quiltnet_b16.npy")}
                                 if a.pair.upper() == "OPEN" else _cm_banks(a.dataset), PR, classes, thr_new, out)
    rep["bundle_bytes"] = {p.name: p.stat().st_size for p in sorted(bundle.iterdir())}
    rep["bundle_total_bytes"] = sum(rep["bundle_bytes"].values())
    rep["verify_bundle"] = export.verify_bundle(bundle)
    (sdir / "int8_drift.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: rep[k] for k in ("method_picked", "bal_acc", "thresholds", "flag_rate_int8", "bundle_total_bytes",
                                         "verify_bundle")}, indent=2))


def _cm_banks(dataset):
    return {"A": np.load(ROOT / "data" / "emb" / dataset / "bank_conch.npy"),
            "B": np.load(ROOT / "data" / "emb" / dataset / "bank_musk.npy")}


if __name__ == "__main__":
    main()
