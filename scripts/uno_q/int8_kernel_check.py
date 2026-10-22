"""Is the board/x86 int8 mismatch caused by x86 U8S8 saturation? Compare board, x86 (default ORT) and x86 with graph
optimisations off (QDQ run as DequantizeLinear -> fp32 op -> QuantizeLinear: no integer kernels, no saturation), each
against the others and against the fp32 student's saved test outputs.

    OMP_NUM_THREADS=4 $BENCH/.venv/bin/python scripts/uno_q/int8_kernel_check.py --repo . \
        --runs results/uno_q/onboard/raw --out results/uno_q/onboard/int8_kernel_check.json

Research prototype, not a diagnostic device.
"""
import argparse
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np


def run_noopt(repo, tag, x, threads=4, batch=64):
    sys.path.insert(0, str(repo / "device"))
    import onnxruntime as ort
    import twinread_runtime as rt
    m = rt.EdgeModel(repo / "bundles" / tag, threads=threads)
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    m.sess = ort.InferenceSession(str(repo / "bundles" / tag / "model.onnx"), so, providers=["CPUExecutionProvider"])
    res = [r for s in range(0, len(x), batch) for r in m.infer(np.ascontiguousarray(x[s:s + batch]))]
    cls = m.classes
    return {"pred": np.array([cls.index(r.label) for r in res]), "review": np.array([r.review for r in res]),
            "probsF": np.array([r.probsF for r in res])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    x = np.load(a.repo / "data" / "crc_test_x.npy", mmap_mode="r")
    y = np.load(a.repo / "data" / "crc_test_y.npy")
    from sklearn.metrics import balanced_accuracy_score as bacc
    import onnxruntime as ort
    cpu = Path("/proc/cpuinfo").read_text()
    flags = set(re.search(r"^flags\s*:\s*(.*)$", cpu, re.M).group(1).split())
    rep = {"host": {"machine": platform.machine(), "cpu": re.search(r"^model name\s*:\s*(.*)$", cpu, re.M).group(1),
                    "avx2": "avx2" in flags, "avx512_vnni": "avx512_vnni" in flags, "avx_vnni": "avx_vnni" in flags,
                    "onnxruntime": ort.__version__}}
    for tag, student in (("open", "open_crc"), ("cm", "cm_crc")):
        b, h = np.load(a.runs / f"board_{tag}.npz"), np.load(a.runs / f"host_{tag}.npz")
        n = run_noopt(a.repo, tag, x)
        np.savez_compressed(a.runs / f"host_noopt_{tag}.npz", **n)
        fp32 = np.load(a.repo / "results" / "student" / student / "outputs.npz")["test_logits"].astype(np.float32)
        fp32_pred = fp32.argmax(1)
        runs = {"board": (b["pred"], b["probsF"], b["review"]), "x86": (h["pred"], h["probsF"], h["review"]),
                "x86_noopt": (n["pred"], n["probsF"], n["review"])}
        r = {"bal_acc": {k: float(bacc(y, v[0])) for k, v in runs.items()},
             "fp32_student_bal_acc": float(bacc(y, fp32_pred)),
             "pred_agreement_with_fp32_student": {k: float((v[0] == fp32_pred).mean()) for k, v in runs.items()}}
        for p, q in (("board", "x86"), ("board", "x86_noopt"), ("x86", "x86_noopt")):
            r[f"{p}_vs_{q}"] = {"pred_mismatches": int((runs[p][0] != runs[q][0]).sum()),
                                "review_mismatches": int((runs[p][2] != runs[q][2]).sum()),
                                "max_abs_diff_probsF": float(np.abs(runs[p][1] - runs[q][1]).max()),
                                "mean_abs_diff_probsF": float(np.abs(runs[p][1] - runs[q][1]).mean())}
        rep[tag] = r
        print(tag, json.dumps(r), flush=True)
    a.out.write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
