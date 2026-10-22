"""Score the UNO Q test-set outputs with the repo's own metrics and compare them tile by tile with the x86 run.

    $BENCH/.venv/bin/python scripts/uno_q/score_onboard.py --repo . --runs results/uno_q/onboard/raw \
        --out results/uno_q/onboard/testset_score.json

Metrics come from scripts/int8_flag_fidelity.py (flag_fidelity, twinread.stats). Research prototype, not a diagnostic
device.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

PAIRS = {"OPEN": ("open", "results/open/OPEN/crc", "results/student/open_crc"),
         "CM": ("cm", "results/cm_local/CM/crc", "results/student/cm_crc")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True, help="checkout holding data/, results/, bundles/ (read only)")
    ap.add_argument("--code", type=Path, help="checkout whose scripts/ and src/ to import (default: --repo)")
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("iff", (a.code or a.repo) / "scripts" / "int8_flag_fidelity.py")
    iff = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(iff)  # sets sys.path to src/ and device/, imports twinread.stats etc.
    y = np.load(a.repo / "data" / "crc_test_y.npy")
    report = {"split": "CRC-VAL-HE-7K test (7,180 tiles)", "pairs": {}}
    for pair, (tag, targets, student) in PAIRS.items():
        b = np.load(a.runs / f"board_{tag}.npz")
        h = np.load(a.runs / f"host_{tag}.npz")
        meta = json.loads((a.runs / f"board_{tag}.json").read_text())
        with np.load(a.repo / targets / "student_targets.npz") as z:
            teacher, y_t = z["review_test"].astype(bool), z["y_test"]
        assert np.array_equal(y_t, y)
        committed = json.loads((a.repo / student / "int8_flag_fidelity.json").read_text())
        bundle = a.repo / "bundles" / tag
        board_fid = iff.flag_fidelity(b["review"], teacher, b["pred"], y)
        bal_board = iff.stats.balanced_accuracy(y, b["pred"])
        bal_host = iff.stats.balanced_accuracy(y, h["pred"])
        report["pairs"][pair] = {
            "bundle_sha256sums_sha256": hashlib.sha256((bundle / "SHA256SUMS").read_bytes()).hexdigest(),
            "board_run": {k: meta[k] for k in ("n", "threads", "batch", "seconds", "ms_per_tile", "tiles_per_s")},
            "board_host": meta["host"],
            "board_temps_c": {"before_max": max(meta["temps_c_before"].values()),
                              "after_max": max(meta["temps_c_after"].values())},
            "int8_bal_acc_board": bal_board,
            "int8_bal_acc_host_x86_rerun": bal_host,
            "int8_bal_acc_committed_x86": committed["checks"]["int8_bal_acc"],
            "board_device_flag": board_fid,
            "committed_x86_device_flag": committed["int8_device_flag"],
            "board_vs_x86": {
                "pred_agreement": float((b["pred"] == h["pred"]).mean()),
                "pred_mismatches": int((b["pred"] != h["pred"]).sum()),
                "review_agreement": float((b["review"] == h["review"]).mean()),
                "review_mismatches": int((b["review"] != h["review"]).sum()),
                "reasons_mismatched_tiles": int((b["reasons"] != h["reasons"]).any(1).sum()),
                "max_abs_diff_probsF": float(np.abs(b["probsF"] - h["probsF"]).max()),
                "max_abs_diff_probsA": float(np.abs(b["probsA"] - h["probsA"]).max()),
                "max_abs_diff_probsB": float(np.abs(b["probsB"] - h["probsB"]).max()),
                "max_abs_diff_review_score": float(np.abs(b["review_score"] - h["review_score"]).max()),
            },
            "checks": {
                "board_bal_acc_matches_committed": abs(bal_board - committed["checks"]["int8_bal_acc"]) < 1e-9,
                "board_kappa_matches_committed":
                    abs(board_fid["kappa"] - committed["int8_device_flag"]["kappa"]) < 1e-9,
            },
        }
    report["note"] = ("Board = Arduino UNO Q (aarch64, onnxruntime CPU) running device/twinread_runtime.EdgeModel; "
                      "host = x86_64 rerun of the same bundle and loop. Research prototype, not a diagnostic device.")
    a.out.write_text(json.dumps(report, indent=1))
    for pair, r in report["pairs"].items():
        print(pair, {"bal_acc_board": round(r["int8_bal_acc_board"], 4),
                     "bal_acc_committed": round(r["int8_bal_acc_committed_x86"], 4),
                     "kappa_board": round(r["board_device_flag"]["kappa"], 4),
                     "ms_per_tile": r["board_run"]["ms_per_tile"]}, r["board_vs_x86"], r["checks"])


if __name__ == "__main__":
    main()
