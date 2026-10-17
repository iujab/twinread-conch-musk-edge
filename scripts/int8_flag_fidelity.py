"""How faithfully does the DEPLOYED int8 device flag reproduce the teacher review flag? (CPU, ~20 s per bundle)

Runs the int8 bundle through device/twinread_runtime.EdgeModel (exactly what the device app runs: onnxruntime CPU,
the bundle's int8-recalibrated thresholds, the device review rule incl. the no-tissue clause) on the full
CRC-VAL-HE-7K test set and compares its flag with the teacher flag (the pre-registered s4 rule at its val threshold,
review_test in student_targets.npz). Test metrics are computed only because preregistration.json exists next to the
targets; nothing is fitted or selected here.

    OMP_NUM_THREADS=4 PYTHONPATH=src .venv/bin/python scripts/int8_flag_fidelity.py --pair CM
    OMP_NUM_THREADS=4 PYTHONPATH=src .venv/bin/python scripts/int8_flag_fidelity.py --pair OPEN

Writes results/student/<pair>_crc/int8_flag_fidelity.json. Research prototype, not a diagnostic device.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "device"))
from twinread import stats  # noqa: E402
from twinread.data import CRC_CLASSES, load_split  # noqa: E402
from twinread.student import _kappa  # noqa: E402

DEFAULTS = {"CM": {"bundle": "bundles/cm", "targets": "results/cm_local/CM/crc", "student": "results/student/cm_crc"},
            "OPEN": {"bundle": "bundles/open", "targets": "results/open/OPEN/crc", "student": "results/student/open_crc"}}


def flag_fidelity(flag: np.ndarray, teacher: np.ndarray, pred: np.ndarray, y: np.ndarray) -> dict:
    """Agreement of a device flag with the teacher flag, plus how many of the device model's own errors it catches."""
    f, t = np.asarray(flag, bool), np.asarray(teacher, bool)
    err = np.asarray(pred) != np.asarray(y)
    return {"n": int(len(f)), "fraction": float(f.mean()), "teacher_fraction": float(t.mean()),
            "agreement": float((f == t).mean()), "kappa": _kappa(f, t),
            "teacher_recall": float(f[t].mean()) if t.any() else None,
            "precision_vs_teacher": float(t[f].mean()) if f.any() else None,
            "error_rate": float(err.mean()), "errors_caught": float(f[err].mean()) if err.any() else None,
            "acc_cleared": float((~err[~f]).mean()) if (~f).any() else None}


def run(pair: str, bundle: Path, targets: Path, student_dir: Path, data: Path, threads: int = 4,
        batch: int = 64) -> dict:
    import twinread_runtime as rt
    if not (targets / "preregistration.json").exists():
        raise SystemExit(f"no preregistration.json in {targets}: test metrics are not computed before it exists")
    x, y = load_split(data, "crc", "test")
    with np.load(targets / "student_targets.npz") as z:
        teacher, y_t = z["review_test"].astype(bool), z["y_test"]
    if not np.array_equal(y_t, y):
        raise SystemExit("student_targets.npz y_test does not match data/crc_test_y.npy")
    m = rt.EdgeModel(bundle, threads=threads)
    t0 = time.time()
    res = [r for s in range(0, len(x), batch) for r in m.infer(x[s:s + batch])]
    secs = time.time() - t0
    flag = np.array([r.review for r in res])
    pred = np.array([CRC_CLASSES.index(r.label) for r in res])
    reasons = {k: float(np.mean([k in r.reasons for r in res])) for k in ("disagree", "js", "review_head", "no_tissue")}
    no_nt = np.array([bool(set(r.reasons) - {"no_tissue"}) for r in res])
    drift = json.loads((student_dir / "int8_drift.json").read_text())
    fp32 = json.loads((student_dir / "metrics.json").read_text())["test"]["clean"]["device_flag"]
    bal = stats.balanced_accuracy(y, pred)
    return {
        "pair": pair, "dataset": "crc", "split": "test (CRC-VAL-HE-7K, evaluated after preregistration.json)",
        "bundle": str(bundle.relative_to(ROOT)) if bundle.is_relative_to(ROOT) else str(bundle),
        "bundle_sha256sums_sha256": hashlib.sha256((bundle / "SHA256SUMS").read_bytes()).hexdigest(),
        "thresholds": m.thr, "runtime": "device/twinread_runtime.EdgeModel, onnxruntime CPU",
        "threads": threads, "host": platform.machine(), "seconds": round(secs, 1),
        "device_flag_definition": "argmax pA != argmax pB OR JS(pA,pB) > thr.js OR sigmoid(review_logit) > thr.review "
                                  "OR no_tissue (the deployed device rule; pA/pB = the two student heads through the "
                                  "aligned probes, on ONE shared student backbone)",
        "teacher_flag_definition": "pre-registered s4 rule of the teacher pair at its val threshold (10% budget), "
                                   "student_targets.npz review_test",
        "int8_device_flag": flag_fidelity(flag, teacher, pred, y),
        "int8_device_flag_without_no_tissue_clause": flag_fidelity(no_nt, teacher, pred, y),
        "int8_clause_fractions": reasons,
        "fp32_device_flag_from_metrics_json": {k: fp32[k] for k in ("fraction", "agreement", "kappa", "teacher_recall",
                                                                     "precision_vs_teacher", "student_errors_caught")},
        "checks": {"int8_bal_acc": bal, "int8_bal_acc_in_int8_drift_json": drift["bal_acc"]["test"]["int8"],
                   "bal_acc_matches_int8_drift": abs(bal - drift["bal_acc"]["test"]["int8"]) < 1e-9,
                   "flag_rate_without_no_tissue": float(no_nt.mean()),
                   "flag_rate_in_int8_drift_json": drift["flag_rate_int8"]["test_recal_thr"],
                   "flag_rate_matches_int8_drift": abs(float(no_nt.mean())
                                                       - drift["flag_rate_int8"]["test_recal_thr"]) < 1e-9},
        "note": "Research prototype, not a diagnostic device.",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pair", required=True, type=str.upper, choices=list(DEFAULTS))
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--targets-dir", type=Path)
    ap.add_argument("--student", type=Path)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    d = DEFAULTS[a.pair]
    student_dir = a.student or ROOT / d["student"]
    rep = run(a.pair, a.bundle or ROOT / d["bundle"], a.targets_dir or ROOT / d["targets"], student_dir, a.data_dir,
              a.threads)
    out = a.out or student_dir / "int8_flag_fidelity.json"
    out.write_text(json.dumps(rep, indent=1))
    f = rep["int8_device_flag"]
    print(out, {k: round(f[k], 4) for k in ("fraction", "kappa", "teacher_recall", "agreement")}, rep["checks"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
