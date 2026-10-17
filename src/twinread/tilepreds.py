"""Per-tile test predictions and triage curves, reproduced from frozen train/val fits (POST-HOC, nothing re-selected).

`run_pair_analysis` stores summary numbers (review.json) but not the per-tile readers' probabilities. This module refits
exactly what the analysis fitted on train/val (same probes, C values from preregistration.json, same fusion primary,
same s4), then checks that the refit reproduces the stored numbers before writing anything:

- every signal's errors-caught at 5/10/20% and AURC in review.json;
- the teacher review label (`review_test`) and fused logits in student_targets.npz (when present; it is git-ignored).

Outputs (in the pair/dataset results dir):
- `tile_preds.npz` (git-ignored): test pA, pB, pF, the review signals, the teacher flag, labels;
- `triage_curves.json` (committed): errors caught vs review fraction for each signal, plus the verification block.

Used by `twinread.figures` (triage curves, blind-spot gallery) and `twinread.app_host` (tile viewer).
Research prototype, not a diagnostic device.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from twinread import review

ROOT = Path(__file__).resolve().parents[2]
SIGNALS = ("s1_disagree", "s2_js", "s3_fused_unc", "s4", "base_A", "base_B")
GRID = tuple(round(i / 200, 3) for i in range(0, 201))  # 0, 0.005, ..., 1.0


def caught_curve(score, correct, fractions=GRID) -> list[float]:
    """Fraction of errors among the top-k scores, k = round(f*n): the same ordering rule as review.errors_caught."""
    score, correct = np.asarray(score, np.float64), np.asarray(correct, bool)
    n, n_err = len(score), int((~correct).sum())
    if n_err == 0:
        return [float("nan")] * len(fractions)
    cum = np.concatenate([[0], np.cumsum(~correct[np.argsort(-score, kind="stable")])])
    return [float(cum[int(round(f * n))] / n_err) for f in fractions]


def _check(name, got, want, tol, report):
    d = float(np.max(np.abs(np.asarray(got, np.float64) - np.asarray(want, np.float64))))
    report[name] = {"max_abs_diff": d, "tol": tol, "ok": bool(d <= tol)}


def reproduce(pair: str, dataset: str, results_dir: Path, data_dir: Path | None = None, epochs: int = 30) -> dict:
    """Refit the train/val models of one pair/dataset, verify against stored results, write tile_preds + curves."""
    import torch
    from twinread import fusion, probes
    from twinread.analyze import _emb, _labels

    torch.set_num_threads(4)
    data_dir = Path(data_dir) if data_dir else ROOT / "data"
    out = Path(results_dir) / pair / dataset
    pre = json.loads((out / "preregistration.json").read_text())
    rv = json.loads((out / "review.json").read_text())
    tgt_p = out / "student_targets.npz"  # git-ignored: checked when present (a clean clone has only the JSON)
    tgt = np.load(tgt_p) if tgt_p.exists() else None
    names, prim = pre["readers"], pre["primary_fusion"]
    y = {s: _labels(data_dir, dataset, s) for s in ("train", "val", "test")}
    Z = {(r, s): _emb(data_dir, dataset, names[r], "probe", s) for r in "AB" for s in ("train", "val", "test")}
    pr = {r: probes.fit_probe(Z[r, "train"], y["train"], Z[r, "val"], y["val"], Cs=(pre["probe_C"][r],)) for r in "AB"}
    zt = lambda r: Z[r, "train"]
    if prim == "concat_lr":
        m = fusion.fit_concat_lr(zt("A"), zt("B"), y["train"], Z["A", "val"], Z["B", "val"], y["val"])
    elif prim == "gated_mlp":
        m = fusion.fit_gated_mlp(zt("A"), zt("B"), y["train"], Z["A", "val"], Z["B", "val"], y["val"], epochs=epochs)

    def pred(s):
        pA, pB = pr["A"].predict_proba(Z["A", s]), pr["B"].predict_proba(Z["B", s])
        pF = fusion.fuse_average(pA, pB) if prim == "average" else m.predict_proba(Z["A", s], Z["B", s])
        return pA, pB, pF

    vA, vB, vF = pred("val")
    s4 = review.fit_s4(review.signals(vA, vB, vF), vF.argmax(1) != y["val"])
    tA, tB, tF = pred("test")
    sig = review.signals(tA, tB, tF)
    sig["s4"] = s4(sig)
    thr10 = pre["s4_val_thresholds"]["0.1"]
    flag = (sig["s4"] > thr10).astype(np.int8)
    corr = {"F": tF.argmax(1) == y["test"], "A": tA.argmax(1) == y["test"], "B": tB.argmax(1) == y["test"]}

    ver = {}
    if tgt is not None:
        _check("fused_logits_test", np.log(np.clip(tF, 1e-8, 1.0)).astype(np.float32), tgt["fused_logits_test"], 1e-5, ver)
        _check("review_test_label", flag, tgt["review_test"], 0, ver)
    for k in SIGNALS:
        got = review.errors_caught(sig[k], corr["F"])
        for b in review.BUDGETS:
            _check(f"{k}_caught_{b}", got[b]["errors_caught"], rv["scores"][k]["caught"][str(b)]["errors_caught"], 1e-9, ver)
        _check(f"{k}_aurc", review.risk_coverage(sig[k], corr["F"])["aurc"], rv["scores"][k]["aurc"], 1e-9, ver)
    for k, r in (("base_A", "A"), ("base_B", "B")):
        got = review.errors_caught(sig[k], corr[r])
        for b in review.BUDGETS:
            _check(f"{k}_own_error_caught_{b}", got[b]["errors_caught"],
                   rv["scores"][k + "_own_error"]["caught"][str(b)]["errors_caught"], 1e-9, ver)
    ok = all(v["ok"] for v in ver.values())
    if not ok:
        bad = {k: v for k, v in ver.items() if not v["ok"]}
        raise AssertionError(f"refit does not reproduce stored results for {pair}/{dataset}: {bad}")

    curves = {k: {"target": "fused-reader errors", "errors_caught": caught_curve(sig[k], corr["F"])} for k in SIGNALS}
    for k, r in (("base_A", "A"), ("base_B", "B")):
        curves[k + "_own_error"] = {"target": f"reader {r}'s own errors", "errors_caught": caught_curve(sig[k], corr[r])}
    classes = pre["classes"]
    pairs = {}
    for i in rv["confident_shared_errors"]["indices"]:
        k = f"{classes[int(y['test'][i])]}->{classes[int(tA[i].argmax())]}"
        pairs[k] = pairs.get(k, 0) + 1
    doc = {"post_hoc": True,
           "note": "POST-HOC reproduction of the frozen train/val fits (probe C from preregistration.json); every "
                   "errors-caught/AURC value in review.json and the stored review label are reproduced exactly "
                   "(see verification). Nothing is selected from test. Research prototype, not a diagnostic device.",
           "pair": pair, "dataset": dataset, "readers": names, "primary_fusion": prim, "n_test": int(len(y["test"])),
           "ci_mode": rv["ci_mode"], "review_fraction": list(GRID), "curves": curves,
           "prereg_operating_point_10": rv["s4_val_threshold_operating_points"]["0.1"],
           "confident_shared_error_pairs": dict(sorted(pairs.items(), key=lambda kv: -kv[1])),
           "confident_shared_errors_flagged_by_prereg_rule": int(flag[rv["confident_shared_errors"]["indices"]].sum()),
           "verification": {"all_ok": ok, "checks": ver}}
    tc = out / "triage_curves.json"
    if tgt is None and tc.exists():
        # Clean clone: the npz checks cannot run, so keep the committed file (whose verification includes them) as
        # long as everything recomputed here matches it; any difference is an error, never a silent rewrite.
        old = json.loads(tc.read_text())
        same = all(old.get(k) == json.loads(json.dumps(v)) for k, v in doc.items() if k != "verification")
        if not same:
            raise AssertionError(f"{tc} differs from the reproduction and student_targets.npz is absent")
        doc = old
    else:
        if tgt is None:
            doc["verification"]["npz_checks"] = "skipped: student_targets.npz absent (git-ignored)"
        tc.write_text(json.dumps(doc, indent=1))
    np.savez_compressed(out / "tile_preds.npz", pA=tA.astype(np.float32), pB=tB.astype(np.float32),
                        pF=tF.astype(np.float32), y=y["test"].astype(np.int64), flag=flag,
                        **{k: sig[k].astype(np.float32) for k in SIGNALS})
    return doc
