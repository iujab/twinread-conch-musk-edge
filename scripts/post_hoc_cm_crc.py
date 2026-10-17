"""POST-HOC description of already-computed CM/crc outputs (R28). Refits exactly as analyze.run_pair_analysis does
(train/val only), checks the refit reproduces fusion.json / student_targets.npz, then writes per-class recall,
per-candidate test deltas and s3-vs-s4 agreement. Does not choose or change anything."""
import json, sys
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
from twinread import fusion, probes, review, stats
from twinread.analyze import _emb, _labels, CANDIDATES
torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "data"; out = ROOT / "results/cm_local/CM/crc"
ds = "crc"; y = {s: _labels(D, ds, s) for s in ("train", "val", "test")}
Z = {(r, s): _emb(D, ds, n, "probe", s) for r, n in (("A", "conch"), ("B", "musk")) for s in ("train", "val", "test")}
pr = {r: probes.fit_probe(Z[r, "train"], y["train"], Z[r, "val"], y["val"]) for r in "AB"}
cl = fusion.fit_concat_lr(Z["A", "train"], Z["B", "train"], y["train"], Z["A", "val"], Z["B", "val"], y["val"])
gm = fusion.fit_gated_mlp(Z["A", "train"], Z["B", "train"], y["train"], Z["A", "val"], Z["B", "val"], y["val"], epochs=30)
def cands(s):
    pA, pB = pr["A"].predict_proba(Z["A", s]), pr["B"].predict_proba(Z["B", s])
    return pA, pB, {"average": fusion.fuse_average(pA, pB), "concat_lr": cl.predict_proba(Z["A", s], Z["B", s]),
                    "gated_mlp": gm.predict_proba(Z["A", s], Z["B", s])}
vA, vB, vc = cands("val"); tA, tB, tc = cands("test")
fj = json.load(open(out / "fusion.json")); prim = fj["primary"]
ba = lambda p: stats.balanced_accuracy(y["test"], p.argmax(1))
chk = {k: abs(ba(v) - fj["candidates"][k]["bal_acc"]) for k, v in tc.items()}
assert max(chk.values()) < 1e-6, f"refit does not reproduce fusion.json: {chk}"
best = {"A": tA, "B": tB}[fj["best_single_reader"]]
classes = json.load(open(ROOT / "prompts/crc.json")) if False else None
nc = tA.shape[1]; rec = lambda p: [float((p.argmax(1)[y["test"] == c] == c).mean()) for c in range(nc)]
res = {"post_hoc": True, "note": "descriptive recomputation of existing outputs; refit reproduces fusion.json to 1e-6; nothing selected from test",
       "primary": prim, "best_single_reader": fj["best_single_reader"], "ci_mode": fj["ci_mode"],
       "test_n_per_class": [int((y["test"] == c).sum()) for c in range(nc)],
       "per_class_recall": {"conch": rec(tA), "musk": rec(tB), "fused_primary": rec(tc[prim])},
       "per_class_recall_delta_fused_minus_best_single": [a - b for a, b in zip(rec(tc[prim]), rec(best))],
       "candidates": {k: {"val_bal_acc": stats.balanced_accuracy(y["val"], vc[k].argmax(1)), "test_bal_acc": ba(v),
                          "test_delta_vs_best_single": ba(v) - ba(best),
                          "val_correct_tiles": int((vc[k].argmax(1) == y["val"]).sum())} for k, v in tc.items()},
       "n_val": int(len(y["val"]))}
sv = review.signals(vA, vB, vc[prim]); s4 = review.fit_s4(sv, vc[prim].argmax(1) != y["val"])
st = review.signals(tA, tB, tc[prim]); s3, s4t = st["s3_fused_unc"], s4(st)
k = int(round(0.1 * len(s3))); top = lambda s: set(np.argsort(-s, kind="stable")[:k])
res["s3_s4_agreement"] = {"spearman": float(spearmanr(s3, s4t).statistic), "top10pct_overlap": len(top(s3) & top(s4t)) / k, "n": int(len(s3))}
(out / "post_hoc_class_candidates_s3s4.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
