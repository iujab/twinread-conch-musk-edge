"""Task 8 step 2: choose the OPEN partner for PathGen-CLIP on VAL ONLY (no test label or test embedding is read).

Rule: among quiltnet_b16 / biomedclip, keep candidates whose val balanced accuracy is within 3 points of the better
candidate, then take the lowest phi-correlation of val probe-error indicators with PathGen-CLIP's. One partner is
chosen for both datasets, so the criterion uses the mean over {crc, cam} of both quantities (per-dataset values are
recorded too).
"""
import json, sys
from pathlib import Path
import numpy as np
from twinread import analyze, probes, stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CANDS = ["quiltnet_b16", "biomedclip"]
ANCHOR = "pathgenclip"


def val_probe(ds, reader):
    emb = lambda s: np.load(DATA / "emb" / ds / f"{reader}_probe_{s}.npy").astype(np.float32)
    ytr, yva = (analyze._labels(DATA, ds, s) for s in ("train", "val"))
    p = probes.fit_probe(emb("train"), ytr, emb("val"), yva)
    pred = p.predict_proba(emb("val")).argmax(1)
    return pred, yva


def phi(a, b):
    a, b = a.astype(float), b.astype(float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


out = {"rule": __doc__.strip(), "anchor": ANCHOR, "per_dataset": {}, "candidates": {}}
for ds in ("crc", "cam"):
    pa, y = val_probe(ds, ANCHOR)
    out["per_dataset"][ds] = {"anchor_val_bal_acc": stats.balanced_accuracy(y, pa)}
    for c in CANDS:
        pc, y2 = val_probe(ds, c)
        assert (y == y2).all()
        out["per_dataset"][ds][c] = {"val_bal_acc": stats.balanced_accuracy(y, pc),
                                     "error_phi_with_anchor": phi(pa != y, pc != y)}
for c in CANDS:
    out["candidates"][c] = {k: float(np.mean([out["per_dataset"][d][c][k] for d in ("crc", "cam")]))
                            for k in ("val_bal_acc", "error_phi_with_anchor")}
best = max(v["val_bal_acc"] for v in out["candidates"].values())
elig = [c for c in CANDS if out["candidates"][c]["val_bal_acc"] >= best - 0.03]
out["eligible"] = elig
out["partner"] = min(elig, key=lambda c: out["candidates"][c]["error_phi_with_anchor"])
Path(sys.argv[1]).write_text(json.dumps(out, indent=2))
print(json.dumps({k: out[k] for k in ("candidates", "eligible", "partner")}, indent=2))
