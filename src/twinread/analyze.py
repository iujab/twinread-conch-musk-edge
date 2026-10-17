"""Pair analysis: fit on train/val -> pre-register -> evaluate test exactly once -> write results.

Output (results_dir/<pair>/<dataset>/): readers.json, fusion.json, review.json, blindspots.csv,
preregistration.json, student_targets.npz, probes_aligned.npz.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from twinread import blindspots, fusion, probes, review, stats
from twinread.embed import PAIRS
from twinread.prompts import PROMPTS_DIR, load_prompts, text_bank, zeroshot_probs

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = ("average", "concat_lr", "gated_mlp")
SPLITS = ("train", "val", "test")
CI = {"method": "percentile bootstrap over test tiles", "n": 1000, "alpha": 0.05, "seed": 0}


# ---------------------------------------------------------------------------------------------------------------
def prompts_sha256(prompts_dir: Path | None = None) -> dict:
    d = Path(prompts_dir) if prompts_dir else PROMPTS_DIR
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("*.json"))}


def _out_dir(results_dir, pair, dataset) -> Path:
    return Path(results_dir) / pair / dataset


def write_preregistration(pair, dataset, results_dir, extra: dict | None = None, prompts_dir=None) -> Path:
    """Freeze the analysis plan. Refuses to overwrite (FileExistsError)."""
    path = _out_dir(results_dir, pair, dataset) / "preregistration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"pair": pair, "dataset": dataset,
           "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "signals": ["s1_disagree", "s2_js", "s3_fused_unc", "s4 (logistic on s1..s3, fitted on val)"],
           "baselines": ["base_A = 1 - max pA", "base_B = 1 - max pB"],
           "budgets": list(review.BUDGETS),
           "fusion_candidates": list(CANDIDATES),
           "primary_fusion_rule": "highest validation balanced accuracy among candidates (ties: listed order)",
           "primary_flag": "s4",
           "review_label": "s4 > validation threshold at the 10% budget",
           "primary_readers": "linear probes on recommended probe features (kind 'probe'); aligned probes = sensitivity",
           "ci": CI,
           "prompts_sha256": prompts_sha256(prompts_dir)}
    doc.update(extra or {})
    with open(path, "x") as f:  # 'x' = refuse to overwrite
        json.dump(doc, f, indent=2, sort_keys=True)
    return path


# ---------------------------------------------------------------------------------------------------------------
def _labels(data_dir: Path, dataset: str, split: str) -> np.ndarray:
    from twinread.data import CAM_TEST_HOSPITAL, split_val
    if dataset == "crc":
        if split == "test":
            return np.load(data_dir / "crc_test_y.npy")
        y = np.load(data_dir / "crc_train_y.npy")
        gp = data_dir / "crc_train_g.npy"
        tr, va = split_val(y, 0.1, 0, groups=np.load(gp) if gp.exists() else None)
        return y[tr if split == "train" else va]
    y, h = np.load(data_dir / "cam_y.npy"), np.load(data_dir / "cam_h.npy")
    return y[np.isin(h, {"train": (0, 3, 4), "val": (1,), "test": (CAM_TEST_HOSPITAL,)}[split])]


def _readers(pair, partner):
    if pair not in PAIRS:
        raise KeyError(f"unknown pair {pair!r}")
    a, b = PAIRS[pair]
    b = b or partner
    if b is None:
        raise ValueError(f"pair {pair} needs an explicit partner reader")
    return a, b


def _emb(data_dir, dataset, reader, kind, split):
    return np.load(Path(data_dir) / "emb" / dataset / f"{reader}_{kind}_{split}.npy").astype(np.float32)


def _bank(data_dir, dataset, reader, banks, prompts_dir):
    if banks and reader in banks:
        return np.asarray(banks[reader])
    cache = Path(data_dir) / "emb" / dataset / f"bank_{reader}.npy"
    if cache.exists():
        return np.load(cache)
    from twinread.readers import load_reader
    r = load_reader(reader, "cpu")
    try:
        b = text_bank(r, dataset, prompts_dir)
    finally:
        r.close()
    np.save(cache, b)
    return b


def _scores(y, probs, n_boot):
    pred = probs.argmax(1)
    f = lambda a, b: stats.balanced_accuracy(a, b)
    return {"bal_acc": stats.balanced_accuracy(y, pred),
            "bal_acc_ci": stats.bootstrap_ci(f, y, pred, n=n_boot),
            "macro_f1": stats.macro_f1(y, pred), "accuracy": float((pred == y).mean())}


def _test_metrics(ctx: dict) -> dict:
    """Every test-label metric. Called exactly once, after preregistration.json exists."""
    y, classes, nb = ctx["y"], ctx["classes"], ctx["n_boot"]
    out = {"readers": {}, "fusion": {}, "review": {}}
    for r in ("A", "B"):
        d = {"name": ctx["names"][r], "probe": _scores(y, ctx["p"][r], nb),
             "aligned_probe": _scores(y, ctx["pal"][r], nb)}
        if ctx.get("zs", {}).get(r) is not None:
            d["zeroshot"] = _scores(y, ctx["zs"][r], nb)
        out["readers"][r] = d
    cand = {k: _scores(y, v, nb) for k, v in ctx["cand"].items()}
    pF = ctx["cand"][ctx["primary"]]
    corr = {"A": ctx["p"]["A"].argmax(1) == y, "B": ctx["p"]["B"].argmax(1) == y, "F": pF.argmax(1) == y}
    best = max("AB", key=lambda r: out["readers"][r]["probe"]["bal_acc"])
    pb = ctx["p"][best].argmax(1)
    yy, pf, pbest = y, pF.argmax(1), pb
    dfn = lambda yv, f_, b_: stats.balanced_accuracy(yv, f_) - stats.balanced_accuracy(yv, b_)
    out["fusion"] = {"candidates": cand, "primary": ctx["primary"], "best_single_reader": best,
                     "delta_bal_acc_vs_best_single": dfn(yy, pf, pbest),
                     "delta_ci": stats.bootstrap_ci(dfn, yy, pf, pbest, n=nb),
                     "mcnemar_fused_vs_A": stats.mcnemar_exact(corr["F"], corr["A"]),
                     "mcnemar_fused_vs_B": stats.mcnemar_exact(corr["F"], corr["B"]),
                     "mcnemar_fused_vs_best_single": stats.mcnemar_exact(corr["F"], corr[best])}
    sig = review.signals(ctx["p"]["A"], ctx["p"]["B"], pF)
    sig["s4"] = ctx["s4"](sig)
    rv = {"target": "error of the primary fused prediction", "scores": {}}
    for k in ("s1_disagree", "s2_js", "s3_fused_unc", "s4", "base_A", "base_B"):
        rc = review.risk_coverage(sig[k], corr["F"])
        caught = review.errors_caught(sig[k], corr["F"])
        fn10 = lambda s, c: review.errors_caught(s, c, (0.10,))[0.10]["errors_caught"]
        rv["scores"][k] = {"aurc": rc["aurc"], "caught": {str(b): v for b, v in caught.items()},
                           "caught_10_ci": stats.bootstrap_ci(fn10, sig[k], corr["F"], n=nb)}
    for k, r in (("base_A", "A"), ("base_B", "B")):
        rv["scores"][k + "_own_error"] = {"aurc": review.risk_coverage(sig[k], corr[r])["aurc"],
                                          "caught": {str(b): v for b, v in review.errors_caught(sig[k], corr[r]).items()}}
    ops = {}
    for b, thr in ctx["thr"].items():
        fl = sig["s4"] > thr
        ops[str(b)] = {"threshold_from_val": thr, "flagged_fraction": float(fl.mean()),
                       "errors_caught": float((~corr["F"][fl]).sum() / max((~corr["F"]).sum(), 1)),
                       "acc_cleared": float(corr["F"][~fl].mean()) if (~fl).any() else float("nan")}
    rv["s4_val_threshold_operating_points"] = ops
    rv["fused_error_rate"] = float(1 - corr["F"].mean())
    out["review"] = rv
    pa, pbb = ctx["p"]["A"], ctx["p"]["B"]
    out["blindspots"] = blindspots.blindspot_table(pa.argmax(1), pbb.argmax(1), y, classes)
    cse = blindspots.confident_shared_errors(pa, pbb, y)
    out["review"]["confident_shared_errors"] = {"n": int(len(cse)), "rate": float(len(cse) / len(y)),
                                                "indices": [int(i) for i in cse[:200]]}
    out["review"]["errors_by_flag"] = {"s1_disagree_rate": float(sig["s1_disagree"].mean())}
    out["_s4_test"] = sig["s4"]
    return out


def _dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def run_pair_analysis(pair: str, dataset: str, results_dir: Path, partner: str | None = None, data_dir=None,
                      labels: dict | None = None, banks: dict | None = None, prompts_dir=None,
                      epochs: int = 30, n_boot: int = CI["n"]) -> dict:
    data_dir = Path(data_dir) if data_dir else ROOT / "data"
    names = dict(zip("AB", _readers(pair, partner)))
    classes = load_prompts(dataset, prompts_dir)[0]
    out = _out_dir(results_dir, pair, dataset)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "preregistration.json").exists():
        raise FileExistsError(f"{out / 'preregistration.json'} exists: analysis already run")
    lab = lambda s: labels[s] if labels is not None else _labels(data_dir, dataset, s)
    y = {s: lab(s) for s in ("train", "val")}
    Z = {(r, k, s): _emb(data_dir, dataset, names[r], k, s) for r in "AB" for k in ("probe", "aligned") for s in SPLITS}

    # ---- fit on train / val only -------------------------------------------------------------------------------
    pr = {r: probes.fit_probe(Z[r, "probe", "train"], y["train"], Z[r, "probe", "val"], y["val"]) for r in "AB"}
    pal = {r: probes.fit_probe(Z[r, "aligned", "train"], y["train"], Z[r, "aligned", "val"], y["val"]) for r in "AB"}
    zp = lambda r, s: Z[r, "probe", s]
    cl = fusion.fit_concat_lr(zp("A", "train"), zp("B", "train"), y["train"], zp("A", "val"), zp("B", "val"), y["val"])
    gm = fusion.fit_gated_mlp(zp("A", "train"), zp("B", "train"), y["train"], zp("A", "val"), zp("B", "val"),
                              y["val"], epochs=epochs)

    def cands(s):
        pA, pB = pr["A"].predict_proba(zp("A", s)), pr["B"].predict_proba(zp("B", s))
        return pA, pB, {"average": fusion.fuse_average(pA, pB),
                        "concat_lr": cl.predict_proba(zp("A", s), zp("B", s)),
                        "gated_mlp": gm.predict_proba(zp("A", s), zp("B", s))}

    vA, vB, vc = cands("val")
    val_scores = {k: stats.balanced_accuracy(y["val"], v.argmax(1)) for k, v in vc.items()}
    primary = max(CANDIDATES, key=lambda k: (val_scores[k], -CANDIDATES.index(k)))
    sv = review.signals(vA, vB, vc[primary])
    s4 = review.fit_s4(sv, vc[primary].argmax(1) != y["val"])
    s4v = s4(sv)
    thr = {b: review.threshold_for_budget(s4v, b) for b in review.BUDGETS}

    # ---- freeze, then evaluate the test split exactly once -----------------------------------------------------
    write_preregistration(pair, dataset, results_dir, prompts_dir=prompts_dir, extra={
        "readers": names, "classes": list(classes), "val_balanced_accuracy": val_scores, "primary_fusion": primary,
        "probe_C": {r: pr[r].C for r in "AB"}, "aligned_probe_C": {r: pal[r].C for r in "AB"},
        "s4_val_thresholds": {str(b): t for b, t in thr.items()}})
    y["test"] = lab("test")
    tA, tB, tc = cands("test")
    ctx = {"y": y["test"], "classes": classes, "n_boot": n_boot, "names": names, "p": {"A": tA, "B": tB},
           "pal": {r: pal[r].predict_proba(Z[r, "aligned", "test"]) for r in "AB"}, "cand": tc, "primary": primary,
           "s4": s4, "thr": thr, "zs": {}}
    for r in "AB":
        bank = _bank(data_dir, dataset, names[r], banks, prompts_dir)
        ctx["zs"][r] = zeroshot_probs(Z[r, "aligned", "test"], bank)
    res = _test_metrics(ctx)

    # ---- artifacts ---------------------------------------------------------------------------------------------
    _dump(out / "readers.json", res["readers"])
    _dump(out / "fusion.json", res["fusion"] | {"val_balanced_accuracy": val_scores})
    _dump(out / "review.json", res["review"])
    res["blindspots"].to_csv(out / "blindspots.csv", index=False)
    arrs = {}
    for r in "AB":
        arrs.update(pal[r].to_arrays(r))
    np.savez(out / "probes_aligned.npz", **arrs)
    tgt = {}
    for s in SPLITS:
        pA, pB, c = cands(s)
        pF = c[primary]
        tgt[f"tA_{s}"], tgt[f"tB_{s}"] = Z["A", "aligned", s], Z["B", "aligned", s]
        tgt[f"fused_logits_{s}"] = np.log(np.clip(pF, 1e-8, 1.0)).astype(np.float32)
        sg = review.signals(pA, pB, pF)
        tgt[f"review_{s}"] = (s4(sg) > thr[0.10]).astype(np.int8)
        tgt[f"y_{s}"] = y[s].astype(np.int64)
    np.savez(out / "student_targets.npz", **tgt)
    return {"primary": primary, "val_balanced_accuracy": val_scores, "out_dir": str(out), **{k: res[k] for k in ("readers", "fusion", "review")}}
