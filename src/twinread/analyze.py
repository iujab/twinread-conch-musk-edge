"""Pair analysis: fit on train/val -> pre-register -> evaluate test exactly once -> write results.

Output (results_dir/<pair>/<dataset>/): readers.json, fusion.json, review.json, blindspots.csv,
preregistration.json, student_targets.npz, probes_aligned.npz.
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

from twinread import blindspots, fusion, probes, review, stats
from twinread.embed import PAIRS
from twinread.prompts import PROMPTS_DIR, load_prompts, text_bank, zeroshot_probs
from twinread.readers import load_reader, spec_of

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = ("average", "concat_lr", "gated_mlp")
SPLITS = ("train", "val", "test")
CI = {"method": "percentile bootstrap", "n": 1000, "alpha": 0.05, "seed": 0}
RESULT_FILES = ("readers.json", "fusion.json", "review.json", "blindspots.csv", "student_targets.npz",
                "probes_aligned.npz")
K_FOLDS = 5
CI_TILE = "tile-level, anti-conservative (no slide ids)"
CI_SLIDE = "cluster bootstrap by slide"


# ---------------------------------------------------------------------------------------------------------------
def prompts_sha256(prompts_dir: Path | None = None) -> dict:
    d = Path(prompts_dir) if prompts_dir else PROMPTS_DIR
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("*.json"))}


def _out_dir(results_dir, pair, dataset) -> Path:
    return Path(results_dir) / pair / dataset


def _prereg_doc(pair, dataset, prompts_dir, extra):
    doc = {"pair": pair, "dataset": dataset,
           "signals": ["s1_disagree", "s2_js", "s3_fused_unc", "s4 (logistic on s1..s3, fitted on val)"],
           "baselines": ["base_A = 1 - max pA", "base_B = 1 - max pB"],
           "budgets": list(review.BUDGETS),
           "fusion_candidates": list(CANDIDATES),
           "primary_fusion_rule": "highest validation balanced accuracy among candidates (ties: listed order)",
           "best_single_reader_rule": "highest validation probe balanced accuracy (chosen on val, not test)",
           "primary_flag": "s4",
           "review_label": "s4 > validation threshold at the 10% budget",
           "primary_readers": "linear probes on recommended probe features (kind 'probe'); aligned probes = sensitivity",
           "train_targets": f"train targets are OOF (K={K_FOLDS}); val/test use full-train fits",
           "ci": CI,
           "prompts_sha256": prompts_sha256(prompts_dir)}
    doc.update(extra or {})
    return doc


def _atomic(path: Path, writer) -> None:
    """Write via a temp file in the same directory, then rename into place."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        writer(f)
    os.replace(tmp, path)


def write_preregistration(pair, dataset, results_dir, extra: dict | None = None, prompts_dir=None,
                          resume_ok: bool = False) -> Path:
    """Freeze the analysis plan atomically. Refuses to overwrite (FileExistsError).

    ``resume_ok`` (used by run_pair_analysis after a crash) accepts an existing file only when no results exist yet and
    the newly computed plan, ignoring ``created_utc``, is identical to the stored one.
    """
    out = _out_dir(results_dir, pair, dataset)
    path = out / "preregistration.json"
    out.mkdir(parents=True, exist_ok=True)
    doc = json.loads(json.dumps(_prereg_doc(pair, dataset, prompts_dir, extra), default=str))
    if path.exists():
        stored = json.loads(path.read_text())
        stored.pop("created_utc", None)
        if resume_ok and not any((out / f).exists() for f in RESULT_FILES) and stored == doc:
            return path
        raise FileExistsError(f"{path} exists and cannot be reused (results exist or the plan differs)")
    doc["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    tmp = path.with_name("preregistration.json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True))
    try:
        os.link(tmp, path)  # atomic exclusive create: fails if another writer got there first
    finally:
        tmp.unlink()
    return path


# ---------------------------------------------------------------------------------------------------------------
def _split_idx(data_dir: Path, dataset: str, split: str) -> np.ndarray:
    """Row indices (into the saved pool) of ``split``."""
    from twinread.data import CAM_TEST_HOSPITAL, split_val
    if dataset == "crc":
        if split == "test":
            return np.arange(len(np.load(data_dir / "crc_test_y.npy")))
        y = np.load(data_dir / "crc_train_y.npy")
        gp = data_dir / "crc_train_g.npy"
        tr, va = split_val(y, 0.1, 0, groups=np.load(gp) if gp.exists() else None)
        return tr if split == "train" else va
    h = np.load(data_dir / "cam_h.npy")
    return np.flatnonzero(np.isin(h, {"train": (0, 3, 4), "val": (1,), "test": (CAM_TEST_HOSPITAL,)}[split]))


def _labels(data_dir: Path, dataset: str, split: str) -> np.ndarray:
    if dataset == "crc" and split == "test":
        return np.load(data_dir / "crc_test_y.npy")
    return np.load(data_dir / f"{dataset}_{'train_' if dataset == 'crc' else ''}y.npy")[_split_idx(data_dir, dataset, split)]


def _train_folds(data_dir: Path, dataset: str, y_tr: np.ndarray, k: int = K_FOLDS):
    """(fold id per train row, scheme string actually used).

    crc: GroupKFold by 100-row parquet group (stratified by class if the group file is absent).
    cam: StratifiedGroupKFold by slide, stratified by label (hospital x label if slide ids are absent).
    """
    n = len(y_tr)
    idx = _split_idx(data_dir, dataset, "train")
    folds = np.zeros(n, int)
    if dataset == "crc" and (data_dir / "crc_train_g.npy").exists():
        g = np.load(data_dir / "crc_train_g.npy")[idx]
        splits, scheme = GroupKFold(k).split(np.zeros(n), y_tr, g), f"crc: GroupKFold(K={k}) by parquet group"
    elif dataset == "cam" and (data_dir / "cam_slide.npy").exists():
        g = np.load(data_dir / "cam_slide.npy")[idx]
        splits = StratifiedGroupKFold(k, shuffle=True, random_state=0).split(np.zeros(n), y_tr, g)
        scheme = f"cam: StratifiedGroupKFold(K={k}) by slide, stratified by label"
    else:
        strata = (np.load(data_dir / "cam_h.npy")[idx] * 2 + y_tr) if dataset == "cam" else y_tr
        splits = StratifiedKFold(k, shuffle=True, random_state=0).split(np.zeros(n), strata)
        scheme = f"{dataset}: StratifiedKFold(K={k}) by " + ("hospital x label" if dataset == "cam" else "label") + " (no group ids)"
    for i, (_, te) in enumerate(splits):
        folds[te] = i
    return folds, scheme


def _test_groups(data_dir: Path, dataset: str):
    """Slide ids of the test tiles for cluster bootstrap, or None."""
    p = data_dir / "cam_slide.npy"
    if dataset == "cam" and p.exists():
        return np.load(p)[_split_idx(data_dir, dataset, "test")]
    return None


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
    side = cache.with_suffix(".json")
    key = {"reader": reader, "spec": dataclasses.asdict(spec_of(reader)), "dataset": dataset,
           "prompts_sha256": prompts_sha256(prompts_dir)}
    if cache.exists() and side.exists() and json.loads(side.read_text()) == json.loads(json.dumps(key)):
        return np.load(cache)
    r = load_reader(reader, "cpu")
    try:
        b = text_bank(r, dataset, prompts_dir)
    finally:
        r.close()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, b)
    side.write_text(json.dumps(key, indent=2))
    return b


def _scores(y, probs, n_boot, groups=None):
    pred = probs.argmax(1)
    f = lambda a, b: stats.balanced_accuracy(a, b)
    ci = stats.bootstrap_ci_info(f, y, pred, n=n_boot, groups=groups)
    return {"bal_acc": stats.balanced_accuracy(y, pred),
            "bal_acc_ci": [ci["lo"], ci["hi"]], "bal_acc_ci_info": ci,
            "macro_f1": stats.macro_f1(y, pred), "accuracy": float((pred == y).mean())}


def _test_metrics(ctx: dict) -> dict:
    """Every test-label metric. Called exactly once, after preregistration.json exists."""
    y, classes, nb, gr = ctx["y"], ctx["classes"], ctx["n_boot"], ctx.get("groups")
    mode = CI_SLIDE if gr is not None else CI_TILE
    out = {"readers": {"ci_mode": mode}, "fusion": {}, "review": {}}
    for r in ("A", "B"):
        d = {"name": ctx["names"][r], "probe": _scores(y, ctx["p"][r], nb, gr),
             "aligned_probe": _scores(y, ctx["pal"][r], nb, gr)}
        if ctx.get("zs", {}).get(r) is not None:
            d["zeroshot"] = _scores(y, ctx["zs"][r], nb, gr)
        out["readers"][r] = d
    cand = {k: _scores(y, v, nb, gr) for k, v in ctx["cand"].items()}
    pF = ctx["cand"][ctx["primary"]]
    corr = {"A": ctx["p"]["A"].argmax(1) == y, "B": ctx["p"]["B"].argmax(1) == y, "F": pF.argmax(1) == y}
    best = ctx["best"]  # chosen on val, frozen in the prereg
    pb = ctx["p"][best].argmax(1)
    yy, pf, pbest = y, pF.argmax(1), pb
    dfn = lambda yv, f_, b_: stats.balanced_accuracy(yv, f_) - stats.balanced_accuracy(yv, b_)
    out["fusion"] = {"candidates": cand, "primary": ctx["primary"], "best_single_reader": best,
                     "delta_bal_acc_vs_best_single": dfn(yy, pf, pbest),
                     "delta_ci_info": (dci := stats.bootstrap_ci_info(dfn, yy, pf, pbest, n=nb, groups=gr)),
                     "delta_ci": [dci["lo"], dci["hi"]], "ci_mode": mode,
                     "mcnemar_note": "tile-level exact test; anti-conservative when tiles cluster by slide",
                     "mcnemar_fused_vs_A": stats.mcnemar_exact(corr["F"], corr["A"]),
                     "mcnemar_fused_vs_B": stats.mcnemar_exact(corr["F"], corr["B"]),
                     "mcnemar_fused_vs_best_single": stats.mcnemar_exact(corr["F"], corr[best])}
    sig = review.signals(ctx["p"]["A"], ctx["p"]["B"], pF)
    sig["s4"] = ctx["s4"](sig)
    rv = {"target": "error of the primary fused prediction", "scores": {}, "ci_mode": mode}
    for k in ("s1_disagree", "s2_js", "s3_fused_unc", "s4", "base_A", "base_B"):
        rc = review.risk_coverage(sig[k], corr["F"])
        caught = review.errors_caught(sig[k], corr["F"])
        fn10 = lambda s, c: review.errors_caught(s, c, (0.10,))[0.10]["errors_caught"]
        cci = stats.bootstrap_ci_info(fn10, sig[k], corr["F"], n=nb, groups=gr)
        rv["scores"][k] = {"aurc": rc["aurc"], "caught": {str(b): v for b, v in caught.items()},
                           "caught_10_ci": [cci["lo"], cci["hi"]], "caught_10_ci_info": cci}
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
    txt = json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    _atomic(path, lambda f: f.write(txt.encode()))


def _oof_train(zA, zB, y, folds, primary, C, zAv, zBv, yv, epochs):
    """Out-of-fold train predictions (pA, pB, pF): each row is predicted by models that never saw it."""
    n, k = len(y), len(np.unique(y))
    pA, pB, pF = (np.zeros((n, k)) for _ in range(3))
    for f in np.unique(folds):
        te, tr = folds == f, folds != f
        ra = probes.fit_probe(zA[tr], y[tr], zAv, yv, Cs=(C["A"],))
        rb = probes.fit_probe(zB[tr], y[tr], zBv, yv, Cs=(C["B"],))
        pA[te], pB[te] = ra.predict_proba(zA[te]), rb.predict_proba(zB[te])
        if primary == "average":
            pF[te] = fusion.fuse_average(pA[te], pB[te])
        elif primary == "concat_lr":
            m = fusion.fit_concat_lr(zA[tr], zB[tr], y[tr], zAv, zBv, yv, Cs=(C["concat_lr"],))
            pF[te] = m.predict_proba(zA[te], zB[te])
        else:
            m = fusion.fit_gated_mlp(zA[tr], zB[tr], y[tr], zAv, zBv, yv, epochs=epochs)
            pF[te] = m.predict_proba(zA[te], zB[te])
    return pA, pB, pF


def run_pair_analysis(pair: str, dataset: str, results_dir: Path, partner: str | None = None, data_dir=None,
                      labels: dict | None = None, banks: dict | None = None, prompts_dir=None,
                      epochs: int = 30, n_boot: int = CI["n"], train_folds=None, test_groups="auto") -> dict:
    torch.set_num_threads(4)
    data_dir = Path(data_dir) if data_dir else ROOT / "data"
    names = dict(zip("AB", _readers(pair, partner)))
    classes = load_prompts(dataset, prompts_dir)[0]
    out = _out_dir(results_dir, pair, dataset)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / f).exists() for f in RESULT_FILES):
        raise FileExistsError(f"results already exist in {out}")
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
    val_reader = {r: stats.balanced_accuracy(y["val"], p.argmax(1)) for r, p in (("A", vA), ("B", vB))}
    best_single = "A" if val_reader["A"] >= val_reader["B"] else "B"
    sv = review.signals(vA, vB, vc[primary])
    s4 = review.fit_s4(sv, vc[primary].argmax(1) != y["val"])
    thr = {b: review.threshold_for_budget(s4(sv), b) for b in review.BUDGETS}

    # ---- everything that needs no test label, BEFORE freezing (a crash after the prereg must not strand the run) --
    if train_folds is not None:
        folds, fold_scheme = np.asarray(train_folds), "provided by caller"
    else:
        folds, fold_scheme = _train_folds(data_dir, dataset, y["train"])
    Cs = {"A": pr["A"].C, "B": pr["B"].C, "concat_lr": cl.info["C"]}
    oof = _oof_train(zp("A", "train"), zp("B", "train"), y["train"], folds, primary, Cs,
                     zp("A", "val"), zp("B", "val"), y["val"], epochs)
    tA, tB, tc = cands("test")
    pal_test = {r: pal[r].predict_proba(Z[r, "aligned", "test"]) for r in "AB"}
    zs = {r: zeroshot_probs(Z[r, "aligned", "test"], _bank(data_dir, dataset, names[r], banks, prompts_dir))
          for r in "AB"}
    groups = _test_groups(data_dir, dataset) if test_groups == "auto" else test_groups

    write_preregistration(pair, dataset, results_dir, prompts_dir=prompts_dir, resume_ok=True, extra={
        "readers": names, "classes": list(classes), "val_balanced_accuracy": val_scores, "primary_fusion": primary,
        "val_probe_balanced_accuracy": val_reader, "best_single_reader": best_single,
        "probe_C": {r: pr[r].C for r in "AB"}, "aligned_probe_C": {r: pal[r].C for r in "AB"},
        "oof_folds": {"k": K_FOLDS, "scheme": fold_scheme},
        "ci_mode": CI_SLIDE if groups is not None else CI_TILE,
        "s4_val_thresholds": {str(b): t for b, t in thr.items()}})

    # ---- test labels are read only now; metrics computed exactly once ------------------------------------------
    y["test"] = lab("test")
    ctx = {"y": y["test"], "classes": classes, "n_boot": n_boot, "names": names, "p": {"A": tA, "B": tB},
           "pal": pal_test, "cand": tc, "primary": primary, "best": best_single, "s4": s4, "thr": thr, "zs": zs,
           "groups": groups}
    res = _test_metrics(ctx)

    # ---- artifacts (atomic) ------------------------------------------------------------------------------------
    arrs = {}
    for r in "AB":
        arrs.update(pal[r].to_arrays(r))
    tgt = {}
    for s, (pA, pB, pF) in (("train", oof), ("val", (vA, vB, vc[primary])), ("test", (tA, tB, tc[primary]))):
        tgt[f"tA_{s}"], tgt[f"tB_{s}"] = Z["A", "aligned", s], Z["B", "aligned", s]
        tgt[f"fused_logits_{s}"] = np.log(np.clip(pF, 1e-8, 1.0)).astype(np.float32)
        tgt[f"review_{s}"] = (s4(review.signals(pA, pB, pF)) > thr[0.10]).astype(np.int8)
        tgt[f"y_{s}"] = y[s].astype(np.int64)
    csv = res["blindspots"].to_csv(index=False).encode()
    _atomic(out / "blindspots.csv", lambda f: f.write(csv))
    _atomic(out / "probes_aligned.npz", lambda f: np.savez(f, **arrs))
    _atomic(out / "student_targets.npz", lambda f: np.savez(f, **tgt))
    _dump(out / "readers.json", res["readers"])
    _dump(out / "fusion.json", res["fusion"] | {"val_balanced_accuracy": val_scores})
    _dump(out / "review.json", res["review"])  # last: its presence marks a complete run
    return {"primary": primary, "best_single_reader": best_single, "val_balanced_accuracy": val_scores,
            "out_dir": str(out), **{k: res[k] for k in ("readers", "fusion", "review")}}
