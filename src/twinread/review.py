"""Review-flag signals, the s4 combiner, risk-coverage and budgeted error capture."""
from __future__ import annotations

import numpy as np
from scipy.special import xlogy
from sklearn.linear_model import LogisticRegression

BUDGETS = (0.05, 0.10, 0.20)
S4_KEYS = ("s1_disagree", "s2_js", "s3_fused_unc")


def js_divergence(p, q) -> np.ndarray:
    """Jensen-Shannon divergence, base 2, per row (0..1)."""
    p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
    m = (p + q) / 2
    kl = lambda a: (xlogy(a, a) - xlogy(a, m)).sum(1)
    return np.clip((kl(p) + kl(q)) / (2 * np.log(2)), 0.0, 1.0)


def signals(pA, pB, pF) -> dict:
    return {"s1_disagree": (np.argmax(pA, 1) != np.argmax(pB, 1)).astype(np.float64),
            "s2_js": js_divergence(pA, pB),
            "s3_fused_unc": 1.0 - np.max(pF, 1),
            "base_A": 1.0 - np.max(pA, 1),
            "base_B": 1.0 - np.max(pB, 1)}


def fit_s4(sig_val: dict, err_val: np.ndarray):
    X = np.stack([sig_val[k] for k in S4_KEYS], 1)
    err = np.asarray(err_val).astype(int)
    if len(np.unique(err)) < 2:  # degenerate: no variation; fall back to the fused uncertainty
        return lambda sig: np.asarray(sig["s3_fused_unc"], np.float64)
    lr = LogisticRegression(C=1e4, max_iter=2000).fit(X, err)
    return lambda sig: lr.predict_proba(np.stack([sig[k] for k in S4_KEYS], 1))[:, 1]


def risk_coverage(score, correct) -> dict:
    """Keep the most confident (lowest score) first; risk = error rate among kept. AURC = mean risk over coverage."""
    score, correct = np.asarray(score, np.float64), np.asarray(correct, bool)
    order = np.argsort(score, kind="stable")
    err = (~correct[order]).astype(np.float64)
    n = len(err)
    k = np.arange(1, n + 1)
    risk = np.cumsum(err) / k
    return {"coverage": k / n, "risk": risk, "aurc": float(risk.mean())}


def errors_caught(score, correct, budgets=BUDGETS) -> dict:
    score, correct = np.asarray(score, np.float64), np.asarray(correct, bool)
    n = len(score)
    order = np.argsort(-score, kind="stable")  # highest score first = flagged first
    n_err = int((~correct).sum())
    out = {}
    for b in budgets:
        k = int(round(b * n))
        flagged = order[:k]
        cleared = order[k:]
        out[b] = {"errors_caught": float((~correct[flagged]).sum() / n_err) if n_err else float("nan"),
                  "acc_cleared": float(correct[cleared].mean()) if len(cleared) else float("nan"),
                  "n_flagged": k}
    return out


def threshold_for_budget(score_val, budget) -> float:
    """Flag score > threshold; chosen so that about `budget` of val is flagged."""
    s = np.sort(np.asarray(score_val, np.float64))
    k = int(round(budget * len(s)))
    if k <= 0:
        return float(s[-1])
    return float(s[-k - 1]) if k < len(s) else float(s[0] - 1e-12)
