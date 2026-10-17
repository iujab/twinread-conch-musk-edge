"""Metrics and paired statistics."""
from __future__ import annotations

import numpy as np
from scipy.stats import binom
from sklearn.metrics import balanced_accuracy_score, f1_score


def balanced_accuracy(y, pred) -> float:
    return float(balanced_accuracy_score(y, pred))


def macro_f1(y, pred) -> float:
    return float(f1_score(y, pred, average="macro"))


def bootstrap_ci(fn, *arrays, n=1000, seed=0, alpha=0.05, groups=None):
    """Percentile bootstrap; fn(*resampled arrays) -> float.

    With ``groups`` (e.g. slide ids) whole clusters are resampled with replacement, which widens the interval when
    tiles of a slide are correlated; without it rows are resampled (tile-level, anti-conservative on clustered data).
    """
    arrays = [np.asarray(a) for a in arrays]
    m = len(arrays[0])
    rng = np.random.default_rng(seed)
    if groups is not None:
        _, inv = np.unique(np.asarray(groups), return_inverse=True)
        members = [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]
    vals = np.empty(n)
    for i in range(n):
        if groups is None:
            idx = rng.integers(0, m, m)
        else:
            idx = np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
        vals[i] = fn(*[a[idx] for a in arrays])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def mcnemar_exact(correct_a, correct_b) -> dict:
    """b = A right & B wrong, c = A wrong & B right; two-sided exact binomial."""
    a = np.asarray(correct_a, bool)
    b_ = np.asarray(correct_b, bool)
    b = int((a & ~b_).sum())
    c = int((~a & b_).sum())
    n = b + c
    p = 1.0 if n == 0 else float(min(1.0, 2 * binom.cdf(min(b, c), n, 0.5)))
    return {"b": b, "c": c, "p": p}
