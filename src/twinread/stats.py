"""Metrics and paired statistics."""
from __future__ import annotations

import numpy as np
from scipy.stats import binom
from sklearn.metrics import balanced_accuracy_score, f1_score


def balanced_accuracy(y, pred) -> float:
    return float(balanced_accuracy_score(y, pred))


def macro_f1(y, pred) -> float:
    return float(f1_score(y, pred, average="macro"))


def bootstrap_ci(fn, *arrays, n=1000, seed=0, alpha=0.05):
    """Percentile bootstrap over rows; fn(*resampled arrays) -> float."""
    arrays = [np.asarray(a) for a in arrays]
    m = len(arrays[0])
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
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
