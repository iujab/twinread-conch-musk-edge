"""Linear probes: standardize + logistic regression, C chosen on validation balanced accuracy."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


class Probe:
    def __init__(self, mean, scale, lr, C):
        self.mean, self.scale, self.lr, self.C = mean, scale, lr, C
        self.classes_ = lr.classes_

    def predict_proba(self, z) -> np.ndarray:
        return self.lr.predict_proba((np.asarray(z, np.float64) - self.mean) / self.scale)

    def softmax_params(self):
        """(coef [C,d], intercept [C]) such that softmax(((z-mean)/scale) @ coef.T + intercept) == predict_proba.

        A binary logistic regression has one coef row; it is expanded to the equivalent 2-row softmax form
        (zeros for class 0, w for class 1), so a numpy-only device can use one code path for any class count.
        """
        w, b = self.lr.coef_.astype(np.float64), self.lr.intercept_.astype(np.float64)
        if w.shape[0] == 1:
            w, b = np.vstack([np.zeros_like(w), w]), np.concatenate([[0.0], b])
        return w, b

    def to_arrays(self, prefix: str) -> dict:
        w, b = self.softmax_params()
        return {f"{prefix}_coef": w.astype(np.float32), f"{prefix}_intercept": b.astype(np.float32),
                f"{prefix}_mean": self.mean.astype(np.float32), f"{prefix}_scale": self.scale.astype(np.float32),
                f"{prefix}_classes": np.asarray(self.classes_).astype(np.int64)}


def standardizer(z):
    z = np.asarray(z, np.float64)
    return z.mean(0), np.maximum(z.std(0), 1e-6)


def fit_probe(z_tr, y_tr, z_val, y_val, Cs=(0.01, 0.1, 1.0, 10.0), seed=0) -> Probe:
    mean, scale = standardizer(z_tr)
    xtr = (np.asarray(z_tr, np.float64) - mean) / scale
    xva = (np.asarray(z_val, np.float64) - mean) / scale
    best = None
    for C in Cs:
        lr = LogisticRegression(C=C, max_iter=3000, random_state=seed).fit(xtr, y_tr)
        s = balanced_accuracy_score(y_val, lr.predict(xva))
        if best is None or s > best[0]:  # ties keep the smaller (more regularized) C
            best = (s, C, lr)
    return Probe(mean, scale, best[2], best[1])
