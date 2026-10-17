"""Fusion candidates: average of probe probabilities, concat logistic regression, gated MLP."""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from twinread.probes import standardizer


def fuse_average(pA, pB) -> np.ndarray:
    p = (np.asarray(pA, np.float64) + np.asarray(pB, np.float64)) / 2
    return p / p.sum(1, keepdims=True)


class Fused:
    def __init__(self, name, fn, info=None):
        self.name, self._fn, self.info = name, fn, info or {}

    def predict_proba(self, zA, zB) -> np.ndarray:
        return self._fn(zA, zB)


def fit_concat_lr(zA_tr, zB_tr, y_tr, zA_val, zB_val, y_val, Cs=(0.01, 0.1, 1.0), seed=0) -> Fused:
    zt = np.hstack([zA_tr, zB_tr])
    zv = np.hstack([zA_val, zB_val])
    mean, scale = standardizer(zt)
    f = lambda z: (np.asarray(z, np.float64) - mean) / scale
    best = None
    for C in Cs:
        lr = LogisticRegression(C=C, max_iter=3000, random_state=seed).fit(f(zt), y_tr)
        s = balanced_accuracy_score(y_val, lr.predict(f(zv)))
        if best is None or s > best[0]:
            best = (s, lr)
    lr = best[1]
    return Fused("concat_lr", lambda a, b: lr.predict_proba(f(np.hstack([a, b]))), {"C": lr.C})


class _Net(nn.Module):
    def __init__(self, dA, dB, n_cls, h=256, p=0.2):
        super().__init__()
        self.mA, self.mB = nn.Linear(dA, h), nn.Linear(dB, h)
        self.gate = nn.Linear(2 * h, h)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(3 * h, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, n_cls))

    def forward(self, a, b):
        ha, hb = self.mA(a), self.mB(b)
        g = torch.sigmoid(self.gate(torch.cat([ha, hb], 1)))
        return self.head(torch.cat([g * ha + (1 - g) * hb, torch.abs(ha - hb), ha * hb], 1))


def fit_gated_mlp(zA_tr, zB_tr, y_tr, zA_val, zB_val, y_val, seed=0, epochs=30) -> Fused:
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    mA, sA = standardizer(zA_tr)
    mB, sB = standardizer(zB_tr)
    tt = lambda z, m, s: torch.tensor((np.asarray(z, np.float64) - m) / s, dtype=torch.float32)
    A, B = tt(zA_tr, mA, sA), tt(zB_tr, mB, sB)
    Av, Bv = tt(zA_val, mA, sA), tt(zB_val, mB, sB)
    classes = np.unique(y_tr)
    y = torch.tensor(np.searchsorted(classes, y_tr))
    net = _Net(A.shape[1], B.shape[1], len(classes))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    g = torch.Generator().manual_seed(seed)
    best, state = -1.0, None
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(y), generator=g)
        for i in range(0, len(y), 256):
            ix = perm[i:i + 256]
            opt.zero_grad()
            nn.functional.cross_entropy(net(A[ix], B[ix]), y[ix]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            s = balanced_accuracy_score(y_val, classes[net(Av, Bv).argmax(1).numpy()])
        if s > best:
            best, state = s, copy.deepcopy(net.state_dict())
    net.load_state_dict(state)
    net.eval()

    def fn(a, b):
        with torch.no_grad():
            return torch.softmax(net(tt(a, mA, sA), tt(b, mB, sB)), 1).numpy().astype(np.float64)

    return Fused("gated_mlp", fn)
