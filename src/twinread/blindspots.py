"""Shared blind spots of two readers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def blindspot_table(predA, predB, y, classes) -> pd.DataFrame:
    predA, predB, y = map(np.asarray, (predA, predB, y))
    ra, rb = predA == y, predB == y
    rows = []
    for i, c in enumerate(classes):
        m = y == i
        both_wrong = m & ~ra & ~rb
        rows.append({"class": c, "n": int(m.sum()),
                     "both_right": int((m & ra & rb).sum()), "A_only": int((m & ra & ~rb).sum()),
                     "B_only": int((m & ~ra & rb).sum()), "both_wrong": int(both_wrong.sum()),
                     "both_wrong_same_label": int((both_wrong & (predA == predB)).sum())})
    return pd.DataFrame(rows)


def confident_shared_errors(pA, pB, y, conf=0.8) -> np.ndarray:
    pA, pB, y = np.asarray(pA), np.asarray(pB), np.asarray(y)
    a, b = pA.argmax(1), pB.argmax(1)
    m = (a != y) & (b != y) & (a == b) & (pA.max(1) >= conf) & (pB.max(1) >= conf)
    return np.flatnonzero(m)
