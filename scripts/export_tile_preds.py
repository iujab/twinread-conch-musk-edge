"""POST-HOC: reproduce per-tile test predictions + triage curves for a pair (see twinread.tilepreds). CPU only.

    OMP_NUM_THREADS=4 PYTHONPATH=src .venv/bin/python scripts/export_tile_preds.py --pair CM --root results/cm_local
"""
import argparse
import json
import time
from pathlib import Path

from twinread.tilepreds import reproduce

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="CM")
    ap.add_argument("--root", default="results/cm_local")
    ap.add_argument("--datasets", nargs="+", default=["crc", "cam"])
    a = ap.parse_args()
    for ds in a.datasets:
        t = time.time()
        doc = reproduce(a.pair, ds, ROOT / a.root)
        print(json.dumps({"pair": a.pair, "dataset": ds, "verified": doc["verification"]["all_ok"],
                          "n_checks": len(doc["verification"]["checks"]), "seconds": round(time.time() - t, 1)}))
