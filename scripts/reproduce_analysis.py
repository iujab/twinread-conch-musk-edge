"""Reproduce a committed pair analysis into a scratch results dir (CPU). Used by run_demo.sh on a clean clone.

The analysis artefacts the student and the bundle need (student_targets.npz, probes_aligned.npz) are git-ignored, and
`run_pair_analysis` refuses to write where results already exist, so a clean clone cannot regenerate them in place.
This script runs the same pre-registered analysis into `--out` (never into results/, and never touching the committed
preregistration.json), then compares the reproduced readers/fusion/review JSON with the committed ones number by
number. Exit 0 = reproduced (written to <out>/<PAIR>/<ds>/reproduce_check.json); exit 4 = a mismatch.

    PYTHONPATH=src .venv/bin/python scripts/reproduce_analysis.py --pair OPEN --dataset crc \
        --committed results/open --out build/reproduce/open
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ("readers.json", "fusion.json", "review.json")


def diff(a, b, path="", tol=1e-9, out=None):
    """Every leaf where the two JSON documents differ (numbers to `tol`)."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: missing on one side")
            else:
                diff(a[k], b[k], f"{path}/{k}", tol, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", tol, out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (math.isclose(a, b, rel_tol=0, abs_tol=tol) or (math.isnan(a) and math.isnan(b))):
            out.append(f"{path}: {a} != {b}")
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=["OPEN", "CM"])
    ap.add_argument("--dataset", required=True, choices=["crc", "cam"])
    ap.add_argument("--committed", required=True, type=Path, help="committed results root, e.g. results/open")
    ap.add_argument("--out", required=True, type=Path, help="scratch results root, e.g. build/reproduce/open")
    a = ap.parse_args(argv)
    committed, out = (ROOT / a.committed).resolve(), (ROOT / a.out).resolve()
    if out == committed or committed in out.parents:
        raise SystemExit("--out must not be (or be inside) the committed results dir")
    cdir, odir = committed / a.pair / a.dataset, out / a.pair / a.dataset
    check = odir / "reproduce_check.json"
    if not (odir / "student_targets.npz").exists():
        from twinread.analyze import run_pair_analysis
        partner = None
        if a.pair == "OPEN":
            partner = json.loads((committed / "partner.json").read_text())["partner"]
        run_pair_analysis(a.pair, a.dataset, results_dir=out, partner=partner)
    mism = {f: diff(json.loads((odir / f).read_text()), json.loads((cdir / f).read_text())) for f in FILES}
    pre_o, pre_c = (json.loads((d / "preregistration.json").read_text()) for d in (odir, cdir))
    pre_o.pop("created_utc", None)
    pre_c.pop("created_utc", None)
    mism["preregistration.json (except created_utc)"] = diff(pre_o, pre_c)
    ok = not any(mism.values())
    check.write_text(json.dumps({"reproduced": ok, "committed": str(cdir.relative_to(ROOT)),
                                 "mismatches": {k: v[:20] for k, v in mism.items()},
                                 "n_mismatches": {k: len(v) for k, v in mism.items()}}, indent=1))
    print(json.dumps({"pair": a.pair, "dataset": a.dataset, "reproduced": ok,
                      "n_mismatches": {k: len(v) for k, v in mism.items()}}))
    return 0 if ok else 4


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    sys.exit(main())
