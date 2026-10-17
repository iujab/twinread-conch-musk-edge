#!/usr/bin/env python
"""Train the TwinRead-Edge student on teacher targets (Task 10). Research prototype, not a diagnostic device.

    python scripts/train_student.py --pair OPEN --dataset crc --backbone mobilenetv4_conv_small --epochs 15 \
        --batch 128 --aug k2 --budget 540          # locally, under flock $B/.gpu.lock
    python scripts/train_student.py --pair CM --dataset crc --backbone <choice> --epochs 15 --aug k2 --seed 2  # NCSA

Exit codes: 0 = done; 3 = the per-invocation time budget (--budget seconds) was used, rerun the same command to resume
from the checkpoint. ``--throughput`` only measures images/s for aug none/torch/k2 and writes throughput.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from twinread import student  # noqa: E402


def default_out(pair: str, dataset: str, aug: str, seed: int) -> Path:
    name = f"{pair.lower()}_{dataset}" + ("" if aug == "k2" else f"_{aug}") + ("" if seed == 0 else f"_seed{seed}")
    return ROOT / "results" / "student" / name


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pair", required=True, type=str.upper, choices=["CM", "OPEN"])
    ap.add_argument("--dataset", required=True, choices=["crc", "cam"])
    ap.add_argument("--backbone", default="mobilenetv4_conv_small",
                    help=f"any timm model; candidates {', '.join(student.BACKBONES)}")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--aug", choices=list(student.AUGS), default="k2")
    ap.add_argument("--out", type=Path, default=None,
                    help="default results/student/<pair>_<dataset>[_<aug> unless k2][_seed<n> unless 0]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--severity-max", type=float, default=1.0)
    ap.add_argument("--targets-dir", type=Path, default=None,
                    help="dir with student_targets.npz + probes_aligned.npz (default: searched under results/)")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--budget", type=float, default=None, help="seconds per invocation; exit 3 = rerun to resume")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--rel-impl", choices=["auto", "torch", "bigbatch"], default="auto")
    ap.add_argument("--throughput", action="store_true",
                    help="only measure images/s (aug none/torch/k2) on real train tiles -> <out>/throughput.json")
    ap.add_argument("--throughput-steps", type=int, default=20)
    return ap.parse_args(argv)


def _throughput(a, out: Path) -> int:
    import numpy as np
    from twinread.data import load_split
    tdir = a.targets_dir or student.resolve_targets_dir(a.pair, a.dataset, a.results_dir)
    with np.load(Path(tdir) / "student_targets.npz") as z:
        dimA, dimB, k = z["tA_train"].shape[1], z["tB_train"].shape[1], z["fused_logits_train"].shape[1]
    x, _ = load_split(a.data_dir or ROOT / "data", a.dataset, "train")
    res = student.measure_throughput(x[:a.batch], a.backbone, dimA, dimB, k, batch=a.batch,
                                     steps=a.throughput_steps, device=a.device, pretrained=not a.no_pretrained)
    out.mkdir(parents=True, exist_ok=True)
    (out / "throughput.json").write_text(json.dumps(student._clean(res), indent=2))
    print(json.dumps(student._clean(res["results"]), indent=1))
    return 0


def main(argv=None) -> int:
    a = parse_args(argv)
    out = a.out or default_out(a.pair, a.dataset, a.aug, a.seed)
    if a.throughput:
        return _throughput(a, out)
    r = student.train_student(a.pair, a.dataset, a.backbone, a.epochs, a.batch, a.aug, out,
                              severity_max=a.severity_max, seed=a.seed, targets_dir=a.targets_dir,
                              data_dir=a.data_dir, results_dir=a.results_dir, pretrained=not a.no_pretrained,
                              budget_s=a.budget, device=a.device, rel_impl=a.rel_impl)
    keys = ("done", "epochs_completed", "resumed_from_epoch", "out_dir", "best_epoch")
    print(json.dumps({k: r.get(k) for k in keys}))
    return 0 if r["done"] else student.EXIT_RERUN


if __name__ == "__main__":
    sys.exit(main())
