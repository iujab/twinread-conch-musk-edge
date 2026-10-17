"""Build a tiny R24-format test bundle (random weights; numbers are meaningless). usage: make_device_bundle.py OUTDIR"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

C, DA, DB = 3, 16, 24
CLASSES = ["ADI", "TUM", "STR"]
THR = {"js": 0.2, "review": 0.9}


def write_sums(d: Path):
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}"
             for p in sorted(d.iterdir()) if p.name != "SHA256SUMS"]
    (d / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def build(d: Path) -> Path:
    import torch
    torch.manual_seed(0)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, 7, stride=8)
            self.hA = torch.nn.Linear(8, DA)
            self.hB = torch.nn.Linear(8, DB)
            self.cls = torch.nn.Linear(8, C)
            self.rev = torch.nn.Linear(8, 1)

        def forward(self, x):
            f = torch.relu(self.conv(x)).mean((2, 3))
            n = torch.nn.functional.normalize
            return n(self.hA(f), dim=-1), n(self.hB(f), dim=-1), self.cls(f), self.rev(f)[:, 0]

    d.mkdir(parents=True, exist_ok=True)
    m = Tiny().eval()
    torch.onnx.export(m, (torch.rand(2, 3, 224, 224),), str(d / "model.onnx"), input_names=["x"],
                      output_names=["hA", "hB", "logits", "review_logit"], dynamic_axes={"x": {0: "N"}}, dynamo=False)
    rng = np.random.default_rng(0)
    np.savez(d / "prompt_bank.npz", bankA=rng.normal(size=(C, DA)).astype("f4"), bankB=rng.normal(size=(C, DB)).astype("f4"))
    pr = {}
    for r, dd in (("A", DA), ("B", DB)):
        pr.update({f"{r}_coef": rng.normal(size=(C, dd)).astype("f4"), f"{r}_intercept": rng.normal(size=C).astype("f4"),
                   f"{r}_mean": rng.normal(size=dd).astype("f4") * 0.01, f"{r}_scale": (rng.random(dd) + 0.5).astype("f4"),
                   f"{r}_classes": np.arange(C)})
    np.savez(d / "probes.npz", **pr)
    (d / "config.json").write_text(json.dumps({"pair": "test", "classes": CLASSES, "thresholds": THR,
                                               "licence_note": "test fixture", "distributable": True}))
    write_sums(d)
    return d




if __name__ == "__main__":
    print(build(Path(sys.argv[1])))
