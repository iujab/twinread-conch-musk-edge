"""Export candidate student backbones (LatencyProxy) to ONNX fp32 + static int8 QDQ for latency benchmarking.

LatencyProxy (controller ruling R8) stands in for TwinStudent: pretrained timm trunk, ImageNet normalisation
inside the module (ONNX input = float [N,3,224,224] in [0,1]), global-pool feature, 4 heads. Head cost is
negligible; the real student is re-measured later. Research prototype, not a diagnostic device.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

BACKBONES = ("mobilenetv4_conv_small", "efficientnet_b0", "mobileone_s0", "regnetx_002", "resnet18", "lcnet_050")
ROOT = Path(__file__).resolve().parents[1]
N_CALIB = 64


def build_proxy(backbone, dimA=512, dimB=1024, n_classes=9, pretrained=True):
    import timm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LatencyProxy(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            self.trunk.eval()
            with torch.no_grad():
                f = self.trunk(torch.zeros(1, 3, 224, 224)).shape[-1]
            self.hA, self.hB = nn.Linear(f, dimA), nn.Linear(f, dimB)
            self.cls, self.review = nn.Linear(f, n_classes), nn.Linear(f, 1)
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x):
            f = self.trunk((x - self.mean) / self.std)
            return (F.normalize(self.hA(f), dim=-1), F.normalize(self.hB(f), dim=-1),
                    self.cls(f), self.review(f))

    return LatencyProxy().eval().cpu()


def LatencyProxy(backbone, dimA=512, dimB=1024, n_classes=9, pretrained=True):
    return build_proxy(backbone, dimA, dimB, n_classes, pretrained)


class _Reader:
    def __init__(self, X, name, bs=8):
        self.X, self.name, self.bs, self.i = X, name, bs, 0

    def get_next(self):
        if self.i >= len(self.X):
            return None
        x = np.ascontiguousarray(self.X[self.i:self.i + self.bs]).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        self.i += self.bs
        return {self.name: x}


def export_backbone(name, out_dir, calib_x, pretrained=True, dimA=512, dimB=1024, n_classes=9):
    import onnx
    import torch
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(_Reader, CalibrationDataReader):
        pass

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32, pre, int8 = out_dir / f"{name}_fp32.onnx", out_dir / f"{name}_pre.onnx", out_dir / f"{name}_int8.onnx"
    m = LatencyProxy(name, dimA, dimB, n_classes, pretrained)
    torch.onnx.export(m, torch.rand(1, 3, 224, 224), str(fp32), dynamo=False, opset_version=17,
                      input_names=["input"], output_names=["hA", "hB", "logits", "review_logit"],
                      dynamic_axes={"input": {0: "b"}})
    onnx.checker.check_model(str(fp32))
    quant_pre_process(str(fp32), str(pre))
    quantize_static(str(pre), str(int8), Reader(calib_x, "input"), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    pre.unlink(missing_ok=True)
    return fp32, int8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "device" / "backbones"))
    ap.add_argument("--calib", default=str(ROOT / "data" / "crc_train_x.npy"))
    ap.add_argument("--backbones", nargs="*", default=list(BACKBONES))
    a = ap.parse_args()
    X = np.load(a.calib, mmap_mode="r")
    idx = np.sort(np.random.RandomState(0).choice(len(X), N_CALIB, replace=False))
    calib = np.ascontiguousarray(X[idx])
    status = {}
    for b in a.backbones:
        try:
            export_backbone(b, a.out, calib)
            status[b] = "ok"
        except Exception as e:  # record and continue
            status[b] = f"FAILED: {type(e).__name__}: {e}"[:300]
        print(b, status[b], flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "export_status.json").write_text(json.dumps(status, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
