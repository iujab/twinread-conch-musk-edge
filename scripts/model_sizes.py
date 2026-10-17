#!/usr/bin/env python
"""Parameter counts and file sizes of the edge students and of the teachers -> results/student/model_sizes.json.

CPU only (no GPU). Students: the trained checkpoints (best.pt) and their int8 ONNX in bundles/<pair>/model.onnx.
Teachers:
  * CONCH (CoCa ViT-B/16): built and loaded on the CPU; vision tower = ``model.visual`` (trunk + attentional pooler +
    projection), the image path of the reader;
  * MUSK (BEiT-3-style multiway transformer, ViT-L): built on the meta device (no weights needed to count); every
    layer has a vision branch ``.A`` and a language branch ``.B``; image path = vision_embed + positions.A + all ``.A``
    branches + vision_head (the language branch, text_embed, language_head and logit_scale are excluded);
  * PathGen-CLIP ViT-L/14-336 (meta device from its config): vision_model + visual_projection;
  * QuiltNet-B-16 (open_clip, CPU): ``model.visual``.
Checkpoint bytes are the files in the local HF cache. Research prototype, not a diagnostic device.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python scripts/model_sizes.py
"""
from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results" / "student" / "model_sizes.json"


def _n(params) -> int:
    return int(sum(p.numel() for p in params))


def _hf_file(repo: str, name: str) -> int | None:
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache(repo, name)
    return os.path.getsize(p) if isinstance(p, str) and os.path.exists(p) else None


def student(run: str, pair: str) -> dict:
    from twinread.student import load_student
    m = load_student(ROOT / "results" / "student" / run / "best.pt")
    trunk = _n(m.trunk.parameters())
    total = _n(m.parameters())
    onnx = ROOT / "bundles" / pair / "model.onnx"
    return {"run": f"results/student/{run}", "backbone": m.config["backbone"], "params_total": total,
            "params_trunk": trunk, "params_heads": total - trunk, "fp32_param_bytes": 4 * total,
            "int8_onnx": f"bundles/{pair}/model.onnx",
            "int8_onnx_bytes": onnx.stat().st_size if onnx.exists() else None,
            "input": "224 px"}


def conch() -> dict:
    from twinread.compat import patch_conch
    patch_conch()
    from conch.open_clip_custom import create_model_from_pretrained
    from huggingface_hub import try_to_load_from_cache
    model, _ = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/CONCH", device="cpu")
    sd = torch.load(try_to_load_from_cache("MahmoodLab/CONCH", "pytorch_model.bin"), map_location="cpu", mmap=True,
                    weights_only=True)
    sd = sd.get("state_dict", sd)
    named = dict(model.named_parameters())
    released = _n(v for k, v in named.items() if k in sd)
    return {"model": "MahmoodLab/CONCH (CoCa, ViT-B/16, 448 px)", "params_total": released,
            "params_total_def": "parameters in the released checkpoint (image encoder + text encoder)",
            "params_text_decoder_not_released": _n(v for k, v in named.items() if k not in sd),
            "params_vision_tower": _n(model.visual.parameters()), "vision_tower_def": "model.visual",
            "checkpoint": "pytorch_model.bin (fp32)", "checkpoint_bytes": _hf_file("MahmoodLab/CONCH", "pytorch_model.bin"),
            "counted_from": "model built and loaded on the CPU; the CoCa text decoder is built by the library but its "
                            "weights are not in the public checkpoint (unused by the reader), so it is not counted"}


def musk() -> dict:
    import timm
    from musk import modeling  # noqa: F401  (registers musk_large_patch16_384)
    with torch.device("meta"):
        m = timm.create_model("musk_large_patch16_384")
    named = list(m.named_parameters())
    lang = lambda n: ".B." in n or n.startswith(("beit3.text_embed", "language_head")) or n == "logit_scale"  # noqa: E731
    return {"model": "xiangjx/musk (multiway BEiT-3 style, ViT-L/16, 384 px)",
            "params_total": _n(p for _, p in named), "params_vision_tower": _n(p for n, p in named if not lang(n)),
            "vision_tower_def": "all parameters except the language branch (.B.), text_embed, language_head, logit_scale",
            "checkpoint": "model.safetensors (fp16)", "checkpoint_bytes": _hf_file("xiangjx/musk", "model.safetensors"),
            "counted_from": "meta device (architecture only, no weights loaded)"}


def pathgenclip() -> dict:
    from transformers import CLIPConfig, CLIPModel
    rid = "jamessyx/pathgenclip-vit-large-patch14-hf"
    with torch.device("meta"):
        m = CLIPModel(CLIPConfig.from_pretrained(rid))
    vis = _n(m.vision_model.parameters()) + _n(m.visual_projection.parameters())
    name = next((f for f in ("pytorch_model.bin", "model.safetensors") if _hf_file(rid, f)), None)
    return {"model": f"{rid} (CLIP ViT-L/14, 336 px)", "params_total": _n(m.parameters()),
            "params_vision_tower": vis, "vision_tower_def": "vision_model + visual_projection",
            "checkpoint": f"{name} (main revision)", "checkpoint_bytes": _hf_file(rid, name) if name else None,
            "counted_from": "meta device from the cached config"}


def quiltnet() -> dict:
    import open_clip
    hub = "hf-hub:wisdomik/QuiltNet-B-16"
    model, _, _ = open_clip.create_model_and_transforms(hub, device="cpu")
    ck = _hf_file("wisdomik/QuiltNet-B-16", "open_clip_pytorch_model.bin")
    return {"model": "wisdomik/QuiltNet-B-16 (CLIP ViT-B/16, 224 px)", "params_total": _n(model.parameters()),
            "params_vision_tower": _n(model.visual.parameters()), "vision_tower_def": "model.visual",
            "checkpoint": "open_clip_pytorch_model.bin", "checkpoint_bytes": ck,
            "counted_from": "model built and loaded on the CPU"}


def main() -> int:
    torch.set_num_threads(4)
    st = {"CM": student("cm_crc", "cm"), "OPEN": student("open_crc", "open")}
    te = {"conch": conch(), "musk": musk(), "pathgenclip": pathgenclip(), "quiltnet_b16": quiltnet()}
    pairs = {"CM": ("conch", "musk"), "OPEN": ("pathgenclip", "quiltnet_b16")}
    ratios = {}
    for pair, (a, b) in pairs.items():
        vis = te[a]["params_vision_tower"] + te[b]["params_vision_tower"]
        ck = (te[a]["checkpoint_bytes"] or 0) + (te[b]["checkpoint_bytes"] or 0)
        ratios[pair] = {
            "teacher_vision_towers_params_sum": vis,
            "student_params_over_teacher_vision_towers": st[pair]["params_total"] / vis,
            "teacher_checkpoint_bytes_sum": ck or None,
            "student_int8_onnx_over_teacher_checkpoints": (st[pair]["int8_onnx_bytes"] / ck
                                                          if ck and st[pair]["int8_onnx_bytes"] else None)}
    doc = {"students": st, "teachers": te, "ratios": ratios,
           "note": "Parameter counts, not FLOPs. The teachers also need a GPU-class device at 384-448 px; the student "
                   "is one 224-px CNN for both readers. Research prototype, not a diagnostic device.",
           "created": datetime.datetime.now().isoformat(timespec="seconds"), "torch": torch.__version__}
    OUT.write_text(json.dumps(doc, indent=2))
    print(json.dumps(ratios, indent=1))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
