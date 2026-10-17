"""Readers: one uniform interface over CONCH, MUSK and three open pathology encoders.

A Reader exposes the model's official preprocessing, an *aligned* embedding (the contrastive image-text space,
L2-normalized), a *probe* embedding (pre-projection features for linear probes / distillation) and a text encoder
into the aligned space. CONCH and MUSK follow the organizer adapters (ashby-hackathon-2026/models.py) exactly.
Weights load lazily inside ``load_reader``; ``spec_of`` needs none.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import PIL.Image
import torch
from torchvision import transforms as T

READER_NAMES = ("conch", "musk", "pathgenclip", "quiltnet_b16", "biomedclip")

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class PreprocessSpec:
    size: int
    crop: int
    resample: str  # "bicubic" (PIL semantics, a=-0.5) or "bilinear"
    mean: tuple
    std: tuple


_SPECS = {
    "conch": PreprocessSpec(448, 448, "bicubic", _CLIP_MEAN, _CLIP_STD),
    "musk": PreprocessSpec(384, 384, "bicubic", (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "pathgenclip": PreprocessSpec(336, 336, "bicubic", _CLIP_MEAN, _CLIP_STD),
    "quiltnet_b16": PreprocessSpec(224, 224, "bicubic", _CLIP_MEAN, _CLIP_STD),
    "biomedclip": PreprocessSpec(224, 224, "bicubic", _CLIP_MEAN, _CLIP_STD),
}


def spec_of(name: str) -> PreprocessSpec:
    return _SPECS[name]


@dataclass
class Reader:
    name: str
    spec: PreprocessSpec
    dim_aligned: int
    dim_probe: int
    preprocess_pil: Callable[[PIL.Image.Image], torch.Tensor]
    encode_aligned: Callable[[torch.Tensor], np.ndarray]
    encode_probe: Callable[[torch.Tensor], np.ndarray]
    encode_texts: Callable[[list], np.ndarray]
    _close: Callable[[], None] = field(default=lambda: None, repr=False)

    def close(self) -> None:
        """Drop model references and free GPU memory."""
        self._close()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _l2(x: torch.Tensor) -> np.ndarray:
    x = x.float()
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return x.cpu().numpy().astype(np.float32)


def _raw(x: torch.Tensor) -> np.ndarray:
    return x.float().cpu().numpy().astype(np.float32)


def _compose(spec: PreprocessSpec) -> T.Compose:
    interp = {"bicubic": T.InterpolationMode.BICUBIC, "bilinear": T.InterpolationMode.BILINEAR}[spec.resample]
    return T.Compose([
        T.Resize(spec.size, interpolation=interp, antialias=True),
        T.CenterCrop(spec.crop),
        T.Lambda(lambda im: im.convert("RGB")),
        T.ToTensor(),
        T.Normalize(spec.mean, spec.std),
    ])


def _holder(*objs):
    """Return a closer that deletes the model references held in a mutable list."""
    box = list(objs)

    def close():
        box.clear()
    return box, close


def _load_conch(device: str) -> Reader:
    from twinread.compat import patch_conch
    patch_conch()
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
    model, pre = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/CONCH")
    model = model.to(device).eval()
    tok = get_tokenizer()

    @torch.no_grad()
    def aligned(b):
        return _l2(model.encode_image(b.to(device), proj_contrast=True, normalize=True))

    @torch.no_grad()
    def probe(b):
        return _raw(model.encode_image(b.to(device), proj_contrast=False, normalize=False))

    @torch.no_grad()
    def texts(ts):
        return _l2(model.encode_text(tokenize(texts=list(ts), tokenizer=tok).to(device)))

    box, close = _holder(model)
    return Reader("conch", _SPECS["conch"], 512, 512, pre, aligned, probe, texts, close)


def _load_musk(device: str) -> Reader:
    import timm
    from huggingface_hub import hf_hub_download
    from musk import modeling  # noqa: F401  (registers musk_large_patch16_384)
    from musk import utils as mu
    from twinread.compat import XLMRSpTokenizer
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = timm.create_model("musk_large_patch16_384")
    mu.load_model_and_may_interpolate(hf_hub_download("xiangjx/musk", "model.safetensors"), model, "model|module", "")
    model = model.to(device=device, dtype=dtype).eval()
    tok = XLMRSpTokenizer(hf_hub_download("xiangjx/musk", "tokenizer.spm"))
    pre = _compose(_SPECS["musk"])

    @torch.no_grad()
    def aligned(b):
        return _l2(model(image=b.to(device=device, dtype=dtype), with_head=True, out_norm=True, ms_aug=False,
                         return_global=True)[0])

    @torch.no_grad()
    def probe(b):
        return _raw(model(image=b.to(device=device, dtype=dtype), with_head=False, out_norm=False, ms_aug=True,
                          return_global=True)[0])

    @torch.no_grad()
    def texts(ts):
        tk = [mu.xlm_tokenizer(t, tok, max_len=100) for t in ts]
        ids = torch.tensor([i for i, _ in tk], device=device)
        pad = torch.tensor([p for _, p in tk], device=device)
        return _l2(model(text_description=ids, padding_mask=pad, with_head=True, out_norm=True, ms_aug=False,
                         return_global=True)[1])

    box, close = _holder(model)
    return Reader("musk", _SPECS["musk"], 1024, 2048, pre, aligned, probe, texts, close)


def _load_pathgenclip(device: str) -> Reader:
    from transformers import CLIPModel, CLIPProcessor
    rid = "jamessyx/pathgenclip-vit-large-patch14-hf"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = CLIPModel.from_pretrained(rid, torch_dtype=dtype).to(device).eval()
    proc = CLIPProcessor.from_pretrained(rid)
    ip = proc.image_processor
    spec = PreprocessSpec(int(ip.size["shortest_edge"]), int(ip.crop_size["height"]), "bicubic",
                          tuple(ip.image_mean), tuple(ip.image_std))
    assert ip.resample == 3, "PathGen-CLIP processor is expected to use bicubic"
    assert spec == _SPECS["pathgenclip"], f"static spec table is stale: {spec}"
    pre = _compose(spec)

    @torch.no_grad()
    def _pooled(b):
        return model.vision_model(pixel_values=b.to(device=device, dtype=dtype)).pooler_output

    @torch.no_grad()
    def aligned(b):
        return _l2(model.visual_projection(_pooled(b)))

    @torch.no_grad()
    def probe(b):
        return _raw(_pooled(b))

    @torch.no_grad()
    def texts(ts):
        enc = proc.tokenizer(list(ts), return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        pooled = model.text_model(**enc).pooler_output
        return _l2(model.text_projection(pooled))

    box, close = _holder(model)
    return Reader("pathgenclip", spec, 768, 1024, pre, aligned, probe, texts, close)


def _load_open_clip(name: str, hub: str, device: str, dim_aligned: int, dim_probe: int) -> Reader:
    import open_clip
    model, _, pre = open_clip.create_model_and_transforms(hub)
    model = model.to(device).eval()
    tok = open_clip.get_tokenizer(hub)
    visual = model.visual

    @torch.no_grad()
    def aligned(b):
        return _l2(model.encode_image(b.to(device)))

    @torch.no_grad()
    def probe(b):
        b = b.to(device)
        if hasattr(visual, "trunk"):  # timm tower (BiomedCLIP): pooled trunk features before head.proj
            return _raw(visual.trunk(b))
        proj, visual.proj = visual.proj, None  # OpenAI-style ViT: skip the final `pooled @ proj`
        try:
            return _raw(visual(b))
        finally:
            visual.proj = proj

    @torch.no_grad()
    def texts(ts):
        return _l2(model.encode_text(tok(list(ts)).to(device)))

    box, close = _holder(model)
    return Reader(name, _SPECS[name], dim_aligned, dim_probe, pre, aligned, probe, texts, close)


def load_reader(name: str, device: str = "cuda") -> Reader:
    """Load one reader's weights. Hold one at a time on a shared GPU and call ``close()`` when done."""
    torch.set_num_threads(4)
    if name == "conch":
        return _load_conch(device)
    if name == "musk":
        return _load_musk(device)
    if name == "pathgenclip":
        return _load_pathgenclip(device)
    if name == "quiltnet_b16":
        return _load_open_clip(name, "hf-hub:wisdomik/QuiltNet-B-16", device, 512, 768)
    if name == "biomedclip":
        return _load_open_clip(name, "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", device, 512, 768)
    raise KeyError(name)
