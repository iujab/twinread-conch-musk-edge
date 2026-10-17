"""Frozen prompt ensembles for zero-shot readers (hash is pre-registered)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def load_prompts(dataset: str, prompts_dir: Path | None = None):
    d = Path(prompts_dir) if prompts_dir else PROMPTS_DIR
    cfg = json.loads((d / f"{dataset}.json").read_text())
    templates = json.loads((d / "templates.json").read_text())["templates"]
    return list(cfg["classes"]), {k: list(v) for k, v in cfg["synonyms"].items()}, list(templates)


def text_bank(reader, dataset: str, prompts_dir: Path | None = None) -> np.ndarray:
    """[C, dim_aligned]: mean over template x synonym embeddings per class, L2-normalized."""
    classes, syn, templates = load_prompts(dataset, prompts_dir)
    rows = []
    for c in classes:
        texts = [t.format(s) for s in syn[c] for t in templates]
        e = np.asarray(reader.encode_texts(texts), dtype=np.float64)
        m = e.mean(0)
        rows.append(m / max(np.linalg.norm(m), 1e-12))
    return np.stack(rows).astype(np.float32)


def zeroshot_probs(z: np.ndarray, bank: np.ndarray, scale: float = 100.0) -> np.ndarray:
    s = scale * (np.asarray(z, dtype=np.float64) @ np.asarray(bank, dtype=np.float64).T)
    s -= s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)
