"""transformers-v5 compatibility shims for the CONCH and MUSK text tokenizers.

In transformers v5, ``XLMRobertaTokenizer(vocab_file=spm)`` silently builds a 5-token vocab (everything maps to
unk), and CONCH's ``PreTrainedTokenizerFast`` path needs the same care. These shims reproduce the v4 outputs
exactly (pinned by tests/fixtures/tokenizer_v4_ids.json, generated with transformers 4.46.3).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def conch_tokenize(tokenizer, texts: list[str]) -> torch.LongTensor:
    """Same as conch.open_clip_custom.custom_tokenizer.tokenize: 127 tokens + one trailing pad -> [N, 128]."""
    enc = tokenizer(list(texts), max_length=127, truncation=True, padding="max_length", return_tensors="pt")
    return F.pad(enc["input_ids"], (0, 1), value=tokenizer.pad_token_id)


def _transformers_major() -> int:
    import transformers
    return int(str(transformers.__version__).split(".")[0])


def patch_conch() -> None:
    """Replace CONCH's tokenize with ``conch_tokenize``. Idempotent; a no-op on transformers < 5."""
    if _transformers_major() < 5:
        return
    import conch.open_clip_custom as oc
    import conch.open_clip_custom.custom_tokenizer as ct
    oc.tokenize = conch_tokenize
    ct.tokenize = conch_tokenize


class XLMRSpTokenizer:
    """transformers-v4 slow XLMRobertaTokenizer semantics over a raw sentencepiece model.

    Piece id i -> i+1 (fairseq offset), except sentencepiece unk (0) -> 3; wrapped in bos(0)/eos(2).
    """
    bos_token_id = 0
    pad_token_id = 1
    eos_token_id = 2
    unk_token_id = 3

    def __init__(self, vocab_file: str):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(vocab_file))

    def encode(self, text: str) -> list[int]:
        ids = [self.unk_token_id if i == 0 else i + 1 for i in self.sp.EncodeAsIds(text)]
        return [self.bos_token_id] + ids + [self.eos_token_id]
