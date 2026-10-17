import importlib.util
import json
from pathlib import Path

import pytest

import twinread.compat as compat
from twinread.compat import XLMRSpTokenizer, conch_tokenize, patch_conch

FIX = json.loads((Path(__file__).parent / "fixtures" / "tokenizer_v4_ids.json").read_text())


def _strings():
    spec = importlib.util.spec_from_file_location(
        "mtf", Path(__file__).parents[1] / "tools" / "make_tokenizer_fixture.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.STRINGS


STRINGS = _strings()


def _spm():
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download("xiangjx/musk", "tokenizer.spm", local_files_only=True)
    except Exception as e:
        pytest.skip(f"MUSK tokenizer.spm not in HF cache: {e}")


def test_conch_tokenize_matches_v4_fixture():
    from conch.open_clip_custom import get_tokenizer
    assert conch_tokenize(get_tokenizer(), STRINGS).tolist() == FIX["conch"]


def test_xlmr_shim_matches_v4_fixture():
    from musk.utils import xlm_tokenizer as musk_xlm_tokenizer
    tok = XLMRSpTokenizer(_spm())
    ids, pads = zip(*[musk_xlm_tokenizer(s, tok, max_len=100) for s in STRINGS])
    assert [list(i) for i in ids] == FIX["musk_ids"] and [list(p) for p in pads] == FIX["musk_pad"]


def test_xlmr_special_ids():
    t = XLMRSpTokenizer(_spm())
    assert (t.bos_token_id, t.pad_token_id, t.eos_token_id, t.unk_token_id) == (0, 1, 2, 3)


def test_patch_conch_idempotent():
    import conch.open_clip_custom as oc
    import conch.open_clip_custom.custom_tokenizer as ct
    patch_conch(); patch_conch()
    assert oc.tokenize is compat.conch_tokenize
    assert ct.tokenize is compat.conch_tokenize


def test_patch_conch_noop_on_transformers_v4(monkeypatch):
    import conch.open_clip_custom as oc
    import conch.open_clip_custom.custom_tokenizer as ct
    orig_oc, orig_ct = oc.tokenize, ct.tokenize
    try:
        oc.tokenize = ct.tokenize = lambda *a, **k: None
        sentinel = oc.tokenize
        monkeypatch.setattr("transformers.__version__", "4.46.3")
        patch_conch()
        assert oc.tokenize is sentinel and ct.tokenize is sentinel
    finally:
        oc.tokenize, ct.tokenize = orig_oc, orig_ct
