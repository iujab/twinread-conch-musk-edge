"""Generate tests/fixtures/tokenizer_v4_ids.json with transformers 4.x (reference semantics).

Run in a throwaway env (transformers==4.46.3, sentencepiece, tokenizers, torch cpu):
  python tools/make_tokenizer_fixture.py <custom_tokenizer.py> <musk_utils.py> <tokenizer.spm> <out.json>
The two source files are the ORIGINAL CONCH/MUSK implementations, loaded by path.
"""
import ast
import importlib.util
import json
import sys

STRINGS = [
    "Invasive ductal carcinoma of the breast, grade 2.",
    "Lung adenocarcinoma with lepidic and acinar growth pattern.",
    "Lung squamous cell carcinoma with keratinization.",
    "Colorectal adenocarcinoma, moderately differentiated.",
    "Normal colonic mucosa with preserved crypt architecture.",
    "Tumor infiltrating lymphocytes are prominent.",
    "Prostate adenocarcinoma, Gleason score 3+4=7.",
    "Hepatocellular carcinoma with trabecular pattern.",
    "Clear cell renal cell carcinoma with cleared cytoplasm.",
    "Papillary thyroid carcinoma with nuclear grooves.",
    "Glioblastoma with necrosis and microvascular proliferation.",
    "Melanoma with epithelioid cells and pigment.",
    "Benign breast tissue with fibrocystic change.",
    "Adipose tissue.",
    "Background: glass, no tissue present.",
    "Stroma with fibroblasts and collagen.",
    "Lymph node metastasis, extranodal extension absent.",
    "Mucinous adenocarcinoma with extracellular mucin pools.",
    "Urothelial carcinoma, high grade, muscle invasive.",
    "Serous ovarian carcinoma with papillary architecture.",
    "A photomicrograph of an H&E stained section showing a tumor.",
    "Poorly differentiated carcinoma with solid sheets.",
    "Necrosis, hemorrhage and debris.",
    "Muscle, smooth muscle bundles.",
    "Mitotic figures are frequent (12 per 10 HPF).",
    "pN0 (0/12)",
    "H&E",
    "µm",
    "",
    "CARCINOMA IN SITU",
    "TUMOR WITH LYMPHOVASCULAR INVASION",
    "ductal carcinoma in situ, cribriform type, nuclear grade 2, with comedo necrosis",
    "café über naïve",
    "Tumor 3.5 cm, margins negative, pT2 pN1a (2/14) M0, ER+ PR+ HER2-.",
    "a",
    "x" * 300,
    "carcinoma " * 40,
    "Small cell carcinoma, neuroendocrine features.",
    "Squamous epithelium with parakeratosis.",
    "Granulomatous inflammation with giant cells.",
]
assert len(STRINGS) == 40, len(STRINGS)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_fn(path, fn):
    # musk/utils.py imports heavy deps; extract just the one function
    tree = ast.parse(open(path).read())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn)
    ns = {}
    exec(compile(ast.Module([node], []), path, "exec"), ns)
    return ns[fn]


def main(ct_path, mu_path, spm, out):
    ct = load(ct_path, "orig_conch_tok")
    import os
    tfile = os.path.join(os.path.dirname(ct_path), "tokenizers", ct.DEFAULT_TOKENIZER)
    from transformers import PreTrainedTokenizerFast, XLMRobertaTokenizer
    tok = PreTrainedTokenizerFast(tokenizer_file=tfile, bos_token="<start_of_text>",
                                  eos_token="<end_of_text>", pad_token="<pad>")
    conch = ct.tokenize(tok, STRINGS).tolist()
    xlm = load_fn(mu_path, "xlm_tokenizer")
    mt = XLMRobertaTokenizer(spm)
    ids, pads = zip(*[xlm(s, mt, max_len=100) for s in STRINGS])
    json.dump({"conch": conch, "musk_ids": [list(i) for i in ids],
               "musk_pad": [list(p) for p in pads]}, open(out, "w"))
    import transformers
    print("transformers", transformers.__version__, "vocab", len(mt))


if __name__ == "__main__":
    main(*sys.argv[1:5])
