"""Pre-download weights on the Delta LOGIN node (compute nodes are offline). Needs HF_TOKEN with CONCH and MUSK access.

    HF_TOKEN=... python ncsa/download_weights.py [--cache DIR]
Run only after the user's explicit OK.
"""
import argparse
import os
import sys

from huggingface_hub import hf_hub_download, snapshot_download

GATED = [("MahmoodLab/CONCH", ["pytorch_model.bin"]), ("xiangjx/musk", ["model.safetensors", "tokenizer.spm"])]
OPEN = [  # (repo, allow_patterns): the open readers used by the OPEN pair
    ("jamessyx/pathgenclip-vit-large-patch14-hf", None),
    ("wisdomik/QuiltNet-B-16", None),
    ("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", None),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None, help="HF cache dir (default: ~/.cache/huggingface)")
    a = ap.parse_args()
    tok = os.environ.get("HF_TOKEN")
    failed = []
    for repo, files in GATED:
        for f in files:
            try:
                print(repo, f, "->", hf_hub_download(repo, f, token=tok, cache_dir=a.cache))
            except Exception as e:  # noqa: BLE001
                failed.append(f"{repo}/{f}: {e}"); print("FAILED", repo, f, e)
    for repo, pats in OPEN:
        try:
            print(repo, "->", snapshot_download(repo, allow_patterns=pats, cache_dir=a.cache))
        except Exception as e:  # noqa: BLE001
            failed.append(f"{repo}: {e}"); print("FAILED", repo, e)
    print("FAILED:" if failed else "all weights cached", *failed, sep="\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
