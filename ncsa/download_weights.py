"""Pre-download weights on the Delta LOGIN node (compute nodes are offline). Needs HF_TOKEN with CONCH and MUSK access.

    HF_TOKEN=... python ncsa/download_weights.py [--cache DIR]
Run only after the user's explicit OK.
"""
import argparse
import os
import sys

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))  # hub cache = $HF_HOME/hub; same default as every sbatch

from huggingface_hub import hf_hub_download, snapshot_download

GATED = [("MahmoodLab/CONCH", ["pytorch_model.bin", "meta.yaml"]), ("xiangjx/musk", ["model.safetensors", "tokenizer.spm"])]
OPEN = [  # (repo, allow_patterns): the open readers used by the OPEN pair
    ("jamessyx/pathgenclip-vit-large-patch14-hf", None),
    ("wisdomik/QuiltNet-B-16", None),
    ("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", None),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None, help="HF hub cache dir (default: $HF_HOME/hub)")
    ap.add_argument("--backbones", nargs="*", default=["mobilenetv3_large_100"],
                    help="timm student backbones to pre-cache (pretrained=True would download on an offline node)")
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
    for b in a.backbones:
        try:
            import timm
            timm.create_model(b, pretrained=True); print("timm backbone cached:", b)
        except Exception as e:  # noqa: BLE001
            failed.append(f"timm {b}: {e}"); print("FAILED timm", b, e)
    print("FAILED:" if failed else "all weights cached", *failed, sep="\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
