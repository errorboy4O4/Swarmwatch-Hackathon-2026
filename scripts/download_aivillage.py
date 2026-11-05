"""List (and optionally download) the gated AI Village dataset. Needs env var HF_TOKEN (read-only token).
Usage: python scripts/download_aivillage.py list
       python scripts/download_aivillage.py get <path_in_repo> [more paths...]"""
import os, sys
from huggingface_hub import HfApi, hf_hub_download

REPO = "aidigestorg/ai-village"
tok = os.environ.get("HF_TOKEN")
if not tok:
    sys.exit("Set HF_TOKEN (read-only Hugging Face token) first.")
if sys.argv[1] == "list":
    for f in HfApi(token=tok).list_repo_tree(REPO, repo_type="dataset", recursive=True):
        print(getattr(f, "size", None), f.path)
else:
    for p in sys.argv[2:]:
        print(hf_hub_download(REPO, p, repo_type="dataset", token=tok, local_dir="data/aivillage"))
