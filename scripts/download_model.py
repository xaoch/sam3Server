#!/usr/bin/env python3
"""Download a Hugging Face model repo (facebook/sam3) to a local folder.

Usage (Linux / WSL):
  HUGGINGFACE_HUB_TOKEN="hf_xxx" python3 scripts/download_model.py --out ./model_cache

Windows PowerShell (temporary env):
  $env:HUGGINGFACE_HUB_TOKEN = 'hf_xxx'; python scripts\download_model.py --out .\model_cache

Or store token in a file at ~/.huggingface/token and run without passing the token.
"""
import argparse
import os
from huggingface_hub import snapshot_download


def load_token_from_files() -> str | None:
    paths = [os.path.expanduser('~/.huggingface/token'), '/root/.huggingface/token']
    for p in paths:
        try:
            if os.path.exists(p):
                with open(p, 'r') as f:
                    t = f.read().strip()
                    if t:
                        return t
        except Exception:
            continue
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default='facebook/sam3', help='repo id on huggingface')
    p.add_argument('--out', default='./model_cache', help='output/cache directory')
    p.add_argument('--token', default=None, help='Hugging Face token (optional)')
    args = p.parse_args()

    token = args.token or os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
    if not token:
        token = load_token_from_files()

    if token:
        # set env so downstream libraries pick it up as well
        os.environ.setdefault('HUGGINGFACE_HUB_TOKEN', token)

    os.makedirs(args.out, exist_ok=True)
    print(f'Downloading {args.repo} to {args.out} ...')
    path = snapshot_download(repo_id=args.repo, cache_dir=args.out, resume_download=True)
    print('Downloaded to', path)


if __name__ == '__main__':
    main()
