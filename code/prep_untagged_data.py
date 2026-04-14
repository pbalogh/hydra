#!/usr/bin/env python3
"""
Prepare untagged (plain) pre-tokenized data for mixed training.
================================================================
Strips all tag tokens from the existing tagged .bin file to produce
a plain-text version using the same GPT-2 tokenizer vocabulary.

The resulting .bin has only word tokens (IDs < 50257), packed
contiguously. This means the same WikiText content but without
any parse/thematic/info-structure tags.

Usage:
    python3 prep_untagged_data.py
"""

import json
import numpy as np
from pathlib import Path

DATA_DIR = Path("/data/hydra/phase05_full_v2")
OUT_DIR = Path("/data/hydra/mixed_training")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_tag_ids():
    """Load tag token IDs from tokenizer JSON."""
    tok_path = DATA_DIR / "tokenizer_all-heads.json"
    with open(tok_path) as f:
        d = json.load(f)
    tag_ids = set()
    for tag, tid in d["tag_to_id"].items():
        tag_ids.add(tid)
    print(f"Loaded {len(tag_ids)} tag IDs (range {min(tag_ids)}-{max(tag_ids)})")
    return tag_ids


def strip_tags(split="train"):
    """Strip tag tokens from a .bin file to produce untagged version."""
    tag_ids = load_tag_ids()
    
    src = DATA_DIR / f"{split}_all-heads.bin"
    dst = OUT_DIR / f"{split}_plain.bin"
    
    print(f"\nProcessing {src}...")
    tokens = np.memmap(src, dtype=np.int32, mode='r')
    print(f"  Source: {len(tokens):,} tokens")
    
    # Filter out tag tokens
    mask = np.ones(len(tokens), dtype=bool)
    for tid in tag_ids:
        mask &= (tokens != tid)
    
    plain_tokens = tokens[mask].copy()
    print(f"  After stripping tags: {len(plain_tokens):,} tokens ({len(plain_tokens)/len(tokens)*100:.1f}%)")
    
    # Save
    plain_tokens.tofile(dst)
    
    # Save metadata
    meta = {
        "n_tokens": int(len(plain_tokens)),
        "n_source_tokens": int(len(tokens)),
        "tag_ratio": round(1 - len(plain_tokens) / len(tokens), 4),
        "dtype": "int32",
        "mode": "plain",
    }
    meta_path = OUT_DIR / f"{split}_plain_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"  Saved to {dst} ({dst.stat().st_size / 1e6:.0f}MB)")
    print(f"  Tag density: {meta['tag_ratio']*100:.1f}%")
    
    return meta


if __name__ == "__main__":
    print("Preparing untagged data for mixed training")
    print("=" * 50)
    
    for split in ["train", "val"]:
        strip_tags(split)
    
    print("\nDone! Untagged data ready in", OUT_DIR)
