#!/usr/bin/env python3
"""
Operator-SAE Alignment Experiment
===================================

Tests whether Hydra's SAE features correlate with operator types from
the unified hierarchy. This is the "holy grail" experiment: if SAE 
features align with operators, then the fiber bundle story is literal —
the model independently learned representations that match our 
theoretically-motivated operator vocabulary.

Pipeline:
  1. Load Hydra model + trained SAE weights
  2. Load sentences with matched operator facts
  3. For each sentence: run through model, extract SAE activations
  4. Label each token with active operators at multiple levels:
     - Primitive: PTRANS, ATRANS, MTRANS, IS_A, PARTOF
     - Transfer type: Emergence, Bond, Creation, Stasis, ...
     - Domain: Biological, Family, Literary, Geographic, ...
     - Full: PTRANS<Emergence,Biological<Human>>
  5. For each (SAE feature, operator label): compute MCC
  6. Report alignment: which features are "operator detectors"?

Requires:
  - Hydra checkpoint (phase 2 best)
  - Trained SAE weights (from SAE interpretability experiment)
  - Augmented training data with operator labels
  - operator_hierarchy_v2.py

Usage:
  python3 operator_sae_alignment.py --checkpoint <path> --sae <path> --data <path>
  python3 operator_sae_alignment.py --extract-only  # just extract activations + labels
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from operator_hierarchy_v2 import classify, all_erasure_levels, OpType, TypeParam


# ═══════════════════════════════════════════════════════════════════════
# TopK Sparse Autoencoder (same architecture as SAE interpretability exp)
# ═══════════════════════════════════════════════════════════════════════

class TopKSAE(torch.nn.Module):
    """TopK sparse autoencoder for extracting interpretable features."""
    
    def __init__(self, d_model: int, d_dict: int, k: int = 10):
        super().__init__()
        self.encoder = torch.nn.Linear(d_model, d_dict, bias=True)
        self.decoder = torch.nn.Linear(d_dict, d_model, bias=True)
        self.k = k
        self.d_dict = d_dict
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input, return sparse activation (only top-k nonzero)."""
        z = self.encoder(x)
        # Keep only top-k activations
        topk_vals, topk_idx = torch.topk(z, self.k, dim=-1)
        sparse = torch.zeros_like(z)
        sparse.scatter_(-1, topk_idx, F.relu(topk_vals))
        return sparse
    
    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, z


# ═══════════════════════════════════════════════════════════════════════
# Label Extraction: Sentence → Operator Labels per Token
# ═══════════════════════════════════════════════════════════════════════

def extract_operator_labels(sentence: dict, facts: List[dict]) -> Dict[str, List[bool]]:
    """
    Given a sentence and its matched facts, produce binary labels 
    for each operator dimension.
    
    Since facts apply to the whole sentence (not individual tokens),
    every token in the sentence gets the same operator labels.
    This is a sentence-level classification, not token-level.
    
    Returns: {label_name: [True/False per token]}
    """
    raw = sentence.get("raw", "")
    n_tokens = len(raw.split())  # approximate
    
    labels = defaultdict(lambda: [False] * n_tokens)
    
    for fact in facts:
        op = classify(
            fact.get("surface_operator", fact.get("operator", "")),
            fact.get("bindings", {}),
            fact.get("property_id")
        )
        
        # Primitive level
        labels[f"prim:{op.primitive}"] = [True] * n_tokens
        
        # Transfer type level (param 0)
        if len(op.params) >= 1:
            labels[f"ttype:{op.params[0].name}"] = [True] * n_tokens
        
        # Domain level (param 1, top-level only)
        if len(op.params) >= 2:
            labels[f"domain:{op.params[1].name}"] = [True] * n_tokens
            # Nested domain
            for child in op.params[1].children:
                labels[f"subdomain:{op.params[1].name}<{child.name}>"] = [True] * n_tokens
        
        # Full type (for fine-grained analysis)
        labels[f"full:{op}"] = [True] * n_tokens
    
    return dict(labels)


# ═══════════════════════════════════════════════════════════════════════
# MCC Computation
# ═══════════════════════════════════════════════════════════════════════

def mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews Correlation Coefficient."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / denom


# ═══════════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════════

def run_alignment(args):
    """
    Full alignment experiment:
    1. Load model + SAE
    2. Process sentences with operator labels
    3. Extract SAE activations
    4. Compute MCC between features and operator labels
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ── Load facts and sentence-fact mapping ──────────────────────────
    print("\nLoading knowledge store...")
    facts = []
    with open(args.facts) as f:
        for line in f:
            facts.append(json.loads(line))
    print(f"  {len(facts)} facts")
    
    sentence_facts = {}
    with open(args.sentence_facts) as f:
        for line in f:
            d = json.loads(line)
            sentence_facts[d["sentence_id"]] = d["fact_ids"]
    print(f"  {len(sentence_facts)} sentence-fact mappings")
    
    # ── Load sentences ────────────────────────────────────────────────
    print("\nLoading sentences with operator labels...")
    
    # Collect all unique operator labels
    all_labels = set()
    labeled_sentences = []
    
    n_sampled = 0
    with open(args.data) as f:
        for line_no, line in enumerate(f):
            if line_no not in sentence_facts:
                continue
            
            d = json.loads(line)
            fact_ids = sentence_facts[line_no]
            matched_facts = [facts[fid] for fid in fact_ids if fid < len(facts)]
            
            if not matched_facts:
                continue
            
            labels = extract_operator_labels(d, matched_facts)
            all_labels.update(labels.keys())
            
            labeled_sentences.append({
                "raw": d.get("raw", ""),
                "line_no": line_no,
                "labels": labels,
                "n_facts": len(matched_facts),
            })
            
            n_sampled += 1
            if n_sampled >= args.max_sentences:
                break
    
    print(f"  {len(labeled_sentences)} labeled sentences")
    print(f"  {len(all_labels)} unique operator labels")
    
    # Report label distribution
    label_counts = Counter()
    for sent in labeled_sentences:
        for label in sent["labels"]:
            label_counts[label] += 1
    
    print(f"\n  Label distribution (top 30):")
    for label, count in label_counts.most_common(30):
        print(f"    {label:45s}: {count:6d} sentences")
    
    # ── Save extracted labels for offline analysis ────────────────────
    if args.extract_only:
        out_path = Path(args.output_dir) / "operator_labels.jsonl"
        with open(out_path, "w") as f:
            for sent in labeled_sentences:
                f.write(json.dumps({
                    "line_no": sent["line_no"],
                    "raw": sent["raw"][:200],
                    "labels": list(sent["labels"].keys()),
                    "n_facts": sent["n_facts"],
                }) + "\n")
        print(f"\n  Saved labels to {out_path}")
        
        # Also save label vocabulary
        with open(Path(args.output_dir) / "operator_label_vocab.json", "w") as f:
            json.dump({
                "labels": sorted(all_labels),
                "counts": dict(label_counts.most_common()),
                "n_sentences": len(labeled_sentences),
                "by_level": {
                    "primitives": sorted(l for l in all_labels if l.startswith("prim:")),
                    "transfer_types": sorted(l for l in all_labels if l.startswith("ttype:")),
                    "domains": sorted(l for l in all_labels if l.startswith("domain:")),
                    "subdomains": sorted(l for l in all_labels if l.startswith("subdomain:")),
                    "full_types": sorted(l for l in all_labels if l.startswith("full:")),
                },
            }, f, indent=2)
        
        print("  Extract-only mode: skipping SAE alignment (no model needed)")
        return
    
    # ── Load model ────────────────────────────────────────────────────
    print("\nLoading Hydra model...")
    # [Model loading code would go here — depends on checkpoint format]
    # For now, we'll set up the framework and fill in model-specific code
    
    print("\n⚠️ Full SAE alignment requires trained SAE weights.")
    print("  Run the SAE training first (from the interpretability experiment),")
    print("  then re-run with --sae <path>.")
    print("\n  For now, we've extracted the operator labels.")
    print("  Next steps:")
    print("    1. Train SAE on 130M model (when training completes)")
    print("    2. Extract activations for labeled sentences")
    print("    3. Compute MCC between SAE features and operator labels")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/data/hydra/phase05_full_v2/hydra_train.jsonl",
                        help="Training data JSONL")
    parser.add_argument("--facts", default="/data/hydra/knowledge_store/facts.jsonl",
                        help="Classified facts JSONL")
    parser.add_argument("--sentence-facts", default="/data/hydra/knowledge_store/sentence_facts.jsonl",
                        help="Sentence-fact mapping JSONL")
    parser.add_argument("--checkpoint", default=None,
                        help="Hydra model checkpoint")
    parser.add_argument("--sae", default=None,
                        help="Trained SAE weights")
    parser.add_argument("--output-dir", default="/data/hydra/knowledge_store/sae_alignment",
                        help="Output directory")
    parser.add_argument("--max-sentences", type=int, default=50000,
                        help="Max sentences to process")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract labels, skip SAE (no model needed)")
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 3, 5, 7],
                        help="Which layers to extract from")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_alignment(args)


if __name__ == "__main__":
    main()
