#!/usr/bin/env python3
"""
SAE Training + Operator Alignment Pipeline
============================================

Trains TopK SAEs on Hydra model activations, then tests alignment
with operator labels from the unified hierarchy.

Phase 1: Extract activations from Hydra model (GPU-light, mostly I/O)
Phase 2: Train SAEs on extracted activations (CPU-friendly)
Phase 3: Compute operator alignment (MCC between features and labels)

Usage:
  python3 train_sae_operators.py --phase extract --checkpoint <path>
  python3 train_sae_operators.py --phase train-sae
  python3 train_sae_operators.py --phase align
  python3 train_sae_operators.py --phase all --checkpoint <path>
"""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

WORK_DIR = Path("/data/hydra/sae_operators")
ACTS_DIR = WORK_DIR / "activations"
SAE_DIR = WORK_DIR / "sae_weights"
RESULTS_DIR = WORK_DIR / "results"

LAYERS_TO_EXTRACT = [0, 1, 3, 5, 7]  # Mamba layers
MAX_TOKENS = 500_000    # token positions to extract
SAE_DICT_MULT = 4       # dictionary size = d_model * this
SAE_K = 10              # top-k sparsity
SAE_EPOCHS = 5
SAE_LR = 1e-3
SAE_L1 = 1e-4
SAE_BATCH = 512


# ═══════════════════════════════════════════════════════════════════════
# TopK SAE
# ═══════════════════════════════════════════════════════════════════════

class TopKSAE(nn.Module):
    def __init__(self, d_model, d_dict, k=10):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_dict)
        self.decoder = nn.Linear(d_dict, d_model)
        self.k = k
        self.d_dict = d_dict
        # Initialize decoder columns to unit norm
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True))
    
    def encode(self, x):
        z = self.encoder(x)
        topk_vals, topk_idx = torch.topk(z, self.k, dim=-1)
        sparse = torch.zeros_like(z)
        sparse.scatter_(-1, topk_idx, F.relu(topk_vals))
        return sparse
    
    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, z
    
    def loss(self, x):
        x_hat, z = self(x)
        recon = F.mse_loss(x_hat, x)
        l1 = z.abs().mean()
        return recon + SAE_L1 * l1, recon.item(), l1.item()


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Extract Activations
# ═══════════════════════════════════════════════════════════════════════

def extract_activations(checkpoint_path: str):
    """Extract hidden states from Hydra model at multiple layers."""
    print("=" * 60)
    print("Phase 1: Extracting Activations")
    print("=" * 60)
    
    ACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    d_model = config.get("d_model", 512)
    n_layer = config.get("n_layer", 8)
    vocab_size = config.get("vocab_size", 50304)
    
    # Detect actual vocab size from checkpoint weights
    model_state = ckpt["model"]
    for key in model_state:
        if "embedding" in key and "weight" in key:
            actual_vocab = model_state[key].shape[0]
            print(f"  Actual vocab from checkpoint: {actual_vocab}")
            vocab_size = actual_vocab
            break
    
    print(f"  d_model={d_model}, n_layer={n_layer}, vocab={vocab_size}")
    
    # Build model
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from mamba_ssm.models.config_mamba import MambaConfig
    mamba_config = MambaConfig(
        d_model=d_model,
        n_layer=n_layer,
        vocab_size=vocab_size,
    )
    model = MambaLMHeadModel(mamba_config).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    
    # Load tokenized data
    data_path = Path("/data/hydra/phase05_full_v2/val_all-heads.bin")
    meta_path = Path("/data/hydra/phase05_full_v2/val_all-heads_meta.json")
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    tokens = np.memmap(data_path, dtype=np.int32, mode='r')
    n_tokens = min(len(tokens), MAX_TOKENS * 512)  # enough sequences
    
    print(f"\n  Extracting from {n_tokens:,} tokens...")
    
    # Hook to capture intermediate activations
    layer_activations = {l: [] for l in LAYERS_TO_EXTRACT if l < n_layer}
    hooks = []
    
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # Mamba layer output is (hidden_states, ...)
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            layer_activations[layer_idx].append(h.detach().cpu())
        return hook_fn
    
    # Register hooks on Mamba layers
    for layer_idx in LAYERS_TO_EXTRACT:
        if layer_idx < n_layer:
            layer = model.backbone.layers[layer_idx]
            hooks.append(layer.register_forward_hook(make_hook(layer_idx)))
    
    # Process sequences
    seq_len = 512
    n_seqs = min(n_tokens // seq_len, MAX_TOKENS // seq_len)
    batch_size = 4  # small batches to save memory
    
    token_count = 0
    
    with torch.no_grad():
        for i in range(0, n_seqs, batch_size):
            batch_end = min(i + batch_size, n_seqs)
            batch_tokens = []
            for j in range(i, batch_end):
                start = j * seq_len
                seq = tokens[start:start + seq_len].astype(np.int64)
                batch_tokens.append(seq)
            
            input_ids = torch.tensor(np.array(batch_tokens), dtype=torch.long, device=device)
            _ = model(input_ids)
            
            token_count += input_ids.numel()
            
            if token_count >= MAX_TOKENS:
                break
            
            if (i // batch_size) % 50 == 0:
                print(f"    {token_count:,} tokens extracted...")
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Save activations
    print(f"\n  Saving activations ({token_count:,} tokens)...")
    for layer_idx, acts_list in layer_activations.items():
        if not acts_list:
            continue
        acts = torch.cat(acts_list, dim=0)  # (n_seqs, seq_len, d_model)
        acts = acts.reshape(-1, d_model)[:MAX_TOKENS]  # (n_tokens, d_model)
        
        out_path = ACTS_DIR / f"layer_{layer_idx}.npy"
        np.save(out_path, acts.numpy())
        print(f"    Layer {layer_idx}: {acts.shape} → {out_path}")
    
    # Save metadata
    with open(ACTS_DIR / "meta.json", "w") as f:
        json.dump({
            "d_model": d_model,
            "n_layer": n_layer,
            "n_tokens": token_count,
            "layers": list(layer_activations.keys()),
            "seq_len": seq_len,
            "checkpoint": checkpoint_path,
        }, f, indent=2)
    
    print(f"\n  Done: {token_count:,} tokens across {len(layer_activations)} layers")


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Train SAEs
# ═══════════════════════════════════════════════════════════════════════

def train_saes():
    """Train TopK SAEs on extracted activations."""
    print("=" * 60)
    print("Phase 2: Training SAEs")
    print("=" * 60)
    
    SAE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    with open(ACTS_DIR / "meta.json") as f:
        meta = json.load(f)
    
    d_model = meta["d_model"]
    d_dict = d_model * SAE_DICT_MULT
    device = torch.device("cpu")  # CPU training as requested
    
    print(f"  d_model={d_model}, d_dict={d_dict}, k={SAE_K}")
    print(f"  Training on CPU")
    
    for layer_idx in meta["layers"]:
        print(f"\n  --- Layer {layer_idx} ---")
        
        # Load activations
        acts = np.load(ACTS_DIR / f"layer_{layer_idx}.npy")
        acts_tensor = torch.tensor(acts, dtype=torch.float32)
        
        # Normalize
        mean = acts_tensor.mean(dim=0)
        std = acts_tensor.std(dim=0).clamp(min=1e-6)
        acts_norm = (acts_tensor - mean) / std
        
        dataset = TensorDataset(acts_norm)
        loader = DataLoader(dataset, batch_size=SAE_BATCH, shuffle=True, num_workers=0)
        
        # Train SAE
        sae = TopKSAE(d_model, d_dict, SAE_K).to(device)
        optimizer = torch.optim.Adam(sae.parameters(), lr=SAE_LR)
        
        for epoch in range(SAE_EPOCHS):
            total_loss = 0
            total_recon = 0
            total_l1 = 0
            n_batches = 0
            
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss, recon, l1 = sae.loss(batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_recon += recon
                total_l1 += l1
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            avg_recon = total_recon / n_batches
            avg_l1 = total_l1 / n_batches
            print(f"    Epoch {epoch+1}/{SAE_EPOCHS}: loss={avg_loss:.4f} "
                  f"(recon={avg_recon:.4f}, l1={avg_l1:.4f})")
        
        # Save SAE weights + normalization params
        save_path = SAE_DIR / f"sae_layer_{layer_idx}.pt"
        torch.save({
            "state_dict": sae.state_dict(),
            "d_model": d_model,
            "d_dict": d_dict,
            "k": SAE_K,
            "mean": mean,
            "std": std,
            "layer": layer_idx,
        }, save_path)
        print(f"    Saved: {save_path}")
    
    print(f"\n  Done: SAEs trained for {len(meta['layers'])} layers")


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Operator Alignment
# ═══════════════════════════════════════════════════════════════════════

def compute_alignment():
    """Compute MCC between SAE features and operator labels."""
    print("=" * 60)
    print("Phase 3: Operator-SAE Alignment")
    print("=" * 60)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    from operator_hierarchy_v2 import classify
    
    # Load metadata
    with open(ACTS_DIR / "meta.json") as f:
        meta = json.load(f)
    
    d_model = meta["d_model"]
    
    # Load facts and sentence-fact mapping
    print("\nLoading knowledge store...")
    facts = []
    with open("/data/hydra/knowledge_store/facts.jsonl") as f:
        for line in f:
            facts.append(json.loads(line))
    
    sentence_facts = {}
    with open("/data/hydra/knowledge_store/sentence_facts.jsonl") as f:
        for line in f:
            d = json.loads(line)
            sentence_facts[d["sentence_id"]] = d["fact_ids"]
    
    # Build token-level operator labels from validation data
    # We need to match token positions in the activation arrays to operator labels
    print("\nBuilding operator labels for extracted token positions...")
    
    # Load validation data to get sentence boundaries
    val_data_path = Path("/data/hydra/phase05_full_v2/val_all-heads.bin")
    val_tokens = np.memmap(val_data_path, dtype=np.int32, mode='r')
    
    # Load tokenizer to identify sentence boundaries
    with open("/data/hydra/phase05_full_v2/tokenizer_all-heads.json") as f:
        tokenizer = json.load(f)
    tag_ids = set(tokenizer.get("tag_to_id", {}).values())
    
    # For each token position in extracted activations, determine operator labels
    # Approach: match val sequences to training sentence_facts by content overlap
    # Simpler approach: since we have 440K matched sentences in TRAINING data,
    # we can also match operator labels to val sentences by entity overlap
    
    # Build entity → operator mapping
    entity_to_ops = defaultdict(set)
    for fact in facts:
        prop_id = fact.get("property_id")
        surface_op = fact.get("surface_operator", fact.get("operator", ""))
        bindings = fact.get("bindings", {})
        
        op = classify(surface_op, bindings, prop_id)
        
        # Index by entity names
        for role, entity in bindings.items():
            if entity and len(entity) > 3:
                entity_lower = entity.lower()
                entity_to_ops[entity_lower].add(f"prim:{op.primitive}")
                if len(op.params) >= 1:
                    entity_to_ops[entity_lower].add(f"ttype:{op.params[0].name}")
                if len(op.params) >= 2:
                    entity_to_ops[entity_lower].add(f"domain:{op.params[1].name}")
    
    print(f"  Entity→operator index: {len(entity_to_ops)} entities")
    
    # For each activation token position, we need the corresponding text
    # to look up entity mentions. This requires decoding the token IDs.
    # For now, we'll use a simpler approach: random sampling with entity matching.
    
    # Load GPT-2 tokenizer for decoding
    from transformers import AutoTokenizer
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    
    seq_len = meta["seq_len"]
    n_tokens = meta["n_tokens"]
    n_seqs = n_tokens // seq_len
    
    # Build labels array: for each token position, which operators are active
    all_labels = sorted(set(l for ops in entity_to_ops.values() for l in ops))
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    n_labels = len(all_labels)
    
    print(f"  {n_labels} unique operator labels")
    
    # Labels matrix: (n_tokens, n_labels), binary
    labels_matrix = np.zeros((n_tokens, n_labels), dtype=np.float32)
    
    # For each sequence, decode and look up entities
    matched_tokens = 0
    for seq_idx in range(n_seqs):
        start = seq_idx * seq_len
        seq_tokens = val_tokens[start:start + seq_len].astype(np.int64)
        
        # Decode non-tag tokens to text
        word_tokens = [t for t in seq_tokens if t < 50257 and t > 0]
        if not word_tokens:
            continue
        
        try:
            text = gpt2_tok.decode(word_tokens).lower()
        except:
            continue
        
        # Find matching entities
        seq_ops = set()
        for entity, ops in entity_to_ops.items():
            if entity in text:
                seq_ops.update(ops)
        
        if seq_ops:
            for op_label in seq_ops:
                if op_label in label_to_idx:
                    idx = label_to_idx[op_label]
                    # Label ALL token positions in this sequence
                    token_start = seq_idx * seq_len
                    token_end = min(token_start + seq_len, n_tokens)
                    labels_matrix[token_start:token_end, idx] = 1.0
            matched_tokens += seq_len
    
    print(f"  {matched_tokens:,} / {n_tokens:,} tokens have operator labels")
    
    # Label distribution
    label_sums = labels_matrix.sum(axis=0)
    print(f"\n  Label frequencies:")
    for i, label in enumerate(all_labels):
        if label_sums[i] > 0:
            print(f"    {label:40s}: {int(label_sums[i]):8d} tokens ({100*label_sums[i]/n_tokens:.1f}%)")
    
    # ── Compute MCC for each (layer, feature, label) ─────────────────
    print(f"\n  Computing alignment...")
    
    all_results = {}
    
    for layer_idx in meta["layers"]:
        print(f"\n  --- Layer {layer_idx} ---")
        
        # Load SAE
        sae_path = SAE_DIR / f"sae_layer_{layer_idx}.pt"
        if not sae_path.exists():
            print(f"    SAE not found: {sae_path}")
            continue
        
        sae_data = torch.load(sae_path, map_location="cpu", weights_only=False)
        d_dict = sae_data["d_dict"]
        sae = TopKSAE(d_model, d_dict, sae_data["k"])
        sae.load_state_dict(sae_data["state_dict"])
        sae.eval()
        
        mean = sae_data["mean"]
        std = sae_data["std"]
        
        # Load activations
        acts = np.load(ACTS_DIR / f"layer_{layer_idx}.npy")
        acts_tensor = torch.tensor(acts, dtype=torch.float32)
        acts_norm = (acts_tensor - mean) / std
        
        # Encode through SAE
        with torch.no_grad():
            # Process in chunks to save memory
            chunk_size = 10000
            all_features = []
            for start in range(0, len(acts_norm), chunk_size):
                chunk = acts_norm[start:start + chunk_size]
                features = sae.encode(chunk)
                all_features.append((features > 0).numpy().astype(np.float32))
            
            feature_matrix = np.concatenate(all_features, axis=0)  # (n_tokens, d_dict)
        
        print(f"    Feature matrix: {feature_matrix.shape}")
        print(f"    Active features per token: {feature_matrix.sum(axis=1).mean():.1f}")
        
        # Compute MCC for each (feature, label) pair
        # Only consider features that are active in at least 100 tokens
        feature_sums = feature_matrix.sum(axis=0)
        active_features = np.where(feature_sums > 100)[0]
        print(f"    Active features (>100 tokens): {len(active_features)} / {d_dict}")
        
        # Only consider labels with enough positive examples
        valid_labels = np.where(label_sums > 100)[0]
        print(f"    Valid labels (>100 tokens): {len(valid_labels)} / {n_labels}")
        
        best_per_label = {}
        
        for li in valid_labels:
            label_name = all_labels[li]
            y_true = labels_matrix[:, li]
            
            best_mcc = 0
            best_feature = -1
            
            for fi in active_features:
                y_pred = feature_matrix[:, fi]
                
                # Fast MCC computation
                tp = np.sum(y_true * y_pred)
                tn = np.sum((1 - y_true) * (1 - y_pred))
                fp = np.sum((1 - y_true) * y_pred)
                fn = np.sum(y_true * (1 - y_pred))
                
                denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                if denom > 0:
                    m = (tp * tn - fp * fn) / denom
                    if abs(m) > abs(best_mcc):
                        best_mcc = m
                        best_feature = int(fi)
            
            best_per_label[label_name] = {
                "mcc": round(best_mcc, 4),
                "feature": best_feature,
                "n_positive": int(label_sums[li]),
            }
        
        # Report
        print(f"\n    Best feature per operator label:")
        for label, info in sorted(best_per_label.items(), key=lambda x: -abs(x[1]["mcc"])):
            if abs(info["mcc"]) > 0.05:
                print(f"      {label:40s}: MCC={info['mcc']:+.3f} "
                      f"(feature {info['feature']}, n={info['n_positive']})")
        
        # Overall alignment score
        aligned = sum(1 for v in best_per_label.values() if abs(v["mcc"]) > 0.1)
        total = len(best_per_label)
        mean_mcc = np.mean([abs(v["mcc"]) for v in best_per_label.values()]) if best_per_label else 0
        
        print(f"\n    Alignment: {aligned}/{total} labels with |MCC|>0.1")
        print(f"    Mean |MCC|: {mean_mcc:.3f}")
        
        all_results[f"layer_{layer_idx}"] = {
            "aligned": aligned,
            "total": total,
            "mean_mcc": round(float(mean_mcc), 4),
            "per_label": best_per_label,
        }
    
    # Save results
    with open(RESULTS_DIR / "operator_alignment.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n  Results saved to {RESULTS_DIR / 'operator_alignment.json'}")
    
    # Summary across layers
    print(f"\n{'='*60}")
    print("SUMMARY: Operator-SAE Alignment")
    print(f"{'='*60}")
    for layer_name, results in sorted(all_results.items()):
        print(f"  {layer_name}: {results['aligned']}/{results['total']} aligned, "
              f"mean |MCC|={results['mean_mcc']:.3f}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["extract", "train-sae", "align", "all"],
                        default="all")
    parser.add_argument("--checkpoint",
                        default="/data/hydra/checkpoints_phase2/hydra_all-heads_best.pt")
    args = parser.parse_args()
    
    for d in [WORK_DIR, ACTS_DIR, SAE_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    if args.phase in ("extract", "all"):
        extract_activations(args.checkpoint)
    
    if args.phase in ("train-sae", "all"):
        train_saes()
    
    if args.phase in ("align", "all"):
        compute_alignment()


if __name__ == "__main__":
    main()
