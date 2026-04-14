#!/usr/bin/env python3
"""
Vectorized SAE Operator Alignment
==================================

Fast, memory-efficient alignment of SAE features against operator labels.
Uses numpy broadcasting to compute MCC for all features simultaneously
instead of looping per-feature.

Memory strategy: loads one layer at a time, processes in feature chunks
to stay under ~2GB RAM even with 500K tokens × 3072 features.

Usage:
    python3 alignment_fast.py --step 25000 [--chunk-size 512]
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch


# ── Config ────────────────────────────────────────────────────────────
OUTPUT_BASE = "/data/hydra/sae_130m"
FACTS_FILE = "/data/hydra/knowledge_store_augmented.jsonl"
VAL_BIN = "/data/hydra/val_tokens.bin"

# Import operator classifier from the pipeline
sys.path.insert(0, "/data/hydra")
from auto_sae_pipeline import classify, TopKSAE, D_MODEL


def build_labels(step: int):
    """Build token-level operator labels. Returns (labels, all_labels, valid_indices, meta)."""
    act_dir = f"{OUTPUT_BASE}/step{step}/activations"
    
    with open(f"{act_dir}/meta.json") as f:
        meta = json.load(f)
    
    n_tokens = meta["n_tokens"]
    
    # Build entity→operator mapping
    facts = [json.loads(l) for l in open(FACTS_FILE)]
    entity_to_ops = defaultdict(set)
    for f_item in facts:
        op = classify(f_item.get("surface_operator", f_item.get("operator", "")),
                      f_item.get("bindings", {}), f_item.get("property_id"))
        for role, ent in f_item.get("bindings", {}).items():
            if ent and len(ent) > 3:
                el = ent.lower()
                entity_to_ops[el].add("prim:" + op.primitive)
                if len(op.params) >= 1:
                    entity_to_ops[el].add("ttype:" + op.params[0].name)
                if len(op.params) >= 2:
                    entity_to_ops[el].add("domain:" + op.params[1].name)
    
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    
    # Pre-tokenize entities
    entity_tokens = {}
    entity_tokens_space = {}
    for ent in entity_to_ops:
        try:
            entity_tokens[ent] = tok.encode(ent)
            entity_tokens_space[ent] = tok.encode(" " + ent)
        except Exception:
            pass
    
    val_tokens = np.memmap(VAL_BIN, dtype=np.int32, mode="r")
    tokens_arr = np.array(val_tokens[:n_tokens], dtype=np.int64)
    
    all_labels = sorted(set(l for ops in entity_to_ops.values() for l in ops))
    l2i = {l: i for i, l in enumerate(all_labels)}
    nl = len(all_labels)
    
    print(f"Building labels ({n_tokens} x {nl})...", flush=True)
    labels = np.zeros((n_tokens, nl), dtype=np.float32)
    
    for ent_idx, (ent, ops) in enumerate(entity_to_ops.items()):
        if ent_idx % 5000 == 0:
            print(f"    Entity {ent_idx}/{len(entity_to_ops)}...", flush=True)
        op_indices = [l2i[op] for op in ops if op in l2i]
        if not op_indices:
            continue
        for token_map in [entity_tokens, entity_tokens_space]:
            if ent not in token_map:
                continue
            ent_ids = token_map[ent]
            if not ent_ids or len(ent_ids) > 20:
                continue
            if len(ent_ids) == 1:
                matches = np.where(tokens_arr == ent_ids[0])[0]
                for pos in matches:
                    for li in op_indices:
                        labels[pos, li] = 1.0
            else:
                first_tok = ent_ids[0]
                candidates = np.where(tokens_arr == first_tok)[0]
                for pos in candidates:
                    if pos + len(ent_ids) <= n_tokens:
                        if all(tokens_arr[pos + j] == ent_ids[j] for j in range(len(ent_ids))):
                            for offset in range(len(ent_ids)):
                                for li in op_indices:
                                    labels[pos + offset, li] = 1.0
    
    lsums = labels.sum(axis=0)
    valid = np.where(lsums > 50)[0]
    total_labeled = int((labels.sum(axis=1) > 0).sum())
    print(f"  {total_labeled}/{n_tokens} labeled, {len(valid)} valid labels", flush=True)
    
    return labels, all_labels, valid, meta


def vectorized_mcc(feat_active: np.ndarray, labels: np.ndarray, 
                    valid: np.ndarray) -> dict:
    """
    Compute MCC for all (feature, label) pairs using vectorized numpy.
    
    feat_active: (n_tokens, n_features) bool
    labels: (n_tokens, n_labels) float32
    valid: indices into label axis with enough positives
    
    Returns dict mapping label_name_index -> {best_mcc, best_feature, P, R}
    """
    n_tokens, n_feat = feat_active.shape
    
    # Convert to float64 for precision
    F = feat_active.astype(np.float64)  # (T, F)
    
    results = {}
    
    for li_idx, li in enumerate(valid):
        Y = labels[:, li].astype(np.float64)  # (T,)
        n_pos = Y.sum()
        n_neg = n_tokens - n_pos
        
        if n_pos == 0 or n_neg == 0:
            results[int(li)] = {"mcc": 0.0, "feature": -1, "P": 0.0, "R": 0.0, "n": int(n_pos)}
            continue
        
        # Vectorized: compute TP, FP, FN, TN for ALL features at once
        # TP[f] = sum(Y * F[:, f])  →  Y @ F = (T,) @ (T, F) = (F,)
        TP = Y @ F                    # (F,)
        FP = (1 - Y) @ F             # (F,)
        FN = n_pos - TP              # (F,)
        TN = n_neg - FP              # (F,)
        
        # MCC denominator
        denom = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        denom = np.where(denom > 0, denom, 1.0)  # avoid division by zero
        
        mcc_all = (TP * TN - FP * FN) / denom  # (F,)
        
        # Find best feature (by absolute MCC)
        best_idx = int(np.argmax(np.abs(mcc_all)))
        best_mcc = float(mcc_all[best_idx])
        
        P = float(TP[best_idx] / (TP[best_idx] + FP[best_idx])) if (TP[best_idx] + FP[best_idx]) > 0 else 0.0
        R = float(TP[best_idx] / (TP[best_idx] + FN[best_idx])) if (TP[best_idx] + FN[best_idx]) > 0 else 0.0
        
        results[int(li)] = {
            "mcc": round(best_mcc, 4),
            "feature": best_idx,
            "P": round(P, 4),
            "R": round(R, 4),
            "n": int(n_pos),
        }
    
    return results


def run_alignment(step: int, chunk_size: int = 512):
    """Run alignment for all layers."""
    
    act_dir = f"{OUTPUT_BASE}/step{step}/activations"
    weight_dir = f"{OUTPUT_BASE}/step{step}/sae_weights"
    result_dir = f"{OUTPUT_BASE}/step{step}/results"
    os.makedirs(result_dir, exist_ok=True)
    
    labels, all_labels, valid, meta = build_labels(step)
    n_tokens = meta["n_tokens"]
    lsums = labels.sum(axis=0)
    
    print(f"\n{'='*60}", flush=True)
    print(f"FAST ALIGNMENT (step {step})", flush=True)
    print(f"{'='*60}", flush=True)
    
    all_results = {}
    
    for layer_idx in meta["layers"]:
        t0 = time.time()
        print(f"\n  === Layer {layer_idx} ===", flush=True)
        
        sae_path = f"{weight_dir}/sae_layer_{layer_idx}.pt"
        if not os.path.exists(sae_path):
            print(f"    SKIP (no weights)", flush=True)
            continue
        
        # Load SAE
        sd = torch.load(sae_path, map_location="cpu", weights_only=False)
        sae = TopKSAE(D_MODEL, sd["d_dict"], sd["k"])
        sae.load_state_dict(sd["state_dict"])
        sae.eval()
        
        # Load activations
        acts = np.load(f"{act_dir}/layer_{layer_idx}.npy")
        acts_t = torch.tensor(acts, dtype=torch.float32)
        acts_norm = (acts_t - sd["mean"]) / sd["std"]
        del acts, acts_t
        gc.collect()
        
        # Encode through SAE in chunks to get feature activations
        # Build sparse boolean array chunk by chunk
        d_dict = sd["d_dict"]
        print(f"    Encoding {n_tokens} tokens through SAE ({d_dict} features)...", flush=True)
        
        # Process in token chunks to build feature activity
        feat_active = np.zeros((n_tokens, d_dict), dtype=np.bool_)
        token_chunk = 50000
        
        with torch.no_grad():
            for s in range(0, n_tokens, token_chunk):
                e = min(s + token_chunk, n_tokens)
                z = sae.encode(acts_norm[s:e])
                feat_active[s:e] = (z > 0).numpy()
        
        del acts_norm
        gc.collect()
        
        # Filter to active features (>50 activations)
        fsums = feat_active.sum(axis=0)
        active_mask = fsums > 50
        n_active = int(active_mask.sum())
        print(f"    Active features: {n_active}/{d_dict}", flush=True)
        
        # Extract only active features to reduce memory
        active_indices = np.where(active_mask)[0]
        feat_subset = feat_active[:, active_indices]  # (T, n_active)
        del feat_active
        gc.collect()
        
        # Vectorized MCC computation
        print(f"    Computing MCC ({n_active} features × {len(valid)} labels)...", flush=True)
        raw_results = vectorized_mcc(feat_subset, labels, valid)
        del feat_subset
        gc.collect()
        
        # Map back to original feature indices
        best = {}
        for li, info in raw_results.items():
            ln = all_labels[li]
            # Map subset index back to original feature index
            orig_feat = int(active_indices[info["feature"]]) if info["feature"] >= 0 else -1
            best[ln] = {
                "mcc": info["mcc"],
                "feature": orig_feat,
                "n": info["n"],
                "P": info["P"],
                "R": info["R"],
            }
        
        # Print results
        for l_name in sorted(best.keys(), key=lambda x: -abs(best[x]["mcc"])):
            info = best[l_name]
            if abs(info["mcc"]) > 0.01:
                print(f"    {l_name:40s}: MCC={info['mcc']:+.4f} P={info['P']:.3f} "
                      f"R={info['R']:.3f} (feat {info['feature']}, n={info['n']})", flush=True)
        
        aligned = sum(1 for v in best.values() if abs(v["mcc"]) > 0.1)
        moderate = sum(1 for v in best.values() if abs(v["mcc"]) > 0.05)
        mean_abs = float(np.mean([abs(v["mcc"]) for v in best.values()])) if best else 0
        max_mcc = max(abs(v["mcc"]) for v in best.values()) if best else 0
        elapsed = time.time() - t0
        
        print(f"    Strong: {aligned}/{len(best)}, Moderate: {moderate}/{len(best)}, "
              f"Mean: {mean_abs:.4f}, Max: {max_mcc:.4f} [{elapsed:.0f}s]", flush=True)
        
        all_results[f"layer_{layer_idx}"] = {
            "strong": aligned, "moderate": moderate, "total": len(best),
            "mean_mcc": round(mean_abs, 4), "max_mcc": round(max_mcc, 4),
            "elapsed_sec": round(elapsed, 1),
            "per_label": best,
        }
    
    # Save results
    with open(f"{result_dir}/alignment_fast.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY — 130M Step {step}", flush=True)
    print(f"{'='*60}", flush=True)
    for lname, r in sorted(all_results.items()):
        print(f"  {lname}: strong={r['strong']}/{r['total']}, "
              f"mean={r['mean_mcc']:.4f}, max={r['max_mcc']:.4f} "
              f"[{r['elapsed_sec']:.0f}s]", flush=True)
    
    # Compare with 39.3M baseline if available
    baseline_path = "/data/hydra/sae_operators/results/operator_alignment_v2.json"
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        print(f"\n  vs 39.3M baseline:", flush=True)
        for lname in sorted(all_results.keys()):
            if lname in baseline:
                new = all_results[lname]
                old = baseline[lname]
                delta = new["mean_mcc"] - old["mean_mcc"]
                print(f"    {lname}: {old['mean_mcc']:.4f} → {new['mean_mcc']:.4f} "
                      f"({'+' if delta >= 0 else ''}{delta:.4f})", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=25000)
    parser.add_argument("--chunk-size", type=int, default=512, help="(unused, kept for compat)")
    args = parser.parse_args()
    
    run_alignment(args.step)
