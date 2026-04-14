#!/usr/bin/env python3
"""SAE-operator alignment v2: token-level labels instead of sequence-level.
Only labels the actual tokens where an entity appears, not the whole sequence."""
import json, math, sys, gc
import numpy as np
import torch
sys.path.insert(0, ".")
from operator_hierarchy_v2 import classify
from train_sae_operators import TopKSAE
from collections import defaultdict

print("Loading facts...", flush=True)
facts = [json.loads(l) for l in open("knowledge_store/facts.jsonl")]

entity_to_ops = defaultdict(set)
for f in facts:
    op = classify(f.get("surface_operator", f.get("operator", "")),
                  f.get("bindings", {}), f.get("property_id"))
    for role, ent in f.get("bindings", {}).items():
        if ent and len(ent) > 3:
            el = ent.lower()
            entity_to_ops[el].add("prim:" + op.primitive)
            if len(op.params) >= 1:
                entity_to_ops[el].add("ttype:" + op.params[0].name)
            if len(op.params) >= 2:
                entity_to_ops[el].add("domain:" + op.params[1].name)

print(f"  {len(entity_to_ops)} entity keys", flush=True)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")

# Pre-tokenize entities for token-level matching
print("Pre-tokenizing entities...", flush=True)
entity_token_ids = {}  # entity -> list of token id sequences
for ent in entity_to_ops:
    try:
        ids = tok.encode(ent)
        if len(ids) >= 1:
            entity_token_ids[ent] = ids
    except Exception:
        pass

# Also tokenize with leading space (GPT-2 tokenizer quirk)
entity_token_ids_space = {}
for ent in entity_to_ops:
    try:
        ids = tok.encode(" " + ent)
        if len(ids) >= 1:
            entity_token_ids_space[ent] = ids
    except Exception:
        pass
print(f"  {len(entity_token_ids)} entities tokenized", flush=True)

val_tokens = np.memmap("phase05_full_v2/val_all-heads.bin", dtype=np.int32, mode="r")
with open("sae_operators/activations/meta.json") as f:
    meta = json.load(f)
seq_len = meta["seq_len"]
n_tokens = meta["n_tokens"]
d_model = meta.get("d_model", 512)
n_seqs = n_tokens // seq_len

all_labels = sorted(set(l for ops in entity_to_ops.values() for l in ops))
l2i = {l: i for i, l in enumerate(all_labels)}
nl = len(all_labels)

print(f"Building TOKEN-LEVEL label matrix ({n_tokens} tokens x {nl} labels)...", flush=True)
labels = np.zeros((n_tokens, nl), dtype=np.float32)
tokens_arr = np.array(val_tokens[:n_tokens], dtype=np.int64)

# For each entity, find token-level matches using subsequence search
labeled_count = 0
for ent_idx, (ent, ops) in enumerate(entity_to_ops.items()):
    if ent_idx % 2000 == 0:
        print(f"  Entity {ent_idx}/{len(entity_to_ops)}...", flush=True)
    
    op_indices = [l2i[op] for op in ops if op in l2i]
    if not op_indices:
        continue
    
    # Try both with and without leading space
    for token_map in [entity_token_ids, entity_token_ids_space]:
        if ent not in token_map:
            continue
        ent_ids = token_map[ent]
        ent_len = len(ent_ids)
        if ent_len == 0 or ent_len > 20:  # skip very long entities
            continue
        
        if ent_len == 1:
            # Single-token entity: fast numpy match
            matches = np.where(tokens_arr == ent_ids[0])[0]
            for pos in matches:
                for li in op_indices:
                    labels[pos, li] = 1.0
                labeled_count += len(matches)
        else:
            # Multi-token: sliding window
            first_tok = ent_ids[0]
            candidates = np.where(tokens_arr == first_tok)[0]
            for pos in candidates:
                if pos + ent_len <= n_tokens:
                    if all(tokens_arr[pos + j] == ent_ids[j] for j in range(ent_len)):
                        for offset in range(ent_len):
                            for li in op_indices:
                                labels[pos + offset, li] = 1.0
                        labeled_count += ent_len

lsums = labels.sum(axis=0)
total_labeled = int((labels.sum(axis=1) > 0).sum())
valid = np.where(lsums > 50)[0]  # lower threshold since token-level is sparser

print(f"\n  Token-level labeling complete:", flush=True)
print(f"  {total_labeled}/{n_tokens} tokens have ANY label ({100*total_labeled/n_tokens:.1f}%)", flush=True)
print(f"  {nl} labels, {len(valid)} with >50 tokens", flush=True)

for li in valid:
    pct = 100 * lsums[li] / n_tokens
    print(f"    {all_labels[li]:40s}: {int(lsums[li]):8d} tokens ({pct:.2f}%)", flush=True)

# Process one layer at a time
all_results = {}

for layer_idx in meta["layers"]:
    print(f"\n{'='*60}", flush=True)
    print(f"Layer {layer_idx}", flush=True)
    print(f"{'='*60}", flush=True)
    gc.collect()

    sd = torch.load(f"sae_operators/sae_weights/sae_layer_{layer_idx}.pt",
                     map_location="cpu", weights_only=False)
    dd = sd["d_dict"]
    sae = TopKSAE(d_model, dd, sd["k"])
    sae.load_state_dict(sd["state_dict"])
    sae.eval()

    acts = np.load(f"sae_operators/activations/layer_{layer_idx}.npy")
    acts_t = torch.tensor(acts, dtype=torch.float32)
    acts_norm = (acts_t - sd["mean"]) / sd["std"]
    del acts, acts_t
    gc.collect()

    feat_active = np.zeros((n_tokens, dd), dtype=np.bool_)
    with torch.no_grad():
        for s in range(0, n_tokens, 50000):
            z = sae.encode(acts_norm[s:s + 50000])
            feat_active[s:s + 50000] = (z > 0).numpy()
    del acts_norm
    gc.collect()

    fsums = feat_active.astype(np.float64).sum(axis=0)
    af = np.where(fsums > 50)[0]
    print(f"  Active features (>50 tokens): {len(af)}/{dd}", flush=True)

    best = {}
    for li_idx, li in enumerate(valid):
        ln = all_labels[li]
        yt = labels[:, li].astype(np.float64)
        best_mcc_val = 0.0
        best_feat = -1
        best_precision = 0.0
        best_recall = 0.0
        for fi in af:
            yp = feat_active[:, fi].astype(np.float64)
            tp = (yt * yp).sum()
            tn = ((1 - yt) * (1 - yp)).sum()
            fp = ((1 - yt) * yp).sum()
            fn = (yt * (1 - yp)).sum()
            denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            if denom > 0:
                mcc_val = (tp * tn - fp * fn) / denom
                if abs(mcc_val) > abs(best_mcc_val):
                    best_mcc_val = mcc_val
                    best_feat = int(fi)
                    best_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    best_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        best[ln] = {
            "mcc": round(best_mcc_val, 4),
            "feature": best_feat,
            "n_positive": int(lsums[li]),
            "precision": round(best_precision, 4),
            "recall": round(best_recall, 4)
        }
        if (li_idx + 1) % 10 == 0:
            print(f"    ...processed {li_idx+1}/{len(valid)} labels", flush=True)

    print(f"\n  Best feature per operator label:", flush=True)
    for l_name in sorted(best.keys(), key=lambda x: -abs(best[x]["mcc"])):
        info = best[l_name]
        if abs(info["mcc"]) > 0.005:
            print(f"    {l_name:40s}: MCC={info['mcc']:+.4f}  "
                  f"P={info['precision']:.3f} R={info['recall']:.3f}  "
                  f"(feat {info['feature']}, n={info['n_positive']})", flush=True)

    aligned = sum(1 for v in best.values() if abs(v["mcc"]) > 0.1)
    moderate = sum(1 for v in best.values() if abs(v["mcc"]) > 0.05)
    mean_abs = float(np.mean([abs(v["mcc"]) for v in best.values()]))
    p90 = float(np.percentile([abs(v["mcc"]) for v in best.values()], 90))
    max_mcc = max(abs(v["mcc"]) for v in best.values())
    print(f"\n  Strong (|MCC|>0.1): {aligned}/{len(best)}", flush=True)
    print(f"  Moderate (|MCC|>0.05): {moderate}/{len(best)}", flush=True)
    print(f"  Mean |MCC|: {mean_abs:.4f}", flush=True)
    print(f"  P90 |MCC|:  {p90:.4f}", flush=True)
    print(f"  Max |MCC|:  {max_mcc:.4f}", flush=True)

    all_results[f"layer_{layer_idx}"] = {
        "aligned_strong": aligned, "aligned_moderate": moderate,
        "total": len(best), "mean_mcc": round(mean_abs, 4),
        "p90_mcc": round(p90, 4), "max_mcc": round(max_mcc, 4),
        "per_label": best
    }

    del feat_active
    gc.collect()

# Save
import pathlib
pathlib.Path("sae_operators/results").mkdir(parents=True, exist_ok=True)
with open("sae_operators/results/operator_alignment_v2.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n" + "=" * 60, flush=True)
print("SUMMARY (v2 token-level)", flush=True)
print("=" * 60, flush=True)
for lname, r in sorted(all_results.items()):
    print(f"  {lname}: strong={r['aligned_strong']}/{r['total']}, "
          f"moderate={r['aligned_moderate']}/{r['total']}, "
          f"mean|MCC|={r['mean_mcc']:.4f}, max={r['max_mcc']:.4f}", flush=True)
print("\nDone!", flush=True)
