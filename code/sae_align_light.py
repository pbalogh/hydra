#!/usr/bin/env python3
"""Lightweight SAE-operator alignment. Runs one layer at a time to save RAM."""
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

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")
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

print(f"Building label matrix ({n_tokens} tokens x {nl} labels)...", flush=True)
labels = np.zeros((n_tokens, nl), dtype=np.float32)
for si in range(n_seqs):
    s = si * seq_len
    toks = val_tokens[s:s + seq_len].astype(np.int64)
    wt = [int(t) for t in toks if t < 50257 and t > 0]
    if not wt:
        continue
    try:
        text = tok.decode(wt).lower()
    except Exception:
        continue
    seq_ops = set()
    for ent, ops in entity_to_ops.items():
        if ent in text:
            seq_ops.update(ops)
    for op_l in seq_ops:
        if op_l in l2i:
            labels[s:min(s + seq_len, n_tokens), l2i[op_l]] = 1.0

lsums = labels.sum(axis=0)
valid = np.where(lsums > 100)[0]
print(f"  {nl} labels, {len(valid)} with >100 tokens", flush=True)

for li in valid:
    pct = 100 * lsums[li] / n_tokens
    print(f"    {all_labels[li]:40s}: {int(lsums[li]):8d} tokens ({pct:.1f}%)", flush=True)

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

    # Encode in chunks, store as bool to save RAM
    feat_active = np.zeros((n_tokens, dd), dtype=np.bool_)
    with torch.no_grad():
        for s in range(0, n_tokens, 50000):
            z = sae.encode(acts_norm[s:s + 50000])
            feat_active[s:s + 50000] = (z > 0).numpy()
    del acts_norm
    gc.collect()

    fsums = feat_active.astype(np.float64).sum(axis=0)
    af = np.where(fsums > 100)[0]
    print(f"  Active features (>100 tokens): {len(af)}/{dd}", flush=True)

    best = {}
    for li_idx, li in enumerate(valid):
        ln = all_labels[li]
        yt = labels[:, li].astype(np.float64)
        best_mcc_val = 0.0
        best_feat = -1
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
        best[ln] = {"mcc": round(best_mcc_val, 4), "feature": best_feat,
                     "n_positive": int(lsums[li])}
        if (li_idx + 1) % 10 == 0:
            print(f"    ...processed {li_idx+1}/{len(valid)} labels", flush=True)

    print(f"\n  Best feature per operator label:", flush=True)
    for l_name in sorted(best.keys(), key=lambda x: -abs(best[x]["mcc"])):
        info = best[l_name]
        if abs(info["mcc"]) > 0.01:
            print(f"    {l_name:40s}: MCC={info['mcc']:+.4f}  "
                  f"(feature {info['feature']}, n={info['n_positive']})", flush=True)

    aligned = sum(1 for v in best.values() if abs(v["mcc"]) > 0.1)
    mean_abs = float(np.mean([abs(v["mcc"]) for v in best.values()]))
    p90 = float(np.percentile([abs(v["mcc"]) for v in best.values()], 90))
    print(f"\n  Aligned (|MCC|>0.1): {aligned}/{len(best)}", flush=True)
    print(f"  Mean |MCC|: {mean_abs:.4f}", flush=True)
    print(f"  P90 |MCC|:  {p90:.4f}", flush=True)

    all_results[f"layer_{layer_idx}"] = {
        "aligned": aligned, "total": len(best),
        "mean_mcc": round(mean_abs, 4), "p90_mcc": round(p90, 4),
        "per_label": best
    }

    del feat_active
    gc.collect()

# Save
import pathlib
pathlib.Path("sae_operators/results").mkdir(parents=True, exist_ok=True)
with open("sae_operators/results/operator_alignment.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n" + "=" * 60, flush=True)
print("SUMMARY", flush=True)
print("=" * 60, flush=True)
for lname, r in sorted(all_results.items()):
    print(f"  {lname}: aligned={r['aligned']}/{r['total']}, "
          f"mean|MCC|={r['mean_mcc']:.4f}, P90={r['p90_mcc']:.4f}", flush=True)
print("\nDone!", flush=True)
