#!/usr/bin/env python3
"""
Lightweight SAE alignment — processes one feature at a time to avoid OOM.
Runs entirely on CPU, designed to coexist with GPU training.

Usage: CUDA_VISIBLE_DEVICES='' python3 alignment_lite.py --step 25000
"""

import json, os, sys, math, gc, argparse
import numpy as np
import torch
from collections import defaultdict

OUTPUT_BASE = "/data/hydra/sae_130m"
FACTS_FILE = "/data/hydra/knowledge_store_augmented.jsonl"
VAL_BIN = "/data/hydra/val_tokens.bin"


def classify_op(surface_op, bindings, prop_id=None):
    """Minimal operator classifier — just returns the surface op."""
    # Import from the main pipeline if available
    try:
        sys.path.insert(0, "/data/hydra")
        from auto_sae_pipeline import classify
        return classify(surface_op, bindings, prop_id)
    except:
        class Op:
            def __init__(self):
                self.primitive = surface_op or "UNKNOWN"
                self.params = []
        return Op()


def build_labels(n_tokens, tokens_arr):
    """Build token-level labels from knowledge store."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    
    facts = [json.loads(l) for l in open(FACTS_FILE)]
    entity_to_ops = defaultdict(set)
    for f_item in facts:
        op = classify_op(
            f_item.get("surface_operator", f_item.get("operator", "")),
            f_item.get("bindings", {}),
            f_item.get("property_id"))
        for role, ent in f_item.get("bindings", {}).items():
            if ent and len(ent) > 3:
                el = ent.lower()
                entity_to_ops[el].add("prim:" + op.primitive)
                if hasattr(op, 'params') and len(op.params) >= 1:
                    entity_to_ops[el].add("ttype:" + op.params[0].name)
                if hasattr(op, 'params') and len(op.params) >= 2:
                    entity_to_ops[el].add("domain:" + op.params[1].name)

    # Pre-tokenize entities
    entity_token_ids = {}
    entity_token_ids_space = {}
    for ent in entity_to_ops:
        try:
            entity_token_ids[ent] = tok.encode(ent)
            entity_token_ids_space[ent] = tok.encode(" " + ent)
        except:
            pass

    all_labels = sorted(set(l for ops in entity_to_ops.values() for l in ops))
    l2i = {l: i for i, l in enumerate(all_labels)}
    nl = len(all_labels)

    print(f"  Building labels ({n_tokens} x {nl})...", flush=True)
    labels = np.zeros((n_tokens, nl), dtype=np.float32)

    for ent_idx, (ent, ops) in enumerate(entity_to_ops.items()):
        if ent_idx % 5000 == 0:
            print(f"    Entity {ent_idx}/{len(entity_to_ops)}...", flush=True)
        op_indices = [l2i[op] for op in ops if op in l2i]
        if not op_indices:
            continue
        for token_map in [entity_token_ids, entity_token_ids_space]:
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
                        if all(tokens_arr[pos+j] == ent_ids[j] for j in range(len(ent_ids))):
                            for offset in range(len(ent_ids)):
                                for li in op_indices:
                                    labels[pos+offset, li] = 1.0

    lsums = labels.sum(axis=0)
    valid = np.where(lsums > 50)[0]
    total_labeled = int((labels.sum(axis=1) > 0).sum())
    print(f"  {total_labeled}/{n_tokens} labeled, {len(valid)} valid labels", flush=True)
    
    return labels, all_labels, valid, lsums


class TopKSAE(torch.nn.Module):
    def __init__(self, d_in, d_dict, k):
        super().__init__()
        self.d_in = d_in
        self.d_dict = d_dict
        self.k = k
        self.encoder = torch.nn.Linear(d_in, d_dict, bias=True)
        self.decoder = torch.nn.Linear(d_dict, d_in, bias=True)

    def encode(self, x):
        z = self.encoder(x)
        topk = torch.topk(z, self.k, dim=-1)
        mask = torch.zeros_like(z)
        mask.scatter_(-1, topk.indices, 1.0)
        return z * mask


def align_layer_chunked(layer_idx, step, labels, all_labels, valid, lsums, n_tokens,
                         chunk_size=256):
    """
    Process one layer's alignment using CHUNKED feature processing.
    Instead of loading all features as a dense boolean array (1.5GB),
    we process features in small chunks.
    """
    act_dir = f"{OUTPUT_BASE}/step{step}/activations"
    weight_dir = f"{OUTPUT_BASE}/step{step}/sae_weights"
    
    sae_path = f"{weight_dir}/sae_layer_{layer_idx}.pt"
    if not os.path.exists(sae_path):
        print(f"  SKIP layer {layer_idx} (no weights)", flush=True)
        return None

    sd = torch.load(sae_path, map_location="cpu", weights_only=False)
    d_dict = sd["d_dict"]
    d_model = sd["state_dict"]["encoder.weight"].shape[1]
    
    sae = TopKSAE(d_model, d_dict, sd["k"])
    sae.load_state_dict(sd["state_dict"])
    sae.eval()

    acts = np.load(f"{act_dir}/layer_{layer_idx}.npy")
    acts_t = torch.tensor(acts, dtype=torch.float32)
    acts_norm = (acts_t - sd["mean"]) / sd["std"]
    del acts, acts_t
    gc.collect()

    # Get feature activity sums first (one pass, chunked by tokens)
    print(f"  Computing feature activity (chunked)...", flush=True)
    fsums = np.zeros(d_dict, dtype=np.float64)
    
    BATCH = 10000
    for s in range(0, n_tokens, BATCH):
        with torch.no_grad():
            z = sae.encode(acts_norm[s:s+BATCH])
            active = (z > 0).numpy().astype(np.float64)
            fsums += active.sum(axis=0)
            del z, active
    
    active_features = np.where(fsums > 50)[0]
    print(f"  Active features: {len(active_features)}/{d_dict}", flush=True)

    # Now compute MCC per label, processing features in chunks
    best = {}
    
    for li_idx, li in enumerate(valid):
        ln = all_labels[li]
        yt = labels[:, li].astype(np.float64)
        n_pos = yt.sum()
        
        best_mcc = 0.0
        best_feat = -1
        best_p = 0.0
        best_r = 0.0
        
        # Process features in chunks to avoid memory explosion
        for chunk_start in range(0, len(active_features), chunk_size):
            chunk_feats = active_features[chunk_start:chunk_start+chunk_size]
            
            # Compute feature activations for this chunk
            chunk_active = np.zeros((n_tokens, len(chunk_feats)), dtype=np.bool_)
            for s in range(0, n_tokens, BATCH):
                end = min(s + BATCH, n_tokens)
                with torch.no_grad():
                    z = sae.encode(acts_norm[s:end])
                    for ci, fi in enumerate(chunk_feats):
                        chunk_active[s:end, ci] = (z[:, fi] > 0).numpy()
                    del z
            
            # Compute MCC for each feature in chunk
            for ci, fi in enumerate(chunk_feats):
                yp = chunk_active[:, ci].astype(np.float64)
                tp = (yt * yp).sum()
                tn = ((1-yt) * (1-yp)).sum()
                fp = ((1-yt) * yp).sum()
                fn = (yt * (1-yp)).sum()
                denom = math.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))
                if denom > 0:
                    mcc = (tp*tn - fp*fn) / denom
                    if abs(mcc) > abs(best_mcc):
                        best_mcc = mcc
                        best_feat = int(fi)
                        best_p = tp/(tp+fp) if (tp+fp) > 0 else 0
                        best_r = tp/(tp+fn) if (tp+fn) > 0 else 0
            
            del chunk_active
            gc.collect()
        
        best[ln] = {"mcc": round(best_mcc, 4), "feature": best_feat,
                     "n": int(lsums[li]), "P": round(best_p, 4), "R": round(best_r, 4)}
    
    del acts_norm
    gc.collect()
    
    # Print results
    for l_name in sorted(best.keys(), key=lambda x: -abs(best[x]["mcc"])):
        info = best[l_name]
        if abs(info["mcc"]) > 0.01:
            print(f"    {l_name:40s}: MCC={info['mcc']:+.4f} P={info['P']:.3f} "
                  f"R={info['R']:.3f} (feat {info['feature']}, n={info['n']})", flush=True)

    aligned = sum(1 for v in best.values() if abs(v["mcc"]) > 0.1)
    moderate = sum(1 for v in best.values() if abs(v["mcc"]) > 0.05)
    mean_abs = float(np.mean([abs(v["mcc"]) for v in best.values()]))
    max_mcc = max(abs(v["mcc"]) for v in best.values()) if best else 0
    print(f"    Strong: {aligned}/{len(best)}, Moderate: {moderate}/{len(best)}, "
          f"Mean: {mean_abs:.4f}, Max: {max_mcc:.4f}", flush=True)

    return {
        "strong": aligned, "moderate": moderate, "total": len(best),
        "mean_mcc": round(mean_abs, 4), "max_mcc": round(max_mcc, 4),
        "per_label": best
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=25000)
    parser.add_argument("--chunk-size", type=int, default=128,
                       help="Features per chunk (lower = less RAM)")
    args = parser.parse_args()

    act_dir = f"{OUTPUT_BASE}/step{args.step}/activations"
    result_dir = f"{OUTPUT_BASE}/step{args.step}/results"
    os.makedirs(result_dir, exist_ok=True)

    with open(f"{act_dir}/meta.json") as f:
        meta = json.load(f)

    n_tokens = meta["n_tokens"]
    val_tokens = np.memmap(VAL_BIN, dtype=np.int32, mode="r")
    tokens_arr = np.array(val_tokens[:n_tokens], dtype=np.int64)

    labels, all_labels, valid, lsums = build_labels(n_tokens, tokens_arr)

    print(f"\n{'='*60}", flush=True)
    print(f"ALIGNMENT LITE (step {args.step})", flush=True)
    print(f"{'='*60}", flush=True)

    all_results = {}
    for layer_idx in meta["layers"]:
        print(f"\n  === Layer {layer_idx} ===", flush=True)
        result = align_layer_chunked(
            layer_idx, args.step, labels, all_labels, valid, lsums, n_tokens,
            chunk_size=args.chunk_size)
        if result:
            all_results[f"layer_{layer_idx}"] = result
        gc.collect()

    with open(f"{result_dir}/alignment_lite.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY — 130M Step {args.step}", flush=True)
    print(f"{'='*60}", flush=True)
    for lname, r in sorted(all_results.items()):
        print(f"  {lname}: strong={r['strong']}/{r['total']}, "
              f"mean={r['mean_mcc']:.4f}, max={r['max_mcc']:.4f}", flush=True)

    print(f"\nResults saved to {result_dir}/alignment_lite.json", flush=True)


if __name__ == "__main__":
    main()
