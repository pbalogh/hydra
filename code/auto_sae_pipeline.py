#!/usr/bin/env python3
"""
Auto-SAE pipeline: monitors 130M training, triggers SAE extraction + alignment at checkpoints.
Run in background: nohup python3 -u auto_sae_pipeline.py > /data/hydra/logs/auto_sae_pipeline.log 2>&1 &
"""
import os, sys, time, json, math, gc, subprocess
import numpy as np
import torch

TRAIN_LOG = "/data/hydra/logs/train_130m_all-heads.jsonl"
CHECKPOINT_DIR = "/data/hydra/checkpoints_130m"
VAL_BIN = "/data/hydra/phase05_full_v2/val_all-heads.bin"
FACTS_FILE = "/data/hydra/knowledge_store/facts.jsonl"
OUTPUT_BASE = "/data/hydra/sae_130m"
TRIGGER_STEPS = [25000, 50000, 75000, 100000]

# 130M model config
D_MODEL = 768
N_LAYER = 24
VOCAB_SIZE = 50304
# Extract from layers spread across the model
EXTRACT_LAYERS = [0, 4, 8, 12, 16, 20, 23]
# SAE config
D_DICT = 3072  # 4x d_model
K = 12
SAE_EPOCHS = 5
SAE_BATCH = 4096
SAE_LR = 1e-3
L1_COEFF = 1e-4
MAX_TOKENS = 500000
SEQ_LEN = 512

sys.path.insert(0, "/data/hydra")
from operator_hierarchy_v2 import classify
from collections import defaultdict


class TopKSAE(torch.nn.Module):
    def __init__(self, d_in, d_dict, k):
        super().__init__()
        self.W_enc = torch.nn.Linear(d_in, d_dict, bias=True)
        self.W_dec = torch.nn.Linear(d_dict, d_in, bias=True)
        self.k = k
        self.d_dict = d_dict

    def encode(self, x):
        z = self.W_enc(x)
        topk = torch.topk(z, self.k, dim=-1)
        mask = torch.zeros_like(z)
        mask.scatter_(-1, topk.indices, 1.0)
        return z * mask

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.W_dec(z)
        return x_hat, z


def get_current_step():
    """Read last step from training log."""
    try:
        with open(TRAIN_LOG) as f:
            lines = f.readlines()
        if lines:
            last = json.loads(lines[-1])
            return last["step"]
    except Exception:
        pass
    return 0


def find_checkpoint(step):
    """Find checkpoint file for given step."""
    # The training script saves best + step checkpoints
    patterns = [
        f"{CHECKPOINT_DIR}/hydra_130m_step{step}.pt",
        f"{CHECKPOINT_DIR}/hydra_130m_best.pt",
    ]
    for p in patterns:
        if os.path.exists(p):
            return p
    # Check for any checkpoint
    if os.path.exists(CHECKPOINT_DIR):
        files = sorted(os.listdir(CHECKPOINT_DIR))
        for f in files:
            if f.endswith(".pt"):
                return os.path.join(CHECKPOINT_DIR, f)
    return None


def extract_activations(checkpoint_path, step):
    """Extract activations from 130M model checkpoint."""
    print(f"\n{'='*60}", flush=True)
    print(f"EXTRACTING ACTIVATIONS (step {step})", flush=True)
    print(f"{'='*60}", flush=True)

    out_dir = f"{OUTPUT_BASE}/step{step}/activations"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    ckpt = torch.load(checkpoint_path, map_location="cuda", weights_only=False)

    # Detect actual vocab size from checkpoint
    embed_key = None
    for k in ckpt.get("model", ckpt).keys():
        if "embedding" in k or "embed" in k:
            embed_key = k
            break
    
    model_state = ckpt.get("model", ckpt)
    actual_vocab = VOCAB_SIZE
    if embed_key:
        actual_vocab = model_state[embed_key].shape[0]
        print(f"  Detected vocab size: {actual_vocab}", flush=True)

    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from mamba_ssm.models.config_mamba import MambaConfig

    config = MambaConfig(
        d_model=D_MODEL,
        n_layer=N_LAYER,
        vocab_size=actual_vocab,
    )
    model = MambaLMHeadModel(config, device="cuda", dtype=torch.float16)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    val_tokens = np.memmap(VAL_BIN, dtype=np.int32, mode="r")
    n_seqs = min(MAX_TOKENS // SEQ_LEN, len(val_tokens) // SEQ_LEN)
    n_tokens = n_seqs * SEQ_LEN
    print(f"  Extracting from {n_tokens} tokens ({n_seqs} sequences)...", flush=True)

    # Hook to capture activations
    layer_acts = {l: [] for l in EXTRACT_LAYERS}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is (hidden_states, residual) or just hidden_states
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            layer_acts[layer_idx].append(h.detach().cpu().float().numpy())
        return hook_fn

    for layer_idx in EXTRACT_LAYERS:
        h = model.backbone.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    with torch.no_grad():
        for si in range(0, n_seqs, 4):  # batch of 4
            batch_end = min(si + 4, n_seqs)
            batch_tokens = []
            for bi in range(si, batch_end):
                s = bi * SEQ_LEN
                toks = val_tokens[s:s + SEQ_LEN].astype(np.int64)
                # Clamp to vocab range
                toks = np.clip(toks, 0, actual_vocab - 1)
                batch_tokens.append(toks)
            
            input_ids = torch.tensor(np.array(batch_tokens), dtype=torch.long, device="cuda")
            model(input_ids)

            if (si // 4 + 1) % 50 == 0:
                print(f"    {(si + batch_end - si) * SEQ_LEN} tokens extracted...", flush=True)

    for h in hooks:
        h.remove()

    # Save activations
    print(f"\n  Saving activations ({n_tokens} tokens)...", flush=True)
    for layer_idx in EXTRACT_LAYERS:
        acts = np.concatenate(layer_acts[layer_idx], axis=0)  # (n_seqs*batch, seq_len, d_model)
        acts = acts.reshape(-1, D_MODEL)[:n_tokens]
        np.save(f"{out_dir}/layer_{layer_idx}.npy", acts)
        print(f"    Layer {layer_idx}: {acts.shape} → {out_dir}/layer_{layer_idx}.npy", flush=True)
        del acts
    
    del model, ckpt
    torch.cuda.empty_cache()
    gc.collect()

    meta = {
        "checkpoint": checkpoint_path,
        "step": step,
        "n_tokens": n_tokens,
        "seq_len": SEQ_LEN,
        "d_model": D_MODEL,
        "layers": EXTRACT_LAYERS,
    }
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("  Done extracting.", flush=True)
    return out_dir


def train_saes(act_dir, step):
    """Train SAEs on extracted activations (CPU)."""
    print(f"\n{'='*60}", flush=True)
    print(f"TRAINING SAEs (step {step})", flush=True)
    print(f"{'='*60}", flush=True)

    weight_dir = f"{OUTPUT_BASE}/step{step}/sae_weights"
    os.makedirs(weight_dir, exist_ok=True)

    with open(f"{act_dir}/meta.json") as f:
        meta = json.load(f)

    for layer_idx in meta["layers"]:
        print(f"\n  --- Layer {layer_idx} ---", flush=True)
        gc.collect()

        acts = np.load(f"{act_dir}/layer_{layer_idx}.npy")
        acts_t = torch.tensor(acts, dtype=torch.float32)
        mean = acts_t.mean(dim=0)
        std = acts_t.std(dim=0).clamp(min=1e-6)
        acts_norm = (acts_t - mean) / std
        del acts, acts_t
        gc.collect()

        sae = TopKSAE(D_MODEL, D_DICT, K)
        opt = torch.optim.Adam(sae.parameters(), lr=SAE_LR)
        n = len(acts_norm)

        for epoch in range(SAE_EPOCHS):
            perm = torch.randperm(n)
            total_loss = 0
            total_recon = 0
            total_l1 = 0
            batches = 0
            for i in range(0, n, SAE_BATCH):
                batch = acts_norm[perm[i:i + SAE_BATCH]]
                x_hat, z = sae(batch)
                recon = ((batch - x_hat) ** 2).mean()
                l1 = z.abs().mean() * L1_COEFF
                loss = recon + l1
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                total_recon += recon.item()
                total_l1 += l1.item()
                batches += 1
            avg_loss = total_loss / batches
            avg_recon = total_recon / batches
            avg_l1 = total_l1 / batches
            print(f"    Epoch {epoch+1}/{SAE_EPOCHS}: loss={avg_loss:.4f} "
                  f"(recon={avg_recon:.4f}, l1={avg_l1:.4f})", flush=True)

        save_path = f"{weight_dir}/sae_layer_{layer_idx}.pt"
        torch.save({
            "state_dict": sae.state_dict(),
            "d_model": D_MODEL,
            "d_dict": D_DICT,
            "k": K,
            "mean": mean,
            "std": std,
            "layer": layer_idx,
            "step": step,
        }, save_path)
        print(f"    Saved: {save_path}", flush=True)
        del acts_norm, sae
        gc.collect()


def run_alignment(step):
    """Run operator alignment v2 (token-level) on 130M SAEs."""
    print(f"\n{'='*60}", flush=True)
    print(f"RUNNING ALIGNMENT (step {step})", flush=True)
    print(f"{'='*60}", flush=True)

    act_dir = f"{OUTPUT_BASE}/step{step}/activations"
    weight_dir = f"{OUTPUT_BASE}/step{step}/sae_weights"
    result_dir = f"{OUTPUT_BASE}/step{step}/results"
    os.makedirs(result_dir, exist_ok=True)

    with open(f"{act_dir}/meta.json") as f:
        meta = json.load(f)

    # Load facts + build entity labels
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
    entity_token_ids = {}
    entity_token_ids_space = {}
    for ent in entity_to_ops:
        try:
            entity_token_ids[ent] = tok.encode(ent)
            entity_token_ids_space[ent] = tok.encode(" " + ent)
        except Exception:
            pass

    val_tokens = np.memmap(VAL_BIN, dtype=np.int32, mode="r")
    n_tokens = meta["n_tokens"]
    tokens_arr = np.array(val_tokens[:n_tokens], dtype=np.int64)

    all_labels = sorted(set(l for ops in entity_to_ops.values() for l in ops))
    l2i = {l: i for i, l in enumerate(all_labels)}
    nl = len(all_labels)

    print(f"  Building token-level labels ({n_tokens} x {nl})...", flush=True)
    labels = np.zeros((n_tokens, nl), dtype=np.float32)

    for ent_idx, (ent, ops) in enumerate(entity_to_ops.items()):
        if ent_idx % 3000 == 0:
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
                        if all(tokens_arr[pos + j] == ent_ids[j] for j in range(len(ent_ids))):
                            for offset in range(len(ent_ids)):
                                for li in op_indices:
                                    labels[pos + offset, li] = 1.0

    lsums = labels.sum(axis=0)
    valid = np.where(lsums > 50)[0]
    total_labeled = int((labels.sum(axis=1) > 0).sum())
    print(f"  {total_labeled}/{n_tokens} tokens labeled, {len(valid)} valid labels", flush=True)

    all_results = {}

    for layer_idx in meta["layers"]:
        print(f"\n  === Layer {layer_idx} ===", flush=True)
        gc.collect()

        sae_path = f"{weight_dir}/sae_layer_{layer_idx}.pt"
        if not os.path.exists(sae_path):
            print(f"    SKIP (no weights)", flush=True)
            continue

        sd = torch.load(sae_path, map_location="cpu", weights_only=False)
        sae = TopKSAE(D_MODEL, sd["d_dict"], sd["k"])
        sae.load_state_dict(sd["state_dict"])
        sae.eval()

        acts = np.load(f"{act_dir}/layer_{layer_idx}.npy")
        acts_t = torch.tensor(acts, dtype=torch.float32)
        acts_norm = (acts_t - sd["mean"]) / sd["std"]
        del acts, acts_t
        gc.collect()

        feat_active = np.zeros((n_tokens, sd["d_dict"]), dtype=np.bool_)
        with torch.no_grad():
            for s in range(0, n_tokens, 50000):
                z = sae.encode(acts_norm[s:s + 50000])
                feat_active[s:s + 50000] = (z > 0).numpy()
        del acts_norm
        gc.collect()

        fsums = feat_active.astype(np.float64).sum(axis=0)
        af = np.where(fsums > 50)[0]
        print(f"    Active features: {len(af)}/{sd['d_dict']}", flush=True)

        best = {}
        for li_idx, li in enumerate(valid):
            ln = all_labels[li]
            yt = labels[:, li].astype(np.float64)
            best_mcc_val = 0.0
            best_feat = -1
            best_p = 0.0
            best_r = 0.0
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
                        best_p = tp / (tp + fp) if (tp + fp) > 0 else 0
                        best_r = tp / (tp + fn) if (tp + fn) > 0 else 0
            best[ln] = {"mcc": round(best_mcc_val, 4), "feature": best_feat,
                        "n": int(lsums[li]), "P": round(best_p, 4), "R": round(best_r, 4)}

        for l_name in sorted(best.keys(), key=lambda x: -abs(best[x]["mcc"])):
            info = best[l_name]
            if abs(info["mcc"]) > 0.01:
                print(f"    {l_name:40s}: MCC={info['mcc']:+.4f} P={info['P']:.3f} "
                      f"R={info['R']:.3f} (feat {info['feature']}, n={info['n']})", flush=True)

        aligned = sum(1 for v in best.values() if abs(v["mcc"]) > 0.1)
        moderate = sum(1 for v in best.values() if abs(v["mcc"]) > 0.05)
        mean_abs = float(np.mean([abs(v["mcc"]) for v in best.values()]))
        max_mcc = max(abs(v["mcc"]) for v in best.values())
        print(f"    Strong: {aligned}/{len(best)}, Moderate: {moderate}/{len(best)}, "
              f"Mean: {mean_abs:.4f}, Max: {max_mcc:.4f}", flush=True)

        all_results[f"layer_{layer_idx}"] = {
            "strong": aligned, "moderate": moderate, "total": len(best),
            "mean_mcc": round(mean_abs, 4), "max_mcc": round(max_mcc, 4),
            "per_label": best
        }
        del feat_active
        gc.collect()

    with open(f"{result_dir}/alignment_v2.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Multi-feature analysis (compositional superposition) ──────────
    run_multifeature_analysis(step, act_dir, weight_dir, result_dir,
                              meta, labels, all_labels, valid, lsums, n_tokens)

    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY — 130M Step {step}", flush=True)
    print(f"{'='*60}", flush=True)
    for lname, r in sorted(all_results.items()):
        print(f"  {lname}: strong={r['strong']}/{r['total']}, "
              f"mean={r['mean_mcc']:.4f}, max={r['max_mcc']:.4f}", flush=True)

    # Compare with 39.3M baseline
    baseline_path = "/data/hydra/sae_operators/results/operator_alignment_v2.json"
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        print(f"\n  vs 39.3M baseline:", flush=True)
        for lname in sorted(all_results.keys()):
            bl = baseline.get(lname, {})
            r = all_results[lname]
            bl_mean = bl.get("mean_mcc", 0)
            delta = r["mean_mcc"] - bl_mean
            print(f"    {lname}: {bl_mean:.4f} → {r['mean_mcc']:.4f} ({delta:+.4f})", flush=True)

    print("\nDone!", flush=True)


def run_multifeature_analysis(step, act_dir, weight_dir, result_dir,
                               meta, labels, all_labels, valid, lsums, n_tokens):
    """Multi-feature operator detection: logistic regression + decision trees.
    Tests whether operators are encoded as feature CONJUNCTIONS (compositional superposition)."""
    print(f"\n{'='*60}", flush=True)
    print(f"MULTI-FEATURE ANALYSIS (step {step})", flush=True)
    print(f"{'='*60}", flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import matthews_corrcoef
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Pick best layer from single-feature results
    with open(f"{result_dir}/alignment_v2.json") as f:
        single_results = json.load(f)
    best_layer = max(single_results.keys(),
                     key=lambda k: single_results[k].get("mean_mcc", 0))
    best_layer_idx = int(best_layer.split("_")[1])
    print(f"  Using {best_layer} (highest mean MCC from single-feature)", flush=True)

    # Load SAE features for best layer
    sd = torch.load(f"{weight_dir}/sae_layer_{best_layer_idx}.pt",
                     map_location="cpu", weights_only=False)
    sae = TopKSAE(D_MODEL, sd["d_dict"], sd["k"])
    sae.load_state_dict(sd["state_dict"])
    sae.eval()

    acts = np.load(f"{act_dir}/layer_{best_layer_idx}.npy")
    acts_t = torch.tensor(acts, dtype=torch.float32)
    acts_norm = (acts_t - sd["mean"]) / sd["std"]
    del acts, acts_t

    feat_active = np.zeros((n_tokens, sd["d_dict"]), dtype=np.bool_)
    with torch.no_grad():
        for s in range(0, n_tokens, 50000):
            z = sae.encode(acts_norm[s:s + 50000])
            feat_active[s:s + 50000] = (z > 0).numpy()
    del acts_norm
    gc.collect()

    # Filter to active features
    fsums = feat_active.astype(np.float64).sum(axis=0)
    af = np.where(fsums > 50)[0]
    X = feat_active[:, af].astype(np.float32)
    print(f"  Feature matrix: {X.shape} ({len(af)} active features)", flush=True)

    # Subsample for speed (logistic regression on 500K x 3072 is slow)
    MAX_SAMPLES = 100000
    if n_tokens > MAX_SAMPLES:
        idx = np.random.RandomState(42).choice(n_tokens, MAX_SAMPLES, replace=False)
        X_sub = X[idx]
        labels_sub = labels[idx]
    else:
        X_sub = X
        labels_sub = labels
        idx = np.arange(n_tokens)

    multi_results = {}

    for li_idx, li in enumerate(valid):
        ln = all_labels[li]
        y = labels_sub[:, li].astype(np.int32)
        n_pos = y.sum()
        if n_pos < 20:
            continue

        # 1. Logistic Regression (multi-feature linear)
        try:
            lr = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced",
                                     solver="lbfgs", random_state=42)
            scores = cross_val_score(lr, X_sub, y, cv=3, scoring="matthews_corrcoef")
            lr_mcc = float(np.mean(scores))
        except Exception:
            lr_mcc = 0.0

        # 2. Decision Tree (discovers conjunctive rules)
        try:
            dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20,
                                         class_weight="balanced", random_state=42)
            dt_scores = cross_val_score(dt, X_sub, y, cv=3, scoring="matthews_corrcoef")
            dt_mcc = float(np.mean(dt_scores))

            # Fit on full data to extract rules
            dt.fit(X_sub, y)
            tree_rules = export_text(dt, max_depth=3,
                                      feature_names=[f"f{af[i]}" for i in range(len(af))])
            # Count features used in tree
            used_features = set()
            for line in tree_rules.split("\n"):
                if "f" in line and "<=" in line:
                    feat_name = line.split("f")[1].split(" ")[0].strip()
                    try:
                        used_features.add(int(feat_name))
                    except ValueError:
                        pass
            n_features_used = len(used_features)
        except Exception:
            dt_mcc = 0.0
            tree_rules = ""
            n_features_used = 0
            used_features = set()

        # Get single-feature MCC for comparison
        single_mcc = abs(single_results[best_layer]["per_label"].get(ln, {}).get("mcc", 0))

        improvement = lr_mcc - single_mcc

        multi_results[ln] = {
            "single_mcc": round(single_mcc, 4),
            "logistic_mcc": round(lr_mcc, 4),
            "tree_mcc": round(dt_mcc, 4),
            "improvement": round(improvement, 4),
            "tree_features_used": n_features_used,
            "tree_feature_ids": sorted(used_features),
            "n_positive": int(n_pos),
        }

        if improvement > 0.02:
            print(f"  📈 {ln:40s}: single={single_mcc:.3f} → LR={lr_mcc:.3f} "
                  f"tree={dt_mcc:.3f} (+{improvement:.3f}, {n_features_used} feats)", flush=True)
        else:
            print(f"     {ln:40s}: single={single_mcc:.3f} → LR={lr_mcc:.3f} "
                  f"tree={dt_mcc:.3f} ({improvement:+.3f})", flush=True)

    # Summary
    improvements = [v["improvement"] for v in multi_results.values()]
    superposed = sum(1 for v in multi_results.values() if v["improvement"] > 0.05)
    mean_improvement = float(np.mean(improvements)) if improvements else 0

    print(f"\n  COMPOSITIONAL SUPERPOSITION SUMMARY:", flush=True)
    print(f"  Labels with >0.05 improvement from multi-feature: "
          f"{superposed}/{len(multi_results)}", flush=True)
    print(f"  Mean improvement (LR over single): {mean_improvement:+.4f}", flush=True)
    print(f"  Mean single-feature MCC: "
          f"{np.mean([v['single_mcc'] for v in multi_results.values()]):.4f}", flush=True)
    print(f"  Mean logistic MCC: "
          f"{np.mean([v['logistic_mcc'] for v in multi_results.values()]):.4f}", flush=True)
    print(f"  Mean tree MCC: "
          f"{np.mean([v['tree_mcc'] for v in multi_results.values()]):.4f}", flush=True)

    # Find shared features across related labels
    print(f"\n  FEATURE SHARING (conjunctive structure):", flush=True)
    # Group by primitive
    prim_features = defaultdict(set)
    ttype_features = defaultdict(set)
    domain_features = defaultdict(set)
    for ln, info in multi_results.items():
        fids = set(info["tree_feature_ids"])
        if ln.startswith("prim:"):
            prim_features[ln] = fids
        elif ln.startswith("ttype:"):
            ttype_features[ln] = fids
        elif ln.startswith("domain:"):
            domain_features[ln] = fids

    # Find features shared within each category (type parameter features)
    for category, feat_dict in [("PRIMITIVES", prim_features),
                                 ("TRANSFER TYPES", ttype_features),
                                 ("DOMAINS", domain_features)]:
        if len(feat_dict) < 2:
            continue
        all_feats = set()
        for fids in feat_dict.values():
            all_feats.update(fids)
        if not all_feats:
            continue
        # For each feature, count how many labels use it
        feat_counts = {}
        for fid in all_feats:
            users = [ln for ln, fids in feat_dict.items() if fid in fids]
            if len(users) > 1:
                feat_counts[fid] = users
        if feat_counts:
            print(f"\n    {category} — shared features:", flush=True)
            for fid, users in sorted(feat_counts.items(), key=lambda x: -len(x[1])):
                print(f"      Feature {fid}: used by {', '.join(users)}", flush=True)

    with open(f"{result_dir}/multifeature_analysis.json", "w") as f:
        json.dump(multi_results, f, indent=2)

    del feat_active, X, X_sub
    gc.collect()


def main():
    print("=" * 60, flush=True)
    print("AUTO-SAE PIPELINE — Monitoring 130M Training", flush=True)
    print(f"  Trigger steps: {TRIGGER_STEPS}", flush=True)
    print(f"  Extract layers: {EXTRACT_LAYERS}", flush=True)
    print(f"  SAE config: d_dict={D_DICT}, k={K}, epochs={SAE_EPOCHS}", flush=True)
    print("=" * 60, flush=True)

    completed = set()
    # Check if any triggers already done
    for step in TRIGGER_STEPS:
        result_file = f"{OUTPUT_BASE}/step{step}/results/alignment_v2.json"
        if os.path.exists(result_file):
            completed.add(step)
            print(f"  Step {step}: already completed", flush=True)

    while True:
        current = get_current_step()
        for trigger in TRIGGER_STEPS:
            if trigger in completed:
                continue
            if current >= trigger:
                print(f"\n🎯 Step {current} >= {trigger} — triggering pipeline!", flush=True)
                ckpt = find_checkpoint(trigger)
                if not ckpt:
                    # Use best checkpoint
                    ckpt = find_checkpoint(current)
                if ckpt:
                    try:
                        act_dir = extract_activations(ckpt, trigger)
                        train_saes(act_dir, trigger)
                        run_alignment(trigger)
                        completed.add(trigger)
                    except Exception as e:
                        print(f"ERROR at step {trigger}: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"  No checkpoint found for step {trigger}", flush=True)

        if len(completed) == len(TRIGGER_STEPS):
            print("\nAll trigger steps completed. Exiting.", flush=True)
            break

        time.sleep(300)  # Check every 5 minutes


if __name__ == "__main__":
    main()
