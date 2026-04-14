#!/usr/bin/env python3
"""
Hydra 130M Mixed Training: Phase 2 (tagged + untagged)
======================================================
Continues training from the best tagged-only checkpoint on a mix
of tagged and untagged sequences. Goal: model that uses structural
tags when present, but can also process plain text.

Key design:
  - 50/50 tagged/untagged batches (alternating)
  - Lower LR (1/5 of original peak) — we're fine-tuning, not training from scratch
  - Word-only PPL tracked separately for tagged AND untagged
  - Bail triggers: if tagged word PPL rises above threshold, stop

Usage:
    python3 hydra_mixed_train.py [--resume CKPT] [--mix-ratio 0.5] [--bail-threshold 3.0]
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# ── Config ─────────────────────────────────────────────────────────────

TAGGED_DIR = Path("/data/hydra/phase05_full_v2")
PLAIN_DIR = Path("/data/hydra/mixed_training")
CKPT_DIR = Path("/data/hydra/checkpoints_mixed")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("/data/hydra/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Model config — must match original
D_MODEL = 768
N_LAYER = 24
VOCAB_SIZE = 50304
MAX_SEQ_LEN = 512

# Training config — conservative for fine-tuning
BATCH_SIZE = 16
GRAD_ACCUM = 4       # effective batch = 64
LR = 1.2e-4          # 1/5 of original 6e-4
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.1
MAX_STEPS = 30000     # ~30% of original (enough to learn plain text mode)

EVAL_EVERY = 2000
SAVE_EVERY = 5000
LOG_EVERY = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True

# Bail triggers
DEFAULT_BAIL_THRESHOLD = 3.5  # stop if tagged word PPL exceeds this (baseline: 2.10)


# ── Dataset ────────────────────────────────────────────────────────────

class PreTokenizedDataset(Dataset):
    """Loads pre-tokenized .bin data (int32 flat array)."""
    
    def __init__(self, bin_path, seq_len=MAX_SEQ_LEN):
        self.seq_len = seq_len
        self.bin_path = str(bin_path)
        
        print(f"Loading pre-tokenized data from {bin_path}...")
        self.tokens = np.memmap(bin_path, dtype=np.int32, mode='r')
        n = (len(self.tokens) // seq_len) * seq_len
        self.n_sequences = n // seq_len
        
        print(f"  {len(self.tokens):,} tokens → {self.n_sequences:,} sequences of {seq_len}")
    
    def __len__(self):
        return self.n_sequences
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.from_numpy(
            self.tokens[start : start + self.seq_len].astype(np.int64).copy()
        )
        return x


class MixedDataLoader:
    """
    Alternates batches from tagged and untagged dataloaders.
    Returns (batch, is_tagged) tuples.
    """
    
    def __init__(self, tagged_loader, plain_loader, mix_ratio=0.5):
        self.tagged_loader = tagged_loader
        self.plain_loader = plain_loader
        self.mix_ratio = mix_ratio  # probability of tagged batch
        self.tagged_iter = iter(tagged_loader)
        self.plain_iter = iter(plain_loader)
        self._rng = np.random.RandomState(42)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        use_tagged = self._rng.random() < self.mix_ratio
        
        if use_tagged:
            try:
                batch = next(self.tagged_iter)
            except StopIteration:
                self.tagged_iter = iter(self.tagged_loader)
                batch = next(self.tagged_iter)
            return batch, True
        else:
            try:
                batch = next(self.plain_iter)
            except StopIteration:
                self.plain_iter = iter(self.plain_loader)
                batch = next(self.plain_iter)
            return batch, False


# ── Tag info ───────────────────────────────────────────────────────────

def load_tag_ids():
    tok_path = TAGGED_DIR / "tokenizer_all-heads.json"
    with open(tok_path) as f:
        d = json.load(f)
    return set(d["tag_to_id"].values())


# ── Model ──────────────────────────────────────────────────────────────

def build_model(vocab_size=VOCAB_SIZE):
    from mamba_ssm import MambaLMHeadModel
    from mamba_ssm.models.config_mamba import MambaConfig
    
    config = MambaConfig(d_model=D_MODEL, n_layer=N_LAYER, vocab_size=vocab_size)
    model = MambaLMHeadModel(config).to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params")
    return model


# ── Evaluation ─────────────────────────────────────────────────────────

def evaluate(model, loader, tag_ids, max_batches=200, label=""):
    """Compute word-only PPL on a dataloader."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    word_loss = 0
    word_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            
            # Handle both (batch,) and (batch, is_tagged) formats
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            
            x = batch.to(DEVICE)
            
            with autocast("cuda", dtype=torch.float16) if USE_AMP else torch.no_grad():
                outputs = model(x[:, :-1])
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            logits_f32 = logits.float()
            targets = x[:, 1:]
            loss_per_token = torch.nn.functional.cross_entropy(
                logits_f32.reshape(-1, logits_f32.size(-1)),
                targets.reshape(-1),
                reduction='none',
            ).reshape(targets.shape)
            
            total_loss += loss_per_token.sum().item()
            total_tokens += targets.numel()
            
            if tag_ids:
                word_mask = torch.ones_like(targets, dtype=torch.bool)
                for tid in tag_ids:
                    word_mask &= (targets != tid)
                if word_mask.sum() > 0:
                    word_loss += (loss_per_token * word_mask).sum().item()
                    word_tokens += word_mask.sum().item()
    
    avg_total = total_loss / max(total_tokens, 1)
    avg_word = word_loss / max(word_tokens, 1) if word_tokens > 0 else avg_total
    word_ppl = math.exp(min(avg_word, 20))
    
    if label:
        print(f"  [{label}] word_loss={avg_word:.4f} word_ppl={word_ppl:.2f} ({word_tokens:,} word tokens)")
    
    return avg_total, avg_word, word_ppl


# ── Training ───────────────────────────────────────────────────────────

def train(args):
    print(f"\n{'='*60}")
    print(f"Hydra 130M MIXED Training")
    print(f"  Mix ratio: {args.mix_ratio} (tagged probability)")
    print(f"  Bail threshold: {args.bail_threshold} (tagged word PPL)")
    print(f"  Max steps: {args.max_steps}")
    print(f"{'='*60}\n")
    
    tag_ids = load_tag_ids()
    print(f"Tag IDs: {len(tag_ids)}")
    
    # Check untagged data exists
    plain_train = PLAIN_DIR / "train_plain.bin"
    plain_val = PLAIN_DIR / "val_plain.bin"
    if not plain_train.exists():
        print(f"ERROR: {plain_train} not found. Run prep_untagged_data.py first.")
        return
    
    # Datasets
    tagged_train_ds = PreTokenizedDataset(TAGGED_DIR / "train_all-heads.bin")
    plain_train_ds = PreTokenizedDataset(plain_train)
    tagged_val_ds = PreTokenizedDataset(TAGGED_DIR / "val_all-heads.bin")
    plain_val_ds = PreTokenizedDataset(plain_val)
    
    tagged_train_loader = DataLoader(
        tagged_train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True,
    )
    plain_train_loader = DataLoader(
        plain_train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True,
    )
    tagged_val_loader = DataLoader(
        tagged_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,
    )
    plain_val_loader = DataLoader(
        plain_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,
    )
    
    mixed_loader = MixedDataLoader(tagged_train_loader, plain_train_loader, args.mix_ratio)
    
    # Build model and load checkpoint
    model = build_model()
    
    resume_path = args.resume or str(Path("/data/hydra/checkpoints_130m/hydra_all-heads_best.pt"))
    if Path(resume_path).exists():
        print(f"\nLoading base checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        base_step = ckpt.get("step", 0)
        base_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Base model: step {base_step}, val_loss={base_val_loss:.4f}")
    else:
        print(f"WARNING: No checkpoint at {resume_path}, training from scratch!")
    
    # Baseline eval before any training
    print("\n--- Baseline evaluation (before mixed training) ---")
    _, _, baseline_tagged_ppl = evaluate(model, tagged_val_loader, tag_ids, label="tagged baseline")
    _, _, baseline_plain_ppl = evaluate(model, plain_val_loader, tag_ids, label="plain baseline")
    print(f"  Baseline tagged word PPL: {baseline_tagged_ppl:.2f}")
    print(f"  Baseline plain word PPL: {baseline_plain_ppl:.2f}")
    
    # Optimizer — fresh (don't reuse old optimizer state for new LR)
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "norm" not in n and "bias" not in n],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if "norm" in n or "bias" in n],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=LR, betas=(0.9, 0.95))
    scaler = GradScaler("cuda") if USE_AMP else None
    
    def lr_schedule(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(args.max_steps - WARMUP_STEPS, 1)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    
    # Training loop
    model.train()
    step = 0
    accum_loss = 0
    accum_count = 0
    tagged_batches = 0
    plain_batches = 0
    log_file = LOG_DIR / "train_130m_mixed.jsonl"
    best_combined_loss = float("inf")
    bailed = False
    
    print(f"\nStarting mixed training for {args.max_steps} steps...")
    t0 = time.time()
    tokens_since_log = 0
    
    optimizer.zero_grad()
    
    while step < args.max_steps:
        batch, is_tagged = next(mixed_loader)
        x = batch.to(DEVICE)
        
        if is_tagged:
            tagged_batches += 1
        else:
            plain_batches += 1
        
        if USE_AMP:
            with autocast("cuda", dtype=torch.float16):
                outputs = model(x[:, :-1])
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                # For tagged batches, compute loss on ALL tokens (including tags)
                # The model needs to learn to predict tags too, for tag-conditioned mode
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    x[:, 1:].reshape(-1),
                )
                loss = loss / GRAD_ACCUM
            scaler.scale(loss).backward()
        else:
            outputs = model(x[:, :-1])
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                x[:, 1:].reshape(-1),
            )
            loss = loss / GRAD_ACCUM
            loss.backward()
        
        accum_loss += loss.item() * GRAD_ACCUM
        accum_count += 1
        tokens_since_log += BATCH_SIZE * MAX_SEQ_LEN
        
        if accum_count % GRAD_ACCUM == 0:
            if USE_AMP:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            
            if step % LOG_EVERY == 0:
                elapsed = time.time() - t0
                avg_loss = accum_loss / (LOG_EVERY * GRAD_ACCUM)
                lr_now = scheduler.get_last_lr()[0]
                tok_per_sec = tokens_since_log / max(elapsed, 0.001)
                ppl = math.exp(min(avg_loss, 20))
                tag_frac = tagged_batches / max(tagged_batches + plain_batches, 1)
                
                log_entry = {
                    "step": step, "loss": round(avg_loss, 4),
                    "ppl": round(ppl, 1), "lr": round(lr_now, 7),
                    "tok_per_sec": round(tok_per_sec),
                    "tagged_frac": round(tag_frac, 3),
                    "elapsed_min": round(elapsed / 60, 1),
                }
                print(f"  step {step:6d} | loss {avg_loss:.4f} | ppl {ppl:8.1f} | "
                      f"lr {lr_now:.2e} | {tok_per_sec:.0f} tok/s | "
                      f"tag_frac {tag_frac:.2f}")
                
                with open(log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + "\n")
                
                accum_loss = 0
                tokens_since_log = 0
                t0 = time.time()
            
            # Evaluation
            if step % EVAL_EVERY == 0:
                print(f"\n--- Eval at step {step} ---")
                _, _, tagged_ppl = evaluate(model, tagged_val_loader, tag_ids, label="tagged")
                _, plain_word_loss, plain_ppl = evaluate(model, plain_val_loader, tag_ids, label="plain")
                
                # Log
                eval_entry = {
                    "step": step, "eval": True,
                    "tagged_word_ppl": round(tagged_ppl, 2),
                    "plain_word_ppl": round(plain_ppl, 2),
                    "baseline_tagged_ppl": round(baseline_tagged_ppl, 2),
                    "baseline_plain_ppl": round(baseline_plain_ppl, 2),
                    "tagged_ppl_delta": round(tagged_ppl - baseline_tagged_ppl, 2),
                }
                with open(log_file, 'a') as f:
                    f.write(json.dumps(eval_entry) + "\n")
                
                print(f"  Tagged PPL: {tagged_ppl:.2f} (baseline: {baseline_tagged_ppl:.2f}, Δ={tagged_ppl - baseline_tagged_ppl:+.2f})")
                print(f"  Plain PPL:  {plain_ppl:.2f} (baseline: {baseline_plain_ppl:.2f}, Δ={plain_ppl - baseline_plain_ppl:+.2f})")
                
                # Bail check
                if tagged_ppl > args.bail_threshold:
                    print(f"\n⚠️ BAIL: Tagged word PPL ({tagged_ppl:.2f}) exceeds threshold ({args.bail_threshold})!")
                    print(f"  Saving checkpoint and stopping.")
                    save_checkpoint(model, optimizer, scaler, step, "bail")
                    bailed = True
                    break
                
                # Save best (combined metric: average of tagged + plain word PPL)
                combined = (tagged_ppl + plain_ppl) / 2
                if combined < best_combined_loss:
                    best_combined_loss = combined
                    save_checkpoint(model, optimizer, scaler, step, "best")
                    print(f"  New best! combined_ppl={combined:.2f}")
                
                model.train()
                t0 = time.time()
            
            if step % SAVE_EVERY == 0:
                save_checkpoint(model, optimizer, scaler, step, f"step{step}")
    
    if not bailed:
        save_checkpoint(model, optimizer, scaler, step, "final")
    
    print(f"\n{'='*60}")
    print(f"Mixed training {'BAILED' if bailed else 'complete'} at step {step}")
    print(f"Best combined PPL: {best_combined_loss:.2f}")
    print(f"{'='*60}")


def save_checkpoint(model, optimizer, scaler, step, tag):
    path = CKPT_DIR / f"hydra_mixed_{tag}.pt"
    save_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": {"d_model": D_MODEL, "n_layer": N_LAYER, "vocab_size": VOCAB_SIZE},
    }
    if scaler is not None:
        save_dict["scaler"] = scaler.state_dict()
    torch.save(save_dict, path)
    size_mb = path.stat().st_size / 1e6
    print(f"  Saved {path} ({size_mb:.0f}MB)")


def main():
    parser = argparse.ArgumentParser(description="Hydra 130M Mixed Training")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint (default: best tagged checkpoint)")
    parser.add_argument("--mix-ratio", type=float, default=0.5,
                       help="Probability of tagged batch (0.5 = equal mix)")
    parser.add_argument("--bail-threshold", type=float, default=DEFAULT_BAIL_THRESHOLD,
                       help="Stop if tagged word PPL exceeds this")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()
    
    if args.lr != LR:
        global LR
        LR = args.lr
    
    train(args)


if __name__ == "__main__":
    main()
