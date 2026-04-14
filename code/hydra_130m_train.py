#!/usr/bin/env python3
"""
Hydra 130M Training: Scale-up from 39.3M to 129.2M
====================================================
d_model=768, n_layer=24, vocab_size=50304 (padded)
Uses pre-tokenized binary data from phase05_full_v2.

Based on Phase 1 architecture (Mamba + multi-stream interleaving).
Key changes from 39.3M run:
  - 3.3x more parameters
  - Larger batch size (effective 64 via grad_accum)
  - seq_len=512 (vs 256)
  - Cosine schedule with longer warmup
  - Mixed precision (fp16) for memory efficiency

Usage:
  python3 hydra_130m_train.py --mode all-heads
  python3 hydra_130m_train.py --mode all-heads --resume /data/hydra/checkpoints_130m/hydra_all-heads_best.pt
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

DATA_DIR = Path("/data/hydra/phase05_full_v2")
CKPT_DIR = Path("/data/hydra/checkpoints_130m")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("/data/hydra/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Model config — 130M
D_MODEL = 768
N_LAYER = 24
VOCAB_SIZE = 50304  # padded to multiple of 64 for efficiency
MAX_SEQ_LEN = 512

# Training config
BATCH_SIZE = 16
GRAD_ACCUM = 4       # effective batch = 64
LR = 3e-4            # slightly lower than 39M run (larger model)
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 2000  # longer warmup for larger model
MAX_STEPS = 100_000
EVAL_EVERY = 2000
SAVE_EVERY = 10000
LOG_EVERY = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True  # mixed precision


# ── Dataset (pre-tokenized binary) ────────────────────────────────────

class PreTokenizedDataset(Dataset):
    """Loads pre-tokenized .bin data (int32 flat array)."""
    
    def __init__(self, bin_path, seq_len=MAX_SEQ_LEN):
        self.seq_len = seq_len
        
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


# ── Tag info (for word-only perplexity) ───────────────────────────────

def load_tag_ids(tokenizer_path):
    """Load tag IDs from tokenizer JSON for word-only eval."""
    with open(tokenizer_path) as f:
        d = json.load(f)
    tag_ids = set()
    for tag, tid in d["tag_to_id"].items():
        tag_ids.add(tid)
    return tag_ids


# ── Model ──────────────────────────────────────────────────────────────

def build_model(vocab_size=VOCAB_SIZE):
    """Build 130M Mamba model."""
    from mamba_ssm import MambaLMHeadModel
    from mamba_ssm.models.config_mamba import MambaConfig
    
    config = MambaConfig(
        d_model=D_MODEL,
        n_layer=N_LAYER,
        vocab_size=vocab_size,
    )
    model = MambaLMHeadModel(config).to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params/1e6:.1f}M params ({n_trainable/1e6:.1f}M trainable)")
    print(f"  d_model={D_MODEL}, n_layer={N_LAYER}, vocab={vocab_size}")
    
    return model


# ── Training Loop ──────────────────────────────────────────────────────

def train(mode, resume_from=None, max_steps=MAX_STEPS):
    """Train 130M Mamba on Hydra data."""
    
    print(f"\n{'='*60}")
    print(f"Hydra 130M Training: mode={mode}")
    print(f"{'='*60}\n")
    
    # Load tag IDs for word-only eval
    tok_path = DATA_DIR / "tokenizer_all-heads.json"
    tag_ids = load_tag_ids(tok_path) if tok_path.exists() else set()
    print(f"Tag IDs loaded: {len(tag_ids)} tags")
    
    # Load pre-tokenized data
    train_dataset = PreTokenizedDataset(DATA_DIR / f"train_{mode}.bin", MAX_SEQ_LEN)
    val_dataset = PreTokenizedDataset(DATA_DIR / f"val_{mode}.bin", MAX_SEQ_LEN)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    
    # Build model
    model = build_model()
    
    # Optimizer — AdamW with decoupled weight decay
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "norm" not in n and "bias" not in n],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if "norm" in n or "bias" in n],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=LR, betas=(0.9, 0.95))
    
    # Mixed precision scaler
    scaler = GradScaler("cuda") if USE_AMP else None
    
    # Cosine schedule with warmup
    def lr_schedule(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(max_steps - WARMUP_STEPS, 1)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))  # min LR = 10% of peak
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    
    # Resume
    start_step = 0
    best_val_loss = float("inf")
    if resume_from and Path(resume_from).exists():
        print(f"Resuming from {resume_from}")
        ckpt = torch.load(resume_from, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        # Advance scheduler
        for _ in range(start_step):
            scheduler.step()
        print(f"  Resumed at step {start_step}, best_val_loss={best_val_loss:.4f}")
    
    # Training
    model.train()
    step = start_step
    accum_loss = 0
    accum_count = 0
    log_file = LOG_DIR / f"train_130m_{mode}.jsonl"
    
    print(f"\nTraining for {max_steps} steps (from step {start_step})...")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"Device: {DEVICE}, AMP: {USE_AMP}")
    print(f"Tokens per step: {BATCH_SIZE * MAX_SEQ_LEN:,}")
    print(f"Tokens per effective step: {BATCH_SIZE * GRAD_ACCUM * MAX_SEQ_LEN:,}")
    total_train_tokens = max_steps * BATCH_SIZE * GRAD_ACCUM * MAX_SEQ_LEN
    print(f"Total training tokens: {total_train_tokens/1e9:.2f}B")
    t0 = time.time()
    tokens_since_log = 0
    
    data_iter = iter(train_loader)
    optimizer.zero_grad()
    
    while step < max_steps:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        x = batch.to(DEVICE)
        
        # Forward with mixed precision
        if USE_AMP:
            with autocast("cuda", dtype=torch.float16):
                outputs = model(x[:, :-1])
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
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
        
        accum_loss += loss.item() * GRAD_ACCUM  # unscale for logging
        accum_count += 1
        tokens_since_log += BATCH_SIZE * MAX_SEQ_LEN
        
        # Gradient step
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
            
            # Logging
            if step % LOG_EVERY == 0:
                elapsed = time.time() - t0
                avg_loss = accum_loss / (LOG_EVERY * GRAD_ACCUM)
                lr_now = scheduler.get_last_lr()[0]
                tok_per_sec = tokens_since_log / max(elapsed, 0.001)
                ppl = math.exp(min(avg_loss, 20))  # cap to avoid overflow
                
                log_entry = {
                    "step": step,
                    "loss": round(avg_loss, 4),
                    "ppl": round(ppl, 1),
                    "lr": round(lr_now, 7),
                    "tok_per_sec": round(tok_per_sec),
                    "elapsed_min": round(elapsed / 60, 1),
                    "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                }
                print(f"  step {step:6d} | loss {avg_loss:.4f} | ppl {ppl:8.1f} | "
                      f"lr {lr_now:.2e} | {tok_per_sec:.0f} tok/s | "
                      f"GPU {log_entry['gpu_mem_gb']:.1f}GB | {elapsed/60:.1f}m")
                
                with open(log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + "\n")
                
                accum_loss = 0
                tokens_since_log = 0
                t0 = time.time()
            
            # Evaluation
            if step % EVAL_EVERY == 0:
                val_loss, word_val_loss = evaluate(model, val_loader, tag_ids)
                word_ppl = math.exp(min(word_val_loss, 20))
                total_ppl = math.exp(min(val_loss, 20))
                
                print(f"  >>> val_loss {val_loss:.4f} (total_ppl {total_ppl:.1f}) | "
                      f"word_loss {word_val_loss:.4f} (word_ppl {word_ppl:.1f})")
                
                with open(log_file, 'a') as f:
                    f.write(json.dumps({
                        "step": step, "eval": True,
                        "val_loss": round(val_loss, 4),
                        "val_ppl": round(total_ppl, 2),
                        "word_val_loss": round(word_val_loss, 4),
                        "word_ppl": round(word_ppl, 2),
                    }) + "\n")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, scaler, step, mode, "best", best_val_loss)
                    print(f"  >>> New best! val_loss={val_loss:.4f}")
                
                model.train()
                t0 = time.time()  # reset timer after eval
            
            # Periodic save
            if step % SAVE_EVERY == 0:
                save_checkpoint(model, optimizer, scaler, step, mode, f"step{step}", best_val_loss)
    
    # Final save
    save_checkpoint(model, optimizer, scaler, step, mode, "final", best_val_loss)
    print(f"\nTraining complete at step {step}.")
    print(f"Best val_loss: {best_val_loss:.4f} (word_ppl ~{math.exp(best_val_loss):.1f})")


def evaluate(model, val_loader, tag_ids, max_batches=200):
    """Compute validation loss (total and word-only)."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    word_loss = 0
    word_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            
            x = batch.to(DEVICE)
            
            with autocast("cuda", dtype=torch.float16) if USE_AMP else torch.no_grad():
                outputs = model(x[:, :-1])
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Per-token loss (in fp32 for accuracy)
            logits_f32 = logits.float()
            targets = x[:, 1:]
            loss_per_token = torch.nn.functional.cross_entropy(
                logits_f32.reshape(-1, logits_f32.size(-1)),
                targets.reshape(-1),
                reduction='none',
            ).reshape(targets.shape)
            
            # Total loss
            total_loss += loss_per_token.sum().item()
            total_tokens += targets.numel()
            
            # Word-only loss (exclude tag tokens)
            if tag_ids:
                word_mask = torch.ones_like(targets, dtype=torch.bool)
                for tid in tag_ids:
                    word_mask &= (targets != tid)
                if word_mask.sum() > 0:
                    word_loss += (loss_per_token * word_mask).sum().item()
                    word_tokens += word_mask.sum().item()
    
    avg_total = total_loss / max(total_tokens, 1)
    avg_word = word_loss / max(word_tokens, 1) if word_tokens > 0 else avg_total
    
    return avg_total, avg_word


def save_checkpoint(model, optimizer, scaler, step, mode, tag, best_val_loss):
    """Save model checkpoint."""
    path = CKPT_DIR / f"hydra_{mode}_{tag}.pt"
    save_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "mode": mode,
        "best_val_loss": best_val_loss,
        "config": {
            "d_model": D_MODEL,
            "n_layer": N_LAYER,
            "vocab_size": VOCAB_SIZE,
        },
    }
    if scaler is not None:
        save_dict["scaler"] = scaler.state_dict()
    torch.save(save_dict, path)
    size_mb = path.stat().st_size / 1e6
    print(f"  Saved {path} ({size_mb:.0f}MB)")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hydra 130M Training")
    parser.add_argument("--mode", default="all-heads",
                       help="Training mode (matches pre-tokenized .bin filename)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()
    
    train(args.mode, resume_from=args.resume, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
