#!/usr/bin/env python3
"""
Hydra Phase 0: Curvature Mapping
================================
Fine-tune GPT-2 Small on register-specific corpora, then measure
per-token embedding drift to identify "curved" vs "flat" tokens.

Usage: python3 hydra_phase0.py [--step download|finetune|extract|all]
"""

import argparse
import json
import os
import numpy as np
import torch
from pathlib import Path

BASE_DIR = Path("/data/hydra")
CORPORA_DIR = BASE_DIR / "corpora"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Fine-tuning config
FT_CONFIG = {
    "epochs": 3,
    "batch_size": 4,
    "grad_accum": 8,  # effective batch 32
    "lr": 5e-5,
    "max_length": 512,
    "warmup_ratio": 0.1,
    "save_steps": 500,
    "logging_steps": 50,
    "max_train_tokens": 10_000_000,  # ~10M tokens per corpus
}

REGISTERS = {
    "aave": {
        "hf_dataset": "wikipedia",  # placeholder — we'll use TwitterAAE or CORAAL
        "description": "African American Vernacular English",
    },
    "legal": {
        "hf_dataset": "pile-of-law/pile-of-law",
        "subset": "courtlistener_docket_entry_documents",
        "description": "Legal/formal English",
    },
    "literary": {
        "hf_dataset": "sedthh/gutenberg_english",
        "description": "Literary/archaic English (pre-1923 Gutenberg)",
    },
    "control": {
        "hf_dataset": "Salesforce/wikitext",
        "subset": "wikitext-103-raw-v1",
        "description": "WikiText-103 (same as base — measures fine-tuning noise)",
    },
}


def download_corpora():
    """Download and prepare register-specific corpora."""
    from datasets import load_dataset
    
    os.makedirs(CORPORA_DIR, exist_ok=True)
    max_chars = FT_CONFIG["max_train_tokens"] * 4  # rough chars-to-tokens ratio
    
    # --- AAVE: Use TwitterAAE (Blodgett et al.) ---
    # If not available on HF, fall back to a synthetic approach
    print("=== Downloading AAVE corpus ===")
    aave_path = CORPORA_DIR / "aave_train.txt"
    if not aave_path.exists():
        try:
            # Try CORAAL or TwitterAAE
            ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
            # Fall back: use a diverse subset we can filter later
            # For now, let's use the Urban Dictionary dataset as a proxy for informal/slang
            ds = load_dataset("Styxxxx/urban_dictionary_full", split="train", streaming=True)
            texts = []
            total_chars = 0
            for i, example in enumerate(ds):
                text = example.get("meaning", "") + " " + example.get("example", "")
                if len(text) > 50:
                    texts.append(text)
                    total_chars += len(text)
                if total_chars > max_chars:
                    break
            with open(aave_path, 'w') as f:
                f.write('\n'.join(texts))
            print(f"  Saved {len(texts)} examples, {total_chars:,} chars")
        except Exception as e:
            print(f"  AAVE download failed: {e}")
            print("  Trying alternative: reddit casual speech...")
            try:
                ds = load_dataset("sentence-transformers/reddit-title-body", split="train", streaming=True)
                texts = []
                total_chars = 0
                for example in ds:
                    text = example.get("body", "")
                    if len(text) > 100:
                        texts.append(text)
                        total_chars += len(text)
                    if total_chars > max_chars:
                        break
                with open(aave_path, 'w') as f:
                    f.write('\n'.join(texts))
                print(f"  Saved {len(texts)} examples, {total_chars:,} chars")
            except Exception as e2:
                print(f"  Alternative also failed: {e2}")
                print("  Will skip AAVE for now")
    else:
        print(f"  Already exists: {aave_path}")
    
    # --- Legal ---
    print("=== Downloading Legal corpus ===")
    legal_path = CORPORA_DIR / "legal_train.txt"
    if not legal_path.exists():
        try:
            ds = load_dataset("pile-of-law/pile-of-law", "courtlistener_docket_entry_documents",
                            split="train", streaming=True, trust_remote_code=True)
            texts = []
            total_chars = 0
            for example in ds:
                text = example.get("text", "")
                if len(text) > 200:
                    texts.append(text[:5000])  # cap individual docs
                    total_chars += len(texts[-1])
                if total_chars > max_chars:
                    break
            with open(legal_path, 'w') as f:
                f.write('\n'.join(texts))
            print(f"  Saved {len(texts)} documents, {total_chars:,} chars")
        except Exception as e:
            print(f"  Legal download failed: {e}")
            print("  Trying alternative: EuroParl...")
            try:
                ds = load_dataset("europarl_bilingual", lang1="en", lang2="fr",
                                split="train", streaming=True)
                texts = []
                total_chars = 0
                for example in ds:
                    text = example["translation"]["en"]
                    if len(text) > 100:
                        texts.append(text)
                        total_chars += len(text)
                    if total_chars > max_chars:
                        break
                with open(legal_path, 'w') as f:
                    f.write('\n'.join(texts))
                print(f"  Saved {len(texts)} examples, {total_chars:,} chars")
            except Exception as e2:
                print(f"  Alternative also failed: {e2}")
    else:
        print(f"  Already exists: {legal_path}")
    
    # --- Literary ---
    print("=== Downloading Literary corpus ===")
    literary_path = CORPORA_DIR / "literary_train.txt"
    if not literary_path.exists():
        try:
            ds = load_dataset("sedthh/gutenberg_english", split="train", streaming=True)
            texts = []
            total_chars = 0
            for example in ds:
                text = example.get("text", "")
                if len(text) > 500:
                    texts.append(text[:10000])
                    total_chars += len(texts[-1])
                if total_chars > max_chars:
                    break
            with open(literary_path, 'w') as f:
                f.write('\n'.join(texts))
            print(f"  Saved {len(texts)} texts, {total_chars:,} chars")
        except Exception as e:
            print(f"  Literary download failed: {e}")
    else:
        print(f"  Already exists: {literary_path}")
    
    # --- Control (WikiText-103) ---
    print("=== Downloading Control corpus ===")
    control_path = CORPORA_DIR / "control_train.txt"
    if not control_path.exists():
        try:
            ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
            texts = [t for t in ds["text"] if len(t) > 50]
            total_chars = sum(len(t) for t in texts)
            # Subsample to match other corpora
            if total_chars > max_chars:
                ratio = max_chars / total_chars
                texts = texts[:int(len(texts) * ratio)]
            with open(control_path, 'w') as f:
                f.write('\n'.join(texts))
            total = sum(len(t) for t in texts)
            print(f"  Saved {len(texts)} passages, {total:,} chars")
        except Exception as e:
            print(f"  Control download failed: {e}")
    else:
        print(f"  Already exists: {control_path}")
    
    # Summary
    print("\n=== Corpus Summary ===")
    for name in ["aave", "legal", "literary", "control"]:
        path = CORPORA_DIR / f"{name}_train.txt"
        if path.exists():
            size = path.stat().st_size
            lines = sum(1 for _ in open(path))
            print(f"  {name:12s}: {size/1e6:.1f}MB, {lines:,} lines")
        else:
            print(f"  {name:12s}: MISSING")


def finetune_all():
    """Fine-tune GPT-2 Small on each register corpus."""
    from transformers import (
        GPT2LMHeadModel, GPT2Tokenizer,
        TextDataset, DataCollatorForLanguageModeling,
        TrainingArguments, Trainer
    )
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    for register in ["aave", "legal", "literary", "control"]:
        corpus_path = CORPORA_DIR / f"{register}_train.txt"
        output_dir = MODELS_DIR / f"gpt2-{register}"
        
        if not corpus_path.exists():
            print(f"Skipping {register}: corpus not found")
            continue
        
        if (output_dir / "pytorch_model.bin").exists() or (output_dir / "model.safetensors").exists():
            print(f"Skipping {register}: already fine-tuned at {output_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Fine-tuning GPT-2 on {register}")
        print(f"{'='*60}")
        
        # Load model fresh each time
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.config.pad_token_id = tokenizer.eos_token_id
        
        # Create dataset
        dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=str(corpus_path),
            block_size=FT_CONFIG["max_length"],
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            num_train_epochs=FT_CONFIG["epochs"],
            per_device_train_batch_size=FT_CONFIG["batch_size"],
            gradient_accumulation_steps=FT_CONFIG["grad_accum"],
            learning_rate=FT_CONFIG["lr"],
            warmup_ratio=FT_CONFIG["warmup_ratio"],
            logging_steps=FT_CONFIG["logging_steps"],
            save_steps=FT_CONFIG["save_steps"],
            save_total_limit=1,
            fp16=True,
            report_to="none",
            dataloader_num_workers=2,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
        
        trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print(f"Saved {register} model to {output_dir}")
        
        # Free GPU memory
        del model, trainer
        torch.cuda.empty_cache()


def extract_curvature():
    """Extract embedding drift per token across all fine-tuned models."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    print("Loading base model embeddings...")
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    base_emb = base_model.transformer.wte.weight.detach().cpu().numpy()  # (50257, 768)
    del base_model
    
    vocab_size = base_emb.shape[0]
    print(f"Vocabulary size: {vocab_size}")
    
    # Collect drifts per register
    drifts = {}
    register_embs = {}
    
    for register in ["aave", "legal", "literary", "control"]:
        model_dir = MODELS_DIR / f"gpt2-{register}"
        if not model_dir.exists():
            print(f"Skipping {register}: model not found")
            continue
        
        print(f"Loading {register} model embeddings...")
        model = GPT2LMHeadModel.from_pretrained(str(model_dir))
        emb = model.transformer.wte.weight.detach().cpu().numpy()
        register_embs[register] = emb
        
        # Per-token L2 drift
        drift = np.linalg.norm(emb - base_emb, axis=1)  # (50257,)
        drifts[register] = drift
        
        print(f"  {register}: mean drift={drift.mean():.4f}, max drift={drift.max():.4f}")
        del model
    
    if not drifts:
        print("No fine-tuned models found! Run --step finetune first.")
        return
    
    # Compute max drift across registers (excluding control)
    real_registers = [r for r in drifts if r != "control"]
    if real_registers:
        max_drift = np.max([drifts[r] for r in real_registers], axis=0)  # (50257,)
    else:
        max_drift = list(drifts.values())[0]
    
    control_drift = drifts.get("control", np.zeros(vocab_size))
    
    # Build curvature map
    curvature_map = []
    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id])
        per_register = {r: float(drifts[r][token_id]) for r in drifts}
        curvature_map.append({
            "token_id": token_id,
            "token": token_str,
            "max_drift": float(max_drift[token_id]),
            "control_drift": float(control_drift[token_id]),
            "signal_noise_ratio": float(max_drift[token_id] / (control_drift[token_id] + 1e-8)),
            "per_register": per_register,
        })
    
    # Sort by max drift descending
    curvature_map.sort(key=lambda x: x["max_drift"], reverse=True)
    
    # Save full results
    results_path = RESULTS_DIR / "curvature_map.json"
    with open(results_path, 'w') as f:
        json.dump(curvature_map, f, indent=2)
    print(f"\nSaved curvature map to {results_path}")
    
    # === Analysis ===
    
    all_drifts = np.array([x["max_drift"] for x in curvature_map])
    all_snr = np.array([x["signal_noise_ratio"] for x in curvature_map])
    
    print(f"\n{'='*60}")
    print("CURVATURE MAP ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\nDrift distribution:")
    print(f"  Mean:   {all_drifts.mean():.4f}")
    print(f"  Median: {np.median(all_drifts):.4f}")
    print(f"  Std:    {all_drifts.std():.4f}")
    print(f"  p10:    {np.percentile(all_drifts, 10):.4f}")
    print(f"  p50:    {np.percentile(all_drifts, 50):.4f}")
    print(f"  p90:    {np.percentile(all_drifts, 90):.4f}")
    print(f"  p99:    {np.percentile(all_drifts, 99):.4f}")
    
    # Check bimodality
    median = np.median(all_drifts)
    below = all_drifts[all_drifts < median]
    above = all_drifts[all_drifts >= median]
    print(f"\n  Below-median mean: {below.mean():.4f}")
    print(f"  Above-median mean: {above.mean():.4f}")
    print(f"  Ratio: {above.mean() / (below.mean() + 1e-8):.1f}x")
    
    print(f"\nControl (fine-tuning noise) drift:")
    ctrl_drifts = np.array([x["control_drift"] for x in curvature_map])
    print(f"  Mean: {ctrl_drifts.mean():.4f}")
    print(f"  Max:  {ctrl_drifts.max():.4f}")
    
    print(f"\nSignal-to-noise ratio (max_drift / control_drift):")
    print(f"  Mean SNR: {all_snr.mean():.1f}")
    print(f"  Median SNR: {np.median(all_snr):.1f}")
    
    # Top 50 most curved tokens
    print(f"\n{'='*60}")
    print("TOP 50 MOST CURVED TOKENS (highest embedding drift)")
    print(f"{'='*60}")
    for i, entry in enumerate(curvature_map[:50]):
        token_repr = repr(entry["token"])
        regs = entry["per_register"]
        reg_str = " | ".join(f"{r}={v:.3f}" for r, v in sorted(regs.items()))
        print(f"  {i+1:3d}. {token_repr:20s} drift={entry['max_drift']:.4f}  SNR={entry['signal_noise_ratio']:.1f}  [{reg_str}]")
    
    # Bottom 50 flattest tokens
    print(f"\n{'='*60}")
    print("TOP 50 FLATTEST TOKENS (lowest embedding drift)")
    print(f"{'='*60}")
    for i, entry in enumerate(curvature_map[-50:]):
        token_repr = repr(entry["token"])
        print(f"  {i+1:3d}. {token_repr:20s} drift={entry['max_drift']:.6f}  SNR={entry['signal_noise_ratio']:.1f}")
    
    # Count curved tokens at various thresholds
    print(f"\n{'='*60}")
    print("CURVED TOKEN COUNTS BY THRESHOLD")
    print(f"{'='*60}")
    for threshold in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        count = sum(1 for d in all_drifts if d > threshold)
        pct = count / vocab_size * 100
        print(f"  drift > {threshold:.1f}: {count:6d} tokens ({pct:.1f}%)")
    
    # Which register causes the most drift per token?
    print(f"\n{'='*60}")
    print("WHICH REGISTER DOMINATES EACH TOKEN'S DRIFT?")
    print(f"{'='*60}")
    register_dominance = {}
    for entry in curvature_map:
        if entry["max_drift"] > 1.0:  # only curved tokens
            max_reg = max(entry["per_register"], key=entry["per_register"].get)
            register_dominance[max_reg] = register_dominance.get(max_reg, 0) + 1
    for reg, count in sorted(register_dominance.items(), key=lambda x: -x[1]):
        print(f"  {reg:12s}: {count:5d} tokens dominated")
    
    # Save summary
    summary = {
        "total_tokens": vocab_size,
        "drift_stats": {
            "mean": float(all_drifts.mean()),
            "median": float(np.median(all_drifts)),
            "std": float(all_drifts.std()),
            "p90": float(np.percentile(all_drifts, 90)),
            "p99": float(np.percentile(all_drifts, 99)),
        },
        "control_drift_mean": float(ctrl_drifts.mean()),
        "curved_counts": {
            f">{t}": int(sum(1 for d in all_drifts if d > t))
            for t in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        },
        "register_dominance": register_dominance,
    }
    summary_path = RESULTS_DIR / "curvature_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Hydra Phase 0: Curvature Mapping")
    parser.add_argument("--step", default="all",
                       choices=["download", "finetune", "extract", "all"],
                       help="Which step to run")
    args = parser.parse_args()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if args.step in ("download", "all"):
        download_corpora()
    
    if args.step in ("finetune", "all"):
        finetune_all()
    
    if args.step in ("extract", "all"):
        extract_curvature()


if __name__ == "__main__":
    main()
