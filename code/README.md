# Hydra Code

## hydra_phase0.py
Curvature mapping experiment: fine-tune GPT-2 Small on register-specific corpora 
(AAVE, legal, literary, control), extract per-token embedding drift, and identify 
"curved" vs "flat" tokens on the speech manifold.

Usage:
```bash
python3 hydra_phase0.py --step download    # fetch corpora
python3 hydra_phase0.py --step finetune    # fine-tune GPT-2 x4
python3 hydra_phase0.py --step extract     # compute curvature map
python3 hydra_phase0.py --step all         # run everything
```

## lambda_bridge.py
Shadow memory system using the modal lambda knowledge store. 
Shadows markdown-based memory with structured lambda calculus facts.
Includes RLHF-style recall tracking (hit/miss/partial scoring).

Usage:
```bash
python3 lambda_bridge.py assert <operator> key=value ...
python3 lambda_bridge.py query [operator] [key=value ...]
python3 lambda_bridge.py search <term> [term2 ...]
python3 lambda_bridge.py history <entity>
python3 lambda_bridge.py recall-score hit|miss|partial [note]
python3 lambda_bridge.py recall-stats
```
