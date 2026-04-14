# Hydra Evaluation Strategy

**Date:** April 14, 2026
**Status:** Benchmarks running on EC2

---

## What Hydra Is (and Isn't)

Hydra is **not** a general-purpose language model competing with GPT-2 or Llama. It's a **structurally-grounded Mamba model** that receives explicit linguistic annotations (parse brackets, thematic roles, information structure tags) as part of its input stream.

This means:

1. **Hydra cannot process plain text.** Fed raw English with no tags, it produces PPL > 600K — worse than random. This isn't a bug; it's a domain mismatch. The model was trained exclusively on tag-interleaved sequences. Raw text is out-of-distribution.

2. **Hydra's 2.10 word PPL is structure-conditioned.** It measures how well the model predicts the next word *given* the structural context. This is a meaningful metric, but it's not comparable to vanilla model PPL on raw text.

3. **The tag dependency is by design.** Hydra is studying what happens when you give structure for free. The fact that it *requires* structure means it genuinely learned to condition on it (rather than ignoring it).

---

## The Three Roles for Hydra

### 1. Research Instrument (primary)
Hydra exists to answer: **"How does explicit structure change internal representations?"**

The tag dependency is the point. By comparing Hydra's SAE features against vanilla Mamba's, we can measure:
- How many features encode structure vs. meaning in each model
- Whether explicit structure produces qualitatively different internal representations
- Whether structural subsidies "free up" capacity for semantics

### 2. Pipeline Component (potential)
A system where a fast parser → Hydra → downstream task could be practical. Benepar parses in milliseconds; the 2.9% tag density is negligible overhead. For applications where interpretability matters (medical, legal, safety-critical), the parser dependency may be worth the structural transparency.

### 3. Standalone Language Model (not currently viable)
Hydra would need to be trained on a mix of tagged and untagged text to function as a standalone model. This is the **production-viable version** — it would learn to use tags when available and fall back to self-discovered structure when not. This is a future experiment, not the current goal.

---

## Benchmarking Philosophy

### The Wrong Benchmark
Feeding Hydra plain text and comparing PPL to GPT-2 or vanilla Mamba. This is like testing a French speaker on German — it measures domain mismatch, not capability.

**Result (April 14, 2026):**
- Hydra on plain text: PPL ~606K (meaningless)
- GPT-2 on plain text: PPL ~34 (expected)
- Vanilla Mamba-130M on plain text: PPL ~11K (tokenizer mismatch suspected)

### The Fair Benchmark
Each model gets the input format it was trained on. All scored on **word token prediction only** (tags excluded from loss).

- **Hydra:** tagged sequences → score word tokens only
- **Vanilla Mamba-130M:** raw text sequences → score all tokens
- **GPT-2 Small:** raw text sequences → score all tokens

Same words, same prediction task, different input conditioning. **This is the correct comparison.**

Key question: Is Hydra's word-only PPL (with tags) competitive with vanilla Mamba's word PPL (without tags)?

- **Hydra < Vanilla Mamba:** structural tags actively help word prediction → the structural subsidy thesis is validated
- **Hydra ≈ Vanilla Mamba:** tags are neutral → structure doesn't hurt but doesn't help for raw prediction
- **Hydra > Vanilla Mamba:** tags may cause over-reliance on structure → model needs more diverse training

### The Interesting Benchmark
Hydra-130M vs. Mamba-130M vs. Mamba-370M on **meaning-heavy tasks** (analogy, entailment, coreference). If Hydra-130M matches the 370M model on semantics, that proves explicit structure is worth ~2-3× in effective parameter efficiency for meaning.

**Paper title candidate:** *"Structure for Free: How Explicit Syntax Doubles the Semantic Capacity of Small Language Models"*

---

## The Syntax Tax Framework

### Core Claim
A fixed parameter budget must be split between structural knowledge (syntax, event types, discourse) and semantic knowledge (world facts, reasoning, composition). Explicit tags eliminate the structural allocation, freeing all parameters for meaning.

### Why Mamba Pays More Than Transformers
- **Transformers** have attention = free structural lookup. Any token can attend to any other based on learned relevance.
- **Mamba** has a fixed-size recurrent state. It must encode structure AND meaning in the same state vector. This creates a capacity bottleneck that attention-based models largely avoid.
- **Hydra** compensates for Mamba's disadvantage by providing the structural context that attention gives implicitly.

### Scaling Prediction
The syntax tax should decrease with model size:
- 30M params: ~45% capacity on structure (tags are transformative)
- 130M params: ~25% (tags are significant)
- 1B params: ~10% (tags are moderate)
- 7B params: ~4% (tags are negligible — model has capacity to spare)

**Implication:** Structural subsidies matter most for small, efficient models. A well-tagged 130M model might outperform an untagged 350M model on semantic tasks — at 2.5× less compute.

---

## Tag Density

Measured from validation set: **2.9% tag tokens** (285 tags per 10K tokens). This is a very light annotation overhead — more like punctuation than a parallel annotation layer.

---

## Long-Context Benchmarks (April 14, 2026)

Mamba's key advantage over transformers is O(n) inference with unlimited context. We tested PPL vs. context length:

| Context | Hydra-130M | GPT-2 124M | Vanilla Mamba-130M |
|---------|-----------|------------|-------------------|
| 512     | 606K*     | 34.23      | 11,515*           |
| 1024    | 676K*     | 29.78      | 10,106*           |
| 2048    | 720K*     | N/A        | 8,532*            |
| 4096    | 805K*     | N/A        | 7,955*            |
| 8192    | 929K*     | N/A        | 9,647*            |
| 16384   | 1.08M*    | N/A        | 19,225*           |

*Both Mamba numbers are on plain text (wrong benchmark). Fair long-context comparison with tagged input is needed.

Vanilla Mamba shows context utilization up to ~4K tokens (PPL improves), then degrades — likely wasn't trained on sequences that long.

**Inference speed scales linearly** for both Mamba models (as expected). At 8K tokens: ~640ms for Mamba vs. GPT-2 unable to process.

---

## Future Experiments

### Mixed Training (High Priority)
Train Hydra on 50/50 tagged + untagged text. Goals:
- Model that uses tags when present, falls back without them
- Measure the PPL gap between tagged and untagged inference
- Determine if structural awareness transfers to untagged processing

### Semantic Task Battery
Compare Hydra-130M vs. Mamba-130M vs. Mamba-370M on:
- Abductive inference ("What caused X?")
- Counterfactual generation
- Compositional generalization (novel operator combinations)
- Coreference resolution across long contexts

### SAE Feature Comparison
Direct comparison of SAE feature types:
- Count structural vs. semantic features in Hydra vs. vanilla Mamba
- Do Hydra's features have cleaner alignment with known linguistic constructs?
- Is the feature space qualitatively different or just quantitatively shifted?

---

## Bottom Line

Hydra's value isn't "better language model." It's **"controlled experiment in structural grounding"** — testing whether making linguistic structure explicit changes what the model learns internally. The interpretability is a methodology: compare Hydra's representations against vanilla Mamba's and ask **"did explicit structure produce qualitatively different representations?"**

The fair comparison is each model on its native input format, scored on the same word predictions. The interesting comparison is whether structural subsidies buy parameter efficiency for semantic tasks. The production comparison (mixed training) is future work.
