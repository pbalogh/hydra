# Hydra: Multi-Head Linguistic Preprocessing Architecture

**Status:** Design phase
**Date:** April 3, 2026
**Predecessor:** Grammar-Mamba (Phase 1 complete, v2 killed)
**Target:** EMNLP 2026 Budapest
**Location:** T4 EC2 (`/data/hydra/`)

---

## Core Thesis

Transformers waste enormous compute on **manifold origami** — folding and unfolding a shared representational space to create local geometries where simple operations can act. Hydra replaces this with **manifold-aware processing from the ground up**: a register head determines *where on the manifold* the current text lives, embeddings are interpolated from sparse landmarks at that location, and parsing heads operate on tokens whose meanings have already been resolved for the current world/register/fiber.

Each head is a **grounding axis** — a coordinate transformation from surface word-space into one dimension of the knowledge manifold. But the foundational axis is **manifold location itself**: which world, which register, which fiber. All other axes depend on it because the same token has different syntactic behavior, different thematic roles, and different discourse functions depending on where you are on the manifold.

**Central architectural claim:** The first grounding axis is manifold location (which world/register/fiber), and all other axes depend on it. This is what makes Hydra different from "Mamba with extra tags."

## Theoretical Foundation

### Words Into Worlds

The fundamental operation of language processing is coordinate transformation: words arrive in surface encoding; knowledge lives in structured worlds. The gap between "here are words" and "here is what they mean" is the mapping from one manifold to another.

### Fiber Bundle Structure

The space of human language has the structure of a **fiber bundle**:
- **Base space:** The token stream (surface words in sequence)
- **Fibers:** Different parsing dimensions, each with its own local rules
- **Projection maps:** Each Hydra head implements one projection from base to fiber
- **Transition functions:** Where fibers interact (e.g., phrase structure constrains thematic roles)

Transformers implement this implicitly through multi-head attention across many layers. Hydra makes it explicit through named, specialized heads.

### Grounding as Coordinate Coverage

The grounding problem decomposes into specific missing coordinate transformations:
- **Phrase structure:** Words → syntactic positions (brackets, attachment)
- **Thematic fit:** Syntax → who-does-what-to-whom (agent, patient, instrument)
- **Information structure:** Roles → discourse relevance (given vs. new, topic vs. focus)
- **[Discoverable]:** Unknown dimensions that emerge from data

Being "ungrounded" = missing axes. Each head adds one grounding dimension.

### No Compromise Embeddings

Standard transformers map each token to ONE point in embedding space. But tokens don't live at points — they live on curves parameterized by context, register, and world. "Bad" in AAVE (positive) and "bad" in standard English (negative) are geometrically distant; no single point serves both. The embedding layer collapses the curve to a point, then the transformer spends layers 1–4 reconstructing what was destroyed.

Worse: the "compromise" embedding (centroid of all meanings) is actively deceptive — the centroid of "terrible" and "excellent" is "mediocre," which is meaningful in the wrong way.

**Hydra's solution: manifold-aware embeddings.** Instead of one vector per token, maintain a sparse set of landmark embeddings at known manifold locations, plus an interpolation function:

```
embed(token, z) = base(token) + Δ(token, z)
```

where `z` is the manifold coordinate from the register head. Only ~10% of tokens need rich landmark sets (the "curved" tokens like "bad," "sick," "fire"); the rest use the base embedding unchanged (the "flat" tokens like "refrigerator"). This roughly doubles embedding parameters but saves all the disambiguation compute in deeper layers.

### Kripkean Worlds as Manifold Landmarks

The lambda knowledge store's possible worlds (actual, fictional, epistemic, hypothetical, deontic, remembered) and Hydra's manifold locations are **discrete and continuous versions of the same structure**:

| Kripke (logic, discrete) | Manifold (geometry, continuous) |
|---|---|
| Named worlds (w0, w_fiction) | Sparse landmarks |
| Accessibility relations | Paths on the manifold |
| Modal operators (□, ◇) | Quantifiers over neighborhoods |
| "True in world w" | "Embedding at location z" |
| Belief protection | Curvature barrier (no smooth path) |

The lambda store's belief protection rule (fiction can't pollute actual) maps to a **curvature barrier** — a topological discontinuity that prevents smooth drift from fictional to actual coordinates without an explicit confirmation operation. Other transitions ARE smooth: hypothetical → actual is a continuous gradient through decreasing uncertainty.

Register/sociolect and modality are *orthogonal axes* of the same manifold. The full coordinate is `(z_modality, z_register, z_domain, ...)` — a multi-dimensional space where Kripkean worlds are specific slices and Hydra register locations are other slices.

### Tokens as Lambda Terms

Every token in the vocabulary is a **lambda term with free variables**:

```
"bad" = λ(register, polarity, domain). meaning
```

The embedding layer partially applies this with default bindings (standard English, negative, general domain). Each subsequent transformer layer further binds free variables using contextual information from attention. By L11, most variables are bound and the exception handler (Darkness Visible) routes the remaining ambiguity.

This reframes the entire transformer as a **variable-binding machine**: it takes lambda terms with free variables (token embeddings) and progressively grounds them using contextual information until they're fully located on the manifold. The grounding axes (register, phrase structure, thematic fit, information structure) are the **specific free variables being bound**.

The "compromise embedding" problem: the standard embedding layer assigns each token a single point — the centroid of all its possible meanings. But the centroid of "terrible" and "excellent" is "mediocre" — *meaningful in the wrong way*. The embedding isn't blurry; it's deceptive. Hydra's manifold-aware embedding layer resolves the correct meaning *before* downstream processing begins.

### Gaussian Splatting and the Shape of Meaning

We discover the manifold through **sparse observations at known locations** — like Gaussian splatting in 3D graphics. Fine-tune at a few landmark registers (standard English, AAVE, legal, literary), measure how embeddings change, interpolate elsewhere.

But the real goal is not interpolation — it's **discerning the overall shape**. Like Dirac studying his equation and discovering that the *structure of the solution space* required the existence of antimatter before anyone observed it, we hope to discover that the topology of the speech manifold *requires* certain registers, parsing dimensions, or meaning-types to exist — locations that the math predicts but that no one has named.

**The procedure:**

1. **Splat:** Train at known landmarks (standard, AAVE, legal, literary...)
2. **Render:** Run inference on diverse text, measure head disagreement and perplexity residuals
3. **Find holes:** Cluster the high-residual tokens by context — what do failure cases share?
4. **Predict:** Symmetry structure of existing landmarks + pattern of holes → hypothesize what new location must exist
5. **Discover:** Find or create a corpus for the predicted register, add as new landmark
6. **Repeat:** Each cycle reveals finer structure

The topology itself becomes scientifically interesting. Does the manifold have **holes** (concepts unreachable from certain starting points)? **Singularities** (where fibers collide — puns, metaphor, irony)? **High-curvature regions** (where all parsing dimensions interact — slang, code-switching)? **Flat regions** (formal technical writing where every token has one meaning)?

If the manifold has topological invariants — genus, Betti numbers, characteristic classes — those constrain how many fundamentally distinct types of language use can exist. Not how many registers we've observed, but how many the topology *requires*.

The ultimate deliverable isn't a language model. It's a **map of the manifold** — the shape of human meaning-space, empirically derived, with predictions about regions no one has visited yet.

### Garden-Path as Fiber Misidentification

Garden-path sentences are what happens when the parser commits to the wrong fiber. "The horse raced past the barn fell" — the phrase structure head says "raced" is the main verb (active fiber), but the thematic head eventually contradicts this (reduced-relative fiber). Unlike transformers that maintain ambiguity as superposition, Hydra's explicit heads force the collision to be visible — making it a model of human parsing that reproduces failure modes, not just successes.

---

## Architecture

### Pipeline (not parallel)

The heads form a **pipeline**, not a parallel bank. Manifold location must be determined before parsing heads can operate, because parsing rules differ by register/world.

```
Layer 0: Register Head → manifold coordinate z
Layer 1: Manifold-Aware Embedding → embed(token, z) from sparse landmarks
Layer 2: Parsing Heads (phrase structure, thematic, info structure, blank)
         operating on CORRECT embeddings for this manifold location
Layer 3: Body (Mamba) operating on fully grounded tokens
```

### Layer 0: Register/World Head
- **What it does:** Determines manifold location — which register, sociolect, domain, modality
- **Input:** Local context window (previous N tokens)
- **Output:** Continuous manifold coordinate `z = (z_register, z_modality, z_domain, ...)`
- **Architecture:** Small classifier or encoder (1-2 layers)
- **Supervision source:** Corpus metadata (AAVE corpus → z_register=AAVE, legal corpus → z_register=legal, etc.)
- **Connection to lambda store:** z_modality axis corresponds to Kripkean world coordinates; discrete landmarks (actual, fictional, etc.) sit at known points on this axis
- **Why it comes first:** "Bad" in AAVE has different phrase structure, different thematic role, different discourse function than "bad" in standard English. All other heads need to know which rulebook applies.

### Layer 1: Manifold-Aware Embeddings
- **What it does:** Interpolates token embedding from sparse landmarks based on manifold coordinate z
- **Formula:** `embed(token, z) = base(token) + Δ(token, z)`
- **Sparse strategy:** Only ~5,000 "curved" tokens (those whose meaning varies significantly across manifold locations) need landmark deltas. ~45,000 "flat" tokens use base embedding unchanged.
- **Landmark identification:** Fine-tune base model on register-specific corpora, measure embedding drift per token. Tokens with drift > threshold are "curved" and get landmarks.
- **Interpolation:** Small MLP that takes (token_id, z) → Δ ∈ R^768, trained to match fine-tuned embeddings at landmark locations
- **Parameter cost:** ~80M total (2× standard embedding layer) but saves disambiguation compute in deeper layers

### Layer 2: Parsing Heads

Each parsing head is a **lightweight encoder** that produces per-token annotations. The annotations are either interleaved as tags (Option A) or fused as embeddings (Option B). These heads now receive manifold-aware embeddings, not raw token embeddings.

#### Head 1: Phrase Structure
- **What it does:** Constituent bracketing, attachment decisions
- **Supervision source:** Benepar constituency parser (we have this from Grammar-Mamba)
- **Output format:** Grammar tags (S, NP, VP, PP, ...) with open/close brackets
- **Psycholinguistic basis:** Clifton — phrase structure is processed independently and early
- **Grounding axis:** Words → syntactic positions

#### Head 2: Thematic Fit
- **What it does:** Argument structure, role assignment, verb subcategorization
- **Supervision source:** PropBank semantic role labels (ARG0=agent, ARG1=patient, etc.)
- **Output format:** Thematic role tags at argument positions
- **Psycholinguistic basis:** Hoeks — thematic fit influences parsing independently of syntax
- **Grounding axis:** Syntax → operator-argument structure
- **Note:** VerbNet class labels could supplement PropBank with selectional restrictions

#### Head 3: Information Structure
- **What it does:** Given/new tracking, topic/focus identification, discourse coherence
- **Supervision source:** This is the hardest — options:
  - Heuristic: sentence-initial = topic, post-verbal = focus, definite = given, indefinite = new
  - Corpus: OntoNotes coreference chains (entities previously mentioned = given)
  - Learned: Train jointly with body, let it discover what "information structure" means
- **Psycholinguistic basis:** Brown — information structure affects processing independently
- **Grounding axis:** Roles → discourse relevance

#### Head 4: The Blank Head (Discovery)
- **What it does:** Unknown — learns its own parsing dimension from data
- **Supervision source:** None. Same architecture as the other heads but no explicit labels
- **Output format:** Learned tags or embeddings
- **Purpose:** Can the architecture discover parsing dimensions linguists haven't named?
- **Analogy:** Operator discovery found 2 novel primitives beyond Schank's 5

### The Body

A **Mamba SSM** (same as Grammar-Mamba baseline, ~130M parameters) that:
- Receives the enriched token stream (words + head annotations)
- Performs language modeling (next-token prediction)
- Benefits from pre-computed geometry provided by the heads

**Why Mamba, not Transformer?**
- Mamba's channel structure naturally separates timescales (fast syntax, slow discourse)
- Grammar-Mamba proved Mamba can consume structural tags effectively
- Linear scaling with sequence length (no quadratic attention cost)
- The heads handle the "manifold selection" that attention would otherwise do

### Fusion Strategy

**Option A: Tag Interleaving (Grammar-Mamba successor)**
```
Input:  The   cat   sat   on    the   mat
Head 1: (S (NP     )NP (VP     (PP (NP     )NP )PP )VP )S
Head 2:       ARG0        V          ARG-LOC
Head 3:       GIVEN       NEW        GIVEN
Head 4:       ???         ???        ???
Output: (S The (NP cat ARG0 GIVEN ) sat V NEW (PP on (NP the mat ARG-LOC GIVEN ) ) )
```
- **Pro:** Proven to work (Grammar-Mamba Phase 1: 100% center-embedding)
- **Con:** 2-4x token count; Grammar-Mamba couldn't generate fluent text

**Option B: Embedding Fusion**
```
Each head produces a d_head-dimensional embedding per token.
Concatenated: [word_emb | head1_emb | head2_emb | head3_emb | head4_emb]
Projected through a learned fusion layer before entering the body.
```
- **Pro:** No token count increase; smoother gradient flow
- **Con:** Less interpretable; unproven

**Option C: Hybrid**
```
Head 1 (phrase structure) uses tag interleaving (proven, explicit).
Heads 2-4 use embedding fusion (softer, can be uncertain).
```
- **Pro:** Uses the proven approach where we have it; allows soft annotation for harder heads
- **Con:** Asymmetric; more complex

**Recommendation: Start with Option A (all interleaving), since we know it works. If fluency problem persists (Grammar-Mamba dissociation), try Option C.**

---

## Training Plan

### Phase 0: Curvature Mapping (validates the entire premise)
- **Purpose:** Empirically measure which tokens have location-dependent embeddings
- **Method:**
  1. Take GPT-2 Small's pretrained embeddings as "base" (location 0: standard English)
  2. Fine-tune GPT-2 on AAVE corpus → extract embeddings (location 1)
  3. Fine-tune GPT-2 on formal/legal text → extract embeddings (location 2)
  4. Fine-tune GPT-2 on literary/archaic text → extract embeddings (location 3)
  5. Compute per-token embedding drift: `drift(token) = max_i ||embed_i(token) - base(token)||`
  6. Rank tokens by drift → **curvature map** (which tokens are flat vs. curved)
- **Expected result:** Function words mostly flat; culturally loaded words ("bad," "sick," "fire," "dog") highly curved; domain terms moderate
- **Deliverable:** Curvature map + list of ~5,000 curved tokens that need landmark embeddings
- **Compute:** 4× GPT-2 fine-tuning ≈ 4-8 hours on T4
- **Success criterion:** Clear bimodal distribution of drift values (flat vs. curved) AND curved tokens are interpretably the polysemous/register-sensitive ones
- **If this fails:** The manifold-aware embedding idea doesn't hold and we fall back to standard embeddings + parsing heads only

### Phase 0.5: Data Preparation
- **WikiText-103** (same dataset as Grammar-Mamba)
- Parse with benepar → phrase structure tags ✅ (already done)
- Parse with AllenNLP SRL → PropBank thematic role labels
- Annotate information structure (heuristic-based initially)
- Interleave all three tag streams with raw text

### Phase 1: Single-Head Reproduction
- Train Hydra with Head 1 only (phrase structure) on WikiText-103
- **Expected result:** Reproduce Grammar-Mamba Phase 1 (100% center-embedding, structural PPL reduction)
- **Purpose:** Verify the architecture works; establish baseline
- **Compute:** ~24-48 hours on T4

### Phase 2: Two-Head (Phrase + Thematic)
- Add Head 2 (thematic roles from PropBank SRL)
- **Key question:** Does adding thematic information improve fluency?
- **Hypothesis:** Grammar-Mamba's fluency failure was partly due to knowing structure but not argument roles — words were syntactically placed but semantically arbitrary
- **Evaluation:**
  - Word-only perplexity (same as Grammar-Mamba eval)
  - Structural tests (center-embedding, agreement)
  - **NEW:** Thematic fit test — "The doctor examined the [patient/table]" — does the model prefer appropriate fillers?
  - **NEW:** Generation fluency — average coherent output length, human judgment

### Phase 3: Three-Head (Full Hydra)
- Add Head 3 (information structure)
- **Key question:** Does discourse awareness further improve coherence?
- **Evaluation:**
  - All Phase 2 metrics
  - **NEW:** Discourse coherence test — entity tracking across sentences, given/new consistency
  - **NEW:** Narrative continuation — can the model maintain topic and introduce new information appropriately?

### Phase 4: Four-Head (Discovery)
- Add Head 4 (blank/learned)
- **Key question:** Does the blank head converge on something interpretable?
- **Analysis:**
  - Cluster the learned tags — do they form discrete categories?
  - Correlate with known linguistic features — does it rediscover something?
  - Ablation — does removing Head 4 hurt performance? On what tasks?

### Phase 5: Garden-Path Experiments
- Feed garden-path sentences through Hydra
- **Key question:** Do the heads disagree at disambiguation points?
- Measure: head-to-head agreement scores at each token position
- **Prediction:** At "fell" in "The horse raced past the barn fell," Head 1 (phrase structure) and Head 2 (thematic) should show maximum disagreement
- Compare with GPT-2 garden-path results (no reanalysis found — one-stage parser)
- **This is publishable independently** as a psycholinguistic modeling result

---

## Evaluation Framework

### Structural Metrics (from Grammar-Mamba)
- Center-embedding accuracy at depths 1-5
- Agreement attraction accuracy (by distractor type)
- Structural perplexity (by complexity bucket)

### Fluency Metrics (the Grammar-Mamba gap)
- Word-only perplexity (exclude tags from computation)
- Average coherent generation length
- Distinct-1, Distinct-2 (lexical diversity)
- Human evaluation of generated text quality

### Grounding Metrics (new for Hydra)
- Thematic fit accuracy (does the model prefer plausible role fillers?)
- Coreference resolution accuracy (does it track entities?)
- Garden-path sensitivity (do heads disagree at disambiguation?)
- Blank head interpretability (clustering analysis of learned tags)

### The Dissociation Test
- **The key metric:** Does Hydra close the Grammar-Mamba dissociation?
- Grammar-Mamba: perfect structure, garbled text
- Hydra prediction: structure + thematic fit + discourse = coherent text
- If the dissociation persists with 3 heads, the missing dimension is NOT any of the three known parsing dimensions — it's something else (content knowledge? world model? the blank head might find it)

---

## The Unified Theory: How Everything Connects

Hydra is not an isolated project — it's the architectural instantiation of a theoretical framework that unifies all our active work. The connections are not post-hoc; they're structural.

### The Stack

```
Layer         │ Project                     │ What it provides
──────────────┼─────────────────────────────┼──────────────────────────────────
Topology      │ Manifold Origami            │ The shape of meaning-space
              │                             │ (fiber bundle structure, curvature,
              │                             │ topological invariants)
──────────────┼─────────────────────────────┼──────────────────────────────────
Logic         │ Lambda Knowledge Store      │ Named landmarks on the manifold
              │                             │ (Kripkean worlds, accessibility,
              │                             │ belief protection = curvature barriers)
──────────────┼─────────────────────────────┼──────────────────────────────────
Coordinates   │ Hydra                       │ The coordinate system itself
              │                             │ (register head, parsing heads,
              │                             │ manifold-aware embeddings)
──────────────┼─────────────────────────────┼──────────────────────────────────
Validation    │ Mentalese Grammar           │ Ground truth for coordinate axes
              │                             │ (SAE case system = thematic head;
              │                             │ relational features = operator-
              │                             │ argument structure)
──────────────┼─────────────────────────────┼──────────────────────────────────
Operators     │ Operator Discovery          │ The functions that act ON the manifold
              │                             │ (Schankian primitives = manifold-local
              │                             │ operators; ATRANS, PTRANS, etc.)
──────────────┼─────────────────────────────┼──────────────────────────────────
Diagnostics   │ Darkness Visible            │ How existing models approximate this
              │                             │ (L11 routing program = implicit
              │                             │ exception handler for manifold
              │                             │ navigation failures)
──────────────┼─────────────────────────────┼──────────────────────────────────
Failure modes │ Garden-Path experiments     │ Where the manifold breaks
              │                             │ (fiber misidentification, head
              │                             │ disagreement at singularities)
──────────────┼─────────────────────────────┼──────────────────────────────────
Retrieval     │ Query as Dialogue           │ Navigating the manifold via questions
              │                             │ (free variables = unresolved
              │                             │ coordinates; clarification = binding)
──────────────┼─────────────────────────────┼──────────────────────────────────
Runtime       │ Continuous Cognition        │ The manifold persists between prompts
              │                             │ (geometry/content separation means
              │                             │ the store stays active for
              │                             │ consolidation while the model sleeps)
```

### The Core Claim

**Language processing is coordinate transformation on a non-Euclidean manifold.** Every project in our portfolio attacks a different aspect of this claim:

- **What is the manifold?** Manifold Origami (the shape), Lambda Store (the landmarks)
- **How do you navigate it?** Hydra (explicit coordinate transforms), Query as Dialogue (navigation via questions)
- **What lives on it?** Operator Discovery (local operators), Mentalese Grammar (local coordinate systems)
- **How do current models approximate it?** Darkness Visible (implicit routing), Grammar-Mamba (single-axis navigation)
- **Where does it break?** Garden-Path (fiber collisions), the Grammar-Mamba dissociation (missing axes)
- **What persists?** Continuous Cognition (the manifold outlives each prompt), AI Memory Architecture (structured persistence)

### Open Theoretical Questions

1. **Does mentalese change across manifold locations?** If SAE features are the vocabulary of mentalese, and the manifold has location-dependent embeddings, then mentalese grammar rules may also be location-dependent. The case system (nominative/accusative) might work differently in different fibers. This would mean there isn't one mentalese but a *family* of mentaleses parameterized by manifold location.

2. **What are the topological invariants?** If we can compute the genus, Betti numbers, or characteristic classes of the speech manifold, these would tell us how many fundamentally distinct types of language use *must* exist — not how many we've observed, but how many the topology requires.

3. **Is the manifold the same across languages?** If a Japanese speaker and an English speaker share the same meaning-manifold with different surface maps (fiber projections), that's evidence for a universal mentalese. If the manifolds differ (different topology, not just different coordinates), that's evidence for linguistic relativity at the geometric level.

4. **Where are the singularities?** Puns, metaphor, irony, code-switching — these are all points where fibers collide or transition functions become singular. Mapping these singularities would give us a geometric theory of figurative language.

5. **Can the manifold predict new phenomena?** Like Dirac predicting antimatter from holes in his equation: can the symmetry structure of known manifold locations predict the existence of registers, parsing dimensions, or meaning-types that no one has named?

## Connection to Other Projects (Summary)

| Project | Connection to Hydra |
|---|---|
| **Grammar-Mamba** | Direct predecessor. Phase 1 proved single-head interleaving works. Hydra generalizes to multiple heads. |
| **Manifold Origami** | Hydra IS the proposed architecture. Heads are explicit manifold selectors. Body operates in pre-computed geometry. |
| **Mentalese Grammar** | SAE case systems (nominative/accusative) are what the thematic head should learn. Validation target. Open question: does mentalese itself vary across manifold locations? |
| **Darkness Visible** | Terminal-layer routing → Hydra makes routing explicit and early. N2123 exception handling → garden-path fiber collision. |
| **Operator Discovery** | Schankian primitives are manifold-local operators. Hydra's thematic head should discover verb classes that align with ATRANS/PTRANS/etc. |
| **Lambda Knowledge Store** | Kripkean worlds are landmarks on Hydra's manifold. Belief protection = curvature barriers. The store IS the pre-computed geometry engine. |
| **Garden-Path** | Fiber collision experiments. Head disagreement at disambiguation = quantified garden-path effect. |
| **Query as Dialogue** | Underdetermined mappings. When heads can't fully resolve coordinates, the system needs clarification — free variables in lambda terms. |
| **Continuous Cognition** | Hydra separates geometry (heads) from content (body+store). The heads can be frozen while the store updates continuously. |

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Tag interleaving causes same fluency failure as Grammar-Mamba | Medium | Try Option C (embedding fusion for soft heads); the thematic head may fix the root cause |
| PropBank SRL quality on WikiText-103 is poor | Medium | Use AllenNLP's BERT-based SRL; evaluate on gold data subset first |
| Information structure heuristics are too noisy | High | Start with simple given/new based on coreference; iterate |
| Blank head doesn't converge | Medium | That's informative too — means 3 dimensions suffice. Not a project failure. |
| Compute budget (T4 is slow) | Low | Grammar-Mamba Phase 1 trained in ~48 hours on T4. Hydra is same architecture + tags. |
| EMNLP 2026 deadline | High | Need results by ~June 2026. Phase 1-2 achievable. Phase 4-5 may be follow-up. |

---

## Open Questions

1. **Head communication:** Should heads see each other's outputs, or operate independently? Independence preserves the fiber bundle analogy but misses constraints (phrase structure limits thematic roles). Maybe: independent forward pass, then a lightweight cross-head attention layer.

2. **Training regime:** Joint training (all heads + body together) or staged (pretrain heads, then fine-tune body)? Staged is cleaner but may miss beneficial interactions.

3. **Head size:** How big do the heads need to be? They're doing syntactic/semantic parsing, not full language modeling. Maybe tiny (1-2 layers of a small transformer or CNN)?

4. **Tokenization interaction:** Grammar tags assume word-level boundaries. BPE subword tokenization complicates this. Use word-level tokenization for Mamba? Or align tags to BPE pieces?

5. **The return trip:** Hydra handles words→worlds (comprehension). What about worlds→words (generation)? Does the body learn to produce appropriate tags during generation, effectively using the heads as a decoder?

---

## Implementation Checklist

### Phase 0: Curvature Mapping (IN PROGRESS — Apr 3, 2026)
- [x] Set up `/data/hydra/` on EC2 T4
- [x] Download register-specific corpora (AAVE-proxy/Reddit, legal/arXiv, literary/Gutenberg, control/WikiText)
- [ ] Fine-tune GPT-2 Small on all 4 corpora (~3 hours, running now)
- [ ] Extract per-token embedding drift → curvature map
- [ ] Validate: bimodal distribution? Curved tokens interpretable? Control drift minimal?
- [ ] Identify ~5,000 curved tokens for landmark embedding layer

### Phase 0.5: Data Preparation
- [ ] Verify benepar phrase structure parses exist from Grammar-Mamba
- [ ] Run AllenNLP SRL on WikiText-103 for PropBank labels
- [ ] Implement information structure heuristics (given/new from coreference)
- [ ] Build multi-stream interleaving script (combine all tag streams)

### Phase 1-5: Model Training
- [ ] Phase 1: single-head (phrase structure) training + evaluation
- [ ] Phase 2: two-head (+ thematic) training + fluency comparison
- [ ] Phase 3: three-head (+ info structure) training + evaluation
- [ ] Phase 4: blank head + manifold-aware embeddings experiments
- [ ] Phase 5: garden-path fiber collision analysis

### Paper & Theory
- [ ] Paper draft (building on manifold origami + Grammar-Mamba)
- [ ] Manifold topology analysis (curvature, singularities, invariants)
- [ ] Unified theory writeup connecting all projects
