# Society of Mind Training: Transferable Patterns and Analysis

*Comprehensive writeup of the SoM architecture, training methodology, data generation, and PAB stability analysis. Extracted for cross-project reuse.*

---

## 1. The Architecture: Specialist Ensemble + Orchestrator

### Core Pattern (Domain-Invariant)

Instead of one monolithic model that learns everything, decompose into:

1. **N specialist networks** — each sees a domain-relevant signal subset at high capacity, plus the full context at lower capacity
2. **One orchestrator** — sees all specialist outputs + position context, learns trust weights via softmax over specialists
3. **Three-stage training** — specialists independently → orchestrator on frozen specialists → joint fine-tuning

This is the Society of Mind (Minsky) pattern implemented as a differentiable ensemble with learned routing.

### Why This Works Better Than a Flat Model

A flat model with the same total parameters (5.1M) trained on the same data plateaus at ~42% with massive overfitting (63% train, 42% eval on 3.6K examples). The SoM achieves 40.1% with minimal overfitting on 111K examples — a genuinely generalizing signal.

The key difference: **inductive bias through decomposition**. Each specialist is forced to learn its domain because it only sees domain-relevant inputs at high capacity. The full context path (lower capacity) prevents catastrophic blindness but doesn't dominate. The orchestrator learns the META-question: "what kind of problem is this, and which specialist should I trust?"

### Concrete Architecture (Chess Instance)

```
6 Specialist Agents (~845K params each):
    domain_encoder: Linear(domain_dims → 512) → LN → GELU → Linear(512 → 512) → LN → GELU → Linear(512 → 256) → LN
    context_encoder: Linear(58 → 256) → LN → GELU
    candidate_scorer: Linear(50 + 256 + 256 → 512) → LN → GELU → Linear(512 → 256) → GELU → Linear(256 → 1)
    position_head: Linear(256 → 32) → GELU → Linear(32 → 1) → Tanh  [auxiliary]

1 Orchestrator (~72K params):
    context_encoder: Linear(58 → 256) → LN → GELU → Linear(256 → 128) → GELU
    trust_head: Linear(128 + 6 → 128) → GELU → Linear(128 → 6)  [softmax → trust weights]
    override_head: Linear(128 → 6) → Sigmoid  [for extreme positions]
    outcome_head: Linear(128 → 32) → GELU → Linear(32 → 1) → Tanh  [auxiliary]
```

### Transferable Design Decisions

**Decision 1: Shared candidate encoder weights.** All candidates use the same scorer weights within each agent. The agent learns "what makes a candidate good in my domain" in general, not "slot 1 is special." This is critical for variable-length candidate sets.

**Decision 2: Domain subset + full context dual-path.** Each specialist sees:
- Its domain signals through a HIGH-capacity encoder (this is what makes it a specialist)
- ALL signals through a LOW-capacity encoder (prevents it from being blind to context)

The asymmetric capacity is the key — not binary domain gating.

**Decision 3: Override mechanism.** The orchestrator has an override head (sigmoid per specialist). When override > 0.8 for any specialist, that specialist's trust weight gets boosted by +5 in logit space before softmax. This handles extreme positions where one specialist should dominate (e.g., checkmate threats → tactical agent takes over).

**Decision 4: Trust as softmax over logits, not learned weights.** The orchestrator outputs trust LOGITS that go through softmax. This means trust weights sum to 1.0 and naturally compete. Adding the override as a logit boost (not weight multiplication) preserves gradient flow.

---

## 2. Training Methodology: Three-Stage with PAB Stability

### The Three Stages

**Stage 1: Train specialists independently**
- All N specialists train on the SAME dataset with DIFFERENT per-specialist labels
- Loss: MSE(specialist_prediction, specialist_label) + 0.05 × MSE(position_assessment, game_outcome)
- The auxiliary position assessment loss provides a weak supervisory signal that grounds each specialist's internal representation
- Optimizer: AdamW with OneCycleLR warmup
- Stops when PAB declares stable OR no eval improvement for 40 checkpoints

**Stage 2: Freeze specialists, train orchestrator**
- Specialists are frozen (no gradient flow to their weights)
- Orchestrator input: position signals + all specialist scores (pre-computed, cached)
- Loss: CrossEntropy(orchestrator_selection, ground_truth_best) + 0.1 × MSE(outcome_pred, game_outcome)
- The orchestrator learns "given these specialist opinions, which one is right?"
- Tends to converge fast (specialists are fixed, search space is small)

**Stage 3: Joint fine-tuning**
- Unfreeze everything. Train end-to-end at 10x lower LR
- Loss: CrossEntropy(final_selection, ground_truth_best)
- This is where compositionality emerges — specialists adapt to the orchestrator's arbitration
- Typically the longest stage and where the biggest eval gains happen

### Why Three Stages Instead of End-to-End

End-to-end training from scratch fails because:
1. The orchestrator has no useful specialist scores to learn from (all random at start)
2. Specialists have no gradient signal from the orchestrator (it doesn't know what to ask for)
3. The system collapses to one dominant specialist (winner-take-all dynamics)

Sequential training solves this:
1. Stage 1 gives each specialist a reasonable domain signal
2. Stage 2 gives the orchestrator reasonable specialist scores to learn from
3. Stage 3 lets them co-adapt from a good initialization

This is **curriculum learning at the architecture level** — not curriculum over data difficulty, but curriculum over which components are trainable.

### Observed Training Dynamics

```
Stage 1 (agents alone):        30.9% eval accuracy
Stage 2 (+ orchestrator):      29.9% eval accuracy  ← slight DROP (orchestrator learning)
Stage 3 epoch 1:               30.3%
Stage 3 epoch 5:               37.2%  ← compositionality kicks in
Stage 3 epoch 17:              39.7%
Stage 3 epoch 25 (best):       40.1%  ← PAB plateau
Stage 3 epoch 40 (PAB stop):   39.6%  ← slight decay, PAB stops
```

The jump from 30% to 37% in early Stage 3 is the compositionality signal — agents + orchestrator learning together produces more than either alone. This is the OTP/COEC prediction: compositional amplification beyond additive.

---

## 3. PAB: Process-Aware Benchmarking for Training Control

### What PAB Is

PAB (from the Balanced Sashimi research) replaces fixed iteration counts with trajectory-based stopping. Instead of "train for 10,000 steps," PAB says "train until the learning trajectory stabilizes."

### The Stability Metric

```python
S(t) = |L(t) - L(t-1)| / (|L(t-1)| + eps)
```

This is the relative change in loss between consecutive evaluation points. When `mean(S)` over a sliding window drops below a threshold, the training is stable — the model has learned what it can learn from this data at this capacity.

### PAB Parameters

| Parameter | What It Controls | Our Values |
|-----------|-----------------|------------|
| `window` | Size of the sliding stability window | 20 checkpoints |
| `threshold` | Stability score below which = converged | 0.015 (S1), 0.01 (S2), 0.008 (S3) |
| `patience` | How many consecutive stable windows before stopping | 5 |
| `max_no_improve` | Stop if eval accuracy hasn't improved in N checkpoints | 40 |

### Why Different Thresholds Per Stage

- **Stage 1 (0.015):** Specialists have noisy domain-specific labels. Higher threshold = more tolerant of oscillation
- **Stage 2 (0.01):** Orchestrator has cleaner labels (cross-entropy on best candidate). Medium threshold
- **Stage 3 (0.008):** Joint fine-tuning should be smooth (already well-initialized). Tightest threshold

### PAB as a Transferable Pattern

PAB applies to ANY multi-stage training process:
1. Define a stability metric per stage
2. Set stage-appropriate thresholds (noisier stages get looser thresholds)
3. Let each stage run until stable, not for a fixed number of steps
4. Track `no_improve` as a safety valve (stops even if stability score is still high)

**Key insight:** The `max_no_improve` safety valve caught our Stage 1 correctly — stability never reached 0.015 (the loss was inherently noisy from the MSE labels), but eval accuracy stopped improving after 40 checkpoints. The safety valve stopped training at the right time even though the primary stability metric hadn't triggered.

---

## 4. Data Generation: The Lichess Pipeline

### The Breakthrough: Pre-Computed Positions

Generating training positions from scratch (SF self-play) was ~12 examples/minute. Switching to the Lichess 342M position eval dataset (pre-computed positions + evals) gave ~90 examples/minute — **7.5x speedup**. The bottleneck shifted from "find interesting positions" to "label candidates with SF."

### Pipeline Architecture

```
Lichess 342M positions (streaming, pre-filtered by eval)
    ↓ Sample 2% of unique FENs
    ↓ Filter: |eval| < 500cp, >= 3 legal moves
    ↓ Run Yami infrastructure → 5 candidates with signal profiles (no SF needed)
    ↓ SF depth 15 on each candidate → label best (5x SF calls per position)
    ↓ PV walk on best candidate → 15-ply signal trajectory
    ↓ Per-agent dimensional delta decomposition
    ↓ Crash-safe checkpoint every 100 examples → JSONL
```

### Per-Agent Labels: Dimensional Delta Decomposition

Stockfish only gives aggregate centipawns — no per-dimension breakdown. Our solution:

1. Compute `PositionSignals` (58 dims) BEFORE the move
2. Push each candidate, compute signals AFTER, compute delta
3. The delta naturally decomposes by agent domain (material dims → tactical agent, pawn dims → positional agent, etc.)
4. Contrastive label: `relevance = L2(agent_dims_delta) / L2(all_dims_delta) × eval_gap / 100`

Each agent gets a continuous relevance score: "how much did YOUR dimensions explain the difference between the best and second-best move?"

### PV Theme Classification

Stockfish's 15-ply principal variation is walked move-by-move, extracting signal trajectories at each ply. The PV's "theme" is classified by which signal dimensions change most over the trajectory:

| Theme | Signal Dimensions That Change | Example |
|-------|------------------------------|---------|
| Tactical | Material balance | PV shows piece capture at ply 5 |
| Positional | Pawn structure, piece activity | PV shows slow structural improvement |
| Endgame | Game phase, passed pawns | PV transitions to endgame |
| Attack | Opponent king safety | PV shows attacker count increasing |
| Defense | Own king safety | PV shows shelter improving |
| Initiative | Mobility, development | PV shows mobility growing |

### Transferable Data Generation Patterns

**Pattern: Crash-safe incremental writes.** Every 100 examples, flush to disk as JSONL append. Loss of power loses at most 99 examples. Multiple machines can generate to separate files and merge later.

**Pattern: Sample rate as a knob.** With 342M source positions, even 2% sampling gives 6.8M candidates — way more than needed. The sample rate controls generation speed vs diversity.

**Pattern: Candidate spread filter.** Only keep positions where the eval spread across candidates exceeds a threshold (10-50cp). Positions where all candidates score similarly are noise — the ranking decision doesn't matter.

**Pattern: Multi-machine generation.** The pipeline is stateless per position. Run on multiple machines with different sample rates (to reduce overlap), merge JSONL files when done. We ran two machines simultaneously producing ~155 examples/min combined.

---

## 5. Signal Architecture: Multi-Scale Lens Decomposition

### Position Signals (58 dimensions)

| Category | Dims | Description |
|----------|------|-------------|
| Material | 5 | Balance, total, imbalance, bishop pair (ours/theirs) |
| Pawn Structure | 12 | Islands, isolated, doubled, backward, passed (ours/theirs) |
| King Safety | 8 | Pawn shelter, open files, attacker count, castled (ours/theirs) |
| Piece Activity | 8 | Mobility, development, good bishop, connected rooks (ours/theirs) |
| Phase/Tempo | 6 | Game phase, move number, center control, space, in check |
| Castling Rights | 4 | K/Q/k/q |
| Navigator | 6 | 6-bank ternary vector (aggression, piece domain, complexity, initiative, king pressure, phase) |
| SoM Context | 2 | Convergence, trajectory trend |
| Strategy | 1 | Match strength |
| Opponent Profile | 5 | Tactical skill, aggressiveness, consistency, pressure, risk tolerance |
| Negative Learning | 1 | Censor rejection rate |

### Per-Candidate Signals (50 dimensions per move)

| Category | Dims | Description |
|----------|------|-------------|
| Navigator OTP | 2 | Score + ternary |
| SoM Agents | 7 | 6 agent scores + convergence |
| Strategy | 2 | Alignment + ternary |
| GM Patterns | 3 | Frequency, win rate, ternary |
| K-Line Memory | 2 | Match score + ternary |
| Look-ahead | 1 | 2-ply blunder detection |
| Censor | 1 | Pass/fail |
| Interference | 1 | OTP constructive/destructive |
| Tactical Motifs | 17 | One-hot: capture, check, fork, pin, etc. |
| Move Geometry | 8 | Piece type, capture, check, castling, SEE, centrality, king distances |
| Post-Move | 6 | Material change, mobility, structure change, outpost, rook file, king tropism |

### The Infrastructure-First Thesis

The model ONLY sees these 58 + 50×5 = 308 signal dimensions. It never sees the raw 64-square board. This means:

1. **Model capacity is bounded by signal quality.** More params can't extract information that isn't in the signals. Our 40% ceiling is a signal quality ceiling, not a model capacity ceiling (proven by testing 12K through 5.1M params — same ceiling).

2. **Improving signals improves everything.** Adding better position signals (deeper look-ahead, more chess knowledge) directly raises the ceiling for any model trained on them.

3. **The infrastructure IS the understanding.** The signals encode human chess knowledge (pawn structure, king safety, piece activity). The model just learns which combinations matter.

---

## 6. Progressive Revelation: Scale-by-Scale Resolution

### The Pipeline

```
Scale 0: Legal + Censors → eliminate impossible moves
Scale 1: Tactical → checkmate/forced wins short-circuit
Scale 2: Positional → navigator OTP narrows candidates
Scale 3: Strategic → OTP interference + COEC amplification
Scale 4: Temporal/SoM → specialist agent debate
Scale 5: Neural → learned fusion model on HARD cases only
```

### Key Properties

**Residual passing:** Each scale receives only the candidates that the previous scale couldn't resolve. Scale 5 sees 1-3 candidates, not 30.

**Ambiguity measurement:** Each scale computes its own confidence. If confident → short-circuit. If ambiguous → pass residual up.

**COEC compositional amplification (Scale 3):** When specific signal pairs both support a move, the combined weight exceeds their sum. Navigator + Strategy agreement → 1.4x multiplier. This is domain-specific — the amplifying pairs must be discovered empirically.

### Transferable Pattern

Any decision pipeline can be restructured as progressive revelation:
1. Order your signal sources from cheapest/most certain to most expensive/uncertain
2. At each scale, measure ambiguity (do signals agree?)
3. Short-circuit when ambiguity drops below threshold
4. Only send AMBIGUOUS cases to expensive computation
5. The expensive computation (neural model) gets a SMALLER, HARDER problem

---

## 7. Key Numbers

| Metric | Value |
|--------|-------|
| Total model parameters | 5,145,305 |
| Per-agent parameters | ~845K |
| Orchestrator parameters | 71,501 |
| Training examples | 111,800 (from 3 sources) |
| Position signal dimensions | 58 |
| Per-candidate signal dimensions | 50 |
| Max candidates per position | 5 |
| Total input dimensions | 308 (58 + 5×50) |
| Eval accuracy (agents alone) | 30.9% |
| Eval accuracy (SoM composed) | 40.1% |
| Compositionality gain | +9.2 percentage points (30% → 40%) |
| Random baseline | 20% (1 in 5 candidates) |
| Training time | 58 minutes (3 stages) |
| Data generation rate (Lichess) | ~90 examples/min per machine |
| Data generation rate (self-play) | ~12 examples/min per machine |
| SF labeling depth | 15 ply |
| PV trajectory depth | 13-14 ply average |
| PAB stability threshold (S3) | 0.008 |
| Stage 3 convergence epoch | ~25 (of 40 max) |

---

## 8. What Worked and What Didn't

### Worked

- **Three-stage training** — essential. End-to-end from scratch fails.
- **Joint fine-tuning (Stage 3)** — this is where compositionality emerges. +9% over agents alone.
- **PAB stability detection** — correctly stopped training at the right time. `max_no_improve` safety valve caught cases where the stability metric was too strict.
- **Lichess dataset for position sourcing** — 7.5x speedup over self-play generation.
- **Crash-safe incremental writes** — saved us from battery deaths and kills.
- **Multi-machine data generation** — trivially parallelizable with JSONL merge.
- **Dual-path specialist architecture** — domain subset (high capacity) + full context (low capacity).

### Didn't Work

- **Per-agent MSE labels (dimensional delta decomposition)** — too sparse. Most agents get ~0.0 relevance for most positions. Cross-entropy on best-candidate-idx would be a denser training signal.
- **Fixed iteration training** — always too short or too long. PAB was strictly better.
- **12K parameter model** — too small to learn anything beyond trivial correlations. But 3.3M was equally bad (same 42% ceiling) — the ceiling was in signal quality, not model capacity.
- **Hand-tuned coherence weights** — fundamentally broken. The learned fusion was always the right approach.
- **2-ply look-ahead** — too shallow. The gap between 2-ply inference signals and 15-ply training labels is where ~60% of the missing accuracy lives.

### Open Questions

- **Would cross-entropy per-agent labels work better than MSE?** Each agent predicts best candidate from its domain signals → ClassificationLoss. Every example gives every agent a training signal.
- **Would 4-6 ply look-ahead at inference close the gap?** Currently 2-ply. Iterative deepening at Scales 3-4 would be expensive but might dramatically improve signal quality.
- **Can we learn COEC pairs from data?** Currently hand-specified. Mining co-occurrence patterns in training data could discover new amplifying pairs.
- **Would adding raw board features (12×64 piece placement) break the ceiling?** The model currently sees only derived signals. Adding the geometric board state as an additional lens might capture patterns that derived signals miss.

---

## 9. Cross-Project Applicability

### The Pattern in Abstract

```
PROBLEM: Complex decision with multiple information sources
SOLUTION:
    1. Decompose information into orthogonal signal banks (domain decomposition)
    2. Build specialist networks per bank (each sees its domain at high capacity)
    3. Build orchestrator (learns trust weights over specialists)
    4. Train: specialists → orchestrator → joint (three-stage curriculum)
    5. Monitor with PAB (train until trajectory stabilizes, not for fixed steps)
    6. Deploy as progressive pipeline (cheap signals resolve easy cases,
       expensive signals handle only the hard residual)
```

### Where This Applies

| Domain | Specialists | Orchestrator Learns |
|--------|-----------|-------------------|
| **Chess** | Tactical, Positional, Endgame, Attack, Defense, Initiative | Which specialist to trust for this position type |
| **Legal retrieval** | Statute, Precedent, Procedural, Constitutional, Regulatory | Which legal domain governs this query |
| **Medical triage** | Vital signs, Chief complaint, History, Labs, Physical exam | Which signal source is most diagnostic |
| **Code quality** | Lint, Type safety, Test coverage, Complexity, Security | Which quality dimension matters most here |
| **Content moderation** | Toxicity, Misinformation, Copyright, Privacy, Violence | Which harm axis is relevant |
| **Portfolio optimization** | Value, Momentum, Quality, Volatility, Sentiment | Which factor dominates this market regime |

The architecture is domain-invariant. Only the signal definitions and specialist domain assignments change.

---

*Analysis v1.0 — March 2026. Yami project, Society of Mind training on 111K examples from Lichess 342M position dataset. 5.1M parameters, 40.1% eval accuracy, PAB-stable at epoch 40.*
