# Learned Signal Fusion: Architecture Design

*Replacing hand-tuned coherence weights with a learned fusion model.*

## The Problem

The coherence engine computes 6+ independent signals per candidate move, then combines them with hand-tuned weights:

```python
# OLD: hand-tuned, broken
final_score = (
    navigator_score * 2.0
    + strategy_score * 2.5
    + temporal_scores * 0.3
    + kline_score * 1.5
    + gm_frequency * 4.0
    + interference * 3.0
    + convergence * 2.0
    + look_ahead * weight
)
```

This was the source of Yami's terrible move selection. The weights were wrong, the priorities were wrong, and the system had no way to learn from its mistakes.

## The Solution

**Kill the hand-tuned weights. Let a small neural model learn the fusion function from data.**

```
OLD:  signals --> hand-tuned fusion --> candidates --> neural re-rank
NEW:  signals --> neural model (learned fusion) --> move selection
```

The infrastructure becomes a **pure feature extractor**. It computes all signals deterministically. The neural model takes the full signal vector for each candidate and predicts which candidate is best. The model learns the weighting function that we were failing to hand-tune.

## Signal Inventory

### Per-Position Signals (computed once per board state)

| Signal | Source | Type | Dims | Description |
|--------|--------|------|------|-------------|
| Navigation vector | `navigator.py` | Ternary | 6 | AGG, PIECE, CMPLX, INIT, KPRES, PHASE |
| Game phase | Board analysis | Float | 1 | 1.0=opening, 0.0=endgame (piece count ratio) |
| Material balance | Board analysis | Float | 1 | (our - opponent) / queen_value |
| Total material | Board analysis | Float | 1 | Total pieces on board, normalized |
| Move number | Board state | Float | 1 | Normalized (move/100) |
| Positional profile | `knowledge_graph.py` | Categorical | 6 | material, structure, activity, safety, opp_safety, tempo |
| Plan type | `knowledge_graph.py` | Categorical | 1 | 7 plan types (one-hot → 7 dims) |
| Plan activation | `knowledge_graph.py` | Float | 1 | How strongly the plan applies |
| Active strategy name | `strategy_library.py` | Categorical | 1 | Which of 20 strategies matches |
| SoM convergence | `temporal_controller.py` | Float | 1 | Agent agreement score |
| SoM trajectory trend | `temporal_controller.py` | Float | 1 | Is the plan converging or diverging? |
| Opponent profile | `opponent_profile.py` | Float | 4 | tactical_skill, aggressiveness, consistency, pressure |
| Opponent risk tolerance | `opponent_profile.py` | Float | 1 | Derived from behavioral axes |
| King safety (own) | Board analysis | Float | 1 | Pawn shield + open file exposure |
| King safety (opponent) | Board analysis | Float | 1 | Same for opponent |
| Pawn structure score | Board analysis | Float | 1 | Isolated + doubled + backward penalty |
| Center control | Board analysis | Float | 1 | Pieces/pawns attacking e4/d4/e5/d5 |
| Development count | Board analysis | Float | 1 | Minor pieces off back rank |
| Castling rights | Board state | Binary | 4 | K, Q, k, q |
| In check | Board state | Binary | 1 | Is the side to move in check? |

**Total per-position: ~35 dimensions**

### Per-Candidate Signals (computed for each of 3-5 candidate moves)

| Signal | Source | Type | Dims | Description |
|--------|--------|------|------|-------------|
| Navigator OTP score | `navigator.py` | Float | 1 | How well move aligns with nav vector |
| Navigator ternary | `navigator.py` | Ternary | 1 | {-1, 0, +1} from OTP threshold |
| Tactical motifs | `tactical_scoper.py` | Binary | 9 | capture, check, fork, pin, etc. |
| Strategy alignment | `strategy_library.py` | Float | 1 | Score against active strategy |
| Strategy ternary | Coherence | Ternary | 1 | {-1, 0, +1} |
| SoM agent scores | `temporal_controller.py` | Float | 6 | One per specialist agent |
| SoM ternary | Coherence | Ternary | 1 | {-1, 0, +1} |
| K-line match | `kline_memory.py` | Float | 1 | Match score from pattern DB |
| K-line ternary | Coherence | Ternary | 1 | {-1, 0, +1} |
| GM pattern frequency | `gm_patterns.py` | Float | 1 | How often GMs play this move |
| GM win rate | `gm_patterns.py` | Float | 1 | Win rate when this move is played |
| GM ternary | Coherence | Ternary | 1 | {-1, 0, +1} |
| Look-ahead score | `coherence.py` | Float | 1 | 2-ply blunder detection |
| Censor pass | `negative_learning.py` | Binary | 1 | Did the censor stack approve? |
| Interference | Coherence | Float | 1 | Constructive/destructive pattern |
| Piece type | Move analysis | One-hot | 6 | P/N/B/R/Q/K |
| Is capture | Move analysis | Binary | 1 | Does this move capture? |
| Is check | Move analysis | Binary | 1 | Does this move give check? |
| Is castling | Move analysis | Binary | 1 | Is this castling? |
| SEE value | `tactical_scoper.py` | Float | 1 | Static exchange evaluation |
| Target centrality | Move geometry | Float | 1 | How central is the target square |
| Distance to opp king | Move geometry | Float | 1 | Manhattan distance |
| Distance to own king | Move geometry | Float | 1 | Manhattan distance |
| Resulting mobility | Post-move | Float | 1 | Opponent legal moves after |
| Pawn structure change | Post-move | Float | 1 | Does this improve/worsen structure? |
| Material change | Post-move | Float | 1 | Net material gain/loss |
| Positional eval | `candidate_filter.py` | Float | 1 | Material diff after move |
| Anchor flags | `navigator.py` | Binary | ~15 | development, center-control, etc. |

**Total per-candidate: ~55 dimensions**

### Full Feature Vector

For a position with 5 candidates:
- Position features: 35 dims
- Candidate features: 5 x 55 dims = 275 dims
- **Total: ~310 dimensions per training example**

## New Signals to Add

From Stockfish classical evaluation (pre-NNUE `evaluate.cpp`), CPW, and Kaufman's research:

### Tier 1: High Impact, Easy to Compute (Static)

| Signal | Type | Dims | Scope | Description |
|--------|------|------|-------|-------------|
| Pawn islands | Int | 2 | Per-pos | Disconnected pawn groups (ours, theirs) |
| Passed pawn count | Int | 2 | Per-pos | Our passed pawns, opponent's |
| Passed pawn max rank | Float | 2 | Per-pos | How far advanced (normalized 0-1) |
| Isolated pawn count | Int | 2 | Per-pos | Isolated pawns (ours, theirs) |
| Doubled pawn count | Int | 2 | Per-pos | Doubled pawns (ours, theirs) |
| Backward pawn count | Int | 2 | Per-pos | Backward pawns (ours, theirs) |
| Bishop pair | Binary | 2 | Per-pos | Do we/they have both bishops? |
| Good/bad bishop | Float | 2 | Per-pos | How many own pawns on bishop's color |
| Connected rooks | Binary | 2 | Per-pos | Are our/their rooks connected? |
| Development count | Int | 2 | Per-pos | Minor pieces off back rank (ours, theirs) |

### Tier 2: High Impact, Medium Complexity

| Signal | Type | Dims | Scope | Description |
|--------|------|------|-------|-------------|
| King pawn shelter | Float | 2 | Per-pos | Pawn shield quality (ours, theirs) |
| Pawn storm | Float | 2 | Per-pos | Enemy pawns advancing on king |
| Attacker count (king zone) | Int | 2 | Per-pos | Pieces attacking squares near king |
| Space advantage | Float | 1 | Per-pos | Safe squares behind pawn chain |
| Piece connectivity | Float | 2 | Per-pos | How many own pieces defend each other |
| Material imbalance | Float | 1 | Per-pos | Kaufman-style pair interactions |
| Outpost (per candidate) | Binary | 1 | Per-cand | Is target an outpost? |
| King tropism (per candidate) | Float | 1 | Per-cand | Piece movement toward opp king |
| Rook on open file (per cand) | Binary | 1 | Per-cand | Does move put rook on open file? |
| Rook on 7th (per candidate) | Binary | 1 | Per-cand | Does move put rook on 7th? |
| Trapped piece | Binary | 1 | Per-cand | Does this move trap a piece? |

### Tier 3: Framework-Level

| Signal | Type | Dims | Scope | Description |
|--------|------|------|-------|-------------|
| Tapered eval phase | Float | 1 | Per-pos | 0=endgame, 1=middlegame (from material) |
| MG/EG split | - | - | Framework | Every positional signal computed twice |
| Initiative factor | Float | 1 | Per-pos | SF-style: pulls eval toward 0 in quiet pos |
| Draw probability | Float | 1 | Per-pos | Opposite-color bishops, fortress, etc. |

### Signal Count After Expansion

```
Current per-position:   ~35 dims
New per-position:       +25 dims (Tier 1 + Tier 2)
Total per-position:     ~60 dims

Current per-candidate:  ~55 dims
New per-candidate:      +5 dims (Tier 2 per-cand)
Total per-candidate:    ~60 dims

Full vector (5 cands):  60 + 5*60 = 360 dims
```

## Training Data Sources

### Source A: SF vs SF at Different ELOs (The Mistake Curriculum)

The key insight: when a weaker engine makes a mistake, the signal profile at that moment reveals which signal combinations are *deceptive*. The model learns "these signals look good but the move is bad."

```
SF(ELO 1500) vs SF(ELO 2500)
SF(ELO 1800) vs SF(ELO 2800)
SF(ELO 2000) vs SF(ELO 3190)

    For each position at the WEAKER engine's turn:
        1. Run Yami infrastructure → full signal profile for all candidates
        2. Record what the weak SF chose
        3. Label with strong SF (depth 20): which candidate is actually best
        4. The DELTA between "what looked good" and "what IS good" is the training signal

    Positions where weak SF and strong SF agree → boring (easy position)
    Positions where they DISAGREE → gold (signal profile is misleading)
```

This creates a natural curriculum: easy positions (all ELOs agree) → medium (1500 wrong, 2000 right) → hard (even 2500 gets it wrong).

### Source B: Expert/GM Games (PGN Import)

Real positions from strong human play. Lichess open database has millions of games rated 2000+.

```
Load GM game PGN
    For each position (sample every 3-5 moves to avoid redundancy):
        1. Run Yami infrastructure → full signal profile
        2. The GM's actual move = primary label
        3. SF depth 20 eval for each candidate = secondary label
        4. If GM move is NOT in Yami's candidate pool → flag as "infrastructure gap"
        5. If GM move IS in pool but not ranked first → "ranking error" (most valuable)
        6. If GM move != SF best move → "human intuition vs engine truth" (edge case)
```

The GM games also generate the BANK SIGNAL PROFILE training data: at each position, we record the full navigator vector, SoM agent scores, strategy match, etc. The model learns which bank profiles correlate with winning chess.

### Source C: Critical Position Data (Currently Generating)

```
SF self-play at moderate depth → extract positions with eval swings >100cp
    - These are positions where ONE specific move changes the evaluation dramatically
    - Full signal profile + SF depth 15 label
    - Currently generating: 1000 critical + 1000 adversarial + 500 endgame
```

### Source D: Candidate Spread Filter (Chess.com Insight)

```
Take any position → compute candidates → get SF eval for each
    - Candidate spread = max_eval - min_eval across candidates
    - KEEP positions where spread > 150cp (the decision matters)
    - DISCARD positions where spread < 30cp (any candidate is fine)

    This is the Chess.com insight: not all positions are equally
    useful for training. Positions where the ranking is decisive
    teach the model to distinguish. Positions where everything is
    equal are noise.
```

### Source E: Bank Profile Optimization (Your Idea)

```
For EVERY training position (from all sources):
    1. Compute the full multi-bank signal profile:
       - Navigator 6-bank vector
       - SoM 6-agent scores + convergence
       - Tactical motif vector
       - Positional signals (pawn structure, king safety, etc.)
       - Strategy library match
       - GM pattern frequency

    2. Label with the GAME RESULT (not just the immediate eval):
       - Did the side that played from this position eventually win?
       - What was the eval trajectory (improving/declining/stable)?

    3. Train the model to predict:
       GIVEN: bank signal profile + candidate signal profiles
       PREDICT: which candidate leads to winning

    The model optimizes for the BANK SIGNAL PROFILE that correlates
    with winning chess. It learns: "when navigator reads AGG=+1 and
    SoM tactical agent is high and there's a fork motif, trust the
    aggressive candidate."
```

This is the key innovation: the model doesn't just learn position → move. It learns **signal profile → outcome**. The signals ARE the representation. The model learns which lenses to trust.

## Model Architecture

The model replaces the coherence engine's hand-tuned weights. It's a **Society of Mind arbitrator**: each signal source is a specialist that votes, and the model learns when to trust each specialist.

```
                    Position Signals (60 dims)
                           |
                  [Position Encoder] (60 → 64, LayerNorm)
                           |
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    Candidate 1 (60)  Candidate 2 (60) ... Candidate 5 (60)
         │                 │                 │
    [Shared Cand          [Shared Cand       [Shared Cand
     Encoder]              Encoder]           Encoder]
    (60 → 32)             (60 → 32)          (60 → 32)
         │                 │                 │
         └────────┬────────┘                 │
                  │                          │
         [Cross-Attention: position context attends to all candidates]
         (64 + 5×32 → 5×32)
                  │
         [Candidate Scorer] (32 → 1) × 5
                  │
              softmax → selection
```

**Key design choices:**

1. **Shared candidate encoder** — all candidates use the same weights. The model learns "what makes a candidate good" in general, not "candidate slot 1 is special."

2. **Cross-attention** — the position context modulates candidate scoring. "This is an attacking position" changes which candidate signals matter.

3. **No raw board** — the model only sees signals from the infrastructure. This enforces the infrastructure-first thesis. Improving signals automatically improves the model.

4. **SoM is native** — the 6 SoM agent scores are direct inputs per candidate. The model learns: "when tactical agent and attack agent agree but positional agent disagrees, trust the tactical/attack coalition." This is Society of Mind arbitration — specialists debate, the model resolves.

### Parameter Budget

```
Position encoder:    60 × 64 + 64     =  3,904
Candidate encoder:   60 × 32 + 32     =  1,952  (shared, not ×5)
Cross-attention:     (64+32) × 32 × 3 =  9,216  (Q, K, V)
Candidate scorer:    32 × 1 + 1       =     33  (shared)
                                     ≈  15,000 parameters
```

~15K parameters. Down from 294K. The infrastructure does the heavy lifting.

### Training Objective

```
L = L_candidate + 0.1 * L_bank_profile + 0.1 * L_signal_attribution

L_candidate:          Cross-entropy on candidate selection (primary)
L_bank_profile:       MSE on predicting game outcome from position signals
                      (teaches the model which signal profiles win)
L_signal_attribution: Which signal source was most important for this decision
                      (auxiliary, interpretability — the model explains itself)
```

The bank profile loss is the key innovation: it ensures the model doesn't just memorize position → answer, but learns the **structural relationship between signals and outcomes**. This is what makes the model generalize — it learns "attacking signal profiles lead to wins" rather than "this specific position → this specific move."

## Wayfinder-Derived Enhancements (v2.0)

Three architectural enhancements derived from the Wayfinder proof-solving project's findings. Wayfinder discovered that formal theorem proving has exploitable bimodal structure: a large cheap majority solvable by infrastructure, and a small structured tail requiring reasoning. The same decomposition applies to chess move selection.

### Enhancement 1: Agent Suppression (Dr. Ducky Bank Filtering)

**Origin:** Dr. Ducky's symbolic proof VM filters irrelevant symbolic banks before execution. "No existential structure → don't run the witness engine." This reduces noise and computation.

**Chess application:** Position-structure-gated SoM agent activation. Before running the 6 SoM agents, read the position signals and suppress agents whose domain dimensions are structurally inert.

```
Opening position (game_phase > 0.5):
  → suppress ENDGAME agent (nothing to say)

No king attack surface (king_safety_theirs low, no aggression):
  → suppress ATTACK agent (noise only)

Our king is safe (king_safety_ours low, no pressure):
  → suppress DEFENSE agent (noise only)
```

**Implementation:** `src/yami/decision_capsule.py::compute_active_agents()`. Suppressed agents get 0.0 scores (they don't contaminate the convergence signal). Convergence is computed only over active agents — a 3-agent consensus on a quiet endgame is more meaningful than 6-agent noise.

**Measured effect:** In the opening position, 3 of 6 agents are correctly suppressed (endgame, attack, defense). The remaining 3 (tactical, positional, initiative) produce cleaner convergence signals.

### Enhancement 2: Decision Residual Packets (Chess.com Move Quality)

**Origin:** When Wayfinder fails, it emits a typed residual packet: goal family, lane trace, candidate banks, pathology. The packet IS the training surface for the next layer. Most positions don't need the expensive layer at all.

**Chess application:** Classify each decision as **infrastructure-resolved** vs **genuine-residual**. When coherence clearly picks a winner (high gap, strong signal agreement), skip the neural model entirely. When it doesn't, emit a typed `DecisionCapsule` that describes exactly what's ambiguous.

**Move quality classification** uses Chess.com-style centipawn-loss buckets for training labels:

| Quality | CP Loss | Score | Description |
|---------|---------|-------|-------------|
| Brilliant | 0 + sacrifice | 1.0 | Best move involving material sacrifice |
| Great | 0 (non-obvious) | 0.9 | Best move, not forced/obvious |
| Best | 0 | 0.85 | Engine's top choice |
| Good | ≤ 30 | 0.7 | Slight imprecision |
| Inaccuracy | 31-100 | 0.4 | Meaningful error |
| Mistake | 101-300 | 0.15 | Serious error |
| Blunder | > 300 | 0.0 | Game-losing |

**Implementation:** `src/yami/decision_capsule.py::classify_decision()`. Infrastructure-resolved positions short-circuit at Scale 3 in the progressive engine. The neural model only sees genuine residuals — positions where signals genuinely disagree.

**Training data implication:** The model should be trained primarily on genuine residuals, not easy positions. Easy positions are noise in the training set. This is the bimodal split: ~70%+ of positions are infrastructure-solvable, the neural model should focus its capacity on the hard 30%.

### Enhancement 3: Decision Capsule (Projector Boundary)

**Origin:** Dr. Ducky's projector boundary: internal symbolic state never crosses into the neural path. The only output is a typed, verified program. This makes every step auditable.

**Chess application:** The `DecisionCapsule` is the ONLY thing the neural model sees. It contains:

```
DecisionCapsule:
  position context:     game_phase, material_balance, 6-bank nav vector
  agent activation:     which agents are active (after suppression)
  ambiguity descriptor: what dimension creates the tie (e.g., "navigator_vs_strategy")
  swing agent:          which agent's opinion would break the tie
  difficulty estimate:  0.0 (trivial) to 1.0 (maximally ambiguous)
  candidates:           list[CandidateCapsule] with per-agent scores

CandidateCapsule:
  coherence signals:    final_score, interference, 5 ternary values
  agent scores:         only from active agents (suppressed = 0.0)
  tactical context:     capture, check, sacrifice, SEE value, motifs
  geometry:             centrality, king distance
  quality label:        Chess.com classification (for training)
```

**Implementation:** `src/yami/decision_capsule.py::build_decision_capsule()`. Built at Scale 3 of the progressive engine, consumed at Scale 5. The capsule forces the question: "what exactly are we asking the model to decide?"

**Vector dimensions:** Position context = 12 dims, per-candidate = 19 dims. For 5 candidates: 12 + 5×19 = 107 dimensions. Down from 310 in the raw signal fusion approach — because the capsule has already compressed the structural information.

### Source F: Residual-Only Training (New)

```
For EVERY training position:
    1. Run full infrastructure pipeline through Scale 3
    2. Compute decision type (infrastructure_resolved vs genuine_residual)
    3. IF infrastructure_resolved:
         → use as VALIDATION only (the model should agree with infrastructure)
         → do NOT train on these (they are the "easy majority")
    4. IF genuine_residual:
         → build DecisionCapsule
         → label each candidate with Chess.com quality (from SF eval)
         → THIS is the training example

    Expected split: ~60-70% infrastructure-resolved, ~30-40% genuine residual
    The model trains on 30-40% of positions but those are the ones that matter.
```

This is the Wayfinder insight applied to training data: the infrastructure compiles away the structural majority. The model learns the residual.

## Implementation Plan

### Phase 1: Signal Expansion (Complete)
1. ~~Build `src/yami/position_signals.py`~~ — 58-dim per-position signals
2. ~~Build `src/yami/candidate_signals.py`~~ — 50-dim per-candidate signals
3. ~~Update signal profile contract~~

### Phase 2: Wayfinder Enhancements (Complete)
4. Build `src/yami/decision_capsule.py` — capsule, agent suppression, move quality
5. Integrate agent suppression into Scale 4 of progressive engine
6. Integrate decision classification + capsule emission into Scale 3
7. Add infrastructure-resolved short-circuit to progressive pipeline

### Phase 3: Data Pipeline
8. Build SF-vs-SF multi-ELO data generator (Source A)
9. Build PGN import pipeline for GM games (Source B)
10. Add candidate spread filter (Source D)
11. **Add residual-only training filter (Source F) — label with Chess.com quality**
12. Regenerate training data with full capsule profiles

### Phase 4: Model
13. Build the fusion model (now operating on capsule vectors, not raw signals)
14. Train on genuine-residual data only (Sources A-F, filtered)
15. Integrate into Scale 5 — model receives DecisionCapsule, not raw features

### Phase 5: Benchmark
16. Compare against old system at all ELO levels
17. Measure infrastructure-resolved ratio across game phases
18. Measure agent suppression effect on convergence quality
19. Ablation: capsule vs raw signals, residual-only vs all-positions training
20. Compare against pure SF at various depths

## What This Preserves

- The infrastructure layers (legal moves, tactical scoping, censors, opening book, endgame tables)
- The signal sources (navigator, SoM, strategy library, GM patterns, K-lines)
- The OTP ternary representation (signals still use {-1, 0, +1})
- The censor stack (still rejects catastrophic moves before the model sees them)
- The progressive revelation architecture (Scales 0-5)

## What This Replaces

- The hand-tuned `final_score` computation in `coherence.py`
- The legacy `rank_moves` in `knowledge_graph.py` (already bypassed)
- The fixed signal weights (2.0, 2.5, 0.3, 1.5, 4.0, 3.0, 2.0)
- **All-SoM-agents-always-on** → position-gated agent suppression
- **Flat feature vector to neural** → typed DecisionCapsule projector boundary
- **Train on all positions equally** → residual-only training with Chess.com quality labels

## The Thesis

The infrastructure-first thesis says: most of chess is structural work that deterministic computation handles trivially. The remaining residual is a pattern recognition problem over the signal space. A tiny model can learn this residual because the infrastructure has already reduced the problem from "evaluate a 64-square board with 32 pieces" to "pick from 5 candidates given 55 signals each."

The Wayfinder extension says: don't just reduce the candidates — reduce the positions. Most positions don't need the model at all. The model should only see genuine residuals, packaged as typed capsules that describe exactly what's ambiguous. The infrastructure compiles away the structural majority. The model resolves the contracted residual.

The signals ARE the understanding. The capsule IS the question. The model just answers it.

---

*Design v2.0 — March 2026. Updated with Wayfinder-derived enhancements: agent suppression, decision capsule projector boundary, Chess.com move quality classification, residual-only training. Supersedes v1.0.*
