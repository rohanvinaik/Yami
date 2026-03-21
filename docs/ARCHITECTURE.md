# Yami: The Missing Layer for LLM Chess

*The Wayfinder architecture applied to chess. Infrastructure-first, neural-residual.*

## The Thesis

LLMs are terrible at chess for the same reason they were "bad" at theorem proving: they're spending neural capacity on structural work that infrastructure handles trivially. Legal move generation, one-move tactics, known endgames, opening theory — these are the chess equivalents of namespace resolution, premise scoping, and tactic family identification.

Wayfinder proved that 58% of Mathlib theorem proving is infrastructure. The prediction: **80-90% of chess decision-making is infrastructure.** The hard residual — long-term strategic planning, prophylactic thinking, complex sacrifices — is where a language model's pattern recognition actually adds value.

Yami is the proof compiler for chess. It handles the structural layers, reduces the decision space from ~30 legal moves to ~3-5 annotated candidates, and hands the LLM a recognition problem instead of a search problem.

## Name

闇 (yami) — darkness, the unseen. The infrastructure works in the dark. The LLM sees only the clean residual.

---

## Architecture Overview

```
Board position
  ┌─────────────────────────────────────────────────────┐
  │ Layer 1: Legal Move Generation (deterministic)       │
  │   python-chess / fairy-stockfish backend             │
  │   Output: ~20-35 legal moves                         │
  ├─────────────────────────────────────────────────────┤
  │ Layer 2: Tactical Scoping (pattern matching)         │
  │   Fork/pin/skewer/discovery/mate threat detection    │
  │   Hanging piece detection, exchange evaluation       │
  │   Output: moves tagged with tactical motifs          │
  ├─────────────────────────────────────────────────────┤
  │ Layer 3: Endgame Resolution (tablebases)             │
  │   Syzygy 7-piece tablebases → exact evaluation       │
  │   If ≤7 pieces: solved. No LLM needed.              │
  ├─────────────────────────────────────────────────────┤
  │ Layer 4: Opening Book (database lookup)              │
  │   First 10-15 moves from master games                │
  │   Lichess opening explorer / Polyglot books          │
  ├─────────────────────────────────────────────────────┤
  │ Layer 5: Positional Knowledge Graph                  │
  │   Pawn structure → plan templates                    │
  │   Piece activity scoring, king safety, space control │
  │   Navigational evaluation like Wayfinder's 6-bank   │
  ├─────────────────────────────────────────────────────┤
  │ Layer 6: Candidate Filtering + Annotation            │
  │   Prune to top 3-5 candidates                        │
  │   Annotate each with: tactical motif, positional     │
  │   evaluation, plan alignment, risk assessment        │
  ├─────────────────────────────────────────────────────┤
  │ Layer 7: LLM Decision (the residual)                 │
  │   Receives: 3-5 annotated candidates + board context │
  │   Decides: which candidate best serves the plan      │
  │   This is recognition, not search.                   │
  ├─────────────────────────────────────────────────────┤
  │ Layer 8: Legal Move Verification (deterministic)     │
  │   Confirm move is legal. The kernel is the judge.    │
  └─────────────────────────────────────────────────────┘
```

---

## Layer-by-Layer Design

### Layer 1: Legal Move Generation

**What:** Generate all legal moves from the current position.

**Implementation:** `python-chess` library. This is a solved problem — the library handles all rules including en passant, castling, promotion, and repetition detection. Zero ambiguity, zero neural cost.

**Wayfinder analogue:** `goal_start()` — creating the valid proof state from which search begins.

**Cost:** Microseconds.

### Layer 2: Tactical Scoping

**What:** Tag each legal move with tactical motifs and detect forcing sequences.

**Implementation:**

```python
class TacticalScoper:
    """Pattern-match tactical motifs on the board."""

    def scope(self, board: chess.Board) -> list[ScopedMove]:
        moves = list(board.legal_moves)
        scoped = []
        for move in moves:
            motifs = []
            board.push(move)

            # Check for tactical consequences
            if board.is_check():
                motifs.append("check")
            if self._creates_fork(board, move):
                motifs.append("fork")
            if self._creates_pin(board, move):
                motifs.append("pin")
            if self._creates_discovery(board, move):
                motifs.append("discovery")
            if self._wins_material(board, move):
                motifs.append("material_gain")
            if self._hangs_piece(board, move):
                motifs.append("hangs_piece")  # CENSOR: suppress this

            board.pop()
            scoped.append(ScopedMove(move=move, motifs=motifs))
        return scoped
```

Key insight: **tactical motifs are the chess equivalent of tactic families.** Just as Wayfinder separates `rw`, `simp`, `apply`, and `exact` into different execution pipelines, Yami separates tactical, positional, prophylactic, and defensive moves into different evaluation paths.

**Wayfinder analogue:** `scope_for_rw()` / family-specific scoping. The scoper reduces ~30 legal moves to ~10 tactically relevant ones.

**The censor principle:** Moves that hang material (detected by static exchange evaluation) are **suppressed** before the LLM sees them. This is Minsky's censor/suppressor applied to chess — learning what NOT to do. The LLM never wastes capacity considering blunders.

**Cost:** ~1ms per position (static analysis).

### Layer 3: Endgame Resolution

**What:** If ≤7 pieces remain on the board, look up the exact evaluation and best move from Syzygy tablebases.

**Implementation:**

```python
class EndgameResolver:
    """Syzygy tablebase lookup for exact endgame play."""

    def __init__(self, tablebase_path: str):
        self.tb = chess.syzygy.open_tablebase(tablebase_path)

    def resolve(self, board: chess.Board) -> Move | None:
        if chess.popcount(board.occupied) > 7:
            return None  # Not in endgame tablebase range

        # Exact solution — no search needed
        best_move = self.tb.probe_root(board)
        return best_move  # Guaranteed optimal
```

**Wayfinder analogue:** `exact?` — Lean's own search that closes the goal when the infrastructure has reduced the problem far enough. In chess, the "far enough" threshold is ≤7 pieces.

**Cost:** Microseconds (disk lookup).

### Layer 4: Opening Book

**What:** For the first 10-15 moves, play from established theory.

**Implementation:**

```python
class OpeningBook:
    """Polyglot opening book + Lichess opening explorer."""

    def lookup(self, board: chess.Board) -> list[BookMove]:
        # Try Polyglot book first (fast, local)
        if self.polyglot.find(board):
            return self.polyglot.weighted_choices(board, n=3)

        # Fall back to Lichess explorer (API, slower)
        if board.fullmove_number <= 15:
            return self.lichess.get_master_moves(board.fen(), n=3)

        return []  # Out of book
```

**Wayfinder analogue:** The navigational knowledge graph. Known positions are like known proof patterns — you don't search, you navigate.

**Cost:** Microseconds (local book) to ~100ms (API).

### Layer 5: Positional Knowledge Graph

**What:** Evaluate positional features and map them to strategic plans. This is the Wayfinder core — a structured semantic network over chess knowledge.

**Implementation:**

The chess knowledge graph has **5 navigational dimensions** (analogous to Wayfinder's 6 banks):

| Dimension | Chess meaning | Values |
|-----------|--------------|--------|
| **MATERIAL** | Piece count balance | {ahead, equal, behind} |
| **STRUCTURE** | Pawn structure type | {open, semi-open, closed, hedgehog, ...} |
| **ACTIVITY** | Piece mobility/coordination | {active, passive, cramped} |
| **SAFETY** | King safety evaluation | {safe, exposed, under_attack} |
| **TEMPO** | Initiative/development | {ahead, equal, behind} |

Each position is placed in this 5-dimensional space. Each placement activates a **plan template** (analogous to Wayfinder's 9 proof templates):

| Plan template | Activation condition | Typical moves |
|--------------|---------------------|---------------|
| **ATTACK_KING** | SAFETY(opp)=exposed, ACTIVITY=active | Piece sacrifices, pawn storms |
| **IMPROVE_PIECES** | ACTIVITY=passive, STRUCTURE=closed | Piece maneuvers, outpost occupation |
| **PAWN_BREAK** | STRUCTURE=closed, TEMPO=ahead | Central/flank pawn advances |
| **SIMPLIFY** | MATERIAL=ahead | Exchanges, endgame transition |
| **FORTIFY** | SAFETY(self)=exposed | Defensive moves, king shelter |
| **EXPLOIT_WEAKNESS** | STRUCTURE has target | Piece pressure on weak squares |
| **PROPHYLAXIS** | Opponent has plan | Prevent opponent's plan |

```python
class PositionalKnowledgeGraph:
    """5-dimensional positional navigation for chess."""

    def evaluate(self, board: chess.Board) -> PositionalProfile:
        return PositionalProfile(
            material=self._eval_material(board),
            structure=self._eval_pawn_structure(board),
            activity=self._eval_piece_activity(board),
            safety=self._eval_king_safety(board),
            tempo=self._eval_tempo(board),
        )

    def suggest_plan(self, profile: PositionalProfile) -> PlanTemplate:
        """Navigate the knowledge graph to find the right plan."""
        # Spreading activation over plan templates
        scores = {}
        for template in PLAN_TEMPLATES:
            scores[template] = self._activation_score(profile, template)
        return max(scores, key=scores.get)

    def rank_moves(self, moves: list[ScopedMove], plan: PlanTemplate) -> list[RankedMove]:
        """Rank moves by alignment with the active plan."""
        ranked = []
        for move in moves:
            alignment = self._plan_alignment(move, plan)
            ranked.append(RankedMove(move=move, alignment=alignment))
        return sorted(ranked, key=lambda r: r.alignment, reverse=True)
```

**Wayfinder analogue:** The 6-bank ternary navigator + template classifier + SubtaskIR. The positional knowledge graph IS the chess proof network.

**Cost:** ~10ms (all features are computable from board state).

### Layer 6: Candidate Filtering + Annotation

**What:** Reduce to 3-5 candidates. Annotate each with full context for the LLM.

**Implementation:**

```python
class CandidateFilter:
    """Produce annotated candidates for the LLM."""

    def filter(self, board, tactical_moves, plan, profile) -> list[AnnotatedCandidate]:
        # Suppress blunders (censor)
        safe_moves = [m for m in tactical_moves if "hangs_piece" not in m.motifs]

        # Rank by plan alignment
        ranked = self.kg.rank_moves(safe_moves, plan)

        # Take top 3-5
        candidates = ranked[:5]

        # Annotate each
        annotated = []
        for c in candidates:
            annotated.append(AnnotatedCandidate(
                move=c.move,
                san=board.san(c.move.move),
                tactical_motifs=c.move.motifs,
                plan_alignment=c.alignment,
                plan_template=plan.name,
                positional_evaluation=self._eval_after_move(board, c.move),
                risk_assessment=self._assess_risk(board, c.move),
                narrative=self._generate_narrative(board, c, plan),
            ))
        return annotated
```

The annotation produces natural-language context:

```
Candidate: Nf5
  Tactical: threatens fork on e3
  Plan: ATTACK_KING (king safety exposed after ...g6)
  Positional: improves knight activity (+0.3), targets weak d6 pawn
  Risk: low (piece is defended, no tactical refutation in 3 moves)
  Narrative: "Centralizes the knight on f5, targeting the weakened kingside.
             Consistent with the attacking plan."
```

**Wayfinder analogue:** The structured residual output — lane attribution, candidate sets, failure diagnostics. The LLM receives the same quality of structured context that Wayfinder provides for unsolved theorems.

### Layer 7: LLM Decision

**What:** The LLM receives 3-5 annotated candidates and chooses the best one.

**Implementation:**

```python
DECISION_PROMPT = """You are playing chess. The position has been analyzed by the infrastructure layer.

Current plan: {plan_template}
Positional profile: {profile_summary}

Candidates:
{annotated_candidates}

Choose the move that best serves the long-term plan. Consider:
- Does this move improve your position or just maintain it?
- Does this move create problems for your opponent?
- Is this move consistent with the plan, or does it require a plan change?

Respond with just the move in SAN notation."""
```

The LLM's job is now **recognition**: read the structured context, understand the strategic situation, and pick from 3-5 pre-validated options. This is what language models are actually good at — understanding context and making judgment calls.

**Wayfinder analogue:** The frontier model receiving the structured residual. The LLM doesn't search — it recognizes.

**Key design choice:** The LLM can also say "none of these — I want to change the plan." This is the equivalent of Wayfinder's temporal controller overriding the lane order. The infrastructure suggests; the LLM decides.

### Layer 8: Legal Move Verification

**What:** Confirm the LLM's chosen move is legal.

**Implementation:** `board.is_legal(move)` — one function call.

**Wayfinder analogue:** Lean 4 verification. The kernel is the judge. If the LLM somehow produces an illegal move (hallucination), it's caught instantly and the next candidate is used.

---

## The Censor Architecture (Negative Learning Applied to Chess)

The most powerful component is what the LLM **never sees.** Following the Negative Learning principle (22x sample efficiency in prior experiments), Yami implements censors at every layer:

| Censor | What it suppresses | Layer |
|--------|-------------------|-------|
| **Blunder censor** | Moves that hang material (SEE < -100cp) | Layer 2 |
| **Tactical censor** | Moves that walk into known tactical patterns | Layer 2 |
| **Repetition censor** | Moves that repeat a position (unless winning) | Layer 1 |
| **Plan censor** | Moves that contradict the active plan (unless forced) | Layer 5 |
| **Endgame censor** | In won endgames, suppress moves that don't make progress | Layer 3 |

Each censor is a `{DON'T}` in OTP terms: -1 in the action space. The LLM only sees the moves that survive all censors. This means:

- It **never** considers hanging its queen
- It **never** considers walking into a fork
- It **never** considers moves that repeat without improvement
- It **never** considers moves that abandon a winning endgame technique

The result: the LLM's effective error rate drops by the product of all censor effectiveness rates. If each censor catches 90% of its target errors, five censors reduce the error rate by 10^5.

---

## The Knowledge Graph: Navigational Chess Intelligence

### Entity Types

| Entity | Count (est.) | Analogue in Wayfinder |
|--------|-------------|----------------------|
| Pawn structures | ~200 named types | Tactic anchors |
| Opening systems | ~500 named openings | Namespace anchors |
| Endgame patterns | ~100 named endings | Type anchors |
| Tactical motifs | ~30 named patterns | Family anchors |
| Strategic plans | ~50 named strategies | Template anchors |
| Piece configurations | ~1000 named patterns | Name anchors |

Total: ~1,900 entities (vs Wayfinder's 242,000). Much smaller graph — faster navigation, cheaper storage.

### Navigation Dimensions

```
Position → [MATERIAL, STRUCTURE, ACTIVITY, SAFETY, TEMPO]
         → Plan template activation
         → Candidate move ranking
```

This is the chess instantiation of GSE (Geometric Semantic Encoding): meaning (the right move) arises from position in structured geometric space, not from learned embeddings.

---

## Stockfish as Oracle (exact? equivalent)

Stockfish plays the role of `exact?` in the Wayfinder architecture:

- **Teacher/oracle:** Evaluate candidates with Stockfish at depth 20 to generate training data
- **Validation:** Confirm that the infrastructure's top candidate matches Stockfish's top choice
- **Fallback:** If the LLM chooses poorly, Stockfish can override (like `exact?` as last-resort)

**Critical design choice:** Stockfish is NOT the runtime engine. It's the teacher. The goal is a system where the infrastructure + LLM plays at 2000+ ELO without Stockfish at runtime. Stockfish is used only during training/evaluation.

This mirrors Wayfinder's approach: `exact?` is the teacher/oracle that establishes the ceiling. The learned system approximates its behavior at a fraction of the cost.

---

## ELO Projections

| Configuration | Estimated ELO | Baseline |
|--------------|---------------|----------|
| Raw LLM (GPT-4, no infrastructure) | ~800-1200 | Published results |
| + Legal move enforcement | ~1000-1400 | No illegal moves |
| + Blunder censor | ~1400-1600 | No hanging pieces |
| + Tactical scoping | ~1600-1800 | Sees 1-move tactics |
| + Opening book | ~1700-1900 | Sound openings |
| + Endgame tablebases | ~1800-2000 | Perfect endgames |
| + Positional knowledge graph | ~1900-2100 | Strategic planning |
| + Plan-aligned candidate filtering | ~2000-2200 | Focused decisions |
| **Full Yami stack** | **~2000-2200** | **Club player to expert** |

Each layer is additive and independently measurable. The final system should play at **expert level (2000-2200 ELO)** — comparable to a strong club player, achieved with a small LLM and deterministic infrastructure.

For comparison, the strongest chess engines (Stockfish, Lc0) play at 3500+ ELO. Yami doesn't compete with engines — it demonstrates that infrastructure makes LLMs competent at a structured task they're currently terrible at.

---

## Implementation Plan

### Phase 1: Foundation (1-2 weeks)
- `python-chess` integration for legal move generation
- Basic tactical scoper (checks, captures, threats)
- Blunder censor (static exchange evaluation)
- Stockfish evaluation oracle
- LLM integration (Claude/GPT-4 API)
- ELO measurement framework (play against Stockfish at fixed levels)

### Phase 2: Knowledge Layers (2-3 weeks)
- Opening book (Polyglot + Lichess explorer)
- Endgame tablebases (Syzygy)
- Pawn structure classifier
- Basic positional evaluation (material, king safety, piece activity)

### Phase 3: Knowledge Graph (2-3 weeks)
- 5-dimensional positional space
- Plan template taxonomy
- Candidate filtering + annotation
- LLM decision prompt engineering

### Phase 4: Censors + Refinement (1-2 weeks)
- Full censor stack (blunder, tactical, repetition, plan, endgame)
- Negative learning from game losses
- Near-miss analysis from close games
- K-line memory for successful game patterns

### Phase 5: Evaluation + Publication (1-2 weeks)
- Play 1000+ games against Stockfish at calibrated levels
- ELO curve with and without each layer (ablation)
- Comparison to raw LLM baselines (published results)
- Write-up: "The Missing Layer for LLM Chess"

**Total: 8-12 weeks** from zero to publishable results.

---

## Why This Matters Beyond Chess

Chess is the demonstration vehicle, not the destination. The architecture proves that the Wayfinder principle — **infrastructure handles the structural layers, neural models handle the residual** — generalizes:

| Domain | Verifier | Structure | Infrastructure | Residual |
|--------|----------|-----------|---------------|----------|
| Theorem proving | Lean kernel | Mathlib knowledge graph | Wayfinder | ~42% hard proofs |
| Chess | Legal move checker | Opening/endgame databases | Yami | ~10-20% strategic decisions |
| Code generation | Compiler/linter | AST, type system, APIs | LintGate | Complex logic |
| Genomics | Biochemical constraints | Pathway databases | GenomeVault | Novel interactions |

Each domain has the same decomposition: most of the problem is structural, cheap, and deterministic. The expensive neural model should only see the hard tail.

Yami makes this argument accessible to everyone. People understand chess. They can see that an LLM making illegal moves is wasting capacity. They can feel the improvement when infrastructure handles the rules. The theorem-proving result is stronger scientifically, but the chess result is stronger communicatively.

Together they say: **the scaling paradigm is optimizing the wrong frontier.** Not model size vs. performance. Infrastructure quality vs. residual hardness.

---

*Yami architecture document v1.0 — March 2026. The Wayfinder thesis applied to chess.*
