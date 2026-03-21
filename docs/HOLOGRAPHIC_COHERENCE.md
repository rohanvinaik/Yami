# Holographic Coherence Architecture: From Surviving to Winning

*The next phase of Yami — multi-signal coherence through holographic answer spaces.*

## The Insight

The answer to "what move should I play?" doesn't exist in any single signal source. It exists as an **interference pattern** across all sources simultaneously. Like a hologram, no single fragment contains the full image — but the superposition of all fragments reconstructs it.

This is the Multi-Scale Lens Architecture applied to chess: multiple orthogonal projections (lenses) at different scales, where each lens captures a unique aspect and the combination provides complete understanding.

## Current State (March 2026)

Yami achieves ~1,000 ELO (100% vs Random, calibrated benchmark) with:
- 6-Bank Navigator (729-bin ternary position classification)
- Temporal Controller (4-phase FSM with 10 strategic plans)
- Strategy Library (13 pre-encoded chess strategies)
- Coherence Layer (4-signal agreement scoring)
- K-Line Memory (7,723 winning patterns in SQLite)
- Negative Learning censors (3 strategy rules)
- Neural model (294K Balanced Sashimi, +640 ELO over infrastructure)

**The gap:** We play perfect defense (0 illegal moves, 0 blunders) but struggle to create winning chances against Stockfish at standard time controls. The system survives but doesn't attack.

## The Three Upgrades

### 1. Grandmaster Pattern Database (Empirical Signal)

**What:** Download top-level PGN databases (Lichess elite games, world championship matches). For each position, find the nearest grandmaster game and see what was played.

**How it works:**
- Index grandmaster games by material signature + nav_vector + key anchors
- At runtime: hash current position → find nearest GM games → extract move frequency
- "In positions like this, GMs played Nd5 80% of the time" → strong signal
- Not a hard override — a weighted suggestion into the coherence layer

**Why this matters for active play:**
GMs don't play defensively. Their move distributions encode *initiative*, *attack timing*, *positional sacrifices* — exactly the aggressive patterns our system lacks. The GM database is a compressed representation of "what winning chess looks like" at the strategic level.

**Implementation:**
- `src/yami/gm_patterns.py` — PGN parser, position indexer, nearest-game lookup
- `scripts/import_gm_games.py` — Download and index Lichess elite database
- SQLite database of position → move → frequency triples
- Integrated as Signal 5 in the coherence layer

**Data sources:**
- Lichess Elite Database (free, ~10M games of 2000+ ELO players)
- World Championship matches (curated, ~1000 games)
- Top-100 player games (downloadable from Lichess API)

### 2. Temporal Society of Mind (Orchestration Upgrade)

**What:** Upgrade the temporal controller from a simple FSM to a full Society of Mind arbiter with specialist agents, trajectory convergence tracking, and multi-move plan evaluation.

**The Wayfinder connection:**
Wayfinder's temporal controller manages proof search with 4 phases (structural_setup → local_close → automation_close → repair_or_replan) and 6 specialist slots. The chess version:

**Specialist agents:**

| Agent | Expertise | Scale | When Active |
|-------|-----------|-------|-------------|
| **Tactical Agent** | Forks, pins, forcing lines | 1-3 moves | Tactical motifs detected |
| **Positional Agent** | Outposts, structure, coordination | 5-10 moves | Quiet positions |
| **Endgame Agent** | Technique, opposition, pawn races | Until game end | ≤12 pieces |
| **Attack Agent** | King safety exploitation, sacrifice patterns | 3-8 moves | Opponent king exposed |
| **Defense Agent** | Fortress, prophylaxis, consolidation | 3-5 moves | Own king in danger |
| **Initiative Agent** | Tempo, development, space | 2-4 moves | Equal positions |

**Trajectory convergence (from sparse-wiki):**
When the Tactical Agent and the Attack Agent agree for 3 consecutive moves, that convergence IS the confidence signal. The temporal arbiter tracks:
- Which agents have been agreeing (convergence → boost)
- Which agents have been contradicting (divergence → replan)
- How long the current plan has been converging (persistence = strength)

This is exactly the sparse-wiki "recursive trajectory disambiguation" — correct strategies converge across layers, wrong ones diverge.

**The holographic principle:**
Each specialist agent provides a partial view (a "lens" in Multi-Scale Lens Architecture terms). The arbiter doesn't pick one agent — it reads the interference pattern across all of them. A move that the Tactical, Attack, AND Initiative agents all support is holographically strong — the answer exists in the superposition.

**Implementation:**
- Upgrade `src/yami/temporal_controller.py` with specialist agent slots
- Each specialist is a lightweight scoring function, not a separate model
- Arbiter tracks trajectory convergence per agent pair
- Convergence history persists across moves (K-line style)

### 3. Holographic Coherence (The Fusion Layer)

**What:** Upgrade the coherence layer from simple weighted averaging to holographic interference pattern detection.

**The key concepts from the research base:**

**Informational Zero (OTP):**
When a signal source scores 0, that's not absence — it's orthogonality. "This signal has nothing to say about this position." The coherence layer must distinguish:
- **+1**: Signal supports this move (constructive interference)
- **0**: Signal is orthogonal (no information — do NOT count as disagreement)
- **-1**: Signal opposes this move (destructive interference)

This is the OTP ternary applied to coherence: {support, irrelevant, contradict}.

**Multi-Scale Lens (decomposition):**
Each signal source is a lens at a different scale:
- Navigator OTP: positional scale (what direction should the game go?)
- Strategy Library: strategic scale (what multi-move plan fits?)
- GM Patterns: empirical scale (what do strong players actually do here?)
- Temporal Agents: tactical/positional/endgame scale (what does each specialist say?)
- K-Line Memory: pattern scale (have we seen this before?)
- Censor Stack: negative scale (what must we NOT do?)

**Catalytic processing:**
Each position is processed once through all lenses. The coherence result is computed. The raw signal data is evicted — only the fused score persists. This is the Catalytic Computing principle: touch once, fuse, evict.

**Trajectory convergence:**
Track coherence scores across moves. A plan where coherence increases over 3+ moves is converging (good). A plan where coherence decreases is diverging (replan). This is the sparse-wiki disambiguation principle applied temporally.

**Implementation:**
- Upgrade `src/yami/coherence.py` with ternary signal handling
- Add trajectory tracking (coherence history per move)
- Implement interference pattern detection:
  - Constructive: 3+ signals agree → strong boost
  - Partial: 2 signals agree, rest orthogonal → moderate boost
  - Destructive: signals contradict → suppress and replan
- Weight by convergence history (persistent agreement > one-time agreement)

## The Holographic Answer Space

The name comes from holography: in a hologram, every piece of the film contains the entire image, just from a different angle. Similarly, every signal source contains partial information about the correct move. The answer exists in all of them simultaneously.

The coherence layer doesn't "choose" between signals. It reads the interference pattern. When the navigator says "attack" and the GM database shows Nd5 and the temporal controller's Attack Agent says "converge on kingside" and the K-line memory recalls a winning knight sacrifice pattern — that four-way constructive interference IS the answer. No single source produced it. It emerged from the superposition.

This is why the system should produce active, aggressive play: the GM database brings initiative. The temporal agents bring plan persistence. The strategy library brings multi-move sequences. The navigator brings positional direction. The holographic coherence layer fuses them into moves that aren't just safe — they're purposeful.

## Data and Dependencies

**GM Database:**
- Lichess Elite Database: https://database.lichess.org/ (free)
- Format: PGN, compressed, ~10GB for 2000+ ELO games
- Processing: index by material + nav_vector + anchors → SQLite

**Specialist Agents:**
- Each is a scoring function in `temporal_controller.py`
- No separate models — uses existing infrastructure (navigator, anchors, piece values)
- Trajectory convergence tracked in TemporalState

**Holographic Coherence:**
- Upgrade to existing `coherence.py`
- Add ternary signal handling (OTP {+1, 0, -1})
- Add trajectory history (list of coherence scores per move)

## Expected Impact

| Component | Expected ELO Impact | Mechanism |
|-----------|-------------------|-----------|
| GM Patterns | +100-200 | Initiative injection — "what do winners do?" |
| Temporal SoM | +50-100 | Plan persistence + multi-agent convergence |
| Holographic Coherence | +50-100 | Better signal fusion, fewer contradictory moves |
| **Combined** | **+150-300** | **Compounding — coherent aggressive play** |

The target: consistently draw Stockfish Skill 0 and occasionally win. Move from ~1,000 ELO (100% vs Random) to ~1,300-1,500 ELO (competing with weak engine settings).

## Connection to Broader Research

| Concept | Source | Application in Yami |
|---------|--------|-------------------|
| Multi-Scale Lens Architecture | GenomeVault → ARC → LintGate | Each signal is a lens at a different scale |
| Informational Zero | OTP (Tier 0 Genesis) | Zero in coherence = orthogonal, not absent |
| Catalytic Computing | GenomeVault | Touch once, fuse, evict — streaming position processing |
| Trajectory Convergence | sparse-wiki | Correct strategies converge across moves/signals |
| Society of Mind | Minsky → Wayfinder | Specialist agents with arbiter orchestration |
| Constrained Hallucination | ShortcutForge → Yami | The GM database constrains the "hallucination" of strategy |
| Near-Miss Learning | Winston | GM games where the second-best move lost |
| K-Lines | Minsky → Wayfinder | Winning patterns as reusable templates |

## Build Order

1. **GM Pattern Database** (Signal 5)
   - Download Lichess elite PGNs
   - Build position → move → frequency index
   - Integrate into coherence layer

2. **Temporal Society of Mind** (Orchestration upgrade)
   - Add 6 specialist agents to temporal controller
   - Track trajectory convergence per agent pair
   - Feed specialist scores into coherence layer

3. **Holographic Coherence** (Fusion upgrade)
   - Ternary signal handling
   - Trajectory history tracking
   - Interference pattern detection
   - Convergence-weighted scoring

4. **Retrain neural model** on 200K data (background task completing)
   - Include coherence features in training data
   - The model learns to exploit the holographic signal

5. **Benchmark**
   - 42-game suites vs Random and Stockfish levels
   - Target: first wins against Stockfish Skill 0

---

*Architecture document v1.0 — March 2026. The Multi-Scale Lens Architecture, Informational Zero, and Catalytic Computing principles applied to chess move selection through holographic multi-signal coherence.*
