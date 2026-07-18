# Chess as Story: The Genesis Frame Architecture

*A chess game is a story. Genesis — Patrick Winston's story-understanding engine, revived at scale on
GSE — is the frame engine that reads it. This document formally encodes that architecture.*

Status legend: **[BUILT]** implemented and verified this session · **[DESIGNED]** specified, not yet
built · **[OPEN]** known gap or research direction.

> **Forward plan:** [`GENESIS_CHESS_SCOPE.md`](GENESIS_CHESS_SCOPE.md) — the full Human-Window stack
> (legal → censored → framed → routed → judged) and the dependency-ordered build units (U1–U8), each with
> a measurement. This document is the current *state*; that is the *plan*.

---

## 1. The Thesis

Yami's original claim is that most of chess is *structural* work the infrastructure handles trivially,
leaving a small recognition residual. This architecture makes the recognition layer literal: **a chess
game is a sequence of who-does-what events — a story — with strict rules and an unusually well-documented
vocabulary of named frames** (openings, pawn structures, tactics, endgames). Genesis reads such stories
deterministically: it derives the facts a sequence *implies but never states*, recognizes the named
patterns, revises beliefs on surprise, and **abstains where nothing follows** rather than fabricating.

Two external systems make this buildable and give it a learning theory:

- **Genesis × GSE** (`~/Genesis`) — deterministic, provenance-carrying story understanding. Every
  inference carries the rule (`via`) that produced it. Semantic-class generalization is native: a rule
  `if x checkmates y then …` fires on any verb whose *meaning* is the class, not the literal token.
- **Semantic Specification Learning** (`rohan-vinaik.github.io/.../Semantic_Specification_Learning`) —
  understanding as the resolution of conceptual degrees of freedom under constraint; a domain is a
  *closed artifact inside an open field*, and the artifact is what is specifiable. Chess is the cleanest
  instance: the open field is perfect play over the 10^44 game tree (intractable, never claimed); the
  closed artifact is legal moves + tactics + endgames + the frame vocabulary (finite, derivable).

## 1A. The Robot Chef — the conceptual frame for the whole problem

*Hold this picture before any of the machinery: it is what the architecture is a machine for.*

The hardest jobs to automate are not the "big think-y" ones (run a company, do conceptual math) — they
are jobs like **head chef in a working kitchen**. Why is obvious in retrospect and it is the whole
problem. The line cook watching your brown sauce gets distracted flirting with the busboy; the sauce
over-reduces. Trivial to fix — 30 seconds for a human. But the kitchen is a **coupled system under two
scarcities at once, time and space**, and every local fix *spends a shared resource* and *displaces a
coupled task*: thin the sauce → costs burner-minutes → the fish that needed the burner slips → the veggies
steaming for the fish overcook → park them (they're cheap) → but now the steamer someone else needed is
occupied and the pastry chef's station is covered in floppy carrots → soothe their ego → and while you do,
the sauce has "reduced" so long it **catches fire**. A 30-second perturbation propagates along the coupling
until it reaches something irreversible.

The skill that resolves this is not computation — it is **understanding the load-bearing decision points**:
recalling the *temporal constraints* for prioritization (the sauce is on a clock), the *resource
constraints* (burners, counter, the shared steamer), and the *acceptable losses* (the carrots are cheap;
sacrifice them to save the dish). What you need is a system that can tell the **story of the kitchen's
operation** — which is why Genesis, not search and not an LLM narrator with no held state. (The thought
experiment is due to a Winston/Minsky student; it is the kitchen-flavored twin of Dennett's frame-problem
robot.)

**Chess is the kitchen, term for term.** Time scarcity = **tempo** (the passer is on a clock; the attack
must land before the defense consolidates). Space scarcity = **the board** (squares, files, coordination —
two pieces cannot hold one square). Acceptable losses = **the sacrifice** (give up the exchange to keep the
initiative — the carrots-are-cheap logic *is* the exchange sac, and brilliancy = the surprise that reveals
the ruined carrots were always meant to be blended). The cascade to fire = **the collapse** (one tempo
lost, one weakness fixed, a won game falling apart move by move). Yami's whole thesis — reduce ~30 legal
moves to ~3–5 annotated candidates, hand a *recognition* problem not a *search* problem — is the chef
recognizing the two things that are load-bearing *right now* instead of simulating every kitchen-state.

## 1B. The SSL construction — the theory forces the architecture

The build is not a design choice; it is read *out of* the Semantic Specification Learning learning-theory
(`~/Projects/rohan-vinaik.github.io/papers/Core Documents/Semantic_Specification_Learning/`). Understanding
= driving conceptual entropy to zero under constraint frames (the **σ_sem-trajectory**); every chess
component is a term in that theory:

| SSL construct | Chess component | Status |
|---|---|---|
| N(T), the candidate-reading neighborhood (GSE generate-wide) | Scale-0 survivors: legal + censored candidate moves | BUILT (`candidate_frames.py`) |
| The σ_sem-trajectory — constraint frames peeling readings, H→0 (§2.3) | the scale stack; each scale a candidate-elimination operator | BUILT (`progressive_engine`) |
| Meta-narrative controller — top-down SoM setting sub-frame priors (§4.1) | the **style-router** (Scale 4): a style biases the elimination | DESIGNED (U6) |
| **Falsifiability guard — σ_sem(T \| frame) > 0, must not suppress surprise (§4.3)** | **break-conditions** — what keep a style non-degenerate | mining BUILT (U3); wiring NEXT |
| Surprise = naive vs context-constrained divergence; retraction log = surprise channel (§7.2) | brilliancy / break-condition retraction | PROVEN (`retraction.py`) |
| Learning = persisted *shape* of surprise across sub-reads (§4.1) | frame-of-frames induction | BUILT (`corpus_learning.py`) |
| Completeness equation, bulk/tail split, L (§2.5) | the thermometer's coarse(bulk) / fine(tail) result | MEASURED (`style_composition_probe.py`) |
| Field/artifact — claim the artifact, not the field (§1.4c) | artifact = legal+tactics+endgames+frames+style-laws | — |

**The load-bearing identity: break-conditions ARE the §4.3 falsifiability guard.** A *style* is a
meta-narrative — Tal's frame and Petrosian's frame are two lawful symmetry-breakings, each zero-entropy in
its own regime (§5, the Stalingrad hinge). But §4.3 names the one degenerate failure: a controller that
makes every position *confirm* its style produces no surprise, collapses N(T) to one, reports σ_sem = 0 —
**Quixote at the windmills**, the chef who burns the kitchen following the ticket. A style-law *with*
break-conditions is a controller that can be *broken by the position* — it keeps σ_sem > 0, the exact
lower bound §4.3 requires. So break-conditions are not fine-style vocabulary; they are what make a style a
*lawful, falsifiable* meta-narrative rather than a one-note obsession. This collapses three units into one
mechanism: the **guarded meta-narrative controller** is simultaneously the style-router (U6), the
falsifiability guard (§4.3), and the fine-style channel — and its surprises, read bottom-up, are the
learning signal. The one place chess is *richer* than the paper: **Stockfish is a fitness oracle on the
open field**, adjudicating *which* lawful style-resolution wins — the paper resolves meaning; chess also
gets to ask which meaning was correct.

## 2. The Spine: Three Levels, Two Directions

The architecture has three cognitive levels, but they resolve to **two directions sharing one currency** —
and keeping that straight is the whole design.

| Level | Scope | Direction | Purpose |
|---|---|---|---|
| **Frames** | recognize the arc *within* one game | ↑ understand | derive what a game *means* (recognition) |
| **Frame-of-frames** | derive structure *across* many games | ↑ understand → **grow laws** | **Learning** — induct the frames the books don't state; map the residual the current frames can't see |
| **SoM routing** | select the move *at one position* | ↓ act → **apply laws** | **Winning** — read the regime, route the style-law that fits, reduce ~30 legal moves to the played move |

**The frame is the shared token.** The same intensional rule is both *"what I recognized in Tal's game"*
(understanding) and *"the law I route by when this position rhymes with Tal"* (action). One direction reads
stories to make laws (LEARNING = frames → frame-of-frames, bottom-up induction + top-down literature); the
other applies laws to make moves (ROUTING = SoM). **Learning grows the law library; routing applies it;
Stockfish judges which routing wins.** They are dual, not stacked.

A frame that fails to see the brilliance of a quiet queen sacrifice is not a bug on the *learning* side:
**that miss is the learning signal** (§6, the intentional residual). But the two directions need *two
lenses over the same games* — a sparse **FRAME lens** (Genesis-derived, marks the highlights, for learning)
and a dense **STYLE lens** (action-frequency profile, for routing/clustering). U1 made this concrete:
rendered onto move *candidates*, the frame lens floods or vanishes (§12) — it is a learning lens, not a
routing lens. Conflating them is the mistake the two-lens split exists to prevent.

## 3. The Three Tiers

```
  A game (PGN)
      │  game_renderer.py  — deterministic SAN → who-does-what event-sentences (Stage 1b)
      ▼
  TIER 1  events        per-move frames fire: tactical + positional STATES
      │  (fixpoint chaining, one --auto pass)
      ▼
  TIER 2  architecture  the rising-action ARC derived OVER the tier-1 trajectory
      │
      ▼
  TIER 3  prescription  diagnostic-state → resolving-move  =  Yami's move selection
      +  cross-game frame-of-frames  =  learning the residual over the whole corpus
```

**Key property [BUILT]:** within a single game, Tier-1 → Tier-2 is *automatic fixpoint chaining* — the
tier-1 states cascade up through the tier-2 plan-rules in one `--auto` pass. The *literal* two-pass
frame-of-frames (re-verb → re-fire) is required only *across* games (separate JVM items cannot chain),
which is exactly the Tier-3 corpus level.

## 3A. The Structural Interface (how the frames seat into the scale stack)

The frame architecture does **not** replace Yami's older move-selection machine — it *seats into it*. The
structural substrate is `progressive_engine.py`'s multi-scale pipeline, which reduces ~30 legal moves to a
decision by progressive revelation, each scale conditioning on the residual below it:

```
Scale 0  LEGAL + CENSORS   legal_moves.py + censor stack    ── the FLOOR the frames stand on
   ↓                       (blunder/tactical/repetition/learned)
Scale 1  TACTICAL          forced wins, checkmate
Scale 2  POSITIONAL        navigator OTP + structure
Scale 3  COHERENCE         OTP interference + COEC
Scale 4  TEMPORAL / SoM    specialist debate  ←────────────── where SoM ROUTING lives (styles-as-laws)
   ↓
Scale 5  NEURAL RESIDUAL   fusion model on the HARD cases only  ── the one black box, quarantined
```

The new levels meet the old machine at **specific, already-existing joints** — this is the whole point of
the "reduce 30 moves to 3-5 annotated candidates, hand a recognition problem not a search problem" thesis:

- **Scale 0 is the floor the frames stand on. [BUILT — U1]** `datagen/candidate_frames.py`
  (`frame_candidates(engine, board)`) makes the frame layer consume Scale-0 survivors — legal *and*
  censor-filtered — so a frame is *never* built on an illegal or blunder move. The renderer stays pure
  (engine-unaware); the seam runs the engine's real censor stack and renders each survivor. Old-meets-new
  happens *at Scale 0, by construction* (measure: every candidate legal, verified with a fail-loud guard).
- **SoM routing is Scale 4. [DESIGNED — U6]** Routing is not a new pipeline bolted alongside; Scale 4 is
  *already* the temporal/SoM slot. U6 fills it with styles-as-laws: read the regime → recall the
  best-matched style-law (K-line resemblance) → route the move. It *matures* an existing scale.
- **The censor is the seam at the "don't" level. [DESIGNED — U2]** Yami's `negative_learning`
  (structural, Scale-0) and Genesis `cannot`/retraction (frame-level) are the **same Minsky near-miss
  mechanism**. Unify them and a blunder learned structurally censors a *frame*, and a refutation learned in
  a frame censors a *move* — the hard/soft bridge is literally where the two architectures fuse.
- **Scale 5 neural is quarantined by design.** The one opaque layer sees only the *hard residual* the
  intensional layers could not resolve. That is the Human-Window discipline made structural: the black box
  is the leftover, never the frontier. Every layer above it is a rule a human could state.
- **Stockfish is the exogenous warrant. [DESIGNED — U7]** It judges which *routing* wins at high ELO,
  making the SoM trainable (RL over the laws, old benchmark infra as the environment) — the layer's warrant
  is outside the layer.

## 4. Tier 1 — Events (the per-move frames)

Transcribed from Yami's existing ontology, not invented.

**Tactical register [BUILT]** — `~/Genesis/gsebridge/rules/chess.rules` (+ `chess.concepts`):
`checkmate` is terminal (`mated → fallen`, `x → victorious`); the vindicated sacrifice
(`sacrifice ∧ checkmate ⇒ brilliant`) recognizes **Brilliancy** and **Checkmate**. Layers on top of the
shipped war/victory/sacrifice archetypes, which already read a game as Sacrifice/Tragedy for the loser
with zero chess authoring.

**Positional register [BUILT]** — `chess_positional.rules`: per-move micro-vocabulary mapping
positional move-verbs → Yami `models.py` state enums (`Activity` active/passive/cramped · `Safety`
safe/exposed · `Structure` isolated/hanging). Fixes the register the dramatic archetypes miss: a
positional grind derived **0** conclusions without it, and **5+** with it.

## 5. Tier 2 — The Rising-Action Arc

`chess_plan.rules` (+ `chess_plan.concepts`) **[BUILT]**, transcribed from
`knowledge_graph._activation_score` (the 7 `PlanType`s). A state-signature licenses a plan:
`control ⇒ ascendant`, `gain ⇒ ascendant`, `penetrate ⇒ dominant`; the bind compounds
(`control ∧ penetrate ⇒ dominant`) and domination makes the win inevitable (`dominant ⇒ prevails`).
Recognizes **Ascendancy** and **Domination**. Verified: the Karpov grind now reads
*Karpov: active → ascendant → dominant; opponent: cramped → constrained → vulnerable.*

## 6. Brilliancy = Surprise (the belief-revision mechanism)

A sacrifice's meaning is **not fixed when it is played** — it is *resolved by the continuation*. At the
moment of the sacrifice the naive reading is `fallen` (material given up); a following checkmate
**retracts** that `fallen`. That divergence between the naive and context-constrained reading *is*
surprise (SSL §3; the Genesis Dr. House retraction). The retraction is a logged event — the
derive→retract hysteresis is the surprise channel. **A brilliancy is the move that makes you retract the
fear it first provoked.**

**Trace-boundary hybrid [BUILT]** — `~/Projects/Regenesis/regenesis/retraction.py` (12 tests). The Java
engine's retraction cascade reconstructs support by type-rematch and *over-retracts* (it stripped the
loser's legitimate `fallen` along with the winner's false one — the impedance mismatch Regenesis Pillar 2
names). The fix keeps Java as the frozen derivation kernel and moves the reasoning to Python: rebuild the
derivation DAG from the emitted trace by symbolic unification, seed retraction from the *legitimate*
censor event, cascade only through real edges, and keep unknown-provenance facts. Verified on the Opera
trace: Java over-retracts **7** facts; the corrected layer retracts **1**; the loser's fall survives.
Rejected alternatives: live Java↔Python marshalling (re-collapses the frame → same bug); full engine
port now (premature). This Python layer is a down payment on Regenesis.

**The intentional residual [OPEN, by design]:** the renderer's SEE-based sacrifice detection catches
capture-sacs but not a quiet piece-*offer* (e.g. `16.Qb8+!!` reads as `checks`, not `sacrifices`). This
is **not patched** — the miss is precisely the residual the cross-game frame-of-frames learns from.

## 7. The Deterministic Renderer (Stage 1b)

`src/yami/datagen/game_renderer.py` **[BUILT]** — PGN → clean who-does-what event-sentences with **zero
chess judgment of our own**:

- **Tactical events** ← `tactical_scoper.scope_moves` motifs (capture/check/checkmate/fork/pin) + SEE
  (a sacrifice is a move whose static exchange gives up material — Yami's number).
- **Positional events** ← `position_signals.extract_position_signals` BEFORE/AFTER deltas (the SOM
  pipeline's dimensional-delta decomposition).
- **Cadence (a):** positional events are DISCRETE inflections tied to a concrete move action
  (castle / pawn-break / win-a-file / create-weakness / infiltrate) confirmed by a real signal jump.
  Mobility/center jitter is dropped — the slow *bind* is the Tier-2 arc, not a per-move event.
- Objects are single bare nouns (`file`, not `the open file`) — the register GSE parses cleanly.

Verified end-to-end: the real Opera Game PGN → **38** event-sentences → Genesis fires all three layers
(**47** derivations, ending correctly on `Morphy checkmates king`).

## 8. Discipline & Provenance

Inherited from the Genesis instrument method (non-negotiable):

- **Derive-or-abstain.** A finding exists only where a frame fires; a draw correctly abstains from any
  decisive Victory/Tragedy frame (verified — the falsifiability discipline holding on chess).
- **Provenance on every line.** Each derivation carries its `via` rule; abstention is a result.
- **Compute-not-impose.** Every read is a run result. The renderer and universes are transcribed from
  Yami's own ontology (`models.py`, `position_signals.py`, `knowledge_graph.py`), never authored from
  chess prior knowledge.

## 9. Component Map

| Component | Location | Status |
|---|---|---|
| Tactical universe | `~/Genesis/gsebridge/rules/chess.rules` + `.concepts` | BUILT |
| Tier-1 positional universe | `~/Genesis/gsebridge/rules/chess_positional.rules` | BUILT |
| Tier-2 arc universe | `~/Genesis/gsebridge/rules/chess_plan.rules` + `.concepts` | BUILT |
| Universe registration | `~/Genesis/gsebridge/archetypes.index` | BUILT |
| Trace-boundary retraction | `~/Projects/Regenesis/regenesis/retraction.py` (+ tests) | BUILT |
| Cross-game frame-of-frames driver | `~/Projects/Regenesis/regenesis/frame_of_frames.py` | BUILT (aimed at Tier 3) |
| Deterministic renderer | `src/yami/datagen/game_renderer.py` | BUILT |
| Scale-0 → frame seam (U1) | `src/yami/datagen/candidate_frames.py` | BUILT |
| Composition thermometer (metric) | `scripts/style_composition_probe.py` | BUILT |
| Literature miner (U3, in progress) | `scripts/u3_mine_literature.py` → `data/genesis_roster/u3_atoms.jsonl` | ACTIVE |
| Roster inputs (19 masters) | `data/genesis_roster/` (stories, labels, PGNs) | BUILT |

## 10. Status & Roadmap

**Built & verified (frame engine):** all three tiers fire on a real game; brilliancy = surprise with the
corrected retraction; the deterministic renderer closes the PGN → Genesis pipeline; discrimination holds
across brilliancy / positional / draw / overreach (each reads differently, draws abstain).

**Built & verified (structural interface + metric):** U1 Scale-0 → frame seam (`candidate_frames.py`,
measure passes); the composition thermometer (`style_composition_probe.py`) reproduced the 78%-coarse/
0%-fine baseline exactly and proved fine style is **atom-gated, not operator-gated** (§12); the 19-master
roster rescued into the repo (`data/genesis_roster/`). U3 literature mining is **active**.

**Next (ordered):**
0. **[ACTIVE] U3 literature mining** — grounded atoms + break-conditions from chess literature
   (`u3_mine_literature.py`), the fine-style unblock the thermometer proved necessary. Then wire the atoms
   into the renderer as signal-detectors (increment 2) and re-run the precedence thermometer.
1. **[DESIGNED] Corpus frame-of-frames (Tier 3, learning)** — run `frame_of_frames.py` across many of one
   player's games → derive their style arc and, crucially, *where the frames go dark* (the residual).
   Unblocked by the renderer (games at scale).
2. **[DESIGNED] Prescriptive move-selection (Tier 3, winning)** — a `diagnostic-state → resolving-move`
   universe (from `_activation_score` conditions) = Yami's actual move selection, closing the loop.

3. **[DESIGNED — U5/U6] Player-lens Society of Mind (Scale 4, the top orchestrator).** Each player's
   corpus yields a **STYLE lens** — an action-frequency profile (a distribution over `(verb, object)`
   frames) read from the *actions*, not the derived outcomes (see §12: the outcome lens is below chance).
   The payoff is the Scale-4 routing layer: **read the position's regime, match the style-law best tuned
   to win *that* kind of game, play in that style** — Yami's Temporal SoM lifted to the top, specialists
   are *players*, the orchestrator picks whose style fits. **Status is gated by vocabulary, not
   engineering:** the STYLE lens already resolves *coarse* style (dynamic vs solid, **78%**) but **0%
   fine**, and the composition thermometer (§12) proved no *ordering* of the current ~18 frames cracks
   fine style — the atoms are the gate. So routing (U6) waits on **U3** (literature atoms + their
   break-conditions), after which the re-probe is *precedence* over the enriched vocabulary. The lenses
   accumulate for free from every corpus the learning loop already fires (`data/player_lenses_v2.json`,
   `data/genesis_roster/`).

**Open gaps [OPEN]:**
- The Tier-2 arc mis-attributed `dominant` to the *loser* (transient early activity not integrated to
  game outcome) — candidate for outcome-integration or the corpus-learning layer.
- The Java retraction cascade over-retraction is fixed at the trace boundary but not in the engine; the
  proper fix belongs in the Regenesis port, not a Java patch.
- Multi-word/possessive object parsing (Stage-1b register) — mitigated by single-noun rendering.

## 11. Lineage

| Layer | System | Role |
|---|---|---|
| Story understanding | Genesis × GSE | derive-or-abstain frame engine over event-sequences |
| Learning theory | Semantic Specification Learning | σ_sem, surprise, the field/artifact completeness law |
| Method | MentalAtlas instrument (`derive_architecture`) | reframe-domain-as-story; re-verb → re-fire = frame-of-frames |
| Reasoning-over-derivations | Regenesis | provenance-native belief revision; the trace-boundary Python layer |
| Chess infrastructure | Yami | legal moves, tactical scoping, 58-dim signals, positional ontology, the renderer |

*This is the third instance of the pattern SSL names — naive-read → context-constrained-read →
meta-narrative-induction — realized in chess: events → arc → the constraint that resolves toward the move.*

## 12. Key Findings (empirical, this session)

- **Style is intensional; the histogram is extensional.** The STYLE lens (action-frequency profile,
  `style_composition_probe.py`, 19 masters, leave-one-out NN) resolves **coarse** style (dynamic vs
  solid, **78%**) but **0% fine** — because fine style is the multi-turn *rule* ("restrain, then strike"),
  not a move-frequency. The derived-**outcome** lens is worse than chance: style lives in the *actions*,
  not the outcomes.
- **Composition is gated on the atoms, not the operator.** Holding the 78%/0% baseline fixed and adding
  order channels: adjacent bigrams move coarse 78→83% (noise), fine 0; directed precedence (the
  "restrain … then strike" operator) nudges fine 0→1/19 (at chance) but *collapses* coarse 78→61% — a
  dilution signature, not signal. **No ordering of ~18 mostly-universal frames cracks fine style.** The
  composition machinery is built and validated; it waits on richer atoms (U3). Precedence — not adjacency —
  is the operator that touched fine at all, so the post-U3 re-probe is *precedence over the rich vocabulary*.
- **The frame lens is a learning lens, not a routing lens (U1).** Rendered onto move *candidates*, the
  Genesis frame render floods (`restrains` on 18/26) or vanishes (empty on 25/29) — sparse-by-design for
  marking a played game's highlights (learning), useless for discriminating quiet candidates (routing).
  This is *why* the two-lens split (§2) exists; U1 made it concrete on real candidates rather than in theory.
- **U3 harvests break-conditions, not the canonical frames.** Yami already transcribed prophylaxis /
  blockade / outpost / restraint into `chess_positional.rules`; the first mined atom re-derived
  `prevents → restrained` (expected). U3's real contribution is *when to abandon* each rule — the
  transition nothing in the current vocabulary encodes, and exactly what precedence-routing needs.

## 13. The Play-Learn Loop — from understanding to WINNING (prescription + negative learning)

The narrative read *proves Winston's thesis* (story-understanding finds meaning in chess — "the chef can
cook"); it is not yet a dish. The dish is the pipeline **position narrative → preferred frame → best moves**,
tasted by Stockfish. The loop, all on built pieces:

1. **UNDERSTAND (cook).** With the prescription channel OFF, the story engine plays coherent legal chess and
   *draws Stockfish Skill-0* from pure understanding — the thesis validated at the level of play.
2. **PRESCRIBE (the dish).** `evaluate_position → suggest_plan` (the narrative frame) → `rank_moves`, over a
   **tactical baseline** (never hang material via SEE, never allow mate-in-1) UNDER the **outcome
   prescription** (`gm_patterns` win_rate = "make this move because it WON"). `scripts/prescribe_taste.py`.
3. **TASTE (Stockfish).** Full games to completion, valid positions, terminate strictly on mate/draw.
4. **LEARN FROM LOSS (fix the recipe) — negative learning.** Play → lose → `mine_negative_examples_from_evals`
   (Stockfish evals locate the turning point = the *surprise*, belief-revision made literal) → read the
   *story of the loss* (frame-of-frames, the winner-style read inverted) → `StrategyCensorRule` auto-written
   (Socratic IV-H): "in narrative X the style I played LOST → censor; the redirect is what won" →
   `filter_moves`. `negative_learning.py` (`LearnedCensorStack`) is the built Minsky-censor store; the
   progressive engine already loads it at Scale-0. Positive from winners, negative from losers.

**ELO-BRACKETED — the best strategy is not a monolith.** Prescription and censors are keyed on
**(narrative, ELO-bracket)**: what wins vs a ~1500 differs from vs a ~2800. The routed *frame must match the
opponent's ELO* — a too-strong frame *censors perception* of threats a weaker player can execute (the
"GM loses in 7 moves" frame-failure: queened a walking passed pawn the GM's frame made invisible). `gm_patterns`
carries `avg_elo`; `opponent_profile.py` estimates opponent skill; `benchmark_elo.py` plays bracketed levels.
Learning order: historical losses → then Yami vs Stockfish, ELO-gated, climbing bracket by bracket.

### 13A. TWO SYSTEMS — the generative engine and the experiential learner are SEPARATE (do not conflate)

Step 4 (LEARN FROM LOSS) is **not** the generative engine wearing a different hat. It is a *distinct,
independent* system with a *different warrant structure*, and confusing the two is a live design error:

- **System 1 — the Genesis generative layer (creative intelligence).** Reads the position as a narrative and
  *generates* candidate frames/plans (steps 1–2). Warrant here is **exogenous-as-JUDGE**: Stockfish and
  outcomes *taste* the dish, they never *select* the move. If Stockfish selected, the generative warrant
  would collapse into a crutch (fluency-without-warrant). This is the §0 purity constraint — it applies HERE.
- **System 2 — the classical experiential learner** (blueprinted in
  `~/…/AI_architecture_papers/law_as_architecture.md`, "Law as Architecture"). A Minsky **Society of Mind** of
  first-order specialists whose learned coefficients are **near-miss censors carved from experienced failure**
  (Winston 1970), held in a **recollective sparse-distributed / K-line memory recalled by resemblance**
  (Kanerva), **stratified by regime = ELO** (§8 of that paper). This layer *legitimately grounds in
  Stockfish-scored experience* — learning from moves that FAILED is the design, not a warrant violation; the
  Stockfish refutation is exactly the "example that just fails" that carves the censor boundary. It is
  classical/symbolic/auditable — **the opposite of the ML anti-project, not an instance of it.**

**Invariant — the experiential corpus is Yami-vs-Stockfish ONLY, never a general Stockfish data source.**
System 2 learns from *Yami's own games against Stockfish* (Yami's own moves in its own losses), so the learned
failure suite is a schema of *Yami's* mistakes, ELO-scoped — not a foreign engine's. (`learn_loop_v2.py`
already honors this: `play()` generates Yami-v-Stockfish games; `mine()` scores only Yami's moves.)

**Integrity rule (System 2's own, NOT §0's):** *learn-then-recall, never live-consult.* Coefficients are
mined offline from played games and **recalled by resemblance** at play time; Stockfish never enters `pick`.
"Query Stockfish live per move" is not learning — it is proxying the oracle, and it destroys the very schema
System 2 exists to build. (Current code obeys this: Stockfish appears only in `mine`/`sf_eval_after`.)

The soft `RiskOverlay` (`scripts/learn_loop_v2.py`) is the *proto* of System 2: `risk[(plan, elo, uci)]` = the
SDM key (context+regime), `learn(eval_drop)` = the near-miss carving, ELO = the regime stratum, the soft
penalty (not a hard ban) = correct because a lost move is *risky*, not *impossible* (hard censors stay reserved
for illegal + mate-in-1, which `pick` already makes unrepresentable). The mature System 2 is a separate future
build; the overlay is its seed.

*Status (2026-07-13): 1–3 wired (`prescribe_taste.py`); 4's turning-point primitive proven on a Capablanca
loss (+273 → −281 located); the soft overlay proven to re-prioritize off learned traps 5/5 (no ban).
**A: the safety-seeking rethink — BUILT & PROVEN** (`learn_loop_v2.py`, `SAFETY_GAIN` gated on rethink
pressure): re-prioritizes off traps 5/5 AND now 5/5 toward a Stockfish-SAFER move (was 3/5), incl.
`Kd1(−9998, into mate) → Qd2(−602)`. Next — **B: the per-style negative lens** from the corpus's losses.*
