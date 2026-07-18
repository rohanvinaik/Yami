# Genesis Chess — Scope & Roadmap

*The build plan for a **Human-Window model of chess understanding**: comprehensible rules-with-exceptions
end to end — legal → censored → framed → routed → judged — not a black-box weight anywhere. Emerged
collaboratively; this document scopes it into dependency-ordered, individually-measured units.*

Status tags: **[BUILT]** done & verified · **[NEXT]** ready, unblocked · **[BLOCKED: X]** waiting on X ·
**[DEBT]** known bug/gap · **[RESEARCH]** downstream, uses the built system.

---

## 1. The Vision

A chess game is a *story*; Genesis (Winston/Minsky story-understanding, revived at scale on GSE) reads
it. Everything is held as **intensional** rules a human could state, follow, and recognize — the Human
Window (Michie/Kopec) — not as extensional move-tables or opaque embeddings. That intelligibility is not
decoration: it is *why* the system can answer "is Tal an answer to Petrosian" and *why* every layer
carries an exogenous warrant instead of self-certifying.

## 2. The Full Stack (every layer a rule a human could state)

```
Rule 0   LEGAL MOVES        hard kernel — python-chess is the judge          [BUILT: legal_moves.py]
   ↓
CENSORS  near-miss "don't"  make blunders UNrepresentable, not penalized     [BUILT both sides; UNIFY]
   ↓                        (Yami negative_learning ⇄ Genesis `cannot`/retraction — the SAME mechanism)
FRAMES   strategic heuristics  intensional rules (from literature + induction) [PARTIAL — coarse only]
   ↓
LAWS     style routing        SoM selects the style-law per move, regime-stratified   [BLOCKED: frames]
   ↓
FITNESS  Stockfish            exogenous warrant — which routing actually WINS  [BLOCKED: laws]
```

**The unification insight (§censors):** near-miss is one mechanism across four systems —
`negative_learning.mine_near_misses`, the Genesis retraction/surprise (naive `fallen` *almost* held →
refuted → `cannot`), `law_as_architecture`'s hard near-miss censors, and the Knight's Tour's "break the
rule at square 11." A censor learned on either side (Yami games ⇄ Genesis frames) must flow to the other.

## 3. The Two Axes

- **LEARNING = frame-of-frames.** Grows the frame/law *library* — from games (induction, bottom-up) and
  from literature (extraction, top-down). Discovers tactics and *entirely new strategies*.
- **ROUTING = SoM.** Applies the right style-*law* per move (regime-stratified, recalled by resemblance,
  censored, composite-supervised — `law_as_architecture` 1:1).

Learning grows the laws; routing applies them; Stockfish judges which routing wins.

## 4. What the Session Proved (the facts the scope stands on)

- **[BUILT] Genesis reads chess.** Tactical + positional + rising-action-arc frames fire on real PGN
  (`chess.rules`, `chess_positional.rules`, `chess_plan.rules`, registered in `archetypes.index`).
- **[BUILT] Brilliancy = surprise.** A sacrifice's meaning is fixed by its continuation; the retraction
  of the naive `fallen` IS the learning signal. Trace-boundary Python layer `regenesis/retraction.py`
  (12 tests) corrects the Java cascade's over-retraction (Java frozen, Python reasons over the DAG).
- **[BUILT] Deterministic renderer.** `game_renderer.py` — PGN → who-does-what event-sentences from
  Yami's own move/signal machinery, zero chess judgment of its own.
- **[BUILT] Learning loop closes.** 150-game Magnus corpus → recurring induction candidates (`capture ⇒
  isolate` 127/150) → principled promotion (conjunctive rule + recognition concept) → the frames the
  corpus taught now recognize (`chess_learned.rules/.concepts`).
- **[BUILT] Cross-agent frame principle.** States are shared, actions are owned → cross-agent frames key
  on the derived STATE (`KingAttack`).
- **[BUILT] Comprehensive roster.** 19 masters spanning the full style taxonomy, fetched from 365chess
  (`fetch_365.py`, Scale-0 symbolic per `STRUCTURED_WEB_KNOWLEDGE_EXTRACTION`).
- **[KEY FINDING] Style is intensional; the histogram is extensional.** Action-frequency profiles resolve
  **coarse** style (aggressive vs solid, **78%**) but **0% fine** — because fine style is the multi-turn
  *rule* ("restrain, then strike"), not a move-frequency. The derived-OUTCOME lens is even worse (below
  chance); style lives in the ACTIONS, not the outcomes.
- **[METRIC] The clustering test.** Nearest-neighbor style-match rate (`action_profile.py`,
  `scripts/style_composition_probe.py`) is the measure of a vocabulary's style-resolving power. Grow
  frames → re-run → does fine-match climb?
- **[KEY FINDING] Composition is gated on the atoms, not the operator.** The thermometer
  (`scripts/style_composition_probe.py`, 19 masters, LOO-NN) held the 78%-coarse/0%-fine baseline fixed
  and added order channels. *Adjacent* bigrams: coarse 78→83% (+1, noise), fine 0. *Directed precedence*
  (window 4, the "restrain … then strike" operator): fine 0→1/19 (at chance) but coarse *collapses*
  78→61% — a dilution signature, not signal. **Verdict: no ordering of ~18 mostly-universal frames
  cracks fine style.** Composition machinery is built and validated; it is waiting on richer atoms (U3).
  Precedence — not adjacency — is the operator that touched fine at all, so the post-U3 re-probe is
  *precedence over the rich vocabulary*. This is the evidence behind "fine style is U3-gated."

## 5. Build Units (dependency-ordered)

### Foundation — makes everything below legal-and-censored by construction

- **U1 [BUILT] Ground the frames on Scale-0.** `datagen/candidate_frames.py` — `frame_candidates(engine,
  board)` runs the engine's Scale-0 (legal + full censor stack, reused verbatim via shared context) and
  renders each survivor with `render_move`, keeping the pure renderer engine-unaware. *Measure met:* every
  returned candidate is legal by construction (fail-loud guard); censored moves never reach the frame
  layer. *Finding:* on candidates the frame lens floods (`restrains` on 18/26) or vanishes (empty on
  25/29) — it is a sparse game-HIGHLIGHT lens (learning), not a discriminative ROUTING lens. Made the
  two-lens split (U5) concrete on real candidates rather than in theory.
- **U2 [NEXT] Unify the censor (the hard/soft bridge).** One censor store, bidirectional: near-misses
  mined from games (`negative_learning`) ↔ Genesis `cannot` frames; Genesis retraction/surprise events
  ↔ Yami negative examples. *Delivers:* a censor learned anywhere applies everywhere. *Measure:* a
  near-miss mined from a Tal game censors the same pattern in a Genesis frame-run, and vice versa.

### Vocabulary growth — the fine-style unblock (the two axes)

- **U3 Literature-mining pass (top-down, intensional).** Extract Human-Window rules **+ their
  break-conditions** from chess literature — canonical named frames (*My System*: prophylaxis,
  overprotection, blockade, outpost, restraint) and match narratives ("slow, then broke through" = the
  arc in prose, Genesis's home turf). Pipeline: `STRUCTURED_WEB_KNOWLEDGE_EXTRACTION` (search → extract →
  structure, **LLM-as-parser-not-source**, grounded). *Grade every extracted rule by the Human-Window
  criterion:* can a human state, follow, recognize it? *Delivers:* the intensional strategic vocabulary
  the histogram can't derive. *Measure:* the clustering test's fine-match climbs above chance.
  - **[BUILT — increment 1: knowledge acquisition]** `scripts/u3_mine_literature.py` (MediaWiki curl +
    local Ollama qwen3.5:35b, `think:false`, `format:"json"`, whitespace-normalized + ≥40-char-chunk
    grounding gate, glossary window-focus, crash-safe JSONL). Output `data/genesis_roster/u3_atoms.jsonl`:
    **12/12 canonical frames grounded, 8 with break-conditions** (~5-6 genuinely useful: IQP→endgame,
    passed-pawn→rook-behind, fianchetto→bishop-exchange). Confirmed U3's real yield is the
    **break-conditions**, not the core frames (Yami already had those).
  - **[TRIED, NEGATIVE — increment 2a: break-condition as emitted TOKEN]** Wired the IQP→endgame atom as
    a signal-guard (`src/yami/datagen/break_guards.py`) emitting `exploits weakness` on
    endgame × opponent-structural-weakness × mobility-pressure (all real signals). Re-rendered the roster
    (`roster_stories_v2.json`) and re-ran the thermometer: **it DEGRADED** — precedence coarse 61→28%,
    fine 1/19→0/19; marginal unchanged at 78%. *Why:* the token tracks **game geometry, not style** —
    Polgar (sharp) emits **505**, Capablanca (technician) **88**, inverting the hypothesis that conversion
    is a technician trait; and firing hundreds of times it re-created the `restrains` flooding pathology.
    Renderer reverted to validated v1; `break_guards.py` + v2 data kept for the post-mortem.
  - **[PROVEN — increment 2b mechanism]** `~/Genesis/gsebridge/rules/chess_breaks.rules` (registered) +
    `scripts/u3_surprise_proof.py`: a break-condition as EXPECTATION (`if x promotes pawn then pawn becomes
    strong`) + `cannot` censor (`if y blocks pawn then pawn cannot become strong`) fires through the JVM;
    the violation retracts the naive reading; Java over-retracts 8, `retraction.py` isolates the **1 real
    surprise** (`pawn → strong`). End-to-end on built machinery. *Gotchas found:* GSE drops unknown object
    nouns ("passer" → empty object — use lexicon-known `pawn`); and **Java's retraction only fires when
    expectation and violation are ADJACENT** (1 intervening sentence → 0) — a blocker for real games.
  - **[SUPERSEDED — trace-surgery]** An earlier plan reconstructed violations at the Regenesis trace
    boundary to beat Java's adjacency window. The SSL derivation shows that is fixing the wrong thing: the
    adjacency failure is §4.1's "sub-stories read in isolation," and the fix is the CONTROLLER, not surgery.
  - **[NEXT — increment 2b: the guarded meta-narrative controller (multi-chained frames)]** Per SSL §7.2
    ("multi-chained frames = the controller") + §4.3 (the falsifiability guard). Each style-law is chained
    with its break-conditions into ONE arc-spanning frame (the shape brilliancy already has — a conjunctive
    rule that binds expectation and violation across the game, same referent), so the global frame's priors
    persist across sub-reads and the surprise binds *whenever* the violation lands, not only on adjacency.
    This single build discharges THREE units at once: the style-router (U6), the **falsifiability guard**
    that keeps σ_sem > 0 (break-conditions = §4.3, not vocabulary), and the fine-style channel. Its
    retractions read bottom-up are the learning signal (§4.1). *Then* fire the roster (via
    `understand_batch`, NOT a naive `understand_file` loop — it wedges at ~40 fires), extract per-player
    surprise profiles (rate × magnitude × polarity), feed the precedence thermometer. *Measure:* fine-match
    climbs; and the guard holds (a style can be refuted by the position — no Quixote).
  - **[SUPERSEDED — increment 2b framing]** The reframe
    the negative result points to (and the Genesis-native one): a break-condition is an **expectation**,
    and its **violation is a surprise** — a retraction event, the SAME mechanism as brilliancy=surprise
    (`~/Projects/Regenesis/regenesis/retraction.py`, already built). The discriminative signal is NOT that
    the guarded state occurred (occurrence tracks *geometry* — the increment-2a failure) but the player's
    **surprise profile**: rate × *magnitude* × which rules they break. A swindler manufactures surprise;
    a technician minimizes it; Tal's are high-magnitude; Petrosian's prophylaxis is surprise-avoidance.
    **"Quality of the surprise" (the retraction magnitude/cascade) IS the signal — not a data-quality
    filter.** The break-conditions that matter are exactly those sharp enough to *be* violated (Tarrasch's
    rook-behind-passer can break; "but there are exceptions" cannot) — so the "filter" and the signal are
    one. Route increment 2 through the surprise/retraction channel, not an occurrence-token. *Measure:*
    the surprise profile feeds the precedence thermometer; does fine-match climb?
- **U4 [BUILT, keep running] Frame-of-frames induction (bottom-up).** Learn the frames the books DON'T
  state, from games (`corpus_learning.py`). Complementary to U3. *Measure:* new recurring candidates
  promoted as corpora grow.

### Style & routing — needs the vocabulary (U3)

- **U5 [NEXT] Formalize the two-lens split.** STYLE lens = action-frequency profile (symbolic, cheap,
  no Genesis fire) for routing/clustering; FRAME lens = Genesis-derived frames for the learning loop.
  *Measure:* style clustering uses the action lens (78%+ coarse), learning uses the frame lens.
- **U6 [DESIGNED + PROTO 2026-07-14 — the router is a PORT of ModelAtlas navigate].** Read the position →
  candidate moves/frames + their significance (the read) → **navigate**: score = significance × bank-alignment
  (multiplicative), banks steered by direction. Two banks answer "learn from both AND learn what wins":
  **significance** (how brilliant/load-bearing) and **outcome** (signed). The direction knob = the two modes —
  `outcome=0` learns from both sides' brilliance, `outcome=+1` prefers the winner (opposed decays hyperbolically);
  ELO is another bank, style/regime = IDF-weighted anchors. Proven on the real Karpov read
  (`scripts/navigate_moves.py`): learning tops with the loser's brilliancy, deciding with the winner's — the
  equal-and-opposite ranking, auditable. **No longer U3-blocked** — the 19 deep style-arcs (via the Regenesis
  MCP) are the style-relevant vocabulary, and ModelAtlas is the routing mechanism ready-made. *Measure:*
  Stockfish (U7). See `docs/GENESIS_CHESS_ARCHITECTURE.md` §14.7.

### Fitness — needs routing (U6)

- **U7 [BLOCKED: U6] Stockfish fitness loop.** Play vs Stockfish at ELO X → render Yami's own game →
  match to nearest style-lens → log (style-match, ELO, result). Aggregate: which style-routings WIN at
  high ELO? Makes the SoM *trainable* — RL over the style-laws with Stockfish as the environment.
  A novel routing that wins where no single master's style did = a genuinely new *effective* style,
  validated by Stockfish, not opinion. *Uses Yami's existing benchmark infra.* *Measure:* win-rate vs
  ELO by routing policy.

### Research — downstream, uses the built system

- **U8 [RESEARCH] The style-history graph.** Antagonism: is a style a *reply* to another (its distinctive
  frames = the anti-frames of another's dominant frames)? → a who-answers-whom directed graph. Era-drift:
  do post-engine masters occupy a distinct region / lower variance (homogenization)? New styles: empty
  regions of lens-space; frames present in no single master. Surprise-engineering: given a master's
  style-law, find moves in its *null space* (the frames they're blind to) — mechanized surprise.

### The Play-Learn loop — from understanding to WINNING (the "taste the dish" axis)

- **U9 [PARTIAL] Prescription: narrative → best moves, tactically grounded.** `evaluate_position →
  suggest_plan` (frame) → `rank_moves`, over a **tactical baseline** (SEE + no mate-in-1) UNDER the
  **outcome prescription** (`gm_patterns` win_rate = "make this move because it WON"). Tasted by full games
  to completion vs Stockfish (`scripts/prescribe_taste.py`; valid positions, terminate on mate/draw).
  *Proven:* the story engine plays coherent chess and draws Skill-0 with prescription OFF (understanding
  alone). *Measure:* game result vs Stockfish (win/draw/loss), longevity, estimated ELO.
- **U10 [NEXT] Negative learning — learn from LOSS.** Play → lose → `mine_negative_examples_from_evals`
  (Stockfish evals locate the turning point = the surprise) → *story of the loss* (frame-of-frames, the
  winner-read inverted) → `StrategyCensorRule` auto-written (Socratic) → `filter_moves`. `negative_learning.py`
  (`LearnedCensorStack`) is the built censor store. *Proven:* turning-point detection on a Capablanca loss
  (+273 → −281 located). Order: historical losses → then Yami vs Stockfish.
  - **[BUILT — the per-style negative lens from the masters' losses, 2026-07-13]** A loss is a win-story the
    result BREAKS — the mirror of brilliancy (there the continuation retracts the naive `fallen`→`brilliant`;
    here the outcome retracts the naive `dominant`→the tragic collapse). NOT a separate "downfall" genre — the
    SAME win universe, read from the loser's side (an early wrong turn was a hand-authored downfall universe;
    dropped). Pieces: (1) **outcome-integration** — `game_renderer.render_game(append_outcome=True)` states the
    result (`Opponent defeats Capablanca`), and `chess.rules` gains the single-antecedent forward pair
    `defeats→victorious`/`defeats→defeated` (distance-robust; a `cannot` retraction was tried and dropped — it
    fires only on Java adjacency, 2/3). The surprise is read at the ANALYSIS layer as the CO-PRESENCE of
    `Domination` and `Tragedy` on the master (`chess_breaks §8`) — fires **3/3**, `Tragedy('master')` now
    recognized. (2) **the negative lens** `scripts/surprise_lens.py` (+ corpora
    `data/genesis_roster/pgn_losses|pgn_draws/`, ~50/master via `fetch_365.py` result-filter) — the WIN
    thermometer's method ported to losses: the ACTION stream (not derived outcomes) + the `+1e-6` IDF-floor
    (zero the universals) + LOO-NN. *Measured (`player_surprise_lenses.json`):* **loss-style 67% coarse / 21%
    fine** (win baseline 78%/0%; chance 50%/8%) — the port discriminates, weaker than wins as expected (GM
    mistakes are subtle). The explicit loss−win *mistake* deviation is at chance (50% coarse) — the subtle
    error is not a different action pattern (U3-gated). Surprise-*localization* (DAG support of the refuted
    frame, `retraction.build_dag`) is **Stage-E-gated**: `dominant` has 0 reconstructable support (blank `via`,
    multiple producers = the Pillar-2 provenance collapse). JVM leak fixed at source (`corpus_learning.analyze_corpus`
    now `close()`s its per-call JVM). *The payoff is downstream — this lens feeds the learning loop / SoM, not a
    perfect number here.*
- **U11 [NEXT] ELO-BRACKETED learning — the best strategy is not a monolith.** Prescription + censors keyed
  on **(narrative, ELO-bracket)**; the routed frame must MATCH the opponent's ELO (a too-strong frame
  censors perception of a weak opponent's real threats — the "GM loses in 7" frame-failure). `gm_patterns`
  has `avg_elo`; `opponent_profile.py` estimates skill; `benchmark_elo.py` plays bracketed levels. *Measure:*
  Yami climbs the ELO ladder bracket by bracket as it learns; each bracket has its own learned rules.

## 6. Open Debts

- **[DEBT] Retraction cascade over-retraction.** Fixed at the trace boundary (`retraction.py`); the
  proper fix belongs in the Regenesis port (provenance-native), not a Java patch.
- **[DEBT] PV encoding bug.** `generate_som_from_lichess.py:152,176` — `best_pv` from candidate 0 walked
  after the `best_idx` move (a pre-existing data-corruption bug from the old neural pipeline).
- **[DEBT] Fine-grained style ceiling.** Unblocked only by U3 (literature). The histogram is at its limit.
- **[DEBT] Data hygiene.** Caruana's Chess960/Freestyle games are correctly dropped at parse; the roster
  fetch should tag and skip variants earlier.

## 7. The Metric Discipline (nothing self-certifies)

- **Human-Window criterion** — quality gate for every frame: statable, followable, recognizable by a human.
- **Clustering test** (`action_profile.py`) — the measure of vocabulary style-resolving power (U3's metric).
- **Stockfish** — the exogenous fitness for routing effectiveness (U7's metric).
- **The retraction/surprise log** — the learning signal (which frames get refuted, when to break a rule).

Every layer's warrant is outside the layer. That is the whole discipline: legal by the kernel, censored
by near-miss, framed by the literature, routed by regime, judged by Stockfish — and intelligible throughout.

---

*Scope v1 — 2026-07-11. Companion to `GENESIS_CHESS_ARCHITECTURE.md` (the built architecture). This is
the forward plan; that is the current state.*
