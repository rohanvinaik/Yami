# Yami 闇

**Watch a chess game and know *why* a move was brilliant — read as a story, at near-zero compute.**

`No minimax · no neural net in the answer path · no JVM · every layer a rule a human could state`

The hardest jobs to automate are not the cerebral ones — run a company, prove a theorem. They are jobs like **head chef in a working kitchen.** A line cook gets distracted and your sauce over-reduces: thirty seconds to fix, for a human. But a kitchen is a coupled system under two scarcities at once, time and space, and every local fix *spends a shared resource* and *displaces a coupled task*. Thin the sauce — costs burner-minutes — the fish that needed the burner slips — the vegetables steaming for it overcook — park them, they're cheap — but now the shared steamer is occupied and the pastry station is buried in floppy carrots, and while you smooth that over the sauce has caught fire. A thirty-second perturbation propagates down the couplings until it reaches something irreversible. The skill that stops it is not computation. It is knowing which two or three decisions are **load-bearing right now** — what is on a clock, what resource is contested, what is cheap enough to sacrifice. You need something that can tell the *story* of the kitchen.

Chess is that kitchen, term for term. Time is **tempo**; space is **the board**; the acceptable loss is **the sacrifice** — carrots-are-cheap *is* the exchange sac, and a brilliancy is the surprise that reveals the ruined carrots were always meant to be blended. The cascade to fire is **the collapse**, a won game coming apart move by move. So Yami does not search thirty moves. Most of chess is structural work — legality, tactics, endgames — that infrastructure handles trivially; what is left is a small **recognition** residual. Reduce the thirty legal moves to three or four annotated candidates, and hand the player a recognition problem, not a search problem.

## Reading the game as a story

The way you recognize is by reading the game as a story. **Genesis** — Patrick Winston's story-understanding engine, running Python-native on Regenesis — takes the moves as who-does-what events and derives what the game *implies but never states*: the tactics, the rising arc, the moment it turned. It scores every derivation by **surprise = −log P** — the improbability of the chain that reconstructs it — and that does automatically what hand-tuning strained toward. The move every player makes, *take material*, prices to zero information. The load-bearing move surfaces because its why-chain is the deepest and least likely. Read against a real Kasparov game, *"this was brilliant"* came out ranked **#1**, the causal chain sitting beneath it.

**Brilliancy falls straight out as belief revision.** A sacrifice reads as `fallen` the instant it is played — material given up — and the checkmate three moves later *retracts* that fear. The retraction is the surprise. A brilliancy is the move that makes you take back the fear it first provoked.

## One read, one knob

Understanding and winning are the same read seen two ways, and the difference is a single knob — `significance × outcome`, multiplicative, every rank auditable:

```
$ python scripts/navigate_moves.py

LEARNING — learn from BOTH, by how instructive    DECIDING — prefer what WINS, now
  1. Opponent: overwhelm king → brilliant  6.36     1. Karpov:   overwhelm king → brilliant  5.66
  2. Karpov:   overwhelm king → brilliant  5.66     2. Karpov:   hunt king                   3.87
  3. Opponent: hunt king                   5.66     3. Opponent: overwhelm king → brilliant  3.18

  LEARNING tops with the OPPONENT's brilliancy — the single most instructive lesson.
  DECIDING tops with KARPOV's winning move   — the one you actually play.
```

Same game, opposite tops, nothing hidden. And a win, a draw, and a loss are not three systems — they are one **outcome axis** with the draw as its zero-point. On a **win**, the winner's climax outranks the loser's. On a **loss**, the win-arc the result refuted is dampened to an `[illusion]` when you are deciding — but marked `[SURPRISE]` when you are learning, because *dominant-then-fell* is exactly the lesson. On a **draw**, the read stays `[neutral]`: the hold is the achievement, and against a much stronger opponent, steering toward the draw *is* the win.

## Two directions, one currency

The **frame** is the shared token. One direction reads stories to grow laws — across many of a master's games, induct the style they play and, more valuably, map where the current frames go dark, the residual the books never state. The other applies laws to pick moves — read the position's regime, route the style-law that fits. Learning grows the law library; routing applies it; **Stockfish judges which routing wins.** Nineteen masters live in the library, each a deep causal arc to their own climax — Tal and Kasparov to the brilliant king-hunt, Petrosian to the strangle, Capablanca to the flawless conversion — and the engine writes the composed rules itself as it reads.

## The Human Window

Every layer is a rule a person could state out loud: **legal → censored → framed → routed → judged.** The one neural box is quarantined to the hard residual the rule layers could not resolve — never the frontier, only the leftover. And the tactical soundness a single read cannot own — significance zeroes material *by design*, so it reads brilliancy but not "don't hang the queen" — is supplied by a **separate experiential system** that learns from Yami's own losses: play, lose, locate the turning point (the surprise), carve a near-miss censor, recall it by resemblance. Recognition generates the plan; experience censors the move. Proven on one lost game: 38 plies taught **7** risk weights keyed to the recognized plan, and at every learned trap the loop re-prioritized off it — 5 for 5.

## Status — honest about the frontier

**Built and verified:** the frame engine (all three tiers fire on real games; brilliancy = surprise with the corrected retraction); the deterministic renderer; the 19-style library, discriminating; the win/draw/loss outcome axis; the navigate router (ported from ModelAtlas); the recognition→experience loop. With the prescription channel off, the story engine already plays coherent legal chess and **draws Stockfish from pure understanding**. **In progress:** mining break-conditions from the literature (the vocabulary that unlocks *fine* style — coarse style already resolves at 78%), then the corpus-scale frame-of-frames and ELO-bracketed climbing. The full plan lives in [`docs/GENESIS_CHESS_ARCHITECTURE.md`](docs/GENESIS_CHESS_ARCHITECTURE.md) and [`docs/GENESIS_CHESS_SCOPE.md`](docs/GENESIS_CHESS_SCOPE.md).

## Lineage

The same low-compute, comprehensible-all-the-way-down paradigm as its siblings — [Wayfinder](https://github.com/rohanvinaik/Wayfinder) proved 63% of Mathlib with 22M parameters on a laptop; Yami is the chess instance.

| Layer | System |
|---|---|
| Story understanding | Genesis × GSE — derive-or-abstain over event-sequences |
| Learning theory | Semantic Specification Learning — σ_sem, surprise, the field/artifact law |
| Reasoning over derivations | Regenesis — provenance-native belief revision |
| Chess infrastructure | Yami — legal moves, tactical scoping, positional ontology, the renderer |

## Quick start

```bash
git clone https://github.com/rohanvinaik/Yami.git && cd Yami
pip install -e ".[dev]"
python scripts/navigate_moves.py       # the read above, on real master games
```

## The name

闇 (yami) — darkness, the unseen. The understanding works in the dark, and hands you only the move and the reason.

---

Deep symbolic understanding at near-zero compute — a system that watches a game and tells you *this was brilliant, and here is the chain of why.* The one thing the scaling paradigm optimizes away.
