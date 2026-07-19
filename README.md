# Yami 闇

**Read the chess game as a story, and recognize the move — instead of searching a tree for it.**

<p align="center">
  <a href="https://github.com/rohanvinaik/Yami/actions"><img src="https://github.com/rohanvinaik/Yami/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-3367d6.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-3367d6.svg" alt="Python 3.11+"></a>
</p>

`chess as a Genesis story · recognition, not search · every rule a human could state · deterministic infrastructure + a tiny recognition residual`

A chess game is a story. Yami renders it as one — plain who-does-what events, with no chess judgment of its own — and hands it to Genesis, Patrick Winston's story-understanding engine, revived at scale on GSE. Here is the Opera Game, read move by move:

```
$ yami reads the Opera Game (Morphy, 1858)

  Morphy develops piece.   Duke develops piece.
  Morphy captures pawn.    Duke captures knight.
  Morphy captures bishop.  Duke exposes king.
  …
  Morphy coordinates rooks.
  Morphy captures knight.  Duke captures rook.
  Morphy forks piece.      Morphy checks king.
  Duke captures queen.       ← Morphy gave the queen away
  Morphy checkmates king.    ← the loss was the plan
```

41 event-sentences, three layers of frames fired, ending — correctly — on *Morphy checkmates king*. Nothing here was generated. The renderer states what happened; Genesis derives what it means, and it carries the rule behind every inference. The queen was not blundered. It was spent, and the surprise that the loss was the plan is exactly what the engine reads as brilliancy.

---

## The robot chef

Hold one picture before any of the machinery. The hardest jobs to automate are not the big-think ones — run a company, do conceptual math. They are jobs like head chef in a working kitchen. The line cook watching your brown sauce gets distracted; the sauce over-reduces. Thirty seconds for a human to fix. But the kitchen is a coupled system under two scarcities at once, time and space, and every local fix spends a shared resource and displaces a coupled task. Thin the sauce, and you cost burner-minutes; the fish that needed the burner slips; the veg steaming for the fish overcooks; you park it, but now the shared steamer is occupied and the pastry station is under floppy carrots. A thirty-second lapse propagates along the coupling until it reaches something you cannot take back.

The skill that resolves that is not computation. It is understanding the load-bearing decision points: the temporal constraints (the sauce is on a clock), the resource constraints (the shared steamer), and the acceptable losses (the carrots are cheap — sacrifice them to save the dish). What you need is a system that can tell the *story of the kitchen's operation*.

Chess is the kitchen, term for term. Time scarcity is **tempo** — the passer is on a clock, the attack must land before the defense consolidates. Space scarcity is **the board** — two pieces cannot hold one square. Acceptable losses are **the sacrifice** — give up the exchange to keep the initiative, and brilliancy is the surprise that reveals the ruined carrots were always meant to be blended. The cascade to fire is **the collapse** — one tempo lost, one weakness fixed, a won game falling apart move by move. So Yami does not simulate every kitchen-state. It recognizes the two things that are load-bearing *right now*.

## Recognition, not search

Most of chess is not hard. Legal moves, tactical motifs, opening theory, endgame tablebases — all deterministic, all microsecond-cost, all handled by infrastructure. The scaling paradigm spends trillions of parameters on work that does not need them. Yami spends none of them there. It decomposes the position into layers, each of which does its cheap work deterministically and eliminates candidates, until roughly thirty legal moves have collapsed to **three to five annotated candidates**. What is left is not a search problem. It is a recognition problem, small enough for a tiny model — the residual the infrastructure could not resolve on its own.

This is read out of a learning theory, not chosen for taste. A domain is a closed artifact inside an open field, and only the artifact is specifiable. Chess's open field is perfect play over the 10⁴⁴ game tree — intractable, and never claimed. The closed artifact is legal moves, tactics, endgames, and the named-frame vocabulary — finite, and derivable.

## Every layer is a rule a human could state

Nothing in Yami is an opaque weight. The whole stack is intensional — rules a person could state, follow, and recognize, the **Human Window** of Michie and Kopec — because that intelligibility is what lets each layer carry an *external* warrant instead of certifying itself:

```
LEGAL      the hard kernel — python-chess is the judge
CENSORS    near-miss "don't" — blunders made UNrepresentable, not penalized
FRAMES     strategic heuristics — intensional rules, from literature and from induction
ROUTING    style-laws — a Society-of-Mind picks the style per move, regime by regime
WARRANT    Stockfish — the exogenous judge of which routing actually wins
```

The censor layer holds a unification worth naming: *near-miss is one mechanism across four systems* — Yami's `negative_learning`, the Genesis retraction that refutes a belief when a naive read almost held, the hard censors of `law_as_architecture`, and the "break the rule at square 11" of the Knight's Tour. A censor learned on one side flows to the others.

Two axes run over the stack. **Learning grows the library** — the frame-of-frames driver induces frames from games bottom-up and extracts them from the literature top-down, and it can discover not just tactics but entirely new strategies. **Routing applies the library** — the Society-of-Mind selects the style-law for the move in front of it, recalled by resemblance and stratified by regime.

## It learned to play the player, not the rules

Yami has a working System-1 → System-2 loop. System 1 recognizes a plan and plays it; System 2 mines *every* game it plays — not just its losses — scoring each of its own moves against the judge and banking it by quality: a bad move becomes a **censor** to avoid, a near-best move an **exemplar** to repeat, each recalled later by resemblance (a Kanerva hamming lookup over navigation vectors) and stratified by the opponent's strength. Its move choice is one line — **theory − censor_penalty + exemplar_reward** — learned from what loses *and* what works, against this kind of opponent. A draw teaches as much as a defeat.

That one line produced strategy no one wrote. About ten games in, Yami drew a game against Stockfish, and the way it drew is the tell. It traded down to an equal position and held it. In a roughly equal spot, every committal move its history has punished gets penalized, so what survives on top is the SEE-safe, non-destabilizing move — and when every move that could spoil the position has been tagged "this kind of move lost," what is left is a rook, bounced back and forth. "Refuse to move into a worse position" is not the absence of a plan. It is the censor mechanism firing. Yami sat on the equal position and made Stockfish commit first (19…Bb4+); only then did it engage, and it navigated the complications to the draw.

"Don't overextend, make the stronger side commit, hold the equal position" is a real, named strategy against a stronger opponent. Nobody handed it to Yami. It fell out of what the system learned *loses*. That is the whole thesis in one game. It did not learn chess as a set of rules. It learned to play the *player* — the way the chef reads the kitchen instead of reciting the recipe.

## Brilliancy is surprise

A brilliant move is one that violates the frame you were reading and turns out to be right — a belief you held, refuted, and revised. Yami reads that with Genesis's trace-boundary retraction (revived Python-native in [Regenesis](https://github.com/rohanvinaik/Regenesis), 12 tests): a naive read expects the material loss to be a mistake, the continuation refutes it, and the surprise of the revision *is* the brilliancy. It is the exchange sac seen as the carrots that were always meant to be blended.

## Lineage and status

Yami is Genesis-native: the tactical, positional, and rising-action frame universes are authored Genesis rules; the deterministic renderer turns a PGN into clean event-sentences with zero neural judgment; brilliancy and cross-game learning run on Regenesis. Built and verified so far: the renderer, the three frame layers, the Scale-0 candidate seam, live recognition-to-move play, the System-1→System-2 play-and-learn loop, resemblance recall, ELO-stratified climbing, dual good-and-bad per-move learning, and a 19-master roster to learn from. Style routing and the frame library are the active build. The full architecture is in [`docs/GENESIS_CHESS_ARCHITECTURE.md`](docs/GENESIS_CHESS_ARCHITECTURE.md); the dependency-ordered plan is in [`docs/GENESIS_CHESS_SCOPE.md`](docs/GENESIS_CHESS_SCOPE.md).

---

MIT — Rohan Vinaik. A chess game is a story; Yami reads it on [Genesis](https://github.com/rohanvinaik/Regenesis).
