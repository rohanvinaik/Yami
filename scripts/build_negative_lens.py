"""Build the roster-wide NEGATIVE LENS STORE — the mirror of the positive lens (build_lens_store.py),
pointed at each master's LOSSES instead of all their games. Literally the same machinery: render the lost
games → fire each through the curated KB (`analyze_corpus`, universe=None = --auto, which INCLUDES the
downfall/disaster/tragedy archetypes) → persist the failure lens (their derived FAILURE mechanics) + the
induction residual + the tier-2 FAILURE architecture derived against the loss-field so far.

The characteristic MISTAKE of a style = its loss-lens's deviation from its win-lens (player_lenses_v3.json)
— read post-hoc, the same way the positive lens's style = deviation from the field. This builder just
produces the loss-lens store; the loss-vs-win mistake read is a separate analysis pass (compare_lenses).

Crash-safe and resumable — a player already in the store is skipped. Losses are SEPARATE from draws (a draw
is often an intentional hold, a different story — the future equilibrium lens, not a failure)."""
import sys

sys.path.insert(0, "/Users/rohanvinaik/Genesis/mcp")
sys.path.insert(0, "/Users/rohanvinaik/Projects/Regenesis")

import json
from pathlib import Path

import chess.pgn
from regenesis.corpus_learning import analyze_corpus, load_lens_store
from yami.datagen.game_renderer import render_game

ROSTER = Path("/Users/rohanvinaik/Projects/Yami/data/genesis_roster")
LOSSES = ROSTER / "pgn_losses"
LENS = str(ROSTER / "player_negative_lenses.json")
KEYS = list(json.loads((ROSTER / "roster.json").read_text()))   # 19 canonical keys


def render_player_losses(key: str) -> list[str]:
    """Roster-convention render of a master's LOST games (player = surname, opponent = 'Opponent')."""
    pgn = LOSSES / f"roster_{key}.pgn"
    if not pgn.exists():
        return []
    out: list[str] = []
    with pgn.open() as f:
        while (g := chess.pgn.read_game(f)) is not None:
            ws = g.headers.get("White", "").split(",")[0].strip()
            white, black = (ws, "Opponent") if key.lower() in ws.lower() else ("Opponent", key)
            sents = render_game(g, white=white, black=black)
            if sents:
                out.append(" ".join(sents))
    return out


def main() -> None:
    store = load_lens_store(LENS)
    for key in KEYS:
        if key in store:
            print(f"  skip (in store): {key}", flush=True)
            continue
        stories = render_player_losses(key)
        if not stories:
            print(f"  THIN (no loss PGNs yet): {key}", flush=True)
            continue
        res = analyze_corpus(stories, key, lens_store=LENS)
        arch = res.get("architecture", {}).get("derivations", [])
        resid = res.get("induction_residual", {})
        print(f"  {key:14s} {res['n_games']:3d} losses · "
              f"{len(resid):2d} residual frames · {len(arch):2d} tier-2 failure derivations", flush=True)
    print(f"\nnegative lens store: {LENS}")
    store = load_lens_store(LENS)
    print(f"players in store: {len(store)}  ({', '.join(sorted(store))})")


if __name__ == "__main__":
    main()
