"""Build the roster-wide LENS STORE — the understanding of how each master takes the basic rules of chess
and actually plays well. For every player: render their games → fire each through the curated KB
(`analyze_corpus`, universe=None = --auto, the proper path) → persist the frequency lens (their derived
mechanics) + the induction residual + whatever tier-2 architecture derives against the field so far.

This is the LEARNING axis's substrate: style = a player's lens's deviation from the field (§ the yardstick).
Crash-safe and resumable — a player already in the store is skipped, so a kill mid-run loses nothing.

NOTE (order): distinctive/architecture are computed vs whatever lenses are already in the store when a
player fires, so early players see a partial field. The persisted LENS itself (the frequency profile) is
order-INDEPENDENT and complete; full-field distinctive/architecture is a cheap re-derive pass over the
finished store (no re-firing of games) — a separate step, not folded in here.
"""
import sys

sys.path.insert(0, "/Users/rohanvinaik/Genesis/mcp")
sys.path.insert(0, "/Users/rohanvinaik/Projects/Regenesis")

import json
from pathlib import Path

import chess.pgn
from regenesis.corpus_learning import analyze_corpus, load_lens_store
from yami.datagen.game_renderer import render_game

ROSTER = Path("/Users/rohanvinaik/Projects/Yami/data/genesis_roster")
LENS = str(ROSTER / "player_lenses_v3.json")
KEYS = list(json.loads((ROSTER / "roster.json").read_text()))   # 19 canonical keys


def render_player(key: str) -> list[str]:
    """Roster-convention render: roster player = their surname, opponent = 'Opponent'."""
    out: list[str] = []
    with (ROSTER / "pgn" / f"roster_{key}.pgn").open() as f:
        while (g := chess.pgn.read_game(f)) is not None:
            ws = g.headers.get("White", "").split(",")[0].strip()
            white, black = (ws, "Opponent") if key.lower() in ws.lower() else ("Opponent", key)
            sents = render_game(g, white=white, black=black)
            if sents:
                out.append(" ".join(sents))
    return out


def main() -> None:
    for key in KEYS:
        if key in load_lens_store(LENS):
            print(f"  skip (in store): {key}", flush=True)
            continue
        stories = render_player(key)
        res = analyze_corpus(stories, key, lens_store=LENS)
        arch = res.get("architecture", {}).get("derivations", [])
        resid = res.get("induction_residual", {})
        print(f"  {key:14s} {res['n_games']:3d} games · "
              f"{len(resid):2d} residual frames · {len(arch):2d} tier-2 derivations", flush=True)
    print(f"\nlens store: {LENS}")
    store = load_lens_store(LENS)
    print(f"players in store: {len(store)}  ({', '.join(sorted(store))})")


if __name__ == "__main__":
    main()
