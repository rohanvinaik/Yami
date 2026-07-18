"""Re-render the 19-master roster through the current renderer → roster_stories_v2.json.

Reproduces the v1 labeling convention (roster player = their surname, opponent = "Opponent") so the ONLY
difference from the v1 baseline is whatever the renderer now emits — here, the U3 increment-2 break-guard
tokens. Feeds the composition thermometer for the v1-vs-v2 comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import chess.pgn

from yami.datagen.game_renderer import render_game

ROSTER = Path(__file__).resolve().parent.parent / "data" / "genesis_roster"
KEYS = list(json.loads((ROSTER / "roster.json").read_text()))   # 19 canonical keys


def roster_side(game: chess.pgn.Game, key: str) -> tuple[str, str]:
    """Label the roster player's side with their surname, the other side 'Opponent' (the v1 convention).
    The roster player is whichever header surname matches the key (case-insensitive)."""
    wsurname = game.headers.get("White", "").split(",")[0].strip()
    if key.lower() in wsurname.lower() or wsurname.lower() in key.lower():
        return wsurname or key, "Opponent"
    bsurname = game.headers.get("Black", "").split(",")[0].strip()
    return "Opponent", (bsurname or key)


def main() -> None:
    stories: dict[str, list[str]] = {}
    for key in KEYS:
        pgn = ROSTER / "pgn" / f"roster_{key}.pgn"
        if not pgn.exists():
            print(f"  MISSING pgn: {key}")
            continue
        games: list[str] = []
        with pgn.open() as f:
            while (g := chess.pgn.read_game(f)) is not None:
                white, black = roster_side(g, key)
                sents = render_game(g, white=white, black=black)
                if sents:
                    games.append(" ".join(sents))
        stories[key] = games
        guards = sum(s.count("exploits weakness") for s in games)
        print(f"  {key:14s} {len(games):3d} games, {guards:3d} break-guard tokens")
    out = ROSTER / "roster_stories_v2.json"
    out.write_text(json.dumps(stories))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
