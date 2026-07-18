"""Per-master SURPRISE PROFILE from LOSSES — the negative lens, done right (the surprise channel).

A loss is a win-story whose ascendancy the result breaks. Fire each of a master's losses through the
EXISTING win universe with the outcome stated (append_outcome=True); the master's win-frame fires
(`Domination`/`dominant`) and the outcome refutes it (`Tragedy`/`defeated`). The SURPRISE is the
CO-PRESENCE of the two on the master (distance-robust, chess_breaks §8 — read here at the analysis layer,
not via the adjacency-fragile Java retraction). The PROFILE is what discriminates style (U3-2b):
  rate       — how often the tragic arc fires (did the win-frame get refuted?)
  which      — which win-frame broke (Domination vs Ascendancy vs a learned frame)
  via / mistake — the master's ACTIONS in the refuted games = their characteristic mistake signature
  magnitude  — how high they rose before the fall (win-frame + aggression height)

Reusable: `python profile_surprise.py [MASTER]` (default Capablanca). Verify on one, then scale to all.
Owns its JVM lifecycle (closes it — no leak)."""
import sys
sys.path[:0] = ["/Users/rohanvinaik/Genesis/mcp", "/Users/rohanvinaik/Projects/Regenesis",
                "/Users/rohanvinaik/Projects/Yami/src"]

import json
from collections import Counter
from pathlib import Path

import chess.pgn
import genesis_access as ga
from regenesis.retraction import parse_fact
from yami.datagen.game_renderer import render_game

LOSSES = Path("/Users/rohanvinaik/Projects/Yami/data/genesis_roster/pgn_losses")
WIN_PATTERNS = {"Domination", "Ascendancy"}          # the rising-action arc (chess_plan.concepts)
FALL_PATTERNS = {"Tragedy", "Disaster"}              # the fall (war archetypes, on the master)
WIN_STATES = {"dominant", "ascendant", "prevailing"}
FALL_STATES = {"defeated", "ruined", "fallen"}       # outcome-refutation states on the master
# The master's ACTIONS (owned verbs) — what they DID; the mistake is which of these preceded the fall.
ACTION_VERBS = {"storm", "penetrate", "sacrifice", "counter", "initiate", "pressure", "infiltrate",
                "restrain", "control", "gain", "obtain", "isolate", "develop", "occupy", "advance"}


def _fire_losses(key: str, jvm, srv, limit: int | None = None):
    pgn = LOSSES / f"roster_{key}.pgn"
    per_game = []
    with pgn.open() as f:
        while (g := chess.pgn.read_game(f)) is not None:
            if limit and len(per_game) >= limit:
                break
            ws = g.headers.get("White", "").split(",")[0].strip()
            white, black = (ws, "Opponent") if key.lower() in ws.lower() else ("Opponent", key)
            story = " ".join(render_game(g, white=white, black=black, append_outcome=True))
            path = srv._persist_story(story, srv._slug(story))
            out = jvm.call({"cmd": "understand_file", "path": str(path)}, timeout=120)
            art = dict((out or {}).get("artifact") or {})
            trace = art.get("trace", [])
            pats = {n for e in trace if e.get("kind") == "pattern"
                    and (n := e.get("name")) and str(e.get("subject", "")).lower() == key.lower()}
            verbs, states = Counter(), set()
            for e in trace:
                if e.get("kind") != "derive":
                    continue
                pf = parse_fact(e.get("fact", ""))
                if pf and pf.subject.lower() == key.lower():
                    if pf.verb == "become":
                        states.add(pf.object)
                    elif pf.verb in ACTION_VERBS:
                        verbs[pf.verb] += 1
            per_game.append({"patterns": pats, "verbs": verbs, "states": states})
            jvm.fires += 1
            if jvm.fires % ga.FIRE_CHUNK == 0:
                jvm.cycle()
    return per_game


def profile(key: str, limit: int | None = None) -> dict:
    srv = ga._server_module()
    jvm = ga.Jvm(universe=None, universe_only=False, dual_projection=False)
    try:
        games = _fire_losses(key, jvm, srv, limit)
    finally:
        jvm.close()

    n = len(games)
    surprises, win_which, mistake, magnitude = [], Counter(), Counter(), []
    for g in games:
        won = (g["patterns"] & WIN_PATTERNS) | {s for s in g["states"] if s in WIN_STATES}
        fell = (g["patterns"] & FALL_PATTERNS) | {s for s in g["states"] if s in FALL_STATES}
        is_surprise = bool(won) and bool(fell)
        surprises.append(is_surprise)
        if is_surprise:
            for w in won:
                win_which[w] += 1
            mistake.update(g["verbs"])                       # the actions that preceded the refuted frame
            magnitude.append(sum(g["verbs"].values()) + len(won))   # height risen before the fall
    rate = sum(surprises) / n if n else 0.0
    return {
        "master": key, "n_losses": n,
        "surprise_rate": round(rate, 3),
        "win_frames_refuted": dict(win_which.most_common()),
        "mistake_signature": dict(mistake.most_common(8)),   # characteristic actions in refuted games
        "avg_magnitude": round(sum(magnitude) / len(magnitude), 2) if magnitude else 0.0,
    }


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "Capablanca"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    prof = profile(key, limit)
    print(json.dumps(prof, indent=2))


if __name__ == "__main__":
    main()
