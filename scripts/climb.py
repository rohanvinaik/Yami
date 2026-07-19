"""U11 step 2 — ELO-BRACKETED CLIMBING: accumulate near-miss censors per regime, bracket by bracket.

The blueprint (`law_as_architecture.md` §8): a coefficient means one thing in a regime, another across one —
in chess the REGIME = ELO. So the near-miss censors are keyed per ELO bracket (`RiskOverlay` already filters
`e == elo`, so a trap learned vs a 1320 does NOT fire vs a 1700 — regime SEPARATION). Climbing = play a bracket,
mine Yami's own losses into that bracket's censors, and rise: "what wins vs a 1500 differs from vs a 2800."

Reuses the built loop verbatim (`play_learn.play`/`mine` — recognition names the plan, `pick` selects, System 2
learns), the resemblance overlay (`learn_loop_v2.RiskOverlay`), and the calibrated bracket map (`benchmark_elo`).

Run:  PYTHONPATH=src python3 scripts/climb.py "0,2" 2 80    # brackets(skill), games/bracket, ply-cap
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path.home() / "Projects" / "Regenesis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chess
import chess.engine
from benchmark_elo import SKILL_TO_ELO
from learn_loop_v2 import SF, RiskOverlay, pick, sf_eval_after
from play_learn import _TEMPLATE, mine, play


TRAJECTORY = "data/trajectory.jsonl"
GAMES_PGN = "data/yami_games.pgn"


def _pgn(g, elo, n):
    """Yami's actual game as PGN movetext — so we can SEE what it played, and load it in any chess viewer."""
    res = {"win": "1-0" if g["us_white"] else "0-1", "loss": "0-1" if g["us_white"] else "1-0",
           "draw": "1/2-1/2"}[g["result"]]
    w, b = ("Yami", "Stockfish") if g["us_white"] else ("Stockfish", "Yami")
    hdr = f'[Event "Yami learning ~{elo}"]\n[Round "{n}"]\n[White "{w}"]\n[Black "{b}"]\n[Result "{res}"]\n\n'
    body = " ".join((f"{i // 2 + 1}." if i % 2 == 0 else "") + s for i, s in enumerate(g.get("san", [])))
    return hdr + body + f" {res}\n\n"


def _log_trajectory(elo, g, library):
    """Append one game to the learning-trajectory log: survival (plies) vs accumulated experience (library).
    The SSL learning curve made concrete — as constraints (censors) accrue, the σ_sem descent should run
    LONGER before the collapse (loss), then hold to a draw. One JSONL line per game, accumulating across runs."""
    import json
    with open(TRAJECTORY, "a") as f:
        f.write(json.dumps({"elo": elo, "result": g["result"], "plies": g["plies"],
                            "recalled": g["recalled"], "library": library}) + "\n")


def climb(skills, games_per, max_plies, overlay_path=None):
    from pathlib import Path
    sf = chess.engine.SimpleEngine.popen_uci(SF)
    limit = chess.engine.Limit(time=0.08)
    reward_path = str(Path(overlay_path).with_name("exemplars.json")) if overlay_path else None
    overlay = RiskOverlay.load(overlay_path) if overlay_path else RiskOverlay()   # BAD moves (censors) IN
    reward = RiskOverlay.load(reward_path) if reward_path else RiskOverlay()       # GOOD moves (exemplars) IN
    if overlay_path:
        print(f"loaded {len(overlay.entries)} censors + {len(reward.entries)} exemplars from prior games", flush=True)
    traps = []                                          # (fen, plan, elo, loser) across all brackets
    game_n = 0
    longest = {"plies": -1}
    for skill in skills:
        elo = SKILL_TO_ELO[skill]
        sf.configure({"Threads": 2, "Skill Level": skill})
        wdl = {"win": 0, "draw": 0, "loss": 0}
        for i in range(games_per):
            game_n += 1
            g = play(sf, limit, elo, i % 2 == 0, overlay, max_plies, reward=reward)
            wdl[g["result"]] += 1
            bad, good = mine(sf, g)                      # §13C: EVERY game teaches, good + bad, per move
            for ne, plan in bad:
                overlay.learn(plan, elo, ne.move_uci, ne.nav_vector, ne.eval_drop)
                traps.append((ne.fen, plan, elo, chess.WHITE if g["us_white"] else chess.BLACK))
            for ne, plan in good:
                reward.learn(plan, elo, ne.move_uci, ne.nav_vector, ne.eval_drop)
            if overlay_path:
                overlay.save(overlay_path)              # persist per GAME (robust across a long grind)
                reward.save(reward_path)
            # LEARNING IN ACTION: `plies` = how far the game-story ran before the collapse (the σ_sem descent
            # extending as constraints accrue, SSL §2.3); `recalled` = moves a learned signal changed THIS game
            print(f"  game {game_n} (Skill {skill}~{elo}): {g['result']:4} · {g['plies']:>3} plies · "
                  f"{g['recalled']} moves changed by recall · +{len(bad)} censors +{len(good)} exemplars · "
                  f"lib {len(overlay.entries)}c/{len(reward.entries)}e",
                  flush=True)
            _log_trajectory(elo, g, len(overlay.entries))
            with open(GAMES_PGN, "a") as f:                 # save the game so we can review its play
                f.write(_pgn(g, elo, game_n))
            if g["plies"] > longest["plies"]:
                longest = {**g, "elo": elo, "n": game_n}
        print(f"bracket Skill {skill} (~{elo} ELO): {wdl['win']}W {wdl['draw']}D {wdl['loss']}L", flush=True)
    if longest["plies"] > 0:                                # SEE WHAT IT DID: its longest-surviving game
        print(f"\n--- Yami's longest game this run (game {longest['n']}, {longest['result']}, "
              f"{longest['plies']} plies) — {'White' if longest['us_white'] else 'Black'} ---", flush=True)
        print("  " + " ".join((f"{i // 2 + 1}." if i % 2 == 0 else "") + s
                               for i, s in enumerate(longest.get("san", []))), flush=True)
    if overlay_path:
        print(f"saved {len(overlay.entries)} censors → {overlay_path} · "
              f"{len(reward.entries)} exemplars → {reward_path}", flush=True)

    # System-2 working, ELO-stratified: at each learned trap, re-prioritize to a SAFER move (recall by regime)
    print("\nTARGETED — recognition+learned overlays re-prioritize off each trap (recall by regime+resemblance):", flush=True)
    safer = changed = 0
    for fen, plan, elo, loser in traps[:8]:
        b = chess.Board(fen)
        tpl = _TEMPLATE.get(plan, _TEMPLATE["IMPROVE_PIECES"])
        theory = pick(b, elo, None, plan=tpl)
        learned = pick(b, elo, overlay, plan=tpl, reward=reward)
        if theory != learned:
            changed += 1
            safer += sf_eval_after(sf, b, learned, loser) > sf_eval_after(sf, b, theory, loser)
    print(f"  re-prioritized {changed}/{len(traps[:8])}, {safer} to a Stockfish-SAFER move", flush=True)

    # REGIME SEPARATION: a censor learned in one bracket is SILENT in another (the ELO stratification)
    if traps:
        fen, plan, elo, _ = traps[0]
        b = chess.Board(fen)
        from yami.navigator import compute_navigation_vector
        nav = compute_navigation_vector(b).as_tuple()
        trap_uci = next((u for p, e, u, n, d in overlay.entries if e == elo), None)
        if trap_uci:
            here = overlay.penalty(plan, elo, trap_uci, nav)
            other = next((SKILL_TO_ELO[s] for s in SKILL_TO_ELO if SKILL_TO_ELO[s] != elo), elo + 400)
            cross = overlay.penalty(plan, other, trap_uci, nav)
            print(f"\nREGIME SEPARATION: trap {trap_uci} penalty at its regime (~{elo}) = {here:.2f}, "
                  f"at ~{other} = {cross:.2f}  {'✓ stratified' if cross == 0 else ''}", flush=True)
    sf.quit()


def main():
    skills = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "0,2").split(",")]
    games_per = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    max_plies = int(sys.argv[3]) if len(sys.argv) > 3 else 80
    overlay_path = sys.argv[4] if len(sys.argv) > 4 else None      # persistent censor library (learn across runs)
    print(f"=== CLIMB — brackets(skill) {skills}, {games_per} games each, cap {max_plies} plies ===", flush=True)
    climb(skills, games_per, max_plies, overlay_path)


if __name__ == "__main__":
    main()
