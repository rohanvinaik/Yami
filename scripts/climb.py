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


def climb(skills, games_per, max_plies):
    sf = chess.engine.SimpleEngine.popen_uci(SF)
    limit = chess.engine.Limit(time=0.08)
    overlay = RiskOverlay()
    traps = []                                          # (fen, plan, elo, loser) across all brackets
    for skill in skills:
        elo = SKILL_TO_ELO[skill]
        sf.configure({"Threads": 2, "Skill Level": skill})
        wdl = {"win": 0, "draw": 0, "loss": 0}
        before = len(overlay.entries)
        for i in range(games_per):
            g = play(sf, limit, elo, i % 2 == 0, overlay, max_plies)
            wdl[g["result"]] += 1
            if g["result"] == "loss":
                for ne, plan in mine(sf, g):
                    overlay.learn(plan, elo, ne.move_uci, ne.nav_vector, ne.eval_drop)
                    traps.append((ne.fen, plan, elo, chess.WHITE if g["us_white"] else chess.BLACK))
        print(f"bracket Skill {skill} (~{elo} ELO): {wdl['win']}W {wdl['draw']}D {wdl['loss']}L · "
              f"+{len(overlay.entries) - before} censors (total {len(overlay.entries)})", flush=True)

    # System-2 working, ELO-stratified: at each learned trap, re-prioritize to a SAFER move (recall by regime)
    print("\nTARGETED — recognition+overlay re-prioritizes off each learned trap (recall by regime+resemblance):", flush=True)
    safer = changed = 0
    for fen, plan, elo, loser in traps[:8]:
        b = chess.Board(fen)
        tpl = _TEMPLATE.get(plan, _TEMPLATE["IMPROVE_PIECES"])
        theory = pick(b, elo, None, plan=tpl)
        learned = pick(b, elo, overlay, plan=tpl)
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
    print(f"=== CLIMB — brackets(skill) {skills}, {games_per} games each, cap {max_plies} plies ===", flush=True)
    climb(skills, games_per, max_plies)


if __name__ == "__main__":
    main()
