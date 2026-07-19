"""U11 — the play-and-learn loop: SYSTEM 1 (recognition) generates, SYSTEM 2 (near-miss censors) refines.

The corrected architecture (docs/GENESIS_CHESS_ARCHITECTURE.md §13A):
  · System 1 — render the game-so-far → understand → recognized_plan NAMES the plan (Stockfish JUDGES, never selects).
  · System 2 — the plan drives the proven learn_loop_v2 selector (theory − learned-risk + safety-seeking rethink);
    losses are mined for near-misses (Winston 1970) and learned into the RiskOverlay, keyed (plan, ELO, move),
    recalled by resemblance. This layer LEGITIMATELY grounds in Yami's own Stockfish-scored failure.
  · Invariant: Yami-vs-Stockfish ONLY; learn-then-recall, Stockfish never in `pick`.

This reuses the built machinery verbatim — RiskOverlay / pick / mine / the safety-rethink (proven 5/5) — with the
one seam (`plan=`) letting recognition drive the plan instead of the heuristic suggest_plan. Reproduces the
targeted proof: at each learned trap, does recognition+overlay re-prioritize toward a Stockfish-SAFER move?
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path.home() / "Projects" / "Regenesis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chess
import chess.engine
from regenesis.instrument import understand as rg_understand
from yami.knowledge_graph import PLAN_TEMPLATES
from yami.negative_learning import mine_negative_examples_from_evals

from learn_loop_v2 import SF, RiskOverlay, pick, sf_eval_after
from play_navigate import recognized_plan, render_live

_TEMPLATE = {t.plan_type.name: t for t in PLAN_TEMPLATES}


def recognized_template(board: chess.Board, us: str, them: str, us_white: bool):
    """System 1: the game-so-far → understand → recognized plan (name + PlanTemplate). One fire per move."""
    story = render_live(board, us, them, us_white)
    if story.strip():
        name, _ = recognized_plan(rg_understand(story, kind="text", shape="clinical"), us)
    else:
        name = "IMPROVE_PIECES"                     # thin opening: develop (no wasted fire)
    return name, _TEMPLATE.get(name, _TEMPLATE["IMPROVE_PIECES"])


def play(sf, limit, elo, us_white, overlay, max_plies):
    """Recognition(System 1) → pick with overlay(System 2). Records (fen, uci, plan_name) per Yami move."""
    board = chess.Board()
    us, them = "Yami", "Opponent"
    us_color = chess.WHITE if us_white else chess.BLACK
    moves, plies = [], 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        if board.turn == us_color:
            name, tpl = recognized_template(board, us, them, us_white)
            mv = pick(board, elo, overlay, plan=tpl)
            if mv:
                moves.append((board.fen(), mv.uci(), name))
        else:
            mv = sf.play(board, limit).move
        if mv is None or mv not in board.legal_moves:
            break
        board.push(mv)
        plies += 1
    oc = board.outcome(claim_draw=True)
    won = "1-0" if us_white else "0-1"
    res = "draw" if not oc or oc.result() == "1/2-1/2" else "win" if oc.result() == won else "loss"
    return {"result": res, "plies": plies, "us_white": us_white, "moves": moves}


def mine(sf, game):
    """Score each Yami move (loser POV) → the disaster moves, carrying the plan Yami followed there."""
    loser = chess.WHITE if game["us_white"] else chess.BLACK
    fens, ucis, evb, eva, plans = [], [], [], [], []
    for fen, uci, name in game["moves"]:
        b = chess.Board(fen)
        e0 = sf.analyse(b, chess.engine.Limit(depth=10))["score"].pov(loser).score(mate_score=10000)
        b.push(chess.Move.from_uci(uci))
        e1 = sf.analyse(b, chess.engine.Limit(depth=10))["score"].pov(loser).score(mate_score=10000)
        fens.append(fen); ucis.append(uci); evb.append(e0); eva.append(e1); plans.append(name)
    negs = mine_negative_examples_from_evals(fens, ucis, evb, eva, threshold_cp=150)
    plan_at = {(f, u): p for f, u, p in zip(fens, ucis, plans)}
    return [(ne, plan_at.get((ne.fen, ne.move_uci), "IMPROVE_PIECES")) for ne in negs]


def main():
    max_games = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    max_plies = int(sys.argv[2]) if len(sys.argv) > 2 else 44
    elo = 1350
    sf = chess.engine.SimpleEngine.popen_uci(SF)
    sf.configure({"Threads": 2, "Skill Level": 0})
    limit = chess.engine.Limit(time=0.08)

    overlay = RiskOverlay()
    disasters = []
    for i in range(max_games):
        g = play(sf, limit, elo, i % 2 == 0, overlay, max_plies)
        print(f"game {i+1}: Yami {'W' if g['us_white'] else 'B'} · {g['plies']} plies · {g['result']}", flush=True)
        if g["result"] == "loss":
            for ne, plan in mine(sf, g):
                overlay.learn(plan, elo, ne.move_uci, ne.nav_vector, ne.eval_drop)
                disasters.append((ne.fen, plan, chess.WHITE if g["us_white"] else chess.BLACK))
        if len(disasters) >= 4:
            break

    print(f"\nlearned {len(overlay.entries)} risk weights from {len(disasters)} disasters", flush=True)
    print("TARGETED — at each learned trap: does recognition+overlay re-prioritize to a SAFER move?", flush=True)
    changed = safer = 0
    for fen, plan, loser in disasters[:5]:
        b = chess.Board(fen)
        tpl = _TEMPLATE.get(plan, _TEMPLATE["IMPROVE_PIECES"])
        theory = pick(b, elo, None, plan=tpl)               # recognition plan, System 2 OFF (≈ the trap)
        learned = pick(b, elo, overlay, plan=tpl)            # recognition plan + the learned risk
        if theory != learned:
            changed += 1
            ev_t = sf_eval_after(sf, b, theory, loser)
            ev_l = sf_eval_after(sf, b, learned, loser)
            safer += ev_l > ev_t
            print(f"  [{plan}] theory={b.san(theory)}({ev_t:+d}) → learned={b.san(learned)}({ev_l:+d})"
                  f"  {'SAFER ✓' if ev_l > ev_t else 'worse'}", flush=True)
        else:
            print(f"  [{plan}] no change (theory={b.san(theory)})", flush=True)
    print(f"\nre-prioritized {changed}/{len(disasters[:5])}, of which {safer} to a Stockfish-SAFER move", flush=True)
    sf.quit()


if __name__ == "__main__":
    main()
