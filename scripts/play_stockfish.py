"""U7 — recognition → move, played to completion vs Stockfish (the exogenous JUDGE).

The live loop closes the whole stack in one place: each of Yami's turns we render the GAME-SO-FAR
(board.move_stack) into event-prose, fire it ONCE through Regenesis `understand` (Python-native, no JVM),
let `recognized_plan` name the plan from the read, and `select_move` play the top sound arc-advancing
candidate. NOT one fire per candidate — one fire per move (the game-so-far read), then cheap symbolic
scoring. Stockfish is never a trainer: it is the OPPONENT and, per move, the exogenous JUDGE — the
centipawn gap between Yami's recognized move and Stockfish's own best is how master-like the recognition is.

Run:  PYTHONPATH=src python3 scripts/play_stockfish.py [max_plies] [skill]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path.home() / "Projects" / "Regenesis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))          # sibling: play_navigate

import chess
import chess.engine
from regenesis.instrument import understand as rg_understand
from yami.knowledge_graph import PLAN_TEMPLATES

from play_navigate import recognized_plan, render_live
from prescribe_taste import prescribe_with_plan

STOCKFISH = "/opt/homebrew/bin/stockfish"
_TEMPLATE = {t.plan_type.name: t for t in PLAN_TEMPLATES}   # PlanType name → PlanTemplate (recognition → selector)


def recognition_move(board: chess.Board, us: str, them: str, us_white: bool) -> tuple[chess.Move | None, str, dict]:
    """The whole live pipeline for one move: render game-so-far → understand → recognized plan → the SOUND
    selector picks the move that enacts it (U7: recognition NAMES the plan; prescribe's tactical baseline +
    gm_patterns win-rate + narrative-fit PICK the move within it). One understand-fire per move."""
    story = render_live(board, us, them, us_white)
    if story.strip():                                # thin opening → no story yet; develop (skip a wasted fire)
        read = rg_understand(story, kind="text", shape="clinical")
        plan, why = recognized_plan(read, us)
    else:
        plan, why = "IMPROVE_PIECES", "(opening: no story yet)"
    _, ranked = prescribe_with_plan(board, _TEMPLATE.get(plan, _TEMPLATE["IMPROVE_PIECES"]))
    mv = next((r.scoped_move.move for r in ranked if r.scoped_move.move in board.legal_moves), None)
    if mv is None:
        legal = list(board.legal_moves)
        mv = legal[0] if legal else None
    return mv, f"{plan}  [{why}]", {}


def _cp(info: dict, pov: chess.Color) -> int:
    return info["score"].pov(pov).score(mate_score=100000)


def play(sf_play, sf_judge, judge_limit, sf_limit, us_white: bool, max_plies: int) -> dict:
    """Yami(recognition) vs Stockfish to mate/draw. sf_play = the (weak) OPPONENT; sf_judge = the
    full-strength JUDGE that scores each Yami move's cp-loss vs its own best. Two engines: a Skill-0
    opponent must NOT also be the yardstick — the judge stays external and strong."""
    board = chess.Board()
    us, them = "Yami", "Opponent"
    us_color = chess.WHITE if us_white else chess.BLACK
    losses: list[int] = []
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        our_turn = board.turn == us_color
        if our_turn:
            best = _cp(sf_judge.analyse(board, judge_limit), us_color)   # the judge's verdict on the position
            mv, tag, _info = recognition_move(board, us, them, us_white)
            if mv is None or mv not in board.legal_moves:
                break
            san = board.san(mv)                                          # name it BEFORE the push
            board.push(mv)
            got = _cp(sf_judge.analyse(board, judge_limit), us_color)    # eval after our recognized move
            loss = max(0, best - got)
            losses.append(loss)
            print(f"  ply {plies+1:>3} · {san:6} · {tag:52.52} · cp_loss {loss}")
        else:
            mv = sf_play.play(board, sf_limit).move
            if mv is None or mv not in board.legal_moves:
                break
            board.push(mv)
        plies += 1
    oc = board.outcome(claim_draw=True)
    result = oc.result() if oc else "*"
    us_res = ("draw" if result == "1/2-1/2"
              else "WIN" if result == ("1-0" if us_white else "0-1") else "loss")
    acpl = round(sum(losses) / len(losses), 1) if losses else 0.0
    return {"plies": plies, "result": result, "us_res": us_res, "acpl": acpl,
            "termination": oc.termination.name if oc else "move_cap",
            "checkmate": board.is_checkmate(), "fen": board.fen(), "moves": len(losses)}


def main() -> None:
    max_plies = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    skill = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    sf_play = chess.engine.SimpleEngine.popen_uci(STOCKFISH)          # the OPPONENT (weak)
    sf_play.configure({"Threads": 2, "Skill Level": skill})
    sf_judge = chess.engine.SimpleEngine.popen_uci(STOCKFISH)         # the JUDGE (full strength, external)
    sf_judge.configure({"Threads": 2})
    sf_limit = chess.engine.Limit(time=0.1)                          # Stockfish's own move (opponent)
    judge_limit = chess.engine.Limit(depth=12)                       # the exogenous-judge eval (consistent)
    print(f"=== Yami(recognition) vs Stockfish (Skill {skill}), Yami=White, cap {max_plies} plies ===")
    r = play(sf_play, sf_judge, judge_limit, sf_limit, us_white=True, max_plies=max_plies)
    sf_play.quit()
    sf_judge.quit()
    print(f"\nresult {r['result']} → Yami {r['us_res']} · {r['plies']} plies · {r['termination']}"
          + ("  [MATE]" if r["checkmate"] else ""))
    print(f"ACPL over {r['moves']} recognized moves: {r['acpl']} cp  (lower = more master-like)")
    assert chess.Board(r["fen"]).is_valid(), "INVALID final position!"


if __name__ == "__main__":
    main()
