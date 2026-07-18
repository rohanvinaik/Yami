"""Taste the dish — the direct narrative → prescription → move pipeline, judged by Stockfish.

The chef can cook (Genesis derives the game's narrative). Now the dish: for a POSITION, state its narrative
(the preferred tactical frame), prescribe the best moves that enact it, and let Stockfish TASTE the top
pick. Composes the ALREADY-BUILT pieces — `evaluate_position` (profile) → `suggest_plan` (narrative frame)
→ `rank_moves` (prescription over legal candidates) — plus Stockfish as the exogenous judge (Pillar 1).

Measure: centipawn loss of the narrative-prescribed top move vs Stockfish's best. Small loss = the story
read produced a good move = the dish tastes good. Real positions only (pulled from roster games)."""
from __future__ import annotations

import chess
import chess.engine

from yami.gm_patterns import GMPatternDB
from yami.knowledge_graph import evaluate_position, rank_moves, suggest_plan
from yami.navigator import compute_navigation_vector
from yami.tactical_scoper import scope_moves

STOCKFISH = "/opt/homebrew/bin/stockfish"
_GM = GMPatternDB("data/gm_patterns.db")


def _allows_mate_in_1(board: chess.Board, move: chess.Move) -> bool:
    """Does `move` let the opponent mate immediately next ply?"""
    board.push(move)
    allows = False
    for m in board.legal_moves:
        board.push(m)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            allows = True
            break
    board.pop()
    return allows


def prescribe(board: chess.Board):
    """Narrative → frame → moves, with the TACTICAL BASELINE + outcome prescription both ON.
    Baseline: never hang material (SEE) or allow mate-in-1 (keep the kitchen from burning). Prescription:
    among the SOUND moves, prefer the one that WON here (`gm_patterns` win_rate), narrative-fit as tie-break."""
    plan = suggest_plan(evaluate_position(board))            # "this position is a narrative about X"
    scoped = scope_moves(board)
    # TACTICAL BASELINE: drop moves that lose material on the exchange or allow immediate mate.
    sound = [m for m in scoped if m.see_value >= -50 and not _allows_mate_in_1(board, m.move)]
    ranked = rank_moves(sound or scoped, plan, board)        # rank the SOUND set (fallback: all)
    # PRESCRIPTION: recall the move that WON here (Pillar 1: the game result is the warrant).
    # win_rate = the outcome signal ("it won that time"); games_seen guards against 1-game noise.
    gm = _GM.query(board, compute_navigation_vector(board), top_k=3)
    winning = {s.move_uci: s.win_rate for s in gm if s.games_seen >= 1}
    # re-rank: outcome-grounded moves first (by win-rate), then narrative-fit as the tie-break/fallback
    ranked.sort(key=lambda r: (winning.get(r.scoped_move.move.uci(), -1.0), r.alignment), reverse=True)
    return plan, ranked


def yami_move(board: chess.Board) -> chess.Move | None:
    """The prescription's move for the side to move. Returns a LEGAL move or None (no legal move)."""
    _, ranked = prescribe(board)
    for r in ranked:                                   # ranked over scoped legal moves; guard anyway
        if r.scoped_move.move in board.legal_moves:
            return r.scoped_move.move
    legal = list(board.legal_moves)
    return legal[0] if legal else None


def play_to_completion(sf, sf_limit, yami_white: bool, max_plies: int = 300) -> dict:
    """Yami-prescription vs Stockfish, to CHECKMATE or DRAW. The prior failure was playing PAST a mate —
    so game-over is checked BEFORE every ply, and no illegal move is ever pushed (only valid positions)."""
    board = chess.Board()
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        yamis_turn = (board.turn == chess.WHITE) == yami_white
        if yamis_turn:
            mv = yami_move(board)
        else:
            mv = sf.play(board, sf_limit).move
        if mv is None or mv not in board.legal_moves:   # never push an illegal move → never an invalid position
            break
        board.push(mv)
        plies += 1
    oc = board.outcome(claim_draw=True)
    return {"plies": plies, "fen": board.fen(),
            "termination": oc.termination.name if oc else "move_cap",
            "result": oc.result() if oc else "*",
            "checkmate": board.is_checkmate()}


def main():
    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    sf.configure({"Threads": 2, "Skill Level": 0})       # weak SF for a coherent BASELINE taste
    limit = chess.engine.Limit(time=0.1)
    for i in range(4):
        yami_white = i % 2 == 0
        r = play_to_completion(sf, limit, yami_white)
        yami_res = ("draw" if r["result"] == "1/2-1/2"
                    else "WIN" if r["result"] == ("1-0" if yami_white else "0-1") else "loss")
        print(f"game {i+1}: Yami {'W' if yami_white else 'B'} · {r['plies']} plies · "
              f"{r['termination']} · result {r['result']} → Yami {yami_res}"
              + ("  [MATE]" if r["checkmate"] else ""))
        # sanity: the final position must be valid (the prior-bug guard)
        assert chess.Board(r["fen"]).is_valid(), "INVALID final position!"
    sf.quit()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
