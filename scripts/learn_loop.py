"""The negative-learning loop, closed and MEASURED — watch it learn from its own losses and play better.

  BASELINE : play N games vs Stockfish, tactical-baseline + prescription, NO learned censors.
  LEARN    : mine Yami's OWN losing moves (mine_negative_examples_from_evals + Stockfish evals) → censors.
  IMPROVED : replay N games WITH the learned censors applied (filter_moves) — does it lose less / last longer?

All built pieces (prescribe = knowledge_graph + gm_patterns; negative_learning = the censor store). Valid
positions only; games terminate strictly on mate/draw (the prior 'mated then kept playing' bug stays fixed)."""
from __future__ import annotations

import chess
import chess.engine

from yami.gm_patterns import GMPatternDB
from yami.knowledge_graph import evaluate_position, rank_moves, suggest_plan
from yami.navigator import compute_navigation_vector
from yami.negative_learning import LearnedCensorStack, mine_negative_examples_from_evals
from yami.tactical_scoper import scope_moves

SF = "/opt/homebrew/bin/stockfish"
_GM = GMPatternDB("data/gm_patterns.db")


def _allows_mate_in_1(board, move):
    board.push(move)
    bad = any((board.push(m), board.is_checkmate(), board.pop())[1] for m in board.legal_moves)
    board.pop()
    return bad


def pick_move(board, censors=None):
    """Tactical baseline + outcome prescription + (optional) learned censors. Returns a legal move."""
    scoped = scope_moves(board)
    sound = [m for m in scoped if m.see_value >= -50 and not _allows_mate_in_1(board, m.move)] or scoped
    nav = compute_navigation_vector(board)
    if censors:
        sound = censors.filter_moves(board, sound, nav) or sound   # drop learned-bad moves
    ranked = rank_moves(sound, suggest_plan(evaluate_position(board)), board)
    win = {s.move_uci: s.win_rate for s in _GM.query(board, nav, top_k=3) if s.games_seen >= 1}
    ranked.sort(key=lambda r: (win.get(r.scoped_move.move.uci(), -1.0), r.alignment), reverse=True)
    for r in ranked:
        if r.scoped_move.move in board.legal_moves:
            return r.scoped_move.move
    return next(iter(board.legal_moves), None)


def play(sf, limit, yami_white, censors=None, max_plies=250):
    board = chess.Board()
    yami_moves = []                                     # (fen_before, move_uci) for Yami's moves
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        if (board.turn == chess.WHITE) == yami_white:
            mv = pick_move(board, censors)
            if mv:
                yami_moves.append((board.fen(), mv.uci()))
        else:
            mv = sf.play(board, limit).move
        if mv is None or mv not in board.legal_moves:
            break
        board.push(mv); plies += 1
    oc = board.outcome(claim_draw=True)
    won = "1-0" if yami_white else "0-1"
    result = ("draw" if not oc or oc.result() == "1/2-1/2"
              else "win" if oc.result() == won else "loss")
    return {"result": result, "plies": plies, "yami_white": yami_white, "moves": yami_moves}


def mine_loss(sf, game):
    """Stockfish-eval Yami's moves in a lost game → negative examples (where Yami's eval dropped)."""
    loser = chess.WHITE if game["yami_white"] else chess.BLACK
    fens, moves, evb, eva = [], [], [], []
    for fen, uci in game["moves"]:
        b = chess.Board(fen)
        ev0 = sf.analyse(b, chess.engine.Limit(depth=10))["score"].pov(loser).score(mate_score=10000)
        b.push(chess.Move.from_uci(uci))
        ev1 = sf.analyse(b, chess.engine.Limit(depth=10))["score"].pov(loser).score(mate_score=10000)
        fens.append(fen); moves.append(uci); evb.append(ev0); eva.append(ev1)
    return mine_negative_examples_from_evals(fens, moves, evb, eva, threshold_cp=150)


def summarize(tag, games):
    w = sum(g["result"] == "win" for g in games)
    d = sum(g["result"] == "draw" for g in games)
    ll = sum(g["result"] == "loss" for g in games)
    avg = sum(g["plies"] for g in games) / len(games)
    print(f"  {tag:9s}: {w}W {d}D {ll}L · avg {avg:.0f} plies", flush=True)
    return w, d, ll, avg


def main():
    sf = chess.engine.SimpleEngine.popen_uci(SF)
    sf.configure({"Threads": 2, "Skill Level": 0})
    limit = chess.engine.Limit(time=0.08)
    N = 4

    print("BASELINE (no learned censors):", flush=True)
    base = [play(sf, limit, i % 2 == 0) for i in range(N)]
    summarize("baseline", base)

    print("LEARN from Yami's own losses:", flush=True)
    censors = LearnedCensorStack()
    mined = 0
    for g in base:
        if g["result"] == "loss":
            for ne in mine_loss(sf, g):
                censors.add_negative_example(ne); mined += 1
    print(f"  mined {mined} negative examples from {sum(g['result']=='loss' for g in base)} losses", flush=True)

    print("IMPROVED (with learned censors):", flush=True)
    improved = [play(sf, limit, i % 2 == 0, censors=censors) for i in range(N)]
    summarize("improved", improved)
    sf.quit()


if __name__ == "__main__":
    main()
