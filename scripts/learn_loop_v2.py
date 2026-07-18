"""Soft-overlay negative learning — a graded, narrative+ELO-keyed RISK penalty on the ranking, NOT a ban.

Default = play theory (plan + prescription). Learned = "in THIS narrative, vs THIS ELO, the theory-move
carried a disaster tail → de-prioritize it and rethink toward a safer move." The penalty is subtracted
from the theory score (soft); a big penalty is the 'rethink'. Keyed on (plan_type, elo_bracket, move) so
it never over-applies across contexts (the failure mode of the hard `(phase, move.uci)` ban).

TARGETED proof (avoids N-game noise): at each disaster position, does the overlay re-prioritize AWAY from
the trap the theory picked, toward a move Stockfish rates SAFER? That is the mechanism, shown directly.

A (safety-seeking rethink): de-prioritizing a trap is only half the spec — the REPLACEMENT must be SAFE, not
the next tactically-blind theory move. So a safety term (internal SEE + the recalled ELO-scoped risk schema)
is gated by the RETHINK PRESSURE (the learned penalty present in this regime): no learned trap → pure theory
(System 1 unchanged); a trap present → safety proportionally leads the reshuffle. This is System 2's
move-refinement (see docs/GENESIS_CHESS_ARCHITECTURE.md §13A) — grounded in Yami's own Stockfish-scored
experience, learn-then-recall, NO live Stockfish in pick."""
from __future__ import annotations

import chess
import chess.engine

from yami.gm_patterns import GMPatternDB
from yami.knowledge_graph import evaluate_position, rank_moves, suggest_plan
from yami.navigator import compute_navigation_vector
from yami.negative_learning import mine_negative_examples_from_evals
from yami.tactical_scoper import scope_moves

SF = "/opt/homebrew/bin/stockfish"
_GM = GMPatternDB("data/gm_patterns.db")


class RiskOverlay:
    """(plan_type, elo_bracket, move_uci) → accumulated disaster risk. Soft penalty, not a blacklist."""
    def __init__(self):
        self.risk: dict[tuple, float] = {}

    def learn(self, plan: str, elo: int, uci: str, eval_drop: int) -> None:
        k = (plan, elo, uci)
        self.risk[k] = self.risk.get(k, 0.0) + eval_drop

    def penalty(self, plan: str, elo: int, uci: str) -> float:
        return self.risk.get((plan, elo, uci), 0.0) / 200.0   # cp → rank-score units


SAFETY_GAIN = 1.2   # how hard a triggered rethink pulls toward safety (0 = pure theory de-prioritization)


def _allows_mate_in_1(board, move):
    board.push(move)
    bad = any((board.push(m), board.is_checkmate(), board.pop())[1] for m in board.legal_moves)
    board.pop()
    return bad


def _see_safety(see_value):
    """Internal static-exchange safety in [0,1]: quiet move → 0.5, good capture → >0.5, loses material → <0.5."""
    return (min(max(see_value, -100), 100) + 100) / 200.0


def _sat(x):
    """Saturating rethink pressure in [0,1): a bigger learned penalty pulls harder toward safety, bounded."""
    return x / (x + 1.0)


def rank_candidates(board, elo, overlay=None):
    """Theory rank (plan + gm win-rate) − learned risk penalty + a SAFETY term gated on rethink pressure.

    A: when this (plan, elo) regime carries a learned trap (max penalty > 0), route the reshuffle toward the
    SEE-safest / lowest-learned-risk survivor — not the next tactically-blind theory move. No trap → pure
    theory (System 1 unchanged). Safety = internal SEE + the recalled schema (via −pen); never live Stockfish."""
    scoped = scope_moves(board)
    sound = [m for m in scoped if m.see_value >= -50 and not _allows_mate_in_1(board, m.move)] or scoped
    nav = compute_navigation_vector(board)
    plan = suggest_plan(evaluate_position(board))
    ranked = rank_moves(sound, plan, board)
    win = {s.move_uci: s.win_rate for s in _GM.query(board, nav, top_k=3) if s.games_seen >= 1}

    pens = {r.scoped_move.move.uci(): (overlay.penalty(plan.plan_type.name, elo, r.scoped_move.move.uci())
                                       if overlay else 0.0) for r in ranked}
    rethink = _sat(max(pens.values(), default=0.0))            # 0 when nothing learned here → theory leads

    def score(r):
        uci = r.scoped_move.move.uci()
        theory = win.get(uci, 0.0) * 1.0 + r.alignment * 0.2
        safety = SAFETY_GAIN * rethink * _see_safety(r.scoped_move.see_value)
        return theory - pens[uci] + safety                     # SOFT de-prioritize the trap, safety leads the rethink

    return sorted(ranked, key=score, reverse=True), plan


def pick(board, elo, overlay=None):
    ranked, _ = rank_candidates(board, elo, overlay)
    for r in ranked:
        if r.scoped_move.move in board.legal_moves:
            return r.scoped_move.move
    return next(iter(board.legal_moves), None)


def play(sf, limit, elo, yami_white, overlay=None, max_plies=250):
    board = chess.Board()
    yami_moves, plies = [], 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        if (board.turn == chess.WHITE) == yami_white:
            mv = pick(board, elo, overlay)
            if mv:
                yami_moves.append((board.fen(), mv.uci()))
        else:
            mv = sf.play(board, limit).move
        if mv is None or mv not in board.legal_moves:
            break
        board.push(mv); plies += 1
    oc = board.outcome(claim_draw=True)
    won = "1-0" if yami_white else "0-1"
    res = "draw" if not oc or oc.result() == "1/2-1/2" else "win" if oc.result() == won else "loss"
    return {"result": res, "plies": plies, "yami_white": yami_white, "moves": yami_moves}


def mine(sf, game):
    loser = chess.WHITE if game["yami_white"] else chess.BLACK
    fens, ms, evb, eva = [], [], [], []
    for fen, uci in game["moves"]:
        b = chess.Board(fen)
        e0 = sf.analyse(b, chess.engine.Limit(depth=10))["score"].pov(loser).score(mate_score=10000)
        b.push(chess.Move.from_uci(uci))
        e1 = sf.analyse(b, chess.engine.Limit(depth=10))["score"].pov(loser).score(mate_score=10000)
        fens.append(fen); ms.append(uci); evb.append(e0); eva.append(e1)
    return mine_negative_examples_from_evals(fens, ms, evb, eva, threshold_cp=150)


def sf_eval_after(sf, board, move, loser):
    b = board.copy(); b.push(move)
    return sf.analyse(b, chess.engine.Limit(depth=10))["score"].pov(loser).score(mate_score=10000)


def main():
    sf = chess.engine.SimpleEngine.popen_uci(SF)
    sf.configure({"Threads": 2, "Skill Level": 0})
    limit = chess.engine.Limit(time=0.08)
    ELO = 1350

    # play until we get a loss with disaster-moves to learn from
    overlay = RiskOverlay()
    disasters = []
    for i in range(6):
        g = play(sf, limit, ELO, i % 2 == 0)
        if g["result"] == "loss":
            for ne in mine(sf, g):
                b = chess.Board(ne.fen)
                plan = suggest_plan(evaluate_position(b)).plan_type.name
                overlay.learn(plan, ELO, ne.move_uci, ne.eval_drop)
                disasters.append((ne.fen, ne.move_uci, ne.eval_drop, plan, chess.WHITE if g["yami_white"] else chess.BLACK))
        if len(disasters) >= 5:
            break

    print(f"learned {len(overlay.risk)} risk weights from {len(disasters)} disasters\n", flush=True)
    print("TARGETED — at each learned trap: did the overlay re-prioritize to a SAFER move?", flush=True)
    changed = safer = 0
    for fen, trap_uci, drop, plan, loser in disasters[:5]:
        b = chess.Board(fen)
        theory = pick(b, ELO, overlay=None)          # what theory alone plays (≈ the trap)
        learned = pick(b, ELO, overlay=overlay)       # with the risk penalty
        if theory != learned:
            changed += 1
            ev_t = sf_eval_after(sf, b, theory, loser)
            ev_l = sf_eval_after(sf, b, learned, loser)
            better = ev_l > ev_t
            safer += better
            print(f"  [{plan}] theory={b.san(theory)}({ev_t:+d}cp) → learned={b.san(learned)}({ev_l:+d}cp)"
                  f"  {'SAFER ✓' if better else 'worse'}", flush=True)
        else:
            print(f"  [{plan}] no change (theory={b.san(theory)}) — penalty didn't outweigh theory", flush=True)
    print(f"\nre-prioritized {changed}/{len(disasters[:5])}, of which {safer} to a Stockfish-SAFER move", flush=True)
    sf.quit()


if __name__ == "__main__":
    main()
