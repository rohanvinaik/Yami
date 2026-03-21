"""Oracle labeling: use Stockfish to label the best candidate.

Wraps StockfishOracle to evaluate each candidate and determine
which one Stockfish considers best.
"""

from __future__ import annotations

import chess

from yami.models import AnnotatedCandidate
from yami.oracle import StockfishOracle


def label_candidates(
    board: chess.Board,
    candidates: list[AnnotatedCandidate],
    oracle: StockfishOracle,
) -> tuple[int, int, int, int]:
    """Evaluate each candidate with Stockfish and return the best.

    Returns:
        (best_idx, best_eval_cp, second_best_idx, eval_gap_cp)
    """
    if not candidates:
        return 0, 0, -1, 0

    evals: list[tuple[int, int]] = []  # (idx, eval_cp)

    for i, c in enumerate(candidates):
        ev = oracle.evaluate_move(board, c.move)
        cp = ev.score_cp if ev.score_cp is not None else 0
        # Mate scores: convert to large centipawn values
        if ev.mate_in is not None:
            cp = 10000 * (1 if ev.mate_in > 0 else -1)
        evals.append((i, cp))

    # Sort by evaluation (best first)
    evals.sort(key=lambda x: x[1], reverse=True)

    best_idx, best_cp = evals[0]
    second_idx = evals[1][0] if len(evals) > 1 else -1
    second_cp = evals[1][1] if len(evals) > 1 else best_cp
    gap = best_cp - second_cp

    return best_idx, best_cp, second_idx, gap
