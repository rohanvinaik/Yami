"""Stockfish Oracle — the teacher/validator, not the runtime engine.

Plays the role of `exact?` in the Wayfinder architecture:
- Evaluate candidates with Stockfish to generate training data
- Validate infrastructure's top candidate against Stockfish's choice
- Fallback when the LLM fails
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

import chess
import chess.engine


@dataclass
class Evaluation:
    """Stockfish evaluation result."""

    score_cp: int | None  # centipawns, from side-to-move perspective
    mate_in: int | None  # mate in N, or None
    best_move: chess.Move | None
    depth: int
    pv: list[chess.Move]  # principal variation


class StockfishOracle:
    """Stockfish UCI engine wrapper for evaluation and validation."""

    def __init__(self, depth: int = 20, time_limit: float | None = None):
        self.depth = depth
        self.time_limit = time_limit
        self._engine: chess.engine.SimpleEngine | None = None

    def _get_engine(self) -> chess.engine.SimpleEngine:
        if self._engine is not None:
            return self._engine

        path = os.environ.get("STOCKFISH_PATH", "")
        if not path:
            path = shutil.which("stockfish") or ""
        if not path:
            raise RuntimeError(
                "Stockfish not found. Set STOCKFISH_PATH or install stockfish."
            )

        self._engine = chess.engine.SimpleEngine.popen_uci(path)
        return self._engine

    def evaluate(self, board: chess.Board) -> Evaluation:
        """Evaluate a position with Stockfish."""
        engine = self._get_engine()
        limit = chess.engine.Limit(depth=self.depth, time=self.time_limit)
        info = engine.analyse(board, limit)

        score = info.get("score")
        pv = info.get("pv", [])
        depth = info.get("depth", self.depth)

        score_cp = None
        mate_in = None
        best_move = pv[0] if pv else None

        if score:
            pov = score.white() if board.turn == chess.WHITE else score.black()
            cp = pov.score()
            if cp is not None:
                score_cp = cp
            mate = pov.mate()
            if mate is not None:
                mate_in = mate

        return Evaluation(
            score_cp=score_cp,
            mate_in=mate_in,
            best_move=best_move,
            depth=depth,
            pv=list(pv),
        )

    def evaluate_move(self, board: chess.Board, move: chess.Move) -> Evaluation:
        """Evaluate the position after a specific move."""
        board.push(move)
        try:
            result = self.evaluate(board)
            # Flip perspective since we pushed a move
            if result.score_cp is not None:
                result = Evaluation(
                    score_cp=-result.score_cp,
                    mate_in=-result.mate_in if result.mate_in else None,
                    best_move=result.best_move,
                    depth=result.depth,
                    pv=result.pv,
                )
            return result
        finally:
            board.pop()

    def best_move(self, board: chess.Board) -> chess.Move | None:
        """Get Stockfish's best move for the position."""
        engine = self._get_engine()
        limit = chess.engine.Limit(depth=self.depth, time=self.time_limit)
        result = engine.play(board, limit)
        return result.move

    def validate_move(
        self, board: chess.Board, move: chess.Move, tolerance_cp: int = 50
    ) -> bool:
        """Check if a move is within tolerance of Stockfish's best.

        Returns True if the move loses at most `tolerance_cp` centipawns
        compared to Stockfish's best move.
        """
        best = self.best_move(board)
        if best is None or move == best:
            return True

        best_eval = self.evaluate_move(board, best)
        move_eval = self.evaluate_move(board, move)

        if best_eval.score_cp is None or move_eval.score_cp is None:
            return True  # can't compare mate scores easily

        return (best_eval.score_cp - move_eval.score_cp) <= tolerance_cp

    def close(self) -> None:
        """Shut down the engine."""
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def __enter__(self) -> StockfishOracle:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
