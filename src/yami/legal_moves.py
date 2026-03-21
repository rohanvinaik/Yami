"""Layer 1 & 8: Legal move generation and verification.

Deterministic layers wrapping python-chess. Zero ambiguity, zero neural cost.
"""

from __future__ import annotations

import chess


def generate_legal_moves(board: chess.Board) -> list[chess.Move]:
    """Generate all legal moves from the current position."""
    return list(board.legal_moves)


def is_legal(board: chess.Board, move: chess.Move) -> bool:
    """Verify a move is legal. The kernel is the judge."""
    return move in board.legal_moves


def parse_move(board: chess.Board, san: str) -> chess.Move | None:
    """Parse a SAN string into a move, returning None if invalid."""
    try:
        return board.parse_san(san)
    except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
        return None


def is_game_over(board: chess.Board) -> bool:
    """Check if the game is over."""
    return board.is_game_over()


def game_result(board: chess.Board) -> str | None:
    """Return the game result string, or None if game is ongoing."""
    if not board.is_game_over():
        return None
    return board.result()
