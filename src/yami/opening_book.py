"""Layer 4: Opening Book — database lookup for the first 10-15 moves.

Known positions are like known proof patterns — you don't search, you navigate.
"""

from __future__ import annotations

import os
from pathlib import Path

import chess
import chess.polyglot

from yami.models import BookMove

_book_reader: chess.polyglot.MemoryMappedReader | None = None
_book_load_attempted = False


def _get_book() -> chess.polyglot.MemoryMappedReader | None:
    """Lazy-load the Polyglot opening book."""
    global _book_reader, _book_load_attempted
    if _book_load_attempted:
        return _book_reader

    _book_load_attempted = True
    book_path = os.environ.get("POLYGLOT_BOOK", "")
    if not book_path:
        return None

    path = Path(book_path)
    if not path.is_file():
        return None

    _book_reader = chess.polyglot.open_reader(str(path))
    return _book_reader


def lookup(board: chess.Board, max_moves: int = 3) -> list[BookMove]:
    """Look up the position in the opening book.

    Returns up to max_moves book moves, sorted by weight.
    """
    reader = _get_book()
    if reader is None:
        return []

    try:
        entries = list(reader.find_all(board))
    except KeyError:
        return []

    if not entries:
        return []

    # Sort by weight (popularity)
    entries.sort(key=lambda e: e.weight, reverse=True)

    return [
        BookMove(move=entry.move, weight=entry.weight)
        for entry in entries[:max_moves]
    ]


def is_in_book(board: chess.Board) -> bool:
    """Check if the current position has book entries."""
    return bool(lookup(board, max_moves=1))


def get_book_move(board: chess.Board) -> chess.Move | None:
    """Get the top book move for the position."""
    moves = lookup(board, max_moves=1)
    if moves:
        return moves[0].move
    return None


# Fallback: built-in opening lines for common positions
# These cover the most common openings without requiring a book file
_BUILTIN_OPENINGS: dict[str, list[str]] = {
    # Starting position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -": ["e2e4", "d2d4", "c2c4"],
    # After 1.e4
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -": ["e7e5", "c7c5", "e7e6"],
    # After 1.d4
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq -": ["d7d5", "g8f6", "e7e6"],
    # After 1.e4 e5
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -": ["g1f3", "f1c4", "b1c3"],
    # After 1.d4 d5
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq -": ["c2c4", "g1f3", "b1c3"],
    # After 1.e4 c5 (Sicilian)
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -": ["g1f3", "b1c3", "c2c3"],
}


def builtin_lookup(board: chess.Board) -> chess.Move | None:
    """Look up position in the built-in opening table."""
    # Normalize FEN by stripping move counters
    fen_parts = board.fen().split()
    key = " ".join(fen_parts[:4])

    moves_uci = _BUILTIN_OPENINGS.get(key)
    if not moves_uci:
        return None

    # Return the first legal move from the list
    for uci in moves_uci:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            return move
    return None
