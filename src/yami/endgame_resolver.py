"""Layer 3: Endgame Resolution — Syzygy tablebase lookup for exact endgame play.

If <= 7 pieces remain, the position is solved. No LLM needed.
The chess equivalent of Lean's `exact?` tactic.
"""

from __future__ import annotations

import os
from pathlib import Path

import chess
import chess.syzygy

_tablebase: chess.syzygy.Tablebase | None = None


def _get_tablebase() -> chess.syzygy.Tablebase | None:
    """Lazy-load the Syzygy tablebase from configured path."""
    global _tablebase
    if _tablebase is not None:
        return _tablebase

    tb_path = os.environ.get("SYZYGY_PATH", "")
    if not tb_path:
        return None

    path = Path(tb_path)
    if not path.is_dir():
        return None

    _tablebase = chess.syzygy.open_tablebase(str(path))
    return _tablebase


def can_resolve(board: chess.Board) -> bool:
    """Check if the position is within tablebase range."""
    return chess.popcount(board.occupied) <= 7 and _get_tablebase() is not None


def resolve(board: chess.Board) -> chess.Move | None:
    """Look up the exact best move from tablebases.

    Returns the best move if position is in tablebase range, None otherwise.
    """
    if chess.popcount(board.occupied) > 7:
        return None

    tb = _get_tablebase()
    if tb is None:
        return None

    best_move = None
    best_wdl = -3  # worst possible

    for move in board.legal_moves:
        board.push(move)
        try:
            wdl = -tb.probe_wdl(board)
        except KeyError:
            board.pop()
            continue
        board.pop()

        if wdl > best_wdl:
            best_wdl = wdl
            best_move = move

    return best_move


def probe_wdl(board: chess.Board) -> int | None:
    """Probe the WDL (Win/Draw/Loss) value for a position.

    Returns 2 (win), 1 (cursed win), 0 (draw), -1 (blessed loss), -2 (loss),
    or None if not in tablebase.
    """
    tb = _get_tablebase()
    if tb is None:
        return None

    if chess.popcount(board.occupied) > 7:
        return None

    try:
        return tb.probe_wdl(board)
    except KeyError:
        return None
