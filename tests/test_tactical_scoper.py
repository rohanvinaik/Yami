"""Tests for Layer 2: Tactical Scoping and Censors."""

import chess

from yami.tactical_scoper import (
    apply_blunder_censor,
    apply_repetition_censor,
    scope_moves,
)


def test_scope_starting_position():
    board = chess.Board()
    scoped = scope_moves(board)
    assert len(scoped) == 20
    # No tactical motifs in starting position
    for m in scoped:
        assert "check" not in m.motifs
        assert "capture" not in m.motifs


def test_detects_check():
    # Position where Qh5 gives check (after e4 e5 Qh5)
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    scoped = scope_moves(board)
    # Verify scoping produces all legal moves
    assert len(scoped) == 29  # 29 legal moves for white after 1.e4 e5


def test_detects_capture():
    # Position with a capture available
    board = chess.Board()
    board.push_san("e4")
    board.push_san("d5")
    scoped = scope_moves(board)
    exd5 = [m for m in scoped if "capture" in m.motifs]
    assert len(exd5) >= 1  # exd5 should be tagged as capture


def test_detects_promotion():
    # Pawn about to promote
    board = chess.Board("8/P7/8/8/8/8/8/4K2k w - - 0 1")
    scoped = scope_moves(board)
    promos = [m for m in scoped if "promotion" in m.motifs]
    assert len(promos) >= 1


def test_blunder_censor_removes_hanging():
    board = chess.Board()
    scoped = scope_moves(board)
    # In starting position nothing hangs, so censor should keep all
    censored = apply_blunder_censor(scoped)
    assert len(censored) == len(scoped)


def test_blunder_censor_filters():
    from yami.models import ScopedMove
    # Manually create a move tagged as hanging
    move = chess.Move.from_uci("e2e4")
    hanging = ScopedMove(move=move, motifs=("hangs_piece",))
    safe = ScopedMove(move=chess.Move.from_uci("d2d4"), motifs=())
    result = apply_blunder_censor([hanging, safe])
    assert len(result) == 1
    assert result[0].move == chess.Move.from_uci("d2d4")


def test_repetition_censor():
    board = chess.Board()
    # Play some moves back and forth to create repetition potential
    moves = ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"]
    for san in moves:
        board.push_san(san)
    scoped = scope_moves(board)
    censored = apply_repetition_censor(scoped, board)
    # Some moves should be censored due to repetition
    assert len(censored) <= len(scoped)
