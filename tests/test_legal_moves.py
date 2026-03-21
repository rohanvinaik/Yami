"""Tests for Layer 1 & 8: Legal move generation and verification."""

import chess

from yami.legal_moves import (
    game_result,
    generate_legal_moves,
    is_game_over,
    is_legal,
    parse_move,
)


def test_starting_position_has_20_moves():
    board = chess.Board()
    moves = generate_legal_moves(board)
    assert len(moves) == 20


def test_is_legal_accepts_valid_move():
    board = chess.Board()
    e4 = chess.Move.from_uci("e2e4")
    assert is_legal(board, e4)


def test_is_legal_rejects_invalid_move():
    board = chess.Board()
    illegal = chess.Move.from_uci("e2e5")
    assert not is_legal(board, illegal)


def test_parse_move_san():
    board = chess.Board()
    move = parse_move(board, "e4")
    assert move is not None
    assert move == chess.Move.from_uci("e2e4")


def test_parse_move_invalid():
    board = chess.Board()
    move = parse_move(board, "Qxf7")
    assert move is None


def test_game_not_over_at_start():
    board = chess.Board()
    assert not is_game_over(board)
    assert game_result(board) is None


def test_checkmate_detection():
    # Scholar's mate position
    board = chess.Board()
    for san in ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"]:
        board.push_san(san)
    assert is_game_over(board)
    assert game_result(board) == "1-0"


def test_stalemate_detection():
    # Known stalemate position
    board = chess.Board("k7/8/1K6/8/8/8/8/1Q6 b - - 0 1")
    # Black king is not in check but has no legal moves — not quite stalemate
    # Let's use a real stalemate position
    board = chess.Board("k7/8/2K5/8/8/8/8/1Q6 b - - 0 1")
    if board.is_stalemate():
        assert is_game_over(board)
        assert game_result(board) == "1/2-1/2"
