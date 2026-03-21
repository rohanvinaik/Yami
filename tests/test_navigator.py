"""Tests for the 6-Bank Chess Navigator."""

import chess

from yami.navigator import (
    NavigationVector,
    compute_navigation_vector,
    detect_anchors,
    otp_score_candidate,
    rank_candidates_by_navigation,
)


def test_navigation_vector_starting_position():
    board = chess.Board()
    nv = compute_navigation_vector(board)
    assert isinstance(nv, NavigationVector)
    assert all(v in (-1, 0, 1) for v in nv.as_tuple())
    assert nv.phase == 1  # opening


def test_navigation_vector_endgame():
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    nv = compute_navigation_vector(board)
    assert nv.phase == -1  # endgame


def test_navigation_vector_tuple():
    nv = NavigationVector(1, -1, 0, 1, 0, -1)
    assert nv.as_tuple() == (1, -1, 0, 1, 0, -1)


def test_hamming_distance():
    a = NavigationVector(1, -1, 0, 1, 0, -1)
    b = NavigationVector(1, -1, 0, 1, 0, -1)
    assert a.hamming_distance(b) == 0

    c = NavigationVector(-1, 1, 0, -1, 0, 1)
    assert a.hamming_distance(c) == 4


def test_detect_anchors_castling():
    board = chess.Board()
    # Simulate position where castling is legal
    board.set_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    castling_move = chess.Move.from_uci("e1g1")
    if castling_move in board.legal_moves:
        anchors = detect_anchors(board, castling_move)
        assert "castling" in anchors


def test_detect_anchors_pawn_break():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("d5")
    # exd5 is a pawn break in the center
    move = board.parse_san("exd5")
    anchors = detect_anchors(board, move)
    # It's a capture of a central pawn
    assert "piece-exchange" in anchors


def test_otp_scoring_prefers_checks_when_attacking():
    board = chess.Board()
    for san in ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6"]:
        board.push_san(san)
    # Qxf7+ is a forcing check — should score high with attacking nav
    nav = NavigationVector(1, 1, -1, 1, 1, 0)  # attacking, forcing
    qxf7 = board.parse_san("Qxf7#")
    anchors = detect_anchors(board, qxf7)
    score = otp_score_candidate(qxf7, anchors, nav, board)
    assert score > 3.0  # high score for attacking + forcing


def test_rank_candidates_returns_sorted():
    board = chess.Board()
    nav = compute_navigation_vector(board)
    moves = list(board.legal_moves)[:10]
    candidates = [(m, None) for m in moves]
    ranked = rank_candidates_by_navigation(board, candidates, nav)
    assert len(ranked) == 10
    # Should be sorted by score descending
    for i in range(len(ranked) - 1):
        assert ranked[i][1] >= ranked[i + 1][1]


def test_navigator_different_positions_different_vectors():
    # Starting position vs endgame should produce different vectors
    start = chess.Board()
    endgame = chess.Board("4k3/4p3/8/8/8/8/4P3/4K3 w - - 0 1")
    nv_start = compute_navigation_vector(start)
    nv_end = compute_navigation_vector(endgame)
    assert nv_start.phase != nv_end.phase
