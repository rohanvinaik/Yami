"""Tests for the GM Pattern Database."""

import chess

from yami.gm_patterns import GMPatternDB
from yami.navigator import compute_navigation_vector


def test_gm_db_seeded():
    db = GMPatternDB()
    assert db.count() > 0
    db.close()


def test_gm_query_returns_suggestions():
    db = GMPatternDB()
    board = chess.Board()
    nav = compute_navigation_vector(board)
    results = db.query(board, nav, top_k=3)
    # Starting position should match some canonical patterns
    assert isinstance(results, list)
    db.close()


def test_gm_store_and_query():
    from yami.gm_patterns import _material_signature
    db = GMPatternDB()
    board = chess.Board()
    mat_sig = _material_signature(board)
    nav = compute_navigation_vector(board)
    # Store a pattern matching the actual starting position signature
    db.store_move(
        mat_sig, nav.as_tuple(),
        "e2e4", "e4", "1-0", 2600,
    )
    results = db.query(board, nav, top_k=5)
    ucis = [r.move_uci for r in results]
    assert "e2e4" in ucis
    db.close()


def test_gm_suggestion_fields():
    db = GMPatternDB()
    db.store_move(
        "TEST_vs_TEST", (0, 0, 0, 0, 0, 0),
        "d2d4", "d4", "1-0", 2500,
    )
    board = chess.Board()
    nav = compute_navigation_vector(board)
    results = db.query(board, nav, top_k=10)
    for r in results:
        assert 0.0 <= r.frequency <= 1.0
        assert 0.0 <= r.win_rate <= 1.0
        assert r.games_seen >= 0
    db.close()


def test_gm_only_legal_moves():
    """GM suggestions should only include legal moves."""
    db = GMPatternDB()
    board = chess.Board()
    nav = compute_navigation_vector(board)
    results = db.query(board, nav, top_k=10)
    for r in results:
        move = chess.Move.from_uci(r.move_uci)
        assert move in board.legal_moves
    db.close()
