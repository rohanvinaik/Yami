"""Tests for the multi-signal coherence layer."""

import chess

from yami.coherence import compute_coherence
from yami.navigator import compute_navigation_vector
from yami.strategy_library import STRATEGY_LIBRARY, query_strategies


def test_coherence_produces_scores():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    nav = compute_navigation_vector(board)
    moves = list(board.legal_moves)[:5]
    result = compute_coherence(board, moves, nav)
    assert len(result.scored_moves) == 5
    assert all(s.final_score >= 0 for s in result.scored_moves)


def test_coherence_sorted_by_final_score():
    board = chess.Board()
    nav = compute_navigation_vector(board)
    moves = list(board.legal_moves)[:8]
    result = compute_coherence(board, moves, nav)
    for i in range(len(result.scored_moves) - 1):
        assert result.scored_moves[i].final_score >= result.scored_moves[i + 1].final_score


def test_strategy_library_not_empty():
    assert len(STRATEGY_LIBRARY) >= 10


def test_query_strategies_returns_matches():
    board = chess.Board()
    # After some moves, create a middlegame position
    for san in ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6"]:
        board.push_san(san)
    nav = compute_navigation_vector(board)
    anchors = {"center-control", "development", "piece-coordination"}
    results = query_strategies(board, nav, anchors, top_k=3)
    assert isinstance(results, list)


def test_coherence_with_tactical_position():
    # Position with clear tactical motifs
    board = chess.Board()
    for san in ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6"]:
        board.push_san(san)
    nav = compute_navigation_vector(board)
    moves = list(board.legal_moves)[:5]
    result = compute_coherence(board, moves, nav)
    # Qxf7# should score high if it's in the candidate list
    assert result.signal_agreement >= 0.0
    assert result.dominant_signal != ""


def test_coherence_agreement_range():
    board = chess.Board()
    nav = compute_navigation_vector(board)
    moves = list(board.legal_moves)[:5]
    result = compute_coherence(board, moves, nav)
    assert 0.0 <= result.signal_agreement <= 1.0


def test_strategy_phase_filtering():
    # Endgame position — should not match opening strategies
    board = chess.Board("4k3/4p3/8/8/8/8/4P3/4K3 w - - 0 1")
    nav = compute_navigation_vector(board)
    anchors = {"endgame-technique", "king-march"}
    results = query_strategies(board, nav, anchors, top_k=5)
    for strat, _ in results:
        assert strat.phase != "opening"


def test_coherence_engine_integration():
    """Smoke test: engine with coherence produces legal moves."""
    from yami.engine import YamiEngine
    engine = YamiEngine(
        use_llm=False, use_neural=False,
        use_navigator=True, use_temporal=True,
        use_opening_book=True,
    )
    decision = engine.decide()
    assert decision.move is not None
    assert decision.move in engine.board.legal_moves
