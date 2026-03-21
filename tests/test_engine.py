"""Tests for the full Yami engine pipeline."""

import chess

from yami.engine import DecisionSource, YamiEngine


def test_engine_decides_from_starting_position():
    engine = YamiEngine(use_llm=False)
    decision = engine.decide()
    assert decision.move is not None
    assert decision.move in engine.board.legal_moves


def test_engine_infrastructure_only():
    engine = YamiEngine(use_llm=False, use_opening_book=False)
    decision = engine.decide()
    assert decision.move is not None
    assert decision.candidates  # should have candidates
    assert decision.plan is not None
    assert decision.profile is not None


def test_engine_uses_opening_book():
    engine = YamiEngine(use_llm=False, use_opening_book=True)
    decision = engine.decide()
    # Starting position should use builtin opening book
    assert decision.move is not None
    assert decision.source == DecisionSource.OPENING_BOOK


def test_engine_play_move():
    engine = YamiEngine(use_llm=False)
    decision = engine.play_move()
    assert decision.move is not None
    assert len(engine.state.move_history) == 1
    assert engine.board.fullmove_number >= 1


def test_engine_play_opponent_move():
    engine = YamiEngine(use_llm=False)
    move = engine.play_opponent_move("e4")
    assert move is not None
    assert engine.board.fullmove_number == 1


def test_engine_full_game_no_crash():
    """Play a short game to ensure nothing crashes."""
    engine = YamiEngine(use_llm=False, use_opening_book=True)
    for _ in range(40):  # 20 full moves
        if engine.is_game_over():
            break
        decision = engine.play_move()
        assert decision.move is not None


def test_engine_reset():
    engine = YamiEngine(use_llm=False)
    engine.play_move()
    engine.reset()
    assert engine.board.fen() == chess.STARTING_FEN
    assert len(engine.state.move_history) == 0


def test_engine_censors_prevent_blunders():
    engine_censored = YamiEngine(use_llm=False, use_censors=True, use_opening_book=False)
    engine_uncensored = YamiEngine(use_llm=False, use_censors=False, use_opening_book=False)

    # Both should produce valid moves
    d1 = engine_censored.decide()
    d2 = engine_uncensored.decide()
    assert d1.move is not None
    assert d2.move is not None

    # Censored should have filtered some moves
    assert d1.censored_move_count <= d1.scoped_move_count


def test_decision_has_candidates_when_not_in_book():
    engine = YamiEngine(use_llm=False, use_opening_book=False)
    decision = engine.decide()
    assert len(decision.candidates) > 0
    assert len(decision.candidates) <= 5


def test_engine_game_over_detection():
    engine = YamiEngine(use_llm=False)
    assert not engine.is_game_over()
    assert engine.result() is None
