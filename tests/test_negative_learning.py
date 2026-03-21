"""Tests for the Negative Learning module."""

import chess

from yami.navigator import NavigationVector
from yami.negative_learning import (
    LearnedCensorStack,
    NegativeExample,
    StrategyCensorRule,
    create_default_censors,
    mine_negative_examples_from_evals,
)


def test_create_default_censors():
    stack = create_default_censors()
    assert len(stack.strategy_rules) >= 3


def test_add_negative_example():
    stack = LearnedCensorStack()
    ex = NegativeExample(
        fen=chess.STARTING_FEN,
        move_uci="e2e4",
        nav_vector=(0, 0, 0, 0, 0, 1),
        position_type="opening",
        eval_before=50,
        eval_after=-200,
        eval_drop=250,
    )
    stack.add_negative_example(ex)
    assert len(stack.move_censors["opening"]) == 1
    assert "e2e4" in stack._bad_move_patterns["opening"]


def test_suppress_bad_move():
    stack = LearnedCensorStack()
    ex = NegativeExample(
        fen=chess.STARTING_FEN,
        move_uci="a2a4",
        nav_vector=(0, 0, 0, 0, 0, 1),
        position_type="opening",
        eval_before=0,
        eval_after=-300,
        eval_drop=300,
    )
    stack.add_negative_example(ex)

    board = chess.Board()
    nav = NavigationVector(0, 0, 0, 0, 0, 1)
    move = chess.Move.from_uci("a2a4")
    assert stack.should_suppress_move(board, move, nav, "opening")


def test_dont_suppress_good_move():
    stack = LearnedCensorStack()
    board = chess.Board()
    nav = NavigationVector(0, 0, 0, 0, 0, 1)
    move = chess.Move.from_uci("e2e4")
    assert not stack.should_suppress_move(board, move, nav, "opening")


def test_strategy_censor():
    stack = LearnedCensorStack()
    stack.add_strategy_rule(StrategyCensorRule(
        strategy="kingside_attack",
        nav_condition={"aggression": -1, "king_pressure": -1},
        confidence=0.9,
        examples_seen=100,
    ))
    board = chess.Board()
    nav = NavigationVector(-1, 0, 0, 0, -1, 0)
    # Any move should be suppressed when nav matches the condition
    move = chess.Move.from_uci("e2e4")
    assert stack.should_suppress_move(board, move, nav, "middlegame")


def test_filter_preserves_at_least_one():
    """Filter should never remove ALL moves."""
    stack = LearnedCensorStack()
    # Add many bad moves
    for uci in ["e2e4", "d2d4", "c2c4", "g1f3"]:
        stack.add_negative_example(NegativeExample(
            fen=chess.STARTING_FEN, move_uci=uci,
            nav_vector=(0, 0, 0, 0, 0, 1), position_type="opening",
            eval_before=0, eval_after=-300, eval_drop=300,
        ))

    board = chess.Board()
    nav = NavigationVector(0, 0, 0, 0, 0, 1)
    moves = list(board.legal_moves)
    filtered = stack.filter_moves(board, moves, nav)
    assert len(filtered) > 0


def test_mine_negative_examples():
    fens = [chess.STARTING_FEN, chess.STARTING_FEN]
    moves = ["e2e4", "a2a3"]
    evals_before = [50, 50]
    evals_after = [40, -200]  # second move is a blunder
    examples = mine_negative_examples_from_evals(
        fens, moves, evals_before, evals_after, threshold_cp=100
    )
    assert len(examples) == 1
    assert examples[0].move_uci == "a2a3"
    assert examples[0].eval_drop == 250
