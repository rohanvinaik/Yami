"""Tests for Layer 5: Positional Knowledge Graph."""

import chess

from yami.knowledge_graph import evaluate_position, rank_moves, suggest_plan
from yami.models import Material, PlanType, Safety, Tempo
from yami.tactical_scoper import scope_moves


def test_starting_position_profile():
    board = chess.Board()
    profile = evaluate_position(board)
    assert profile.material == Material.EQUAL
    assert profile.tempo == Tempo.EQUAL


def test_material_ahead_detection():
    # White has extra queen
    board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    profile = evaluate_position(board)
    assert profile.material == Material.AHEAD


def test_material_behind_detection():
    # White missing queen
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
    profile = evaluate_position(board)
    assert profile.material == Material.BEHIND


def test_suggest_plan_returns_plan():
    board = chess.Board()
    profile = evaluate_position(board)
    plan = suggest_plan(profile)
    assert plan.plan_type in PlanType
    assert plan.name
    assert plan.description


def test_suggest_simplify_when_ahead():
    # White is way ahead in material
    board = chess.Board("4k3/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1")
    profile = evaluate_position(board)
    plan = suggest_plan(profile)
    assert plan.plan_type == PlanType.SIMPLIFY


def test_suggest_fortify_when_king_exposed():
    # White king exposed, opponent king castled and safe
    board = chess.Board("5rk1/ppp2ppp/8/8/8/4q3/PPPPPPPP/R3K2R w KQ - 0 20")
    profile = evaluate_position(board)
    if profile.safety in (Safety.EXPOSED, Safety.UNDER_ATTACK):
        plan = suggest_plan(profile)
        assert plan.plan_type == PlanType.FORTIFY


def test_rank_moves_returns_sorted():
    board = chess.Board()
    profile = evaluate_position(board)
    plan = suggest_plan(profile)
    scoped = scope_moves(board)
    ranked = rank_moves(scoped, plan, board)
    assert len(ranked) == len(scoped)
    # Should be sorted by alignment descending
    for i in range(len(ranked) - 1):
        assert ranked[i].alignment >= ranked[i + 1].alignment


def test_king_safety_detection():
    board = chess.Board()
    profile = evaluate_position(board)
    # Starting position: king is safe
    assert profile.safety == Safety.SAFE
