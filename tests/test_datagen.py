"""Tests for the data generation pipeline."""

import json
import tempfile
from pathlib import Path

import chess

from yami.datagen.contracts import (
    MOTIF_VOCAB,
    CandidateFeatures,
    ChessExample,
    load_dataset,
    save_dataset,
)
from yami.datagen.feature_extractor import board_to_example, extract_features


def _make_example() -> ChessExample:
    """Create a minimal valid ChessExample."""
    cands = []
    for i in range(3):
        cands.append(CandidateFeatures(
            move_uci=f"e2e{4 + i}",
            move_san=f"e{4 + i}",
            motif_flags=[0] * len(MOTIF_VOCAB),
            plan_alignment=float(i),
            positional_eval=0.0,
            risk_level=0,
            is_capture=False,
            is_check=False,
            see_value=0.0,
        ))
    # Pad to 5
    pad = CandidateFeatures(
        move_uci="0000", move_san="--",
        motif_flags=[0] * len(MOTIF_VOCAB),
        plan_alignment=0.0, positional_eval=0.0,
        risk_level=0, is_capture=False, is_check=False, see_value=0.0,
    )
    cands.extend([pad, pad])

    return ChessExample(
        fen=chess.STARTING_FEN,
        material=1, structure=1, activity=0,
        safety=0, opponent_safety=0, tempo=1,
        plan_type=0, plan_activation=1.5,
        candidates=cands,
        num_candidates=3,
        best_candidate_idx=0,
        oracle_eval_cp=50,
    )


def test_chess_example_roundtrip():
    ex = _make_example()
    json_str = ex.to_json()
    recovered = ChessExample.from_json(json_str)
    assert recovered.fen == ex.fen
    assert recovered.num_candidates == 3
    assert recovered.best_candidate_idx == 0
    assert len(recovered.candidates) == 5


def test_save_load_dataset():
    examples = [_make_example() for _ in range(5)]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test.jsonl"
        save_dataset(examples, path)
        loaded = load_dataset(path)
        assert len(loaded) == 5
        assert loaded[0].fen == chess.STARTING_FEN


def test_extract_features_starting_position():
    board = chess.Board()
    cands, profile, plan_idx = extract_features(board)
    assert len(cands) == 5  # padded to 5
    assert 0 <= plan_idx <= 6
    assert "material" in profile
    assert "safety" in profile


def test_board_to_example():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    ex = board_to_example(board, best_idx=0, oracle_eval_cp=30)
    assert ex.fen == board.fen()
    assert ex.best_candidate_idx == 0
    assert ex.num_candidates >= 2
    assert len(ex.candidates) == 5


def test_json_serialization_valid():
    ex = _make_example()
    parsed = json.loads(ex.to_json())
    assert isinstance(parsed["candidates"], list)
    assert parsed["num_candidates"] == 3
