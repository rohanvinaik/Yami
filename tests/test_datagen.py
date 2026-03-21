"""Tests for the data generation pipeline."""

import json
import tempfile
from pathlib import Path

import chess

from yami.datagen.contracts import (
    CANDIDATE_FEAT_DIM,
    MOTIF_VOCAB,
    CandidateFeatures,
    ChessExample,
    load_dataset,
    save_dataset,
)
from yami.datagen.feature_extractor import board_to_example, extract_features


def _make_candidate(uci: str = "e2e4", san: str = "e4") -> CandidateFeatures:
    return CandidateFeatures(
        move_uci=uci, move_san=san,
        motif_flags=[0] * len(MOTIF_VOCAB),
        plan_alignment=0.0, positional_eval=0.0,
        risk_level=0, is_capture=False, is_check=False, see_value=0.0,
        piece_type_onehot=[0] * 6, target_centrality=0.0,
        dist_to_opp_king=0.0, dist_to_own_king=0.0,
        is_castling=False, opponent_mobility=0.0, pawn_structure_change=0.0,
    )


def _make_pad() -> CandidateFeatures:
    return _make_candidate(uci="0000", san="--")


def _make_example() -> ChessExample:
    cands = [_make_candidate() for _ in range(3)]
    cands.extend([_make_pad(), _make_pad()])
    return ChessExample(
        fen=chess.STARTING_FEN,
        material=1, structure=1, activity=0,
        safety=0, opponent_safety=0, tempo=1,
        plan_type=0, plan_activation=1.5,
        candidates=cands, num_candidates=3,
        game_phase=0.9, total_material=0.9, move_number=0.01,
        best_candidate_idx=0, oracle_eval_cp=50,
    )


def test_chess_example_roundtrip():
    ex = _make_example()
    json_str = ex.to_json()
    recovered = ChessExample.from_json(json_str)
    assert recovered.fen == ex.fen
    assert recovered.num_candidates == 3
    assert recovered.game_phase == 0.9
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
    assert len(cands) == 5
    assert 0 <= plan_idx <= 6
    assert "material" in profile
    # Verify new features are populated
    real_cands = [c for c in cands if c.move_uci != "0000"]
    assert len(real_cands) >= 2
    c0 = real_cands[0]
    assert len(c0.piece_type_onehot) == 6
    assert sum(c0.piece_type_onehot) == 1  # exactly one piece type


def test_board_to_example():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    ex = board_to_example(board, best_idx=0, oracle_eval_cp=30)
    assert ex.fen == board.fen()
    assert ex.num_candidates >= 2
    assert len(ex.candidates) == 5
    assert ex.game_phase > 0.0
    assert ex.move_number > 0.0


def test_candidate_feat_dim_matches():
    """Verify CANDIDATE_FEAT_DIM matches actual feature count."""
    assert CANDIDATE_FEAT_DIM == 30


def test_json_serialization_valid():
    ex = _make_example()
    parsed = json.loads(ex.to_json())
    assert isinstance(parsed["candidates"], list)
    assert parsed["num_candidates"] == 3
    assert "game_phase" in parsed
