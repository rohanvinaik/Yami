"""Dataset builder: generate training data from positions + Stockfish labels.

Orchestrates: positions → Yami infrastructure → Stockfish labeling → JSONL.
"""

from __future__ import annotations

import random
from pathlib import Path

import chess

from yami.candidate_filter import filter_and_annotate
from yami.datagen.contracts import ChessExample, save_dataset
from yami.datagen.feature_extractor import board_to_example
from yami.datagen.label_oracle import label_candidates
from yami.oracle import StockfishOracle
from yami.tactical_scoper import (
    apply_blunder_censor,
    apply_repetition_censor,
    apply_tactical_censor,
    scope_moves,
)


def generate_from_self_play(
    num_positions: int,
    oracle: StockfishOracle,
    output_path: Path,
    max_game_moves: int = 120,
    stockfish_play_depth: int = 8,
) -> list[ChessExample]:
    """Generate training data from random self-play positions.

    Uses Stockfish at low depth for move generation (to create diverse
    positions) and at high depth for labeling (to get accurate labels).
    """
    examples: list[ChessExample] = []
    games_played = 0

    while len(examples) < num_positions:
        board = chess.Board()
        games_played += 1

        for move_num in range(max_game_moves):
            if board.is_game_over():
                break

            # Every few moves, extract a training example
            if move_num >= 6 and random.random() < 0.3:
                ex = _extract_example(board, oracle)
                if ex is not None:
                    examples.append(ex)
                    if len(examples) >= num_positions:
                        break

            # Play a move (mix of Stockfish and random for diversity)
            if random.random() < 0.7:
                move = oracle.best_move(board)
            else:
                legal = list(board.legal_moves)
                move = random.choice(legal) if legal else None

            if move is None:
                break
            board.push(move)

    save_dataset(examples, output_path)
    return examples


def generate_from_positions(
    positions: list[str],
    oracle: StockfishOracle,
    output_path: Path,
) -> list[ChessExample]:
    """Generate training data from a list of FEN positions."""
    examples: list[ChessExample] = []

    for fen in positions:
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        ex = _extract_example(board, oracle)
        if ex is not None:
            examples.append(ex)

    save_dataset(examples, output_path)
    return examples


def _extract_example(
    board: chess.Board,
    oracle: StockfishOracle,
) -> ChessExample | None:
    """Extract one training example from a position."""
    # Run infrastructure pipeline
    scoped = scope_moves(board)
    censored = apply_blunder_censor(scoped)
    censored = apply_tactical_censor(censored, board)
    censored = apply_repetition_censor(censored, board)
    if not censored:
        censored = scoped

    candidates, plan, profile = filter_and_annotate(board, censored)
    if len(candidates) < 2:
        return None  # Need at least 2 candidates to learn from

    # Label with Stockfish
    best_idx, best_cp, second_idx, gap = label_candidates(
        board, candidates, oracle
    )

    return board_to_example(
        board,
        best_idx=best_idx,
        oracle_eval_cp=best_cp,
        second_best_idx=second_idx,
        eval_gap_cp=gap,
    )
