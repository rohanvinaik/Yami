#!/usr/bin/env python3
"""Generate training data for the Yami neural Layer 7.

Uses Stockfish at depth 10 for move generation (diverse positions)
and depth 15 for candidate labeling (accurate labels).

Usage:
    python scripts/generate_data.py --num-train 10000 --num-eval 1000
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import chess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from yami.candidate_filter import filter_and_annotate
from yami.datagen.contracts import save_dataset
from yami.datagen.feature_extractor import board_to_example
from yami.datagen.label_oracle import label_candidates
from yami.models import AnnotatedCandidate
from yami.oracle import StockfishOracle
from yami.tactical_scoper import (
    apply_blunder_censor,
    apply_repetition_censor,
    apply_tactical_censor,
    scope_moves,
)


def extract_example(board, candidates, oracle_label):
    """Build a training example from infrastructure output + oracle label."""
    best_idx, best_cp, second_idx, gap = oracle_label
    return board_to_example(
        board,
        best_idx=best_idx,
        oracle_eval_cp=best_cp,
        second_best_idx=second_idx,
        eval_gap_cp=gap,
    )


def generate_positions_and_label(
    num_examples: int,
    play_oracle: StockfishOracle,
    label_oracle: StockfishOracle,
    progress_interval: int = 100,
):
    """Generate training examples from diverse game positions."""
    examples = []
    games = 0
    skipped = 0
    infra_agrees = 0
    infra_disagrees = 0
    t0 = time.time()

    while len(examples) < num_examples:
        board = chess.Board()
        games += 1

        # Play a game using Stockfish at low depth for diversity
        for move_num in range(200):
            if board.is_game_over():
                break

            # Extract training example from ~30% of middlegame positions
            if move_num >= 6 and move_num <= 150 and random.random() < 0.35:
                ex = _try_extract(board, label_oracle)
                if ex is not None:
                    # Track infrastructure agreement
                    if ex.best_candidate_idx == 0:
                        infra_agrees += 1
                    else:
                        infra_disagrees += 1
                    examples.append(ex)

                    if len(examples) % progress_interval == 0:
                        elapsed = time.time() - t0
                        rate = len(examples) / elapsed
                        agree_pct = infra_agrees / max(infra_agrees + infra_disagrees, 1) * 100
                        print(
                            f"  [{len(examples):>6}/{num_examples}] "
                            f"{rate:.1f} ex/s | "
                            f"games={games} | "
                            f"infra agrees={agree_pct:.1f}% | "
                            f"skipped={skipped}"
                        )

                    if len(examples) >= num_examples:
                        break
                else:
                    skipped += 1

            # Play a move (mix for diversity)
            r = random.random()
            if r < 0.6:
                move = play_oracle.best_move(board)
            elif r < 0.85:
                # Play a slightly sub-optimal move for diversity
                legal = list(board.legal_moves)
                if len(legal) > 1:
                    move = random.choice(legal[:min(5, len(legal))])
                else:
                    move = legal[0] if legal else None
            else:
                legal = list(board.legal_moves)
                move = random.choice(legal) if legal else None

            if move is None:
                break
            board.push(move)

    elapsed = time.time() - t0
    agree_pct = infra_agrees / max(infra_agrees + infra_disagrees, 1) * 100
    print(f"\nGeneration complete:")
    print(f"  Examples: {len(examples)}")
    print(f"  Games played: {games}")
    print(f"  Time: {elapsed:.1f}s ({len(examples)/elapsed:.1f} ex/s)")
    print(f"  Infrastructure agrees with Stockfish: {agree_pct:.1f}%")
    print(f"  Disagreement rate: {100 - agree_pct:.1f}%")
    print(f"  Skipped positions: {skipped}")

    return examples


def _try_extract(board, oracle):
    """Try to extract a training example from a position."""
    scoped = scope_moves(board)
    if len(scoped) < 3:
        return None

    censored = apply_blunder_censor(scoped)
    censored = apply_tactical_censor(censored, board)
    censored = apply_repetition_censor(censored, board)
    if not censored or len(censored) < 2:
        return None

    candidates, plan, profile = filter_and_annotate(board, censored)
    if len(candidates) < 2:
        return None

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


def main():
    parser = argparse.ArgumentParser(description="Generate Yami training data")
    parser.add_argument("--num-train", type=int, default=10000)
    parser.add_argument("--num-eval", type=int, default=1000)
    parser.add_argument("--play-depth", type=int, default=8)
    parser.add_argument("--label-depth", type=int, default=12)
    parser.add_argument("--output-dir", type=str, default="data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Yami Training Data Generation ===")
    print(f"Play depth: {args.play_depth}, Label depth: {args.label_depth}")
    print(f"Target: {args.num_train} train, {args.num_eval} eval\n")

    play_oracle = StockfishOracle(depth=args.play_depth, time_limit=0.05)
    label_oracle = StockfishOracle(depth=args.label_depth, time_limit=0.1)

    try:
        # Generate training data
        print("--- Generating training data ---")
        train_examples = generate_positions_and_label(
            args.num_train, play_oracle, label_oracle
        )
        train_path = output_dir / "chess_train.jsonl"
        save_dataset(train_examples, train_path)
        print(f"Saved to {train_path}\n")

        # Generate eval data (use different random seed for diversity)
        random.seed(42)
        print("--- Generating eval data ---")
        eval_examples = generate_positions_and_label(
            args.num_eval, play_oracle, label_oracle, progress_interval=50
        )
        eval_path = output_dir / "chess_eval.jsonl"
        save_dataset(eval_examples, eval_path)
        print(f"Saved to {eval_path}\n")

        # Summary statistics
        print("=== Dataset Summary ===")
        print(f"Train: {len(train_examples)} examples")
        print(f"Eval:  {len(eval_examples)} examples")

        # Distribution of best candidate index
        from collections import Counter
        train_dist = Counter(ex.best_candidate_idx for ex in train_examples)
        print(f"\nBest candidate distribution (train):")
        for idx in sorted(train_dist.keys()):
            pct = train_dist[idx] / len(train_examples) * 100
            print(f"  Candidate {idx}: {train_dist[idx]:>5} ({pct:.1f}%)")

        # Eval gap distribution
        gaps = [ex.eval_gap_cp for ex in train_examples if ex.eval_gap_cp > 0]
        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            print(f"\nAvg eval gap (best vs second): {avg_gap:.0f}cp")
            close = sum(1 for g in gaps if g < 30)
            print(f"Close decisions (<30cp gap): {close} ({close/len(gaps)*100:.1f}%)")

    finally:
        play_oracle.close()
        label_oracle.close()


if __name__ == "__main__":
    main()
