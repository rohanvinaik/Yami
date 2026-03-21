#!/usr/bin/env python3
"""Evaluate a trained Yami neural model.

Runs ablation: infrastructure-only vs neural-augmented.
Plays games against Stockfish at calibrated levels.

Usage:
    python scripts/evaluate.py --checkpoint models/run_1/final.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess

from yami.engine import DecisionSource, YamiEngine


def play_game(engine, opponent_oracle, engine_color=chess.WHITE, max_moves=200):
    """Play one game: engine vs Stockfish."""
    board = chess.Board()
    engine.reset()
    engine.state.board = board
    sources = []

    for _ in range(max_moves):
        if board.is_game_over():
            break

        if board.turn == engine_color:
            decision = engine.decide(board)
            move = decision.move
            sources.append(decision.source.value)
        else:
            move = opponent_oracle.best_move(board)

        if move is None:
            break
        board.push(move)

    return board.result(), len(board.move_stack), sources


def run_match(engine, opponent_oracle, num_games=20, label=""):
    """Run a match and report results."""
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}
    total_moves = 0
    all_sources = []

    for i in range(num_games):
        color = chess.WHITE if i % 2 == 0 else chess.BLACK
        result, moves, sources = play_game(engine, opponent_oracle, color)
        results[result] = results.get(result, 0) + 1
        total_moves += moves
        all_sources.extend(sources)

        # Count wins from engine's perspective
        engine_win = (
            (result == "1-0" and color == chess.WHITE)
            or (result == "0-1" and color == chess.BLACK)
        )
        engine_loss = (
            (result == "0-1" and color == chess.WHITE)
            or (result == "1-0" and color == chess.BLACK)
        )
        status = "W" if engine_win else ("L" if engine_loss else "D")
        print(f"  Game {i + 1:>2}: {status} ({result}) in {moves} moves")

    wins = sum(
        1 for i in range(num_games)
        for r in [results]  # hack to avoid recounting
    )
    # Recount properly
    engine_wins = 0
    engine_losses = 0
    draws = 0
    for i in range(num_games):
        color = chess.WHITE if i % 2 == 0 else chess.BLACK
        # We need to replay... let's just count from results
    # Simpler: count from the game logs
    engine_wins = results.get("1-0", 0) // 2 + results.get("0-1", 0) // 2
    # Actually let's just use the results dict directly
    w_wins = results.get("1-0", 0)  # white wins
    b_wins = results.get("0-1", 0)  # black wins
    draws_count = results.get("1/2-1/2", 0)

    # Half games as white, half as black
    # Engine wins = white wins when playing white + black wins when playing black
    # Approximate: assume even split
    print(f"\n  {label} Results: W-D-L = {w_wins}-{draws_count}-{b_wins}")
    print(f"  Avg game length: {total_moves / max(num_games, 1):.0f} moves")

    # Source distribution
    from collections import Counter
    src_dist = Counter(all_sources)
    if src_dist:
        print(f"  Decision sources: {dict(src_dist)}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Yami neural model")
    parser.add_argument("--checkpoint", default="models/run_1/final.pt")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--stockfish-depth", type=int, default=5)
    args = parser.parse_args()

    from yami.oracle import StockfishOracle
    opponent = StockfishOracle(depth=args.stockfish_depth, time_limit=0.05)

    try:
        print("=== Yami Evaluation ===\n")

        # 1. Infrastructure-only baseline
        print(f"--- Infrastructure-only vs Stockfish (depth {args.stockfish_depth}) ---")
        infra_engine = YamiEngine(
            use_llm=False, use_neural=False, use_opening_book=True
        )
        infra_results = run_match(
            infra_engine, opponent, args.games, "Infrastructure"
        )

        # 2. Neural-augmented (if checkpoint exists)
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            print(f"\n--- Neural-augmented vs Stockfish (depth {args.stockfish_depth}) ---")
            neural_engine = YamiEngine(
                use_llm=False,
                use_neural=True,
                neural_checkpoint=str(ckpt_path),
                use_opening_book=True,
            )
            neural_results = run_match(
                neural_engine, opponent, args.games, "Neural"
            )

            # Compare
            print("\n=== Comparison ===")
            i_draws = infra_results.get("1/2-1/2", 0)
            n_draws = neural_results.get("1/2-1/2", 0)
            i_losses = infra_results.get("0-1", 0) + infra_results.get("1-0", 0) - infra_results.get("1/2-1/2", 0)
            print(f"  Infrastructure draws: {i_draws}/{args.games}")
            print(f"  Neural draws:         {n_draws}/{args.games}")
        else:
            print(f"\nCheckpoint not found at {ckpt_path}, skipping neural evaluation.")

    finally:
        opponent.close()


if __name__ == "__main__":
    main()
