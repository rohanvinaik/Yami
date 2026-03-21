#!/usr/bin/env python3
"""Mine K-line patterns from Stockfish games.

Plays games using Stockfish, extracts winning move sequences,
and stores them in a SQLite K-line database.

Usage:
    python scripts/mine_klines.py --games 500 --db data/klines.db
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess

from yami.kline_memory import KLineMemory, mine_kline_from_game
from yami.oracle import StockfishOracle


def play_and_mine(
    oracle: StockfishOracle,
    num_games: int,
    db_path: str,
    play_depth: int = 8,
    eval_depth: int = 10,
) -> int:
    """Play games and mine K-line patterns."""
    memory = KLineMemory(db_path)
    eval_oracle = StockfishOracle(depth=eval_depth, time_limit=0.1)
    total_patterns = 0

    try:
        for game_num in range(num_games):
            board = chess.Board()
            moves: list[chess.Move] = []
            evals: list[int] = []

            # Play a full game
            for _ in range(200):
                if board.is_game_over():
                    break

                # Get eval before move
                ev = eval_oracle.evaluate(board)
                eval_cp = ev.score_cp if ev.score_cp is not None else 0
                if ev.mate_in is not None:
                    eval_cp = 10000 * (1 if ev.mate_in > 0 else -1)
                # Normalize to white's perspective
                if board.turn == chess.BLACK:
                    eval_cp = -eval_cp
                evals.append(eval_cp)

                # Play move (mix of Stockfish and random for diversity)
                r = random.random()
                if r < 0.65:
                    move = oracle.best_move(board)
                elif r < 0.85:
                    legal = list(board.legal_moves)
                    move = random.choice(legal[:min(5, len(legal))])
                else:
                    legal = list(board.legal_moves)
                    move = random.choice(legal)

                if move is None:
                    break
                moves.append(move)
                board.push(move)

            # Mine K-line patterns from this game
            if len(moves) >= 8:
                patterns = mine_kline_from_game(moves, evals)
                for p in patterns:
                    memory.store(p)
                    total_patterns += 1

            if (game_num + 1) % 50 == 0:
                print(
                    f"  [{game_num + 1}/{num_games}] "
                    f"patterns={total_patterns} "
                    f"db_size={memory.count()}"
                )
    finally:
        eval_oracle.close()
        memory.close()

    return total_patterns


def main():
    parser = argparse.ArgumentParser(description="Mine K-line patterns")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--db", type=str, default="data/klines.db")
    parser.add_argument("--play-depth", type=int, default=6)
    parser.add_argument("--eval-depth", type=int, default=8)
    args = parser.parse_args()

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)

    print("=== K-Line Pattern Mining ===")
    print(f"Games: {args.games}")
    print(f"Play depth: {args.play_depth}, Eval depth: {args.eval_depth}")
    print()

    oracle = StockfishOracle(depth=args.play_depth, time_limit=0.05)
    try:
        t0 = time.time()
        total = play_and_mine(
            oracle, args.games, args.db,
            play_depth=args.play_depth,
            eval_depth=args.eval_depth,
        )
        elapsed = time.time() - t0
        print(f"\nDone: {total} patterns mined in {elapsed:.0f}s")
        print(f"Database: {args.db}")

        # Show stats
        mem = KLineMemory(args.db)
        print(f"Total entries: {mem.count()}")
        mem.close()
    finally:
        oracle.close()


if __name__ == "__main__":
    main()
