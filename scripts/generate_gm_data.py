#!/usr/bin/env python3
"""Generate synthetic GM data from Stockfish at high depth.

Plays Stockfish at depth 15 vs itself at depth 8, imports all moves
into the GM pattern database. This creates high-quality "GM-level"
move suggestions — what does the best engine do in each position?

Usage:
    python scripts/generate_gm_data.py --games 500 --db data/gm_patterns.db
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
import chess.engine

from yami.gm_patterns import GMPatternDB, _material_signature
from yami.navigator import compute_navigation_vector


def play_and_import(
    num_games: int,
    db_path: str,
    strong_depth: int = 15,
    weak_depth: int = 8,
    strong_time: float = 0.2,
    weak_time: float = 0.05,
) -> int:
    """Play games between Stockfish at different depths, import moves."""
    db = GMPatternDB(db_path)
    sf = chess.engine.SimpleEngine.popen_uci("stockfish")
    total_positions = 0

    try:
        for game_num in range(num_games):
            board = chess.Board()
            game_moves: list[tuple[str, str, str, int]] = []  # fen, uci, san, depth

            # Alternate who plays strong
            strong_color = chess.WHITE if game_num % 2 == 0 else chess.BLACK

            for move_num in range(200):
                if board.is_game_over():
                    break

                # Choose depth based on color
                if board.turn == strong_color:
                    sf.configure({"Skill Level": 20})
                    result = sf.play(
                        board,
                        chess.engine.Limit(depth=strong_depth, time=strong_time),
                    )
                    depth_used = strong_depth
                else:
                    # Weaker opponent for diversity
                    sf.configure({"Skill Level": random.randint(5, 15)})
                    result = sf.play(
                        board,
                        chess.engine.Limit(depth=weak_depth, time=weak_time),
                    )
                    depth_used = weak_depth

                move = result.move
                if move is None:
                    break

                # Record the strong player's moves (GM-quality)
                if board.turn == strong_color and move_num >= 4:
                    try:
                        san = board.san(move)
                        game_moves.append((
                            board.fen(), move.uci(), san, depth_used
                        ))
                    except (chess.InvalidMoveError, chess.IllegalMoveError):
                        pass

                board.push(move)

            # Determine game result
            game_result = board.result()

            # Import all strong-player moves
            for fen, uci, san, _depth in game_moves:
                pos_board = chess.Board(fen)
                nav = compute_navigation_vector(pos_board)
                mat_sig = _material_signature(pos_board)

                # Adjust result for side to move
                adjusted = game_result
                if pos_board.turn == chess.BLACK:
                    if game_result == "1-0":
                        adjusted = "0-1"
                    elif game_result == "0-1":
                        adjusted = "1-0"

                db.store_move(
                    mat_sig, nav.as_tuple(),
                    uci, san, adjusted,
                    elo=2800,  # Stockfish at depth 15 plays like 2800+
                )
                total_positions += 1

            if (game_num + 1) % 50 == 0:
                print(
                    f"  [{game_num + 1}/{num_games}] "
                    f"positions={total_positions} "
                    f"db_size={db.count()} "
                    f"last_result={game_result}"
                )

    finally:
        sf.quit()
        db.close()

    return total_positions


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic GM data")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--db", type=str, default="data/gm_patterns.db")
    parser.add_argument("--strong-depth", type=int, default=15)
    parser.add_argument("--weak-depth", type=int, default=8)
    args = parser.parse_args()

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)

    print("=== Synthetic GM Data Generation ===")
    print(f"Games: {args.games}")
    print(f"Strong depth: {args.strong_depth}, Weak depth: {args.weak_depth}")
    print()

    t0 = time.time()
    total = play_and_import(
        args.games, args.db,
        strong_depth=args.strong_depth,
        weak_depth=args.weak_depth,
    )
    elapsed = time.time() - t0

    print(f"\nDone: {total} positions imported in {elapsed:.0f}s")

    db = GMPatternDB(args.db)
    print(f"Total GM pattern entries: {db.count()}")
    db.close()


if __name__ == "__main__":
    main()
