#!/usr/bin/env python3
"""Benchmark Yami's ELO against Stockfish at calibrated skill levels.

Replicates the llm_chess benchmark methodology:
- Play games against Stockfish at skill levels 0-20
- Skill level 0 ≈ 1320 ELO, Level 5 ≈ 1700, Level 10 ≈ 2000,
  Level 15 ≈ 2300, Level 20 ≈ 3200 (approximate)
- Compute ELO from win/draw/loss ratios

Usage:
    python scripts/benchmark_elo.py --games 30
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
import chess.engine

from yami.engine import YamiEngine

# Stockfish skill level to approximate ELO mapping
# Based on community testing and Lichess calibration
SKILL_TO_ELO = {
    0: 1320,
    1: 1420,
    2: 1500,
    3: 1580,
    4: 1650,
    5: 1700,
    6: 1780,
    7: 1850,
    8: 1900,
    10: 2000,
    12: 2100,
    15: 2300,
    20: 3200,
}


def play_game(
    engine: YamiEngine,
    sf_engine: chess.engine.SimpleEngine,
    sf_skill: int,
    sf_time: float,
    engine_color: chess.Color,
    max_moves: int = 200,
) -> tuple[str, int]:
    """Play one game. Returns (result, move_count)."""
    board = chess.Board()
    engine.reset()
    engine.state.board = board

    sf_engine.configure({"Skill Level": sf_skill})

    for _ in range(max_moves):
        if board.is_game_over():
            break

        if board.turn == engine_color:
            d = engine.decide(board)
            move = d.move
        else:
            result = sf_engine.play(board, chess.engine.Limit(time=sf_time))
            move = result.move

        if move is None:
            break
        board.push(move)

    return board.result(), len(board.move_stack)


def estimate_elo(wins: int, draws: int, losses: int, opponent_elo: float) -> float:
    """Estimate ELO from game results."""
    total = wins + draws + losses
    if total == 0:
        return opponent_elo

    score = (wins + 0.5 * draws) / total
    score = max(0.001, min(0.999, score))
    elo_diff = -400 * math.log10(1 / score - 1)
    return opponent_elo + elo_diff


def main():
    parser = argparse.ArgumentParser(description="Benchmark Yami ELO")
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--checkpoint", type=str, default="models/v4_nav/final.pt")
    parser.add_argument("--sf-time", type=float, default=0.1)
    parser.add_argument(
        "--levels", type=str, default="0,3,5,8",
        help="Comma-separated Stockfish skill levels to test",
    )
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",")]

    # Build Yami engine
    ckpt = Path(args.checkpoint)
    use_neural = ckpt.exists()
    engine = YamiEngine(
        use_llm=False,
        use_neural=use_neural,
        neural_checkpoint=str(ckpt) if use_neural else None,
        use_navigator=True,
        use_temporal=True,
        use_opening_book=True,
    )
    config_label = "full stack" if use_neural else "navigator only"

    print("=" * 65)
    print(f"  YAMI ELO BENCHMARK ({config_label})")
    print(f"  {args.games} games per skill level, SF time={args.sf_time}s")
    print("=" * 65)

    sf = chess.engine.SimpleEngine.popen_uci("stockfish")

    try:
        all_results = {}
        for skill in levels:
            opp_elo = SKILL_TO_ELO.get(skill, 1320 + skill * 80)
            wins, draws, losses = 0, 0, 0
            total_moves = 0

            print(f"\n--- Skill Level {skill} (~{opp_elo} ELO) ---")
            t0 = time.time()

            for i in range(args.games):
                color = chess.WHITE if i % 2 == 0 else chess.BLACK
                result, moves = play_game(
                    engine, sf, skill, args.sf_time, color,
                )
                total_moves += moves

                ew = (
                    (result == "1-0" and color == chess.WHITE)
                    or (result == "0-1" and color == chess.BLACK)
                )
                el = (
                    (result == "0-1" and color == chess.WHITE)
                    or (result == "1-0" and color == chess.BLACK)
                )
                if ew:
                    wins += 1
                elif el:
                    losses += 1
                else:
                    draws += 1

                status = "W" if ew else ("L" if el else "D")
                if (i + 1) % 10 == 0:
                    print(
                        f"  [{i + 1}/{args.games}] "
                        f"W={wins} D={draws} L={losses} "
                        f"last={status}({moves}m)"
                    )

            elapsed = time.time() - t0
            avg_moves = total_moves / max(args.games, 1)
            est_elo = estimate_elo(wins, draws, losses, opp_elo)
            score = (wins + 0.5 * draws) / max(args.games, 1)

            all_results[skill] = {
                "opponent_elo": opp_elo,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "avg_moves": round(avg_moves, 1),
                "score": round(score, 3),
                "estimated_elo": round(est_elo, 0),
                "time": round(elapsed, 1),
            }

            print(
                f"  Result: W={wins} D={draws} L={losses} | "
                f"Score={score:.1%} | Avg={avg_moves:.0f} moves | "
                f"ELO estimate: {est_elo:.0f}"
            )

        # Summary
        print("\n" + "=" * 65)
        print("  RESULTS SUMMARY")
        print("=" * 65)
        print(
            f"{'Level':>6} {'Opp ELO':>8} {'W':>3} {'D':>3} {'L':>3} "
            f"{'Score':>7} {'Avg Mov':>7} {'Est ELO':>8}"
        )
        print("-" * 55)

        elo_estimates = []
        for skill in levels:
            r = all_results[skill]
            print(
                f"{skill:>6} {r['opponent_elo']:>8} "
                f"{r['wins']:>3} {r['draws']:>3} {r['losses']:>3} "
                f"{r['score']:>6.1%} {r['avg_moves']:>7} "
                f"{r['estimated_elo']:>8.0f}"
            )
            elo_estimates.append(r["estimated_elo"])

        avg_elo = sum(elo_estimates) / len(elo_estimates)
        print(f"\n  Average estimated ELO: {avg_elo:.0f}")
        print(f"  Range: {min(elo_estimates):.0f} - {max(elo_estimates):.0f}")

    finally:
        sf.quit()


if __name__ == "__main__":
    main()
