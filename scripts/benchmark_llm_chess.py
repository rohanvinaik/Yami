#!/usr/bin/env python3
"""Run Yami against the llm_chess benchmark framework.

Integrates Yami as a custom chess engine agent in the llm_chess
benchmark, playing against Random Player and Stockfish at various
levels — the same opponents used for the leaderboard.

Usage:
    # vs Random Player (30 games)
    python scripts/benchmark_llm_chess.py --opponent random --games 30

    # vs Stockfish at skill level 5 (30 games)
    python scripts/benchmark_llm_chess.py --opponent stockfish --sf-level 5 --games 30

    # Full benchmark suite (Random + multiple SF levels)
    python scripts/benchmark_llm_chess.py --full-suite
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
import chess.engine

from yami.engine import YamiEngine


def play_game(
    yami: YamiEngine,
    opponent_type: str,
    sf_engine: chess.engine.SimpleEngine | None = None,
    sf_level: int | None = None,
    sf_time: float = 0.1,
    yami_color: chess.Color = chess.BLACK,
    max_moves: int = 200,
) -> dict:
    """Play one game in the llm_chess benchmark format.

    Returns a dict matching the benchmark's game_stats schema.
    """
    board = chess.Board()
    yami.reset()
    yami.state.board = board

    wrong_moves = 0
    wrong_actions = 0
    move_count = 0
    moves_san = []

    if sf_engine and sf_level is not None:
        sf_engine.configure({"Skill Level": sf_level})

    for _ in range(max_moves):
        if board.is_game_over():
            break

        if board.turn == yami_color:
            # Yami's turn
            decision = yami.decide(board)
            move = decision.move
            if move is None or move not in board.legal_moves:
                wrong_moves += 1
                # Fallback to first legal move
                legal = list(board.legal_moves)
                move = legal[0] if legal else None
        else:
            # Opponent's turn
            if opponent_type == "random":
                import random
                legal = list(board.legal_moves)
                move = random.choice(legal) if legal else None
            elif opponent_type == "stockfish" and sf_engine:
                result = sf_engine.play(board, chess.engine.Limit(time=sf_time))
                move = result.move
            else:
                move = None

        if move is None:
            break

        moves_san.append(board.san(move))
        board.push(move)
        move_count += 1

    # Determine result
    result = board.result()
    yami_won = (
        (result == "1-0" and yami_color == chess.WHITE)
        or (result == "0-1" and yami_color == chess.BLACK)
    )
    yami_lost = (
        (result == "0-1" and yami_color == chess.WHITE)
        or (result == "1-0" and yami_color == chess.BLACK)
    )

    # Material count (benchmark format)
    white_material = sum(
        {1: 1, 2: 3, 3: 3, 4: 5, 5: 9}.get(p.piece_type, 0)
        for p in board.piece_map().values()
        if p.color == chess.WHITE
    )
    black_material = sum(
        {1: 1, 2: 3, 3: 3, 4: 5, 5: 9}.get(p.piece_type, 0)
        for p in board.piece_map().values()
        if p.color == chess.BLACK
    )

    if yami_color == chess.BLACK:
        player_material = black_material
        opponent_material = white_material
    else:
        player_material = white_material
        opponent_material = black_material

    return {
        "result": result,
        "yami_won": yami_won,
        "yami_lost": yami_lost,
        "draw": not yami_won and not yami_lost,
        "moves": move_count,
        "wrong_moves": wrong_moves,
        "wrong_actions": wrong_actions,
        "player_material": player_material,
        "opponent_material": opponent_material,
        "moves_san": moves_san[:20],  # first 20 for display
    }


def run_match(
    opponent_type: str,
    num_games: int = 30,
    sf_level: int | None = None,
    sf_time: float = 0.1,
    checkpoint: str | None = None,
) -> dict:
    """Run a match and compute benchmark statistics."""
    use_neural = checkpoint is not None and Path(checkpoint).exists()
    yami = YamiEngine(
        use_llm=False,
        use_neural=use_neural,
        neural_checkpoint=checkpoint if use_neural else None,
        use_navigator=True,
        use_temporal=True,
        use_opening_book=True,
    )

    sf_engine = None
    if opponent_type == "stockfish":
        sf_engine = chess.engine.SimpleEngine.popen_uci("stockfish")

    wins, losses, draws = 0, 0, 0
    total_moves = 0
    total_wrong_moves = 0
    total_player_material = 0
    total_opponent_material = 0
    game_durations = []

    try:
        for i in range(num_games):
            color = chess.BLACK if i % 2 == 0 else chess.WHITE
            game = play_game(
                yami, opponent_type, sf_engine, sf_level, sf_time, color,
            )

            total_moves += game["moves"]
            total_wrong_moves += game["wrong_moves"]
            total_player_material += game["player_material"]
            total_opponent_material += game["opponent_material"]
            game_durations.append(game["moves"] / 200.0)  # as % of max

            if game["yami_won"]:
                wins += 1
            elif game["yami_lost"]:
                losses += 1
            else:
                draws += 1

            status = "W" if game["yami_won"] else ("L" if game["yami_lost"] else "D")
            if (i + 1) % 10 == 0:
                print(
                    f"  [{i + 1}/{num_games}] W={wins} D={draws} L={losses} "
                    f"last={status}({game['moves']}m)"
                )
    finally:
        if sf_engine:
            sf_engine.quit()

    # Compute benchmark-compatible statistics
    avg_moves = total_moves / max(num_games, 1)
    win_pct = wins / max(num_games, 1) * 100
    draw_pct = draws / max(num_games, 1) * 100
    wrong_per_1k = total_wrong_moves / max(total_moves, 1) * 1000
    avg_game_duration = sum(game_durations) / max(len(game_durations), 1)
    score = (wins + 0.5 * draws) / max(num_games, 1)

    return {
        "total_games": num_games,
        "player_wins": wins,
        "opponent_wins": losses,
        "draws": draws,
        "player_wins_percent": round(win_pct, 1),
        "player_draws_percent": round(draw_pct, 1),
        "average_moves": round(avg_moves, 1),
        "total_moves": total_moves,
        "wrong_moves_per_1000moves": round(wrong_per_1k, 1),
        "player_avg_material": round(total_player_material / max(num_games, 1), 1),
        "opponent_avg_material": round(total_opponent_material / max(num_games, 1), 1),
        "game_duration": round(avg_game_duration, 3),
        "score": round(score, 3),
    }


def estimate_elo(score: float, opponent_elo: float) -> float:
    s = max(0.001, min(0.999, score))
    return opponent_elo + (-400 * math.log10(1 / s - 1))


def main():
    parser = argparse.ArgumentParser(description="Yami llm_chess benchmark")
    parser.add_argument("--opponent", choices=["random", "stockfish"], default="random")
    parser.add_argument("--sf-level", type=int, default=None)
    parser.add_argument("--sf-time", type=float, default=0.1)
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--checkpoint", type=str, default="models/v4_nav/final.pt")
    parser.add_argument("--full-suite", action="store_true")
    args = parser.parse_args()

    if args.full_suite:
        run_full_suite(args.games, args.checkpoint, args.sf_time)
        return

    opponent_label = args.opponent
    if args.opponent == "stockfish" and args.sf_level is not None:
        opponent_label = f"stockfish-lvl-{args.sf_level}"

    print(f"{'='*60}")
    print(f"  YAMI vs {opponent_label} ({args.games} games)")
    print(f"{'='*60}")

    result = run_match(
        args.opponent, args.games, args.sf_level, args.sf_time, args.checkpoint,
    )

    print("\n--- Results ---")
    print(json.dumps(result, indent=2))


def run_full_suite(games_per: int, checkpoint: str, sf_time: float):
    """Run the full benchmark suite matching llm_chess methodology."""

    configs = [
        ("Random Player", "random", None, None),
        ("Stockfish Skill 0", "stockfish", 0, 800),
        ("Stockfish Skill 3", "stockfish", 3, 1200),
        ("Stockfish Skill 5", "stockfish", 5, 1500),
        ("Stockfish Skill 8", "stockfish", 8, 1800),
    ]

    print("=" * 70)
    print("  YAMI FULL BENCHMARK SUITE (llm_chess compatible)")
    print(f"  {games_per} games per opponent")
    print("=" * 70)

    all_results = {}
    t0 = time.time()

    for label, opp_type, sf_level, approx_elo in configs:
        print(f"\n--- {label} ---")
        result = run_match(opp_type, games_per, sf_level, sf_time, checkpoint)
        result["opponent"] = label
        result["opponent_approx_elo"] = approx_elo

        if approx_elo:
            est_elo = estimate_elo(result["score"], approx_elo)
            result["estimated_elo"] = round(est_elo, 0)
        elif result["score"] > 0:
            # vs Random ≈ 400 ELO
            est_elo = estimate_elo(result["score"], 400)
            result["estimated_elo"] = round(est_elo, 0)

        all_results[label] = result
        print(
            f"  W={result['player_wins']} D={result['draws']} "
            f"L={result['opponent_wins']} | "
            f"Score={result['score']:.1%} | "
            f"Avg={result['average_moves']:.0f}m | "
            f"Wrong={result['wrong_moves_per_1000moves']:.1f}/1k | "
            f"ELO≈{result.get('estimated_elo', '?')}"
        )

    elapsed = time.time() - t0

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY (total: {elapsed:.0f}s)")
    print(f"{'='*70}")
    print(
        f"{'Opponent':25s} {'W':>3} {'D':>3} {'L':>3} "
        f"{'Score':>7} {'AvgMov':>6} {'Wrong':>6} {'ELO':>6}"
    )
    print("-" * 65)
    for label, r in all_results.items():
        print(
            f"{label:25s} {r['player_wins']:>3} {r['draws']:>3} "
            f"{r['opponent_wins']:>3} {r['score']:>6.1%} "
            f"{r['average_moves']:>6.0f} "
            f"{r['wrong_moves_per_1000moves']:>5.1f} "
            f"{r.get('estimated_elo', '?'):>6}"
        )

    # Save results
    output_path = Path("results/benchmark_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
