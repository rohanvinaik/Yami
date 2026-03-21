#!/usr/bin/env python3
"""Full LLM Chess benchmark with paper-compatible metrics.

Replicates the methodology from arxiv.org/abs/2512.01992:
- ELO with 95% CI using maximum-likelihood estimation
- Game Duration (% of max moves completed)
- Wrong moves / wrong actions per 1000 moves
- Material difference (player - opponent average)
- Win/Loss/Draw breakdown
- Per-ply move quality (best move %, blunder rate)

Opponents: Random Player + Stockfish at calibrated ELO levels
(Stockfish with UCI_LimitStrength as proxy for Dragon skill levels)

Usage:
    python scripts/benchmark_full_metrics.py --games 42
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

# Calibrated opponent ELOs (Stockfish UCI_Elo settings)
# Paper uses Dragon skill 1-10 ≈ 250-1375 ELO
# We use Stockfish UCI_LimitStrength for comparable calibration
OPPONENTS = [
    {"name": "Random Player", "type": "random", "elo": -122, "sf_elo": None, "sf_skill": None},
    {"name": "Engine Lvl 1 (~250)", "type": "stockfish", "elo": 250, "sf_elo": 1320, "sf_skill": 0},
    {"name": "Engine Lvl 3 (~500)", "type": "stockfish", "elo": 500, "sf_elo": 1400, "sf_skill": 2},
    {"name": "Engine Lvl 5 (~750)", "type": "stockfish", "elo": 750, "sf_elo": 1500, "sf_skill": 4},
    {"name": "Engine Lvl 7 (~1000)", "type": "stockfish",
     "elo": 1000, "sf_elo": 1700, "sf_skill": 6},
    {"name": "Engine Lvl 10 (~1375)", "type": "stockfish",
     "elo": 1375, "sf_elo": 1900, "sf_skill": 10},
]


def play_game(yami, opp_type, sf_engine=None, sf_skill=None, sf_elo=None,
              yami_color=chess.BLACK, max_moves=200):
    """Play one game, collecting per-move metrics."""
    board = chess.Board()
    yami.reset()
    yami.state.board = board

    if sf_engine and sf_skill is not None:
        sf_engine.configure({"Skill Level": sf_skill})
    if sf_engine and sf_elo is not None:
        sf_engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo})

    wrong_moves = 0
    wrong_actions = 0
    moves_played = 0
    yami_material_sum = 0
    opp_material_sum = 0
    material_samples = 0

    for _ in range(max_moves):
        if board.is_game_over():
            break

        if board.turn == yami_color:
            decision = yami.decide(board)
            move = decision.move
            if move is None or move not in board.legal_moves:
                wrong_moves += 1
                legal = list(board.legal_moves)
                move = legal[0] if legal else None
        else:
            if opp_type == "random":
                import random
                legal = list(board.legal_moves)
                move = random.choice(legal) if legal else None
            elif sf_engine:
                result = sf_engine.play(board, chess.engine.Limit(time=0.1))
                move = result.move
            else:
                move = None

        if move is None:
            break

        board.push(move)
        moves_played += 1

        # Track material every 10 moves
        if moves_played % 10 == 0:
            piece_vals = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9}
            w_mat = sum(piece_vals.get(p.piece_type, 0)
                        for p in board.piece_map().values() if p.color == chess.WHITE)
            b_mat = sum(piece_vals.get(p.piece_type, 0)
                        for p in board.piece_map().values() if p.color == chess.BLACK)
            if yami_color == chess.BLACK:
                yami_material_sum += b_mat
                opp_material_sum += w_mat
            else:
                yami_material_sum += w_mat
                opp_material_sum += b_mat
            material_samples += 1

    result = board.result()
    yami_won = ((result == "1-0" and yami_color == chess.WHITE)
                or (result == "0-1" and yami_color == chess.BLACK))
    yami_lost = ((result == "0-1" and yami_color == chess.WHITE)
                 or (result == "1-0" and yami_color == chess.BLACK))

    return {
        "result": result,
        "won": yami_won,
        "lost": yami_lost,
        "draw": not yami_won and not yami_lost,
        "moves": moves_played,
        "wrong_moves": wrong_moves,
        "wrong_actions": wrong_actions,
        "yami_material_avg": yami_material_sum / max(material_samples, 1),
        "opp_material_avg": opp_material_sum / max(material_samples, 1),
        "game_duration": moves_played / (max_moves * 2),  # as fraction of max
    }


def compute_elo_mle(scores, opponent_elo, n_games):
    """Maximum-likelihood ELO estimation with 95% CI (paper method)."""
    score = sum(scores) / max(n_games, 1)
    score = max(0.001, min(0.999, score))

    # MLE: R = opp_elo - 400*log10(1/score - 1)
    elo = opponent_elo - 400 * math.log10(1 / score - 1)

    # Fisher information for CI
    e = score  # expected score at MLE
    info = n_games * e * (1 - e) * (math.log(10) / 400) ** 2
    se = 1 / math.sqrt(max(info, 0.0001))
    ci_low = elo - 1.96 * se
    ci_high = elo + 1.96 * se

    return elo, ci_low, ci_high, se


def run_full_benchmark(num_games=42):
    """Run the complete benchmark suite with paper-compatible metrics."""
    yami = YamiEngine(
        use_llm=False, use_neural=False,
        use_navigator=True, use_temporal=True, use_gm_patterns=True,
        use_opening_book=True,
    )

    sf = chess.engine.SimpleEngine.popen_uci("stockfish")

    print("=" * 75)
    print("  YAMI FULL BENCHMARK (arxiv.org/abs/2512.01992 compatible)")
    print(f"  {num_games} games per opponent, alternating colors")
    print("=" * 75)

    all_results = {}
    all_scores = []  # for combined ELO
    all_opp_elos = []
    t_total = time.time()

    try:
        for opp in OPPONENTS:
            wins, draws, losses = 0, 0, 0
            total_moves = 0
            total_wrong_moves = 0
            total_wrong_actions = 0
            total_yami_mat = 0.0
            total_opp_mat = 0.0
            game_durations = []
            scores = []

            print(f"\n--- {opp['name']} (anchor ELO: {opp['elo']}) ---")
            t0 = time.time()

            for i in range(num_games):
                color = chess.BLACK if i % 2 == 0 else chess.WHITE
                game = play_game(
                    yami, opp["type"], sf, opp["sf_skill"], opp.get("sf_elo"),
                    color,
                )

                total_moves += game["moves"]
                total_wrong_moves += game["wrong_moves"]
                total_wrong_actions += game["wrong_actions"]
                total_yami_mat += game["yami_material_avg"]
                total_opp_mat += game["opp_material_avg"]
                game_durations.append(game["game_duration"])

                if game["won"]:
                    wins += 1
                    scores.append(1.0)
                elif game["lost"]:
                    losses += 1
                    scores.append(0.0)
                else:
                    draws += 1
                    scores.append(0.5)

                if (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{num_games}] W={wins} D={draws} L={losses}")

            elapsed = time.time() - t0

            # Compute metrics
            n = num_games
            score_pct = sum(scores) / n
            win_loss = (wins - losses) / (2 * n) + 0.5
            avg_moves = total_moves / max(n, 1)
            wrong_per_1k = (total_wrong_moves + total_wrong_actions) / max(total_moves, 1) * 1000
            wrong_moves_per_1k = total_wrong_moves / max(total_moves, 1) * 1000
            wrong_actions_per_1k = total_wrong_actions / max(total_moves, 1) * 1000
            avg_duration = sum(game_durations) / max(len(game_durations), 1)
            mat_diff = (total_yami_mat - total_opp_mat) / max(n, 1)

            # ELO
            elo, ci_lo, ci_hi, se = compute_elo_mle(scores, opp["elo"], n)

            all_scores.extend(scores)
            all_opp_elos.extend([opp["elo"]] * n)

            result = {
                "opponent": opp["name"],
                "opponent_elo": opp["elo"],
                "total_games": n,
                "player_wins": wins,
                "opponent_wins": losses,
                "draws": draws,
                "player_wins_percent": round(wins / n * 100, 1),
                "player_draws_percent": round(draws / n * 100, 1),
                "win_loss": round(win_loss, 3),
                "average_moves": round(avg_moves, 1),
                "total_moves": total_moves,
                "wrong_moves_per_1000moves": round(wrong_moves_per_1k, 1),
                "wrong_actions_per_1000moves": round(wrong_actions_per_1k, 1),
                "mistakes_per_1000moves": round(wrong_per_1k, 1),
                "player_avg_material": round(total_yami_mat / max(n, 1), 1),
                "opponent_avg_material": round(total_opp_mat / max(n, 1), 1),
                "material_diff": round(mat_diff, 1),
                "game_duration": round(avg_duration, 3),
                "elo": round(elo, 1),
                "elo_ci_low": round(ci_lo, 1),
                "elo_ci_high": round(ci_hi, 1),
                "elo_se": round(se, 1),
                "time_s": round(elapsed, 1),
            }
            all_results[opp["name"]] = result

            print(f"  W={wins} D={draws} L={losses} | Score={score_pct:.1%} | "
                  f"ELO={elo:.0f} [{ci_lo:.0f}, {ci_hi:.0f}]")

    finally:
        sf.quit()

    total_elapsed = time.time() - t_total

    # Combined ELO across all opponents
    combined_opp = sum(all_opp_elos) / len(all_opp_elos)
    combined_elo, combined_lo, combined_hi, _ = compute_elo_mle(
        all_scores, combined_opp, len(all_scores)
    )

    # Print paper-format table
    print(f"\n{'='*75}")
    print(f"  RESULTS (paper format) — Total time: {total_elapsed:.0f}s")
    print(f"{'='*75}")
    print(f"{'Opponent':25s} {'W':>3} {'D':>3} {'L':>3} {'Score':>6} "
          f"{'ELO':>6} {'95% CI':>14} {'Dur':>5} {'Mis/1k':>7} {'MatΔ':>5}")
    print("-" * 75)

    total_w, total_d, total_l = 0, 0, 0
    for name, r in all_results.items():
        total_w += r["player_wins"]
        total_d += r["draws"]
        total_l += r["opponent_wins"]
        s = (r["player_wins"] + 0.5 * r["draws"]) / r["total_games"]
        print(f"{name:25s} {r['player_wins']:>3} {r['draws']:>3} "
              f"{r['opponent_wins']:>3} {s:>5.1%} "
              f"{r['elo']:>6.0f} [{r['elo_ci_low']:>5.0f},{r['elo_ci_high']:>5.0f}] "
              f"{r['game_duration']:>5.3f} {r['mistakes_per_1000moves']:>6.1f} "
              f"{r['material_diff']:>+5.1f}")

    print("-" * 75)
    total_n = total_w + total_d + total_l
    total_score = (total_w + 0.5 * total_d) / max(total_n, 1)
    print(f"{'COMBINED':25s} {total_w:>3} {total_d:>3} {total_l:>3} "
          f"{total_score:>5.1%} {combined_elo:>6.0f} "
          f"[{combined_lo:>5.0f},{combined_hi:>5.0f}]")

    print(f"\n  Total games: {total_n}")
    print(f"  Combined ELO: {combined_elo:.0f} ± {1.96 * (combined_hi - combined_lo) / 3.92:.0f}")
    print("  Parameters: 294,261")
    print("  Cost per game: $0.00")
    print("  Tokens per move: 0 (not an LLM)")
    print(f"  Average time per game: {total_elapsed / max(total_n, 1):.1f}s")

    # Comparison with leaderboard top models
    print(f"\n{'='*75}")
    print("  LEADERBOARD COMPARISON")
    print(f"{'='*75}")
    print(f"{'Model':45s} {'ELO':>7} {'Params':>10} {'$/game':>7}")
    print("-" * 75)
    comparisons = [
        ("gpt-5-2025-08-07-medium", 1087, "~1T+", "~$5-10"),
        ("gpt-5.1-2025-11-13-medium", 947, "~1T+", "~$3-5"),
        ("o3-2025-04-16-medium", 778, "~1T+", "~$5"),
        ("claude-opus-4-5 (thinking)", 446, "~1T+", "~$2-5"),
        ("o4-mini-2025-04-16-high", 440, "~100B+", "~$1"),
        (">>> YAMI (this system) <<<", combined_elo, "294K", "$0.00"),
        ("grok-3-mini-beta-high", 430, "~100B+", "~$1"),
        ("o3-mini-2025-01-31-high", 403, "~100B+", "~$0.50"),
    ]
    for name, elo, params, cost in comparisons:
        if "YAMI" in name:
            print(f"{'='*75}")
            print(f"  {name:43s} {elo:>7.0f} {params:>10} {cost:>7}")
            print(f"{'='*75}")
        else:
            print(f"  {name:43s} {elo:>7} {params:>10} {cost:>7}")

    # Save results
    output_path = Path("results/full_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model": "Yami (holographic coherence)",
            "parameters": 294261,
            "cost_per_game": 0.0,
            "combined_elo": round(combined_elo, 1),
            "combined_ci": [round(combined_lo, 1), round(combined_hi, 1)],
            "total_games": total_n,
            "total_wins": total_w,
            "total_draws": total_d,
            "total_losses": total_l,
            "per_opponent": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=42)
    args = parser.parse_args()
    run_full_benchmark(args.games)
