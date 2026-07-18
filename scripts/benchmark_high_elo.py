#!/usr/bin/env python3
"""Focused benchmark at high ELO levels with neural model support.

Tests specifically at 2000, 2500, 2800, and 3190 ELO to measure
win rate improvements from elite fine-tuning.

Usage:
    # Infrastructure only (baseline)
    python scripts/benchmark_high_elo.py --games 20

    # With neural model
    python scripts/benchmark_high_elo.py --games 20 \
        --checkpoint results/phase2_200k/phase2_A/final.pt

    # After elite fine-tuning
    python scripts/benchmark_high_elo.py --games 20 \
        --checkpoint results/elite_finetune/best.pt
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

HIGH_ELO_OPPONENTS = [
    {"name": "SF ELO 2000", "sf_elo": 2000, "sf_skill": 15},
    {"name": "SF ELO 2500", "sf_elo": 2500, "sf_skill": 19},
    {"name": "SF ELO 2800", "sf_elo": 2800, "sf_skill": 20},
    {"name": "SF ELO 3190", "sf_elo": 3190, "sf_skill": 20},
]


def play_game(yami, sf_engine, sf_elo, sf_skill, yami_color, max_moves=200):
    """Play one game against calibrated Stockfish."""
    board = chess.Board()
    yami.reset()
    yami.state.board = board

    sf_engine.configure({
        "Skill Level": sf_skill,
        "UCI_LimitStrength": True,
        "UCI_Elo": sf_elo,
    })

    moves_played = 0
    for _ in range(max_moves):
        if board.is_game_over():
            break

        if board.turn == yami_color:
            decision = yami.decide(board)
            move = decision.move
            if move is None or move not in board.legal_moves:
                legal = list(board.legal_moves)
                move = legal[0] if legal else None
        else:
            result = sf_engine.play(board, chess.engine.Limit(time=0.1))
            move = result.move

        if move is None:
            break

        board.push(move)
        moves_played += 1

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
    }


def compute_elo_mle(scores, opponent_elo, n_games):
    """MLE ELO estimation with 95% CI."""
    score = sum(scores) / max(n_games, 1)
    score = max(0.001, min(0.999, score))
    elo = opponent_elo - 400 * math.log10(1 / score - 1)
    e = score
    info = n_games * e * (1 - e) * (math.log(10) / 400) ** 2
    se = 1 / math.sqrt(max(info, 0.0001))
    return elo, elo - 1.96 * se, elo + 1.96 * se


def main():
    parser = argparse.ArgumentParser(description="High-ELO focused benchmark")
    parser.add_argument("--games", type=int, default=20,
                        help="Games per opponent (alternating colors)")
    parser.add_argument("--checkpoint", default=None,
                        help="Neural model checkpoint (None = infrastructure only)")
    parser.add_argument("--label", default=None,
                        help="Label for this run (e.g., 'phase2', 'elite_v1')")
    parser.add_argument("--output", default=None,
                        help="JSON output path")
    args = parser.parse_args()

    use_neural = args.checkpoint is not None
    if use_neural and not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    label = args.label or ("neural" if use_neural else "infra_only")

    # Build engine
    engine_kwargs = dict(
        use_llm=False,
        use_navigator=True,
        use_temporal=True,
        use_gm_patterns=True,
        use_opening_book=True,
    )
    if use_neural:
        from yami.neural.config import NeuralConfig
        engine_kwargs["use_neural"] = True
        engine_kwargs["neural_config"] = NeuralConfig.from_variant("A")
        engine_kwargs["neural_checkpoint"] = args.checkpoint
    else:
        engine_kwargs["use_neural"] = False

    yami = YamiEngine(**engine_kwargs)
    sf = chess.engine.SimpleEngine.popen_uci("stockfish")

    print("=" * 65)
    print(f"  HIGH-ELO BENCHMARK — {label}")
    print(f"  Neural: {'YES (' + args.checkpoint + ')' if use_neural else 'NO (infrastructure only)'}")
    print(f"  Games per opponent: {args.games}")
    print("=" * 65)

    results = {}
    t_total = time.time()

    try:
        for opp in HIGH_ELO_OPPONENTS:
            wins, draws, losses = 0, 0, 0
            scores = []

            print(f"\n--- {opp['name']} ---")
            t0 = time.time()

            for i in range(args.games):
                color = chess.BLACK if i % 2 == 0 else chess.WHITE
                game = play_game(
                    yami, sf, opp["sf_elo"], opp["sf_skill"], color,
                )

                if game["won"]:
                    wins += 1
                    scores.append(1.0)
                elif game["lost"]:
                    losses += 1
                    scores.append(0.0)
                else:
                    draws += 1
                    scores.append(0.5)

                if (i + 1) % 5 == 0:
                    print(f"  [{i+1}/{args.games}] W={wins} D={draws} L={losses}")

            elapsed = time.time() - t0
            score_pct = sum(scores) / len(scores)
            elo, ci_lo, ci_hi = compute_elo_mle(scores, opp["sf_elo"], len(scores))

            results[opp["name"]] = {
                "opponent_elo": opp["sf_elo"],
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "score": round(score_pct, 3),
                "est_elo": round(elo, 1),
                "ci_low": round(ci_lo, 1),
                "ci_high": round(ci_hi, 1),
                "time_s": round(elapsed, 1),
            }

            print(f"  Result: W={wins} D={draws} L={losses} | "
                  f"Score={score_pct:.1%} | ELO={elo:.0f} [{ci_lo:.0f}, {ci_hi:.0f}] "
                  f"({elapsed:.0f}s)")

    finally:
        sf.quit()

    total_elapsed = time.time() - t_total

    # Summary
    print(f"\n{'=' * 65}")
    print(f"  SUMMARY — {label}")
    print(f"{'=' * 65}")
    print(f"{'Opponent':20s} {'W':>3} {'D':>3} {'L':>3} {'Score':>6} "
          f"{'Est. ELO':>8} {'95% CI':>14}")
    print("-" * 65)

    total_w, total_d, total_l = 0, 0, 0
    for name, r in results.items():
        total_w += r["wins"]
        total_d += r["draws"]
        total_l += r["losses"]
        print(f"{name:20s} {r['wins']:>3} {r['draws']:>3} {r['losses']:>3} "
              f"{r['score']:>5.1%} {r['est_elo']:>8.0f} "
              f"[{r['ci_low']:>5.0f},{r['ci_high']:>5.0f}]")

    print("-" * 65)
    total_n = total_w + total_d + total_l
    total_score = (total_w + 0.5 * total_d) / max(total_n, 1)
    print(f"{'TOTAL':20s} {total_w:>3} {total_d:>3} {total_l:>3} "
          f"{total_score:>5.1%}")
    print(f"\nWin rate at 2000+: {total_w}/{total_n} ({total_w/max(total_n,1):.1%})")
    print(f"Loss rate: {total_l}/{total_n} ({total_l/max(total_n,1):.1%})")
    print(f"Time: {total_elapsed:.0f}s")

    # Save results
    output_path = args.output or f"results/high_elo_{label}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "label": label,
            "neural": use_neural,
            "checkpoint": args.checkpoint,
            "games_per_opponent": args.games,
            "total_games": total_n,
            "total_wins": total_w,
            "total_draws": total_d,
            "total_losses": total_l,
            "per_opponent": results,
        }, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
