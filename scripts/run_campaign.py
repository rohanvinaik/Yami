#!/usr/bin/env python3
"""Orchestrate the Yami neural training campaign.

Runs architecture variants, tracks results, selects winners.

Usage:
    python scripts/run_campaign.py --phase 1 --data data/chess_train.jsonl
    python scripts/run_campaign.py --phase 2 --data data/train_200k.jsonl --variants A,B
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess

from yami.engine import YamiEngine
from yami.neural.config import NeuralConfig
from yami.neural.trainer import YamiTrainer
from yami.oracle import StockfishOracle


def train_variant(
    variant: str,
    train_path: str,
    eval_path: str,
    max_iterations: int,
    device: str,
    checkpoint_dir: str,
) -> dict:
    """Train one architecture variant and return metrics."""
    config = NeuralConfig.from_variant(variant)
    config.max_iterations = max_iterations
    config.device = device
    config.train_path = train_path
    config.eval_path = eval_path
    config.campaign_name = f"campaign_{variant}"

    print(f"\n{'='*60}")
    print(f"  VARIANT {variant}: {_variant_desc(variant)}")
    print(f"{'='*60}")

    trainer = YamiTrainer(config)
    params = trainer.param_count()
    print(f"  Parameters: {params:,}")

    t0 = time.time()
    metrics = trainer.train(
        train_path=train_path,
        eval_path=eval_path,
        checkpoint_dir=checkpoint_dir,
    )
    elapsed = time.time() - t0

    final_train_acc = (
        sum(metrics.candidate_accuracies[-20:])
        / max(len(metrics.candidate_accuracies[-20:]), 1)
    )
    final_eval_acc = metrics.eval_accuracies[-1] if metrics.eval_accuracies else 0.0
    best_eval_acc = max(metrics.eval_accuracies) if metrics.eval_accuracies else 0.0

    result = {
        "variant": variant,
        "params": params,
        "steps": metrics.steps,
        "time_s": round(elapsed, 1),
        "final_train_acc": round(final_train_acc, 4),
        "final_eval_acc": round(final_eval_acc, 4),
        "best_eval_acc": round(best_eval_acc, 4),
        "final_loss": round(metrics.train_losses[-1], 4) if metrics.train_losses else 0,
        "checkpoint": f"{checkpoint_dir}/final.pt",
    }

    print(f"\n  Result: train_acc={final_train_acc:.3f} eval_acc={final_eval_acc:.3f} "
          f"best_eval={best_eval_acc:.3f} time={elapsed:.0f}s")

    return result


def evaluate_model(
    checkpoint_path: str, config: NeuralConfig,
    sf_depth: int = 3, num_games: int = 20,
) -> dict:
    """Evaluate a trained model by playing games against Stockfish."""
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        return {"error": "checkpoint not found"}

    engine = YamiEngine(
        use_llm=False, use_neural=True,
        neural_checkpoint=str(ckpt), neural_config=config,
        use_opening_book=True,
    )

    oracle = StockfishOracle(depth=sf_depth, time_limit=0.05)
    wins = 0
    losses = 0
    draws = 0
    total_moves = 0

    try:
        for i in range(num_games):
            board = chess.Board()
            engine.reset()
            engine.state.board = board
            color = chess.WHITE if i % 2 == 0 else chess.BLACK

            for _ in range(200):
                if board.is_game_over():
                    break
                if board.turn == color:
                    d = engine.decide(board)
                    move = d.move
                else:
                    move = oracle.best_move(board)
                if move is None:
                    break
                board.push(move)

            total_moves += len(board.move_stack)
            result = board.result()
            engine_won = (result == "1-0" and color == chess.WHITE) or \
                         (result == "0-1" and color == chess.BLACK)
            engine_lost = (result == "0-1" and color == chess.WHITE) or \
                          (result == "1-0" and color == chess.BLACK)
            if engine_won:
                wins += 1
            elif engine_lost:
                losses += 1
            else:
                draws += 1
    finally:
        oracle.close()

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "avg_moves": round(total_moves / max(num_games, 1), 1),
        "win_rate": round(wins / max(num_games, 1), 3),
        "sf_depth": sf_depth,
    }


def run_campaign(
    phase: int,
    variants: list[str],
    train_path: str,
    eval_path: str,
    max_iterations: int,
    device: str,
    results_dir: str,
    sf_depth: int = 3,
    eval_games: int = 20,
):
    """Run a full campaign phase."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  YAMI TRAINING CAMPAIGN — PHASE {phase}")
    print(f"  Variants: {', '.join(variants)}")
    print(f"  Data: {train_path}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Device: {device}")
    print(f"{'#'*60}")

    all_results = {}
    t_campaign = time.time()

    for variant in variants:
        ckpt_dir = str(results_path / f"phase{phase}_{variant}")

        result = train_variant(
            variant=variant,
            train_path=train_path,
            eval_path=eval_path,
            max_iterations=max_iterations,
            device=device,
            checkpoint_dir=ckpt_dir,
        )

        # Evaluate against Stockfish
        print(f"\n  Evaluating variant {variant} vs Stockfish depth {sf_depth}...")
        eval_result = evaluate_model(
            result["checkpoint"],
            NeuralConfig.from_variant(variant),
            sf_depth=sf_depth,
            num_games=eval_games,
        )
        result["eval_vs_sf"] = eval_result
        all_results[variant] = result

        # Save incrementally
        manifest = results_path / f"phase{phase}_results.json"
        with open(manifest, "w") as f:
            json.dump(all_results, f, indent=2)

    elapsed = time.time() - t_campaign

    # Also evaluate infrastructure-only baseline
    print("\n  Evaluating infrastructure-only baseline...")
    infra_engine = YamiEngine(use_llm=False, use_neural=False, use_opening_book=True)
    oracle = StockfishOracle(depth=sf_depth, time_limit=0.05)
    infra_wins, infra_losses, infra_draws, infra_moves = 0, 0, 0, 0
    try:
        for i in range(eval_games):
            board = chess.Board()
            infra_engine.reset()
            infra_engine.state.board = board
            color = chess.WHITE if i % 2 == 0 else chess.BLACK
            for _ in range(200):
                if board.is_game_over():
                    break
                if board.turn == color:
                    d = infra_engine.decide(board)
                    move = d.move
                else:
                    move = oracle.best_move(board)
                if move is None:
                    break
                board.push(move)
            infra_moves += len(board.move_stack)
            result = board.result()
            ew = (result == "1-0" and color == chess.WHITE) or \
                 (result == "0-1" and color == chess.BLACK)
            el = (result == "0-1" and color == chess.WHITE) or \
                 (result == "1-0" and color == chess.BLACK)
            if ew:
                infra_wins += 1
            elif el:
                infra_losses += 1
            else:
                infra_draws += 1
    finally:
        oracle.close()

    all_results["infrastructure"] = {
        "wins": infra_wins, "losses": infra_losses, "draws": infra_draws,
        "avg_moves": round(infra_moves / max(eval_games, 1), 1),
    }

    # Save final results
    manifest = results_path / f"phase{phase}_results.json"
    with open(manifest, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"  PHASE {phase} RESULTS (total time: {elapsed:.0f}s)")
    print(f"{'='*80}")
    print(f"{'Variant':<12} {'Params':>8} {'Train':>7} {'Eval':>7} {'Best':>7} "
          f"{'W':>3} {'D':>3} {'L':>3} {'AvgMov':>7} {'Time':>6}")
    print("-" * 80)

    for v, r in all_results.items():
        if v == "infrastructure":
            print(f"{'INFRA':<12} {'—':>8} {'—':>7} {'—':>7} {'—':>7} "
                  f"{r['wins']:>3} {r['draws']:>3} {r['losses']:>3} "
                  f"{r['avg_moves']:>7} {'—':>6}")
        else:
            sf = r.get("eval_vs_sf", {})
            print(f"{v:<12} {r['params']:>8,} {r['final_train_acc']:>7.3f} "
                  f"{r['final_eval_acc']:>7.3f} {r['best_eval_acc']:>7.3f} "
                  f"{sf.get('wins', 0):>3} {sf.get('draws', 0):>3} "
                  f"{sf.get('losses', 0):>3} {sf.get('avg_moves', 0):>7} "
                  f"{r['time_s']:>5.0f}s")

    # Select winners
    ranked = sorted(
        [(v, r) for v, r in all_results.items() if v != "infrastructure"],
        key=lambda x: x[1]["best_eval_acc"],
        reverse=True,
    )
    winners = [v for v, _ in ranked[:2]]
    print(f"\n  Top 2 variants for Phase {phase + 1}: {', '.join(winners)}")

    return all_results, winners


def _variant_desc(v: str) -> str:
    descs = {
        "A": "Continuous baseline (256h, 2L)",
        "B": "Bigger continuous (512h, 3L)",
        "C": "Partial ternary (ternary hidden, continuous heads)",
        "D": "Staged (continuous warmup → ternarize)",
        "E": "Attention decoder (self-attn over candidates)",
    }
    return descs.get(v, v)


def main():
    parser = argparse.ArgumentParser(description="Yami training campaign")
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--variants", type=str, default="A,B,C,D,E")
    parser.add_argument("--data", type=str, default="data/chess_train.jsonl")
    parser.add_argument("--eval-data", type=str, default="data/chess_eval.jsonl")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--sf-depth", type=int, default=3)
    parser.add_argument("--eval-games", type=int, default=20)
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",")]

    run_campaign(
        phase=args.phase,
        variants=variants,
        train_path=args.data,
        eval_path=args.eval_data,
        max_iterations=args.iterations,
        device=args.device,
        results_dir=args.results_dir,
        sf_depth=args.sf_depth,
        eval_games=args.eval_games,
    )


if __name__ == "__main__":
    main()
