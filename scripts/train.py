#!/usr/bin/env python3
"""Train the Balanced Sashimi chess model.

Usage:
    python scripts/train.py
    python scripts/train.py --iterations 2000 --device cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from yami.neural.config import NeuralConfig
from yami.neural.trainer import YamiTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Yami neural Layer 7")
    parser.add_argument("--train-data", default="data/chess_train.jsonl")
    parser.add_argument("--eval-data", default="data/chess_eval.jsonl")
    parser.add_argument("--checkpoint-dir", default="models/run_1")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--ternary", action="store_true", default=True)
    parser.add_argument("--no-ternary", dest="ternary", action="store_false")
    args = parser.parse_args()

    config = NeuralConfig(
        max_iterations=args.iterations,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        ternary_enabled=args.ternary,
        train_path=args.train_data,
        eval_path=args.eval_data,
    )

    print("=== Yami Neural Training ===")
    print(f"Device: {config.device}")
    print(f"Ternary: {config.ternary_enabled}")
    print(f"Iterations: {config.max_iterations}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print()

    trainer = YamiTrainer(config)
    param_count = trainer.param_count()
    print(f"Model parameters: {param_count:,}")
    print(f"  Encoder: {sum(p.numel() for p in trainer.encoder.parameters()):,}")
    print(f"  Bridge:  {sum(p.numel() for p in trainer.bridge.parameters()):,}")
    print(f"  Decoder: {sum(p.numel() for p in trainer.decoder.parameters()):,}")
    print()

    t0 = time.time()
    metrics = trainer.train(
        train_path=args.train_data,
        eval_path=args.eval_data,
        checkpoint_dir=args.checkpoint_dir,
    )
    elapsed = time.time() - t0

    print(f"\n=== Training Complete ===")
    print(f"Time: {elapsed:.1f}s")
    print(f"Steps: {metrics.steps}")

    if metrics.candidate_accuracies:
        final_acc = sum(metrics.candidate_accuracies[-10:]) / min(
            10, len(metrics.candidate_accuracies)
        )
        print(f"Final candidate accuracy (last 10): {final_acc:.3f}")

    if metrics.eval_accuracies:
        print(f"Final eval accuracy: {metrics.eval_accuracies[-1]:.3f}")

    if metrics.train_losses:
        final_loss = sum(metrics.train_losses[-10:]) / min(
            10, len(metrics.train_losses)
        )
        print(f"Final train loss (last 10): {final_loss:.3f}")

    print(f"\nCheckpoint: {args.checkpoint_dir}/final.pt")


if __name__ == "__main__":
    main()
