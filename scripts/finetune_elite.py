#!/usr/bin/env python3
"""Fine-tune the Phase 2 model on elite training data.

Curriculum learning: loads the general-data checkpoint and fine-tunes
on decisive positions (critical eval swings, adversarial near-misses,
endgame conversion). Uses lower LR and optional data mixing to prevent
catastrophic forgetting.

Usage:
    python scripts/finetune_elite.py
    python scripts/finetune_elite.py --elite-data data/elite_train.jsonl \
        --base-checkpoint results/phase2_200k/phase2_A/final.pt \
        --mix-ratio 0.3
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from yami.neural.config import NeuralConfig
from yami.neural.data import ChessDataset
from yami.neural.trainer import YamiTrainer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on elite data")
    parser.add_argument(
        "--elite-data", default="data/elite_train.jsonl",
        help="Elite training data (decisive positions)",
    )
    parser.add_argument(
        "--elite-eval", default="data/elite_eval.jsonl",
        help="Elite eval data",
    )
    parser.add_argument(
        "--general-data", default="data/chess_train.jsonl",
        help="General training data (for mixing)",
    )
    parser.add_argument(
        "--general-eval", default="data/chess_eval.jsonl",
        help="General eval data",
    )
    parser.add_argument(
        "--base-checkpoint",
        default="results/phase2_200k/phase2_A/final.pt",
        help="Phase 2 checkpoint to fine-tune from",
    )
    parser.add_argument(
        "--output-dir", default="results/elite_finetune",
        help="Output directory for fine-tuned checkpoints",
    )
    parser.add_argument(
        "--mix-ratio", type=float, default=0.3,
        help="Fraction of each batch from general data (0=elite only, 0.5=50/50)",
    )
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Lower LR for fine-tuning (default 5e-5 vs 3e-4)")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    # Validate inputs
    if not Path(args.elite_data).exists():
        print(f"ERROR: Elite data not found at {args.elite_data}")
        print("Run scripts/generate_elite_data.py first.")
        sys.exit(1)

    if not Path(args.base_checkpoint).exists():
        print(f"ERROR: Base checkpoint not found at {args.base_checkpoint}")
        sys.exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=== Elite Fine-Tuning ===")
    print(f"Base checkpoint: {args.base_checkpoint}")
    print(f"Elite data: {args.elite_data}")
    print(f"Mix ratio (general): {args.mix_ratio}")
    print(f"Learning rate: {args.lr} (vs 3e-4 base)")
    print(f"Iterations: {args.iterations}")
    print()

    # Build config — same architecture as Phase 2 (variant A), lower LR
    config = NeuralConfig.from_variant(
        "A",
        max_iterations=args.iterations,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_interval=100,
    )

    # Build trainer and load base checkpoint
    trainer = YamiTrainer(config)
    checkpoint = torch.load(args.base_checkpoint, map_location=config.device,
                            weights_only=True)
    trainer.encoder.load_state_dict(checkpoint["encoder"])
    trainer.bridge.load_state_dict(checkpoint["bridge"])
    trainer.decoder.load_state_dict(checkpoint["decoder"])
    if "loss_fn" in checkpoint:
        trainer.loss_fn.load_state_dict(checkpoint["loss_fn"])
    print(f"Loaded base checkpoint ({trainer.param_count():,} params)")

    # Build datasets
    elite_ds = ChessDataset(Path(args.elite_data))
    print(f"Elite dataset: {len(elite_ds)} examples")

    elite_eval_path = args.elite_eval
    general_eval_path = args.general_eval

    if args.mix_ratio > 0 and Path(args.general_data).exists():
        general_ds = ChessDataset(Path(args.general_data))
        # Sample a subset of general data proportional to mix_ratio
        n_general = int(len(elite_ds) * args.mix_ratio / (1 - args.mix_ratio))
        n_general = min(n_general, len(general_ds))
        indices = random.sample(range(len(general_ds)), n_general)
        general_subset = Subset(general_ds, indices)
        train_ds = ConcatDataset([elite_ds, general_subset])
        print(f"General data mixed in: {n_general} examples ({args.mix_ratio:.0%})")
    else:
        train_ds = elite_ds

    print(f"Total training examples: {len(train_ds)}")
    print()

    # Build data loaders
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True,
    )

    eval_loader = None
    if Path(elite_eval_path).exists():
        eval_ds = ChessDataset(Path(elite_eval_path))
        eval_loader = DataLoader(eval_ds, batch_size=config.batch_size)
        print(f"Elite eval: {len(eval_ds)} examples")
    elif Path(general_eval_path).exists():
        eval_ds = ChessDataset(Path(general_eval_path))
        eval_loader = DataLoader(eval_ds, batch_size=config.batch_size)
        print(f"General eval: {len(eval_ds)} examples")

    # Train
    t0 = time.time()
    step = 0
    best_eval_acc = 0.0

    for epoch in range(1000):
        for batch in train_loader:
            if step >= config.max_iterations:
                break

            loss_dict, acc = trainer._train_step(batch)
            trainer.metrics.steps = step
            trainer.metrics.train_losses.append(loss_dict["total"].item())
            trainer.metrics.candidate_accuracies.append(acc)

            if step > 0 and step % config.checkpoint_interval == 0:
                avg_loss = (
                    sum(trainer.metrics.train_losses[-config.checkpoint_interval:])
                    / config.checkpoint_interval
                )
                avg_acc = (
                    sum(trainer.metrics.candidate_accuracies[-config.checkpoint_interval:])
                    / config.checkpoint_interval
                )
                log = f"  step {step:>5} | loss={avg_loss:.3f} | acc={avg_acc:.3f}"

                if eval_loader:
                    eval_loss, eval_acc = trainer._evaluate(eval_loader)
                    trainer.metrics.eval_losses.append(eval_loss)
                    trainer.metrics.eval_accuracies.append(eval_acc)
                    log += f" | eval_acc={eval_acc:.3f}"

                    if eval_acc > best_eval_acc:
                        best_eval_acc = eval_acc
                        trainer._save_checkpoint(
                            Path(args.output_dir) / "best.pt"
                        )
                        log += " *best*"

                print(log)
                trainer._save_checkpoint(
                    Path(args.output_dir) / f"step_{step}.pt"
                )

            step += 1

        if step >= config.max_iterations:
            break

    # Save final
    trainer._save_checkpoint(Path(args.output_dir) / "final.pt")
    elapsed = time.time() - t0

    print(f"\n=== Elite Fine-Tuning Complete ===")
    print(f"Time: {elapsed:.1f}s")
    print(f"Steps: {step}")

    if trainer.metrics.candidate_accuracies:
        final_acc = sum(trainer.metrics.candidate_accuracies[-10:]) / min(
            10, len(trainer.metrics.candidate_accuracies)
        )
        print(f"Final train accuracy (last 10): {final_acc:.3f}")

    if trainer.metrics.eval_accuracies:
        print(f"Best eval accuracy: {best_eval_acc:.3f}")
        print(f"Final eval accuracy: {trainer.metrics.eval_accuracies[-1]:.3f}")

    print(f"\nCheckpoints: {args.output_dir}/")
    print(f"  final.pt — last checkpoint")
    print(f"  best.pt  — best eval accuracy")


if __name__ == "__main__":
    main()
