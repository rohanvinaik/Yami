#!/usr/bin/env python3
"""Train the learned signal fusion model.

Takes signal-profile training data (from generate_signal_data.py) and
trains the 12K-parameter fusion model to select the best candidate
from signal profiles.

Usage:
    python scripts/train_fusion.py --data data/signal_train.jsonl
    python scripts/train_fusion.py --data data/signal_train.jsonl --iterations 5000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader, Dataset

from yami.neural.fusion_model import FusionLoss, SignalFusionModel


class SignalDataset(Dataset):
    """Load signal-profile examples for the fusion model."""

    def __init__(self, path: Path):
        self.examples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return {
            "position_signals": torch.tensor(
                ex["position_signals"], dtype=torch.float32
            ),
            "candidate_signals": torch.tensor(
                ex["candidate_signals"], dtype=torch.float32
            ),
            "candidate_mask": torch.tensor(
                [i < ex["num_candidates"] for i in range(5)],
            ),
            "best_idx": torch.tensor(ex["best_candidate_idx"], dtype=torch.long),
            "game_outcome": torch.tensor(
                [ex.get("game_outcome", 0.0)], dtype=torch.float32
            ),
        }


def main():
    parser = argparse.ArgumentParser(description="Train fusion model")
    parser.add_argument("--data", default="data/signal_train.jsonl")
    parser.add_argument("--eval-data", default="data/signal_eval.jsonl")
    parser.add_argument("--checkpoint-dir", default="models/fusion_v1")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"ERROR: Training data not found: {args.data}")
        sys.exit(1)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Data
    train_ds = SignalDataset(Path(args.data))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    eval_loader = None
    if Path(args.eval_data).exists():
        eval_ds = SignalDataset(Path(args.eval_data))
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size)

    # Model
    model = SignalFusionModel().to(device)
    loss_fn = FusionLoss(bank_weight=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    print("=== Fusion Model Training ===", flush=True)
    print(f"Parameters: {model.param_count():,}", flush=True)
    print(f"Train examples: {len(train_ds)}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"LR: {args.lr}, Batch: {args.batch_size}", flush=True)
    print(flush=True)

    # Training loop
    step = 0
    best_eval_acc = 0.0
    train_losses = []
    train_accs = []

    t0 = time.time()
    for epoch in range(1000):
        for batch in train_loader:
            if step >= args.iterations:
                break

            model.train()
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(
                batch["position_signals"],
                batch["candidate_signals"],
                batch["candidate_mask"],
            )

            losses = loss_fn(
                out["candidate_logits"],
                out["bank_profile_pred"],
                batch["best_idx"],
                batch["game_outcome"],
            )

            total_loss = losses["total"]
            if torch.isnan(total_loss):
                continue

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Accuracy
            preds = out["candidate_logits"].argmax(dim=-1)
            acc = (preds == batch["best_idx"]).float().mean().item()
            train_losses.append(total_loss.item())
            train_accs.append(acc)

            # Log every 100 steps
            if step > 0 and step % 100 == 0:
                avg_loss = sum(train_losses[-100:]) / 100
                avg_acc = sum(train_accs[-100:]) / 100
                log = f"  step {step:>5} | loss={avg_loss:.3f} | acc={avg_acc:.3f}"

                if eval_loader:
                    eval_acc = evaluate(model, eval_loader, device)
                    log += f" | eval_acc={eval_acc:.3f}"
                    if eval_acc > best_eval_acc:
                        best_eval_acc = eval_acc
                        save_checkpoint(
                            model, Path(args.checkpoint_dir) / "best.pt"
                        )
                        log += " *best*"

                print(log, flush=True)

            step += 1

        if step >= args.iterations:
            break

    # Final save
    save_checkpoint(model, Path(args.checkpoint_dir) / "final.pt")
    elapsed = time.time() - t0

    print(f"\n=== Training Complete ===", flush=True)
    print(f"Time: {elapsed:.1f}s", flush=True)
    print(f"Steps: {step}", flush=True)
    if train_accs:
        print(f"Final train acc: {sum(train_accs[-10:])/min(10,len(train_accs)):.3f}",
              flush=True)
    if best_eval_acc > 0:
        print(f"Best eval acc: {best_eval_acc:.3f}", flush=True)
    print(f"Checkpoints: {args.checkpoint_dir}/", flush=True)


@torch.no_grad()
def evaluate(
    model: SignalFusionModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            batch["position_signals"],
            batch["candidate_signals"],
            batch["candidate_mask"],
        )
        preds = out["candidate_logits"].argmax(dim=-1)
        correct += (preds == batch["best_idx"]).sum().item()
        total += batch["best_idx"].shape[0]
    return correct / max(total, 1)


def save_checkpoint(model: SignalFusionModel, path: Path) -> None:
    torch.save({"model": model.state_dict()}, path)


if __name__ == "__main__":
    main()
