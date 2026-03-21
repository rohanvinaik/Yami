"""Training loop for the chess neural Layer 7.

Adapted from ShortcutForge's BalancedSashimiTrainer. Pipeline:
ChessPositionEncoder → InformationBridge → ChessTernaryDecoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from yami.neural.bridge import InformationBridge
from yami.neural.config import NeuralConfig
from yami.neural.data import ChessDataset
from yami.neural.decoder import ChessTernaryDecoder
from yami.neural.encoder import ChessPositionEncoder
from yami.neural.losses import ChessCompositeLoss


@dataclass
class TrainMetrics:
    """Metrics from a training run."""

    steps: int = 0
    train_losses: list[float] = field(default_factory=list)
    candidate_accuracies: list[float] = field(default_factory=list)
    plan_accuracies: list[float] = field(default_factory=list)
    eval_losses: list[float] = field(default_factory=list)
    eval_accuracies: list[float] = field(default_factory=list)


class YamiTrainer:
    """Training loop for the Balanced Sashimi chess model."""

    def __init__(self, config: NeuralConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

        # Build model
        self.encoder = ChessPositionEncoder(
            output_dim=config.encoder_output_dim,
            candidate_dim=config.candidate_dim,
            embed_dim=config.embed_dim,
        ).to(self.device)

        self.bridge = InformationBridge(
            input_dim=config.encoder_output_dim,
            bridge_dim=config.bridge_dim,
        ).to(self.device)

        self.decoder = ChessTernaryDecoder(
            input_dim=config.bridge_dim,
            hidden_dim=config.decoder_hidden_dim,
            num_layers=config.decoder_num_layers,
            ternary_enabled=config.ternary_enabled,
            partial_ternary=config.partial_ternary,
        ).to(self.device)

        self.loss_fn = ChessCompositeLoss(
            margin=config.margin,
            plan_weight=config.plan_weight,
        ).to(self.device)

        # Optimizer
        all_params = (
            list(self.encoder.parameters())
            + list(self.bridge.parameters())
            + list(self.decoder.parameters())
            + list(self.loss_fn.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.metrics = TrainMetrics()

    def param_count(self) -> int:
        """Total trainable parameters."""
        return sum(
            p.numel() for p in
            list(self.encoder.parameters())
            + list(self.bridge.parameters())
            + list(self.decoder.parameters())
            if p.requires_grad
        )

    def train(
        self,
        train_path: Path | str,
        eval_path: Path | str | None = None,
        checkpoint_dir: Path | str | None = None,
    ) -> TrainMetrics:
        """Run the full training loop."""
        train_ds = ChessDataset(Path(train_path))
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        eval_loader = None
        if eval_path and Path(eval_path).exists():
            eval_ds = ChessDataset(Path(eval_path))
            eval_loader = DataLoader(eval_ds, batch_size=self.config.batch_size)

        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        step = 0
        for _epoch in range(1000):  # iterate epochs until max_iterations
            for batch in train_loader:
                if step >= self.config.max_iterations:
                    break

                loss_dict, acc = self._train_step(batch)
                self.metrics.steps = step
                self.metrics.train_losses.append(loss_dict["total"].item())
                self.metrics.candidate_accuracies.append(acc)

                # Checkpoint + logging
                if step > 0 and step % self.config.checkpoint_interval == 0:
                    avg_loss = (
                        sum(self.metrics.train_losses[-self.config.checkpoint_interval:])
                        / self.config.checkpoint_interval
                    )
                    avg_acc = (
                        sum(self.metrics.candidate_accuracies[-self.config.checkpoint_interval:])
                        / self.config.checkpoint_interval
                    )
                    log = f"  step {step:>5} | loss={avg_loss:.3f} | acc={avg_acc:.3f}"

                    if eval_loader:
                        eval_loss, eval_acc = self._evaluate(eval_loader)
                        self.metrics.eval_losses.append(eval_loss)
                        self.metrics.eval_accuracies.append(eval_acc)
                        log += f" | eval_acc={eval_acc:.3f}"

                    print(log)

                    if checkpoint_dir:
                        self._save_checkpoint(
                            Path(checkpoint_dir) / f"step_{step}.pt"
                        )

                step += 1

            if step >= self.config.max_iterations:
                break

        # Final checkpoint
        if checkpoint_dir:
            self._save_checkpoint(Path(checkpoint_dir) / "final.pt")

        return self.metrics

    def _train_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], float]:
        """Single training step."""
        self.encoder.train()
        self.bridge.train()
        self.decoder.train()

        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward
        enc = self.encoder(
            batch["profile"],
            batch["plan_type"],
            batch["plan_activation"],
            batch["candidate_features"],
            batch["candidate_mask"],
        )
        bridge_out = self.bridge(enc)
        outputs = self.decoder(bridge_out, batch["candidate_mask"])

        # Loss
        second_targets = None
        if batch["has_second"].any():
            second_targets = batch["second_idx"]

        loss_dict = self.loss_fn(
            outputs["candidate_logits"],
            outputs["plan_logits"],
            batch["best_idx"],
            batch["plan_type"],
            second_targets,
        )

        # Gradient safety
        total_loss = loss_dict["total"]
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return loss_dict, 0.0

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.bridge.parameters())
            + list(self.decoder.parameters()),
            self.config.max_grad_norm,
        )
        self.optimizer.step()

        # Accuracy
        preds = outputs["candidate_logits"].argmax(dim=-1)
        acc = (preds == batch["best_idx"]).float().mean().item()

        return loss_dict, acc

    @torch.no_grad()
    def _evaluate(
        self, loader: DataLoader  # type: ignore[type-arg]
    ) -> tuple[float, float]:
        """Evaluate on a dataset."""
        self.encoder.eval()
        self.bridge.eval()
        self.decoder.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            enc = self.encoder(
                batch["profile"],
                batch["plan_type"],
                batch["plan_activation"],
                batch["candidate_features"],
                batch["candidate_mask"],
            )
            bridge_out = self.bridge(enc)
            outputs = self.decoder(bridge_out, batch["candidate_mask"])

            loss_dict = self.loss_fn(
                outputs["candidate_logits"],
                outputs["plan_logits"],
                batch["best_idx"],
                batch["plan_type"],
            )
            total_loss += loss_dict["total"].item() * batch["best_idx"].shape[0]

            preds = outputs["candidate_logits"].argmax(dim=-1)
            total_correct += (preds == batch["best_idx"]).sum().item()
            total_samples += batch["best_idx"].shape[0]

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_correct / max(total_samples, 1)
        return avg_loss, avg_acc

    def _save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint."""
        torch.save({
            "encoder": self.encoder.state_dict(),
            "bridge": self.bridge.state_dict(),
            "decoder": self.decoder.state_dict(),
            "loss_fn": self.loss_fn.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metrics": {
                "steps": self.metrics.steps,
                "train_losses": self.metrics.train_losses[-10:],
                "candidate_accuracies": self.metrics.candidate_accuracies[-10:],
            },
        }, path)
