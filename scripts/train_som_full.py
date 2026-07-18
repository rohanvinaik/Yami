#!/usr/bin/env python3
"""Full SoM training with PAB stability detection and scaled-up agents.

No fixed iteration limits. Each stage trains until its learning
trajectory stabilizes (PAB stability metric < threshold over a window).

PAB Stability: S(t) = |L(t) - L(t-1)| / (L(t-1) + eps)
When mean(S) over a window drops below threshold → training is stable → stop.

Usage:
    python scripts/train_som_full.py --data data/som_merged_train.jsonl
    python scripts/train_som_full.py --data data/som_merged_train.jsonl --hidden 1024
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from yami.neural.som_agents import AGENT_NAMES, SoMAgentEnsemble
from yami.neural.som_orchestrator import SoMOrchestrator, SoMSystem


class SoMDataset(Dataset):
    def __init__(self, path: Path):
        self.data = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.data[idx]
        agent_labels = ex.get("agent_labels", {})
        return {
            "position_signals": torch.tensor(ex["position_signals"], dtype=torch.float32),
            "candidate_signals": torch.tensor(ex["candidate_signals"], dtype=torch.float32),
            "candidate_mask": torch.tensor([i < ex["num_candidates"] for i in range(5)]),
            "best_idx": torch.tensor(ex["best_candidate_idx"], dtype=torch.long),
            "game_outcome": torch.tensor([ex.get("game_outcome", 0.0)], dtype=torch.float32),
            "label_tactical": torch.tensor(agent_labels.get("tactical", 0.0), dtype=torch.float32),
            "label_positional": torch.tensor(agent_labels.get("positional", 0.0), dtype=torch.float32),
            "label_endgame": torch.tensor(agent_labels.get("endgame", 0.0), dtype=torch.float32),
            "label_attack": torch.tensor(agent_labels.get("attack", 0.0), dtype=torch.float32),
            "label_defense": torch.tensor(agent_labels.get("defense", 0.0), dtype=torch.float32),
            "label_initiative": torch.tensor(agent_labels.get("initiative", 0.0), dtype=torch.float32),
        }


class PABTracker:
    """Process-Aware Benchmarking stability tracker.

    Monitors the learning trajectory and signals when training is stable.
    Stability: mean of |L(t) - L(t-1)| / (L(t-1) + eps) over a window.
    """

    def __init__(self, window: int = 20, threshold: float = 0.02, patience: int = 5):
        self.window = window
        self.threshold = threshold
        self.patience = patience
        self.losses = deque(maxlen=window + 1)
        self.eval_accs = deque(maxlen=window)
        self.stability_scores = deque(maxlen=window)
        self.stable_count = 0
        self.best_eval_acc = 0.0
        self.steps_since_best = 0

    def record(self, loss: float, eval_acc: float | None = None) -> None:
        self.losses.append(loss)
        if eval_acc is not None:
            self.eval_accs.append(eval_acc)
            if eval_acc > self.best_eval_acc:
                self.best_eval_acc = eval_acc
                self.steps_since_best = 0
            else:
                self.steps_since_best += 1

        if len(self.losses) >= 2:
            prev = self.losses[-2]
            curr = self.losses[-1]
            s = abs(curr - prev) / (abs(prev) + 1e-8)
            self.stability_scores.append(s)

    def is_stable(self) -> bool:
        """Has training converged?"""
        if len(self.stability_scores) < self.window:
            return False

        mean_stability = sum(self.stability_scores) / len(self.stability_scores)
        if mean_stability < self.threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0

        return self.stable_count >= self.patience

    def should_stop(self, max_no_improve: int = 30) -> bool:
        """Should we stop? Stable OR no improvement for too long."""
        return self.is_stable() or self.steps_since_best >= max_no_improve

    def status(self) -> str:
        if len(self.stability_scores) < 3:
            return "warming up"
        mean_s = sum(self.stability_scores) / len(self.stability_scores)
        return f"stability={mean_s:.4f} (threshold={self.threshold}), best_eval={self.best_eval_acc:.3f}, no_improve={self.steps_since_best}"


@torch.no_grad()
def evaluate(model, loader, device, mode="agents"):
    """Evaluate model accuracy."""
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if mode == "agents":
            scores = model.get_all_scores(
                batch["position_signals"], batch["candidate_signals"], batch["candidate_mask"],
            )
            combined = scores.sum(dim=1)
            combined = combined.masked_fill(~batch["candidate_mask"], float("-inf"))
            preds = combined.argmax(dim=-1)
        elif mode == "system":
            out = model(
                batch["position_signals"], batch["candidate_signals"], batch["candidate_mask"],
            )
            preds = out["candidate_logits"].argmax(dim=-1)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        correct += (preds == batch["best_idx"]).sum().item()
        total += batch["best_idx"].shape[0]
    return correct / max(total, 1)


def train_stage_1(agents, train_loader, eval_loader, device, lr, max_epochs=200):
    """Train agents until PAB stable."""
    print("=== Stage 1: Training Specialist Agents (until PAB stable) ===", flush=True)
    print(f"  Params: {agents.param_count():,}", flush=True)

    optimizer = torch.optim.AdamW(agents.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=max_epochs,
        steps_per_epoch=len(train_loader), pct_start=0.05,
    )
    pab = PABTracker(window=20, threshold=0.015, patience=5)

    step = 0
    for epoch in range(max_epochs):
        for batch in train_loader:
            agents.train()
            batch = {k: v.to(device) for k, v in batch.items()}

            results = agents(
                batch["position_signals"], batch["candidate_signals"], batch["candidate_mask"],
            )

            total_loss = torch.tensor(0.0, device=device)
            for agent_name in AGENT_NAMES:
                agent_out = results[agent_name]
                scores = agent_out["candidate_scores"]
                label = batch[f"label_{agent_name}"]
                best_idx = batch["best_idx"]
                pred_score = scores.gather(1, best_idx.unsqueeze(1)).squeeze(1)
                total_loss = total_loss + nn.functional.mse_loss(pred_score, label)
                total_loss = total_loss + 0.05 * nn.functional.mse_loss(
                    agent_out["position_assessment"], batch["game_outcome"]
                )

            if torch.isnan(total_loss):
                continue

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(agents.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 200 == 0:
                eval_acc = evaluate(agents, eval_loader, device, mode="agents")
                pab.record(total_loss.item(), eval_acc)
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"  step {step:>6} | epoch {epoch+1:>3} | loss={total_loss.item():.4f} | "
                    f"eval_acc={eval_acc:.3f} | lr={current_lr:.2e} | {pab.status()}",
                    flush=True,
                )

                if pab.should_stop(max_no_improve=40):
                    print(f"  PAB STABLE at step {step} (epoch {epoch+1})", flush=True)
                    return step

    print(f"  Max epochs reached at step {step}", flush=True)
    return step


def train_stage_2(agents, orch, train_loader, eval_loader, device, lr, max_epochs=100):
    """Train orchestrator until PAB stable (agents frozen)."""
    print("\n=== Stage 2: Training Orchestrator (until PAB stable) ===", flush=True)
    print(f"  Params: {orch.param_count():,}", flush=True)

    agents.eval()
    for p in agents.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(orch.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=max_epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1,
    )
    loss_fn = nn.CrossEntropyLoss()
    pab = PABTracker(window=20, threshold=0.01, patience=5)

    step = 0
    for epoch in range(max_epochs):
        for batch in train_loader:
            orch.train()
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                agent_scores = agents.get_all_scores(
                    batch["position_signals"], batch["candidate_signals"], batch["candidate_mask"],
                )

            out = orch(batch["position_signals"], agent_scores, batch["candidate_mask"])
            total_loss = loss_fn(out["candidate_logits"], batch["best_idx"])
            total_loss = total_loss + 0.1 * nn.functional.mse_loss(
                out["outcome_pred"], batch["game_outcome"]
            )

            if torch.isnan(total_loss):
                continue

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(orch.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 200 == 0:
                eval_acc = evaluate(
                    SoMSystem(agents, orch), eval_loader, device, mode="system"
                )
                pab.record(total_loss.item(), eval_acc)
                print(
                    f"  step {step:>6} | epoch {epoch+1:>3} | loss={total_loss.item():.4f} | "
                    f"eval_acc={eval_acc:.3f} | {pab.status()}",
                    flush=True,
                )

                if pab.should_stop(max_no_improve=30):
                    print(f"  PAB STABLE at step {step} (epoch {epoch+1})", flush=True)
                    for p in agents.parameters():
                        p.requires_grad = True
                    return step

    for p in agents.parameters():
        p.requires_grad = True
    print(f"  Max epochs reached at step {step}", flush=True)
    return step


def train_stage_3(system, train_loader, eval_loader, device, lr, max_epochs=50):
    """Joint fine-tuning until PAB stable."""
    print("\n=== Stage 3: Joint Fine-Tuning (until PAB stable) ===", flush=True)
    print(f"  Total params: {system.param_count():,}", flush=True)

    optimizer = torch.optim.AdamW(system.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=max_epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1,
    )
    loss_fn = nn.CrossEntropyLoss()
    pab = PABTracker(window=20, threshold=0.008, patience=5)

    step = 0
    for epoch in range(max_epochs):
        for batch in train_loader:
            system.train()
            batch = {k: v.to(device) for k, v in batch.items()}

            out = system(
                batch["position_signals"], batch["candidate_signals"], batch["candidate_mask"],
            )
            total_loss = loss_fn(out["candidate_logits"], batch["best_idx"])

            if torch.isnan(total_loss):
                continue

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(system.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 200 == 0:
                eval_acc = evaluate(system, eval_loader, device, mode="system")
                pab.record(total_loss.item(), eval_acc)
                print(
                    f"  step {step:>6} | epoch {epoch+1:>3} | loss={total_loss.item():.4f} | "
                    f"eval_acc={eval_acc:.3f} | {pab.status()}",
                    flush=True,
                )

                if pab.should_stop(max_no_improve=40):
                    print(f"  PAB STABLE at step {step} (epoch {epoch+1})", flush=True)
                    return step

    print(f"  Max epochs reached at step {step}", flush=True)
    return step


def main():
    parser = argparse.ArgumentParser(description="Full SoM training with PAB stability")
    parser.add_argument("--data", default="data/som_merged_train.jsonl")
    parser.add_argument("--eval-data", default="data/som_merged_eval.jsonl")
    parser.add_argument("--checkpoint-dir", default="models/som_v3")
    parser.add_argument("--hidden", type=int, default=512, help="Agent hidden dim")
    parser.add_argument("--context", type=int, default=256, help="Agent context dim")
    parser.add_argument("--orch-hidden", type=int, default=256, help="Orchestrator hidden dim")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr1", type=float, default=3e-4, help="Stage 1 LR")
    parser.add_argument("--lr2", type=float, default=1e-3, help="Stage 2 LR")
    parser.add_argument("--lr3", type=float, default=3e-5, help="Stage 3 LR")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    device = torch.device(args.device)
    cp_dir = Path(args.checkpoint_dir)
    cp_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SoMDataset(Path(args.data))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    eval_ds = SoMDataset(Path(args.eval_data))
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size)

    print(f"=== SoM Full Training (PAB Stability) ===", flush=True)
    print(f"Train: {len(train_ds):,} examples", flush=True)
    print(f"Eval: {len(eval_ds):,} examples", flush=True)
    print(f"Agent hidden: {args.hidden}, context: {args.context}", flush=True)
    print(f"Device: {device}", flush=True)
    print(flush=True)

    # Build models
    agents = SoMAgentEnsemble(hidden_dim=args.hidden, context_dim=args.context).to(device)
    orch = SoMOrchestrator(hidden_dim=args.orch_hidden).to(device)
    system = SoMSystem(agents, orch)

    print(f"Agent params: {agents.param_count():,}", flush=True)
    print(f"Orchestrator params: {orch.param_count():,}", flush=True)
    print(f"Total: {system.param_count():,}", flush=True)
    print(flush=True)

    t0 = time.time()

    # Stage 1: Agents
    s1_steps = train_stage_1(agents, train_loader, eval_loader, device, lr=args.lr1)
    torch.save({"agents": agents.state_dict()}, cp_dir / "agents_stable.pt")

    # Stage 2: Orchestrator
    s2_steps = train_stage_2(agents, orch, train_loader, eval_loader, device, lr=args.lr2)
    torch.save({"orchestrator": orch.state_dict()}, cp_dir / "orch_stable.pt")

    # Stage 3: Joint
    s3_steps = train_stage_3(system, train_loader, eval_loader, device, lr=args.lr3)
    torch.save({
        "agents": agents.state_dict(),
        "orchestrator": orch.state_dict(),
    }, cp_dir / "som_stable.pt")

    elapsed = time.time() - t0
    print(f"\n=== Training Complete ===", flush=True)
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"Steps: S1={s1_steps}, S2={s2_steps}, S3={s3_steps}", flush=True)
    print(f"Checkpoints: {cp_dir}/", flush=True)


if __name__ == "__main__":
    main()
