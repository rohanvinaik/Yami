#!/usr/bin/env python3
"""Train the Society of Mind: 6 agents + orchestrator.

Three-stage training procedure:
  Stage 1: Train 6 agents independently (parallel on same data, different labels)
  Stage 2: Freeze agents, train orchestrator on agent outputs
  Stage 3: Joint fine-tuning (optional, low LR)

Usage:
    python scripts/train_som.py --data data/som_train.jsonl
    python scripts/train_som.py --data data/som_train.jsonl --stage 2 --agents-checkpoint models/som/agents.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
            # Per-agent labels
            "label_tactical": torch.tensor(agent_labels.get("tactical", 0.0), dtype=torch.float32),
            "label_positional": torch.tensor(agent_labels.get("positional", 0.0), dtype=torch.float32),
            "label_endgame": torch.tensor(agent_labels.get("endgame", 0.0), dtype=torch.float32),
            "label_attack": torch.tensor(agent_labels.get("attack", 0.0), dtype=torch.float32),
            "label_defense": torch.tensor(agent_labels.get("defense", 0.0), dtype=torch.float32),
            "label_initiative": torch.tensor(agent_labels.get("initiative", 0.0), dtype=torch.float32),
        }


def train_agents(
    train_loader: DataLoader,
    eval_loader: DataLoader | None,
    device: torch.device,
    iterations: int = 5000,
    lr: float = 3e-4,
    checkpoint_dir: Path | None = None,
) -> SoMAgentEnsemble:
    """Stage 1: Train all 6 agents independently on domain-specific labels."""
    print("=== Stage 1: Training 6 Specialist Agents ===", flush=True)

    agents = SoMAgentEnsemble().to(device)
    print(f"Total agent params: {agents.param_count():,}", flush=True)

    optimizer = torch.optim.AdamW(agents.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    step = 0
    best_eval_acc = 0.0

    for epoch in range(1000):
        for batch in train_loader:
            if step >= iterations:
                break

            agents.train()
            batch = {k: v.to(device) for k, v in batch.items()}

            # Run all agents
            results = agents(
                batch["position_signals"],
                batch["candidate_signals"],
                batch["candidate_mask"],
            )

            # Per-agent MSE loss on domain-specific labels
            total_loss = torch.tensor(0.0, device=device)
            for agent_name in AGENT_NAMES:
                agent_out = results[agent_name]
                scores = agent_out["candidate_scores"]
                label = batch[f"label_{agent_name}"]

                # Agent loss: the agent's top candidate score should correlate with its label
                # Use the score of the SF-best candidate as the prediction
                best_idx = batch["best_idx"]
                pred_score = scores.gather(1, best_idx.unsqueeze(1)).squeeze(1)

                agent_loss = nn.functional.mse_loss(pred_score, label)
                total_loss = total_loss + agent_loss

                # Auxiliary: position assessment
                pos_pred = agent_out["position_assessment"]
                pos_loss = nn.functional.mse_loss(pos_pred, batch["game_outcome"])
                total_loss = total_loss + 0.05 * pos_loss

            if torch.isnan(total_loss):
                continue

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(agents.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step > 0 and step % 200 == 0:
                log = f"  step {step:>5} | loss={total_loss.item():.4f}"

                if eval_loader:
                    eval_acc = evaluate_agents(agents, eval_loader, device)
                    log += f" | eval_acc={eval_acc:.3f}"
                    if eval_acc > best_eval_acc:
                        best_eval_acc = eval_acc
                        if checkpoint_dir:
                            torch.save({"agents": agents.state_dict()},
                                       checkpoint_dir / "agents_best.pt")
                        log += " *best*"

                print(log, flush=True)

            step += 1

        if step >= iterations:
            break

    if checkpoint_dir:
        torch.save({"agents": agents.state_dict()}, checkpoint_dir / "agents_final.pt")

    print(f"  Best eval accuracy: {best_eval_acc:.3f}", flush=True)
    return agents


def train_orchestrator(
    agents: SoMAgentEnsemble,
    train_loader: DataLoader,
    eval_loader: DataLoader | None,
    device: torch.device,
    iterations: int = 3000,
    lr: float = 1e-3,
    checkpoint_dir: Path | None = None,
) -> SoMOrchestrator:
    """Stage 2: Freeze agents, train orchestrator on their outputs."""
    print("\n=== Stage 2: Training Orchestrator ===", flush=True)

    agents.eval()
    for p in agents.parameters():
        p.requires_grad = False

    orch = SoMOrchestrator().to(device)
    print(f"Orchestrator params: {orch.param_count():,}", flush=True)

    optimizer = torch.optim.AdamW(orch.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    loss_fn = nn.CrossEntropyLoss()

    step = 0
    best_eval_acc = 0.0

    for epoch in range(1000):
        for batch in train_loader:
            if step >= iterations:
                break

            orch.train()
            batch = {k: v.to(device) for k, v in batch.items()}

            # Get frozen agent scores
            with torch.no_grad():
                agent_scores = agents.get_all_scores(
                    batch["position_signals"],
                    batch["candidate_signals"],
                    batch["candidate_mask"],
                )

            # Run orchestrator
            out = orch(batch["position_signals"], agent_scores, batch["candidate_mask"])

            # Loss: cross-entropy on candidate selection
            cand_loss = loss_fn(out["candidate_logits"], batch["best_idx"])
            outcome_loss = nn.functional.mse_loss(out["outcome_pred"], batch["game_outcome"])
            total_loss = cand_loss + 0.1 * outcome_loss

            if torch.isnan(total_loss):
                continue

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(orch.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step > 0 and step % 200 == 0:
                acc = (out["candidate_logits"].argmax(-1) == batch["best_idx"]).float().mean()
                log = f"  step {step:>5} | loss={total_loss.item():.4f} | train_acc={acc:.3f}"

                if eval_loader:
                    eval_acc = evaluate_system(agents, orch, eval_loader, device)
                    log += f" | eval_acc={eval_acc:.3f}"
                    if eval_acc > best_eval_acc:
                        best_eval_acc = eval_acc
                        if checkpoint_dir:
                            torch.save({"orchestrator": orch.state_dict()},
                                       checkpoint_dir / "orch_best.pt")
                        log += " *best*"

                print(log, flush=True)

            step += 1

        if step >= iterations:
            break

    if checkpoint_dir:
        torch.save({"orchestrator": orch.state_dict()}, checkpoint_dir / "orch_final.pt")

    # Unfreeze agents for potential Stage 3
    for p in agents.parameters():
        p.requires_grad = True

    print(f"  Best eval accuracy: {best_eval_acc:.3f}", flush=True)
    return orch


def train_joint(
    system: SoMSystem,
    train_loader: DataLoader,
    eval_loader: DataLoader | None,
    device: torch.device,
    iterations: int = 1000,
    lr: float = 3e-5,
    checkpoint_dir: Path | None = None,
) -> None:
    """Stage 3: Joint fine-tuning at low LR."""
    print("\n=== Stage 3: Joint Fine-Tuning ===", flush=True)

    system.to(device)
    optimizer = torch.optim.AdamW(system.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    step = 0
    for epoch in range(1000):
        for batch in train_loader:
            if step >= iterations:
                break

            system.train()
            batch = {k: v.to(device) for k, v in batch.items()}

            out = system(
                batch["position_signals"],
                batch["candidate_signals"],
                batch["candidate_mask"],
            )

            total_loss = loss_fn(out["candidate_logits"], batch["best_idx"])

            if torch.isnan(total_loss):
                continue

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(system.parameters(), 0.5)
            optimizer.step()

            if step > 0 and step % 200 == 0:
                acc = (out["candidate_logits"].argmax(-1) == batch["best_idx"]).float().mean()
                log = f"  step {step:>5} | loss={total_loss.item():.4f} | acc={acc:.3f}"

                if eval_loader:
                    eval_acc = evaluate_full(system, eval_loader, device)
                    log += f" | eval_acc={eval_acc:.3f}"

                print(log, flush=True)

            step += 1

        if step >= iterations:
            break

    if checkpoint_dir:
        torch.save({
            "agents": system.agents.state_dict(),
            "orchestrator": system.orchestrator.state_dict(),
        }, checkpoint_dir / "som_final.pt")


@torch.no_grad()
def evaluate_agents(agents: SoMAgentEnsemble, loader: DataLoader, device: torch.device) -> float:
    """Evaluate: do agents collectively pick the SF-best candidate?"""
    agents.eval()
    correct = total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        scores = agents.get_all_scores(
            batch["position_signals"], batch["candidate_signals"], batch["candidate_mask"],
        )
        # Simple sum across agents as baseline
        combined = scores.sum(dim=1)
        combined = combined.masked_fill(~batch["candidate_mask"], float("-inf"))
        preds = combined.argmax(dim=-1)
        correct += (preds == batch["best_idx"]).sum().item()
        total += batch["best_idx"].shape[0]
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_system(
    agents: SoMAgentEnsemble, orch: SoMOrchestrator,
    loader: DataLoader, device: torch.device,
) -> float:
    agents.eval()
    orch.eval()
    correct = total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        scores = agents.get_all_scores(
            batch["position_signals"], batch["candidate_signals"], batch["candidate_mask"],
        )
        out = orch(batch["position_signals"], scores, batch["candidate_mask"])
        preds = out["candidate_logits"].argmax(dim=-1)
        correct += (preds == batch["best_idx"]).sum().item()
        total += batch["best_idx"].shape[0]
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_full(system: SoMSystem, loader: DataLoader, device: torch.device) -> float:
    system.eval()
    correct = total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = system(
            batch["position_signals"], batch["candidate_signals"], batch["candidate_mask"],
        )
        preds = out["candidate_logits"].argmax(dim=-1)
        correct += (preds == batch["best_idx"]).sum().item()
        total += batch["best_idx"].shape[0]
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Train SoM system")
    parser.add_argument("--data", default="data/som_train.jsonl")
    parser.add_argument("--eval-data", default="data/som_eval.jsonl")
    parser.add_argument("--checkpoint-dir", default="models/som")
    parser.add_argument("--stage", type=int, default=0, help="0=all, 1=agents, 2=orch, 3=joint")
    parser.add_argument("--agents-checkpoint", default=None)
    parser.add_argument("--orch-checkpoint", default=None)
    parser.add_argument("--iterations-1", type=int, default=5000)
    parser.add_argument("--iterations-2", type=int, default=3000)
    parser.add_argument("--iterations-3", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    device = torch.device(args.device)
    cp_dir = Path(args.checkpoint_dir)
    cp_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SoMDataset(Path(args.data))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    eval_loader = None
    if Path(args.eval_data).exists():
        eval_ds = SoMDataset(Path(args.eval_data))
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size)

    print(f"Train: {len(train_ds)} examples, Device: {device}", flush=True)
    t0 = time.time()

    # Stage 1: Train agents
    if args.stage in (0, 1):
        agents = train_agents(train_loader, eval_loader, device,
                              iterations=args.iterations_1, checkpoint_dir=cp_dir)
    else:
        agents = SoMAgentEnsemble().to(device)
        if args.agents_checkpoint:
            cp = torch.load(args.agents_checkpoint, map_location=device, weights_only=True)
            agents.load_state_dict(cp["agents"])
            print(f"Loaded agents from {args.agents_checkpoint}", flush=True)

    # Stage 2: Train orchestrator
    if args.stage in (0, 2):
        orch = train_orchestrator(agents, train_loader, eval_loader, device,
                                  iterations=args.iterations_2, checkpoint_dir=cp_dir)
    else:
        orch = SoMOrchestrator().to(device)
        if args.orch_checkpoint:
            cp = torch.load(args.orch_checkpoint, map_location=device, weights_only=True)
            orch.load_state_dict(cp["orchestrator"])
            print(f"Loaded orchestrator from {args.orch_checkpoint}", flush=True)

    # Stage 3: Joint fine-tuning
    if args.stage in (0, 3):
        system = SoMSystem(agents, orch)
        train_joint(system, train_loader, eval_loader, device,
                    iterations=args.iterations_3, checkpoint_dir=cp_dir)

    elapsed = time.time() - t0
    print(f"\n=== Done in {elapsed:.0f}s ===", flush=True)
    print(f"Checkpoints: {cp_dir}/", flush=True)


if __name__ == "__main__":
    main()
