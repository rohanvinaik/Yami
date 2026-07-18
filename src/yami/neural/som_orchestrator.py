"""Society of Mind Orchestrator — the trust-weight arbitrator.

Takes position context + all 6 agent scores and learns WHICH agent
to trust in each position type. Does not re-evaluate moves — it
arbitrates the agent debate.

Architecture:
    position_signals(58) + agent_scores(6×5=30) + convergence(6)
    → context_encoder → trust_head → 6 weights (softmax)
    → final_score[cand_i] = sum(trust[agent_j] * agent_j.score[cand_i])
    → select argmax

~200K parameters. The agents do the evaluation. The orchestrator picks
which evaluation to believe.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from yami.neural.som_agents import AGENT_NAMES, MAX_CANDIDATES
from yami.position_signals import PositionSignals

N_AGENTS = len(AGENT_NAMES)
POS_DIM = PositionSignals.vector_dim()


class SoMOrchestrator(nn.Module):
    """Trust-weight arbitrator for the Society of Mind.

    Learns: "given this position type and these agent opinions,
    which agent should I listen to?"
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        # Position context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(POS_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        # Agent convergence features: variance, agreement patterns
        convergence_dim = N_AGENTS  # one convergence feature per agent

        # Trust head: position context + convergence → agent weights
        self.trust_head = nn.Sequential(
            nn.Linear(hidden_dim // 2 + convergence_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, N_AGENTS),
            # No softmax here — applied in forward() for numerical stability
        )

        # Override head: detect positions where one agent should dominate
        self.override_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, N_AGENTS),
            nn.Sigmoid(),  # per-agent override probability
        )

        # Auxiliary: game outcome prediction
        self.outcome_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        position_signals: torch.Tensor,
        agent_scores: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        position_signals: (batch, 58)
        agent_scores: (batch, n_agents, max_candidates)
        candidate_mask: (batch, max_candidates)

        Returns:
            candidate_logits: (batch, max_candidates) — final fused scores
            trust_weights: (batch, n_agents) — how much to trust each agent
            override_probs: (batch, n_agents) — override probability per agent
            outcome_pred: (batch, 1) — predicted game outcome
        """
        batch = position_signals.shape[0]

        # Encode position context
        context = self.context_encoder(position_signals)  # (batch, hidden//2)

        # Compute convergence features from agent scores
        # Replace -inf with 0 for computation, then measure agreement
        safe_scores = agent_scores.masked_fill(
            ~candidate_mask.unsqueeze(1).expand_as(agent_scores), 0.0
        )
        agent_means = safe_scores.mean(dim=1, keepdim=True)  # (batch, 1, n_cand)
        agent_agreement = 1.0 - (safe_scores - agent_means).abs().mean(dim=2)  # (batch, n_agents)
        convergence_features = agent_agreement

        # Trust weights
        trust_input = torch.cat([context, convergence_features], dim=-1)
        trust_logits = self.trust_head(trust_input)  # (batch, n_agents)
        trust_weights = F.softmax(trust_logits, dim=-1)  # (batch, n_agents)

        # Override detection
        override_probs = self.override_head(context)  # (batch, n_agents)

        # Apply override: if any agent has override > 0.8, boost its trust
        override_mask = (override_probs > 0.8).float()
        if override_mask.sum() > 0:
            boosted_logits = trust_logits + override_mask * 5.0
            trust_weights = F.softmax(boosted_logits, dim=-1)

        # Fuse agent scores: weighted sum per candidate
        # Use safe_scores (no -inf) for the weighted sum
        weighted = safe_scores * trust_weights.unsqueeze(-1)  # (batch, n_agents, n_cand)
        candidate_logits = weighted.sum(dim=1)  # (batch, n_cand)

        # Mask invalid candidates
        candidate_logits = candidate_logits.masked_fill(~candidate_mask, float("-inf"))

        # Auxiliary outcome prediction
        outcome_pred = self.outcome_head(context)

        return {
            "candidate_logits": candidate_logits,
            "trust_weights": trust_weights,
            "override_probs": override_probs,
            "outcome_pred": outcome_pred,
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SoMSystem(nn.Module):
    """Complete Society of Mind: agents + orchestrator.

    This is the full inference pipeline — takes raw signals, runs
    all agents, and the orchestrator picks the move.
    """

    def __init__(
        self,
        agents: nn.Module,  # SoMAgentEnsemble
        orchestrator: SoMOrchestrator,
    ):
        super().__init__()
        self.agents = agents
        self.orchestrator = orchestrator

    def forward(
        self,
        position_signals: torch.Tensor,
        candidate_signals: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full SoM inference: agents score → orchestrator arbitrates."""
        # Run all agents
        agent_scores = self.agents.get_all_scores(
            position_signals, candidate_signals, candidate_mask,
        )  # (batch, 6, n_cand)

        # Orchestrator arbitrates
        orch_out = self.orchestrator(
            position_signals, agent_scores, candidate_mask,
        )

        return {
            "candidate_logits": orch_out["candidate_logits"],
            "trust_weights": orch_out["trust_weights"],
            "agent_scores": agent_scores,
            "outcome_pred": orch_out["outcome_pred"],
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
