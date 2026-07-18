"""Society of Mind Specialist Agents — learned domain experts.

Six specialist models, each ~500K-1M parameters, trained on domain-specific
labels derived from Stockfish's 15-ply principal variation analysis.

Each agent sees:
  1. Its domain-relevant signal subset (high capacity)
  2. The full position context (lower capacity, prevents blindness)
  3. Per-candidate signals

Each agent outputs a scalar score per candidate: "how good is this move
from my perspective?"

The orchestrator (som_orchestrator.py) learns which agent to trust.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from yami.position_signals import PositionSignals
from yami.candidate_signals import CandidateSignalVector


# Agent domain → position signal dimension indices
# Must match AGENT_POSITION_DIMS in generate_som_data.py
AGENT_DOMAINS = {
    "tactical": {
        "pos_dims": list(range(0, 5)),  # material signals
        "description": "Material, captures, threats",
    },
    "positional": {
        "pos_dims": list(range(5, 17)) + list(range(25, 33)),  # pawn + piece activity
        "description": "Pawn structure, piece placement",
    },
    "endgame": {
        "pos_dims": [13, 14, 15, 16, 33],  # passed pawns + game phase
        "description": "Conversion, king activity, passed pawns",
    },
    "attack": {
        "pos_dims": [19, 21, 23],  # king safety theirs
        "description": "King pressure, attacking potential",
    },
    "defense": {
        "pos_dims": [17, 20, 22, 24],  # king safety ours + castled
        "description": "King safety, vulnerability reduction",
    },
    "initiative": {
        "pos_dims": [25, 26, 27, 28, 34, 35, 36, 37],  # mobility + center + space
        "description": "Tempo, space, development",
    },
}

AGENT_NAMES = ["tactical", "positional", "endgame", "attack", "defense", "initiative"]
FULL_POS_DIM = PositionSignals.vector_dim()  # 58
CAND_DIM = CandidateSignalVector.vector_dim()  # 50
MAX_CANDIDATES = 5


class SpecialistAgent(nn.Module):
    """One SoM specialist agent.

    Architecture:
        domain_encoder(subset) → 64-dim domain representation
        context_encoder(all_58) → 32-dim full context (lower capacity)
        candidate_scorer(cand_50 + domain_64 + context_32) → scalar per candidate
        position_head(domain_64) → game outcome prediction (auxiliary)
    """

    def __init__(
        self,
        agent_name: str,
        domain_dims: list[int],
        hidden_dim: int = 128,
        context_dim: int = 64,
    ):
        super().__init__()
        self.agent_name = agent_name
        self.domain_dims = domain_dims
        domain_input_dim = len(domain_dims)

        # Domain-specific encoder (high capacity)
        self.domain_encoder = nn.Sequential(
            nn.Linear(domain_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, context_dim),
            nn.LayerNorm(context_dim),
        )

        # Full context encoder (lower capacity — prevents domain blindness)
        self.context_encoder = nn.Sequential(
            nn.Linear(FULL_POS_DIM, context_dim),
            nn.LayerNorm(context_dim),
            nn.GELU(),
        )

        # Per-candidate scorer (shared across candidates)
        scorer_input = CAND_DIM + context_dim + context_dim  # cand + domain + full_context
        self.candidate_scorer = nn.Sequential(
            nn.Linear(scorer_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Auxiliary: position assessment
        self.position_head = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Tanh(),  # -1 to +1: losing to winning
        )

    def forward(
        self,
        position_signals: torch.Tensor,
        candidate_signals: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        position_signals: (batch, 58)
        candidate_signals: (batch, max_cands, 50)
        candidate_mask: (batch, max_cands)

        Returns:
            candidate_scores: (batch, max_cands)
            position_assessment: (batch, 1)
        """
        batch, n_cand, cand_feat = candidate_signals.shape

        # Extract domain-relevant signals
        domain_input = position_signals[:, self.domain_dims]  # (batch, domain_dims)

        # Encode
        domain_enc = self.domain_encoder(domain_input)  # (batch, context_dim)
        context_enc = self.context_encoder(position_signals)  # (batch, context_dim)

        # Score each candidate
        domain_expanded = domain_enc.unsqueeze(1).expand(-1, n_cand, -1)
        context_expanded = context_enc.unsqueeze(1).expand(-1, n_cand, -1)

        scorer_input = torch.cat(
            [candidate_signals, domain_expanded, context_expanded], dim=-1
        )  # (batch, n_cand, cand_dim + 2*context_dim)

        flat = scorer_input.reshape(batch * n_cand, -1)
        scores = self.candidate_scorer(flat).reshape(batch, n_cand)

        # Mask invalid candidates
        scores = scores.masked_fill(~candidate_mask, float("-inf"))

        # Position assessment (auxiliary)
        pos_assessment = self.position_head(domain_enc)

        return {
            "candidate_scores": scores,
            "position_assessment": pos_assessment,
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SoMAgentEnsemble(nn.Module):
    """All 6 specialist agents as a single module.

    Convenience wrapper for saving/loading all agents together.
    """

    def __init__(self, hidden_dim: int = 128, context_dim: int = 64):
        super().__init__()
        self.agents = nn.ModuleDict()
        for name in AGENT_NAMES:
            domain = AGENT_DOMAINS[name]
            self.agents[name] = SpecialistAgent(
                agent_name=name,
                domain_dims=domain["pos_dims"],
                hidden_dim=hidden_dim,
                context_dim=context_dim,
            )

    def forward(
        self,
        position_signals: torch.Tensor,
        candidate_signals: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Run all agents and return per-agent outputs."""
        results = {}
        for name, agent in self.agents.items():
            results[name] = agent(position_signals, candidate_signals, candidate_mask)
        return results

    def get_all_scores(
        self,
        position_signals: torch.Tensor,
        candidate_signals: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get stacked agent scores: (batch, n_agents, n_candidates)."""
        all_scores = []
        for name in AGENT_NAMES:
            out = self.agents[name](position_signals, candidate_signals, candidate_mask)
            all_scores.append(out["candidate_scores"])
        return torch.stack(all_scores, dim=1)  # (batch, 6, n_cand)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def per_agent_counts(self) -> dict[str, int]:
        return {name: agent.param_count() for name, agent in self.agents.items()}
