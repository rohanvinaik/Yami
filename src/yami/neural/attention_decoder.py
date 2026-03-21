"""Attention-based candidate decoder (Config E).

Instead of scoring candidates from a flat bridge output, this decoder
applies self-attention over per-candidate encodings so candidates can
interact (e.g., "this move is better BECAUSE the alternative is risky").
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionCandidateDecoder(nn.Module):
    """Decoder with self-attention over candidate encodings.

    Architecture:
      - Project bridge output to context vector
      - Cross-attend: context queries candidate encodings
      - Per-candidate scoring from attended representations
      - Plan head from bridge output directly

    Args:
        bridge_dim: Bridge output dimension.
        candidate_dim: Per-candidate encoding dimension from encoder.
        num_heads: Attention heads.
        hidden_dim: FF hidden dimension.
        max_candidates: Max candidate slots.
        num_plan_types: Plan type count.
    """

    def __init__(
        self,
        bridge_dim: int = 128,
        candidate_dim: int = 48,
        num_heads: int = 4,
        hidden_dim: int = 256,
        max_candidates: int = 5,
        num_plan_types: int = 7,
    ) -> None:
        super().__init__()

        # Project bridge output to query for cross-attention
        self.bridge_to_query = nn.Linear(bridge_dim, candidate_dim)

        # Self-attention over candidates
        self.self_attn = nn.MultiheadAttention(
            embed_dim=candidate_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Cross-attention: bridge queries candidates
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=candidate_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Per-candidate scoring
        self.candidate_scorer = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Plan head (from bridge directly)
        self.plan_head = nn.Sequential(
            nn.Linear(bridge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_plan_types),
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(bridge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        bridge_output: torch.Tensor,
        candidate_encodings: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode using attention over candidates.

        Args:
            bridge_output: [batch, bridge_dim] from InformationBridge.
            candidate_encodings: [batch, 5, candidate_dim] from encoder.
            candidate_mask: [batch, 5] bool, True = valid candidate.

        Returns:
            Dict with "plan_logits", "candidate_logits", "confidence".
        """
        # Self-attention over candidates
        key_padding_mask = None
        if candidate_mask is not None:
            key_padding_mask = ~candidate_mask  # True = ignore

        attended, _ = self.self_attn(
            candidate_encodings, candidate_encodings, candidate_encodings,
            key_padding_mask=key_padding_mask,
        )

        # Cross-attention: bridge queries attended candidates
        query = self.bridge_to_query(bridge_output).unsqueeze(1)  # [batch, 1, cand_dim]
        cross_out, _ = self.cross_attn(
            query, attended, attended,
            key_padding_mask=key_padding_mask,
        )

        # Score each candidate individually
        scores = self.candidate_scorer(attended).squeeze(-1)  # [batch, 5]

        if candidate_mask is not None:
            scores = scores.masked_fill(~candidate_mask, float("-inf"))

        # Plan and confidence from bridge
        plan_logits = self.plan_head(bridge_output)
        confidence = torch.sigmoid(self.confidence_head(bridge_output))

        return {
            "plan_logits": plan_logits,
            "candidate_logits": scores,
            "confidence": confidence,
        }
