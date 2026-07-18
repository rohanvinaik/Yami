"""Learned Signal Fusion Model — the SoM arbitrator.

Replaces hand-tuned coherence weights with a learned fusion function.
The infrastructure computes all signals deterministically; this model
learns which signal combinations predict good moves.

Architecture:
    Position signals (58) → PositionEncoder (→ 64)
    Candidate signals (50) × 5 → SharedCandidateEncoder (→ 32) × 5
    Cross-attention: position context modulates candidate scoring
    Candidate scorer: (32 → 1) × 5 → softmax → selection

~15K parameters. The infrastructure does the understanding.
The model just learns which signals to trust.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from yami.position_signals import PositionSignals
from yami.candidate_signals import CandidateSignalVector


class PositionEncoder(nn.Module):
    """Encode per-position signals into a context vector."""

    def __init__(self, input_dim: int = 58, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) → (batch, output_dim)"""
        return self.net(x)


class CandidateEncoder(nn.Module):
    """Shared encoder for per-candidate signals. Same weights for all slots."""

    def __init__(self, input_dim: int = 50, output_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, num_candidates, input_dim) → (batch, num_candidates, output_dim)"""
        batch, n_cand, feat = x.shape
        flat = x.reshape(batch * n_cand, feat)
        encoded = self.net(flat)
        return encoded.reshape(batch, n_cand, -1)


class SignalCrossAttention(nn.Module):
    """Position context attends to candidates to modulate scoring.

    The position says "this is an attacking position" — that changes
    which candidate signals matter (tactical > positional).
    """

    def __init__(
        self, pos_dim: int = 64, cand_dim: int = 32, num_heads: int = 2
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = cand_dim // num_heads

        self.q_proj = nn.Linear(pos_dim, cand_dim)  # query from position
        self.k_proj = nn.Linear(cand_dim, cand_dim)  # keys from candidates
        self.v_proj = nn.Linear(cand_dim, cand_dim)  # values from candidates
        self.out_proj = nn.Linear(cand_dim, cand_dim)

    def forward(
        self,
        position: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        position: (batch, pos_dim)
        candidates: (batch, num_cand, cand_dim)
        mask: (batch, num_cand) — True for valid candidates
        Returns: (batch, num_cand, cand_dim)
        """
        batch, n_cand, cand_dim = candidates.shape

        # Query from position context (expand to attend to each candidate)
        q = self.q_proj(position).unsqueeze(1)  # (batch, 1, cand_dim)
        k = self.k_proj(candidates)  # (batch, n_cand, cand_dim)
        v = self.v_proj(candidates)  # (batch, n_cand, cand_dim)

        # Multi-head reshape
        q = q.reshape(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, n_cand, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, n_cand, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scale = self.head_dim ** 0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # (batch, heads, 1, n_cand)

        if mask is not None:
            attn_mask = ~mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, n_cand)
            attn = attn.masked_fill(attn_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, 0.0)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch, heads, 1, head_dim)
        out = out.transpose(1, 2).reshape(batch, 1, cand_dim)

        # Combine with original candidates (residual)
        context = self.out_proj(out)  # (batch, 1, cand_dim)
        return candidates + context  # broadcast: position context added to all candidates


class CandidateScorer(nn.Module):
    """Score each candidate. Shared weights."""

    def __init__(self, input_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, num_candidates, input_dim) → (batch, num_candidates)"""
        return self.net(x).squeeze(-1)


class BankProfileHead(nn.Module):
    """Auxiliary head: predict game outcome from position signals.

    Teaches the model which signal profiles correlate with winning.
    """

    def __init__(self, pos_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pos_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Tanh(),  # output in [-1, 1]: -1=losing, 0=drawn, +1=winning
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, pos_dim) → (batch, 1)"""
        return self.net(x)


class SignalFusionModel(nn.Module):
    """The complete learned signal fusion model.

    Takes position signals + candidate signals, outputs candidate selection.
    ~15K parameters. The SoM arbitrator.
    """

    def __init__(
        self,
        pos_dim: int = 58,
        cand_dim: int = 50,
        pos_hidden: int = 64,
        cand_hidden: int = 32,
        num_heads: int = 2,
        max_candidates: int = 5,
    ):
        super().__init__()
        self.max_candidates = max_candidates

        self.pos_encoder = PositionEncoder(pos_dim, pos_hidden)
        self.cand_encoder = CandidateEncoder(cand_dim, cand_hidden)
        self.cross_attention = SignalCrossAttention(pos_hidden, cand_hidden, num_heads)
        self.scorer = CandidateScorer(cand_hidden)
        self.bank_profile_head = BankProfileHead(pos_hidden)

    def forward(
        self,
        position_signals: torch.Tensor,
        candidate_signals: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        position_signals: (batch, pos_dim)
        candidate_signals: (batch, max_candidates, cand_dim)
        candidate_mask: (batch, max_candidates) — True for valid candidates

        Returns dict with:
            candidate_logits: (batch, max_candidates) — raw scores
            bank_profile_pred: (batch, 1) — predicted game outcome
        """
        # Encode
        pos_enc = self.pos_encoder(position_signals)  # (batch, pos_hidden)
        cand_enc = self.cand_encoder(candidate_signals)  # (batch, n_cand, cand_hidden)

        # Cross-attention: position modulates candidates
        cand_contextualized = self.cross_attention(
            pos_enc, cand_enc, candidate_mask
        )

        # Score candidates
        logits = self.scorer(cand_contextualized)  # (batch, n_cand)

        # Mask invalid candidates
        logits = logits.masked_fill(~candidate_mask, float("-inf"))

        # Bank profile prediction (auxiliary)
        bank_pred = self.bank_profile_head(pos_enc)

        return {
            "candidate_logits": logits,
            "bank_profile_pred": bank_pred,
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FusionLoss(nn.Module):
    """Combined loss for the fusion model.

    L = L_candidate + alpha * L_bank_profile
    """

    def __init__(self, bank_weight: float = 0.1):
        super().__init__()
        self.bank_weight = bank_weight
        self.candidate_loss = nn.CrossEntropyLoss()
        self.bank_loss = nn.MSELoss()

    def forward(
        self,
        candidate_logits: torch.Tensor,
        bank_profile_pred: torch.Tensor,
        best_idx: torch.Tensor,
        game_outcome: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        candidate_logits: (batch, max_candidates)
        bank_profile_pred: (batch, 1)
        best_idx: (batch,) — index of best candidate
        game_outcome: (batch, 1) — optional, -1/0/+1
        """
        l_cand = self.candidate_loss(candidate_logits, best_idx)

        total = l_cand
        result = {"total": total, "candidate": l_cand}

        if game_outcome is not None:
            l_bank = self.bank_loss(bank_profile_pred, game_outcome)
            total = l_cand + self.bank_weight * l_bank
            result["total"] = total
            result["bank_profile"] = l_bank

        return result
