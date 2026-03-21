"""Composite loss for chess candidate selection.

Adapted from ShortcutForge's CompositeLoss with UW-SO adaptive weighting.

Components:
  L_candidate: Cross-entropy over candidate selection (primary)
  L_plan: Cross-entropy over plan type validation (secondary)
  L_margin: Contrastive margin between best and second-best candidate
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as f


class ChessCompositeLoss(nn.Module):
    """Multi-task loss with UW-SO (Uncertainty-Weighted Soft-Optimal) weighting.

    Learns per-component log-sigma parameters that automatically balance
    the loss components during training.
    """

    def __init__(self, margin: float = 0.3, plan_weight: float = 0.2) -> None:
        super().__init__()
        self.margin = margin
        self.plan_weight = plan_weight

        # UW-SO: learnable log-sigma per component
        self.log_sigma_candidate = nn.Parameter(torch.zeros(1))
        self.log_sigma_plan = nn.Parameter(torch.zeros(1))
        self.log_sigma_margin = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        candidate_logits: torch.Tensor,
        plan_logits: torch.Tensor,
        candidate_targets: torch.Tensor,
        plan_targets: torch.Tensor,
        second_best_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute composite loss.

        Args:
            candidate_logits: [batch, 5] from decoder.
            plan_logits: [batch, 7] from decoder.
            candidate_targets: [batch] int, best candidate index.
            plan_targets: [batch] int, plan type index.
            second_best_targets: [batch] int, second-best index (for margin).

        Returns:
            Dict with "total", "candidate", "plan", "margin" losses.
        """
        # L_candidate: primary objective
        l_cand = f.cross_entropy(candidate_logits, candidate_targets)

        # L_plan: secondary objective
        l_plan = f.cross_entropy(plan_logits, plan_targets)

        # L_margin: contrastive between best and second-best
        l_margin = torch.tensor(0.0, device=candidate_logits.device)
        if second_best_targets is not None:
            best_scores = candidate_logits.gather(
                1, candidate_targets.unsqueeze(1)
            ).squeeze(1)
            second_scores = candidate_logits.gather(
                1, second_best_targets.unsqueeze(1)
            ).squeeze(1)
            # margin loss: best should be at least `margin` better than second
            l_margin = f.relu(
                self.margin - (best_scores - second_scores)
            ).mean()

        # UW-SO weighting
        w_cand = torch.exp(-self.log_sigma_candidate)
        w_plan = torch.exp(-self.log_sigma_plan)
        w_margin = torch.exp(-self.log_sigma_margin)

        total = (
            w_cand * l_cand + self.log_sigma_candidate
            + self.plan_weight * (w_plan * l_plan + self.log_sigma_plan)
            + w_margin * l_margin + self.log_sigma_margin
        )

        return {
            "total": total,
            "candidate": l_cand,
            "plan": l_plan,
            "margin": l_margin,
        }
