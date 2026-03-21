"""Chess ternary decoder — {-1, 0, +1} weight layers for candidate selection.

Adapted from ShortcutForge's TernaryDecoder. Uses Straight-Through Estimator
(STE) for training: fp32 shadow weights with quantized forward pass.

Tier 1: Plan type validation (7-class)
Tier 2: Candidate selection (5-class with masking)
Confidence: Scalar (continuous, not ternary)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as f


def ternary_quantize(weights: torch.Tensor) -> torch.Tensor:
    """Quantize continuous weights to {-1, 0, +1} via STE.

    Per-row threshold: 0.7 * mean(|w|).
    STE: gradient of this function is identity (straight-through).
    """
    threshold = 0.7 * weights.abs().mean(dim=-1, keepdim=True)
    quantized = torch.zeros_like(weights)
    quantized[weights > threshold] = 1.0
    quantized[weights < -threshold] = -1.0
    return weights + (quantized - weights).detach()


class TernaryLinear(nn.Module):
    """Linear layer with ternary weights via STE.

    Maintains fp32 shadow weights for gradient computation.
    Forward pass uses quantized {-1, 0, +1} weights.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_weight = ternary_quantize(self.weight)
        return f.linear(x, q_weight, self.bias)


class ChessTernaryDecoder(nn.Module):
    """Ternary decoder for chess plan validation + candidate selection.

    Args:
        input_dim: Bridge output dimension.
        hidden_dim: Internal layer dimension.
        num_plan_types: Number of plan types (default 7).
        max_candidates: Maximum candidates (default 5).
        num_layers: Number of hidden TernaryLinear layers.
        ternary_enabled: Use ternary quantization (disable for ablation).
        partial_ternary: Ternary hidden layers, continuous heads.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_plan_types: int = 7,
        max_candidates: int = 5,
        num_layers: int = 2,
        ternary_enabled: bool = True,
        partial_ternary: bool = False,
    ) -> None:
        super().__init__()

        hidden_cls = TernaryLinear if ternary_enabled else nn.Linear
        head_cls = nn.Linear if (not ternary_enabled or partial_ternary) else TernaryLinear

        # Shared hidden stack
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(hidden_cls(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.layers = nn.Sequential(*layers)

        # Prediction heads
        self.plan_head = head_cls(hidden_dim, num_plan_types)
        self.candidate_head = head_cls(hidden_dim, max_candidates)
        self.confidence_head = nn.Linear(hidden_dim, 1)  # always continuous

    def forward(
        self,
        bridge_output: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode bridge representation to plan + candidate logits.

        Args:
            bridge_output: [batch, input_dim] from InformationBridge.
            candidate_mask: [batch, 5] bool, True = valid candidate.

        Returns:
            Dict with "plan_logits", "candidate_logits", "confidence".
        """
        hidden = self.layers(bridge_output)

        plan_logits = self.plan_head(hidden)
        candidate_logits = self.candidate_head(hidden)
        confidence = torch.sigmoid(self.confidence_head(hidden))

        # Mask invalid candidates with large negative value
        if candidate_mask is not None:
            candidate_logits = candidate_logits.masked_fill(
                ~candidate_mask, float("-inf")
            )

        return {
            "plan_logits": plan_logits,
            "candidate_logits": candidate_logits,
            "confidence": confidence,
        }
