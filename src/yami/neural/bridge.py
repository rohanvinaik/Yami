"""Information bridge — bottleneck between continuous encoder and ternary decoder.

Reused verbatim from ShortcutForge's Balanced Sashimi architecture.
Compresses encoder output to a fixed-dim representation suitable for
the ternary decoder's discrete decision space.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class InformationBridge(nn.Module):
    """Bottleneck bridge from continuous to discrete space.

    Args:
        input_dim: Encoder output dimension.
        bridge_dim: Compressed representation dimension.
    """

    def __init__(self, input_dim: int = 384, bridge_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.bridge_dim = bridge_dim
        self.projection = nn.Linear(input_dim, bridge_dim)
        self.norm = nn.LayerNorm(bridge_dim)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Compress encoder output through bottleneck.

        Args:
            encoder_output: [batch, input_dim] from ChessPositionEncoder.

        Returns:
            [batch, bridge_dim] compressed representation.
        """
        return self.norm(self.projection(encoder_output))
