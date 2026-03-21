"""Chess position encoder — structured features to fixed-dim embedding.

Replaces all-MiniLM-L6-v2 from ShortcutForge. The input is structured
chess data (not natural language), so we encode it directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from yami.datagen.contracts import MOTIF_VOCAB


class CandidateEncoder(nn.Module):
    """Encode a single candidate's features into a fixed-dim vector.

    Shared weights across all 5 candidate slots — the model learns
    candidate quality features, not positional bias.
    """

    def __init__(self, out_dim: int = 48) -> None:
        super().__init__()
        # Input features per candidate:
        #   motif_flags (9) + plan_alignment (1) + positional_eval (1)
        #   + risk_level_onehot (4) + is_capture (1) + is_check (1) + see_value (1)
        input_dim = len(MOTIF_VOCAB) + 1 + 1 + 4 + 1 + 1 + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode candidate features.

        Args:
            x: [batch, input_dim] candidate feature vector.

        Returns:
            [batch, out_dim] encoded candidate.
        """
        return self.net(x)


class ChessPositionEncoder(nn.Module):
    """Encode structured chess position into a fixed-dim embedding.

    Architecture:
      - Position profile: 6 categorical → embeddings → 64-dim
      - Plan context: 1 categorical + 1 float → 32-dim
      - Per-candidate: shared CandidateEncoder → 48-dim × 5
      - Concat → Linear → output_dim (384)

    Args:
        output_dim: Final embedding dimension (matches bridge input).
        candidate_dim: Per-candidate encoding dimension.
    """

    def __init__(
        self,
        output_dim: int = 384,
        candidate_dim: int = 48,
        embed_dim: int = 8,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim

        # Positional profile embeddings
        self.material_emb = nn.Embedding(3, embed_dim)   # behind/equal/ahead
        self.structure_emb = nn.Embedding(6, embed_dim)  # 6 structure types
        self.activity_emb = nn.Embedding(3, embed_dim)
        self.safety_emb = nn.Embedding(3, embed_dim)
        self.opp_safety_emb = nn.Embedding(3, embed_dim)
        self.tempo_emb = nn.Embedding(3, embed_dim)

        profile_dim = 6 * embed_dim  # 48
        self.profile_proj = nn.Sequential(
            nn.Linear(profile_dim, 64),
            nn.ReLU(),
        )

        # Plan context
        self.plan_type_emb = nn.Embedding(7, 16)  # 7 plan types
        self.plan_proj = nn.Sequential(
            nn.Linear(16 + 1, 32),  # +1 for activation score
            nn.ReLU(),
        )

        # Shared candidate encoder
        self.candidate_encoder = CandidateEncoder(out_dim=candidate_dim)

        # Final projection
        # 64 (profile) + 32 (plan) + 5*candidate_dim (candidates)
        total_dim = 64 + 32 + 5 * candidate_dim
        self.final_proj = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        profile: torch.Tensor,
        plan_type: torch.Tensor,
        plan_activation: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a chess position.

        Args:
            profile: [batch, 6] int tensor (material, structure, activity,
                     safety, opp_safety, tempo).
            plan_type: [batch] int tensor (0-6).
            plan_activation: [batch, 1] float tensor.
            candidate_features: [batch, 5, feat_dim] float tensor.
            candidate_mask: [batch, 5] bool tensor (True = real candidate).

        Returns:
            [batch, output_dim] position embedding.
        """
        # Encode positional profile
        p_mat = self.material_emb(profile[:, 0])
        p_str = self.structure_emb(profile[:, 1])
        p_act = self.activity_emb(profile[:, 2])
        p_saf = self.safety_emb(profile[:, 3])
        p_opp = self.opp_safety_emb(profile[:, 4])
        p_tmp = self.tempo_emb(profile[:, 5])
        profile_cat = torch.cat([p_mat, p_str, p_act, p_saf, p_opp, p_tmp], dim=-1)
        profile_enc = self.profile_proj(profile_cat)  # [batch, 64]

        # Encode plan
        plan_emb = self.plan_type_emb(plan_type)  # [batch, 16]
        plan_cat = torch.cat([plan_emb, plan_activation], dim=-1)  # [batch, 17]
        plan_enc = self.plan_proj(plan_cat)  # [batch, 32]

        # Encode candidates (shared weights, applied per-slot)
        cand_encs = []
        for i in range(5):
            cand_feat = candidate_features[:, i, :]  # [batch, feat_dim]
            enc = self.candidate_encoder(cand_feat)  # [batch, 48]
            # Zero out padded candidates
            mask_i = candidate_mask[:, i].unsqueeze(-1).float()
            enc = enc * mask_i
            cand_encs.append(enc)
        cand_enc = torch.cat(cand_encs, dim=-1)  # [batch, 5*48]

        # Concat and project
        combined = torch.cat([profile_enc, plan_enc, cand_enc], dim=-1)
        return self.final_proj(combined)
