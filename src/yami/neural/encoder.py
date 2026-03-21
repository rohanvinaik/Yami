"""Chess position encoder — structured features to fixed-dim embedding.

Encodes structured chess data directly (not natural language).
Supports enriched 30-dim candidate features and 3-dim board-level continuous features.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from yami.datagen.contracts import CANDIDATE_FEAT_DIM


class CandidateEncoder(nn.Module):
    """Encode a single candidate's features into a fixed-dim vector.

    Shared weights across all 5 candidate slots — the model learns
    candidate quality features, not positional bias.
    """

    def __init__(self, input_dim: int = CANDIDATE_FEAT_DIM, out_dim: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Linear(96, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChessPositionEncoder(nn.Module):
    """Encode structured chess position into a fixed-dim embedding.

    Architecture:
      - Position profile: 6 categorical + 3 continuous → 64-dim
      - Plan context: 1 categorical + 1 float → 32-dim
      - Per-candidate: shared CandidateEncoder → candidate_dim × 5
      - Concat → Linear → output_dim (384)

    Args:
        output_dim: Final embedding dimension (matches bridge input).
        candidate_dim: Per-candidate encoding dimension.
        embed_dim: Categorical embedding dimension.
        profile_continuous_dim: Number of continuous profile features.
        candidate_input_dim: Per-candidate feature dimension.
        return_candidate_encodings: If True, also return per-candidate encodings
            (needed for Config E attention decoder).
    """

    def __init__(
        self,
        output_dim: int = 384,
        candidate_dim: int = 48,
        embed_dim: int = 8,
        profile_continuous_dim: int = 9,  # 3 board features + 6 nav_vector
        candidate_input_dim: int = CANDIDATE_FEAT_DIM,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.candidate_dim = candidate_dim

        # Positional profile embeddings
        self.material_emb = nn.Embedding(3, embed_dim)
        self.structure_emb = nn.Embedding(6, embed_dim)
        self.activity_emb = nn.Embedding(3, embed_dim)
        self.safety_emb = nn.Embedding(3, embed_dim)
        self.opp_safety_emb = nn.Embedding(3, embed_dim)
        self.tempo_emb = nn.Embedding(3, embed_dim)

        profile_dim = 6 * embed_dim + profile_continuous_dim  # 48 + 3 = 51
        self.profile_proj = nn.Sequential(
            nn.Linear(profile_dim, 64),
            nn.ReLU(),
        )

        # Plan context
        self.plan_type_emb = nn.Embedding(7, 16)
        self.plan_proj = nn.Sequential(
            nn.Linear(16 + 1, 32),
            nn.ReLU(),
        )

        # Shared candidate encoder
        self.candidate_encoder = CandidateEncoder(
            input_dim=candidate_input_dim, out_dim=candidate_dim
        )

        # Final projection: 64 + 32 + 5*candidate_dim
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
        profile_continuous: torch.Tensor | None = None,
        return_candidate_encodings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode a chess position.

        Args:
            profile: [batch, 6] int tensor.
            plan_type: [batch] int tensor.
            plan_activation: [batch, 1] float tensor.
            candidate_features: [batch, 5, feat_dim] float tensor.
            candidate_mask: [batch, 5] bool tensor.
            profile_continuous: [batch, 3] float tensor (optional).
            return_candidate_encodings: Also return [batch, 5, cand_dim].

        Returns:
            [batch, output_dim] position embedding, or
            tuple of (embedding, [batch, 5, cand_dim]) if return_candidate_encodings.
        """
        # Encode positional profile
        p_mat = self.material_emb(profile[:, 0])
        p_str = self.structure_emb(profile[:, 1])
        p_act = self.activity_emb(profile[:, 2])
        p_saf = self.safety_emb(profile[:, 3])
        p_opp = self.opp_safety_emb(profile[:, 4])
        p_tmp = self.tempo_emb(profile[:, 5])
        profile_cat = torch.cat([p_mat, p_str, p_act, p_saf, p_opp, p_tmp], dim=-1)

        if profile_continuous is not None:
            profile_cat = torch.cat([profile_cat, profile_continuous], dim=-1)
        else:
            # Pad with zeros if not provided (backward compat)
            pad_dim = self.profile_proj[0].in_features - profile_cat.shape[-1]
            zeros = torch.zeros(profile_cat.shape[0], pad_dim, device=profile_cat.device)
            profile_cat = torch.cat([profile_cat, zeros], dim=-1)

        profile_enc = self.profile_proj(profile_cat)

        # Encode plan
        plan_emb = self.plan_type_emb(plan_type)
        plan_cat = torch.cat([plan_emb, plan_activation], dim=-1)
        plan_enc = self.plan_proj(plan_cat)

        # Encode candidates (shared weights, applied per-slot)
        cand_encs = []
        for i in range(5):
            cand_feat = candidate_features[:, i, :]
            enc = self.candidate_encoder(cand_feat)
            mask_i = candidate_mask[:, i].unsqueeze(-1).float()
            enc = enc * mask_i
            cand_encs.append(enc)

        cand_enc_stacked = torch.stack(cand_encs, dim=1)  # [batch, 5, cand_dim]
        cand_enc_flat = cand_enc_stacked.reshape(cand_enc_stacked.shape[0], -1)

        # Concat and project
        combined = torch.cat([profile_enc, plan_enc, cand_enc_flat], dim=-1)
        output = self.final_proj(combined)

        if return_candidate_encodings:
            return output, cand_enc_stacked
        return output
