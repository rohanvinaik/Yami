"""Fast inference wrapper for the trained neural model.

Drop-in replacement for the Claude API call in llm_decision.py.
Loads a trained checkpoint and performs inference in <10ms on CPU.
"""

from __future__ import annotations

from pathlib import Path

import chess
import torch

from yami.datagen.contracts import MOTIF_VOCAB, RISK_LEVELS
from yami.datagen.feature_extractor import (
    _ACTIVITY_MAP,
    _MATERIAL_MAP,
    _PLAN_MAP,
    _SAFETY_MAP,
    _STRUCTURE_MAP,
    _TEMPO_MAP,
    MOTIF_TO_IDX,
)
from yami.models import AnnotatedCandidate, PlanTemplate, PositionalProfile
from yami.neural.bridge import InformationBridge
from yami.neural.config import NeuralConfig
from yami.neural.decoder import ChessTernaryDecoder
from yami.neural.encoder import ChessPositionEncoder


class NeuralDecider:
    """Neural candidate selector — replaces the LLM API call.

    Loads a trained encoder-bridge-decoder checkpoint and selects
    the best candidate from the infrastructure's annotated set.
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        config: NeuralConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = config or NeuralConfig()
        self.device = torch.device(device)

        # Build model
        self.encoder = ChessPositionEncoder(
            output_dim=self.config.encoder_output_dim,
            candidate_dim=self.config.candidate_dim,
            embed_dim=self.config.embed_dim,
        )
        self.bridge = InformationBridge(
            input_dim=self.config.encoder_output_dim,
            bridge_dim=self.config.bridge_dim,
        )
        self.decoder = ChessTernaryDecoder(
            input_dim=self.config.bridge_dim,
            hidden_dim=self.config.decoder_hidden_dim,
            num_layers=self.config.decoder_num_layers,
            ternary_enabled=self.config.ternary_enabled,
            partial_ternary=self.config.partial_ternary,
        )

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.bridge.load_state_dict(ckpt["bridge"])
        self.decoder.load_state_dict(ckpt["decoder"])

        self.encoder.to(self.device).eval()
        self.bridge.to(self.device).eval()
        self.decoder.to(self.device).eval()

    @torch.no_grad()
    def decide(
        self,
        board: chess.Board,
        candidates: list[AnnotatedCandidate],
        plan: PlanTemplate,
        profile: PositionalProfile,
    ) -> tuple[chess.Move | None, float]:
        """Select the best candidate using the neural model.

        Returns:
            (chosen_move, confidence)
        """
        if not candidates:
            return None, 0.0

        # Build input tensors
        tensors = self._build_tensors(candidates, plan, profile)

        # Forward pass
        enc = self.encoder(
            tensors["profile"],
            tensors["plan_type"],
            tensors["plan_activation"],
            tensors["candidate_features"],
            tensors["candidate_mask"],
        )
        bridge_out = self.bridge(enc)
        outputs = self.decoder(bridge_out, tensors["candidate_mask"])

        # Get prediction
        logits = outputs["candidate_logits"][0]  # [5]
        confidence = outputs["confidence"][0, 0].item()
        pred_idx = logits.argmax().item()

        if pred_idx < len(candidates):
            return candidates[pred_idx].move, confidence
        return candidates[0].move, confidence

    def _build_tensors(
        self,
        candidates: list[AnnotatedCandidate],
        plan: PlanTemplate,
        profile: PositionalProfile,
    ) -> dict[str, torch.Tensor]:
        """Convert Yami types to model input tensors."""
        dev = self.device

        # Profile
        p = torch.tensor([[
            _MATERIAL_MAP.get(profile.material, 1),
            _STRUCTURE_MAP.get(profile.structure, 1),
            _ACTIVITY_MAP.get(profile.activity, 0),
            _SAFETY_MAP.get(profile.safety, 0),
            _SAFETY_MAP.get(profile.opponent_safety, 0),
            _TEMPO_MAP.get(profile.tempo, 1),
        ]], dtype=torch.long, device=dev)

        plan_type = torch.tensor(
            [_PLAN_MAP.get(plan.plan_type, 0)], dtype=torch.long, device=dev
        )
        plan_act = torch.tensor(
            [[plan.activation_score]], dtype=torch.float32, device=dev
        )

        # Candidates
        feat_dim = len(MOTIF_VOCAB) + 1 + 1 + 4 + 1 + 1 + 1
        cand_feats = torch.zeros(1, 5, feat_dim, device=dev)
        cand_mask = torch.zeros(1, 5, dtype=torch.bool, device=dev)

        for i, c in enumerate(candidates[:5]):
            cand_mask[0, i] = True
            motif_flags = [0.0] * len(MOTIF_VOCAB)
            for m in c.tactical_motifs:
                idx = MOTIF_TO_IDX.get(m)
                if idx is not None:
                    motif_flags[idx] = 1.0

            risk_onehot = [0.0] * 4
            risk_idx = RISK_LEVELS.get(c.risk, 0)
            risk_onehot[risk_idx] = 1.0

            feats = (
                motif_flags
                + [c.plan_alignment]
                + [c.positional_eval]
                + risk_onehot
                + [float("capture" in c.tactical_motifs)]
                + [float("check" in c.tactical_motifs)]
                + [0.0]  # see_value
            )
            cand_feats[0, i] = torch.tensor(feats, dtype=torch.float32)

        return {
            "profile": p,
            "plan_type": plan_type,
            "plan_activation": plan_act,
            "candidate_features": cand_feats,
            "candidate_mask": cand_mask,
        }
