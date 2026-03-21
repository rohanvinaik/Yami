"""Fast inference wrapper for the trained neural model.

Drop-in replacement for the Claude API call in llm_decision.py.
Loads a trained checkpoint and performs inference in <10ms on CPU.
"""

from __future__ import annotations

from pathlib import Path

import chess
import torch

from yami.datagen.contracts import CANDIDATE_FEAT_DIM, MOTIF_VOCAB, RISK_LEVELS
from yami.datagen.feature_extractor import (
    _ACTIVITY_MAP,
    _MATERIAL_MAP,
    _PIECE_TYPE_IDX,
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
from yami.tactical_scoper import PIECE_VALUES

_MAX_MATERIAL = 2 * (900 + 2 * 500 + 2 * 330 + 2 * 320 + 8 * 100)


class NeuralDecider:
    """Neural candidate selector — replaces the LLM API call."""

    def __init__(
        self,
        checkpoint_path: Path | str,
        config: NeuralConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = config or NeuralConfig()
        self.device = torch.device(device)

        self.encoder = ChessPositionEncoder(
            output_dim=self.config.encoder_output_dim,
            candidate_dim=self.config.candidate_dim,
            embed_dim=self.config.embed_dim,
        )
        self.bridge = InformationBridge(
            input_dim=self.config.encoder_output_dim,
            bridge_dim=self.config.bridge_dim,
        )
        self.use_attention = self.config.variant == "E"
        if self.use_attention:
            from yami.neural.attention_decoder import AttentionCandidateDecoder
            self.decoder = AttentionCandidateDecoder(
                bridge_dim=self.config.bridge_dim,
                candidate_dim=self.config.candidate_dim,
                num_heads=self.config.attention_heads,
                hidden_dim=self.config.decoder_hidden_dim,
            )
        else:
            self.decoder = ChessTernaryDecoder(
                input_dim=self.config.bridge_dim,
                hidden_dim=self.config.decoder_hidden_dim,
                num_layers=self.config.decoder_num_layers,
                ternary_enabled=self.config.ternary_enabled,
                partial_ternary=self.config.partial_ternary,
            )

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
        if not candidates:
            return None, 0.0

        tensors = self._build_tensors(board, candidates, plan, profile)

        enc_args = dict(
            profile=tensors["profile"],
            plan_type=tensors["plan_type"],
            plan_activation=tensors["plan_activation"],
            candidate_features=tensors["candidate_features"],
            candidate_mask=tensors["candidate_mask"],
            profile_continuous=tensors["profile_continuous"],
        )
        if self.use_attention:
            enc_args["return_candidate_encodings"] = True
            enc_out, cand_encs = self.encoder(**enc_args)
            bridge_out = self.bridge(enc_out)
            outputs = self.decoder(bridge_out, cand_encs, tensors["candidate_mask"])
        else:
            enc_out = self.encoder(**enc_args)
            bridge_out = self.bridge(enc_out)
            outputs = self.decoder(bridge_out, tensors["candidate_mask"])

        logits = outputs["candidate_logits"][0]
        confidence = outputs["confidence"][0, 0].item()
        pred_idx = logits.argmax().item()

        if pred_idx < len(candidates):
            return candidates[pred_idx].move, confidence
        return candidates[0].move, confidence

    def _build_tensors(
        self,
        board: chess.Board,
        candidates: list[AnnotatedCandidate],
        plan: PlanTemplate,
        profile: PositionalProfile,
    ) -> dict[str, torch.Tensor]:
        dev = self.device

        p = torch.tensor([[
            _MATERIAL_MAP.get(profile.material, 1),
            _STRUCTURE_MAP.get(profile.structure, 1),
            _ACTIVITY_MAP.get(profile.activity, 0),
            _SAFETY_MAP.get(profile.safety, 0),
            _SAFETY_MAP.get(profile.opponent_safety, 0),
            _TEMPO_MAP.get(profile.tempo, 1),
        ]], dtype=torch.long, device=dev)

        # Board-level continuous features + nav_vector
        total_mat = sum(
            PIECE_VALUES.get(pc.piece_type, 0) for pc in board.piece_map().values()
        )
        from yami.navigator import compute_navigation_vector
        nav = compute_navigation_vector(board)
        profile_cont = torch.tensor([[
            total_mat / _MAX_MATERIAL,
            total_mat / _MAX_MATERIAL,
            min(board.fullmove_number, 100) / 100.0,
            float(nav.aggression),
            float(nav.piece_domain),
            float(nav.complexity),
            float(nav.initiative),
            float(nav.king_pressure),
            float(nav.phase),
            0.0,  # gm_top_move_freq (computed at data gen, not inference)
            0.0,  # som_convergence
            0.0,  # interference_score
        ]], dtype=torch.float32, device=dev)

        plan_type = torch.tensor(
            [_PLAN_MAP.get(plan.plan_type, 0)], dtype=torch.long, device=dev
        )
        plan_act = torch.tensor(
            [[plan.activation_score]], dtype=torch.float32, device=dev
        )

        our_king = board.king(board.turn)
        opp_king = board.king(not board.turn)

        cand_feats = torch.zeros(1, 5, CANDIDATE_FEAT_DIM, device=dev)
        cand_mask = torch.zeros(1, 5, dtype=torch.bool, device=dev)

        for i, c in enumerate(candidates[:5]):
            cand_mask[0, i] = True

            motif_flags = [0.0] * len(MOTIF_VOCAB)
            for m in c.tactical_motifs:
                idx = MOTIF_TO_IDX.get(m)
                if idx is not None:
                    motif_flags[idx] = 1.0

            risk_onehot = [0.0] * 4
            risk_onehot[RISK_LEVELS.get(c.risk, 0)] = 1.0

            # Piece type
            piece = board.piece_at(c.move.from_square)
            piece_onehot = [0.0] * 6
            if piece:
                pt_idx = _PIECE_TYPE_IDX.get(piece.piece_type)
                if pt_idx is not None:
                    piece_onehot[pt_idx] = 1.0

            # Geometry
            to_sq = c.move.to_square
            to_f = chess.square_file(to_sq)
            to_r = chess.square_rank(to_sq)
            centrality = (4.0 - (abs(3.5 - to_f) + abs(3.5 - to_r))) / 4.0
            d_opp = chess.square_distance(to_sq, opp_king) / 7.0 if opp_king else 1.0
            d_own = chess.square_distance(to_sq, our_king) / 7.0 if our_king else 1.0

            feats = (
                motif_flags  # 9
                + [c.plan_alignment]  # 1
                + [c.positional_eval]  # 1
                + risk_onehot  # 4
                + [float("capture" in c.tactical_motifs)]  # 1
                + [float("check" in c.tactical_motifs)]  # 1
                + [c.see_value / 900.0]  # 1
                + piece_onehot  # 6
                + [centrality]  # 1
                + [d_opp]  # 1
                + [d_own]  # 1
                + [float(board.is_castling(c.move))]  # 1
                + [0.0]  # opponent_mobility (skip at inference for speed)
                + [0.0]  # pawn_structure_change (skip at inference)
            )  # total = 30
            cand_feats[0, i] = torch.tensor(feats, dtype=torch.float32)

        return {
            "profile": p,
            "profile_continuous": profile_cont,
            "plan_type": plan_type,
            "plan_activation": plan_act,
            "candidate_features": cand_feats,
            "candidate_mask": cand_mask,
        }
