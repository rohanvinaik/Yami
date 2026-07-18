"""Multi-Scale Progressive Revelation Engine.

Replaces the flat pipeline in engine.py with a scale-by-scale architecture
where each scale resolves uncertainties from the scale below. Information
gain is CONDITIONAL — each scale sees a smaller, harder problem because
previous scales already resolved the easy parts.

Architecture (from Data Geometry Architecture, Pattern 3):

    Scale 0: Legal + Censors → eliminate impossible moves
    Scale 1: Tactical Resolution → forced wins, checkmate
    Scale 2: Positional Evaluation → navigator OTP + structure
    Scale 3: Strategic Coherence → OTP interference + COEC amplification
    Scale 4: Temporal/SoM Context → specialist agent debate
    Scale 5: Neural Residual → fusion model on the HARD cases only

Key properties:
    - Progressive Revelation: each scale conditions on previous scales
    - COEC Amplification: co-supporting signal pairs > sum of parts
    - Residual Passing: each scale gets a SMALLER, HARDER problem
    - Minority Channel Advantage: sparse ternary, non-zero = high info
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import chess

from yami.models import (
    AnnotatedCandidate,
    GameState,
    PlanTemplate,
    PositionalProfile,
    ScopedMove,
)
from yami.navigator import NavigationVector
from yami.temporal_controller import TemporalController, TemporalState


class DecisionSource(Enum):
    SCALE_0_CENSOR = "scale_0_censor"
    SCALE_1_TACTICAL = "scale_1_tactical"
    SCALE_2_POSITIONAL = "scale_2_positional"
    SCALE_3_COHERENCE = "scale_3_coherence"
    SCALE_4_TEMPORAL = "scale_4_temporal"
    SCALE_5_NEURAL = "scale_5_neural"
    ENDGAME_TABLEBASE = "endgame_tablebase"
    OPENING_BOOK = "opening_book"


@dataclass
class ScoredCandidate:
    """A candidate move with accumulated scores from all scales."""

    move: chess.Move
    scoped: ScopedMove
    anchors: set[str] = field(default_factory=set)

    # Accumulated per-scale scores
    tactical_score: float = 0.0
    positional_score: float = 0.0
    coherence_score: float = 0.0
    temporal_score: float = 0.0

    # OTP ternary signals (built up through scales)
    ternary_signals: dict[str, int] = field(default_factory=dict)

    # COEC compositional amplification bonus
    coec_bonus: float = 0.0

    # Combined score (accumulated across scales)
    total_score: float = 0.0


@dataclass
class ScaleResult:
    """Output of one scale in the progressive pipeline."""

    selected: chess.Move | None
    residual: list[ScoredCandidate]
    ambiguity: float  # 0.0 = fully resolved, 1.0 = maximally ambiguous
    scale_signals: dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


@dataclass
class ProgressiveDecision:
    """Full result of the progressive revelation pipeline."""

    move: chess.Move | None
    source: DecisionSource
    resolving_scale: int
    scale_results: list[ScaleResult] = field(default_factory=list)
    nav_vector: NavigationVector | None = None
    temporal_state: TemporalState | None = None
    legal_move_count: int = 0
    candidates_at_resolution: int = 0
    # Wayfinder-derived: decision capsule and classification
    decision_capsule: Any | None = None
    infrastructure_resolved: bool = False


# Type alias for scale functions
ScaleFunction = Callable[
    [chess.Board, list[ScoredCandidate], dict[str, Any]],
    ScaleResult,
]


# Default ambiguity thresholds per scale
DEFAULT_THRESHOLDS = {
    0: 0.05,  # Scale 0: only 1 candidate survives censoring
    1: 0.10,  # Scale 1: forced tactical sequence
    2: 0.20,  # Scale 2: positional signals unanimous
    3: 0.25,  # Scale 3: coherence strongly agrees
    4: 0.30,  # Scale 4: SoM convergence high
    # Scale 5 always resolves (terminal)
}


class ProgressiveRevealEngine:
    """Multi-scale progressive revelation engine.

    Each scale receives only the RESIDUAL candidates from below.
    Each scale measures its own AMBIGUITY.
    If ambiguity < threshold, short-circuit (no higher scales needed).
    The neural model at Scale 5 only sees the HARD cases.
    """

    def __init__(
        self,
        use_opening_book: bool = True,
        use_endgame_tables: bool = True,
        use_gm_patterns: bool = True,
        use_klines: bool = False,
        use_neural: bool = False,
        neural_checkpoint: str | None = None,
        kline_db_path: str | None = None,
        gm_db_path: str | None = None,
        ambiguity_thresholds: dict[int, float] | None = None,
    ):
        self.use_opening_book = use_opening_book
        self.use_endgame_tables = use_endgame_tables
        self.thresholds = ambiguity_thresholds or DEFAULT_THRESHOLDS
        self.state = GameState()

        # Subsystems (initialized once, shared across scales via context)
        from yami.negative_learning import create_default_censors

        self._learned_censors = create_default_censors()
        self._temporal = TemporalController()

        from yami.opponent_profile import OpponentProfiler

        self._opponent_profiler = OpponentProfiler()

        self._gm_db = None
        if use_gm_patterns:
            from yami.gm_patterns import GMPatternDB

            self._gm_db = GMPatternDB(gm_db_path)

        self._klines = None
        if use_klines and kline_db_path:
            from yami.kline_memory import KLineMemory

            self._klines = KLineMemory(kline_db_path)

        self._fusion_model = None
        if use_neural and neural_checkpoint:
            import torch
            from yami.neural.fusion_model import SignalFusionModel

            self._fusion_model = SignalFusionModel()
            cp = torch.load(neural_checkpoint, map_location="cpu", weights_only=True)
            self._fusion_model.load_state_dict(cp["model"])
            self._fusion_model.eval()

        # Build scale function list
        from yami.scale_functions import (
            scale_0_legal_and_censors,
            scale_1_tactical_resolution,
            scale_2_positional_evaluation,
            scale_3_strategic_coherence,
            scale_4_temporal_context,
            scale_5_neural_residual,
        )

        self.scales: list[ScaleFunction] = [
            scale_0_legal_and_censors,
            scale_1_tactical_resolution,
            scale_2_positional_evaluation,
            scale_3_strategic_coherence,
            scale_4_temporal_context,
            scale_5_neural_residual,
        ]

    def reset(self) -> None:
        self.state = GameState()
        self._temporal.reset()
        self._opponent_profiler.reset()

    def decide(self, board: chess.Board | None = None) -> ProgressiveDecision:
        """Run the progressive revelation pipeline."""
        if board is None:
            board = self.state.board

        # Pre-pipeline: endgame tablebase short-circuit
        if self.use_endgame_tables:
            from yami import endgame_resolver

            if endgame_resolver.can_resolve(board):
                tb_move = endgame_resolver.resolve(board)
                if tb_move is not None:
                    return ProgressiveDecision(
                        move=tb_move,
                        source=DecisionSource.ENDGAME_TABLEBASE,
                        resolving_scale=-1,
                        legal_move_count=board.legal_moves.count(),
                    )

        # Pre-pipeline: opening book short-circuit
        if self.use_opening_book and self.state.in_book:
            from yami import opening_book

            book_move = opening_book.get_book_move(board)
            if book_move is None:
                book_move = opening_book.builtin_lookup(board)
            if book_move is not None:
                return ProgressiveDecision(
                    move=book_move,
                    source=DecisionSource.OPENING_BOOK,
                    resolving_scale=-1,
                    legal_move_count=board.legal_moves.count(),
                )
            else:
                self.state.in_book = False

        # Build shared context (computed ONCE, shared across all scales)
        context = self._build_context(board)

        # Start with empty candidates (Scale 0 generates them)
        candidates: list[ScoredCandidate] = []
        scale_results: list[ScaleResult] = []

        for scale_idx, scale_fn in enumerate(self.scales):
            result = scale_fn(board, candidates, context)
            scale_results.append(result)

            threshold = self.thresholds.get(scale_idx, 0.5)
            if result.resolved or result.ambiguity < threshold:
                move = result.selected
                if move is None and result.residual:
                    move = result.residual[0].move

                # Verify legality
                if move is not None and move not in board.legal_moves:
                    move = list(board.legal_moves)[0] if board.legal_moves else None

                # Check if this was an infrastructure-resolved decision
                infra_resolved = (
                    result.scale_signals.get("decision_type")
                    == "infrastructure_resolved"
                )

                return ProgressiveDecision(
                    move=move,
                    source=DecisionSource(f"scale_{scale_idx}_"
                                          + ["censor", "tactical", "positional",
                                             "coherence", "temporal", "neural"][scale_idx]),
                    resolving_scale=scale_idx,
                    scale_results=scale_results,
                    nav_vector=context.get("nav_vector"),
                    temporal_state=context.get("temporal_state"),
                    legal_move_count=context.get("legal_count", 0),
                    candidates_at_resolution=len(result.residual),
                    decision_capsule=context.get("decision_capsule"),
                    infrastructure_resolved=infra_resolved,
                )

            # Pass residual to next scale
            candidates = result.residual

        # Fallback (should not reach here — Scale 5 always resolves)
        move = candidates[0].move if candidates else None
        if move is not None and move not in board.legal_moves:
            move = list(board.legal_moves)[0] if board.legal_moves else None

        return ProgressiveDecision(
            move=move,
            source=DecisionSource.SCALE_5_NEURAL,
            resolving_scale=5,
            scale_results=scale_results,
            nav_vector=context.get("nav_vector"),
            legal_move_count=context.get("legal_count", 0),
        )

    def _build_context(self, board: chess.Board) -> dict[str, Any]:
        """Build shared context computed ONCE for all scales."""
        from yami.navigator import compute_navigation_vector
        from yami.position_signals import extract_position_signals

        nav_vector = compute_navigation_vector(board)
        pos_signals = extract_position_signals(board)

        # Fill navigator signals into position signals
        pos_signals.nav_aggression = nav_vector.aggression
        pos_signals.nav_piece_domain = nav_vector.piece_domain
        pos_signals.nav_complexity = nav_vector.complexity
        pos_signals.nav_initiative = nav_vector.initiative
        pos_signals.nav_king_pressure = nav_vector.king_pressure
        pos_signals.nav_phase = nav_vector.phase

        # Opponent profile
        opp = self._opponent_profiler.profile
        pos_signals.opp_tactical_skill = opp.tactical_skill
        pos_signals.opp_aggressiveness = opp.aggressiveness
        pos_signals.opp_consistency = opp.consistency
        pos_signals.opp_pressure_rate = opp.pressure_rate
        pos_signals.opp_risk_tolerance = opp.risk_tolerance()

        return {
            "nav_vector": nav_vector,
            "pos_signals": pos_signals,
            "temporal": self._temporal,
            "gm_db": self._gm_db,
            "klines": self._klines,
            "learned_censors": self._learned_censors,
            "opponent_profiler": self._opponent_profiler,
            "fusion_model": self._fusion_model,
            "look_ahead_weight": opp.look_ahead_weight(),
        }
