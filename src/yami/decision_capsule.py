"""Decision Capsule — typed boundary between infrastructure and neural reasoning.

Inspired by Wayfinder's Dr. Ducky architecture: the expensive model should never
start from the raw position. It should start from the compiled decision state.

Three Wayfinder-derived enhancements:

1. **Agent Suppression** (← Dr. Ducky's bank filtering):
   Position-structure-gated activation. If a position's structure makes an agent
   domain inert (e.g., no king safety issues → suppress attack/defense agents),
   don't run that agent. Reduces noise and computation.

2. **Decision Residual Packets** (← Wayfinder's structured failure):
   Classify each decision as infrastructure-resolved vs genuine-residual.
   When coherence clearly picks a winner, skip neural. When it doesn't,
   emit a typed residual describing exactly what's ambiguous.

3. **Capsule Boundary** (← Dr. Ducky's projector):
   The DecisionCapsule is the ONLY thing the neural model sees.
   Internal symbolic state never crosses this boundary.

Move quality uses Chess.com-style centipawn-loss classification:
   brilliant / great / best / good / inaccuracy / mistake / blunder
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import chess

from yami.position_signals import PositionSignals


# ---------------------------------------------------------------------------
# Move quality classification (Chess.com style)
# ---------------------------------------------------------------------------

class MoveQuality(Enum):
    """Chess.com-style move quality based on centipawn loss vs best move.

    Centipawn loss thresholds adapted from Chess.com's classification:
    https://support.chess.com/en/articles/8572705
    """

    BRILLIANT = "brilliant"    # Best move + involves sacrifice or deep tactic
    GREAT = "great"            # cp_loss == 0, non-obvious (not recapture, not forced)
    BEST = "best"              # cp_loss == 0, the engine's top choice
    GOOD = "good"              # cp_loss <= 30 — reasonable, slight imprecision
    INACCURACY = "inaccuracy"  # 30 < cp_loss <= 100
    MISTAKE = "mistake"        # 100 < cp_loss <= 300
    BLUNDER = "blunder"        # cp_loss > 300


def classify_move_quality(
    cp_loss: int,
    is_best: bool = False,
    is_sacrifice: bool = False,
    is_only_move: bool = False,
    is_forced_recapture: bool = False,
) -> MoveQuality:
    """Classify a move's quality from its centipawn loss and context.

    Args:
        cp_loss: Centipawn loss relative to engine's best move (always >= 0).
        is_best: Whether this was the engine's #1 choice.
        is_sacrifice: Whether the move involves material sacrifice with compensation.
        is_only_move: Whether there was only one reasonable move.
        is_forced_recapture: Whether this is an obvious recapture.
    """
    if cp_loss > 300:
        return MoveQuality.BLUNDER
    if cp_loss > 100:
        return MoveQuality.MISTAKE
    if cp_loss > 30:
        return MoveQuality.INACCURACY
    # cp_loss <= 30
    if is_best and is_sacrifice:
        return MoveQuality.BRILLIANT
    if is_best and not is_forced_recapture and not is_only_move:
        return MoveQuality.GREAT
    if is_best:
        return MoveQuality.BEST
    return MoveQuality.GOOD


# Numeric encoding for training: higher = better move
QUALITY_SCORE = {
    MoveQuality.BRILLIANT: 1.0,
    MoveQuality.GREAT: 0.9,
    MoveQuality.BEST: 0.85,
    MoveQuality.GOOD: 0.7,
    MoveQuality.INACCURACY: 0.4,
    MoveQuality.MISTAKE: 0.15,
    MoveQuality.BLUNDER: 0.0,
}


# ---------------------------------------------------------------------------
# Agent suppression — position-structure-gated activation
# ---------------------------------------------------------------------------

# Agent domain → position signal conditions that make the agent inert.
# Each condition is a callable: (PositionSignals) → bool (True = suppress).
AGENT_SUPPRESSION_RULES: dict[str, list[tuple[str, Any]]] = {
    "tactical": [
        # Suppress in dead-equal endgames with no material tension
        ("material_balance_zero_and_endgame", None),
    ],
    "positional": [
        # Positional agent is always relevant — never suppress
    ],
    "endgame": [
        # Suppress in opening/early middlegame (game_phase > 0.6)
        ("early_game", None),
    ],
    "attack": [
        # Suppress when opponent king is castled with full pawn shelter
        # and we have no attacking pieces near their king
        ("no_king_attack_surface", None),
    ],
    "defense": [
        # Suppress when our king is safe and opponent has no attacking pieces
        ("no_king_threat", None),
    ],
    "initiative": [
        # Initiative agent is always relevant — never suppress
    ],
}


def compute_active_agents(
    pos_signals: PositionSignals,
) -> dict[str, bool]:
    """Determine which SoM agents should be active for this position.

    Returns a dict of agent_name → is_active. Suppressed agents are False.
    This is the chess equivalent of Dr. Ducky's bank filtering:
    "no existential structure → don't spend budget on witness construction."
    """
    agents = {
        "tactical": True,
        "positional": True,
        "endgame": True,
        "attack": True,
        "defense": True,
        "initiative": True,
    }

    # Game phase: vec index depends on PositionSignals field order.
    # We use the named attributes directly for clarity.
    game_phase = getattr(pos_signals, "game_phase", 0.5)
    material_balance = getattr(pos_signals, "material_balance", 0.0)
    king_safety_ours = getattr(pos_signals, "king_safety_ours", 0.0)
    king_safety_theirs = getattr(pos_signals, "king_safety_theirs", 0.0)
    nav_aggression = getattr(pos_signals, "nav_aggression", 0)
    nav_king_pressure = getattr(pos_signals, "nav_king_pressure", 0)
    pawn_structure_tension = getattr(pos_signals, "pawn_structure_tension", 0.0)

    # Suppress tactical agent in dead-equal endgames
    if (game_phase < -0.3
            and abs(material_balance) < 0.1
            and pawn_structure_tension < 0.1):
        agents["tactical"] = False

    # Suppress endgame agent in opening/early middlegame
    if game_phase > 0.5:
        agents["endgame"] = False

    # Suppress attack agent when no king attack surface
    if (king_safety_theirs < 0.3
            and nav_aggression <= 0
            and nav_king_pressure <= 0):
        agents["attack"] = False

    # Suppress defense agent when our king is safe
    if (king_safety_ours < 0.3
            and nav_king_pressure >= 0):
        agents["defense"] = False

    return agents


# ---------------------------------------------------------------------------
# Decision classification — infrastructure-resolved vs genuine residual
# ---------------------------------------------------------------------------

class DecisionType(Enum):
    """Whether the infrastructure can resolve this decision alone."""

    INFRASTRUCTURE_RESOLVED = "infrastructure_resolved"
    GENUINE_RESIDUAL = "genuine_residual"


def classify_decision(
    coherence_gap: float,
    signal_agreement: float,
    constructive_count: int,
    top_score: float,
    candidate_count: int,
) -> DecisionType:
    """Classify whether the position needs neural reasoning.

    Infrastructure-resolved: coherence clearly picks a winner.
    Genuine residual: ambiguous, needs the neural model.

    Args:
        coherence_gap: Score gap between 1st and 2nd candidate (normalized).
        signal_agreement: Fraction of ternary signals supporting top candidate.
        constructive_count: Number of moves with 3+ agreeing signals.
        top_score: Top candidate's coherence score.
        candidate_count: Number of viable candidates remaining.
    """
    # Strong agreement + clear gap → infrastructure handles it
    if coherence_gap > 0.35 and signal_agreement > 0.6:
        return DecisionType.INFRASTRUCTURE_RESOLVED

    # Single constructive candidate with high score
    if constructive_count == 1 and top_score > 8.0:
        return DecisionType.INFRASTRUCTURE_RESOLVED

    # Only 2 candidates and one is clearly better
    if candidate_count <= 2 and coherence_gap > 0.5:
        return DecisionType.INFRASTRUCTURE_RESOLVED

    return DecisionType.GENUINE_RESIDUAL


# ---------------------------------------------------------------------------
# The Decision Capsule — the projector boundary
# ---------------------------------------------------------------------------

@dataclass
class CandidateCapsule:
    """Per-candidate data inside the decision capsule."""

    move_uci: str
    # Chess.com-style quality (set during labeling, not inference)
    quality: MoveQuality | None = None
    quality_score: float = 0.0

    # Coherence signals (from infrastructure)
    coherence_score: float = 0.0
    interference: float = 0.0
    ternary_navigator: int = 0
    ternary_strategy: int = 0
    ternary_temporal: int = 0
    ternary_kline: int = 0
    ternary_gm: int = 0

    # SoM agent scores (only from active agents)
    agent_scores: dict[str, float] = field(default_factory=dict)

    # Tactical context
    is_capture: bool = False
    is_check: bool = False
    is_sacrifice: bool = False
    see_value: float = 0.0
    tactical_motifs: list[str] = field(default_factory=list)

    # Geometry
    target_centrality: float = 0.0
    dist_to_opp_king: float = 0.0

    def to_vector(self) -> list[float]:
        """Flatten to numeric vector for model input."""
        vec = [
            self.coherence_score,
            self.interference,
            float(self.ternary_navigator),
            float(self.ternary_strategy),
            float(self.ternary_temporal),
            float(self.ternary_kline),
            float(self.ternary_gm),
            float(self.is_capture),
            float(self.is_check),
            float(self.is_sacrifice),
            self.see_value,
            self.target_centrality,
            self.dist_to_opp_king,
        ]
        # Agent scores (6 slots, 0.0 for suppressed agents)
        for agent in ["tactical", "positional", "endgame",
                       "attack", "defense", "initiative"]:
            vec.append(self.agent_scores.get(agent, 0.0))
        return vec

    @staticmethod
    def vector_dim() -> int:
        return 19  # 13 base + 6 agent scores


@dataclass
class DecisionCapsule:
    """The typed boundary object between infrastructure and neural model.

    This is the chess equivalent of Dr. Ducky's GoalCapsule:
    - Contains the position type and active agents (after suppression)
    - Contains per-candidate agent scores and ambiguity dimensions
    - Contains the specific ambiguity the neural model needs to resolve
    - Is the ONLY thing the neural model sees

    The capsule is the projector boundary. Internal symbolic state
    (raw board, move generation, censor internals) never crosses it.
    """

    # Position context
    game_phase: float = 0.0  # -1=endgame, +1=opening
    material_balance: float = 0.0

    # Navigation context (6-bank ternary)
    nav_aggression: int = 0
    nav_piece_domain: int = 0
    nav_complexity: int = 0
    nav_initiative: int = 0
    nav_king_pressure: int = 0
    nav_phase: int = 0

    # Agent activation state (after suppression)
    active_agents: dict[str, bool] = field(default_factory=dict)
    suppressed_agent_count: int = 0

    # Decision classification
    decision_type: DecisionType = DecisionType.GENUINE_RESIDUAL
    coherence_gap: float = 0.0  # score gap between #1 and #2
    signal_agreement: float = 0.0  # fraction of signals supporting top

    # The ambiguity description — what exactly is the model being asked?
    ambiguity_dimension: str = ""  # e.g., "tactical_vs_positional"
    swing_agent: str = ""  # which agent's opinion would break the tie

    # Candidates (the capsule contents)
    candidates: list[CandidateCapsule] = field(default_factory=list)

    # Difficulty estimate
    difficulty: float = 0.0  # 0=trivial, 1=maximally ambiguous

    def to_vector(self) -> list[float]:
        """Flatten position context to numeric vector."""
        return [
            self.game_phase,
            self.material_balance,
            float(self.nav_aggression),
            float(self.nav_piece_domain),
            float(self.nav_complexity),
            float(self.nav_initiative),
            float(self.nav_king_pressure),
            float(self.nav_phase),
            float(self.suppressed_agent_count),
            self.coherence_gap,
            self.signal_agreement,
            self.difficulty,
        ]

    @staticmethod
    def position_vector_dim() -> int:
        return 12


def build_decision_capsule(
    board: chess.Board,
    pos_signals: PositionSignals,
    scored_moves: list[Any],  # list of MoveSignals from coherence
    active_agents: dict[str, bool],
    coherence_gap: float,
    signal_agreement: float,
    constructive_count: int,
) -> DecisionCapsule:
    """Build a DecisionCapsule from infrastructure outputs.

    This is the projector function: it takes internal state and produces
    the typed boundary object that the neural model will consume.
    """
    decision_type = classify_decision(
        coherence_gap=coherence_gap,
        signal_agreement=signal_agreement,
        constructive_count=constructive_count,
        top_score=scored_moves[0].final_score if scored_moves else 0.0,
        candidate_count=len(scored_moves),
    )

    # Build candidate capsules
    cand_capsules = []
    for sig in scored_moves[:5]:
        cc = CandidateCapsule(
            move_uci=sig.move.uci(),
            coherence_score=sig.final_score,
            interference=sig.interference,
            ternary_navigator=sig.ternary_navigator,
            ternary_strategy=sig.ternary_strategy,
            ternary_temporal=sig.ternary_temporal,
            ternary_kline=sig.ternary_kline,
            ternary_gm=sig.ternary_gm,
            is_capture=board.is_capture(sig.move),
            is_check=board.gives_check(sig.move),
            see_value=0.0,
            tactical_motifs=[],
        )

        # Agent scores (only for active agents)
        for agent_name, is_active in active_agents.items():
            if is_active:
                cc.agent_scores[agent_name] = sig.temporal_scores.get(agent_name, 0.0)
            # Suppressed agents get 0.0 (default)

        cand_capsules.append(cc)

    # Identify ambiguity dimension
    ambiguity_dim, swing = _identify_ambiguity(scored_moves, active_agents)

    suppressed = sum(1 for v in active_agents.values() if not v)

    capsule = DecisionCapsule(
        game_phase=getattr(pos_signals, "game_phase", 0.0),
        material_balance=getattr(pos_signals, "material_balance", 0.0),
        nav_aggression=getattr(pos_signals, "nav_aggression", 0),
        nav_piece_domain=getattr(pos_signals, "nav_piece_domain", 0),
        nav_complexity=getattr(pos_signals, "nav_complexity", 0),
        nav_initiative=getattr(pos_signals, "nav_initiative", 0),
        nav_king_pressure=getattr(pos_signals, "nav_king_pressure", 0),
        nav_phase=getattr(pos_signals, "nav_phase", 0),
        active_agents=active_agents,
        suppressed_agent_count=suppressed,
        decision_type=decision_type,
        coherence_gap=coherence_gap,
        signal_agreement=signal_agreement,
        ambiguity_dimension=ambiguity_dim,
        swing_agent=swing,
        candidates=cand_capsules,
        difficulty=1.0 - signal_agreement if signal_agreement > 0 else 1.0,
    )

    return capsule


def _identify_ambiguity(
    scored_moves: list[Any],
    active_agents: dict[str, bool],
) -> tuple[str, str]:
    """Identify what dimension creates the ambiguity between top candidates.

    Returns (ambiguity_dimension, swing_agent).
    """
    if len(scored_moves) < 2:
        return ("no_ambiguity", "")

    top = scored_moves[0]
    second = scored_moves[1]

    # Which ternary signals disagree between top two?
    disagreements = []
    for name, get_t in [
        ("navigator", lambda s: s.ternary_navigator),
        ("strategy", lambda s: s.ternary_strategy),
        ("temporal", lambda s: s.ternary_temporal),
        ("kline", lambda s: s.ternary_kline),
        ("gm", lambda s: s.ternary_gm),
    ]:
        if get_t(top) != get_t(second):
            disagreements.append(name)

    if not disagreements:
        return ("score_margin", "")

    # The ambiguity dimension is the signal that most differentiates
    # Example: navigator agrees with top, strategy agrees with second
    #   → "navigator_vs_strategy"
    if len(disagreements) >= 2:
        dim = f"{disagreements[0]}_vs_{disagreements[1]}"
    else:
        dim = f"{disagreements[0]}_split"

    # Swing agent: which active agent's score differs most?
    swing = ""
    max_diff = 0.0
    for agent, is_active in active_agents.items():
        if not is_active:
            continue
        top_score = top.temporal_scores.get(agent, 0.0)
        second_score = second.temporal_scores.get(agent, 0.0)
        diff = abs(top_score - second_score)
        if diff > max_diff:
            max_diff = diff
            swing = agent

    return (dim, swing)
