"""Temporal Controller — 4-phase FSM for multi-move planning.

Tracks plan persistence across moves. The key missing ingredient:
instead of evaluating each position independently, the controller
maintains strategic continuity — "don't abandon a working attack
for a marginal positional improvement."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from yami.navigator import NavigationVector


class Phase(Enum):
    RECOGNITION = "recognition"
    PLANNING = "planning"
    EXECUTION = "execution"
    VERIFICATION = "verification"


@dataclass
class StrategicPlan:
    """A multi-move strategic plan."""

    name: str
    description: str
    expected_moves: int  # how many moves this plan typically takes
    # Navigation bias: which banks to boost
    nav_bias: dict[str, int] = field(default_factory=dict)
    # Activation conditions
    min_aggression: int = -1
    min_king_pressure: int = -1
    min_initiative: int = -1
    # Progress tracking
    progress_anchors: list[str] = field(default_factory=list)
    # Abandonment conditions
    abandon_if_behind_material: bool = False


# --- Plan Library ---

PLAN_LIBRARY: list[StrategicPlan] = [
    StrategicPlan(
        "kingside_attack",
        "Converge pieces on kingside, open files, sacrifice if needed",
        expected_moves=6,
        nav_bias={"aggression": 1, "king_pressure": 1, "complexity": 1},
        min_king_pressure=1,
        progress_anchors=["tempo-gain", "sacrifice", "open-file"],
    ),
    StrategicPlan(
        "queenside_attack",
        "Minority attack, open queenside files",
        expected_moves=5,
        nav_bias={"aggression": 1, "piece_domain": -1},
        progress_anchors=["pawn-break", "minority-attack", "open-file"],
    ),
    StrategicPlan(
        "central_breakthrough",
        "Pawn break in center, open lines for pieces",
        expected_moves=4,
        nav_bias={"initiative": 1, "piece_domain": -1},
        progress_anchors=["pawn-break", "center-control", "open-file"],
    ),
    StrategicPlan(
        "promote_pawn",
        "Advance passed pawn, support with pieces",
        expected_moves=8,
        nav_bias={"piece_domain": -1, "complexity": -1},
        progress_anchors=["passed-pawn", "pawn-race", "endgame-technique"],
    ),
    StrategicPlan(
        "simplify_to_endgame",
        "Trade pieces when ahead in material",
        expected_moves=4,
        nav_bias={"complexity": -1, "piece_domain": 1},
        progress_anchors=["simplification", "piece-exchange"],
        abandon_if_behind_material=True,
    ),
    StrategicPlan(
        "fortress",
        "Defensive setup when behind, aim for draw",
        expected_moves=6,
        nav_bias={"aggression": -1, "complexity": -1},
        progress_anchors=["fortress", "blockade", "king-safety"],
    ),
    StrategicPlan(
        "king_march",
        "Activate king in endgame",
        expected_moves=5,
        nav_bias={"piece_domain": 1, "complexity": -1},
        progress_anchors=["king-march", "king-activity", "opposition"],
    ),
    StrategicPlan(
        "piece_coordination",
        "Improve worst piece, control key squares",
        expected_moves=4,
        nav_bias={"initiative": 1, "piece_domain": 1},
        progress_anchors=["piece-coordination", "outpost", "connected-rooks"],
    ),
    StrategicPlan(
        "prophylaxis",
        "Prevent opponent's plan before it materializes",
        expected_moves=3,
        nav_bias={"aggression": -1, "initiative": 0},
        progress_anchors=["prophylaxis", "blockade"],
    ),
    StrategicPlan(
        "initiative_seize",
        "Gain tempo, force opponent to react",
        expected_moves=3,
        nav_bias={"initiative": 1, "aggression": 1, "complexity": -1},
        progress_anchors=["tempo-gain", "development"],
    ),
]

PLAN_BY_NAME = {p.name: p for p in PLAN_LIBRARY}


@dataclass
class TemporalState:
    """Current temporal state of the game."""

    phase: Phase = Phase.RECOGNITION
    active_plan: str | None = None
    plan_moves_remaining: int = 0
    plan_progress: float = 0.0
    moves_in_plan: int = 0
    prior_nav_vectors: list[tuple[int, ...]] = field(default_factory=list)
    escalation_level: int = 0  # 0=normal, 1=urgent, 2=desperate


class TemporalController:
    """4-phase FSM for multi-move planning."""

    def __init__(self, max_history: int = 10) -> None:
        self.state = TemporalState()
        self.max_history = max_history

    def reset(self) -> None:
        self.state = TemporalState()

    def update(
        self,
        nav_vector: NavigationVector,
        move_anchors: set[str],
        material_balance: int = 0,
    ) -> TemporalState:
        """Update temporal state after a move.

        Args:
            nav_vector: Current position's navigation vector.
            move_anchors: Anchors activated by the move just played.
            material_balance: Centipawns advantage (positive = ahead).
        """
        nv = nav_vector.as_tuple()
        self.state.prior_nav_vectors.append(nv)
        if len(self.state.prior_nav_vectors) > self.max_history:
            self.state.prior_nav_vectors.pop(0)

        if self.state.phase == Phase.RECOGNITION:
            self._do_recognition(nav_vector, material_balance)
        elif self.state.phase == Phase.PLANNING:
            self._do_planning(nav_vector, material_balance)
        elif self.state.phase == Phase.EXECUTION:
            self._do_execution(nav_vector, move_anchors, material_balance)
        elif self.state.phase == Phase.VERIFICATION:
            self._do_verification(nav_vector, move_anchors)

        return self.state

    def get_plan_bias(self) -> dict[str, float]:
        """Return scoring biases based on current plan.

        Returns dict of {bank_name: bias_value} to add to OTP scores.
        """
        if self.state.active_plan is None:
            return {}

        plan = PLAN_BY_NAME.get(self.state.active_plan)
        if plan is None:
            return {}

        # Bias strength scales with plan progress
        progress_mult = 0.5 + 0.5 * self.state.plan_progress
        bias = {}
        for bank, direction in plan.nav_bias.items():
            bias[bank] = direction * progress_mult * 2.0

        return bias

    def _do_recognition(
        self, nav: NavigationVector, material: int
    ) -> None:
        """Assess position type and transition to planning."""
        # Select best plan based on current navigation vector
        best_plan = None
        best_score = -1.0

        for plan in PLAN_LIBRARY:
            score = self._plan_activation_score(plan, nav, material)
            if score > best_score:
                best_score = score
                best_plan = plan

        if best_plan and best_score > 1.0:
            self.state.active_plan = best_plan.name
            self.state.plan_moves_remaining = best_plan.expected_moves
            self.state.plan_progress = 0.0
            self.state.moves_in_plan = 0
            self.state.phase = Phase.EXECUTION
        else:
            # No strong plan — stay in recognition
            self.state.active_plan = None

    def _do_planning(
        self, nav: NavigationVector, material: int
    ) -> None:
        """Replan after verification failure."""
        # Reset and re-recognize
        self.state.active_plan = None
        self.state.phase = Phase.RECOGNITION
        self._do_recognition(nav, material)

    def _do_execution(
        self,
        nav: NavigationVector,
        move_anchors: set[str],
        material: int,
    ) -> None:
        """Execute the current plan, tracking progress."""
        plan = PLAN_BY_NAME.get(self.state.active_plan or "")
        if plan is None:
            self.state.phase = Phase.RECOGNITION
            return

        self.state.moves_in_plan += 1
        self.state.plan_moves_remaining -= 1

        # Track progress via anchor matches
        progress_hits = sum(
            1 for a in plan.progress_anchors if a in move_anchors
        )
        if plan.progress_anchors:
            self.state.plan_progress = min(
                1.0,
                self.state.plan_progress
                + progress_hits / max(len(plan.progress_anchors), 1) * 0.3,
            )

        # Check abandonment conditions
        if plan.abandon_if_behind_material and material < -200:
            self.state.phase = Phase.VERIFICATION
            return

        # Check if plan is complete or expired
        if self.state.plan_moves_remaining <= 0:
            self.state.phase = Phase.VERIFICATION

    def _do_verification(
        self, nav: NavigationVector, move_anchors: set[str]
    ) -> None:
        """Verify plan outcome and decide: continue, modify, or replan."""
        if self.state.plan_progress >= 0.6:
            # Plan succeeded enough — check if we should continue
            plan = PLAN_BY_NAME.get(self.state.active_plan or "")
            if plan:
                self.state.plan_moves_remaining = plan.expected_moves // 2
                self.state.phase = Phase.EXECUTION
                return

        # Plan failed or completed — replan
        self.state.phase = Phase.PLANNING
        self._do_planning(nav, 0)

    def _plan_activation_score(
        self, plan: StrategicPlan, nav: NavigationVector, material: int
    ) -> float:
        """Score how well a plan fits the current position."""
        score = 0.0
        nv = nav.as_tuple()
        banks = ["aggression", "piece_domain", "complexity",
                 "initiative", "king_pressure", "phase"]

        for i, bank_name in enumerate(banks):
            if bank_name in plan.nav_bias:
                desired = plan.nav_bias[bank_name]
                actual = nv[i]
                if desired == actual:
                    score += 2.0
                elif desired == 0 or actual == 0:
                    score += 0.5
                elif desired == -actual:
                    score -= 1.0

        # Material-dependent plans
        if plan.name == "simplify_to_endgame" and material > 200:
            score += 3.0
        if plan.name == "fortress" and material < -200:
            score += 3.0
        if plan.name == "promote_pawn" and nav.phase == -1:
            score += 2.0

        return score
