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
    # Society of Mind: specialist agent scores per move
    agent_scores: dict[str, float] = field(default_factory=dict)
    # Trajectory convergence: history of agent agreement
    convergence_history: list[float] = field(default_factory=list)


class TemporalController:
    """Society of Mind temporal controller with specialist agents.

    Six specialist agents each score moves from their domain. The arbiter
    tracks trajectory convergence — when agents agree across moves, that
    convergence IS the confidence signal.
    """

    AGENT_NAMES = [
        "tactical", "positional", "endgame", "attack", "defense", "initiative",
    ]

    def __init__(self, max_history: int = 10) -> None:
        self.state = TemporalState()
        self.max_history = max_history

    def reset(self) -> None:
        self.state = TemporalState()

    def score_move_by_agents(
        self,
        nav_vector: NavigationVector,
        move_anchors: set[str],
        material_balance: int = 0,
    ) -> dict[str, float]:
        """Score a position/move through all specialist agents.

        Returns dict of {agent_name: score} where positive = agent supports,
        negative = agent opposes, zero = agent is orthogonal (OTP).
        """
        nv = nav_vector.as_tuple()
        scores: dict[str, float] = {}

        # Tactical agent: forcing moves, captures, checks
        tactical = 0.0
        if "tempo-gain" in move_anchors:
            tactical += 2.0
        if "fork" in move_anchors or "pin" in move_anchors:
            tactical += 3.0
        if "sacrifice" in move_anchors:
            tactical += 2.5
        if nv[2] == -1:  # COMPLEXITY = forcing
            tactical += 1.0
        scores["tactical"] = tactical

        # Positional agent: structure, coordination, outposts
        positional = 0.0
        if "outpost" in move_anchors or "knight-outpost" in move_anchors:
            positional += 2.0
        if "piece-coordination" in move_anchors:
            positional += 1.5
        if "open-file" in move_anchors or "rook-on-open-file" in move_anchors:
            positional += 1.5
        if "connected-rooks" in move_anchors:
            positional += 1.0
        if nv[1] == 1:  # PIECE_DOMAIN = major piece activity
            positional += 1.0
        scores["positional"] = positional

        # Endgame agent: king activity, passed pawns, technique
        endgame = 0.0
        if nv[5] == -1:  # PHASE = endgame
            if "king-march" in move_anchors or "king-activity" in move_anchors:
                endgame += 3.0
            if "passed-pawn" in move_anchors:
                endgame += 3.0
            if "endgame-technique" in move_anchors:
                endgame += 2.0
            if "opposition" in move_anchors:
                endgame += 2.5
        scores["endgame"] = endgame

        # Attack agent: king pressure, sacrifices, convergence
        attack = 0.0
        if nv[4] == 1:  # KING_PRESSURE = targeting opponent
            attack += 2.0
        if nv[0] == 1:  # AGGRESSION = attacking
            attack += 1.5
        if "sacrifice" in move_anchors:
            attack += 2.0
        if "back-rank-threat" in move_anchors:
            attack += 3.0
        if "king-hunt" in move_anchors:
            attack += 3.0
        scores["attack"] = attack

        # Defense agent: king safety, fortress, prophylaxis
        defense = 0.0
        if nv[4] == -1:  # KING_PRESSURE = own king danger
            defense += 3.0
        if "king-safety" in move_anchors or "castling" in move_anchors:
            defense += 2.0
        if "fortress" in move_anchors or "blockade" in move_anchors:
            defense += 2.0
        if "prophylaxis" in move_anchors:
            defense += 1.5
        if material_balance < -200:
            defense += 1.0
        scores["defense"] = defense

        # Initiative agent: tempo, development, space
        initiative = 0.0
        if nv[3] == 1:  # INITIATIVE = dictating
            initiative += 2.0
        if "tempo-gain" in move_anchors:
            initiative += 2.5
        if "development" in move_anchors:
            initiative += 2.0
        if "center-control" in move_anchors:
            initiative += 1.5
        if "space-advantage" in move_anchors:
            initiative += 1.0
        scores["initiative"] = initiative

        return scores

    def compute_convergence(self, agent_scores: dict[str, float]) -> float:
        """Compute convergence between current and previous agent scores.

        High convergence = agents have been agreeing across moves.
        Low convergence = agents are contradicting or changing their minds.
        """
        self.state.agent_scores = agent_scores

        if not self.state.convergence_history:
            # First move — no history to compare
            self.state.convergence_history.append(0.5)
            return 0.5

        # Get non-zero agent scores (agents that have opinions)
        active = {k: v for k, v in agent_scores.items() if v > 0.5}

        if len(active) < 2:
            convergence = 0.3  # too few signals
        else:
            values = list(active.values())
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            # Low variance among active agents = high convergence
            convergence = max(0.0, 1.0 - variance / max(mean * mean, 0.01))

        self.state.convergence_history.append(convergence)
        if len(self.state.convergence_history) > self.max_history:
            self.state.convergence_history.pop(0)

        return convergence

    def get_trajectory_trend(self) -> float:
        """Get the trend of convergence over recent moves.

        Positive = convergence increasing (plan is working).
        Negative = convergence decreasing (plan is failing).
        Zero = stable.
        """
        hist = self.state.convergence_history
        if len(hist) < 3:
            return 0.0

        recent = hist[-3:]
        trend = recent[-1] - recent[0]
        return trend

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
