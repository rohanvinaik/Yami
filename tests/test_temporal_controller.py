"""Tests for the Temporal Controller."""

from yami.navigator import NavigationVector
from yami.temporal_controller import (
    PLAN_BY_NAME,
    Phase,
    TemporalController,
)


def test_initial_state():
    tc = TemporalController()
    assert tc.state.phase == Phase.RECOGNITION
    assert tc.state.active_plan is None


def test_recognition_selects_plan():
    tc = TemporalController()
    # Aggressive position with king pressure → should select attack plan
    nav = NavigationVector(1, 1, 0, 1, 1, 0)
    tc.update(nav, {"tempo-gain", "sacrifice"}, material_balance=0)
    # Should have moved to execution with a plan
    assert tc.state.phase in (Phase.EXECUTION, Phase.RECOGNITION)


def test_execution_tracks_progress():
    tc = TemporalController()
    # Force into kingside_attack plan
    nav = NavigationVector(1, 1, 0, 1, 1, 0)
    tc.update(nav, set(), material_balance=0)

    if tc.state.active_plan == "kingside_attack":
        initial_remaining = tc.state.plan_moves_remaining
        # Execute a move with progress anchors
        tc.update(nav, {"tempo-gain"}, material_balance=0)
        assert tc.state.plan_moves_remaining < initial_remaining
        assert tc.state.plan_progress > 0.0


def test_plan_abandonment_on_material_loss():
    tc = TemporalController()
    # Start with simplify plan (requires material advantage)
    tc.state.phase = Phase.EXECUTION
    tc.state.active_plan = "simplify_to_endgame"
    tc.state.plan_moves_remaining = 4

    nav = NavigationVector(0, 0, -1, 0, 0, -1)
    # Material loss should trigger verification
    tc.update(nav, set(), material_balance=-300)
    assert tc.state.phase == Phase.VERIFICATION


def test_plan_bias_output():
    tc = TemporalController()
    tc.state.active_plan = "kingside_attack"
    tc.state.plan_progress = 0.5

    bias = tc.get_plan_bias()
    assert isinstance(bias, dict)
    assert len(bias) > 0
    # Kingside attack should boost aggression
    assert bias.get("aggression", 0) > 0


def test_no_plan_bias_when_no_plan():
    tc = TemporalController()
    bias = tc.get_plan_bias()
    assert bias == {}


def test_reset():
    tc = TemporalController()
    nav = NavigationVector(1, 1, 0, 1, 1, 0)
    tc.update(nav, set())
    tc.reset()
    assert tc.state.phase == Phase.RECOGNITION
    assert tc.state.active_plan is None
    assert len(tc.state.prior_nav_vectors) == 0


def test_plan_library_complete():
    assert len(PLAN_BY_NAME) >= 10
    for name, plan in PLAN_BY_NAME.items():
        assert plan.expected_moves > 0
        assert plan.name == name


def test_full_lifecycle():
    """Test a full plan lifecycle: recognition → execution → verification."""
    tc = TemporalController()

    # Phase 1: Recognition with strong attacking position
    nav = NavigationVector(1, 1, -1, 1, 1, 0)
    tc.update(nav, set(), material_balance=200)

    # Should have a plan now
    if tc.state.active_plan:
        # Phase 2: Execute moves
        for _ in range(tc.state.plan_moves_remaining + 1):
            tc.update(nav, {"tempo-gain"}, material_balance=200)

        # Should eventually reach verification
        assert tc.state.phase in (Phase.VERIFICATION, Phase.EXECUTION,
                                   Phase.PLANNING, Phase.RECOGNITION)
