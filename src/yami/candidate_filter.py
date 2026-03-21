"""Layer 6: Candidate Filtering + Annotation.

Reduce to 3-5 candidates. Annotate each with full context for the LLM.
The structured residual output — the LLM receives recognition-ready context.
"""

from __future__ import annotations

import chess

from yami.knowledge_graph import evaluate_position, rank_moves, suggest_plan
from yami.models import (
    AnnotatedCandidate,
    PlanTemplate,
    PositionalProfile,
    RankedMove,
    ScopedMove,
)
from yami.tactical_scoper import PIECE_VALUES


def filter_and_annotate(
    board: chess.Board,
    moves: list[ScopedMove],
    max_candidates: int = 5,
) -> tuple[list[AnnotatedCandidate], PlanTemplate, PositionalProfile]:
    """Produce annotated candidates for the LLM.

    Returns (candidates, active_plan, positional_profile).
    """
    profile = evaluate_position(board)
    plan = suggest_plan(profile)

    ranked = rank_moves(moves, plan, board)
    top = ranked[:max_candidates]

    candidates = [_annotate(board, rm, plan, profile) for rm in top]
    return candidates, plan, profile


def _annotate(
    board: chess.Board,
    ranked: RankedMove,
    plan: PlanTemplate,
    profile: PositionalProfile,
) -> AnnotatedCandidate:
    """Build a fully annotated candidate from a ranked move."""
    sm = ranked.scoped_move
    san = board.san(sm.move)
    pos_eval = _eval_after_move(board, sm)
    risk = _assess_risk(board, sm)
    narrative = _generate_narrative(san, sm, plan, profile, pos_eval, risk)

    return AnnotatedCandidate(
        move=sm.move,
        san=san,
        tactical_motifs=sm.motifs,
        plan_alignment=ranked.alignment,
        plan_name=plan.name,
        positional_eval=pos_eval,
        see_value=sm.see_value,
        risk=risk,
        narrative=narrative,
    )


def _eval_after_move(board: chess.Board, move: ScopedMove) -> float:
    """Simple positional evaluation after the move in centipawns."""
    board.push(move.move)
    # Material count from our perspective (we just moved, so now it's opponent's turn)
    our_color = not board.turn
    our_material = sum(
        PIECE_VALUES.get(p.piece_type, 0)
        for p in board.piece_map().values()
        if p.color == our_color
    )
    opp_material = sum(
        PIECE_VALUES.get(p.piece_type, 0)
        for p in board.piece_map().values()
        if p.color != our_color
    )
    board.pop()
    return (our_material - opp_material) / 100.0


def _assess_risk(board: chess.Board, move: ScopedMove) -> str:
    """Assess the risk level of a move."""
    risks: list[str] = []

    if "hangs_piece" in move.motifs:
        return "critical"

    board.push(move.move)
    # Check if opponent has forcing responses
    checks = 0
    captures = 0
    for opp_move in board.legal_moves:
        if board.is_capture(opp_move):
            captures += 1
        board.push(opp_move)
        if board.is_check():
            checks += 1
        board.pop()
    board.pop()

    if checks >= 2:
        risks.append("opponent has multiple checks")
    if captures >= 5:
        risks.append("many captures available")

    if move.see_value < -200:
        return "high"
    if risks:
        return "medium"
    return "low"


def _generate_narrative(
    san: str,
    move: ScopedMove,
    plan: PlanTemplate,
    profile: PositionalProfile,
    pos_eval: float,
    risk: str,
) -> str:
    """Generate a natural-language narrative for the candidate."""
    parts: list[str] = []

    # Describe what the move does
    if move.motifs:
        motif_str = ", ".join(m.replace("_", " ") for m in move.motifs if m != "hangs_piece")
        if motif_str:
            parts.append(f"{san} ({motif_str})")
        else:
            parts.append(san)
    else:
        parts.append(f"{san} — quiet move")

    # Plan alignment
    parts.append(f"Aligns with plan: {plan.name}.")

    # Positional assessment
    if pos_eval > 1.0:
        parts.append(f"Maintains advantage ({pos_eval:+.1f}).")
    elif pos_eval < -1.0:
        parts.append(f"Behind in material ({pos_eval:+.1f}).")
    else:
        parts.append("Roughly equal position.")

    # Risk
    parts.append(f"Risk: {risk}.")

    return " ".join(parts)


def format_candidates_for_llm(
    candidates: list[AnnotatedCandidate],
    plan: PlanTemplate,
    profile: PositionalProfile,
) -> str:
    """Format annotated candidates into the LLM decision prompt context."""
    lines = [
        f"Current plan: {plan.name} — {plan.description}",
        f"Position: material={profile.material.value}, structure={profile.structure.value}, "
        f"activity={profile.activity.value}, safety={profile.safety.value}, "
        f"tempo={profile.tempo.value}",
        "",
        "Candidates:",
    ]

    for i, c in enumerate(candidates, 1):
        motifs = ", ".join(c.tactical_motifs) if c.tactical_motifs else "none"
        lines.append(f"\n{i}. {c.san}")
        lines.append(f"   Tactical: {motifs}")
        lines.append(f"   Plan alignment: {c.plan_alignment:.1f}")
        lines.append(f"   Evaluation: {c.positional_eval:+.1f}")
        lines.append(f"   Risk: {c.risk}")
        lines.append(f"   {c.narrative}")

    return "\n".join(lines)
