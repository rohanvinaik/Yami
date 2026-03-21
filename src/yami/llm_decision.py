"""Layer 7: LLM Decision — the residual.

The LLM receives 3-5 annotated candidates and chooses the best one.
This is recognition, not search. Constrained hallucination applied to chess.
"""

from __future__ import annotations

import os

import chess

from yami.candidate_filter import format_candidates_for_llm
from yami.models import AnnotatedCandidate, PlanTemplate, PositionalProfile

DECISION_PROMPT = """\
You are playing chess as {color}. \
The position has been analyzed by the infrastructure layer.

Current position (FEN): {fen}

{candidate_context}

Choose the move that best serves the long-term plan. Consider:
- Does this move improve your position or just maintain it?
- Does this move create problems for your opponent?
- Is this move consistent with the plan, or does it require a plan change?
- What is the risk/reward tradeoff?

Respond with ONLY the move in standard algebraic notation (e.g., Nf5, e4, Qxd7).
Do not include move numbers, annotations, or explanations."""


async def choose_move(
    board: chess.Board,
    candidates: list[AnnotatedCandidate],
    plan: PlanTemplate,
    profile: PositionalProfile,
) -> chess.Move | None:
    """Ask the LLM to choose among annotated candidates.

    Returns the chosen move, or None if the LLM fails to produce a valid move.
    """
    try:
        import anthropic
    except ImportError:
        return _fallback_choice(candidates)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _fallback_choice(candidates)

    color = "White" if board.turn == chess.WHITE else "Black"
    context = format_candidates_for_llm(candidates, plan, profile)
    prompt = DECISION_PROMPT.format(
        color=color,
        fen=board.fen(),
        candidate_context=context,
    )

    client = anthropic.AsyncAnthropic(api_key=api_key)
    message = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text.strip()
    return _parse_llm_response(board, response_text, candidates)


def _parse_llm_response(
    board: chess.Board,
    response: str,
    candidates: list[AnnotatedCandidate],
) -> chess.Move | None:
    """Parse the LLM's response into a legal move.

    Tries exact SAN match first, then fuzzy matching against candidates.
    Falls back to the top candidate if parsing fails.
    """
    # Clean response — strip punctuation, move numbers
    cleaned = response.strip().rstrip(".!?+#")
    # Remove move number prefix like "1." or "15..."
    parts = cleaned.split()
    if parts:
        candidate_str = parts[-1].rstrip(".!?+#")
    else:
        return _fallback_choice(candidates)

    # Try direct SAN parse
    try:
        move = board.parse_san(candidate_str)
        if move in board.legal_moves:
            return move
    except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
        pass

    # Try matching against candidate SANs
    for c in candidates:
        if candidate_str.lower() == c.san.lower():
            return c.move
        # Fuzzy: strip check/mate symbols
        clean_san = c.san.rstrip("+#")
        if candidate_str.rstrip("+#").lower() == clean_san.lower():
            return c.move

    # Fallback to top candidate
    return _fallback_choice(candidates)


def _fallback_choice(candidates: list[AnnotatedCandidate]) -> chess.Move | None:
    """Fall back to the highest-ranked candidate."""
    if not candidates:
        return None
    # Candidates are already sorted by plan alignment
    return candidates[0].move


def choose_move_sync(
    board: chess.Board,
    candidates: list[AnnotatedCandidate],
    plan: PlanTemplate,
    profile: PositionalProfile,
) -> chess.Move | None:
    """Synchronous wrapper for choose_move."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already in an async context — use fallback
        return _fallback_choice(candidates)

    return asyncio.run(choose_move(board, candidates, plan, profile))
