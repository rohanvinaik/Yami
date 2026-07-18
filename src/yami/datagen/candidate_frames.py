"""U1 — the Scale-0 → frame-layer bridge.

The renderer (`game_renderer`) is deliberately pure: it turns a *given* move into event-sentences and
makes zero chess judgments of its own. That purity is load-bearing, so the renderer must NOT know about
engines or censors. This module is the seam that keeps it pure while grounding the frame layer:

    ~30 legal moves ──(Scale-0: legal + censor stack)──▶ ~3-5 survivors ──(render_move)──▶ annotated frames

Every frame the router (U6) ever sees is therefore legal-and-not-a-blunder *by construction* — the router
gets a recognition problem over a handful of annotated candidates, not a search over the raw legal set.

The censor stack is the ENGINE's stack (blunder/tactical/repetition + learned negative-learning censors),
reused verbatim via the engine's shared context — not a re-implementation — so a censor learned during
play (U2) automatically prunes the candidate set the frames are built on.
"""
from __future__ import annotations

from dataclasses import dataclass

import chess

from yami.datagen.game_renderer import render_move


@dataclass
class FrameCandidate:
    """A Scale-0-surviving candidate annotated with the event-sentences it licenses.

    `frames` is the renderer's output for this move — the who-does-what prose the Genesis chess universes
    parse. Empty frames means the move is legal-and-safe but licenses no *named* tactical/positional
    inflection (a quiet move); that is information, not a defect."""

    move: chess.Move
    san: str
    frames: list[str]


def frame_candidates(engine, board: chess.Board, name: str | None = None) -> list[FrameCandidate]:
    """Render the Scale-0 candidate set into annotated frames.

    `engine` is a `ProgressiveRevealEngine` — we reuse its shared context (nav vector, position signals,
    learned censors) so the candidate set is EXACTLY the one the engine would route over. We run Scale-0
    only; higher scales are the router's job (U6), not the frame layer's.

    Postcondition (the U1 measure): every returned candidate is legal on `board`. Censored moves never
    reach the frame layer — they are gone before `render_move` is ever called.
    """
    from yami.scale_functions import scale_0_legal_and_censors

    name = name or ("White" if board.turn == chess.WHITE else "Black")
    context = engine._build_context(board)
    result = scale_0_legal_and_censors(board, [], context)

    out: list[FrameCandidate] = []
    for cand in result.residual:
        move = cand.move
        # Invariant guard: the seam must never leak an illegal move into the frame layer. scope_moves
        # only emits legal moves, so this can only fire on a genuine upstream regression — fail loud.
        if move not in board.legal_moves:
            raise AssertionError(f"Scale-0 leaked an illegal move into the frame layer: {move}")
        out.append(FrameCandidate(move=move, san=board.san(move), frames=render_move(board, move, name)))
    return out
