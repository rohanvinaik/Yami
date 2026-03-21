"""Yami Engine — the full pipeline coordinator.

Wires all 8 layers together into a single decision pipeline:
Board → Legal Moves → Tactical Scoping → Endgame/Opening → Knowledge Graph
→ Candidate Filtering → LLM Decision → Verification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import chess

from yami import endgame_resolver, opening_book
from yami.candidate_filter import filter_and_annotate
from yami.legal_moves import generate_legal_moves, is_game_over, is_legal, parse_move
from yami.llm_decision import choose_move_sync
from yami.models import AnnotatedCandidate, GameState, PlanTemplate, PositionalProfile
from yami.tactical_scoper import (
    apply_blunder_censor,
    apply_repetition_censor,
    apply_tactical_censor,
    scope_moves,
)


class DecisionSource(Enum):
    """Where the move decision came from."""

    ENDGAME_TABLEBASE = "endgame_tablebase"
    OPENING_BOOK = "opening_book"
    LLM_DECISION = "llm_decision"
    INFRASTRUCTURE_FALLBACK = "infrastructure_fallback"
    NO_LEGAL_MOVES = "no_legal_moves"


@dataclass
class MoveDecision:
    """The result of the Yami pipeline for one position."""

    move: chess.Move | None
    source: DecisionSource
    candidates: list[AnnotatedCandidate] = field(default_factory=list)
    plan: PlanTemplate | None = None
    profile: PositionalProfile | None = None
    legal_move_count: int = 0
    scoped_move_count: int = 0
    censored_move_count: int = 0


class YamiEngine:
    """The full Yami infrastructure + LLM chess engine."""

    def __init__(
        self,
        use_llm: bool = True,
        use_opening_book: bool = True,
        use_endgame_tables: bool = True,
        use_censors: bool = True,
        max_candidates: int = 5,
    ):
        self.use_llm = use_llm
        self.use_opening_book = use_opening_book
        self.use_endgame_tables = use_endgame_tables
        self.use_censors = use_censors
        self.max_candidates = max_candidates
        self.state = GameState()

    def reset(self) -> None:
        """Reset the engine for a new game."""
        self.state = GameState()

    @property
    def board(self) -> chess.Board:
        return self.state.board

    def decide(self, board: chess.Board | None = None) -> MoveDecision:
        """Run the full Yami pipeline and decide on a move.

        This is the main entry point — it runs all 8 layers.
        """
        if board is None:
            board = self.state.board

        # Layer 1: Legal move generation
        legal_moves = generate_legal_moves(board)
        if not legal_moves:
            return MoveDecision(move=None, source=DecisionSource.NO_LEGAL_MOVES)

        # Layer 3: Endgame resolution (short-circuit)
        if self.use_endgame_tables and endgame_resolver.can_resolve(board):
            tb_move = endgame_resolver.resolve(board)
            if tb_move is not None:
                return MoveDecision(
                    move=tb_move,
                    source=DecisionSource.ENDGAME_TABLEBASE,
                    legal_move_count=len(legal_moves),
                )

        # Layer 4: Opening book (short-circuit)
        if self.use_opening_book and self.state.in_book:
            book_move = opening_book.get_book_move(board)
            if book_move is None:
                book_move = opening_book.builtin_lookup(board)
            if book_move is not None:
                return MoveDecision(
                    move=book_move,
                    source=DecisionSource.OPENING_BOOK,
                    legal_move_count=len(legal_moves),
                )
            else:
                self.state.in_book = False

        # Layer 2: Tactical scoping
        scoped = scope_moves(board)

        # Censors
        if self.use_censors:
            censored = apply_blunder_censor(scoped)
            censored = apply_tactical_censor(censored, board)
            censored = apply_repetition_censor(censored, board)
        else:
            censored = scoped

        # Ensure we have at least one move
        if not censored:
            censored = scoped  # fall back to uncensored if all censored

        # Layers 5-6: Knowledge graph + candidate filtering
        candidates, plan, profile = filter_and_annotate(
            board, censored, max_candidates=self.max_candidates
        )

        if not candidates:
            # Shouldn't happen, but safety fallback
            return MoveDecision(
                move=legal_moves[0],
                source=DecisionSource.INFRASTRUCTURE_FALLBACK,
                legal_move_count=len(legal_moves),
                scoped_move_count=len(scoped),
                censored_move_count=len(censored),
            )

        # Layer 7: LLM decision
        if self.use_llm:
            chosen = choose_move_sync(board, candidates, plan, profile)
        else:
            # Infrastructure-only mode: use top candidate
            chosen = candidates[0].move

        # Layer 8: Legal move verification
        if chosen is not None and not is_legal(board, chosen):
            # LLM hallucinated an illegal move — use top candidate
            chosen = candidates[0].move

        return MoveDecision(
            move=chosen,
            source=(
                DecisionSource.LLM_DECISION
                if self.use_llm
                else DecisionSource.INFRASTRUCTURE_FALLBACK
            ),
            candidates=candidates,
            plan=plan,
            profile=profile,
            legal_move_count=len(legal_moves),
            scoped_move_count=len(scoped),
            censored_move_count=len(censored),
        )

    def play_move(self, move: chess.Move | None = None) -> MoveDecision:
        """Decide on a move and play it on the internal board."""
        if move is not None:
            # External move (e.g., opponent's move)
            self.state.board.push(move)
            self.state.move_history.append(move)
            return MoveDecision(move=move, source=DecisionSource.INFRASTRUCTURE_FALLBACK)

        decision = self.decide()
        if decision.move is not None:
            self.state.board.push(decision.move)
            self.state.move_history.append(decision.move)
            if decision.plan:
                self.state.plan_history.append(decision.plan.plan_type)
        return decision

    def play_opponent_move(self, san: str) -> chess.Move | None:
        """Parse and play the opponent's move in SAN notation."""
        move = parse_move(self.state.board, san)
        if move is not None:
            self.state.board.push(move)
            self.state.move_history.append(move)
        return move

    def is_game_over(self) -> bool:
        return is_game_over(self.state.board)

    def result(self) -> str | None:
        from yami.legal_moves import game_result
        return game_result(self.state.board)
