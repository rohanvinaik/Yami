"""Yami Engine — the full pipeline coordinator.

Wires all layers together into a single decision pipeline:
Board → Legal Moves → Tactical Scoping → Endgame/Opening
→ 6-Bank Navigator → K-Line Memory → Temporal Controller
→ Candidate Filtering → Neural/LLM Decision → Verification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import chess

from yami import endgame_resolver, opening_book
from yami.candidate_filter import filter_and_annotate
from yami.kline_memory import KLineMemory
from yami.legal_moves import generate_legal_moves, is_game_over, is_legal, parse_move
from yami.llm_decision import choose_move_sync
from yami.models import AnnotatedCandidate, GameState, PlanTemplate, PositionalProfile
from yami.navigator import (
    NavigationVector,
    compute_navigation_vector,
    detect_anchors,
)
from yami.tactical_scoper import (
    apply_blunder_censor,
    apply_repetition_censor,
    apply_tactical_censor,
    scope_moves,
)
from yami.temporal_controller import TemporalController, TemporalState


class DecisionSource(Enum):
    """Where the move decision came from."""

    ENDGAME_TABLEBASE = "endgame_tablebase"
    OPENING_BOOK = "opening_book"
    KLINE_MEMORY = "kline_memory"
    LLM_DECISION = "llm_decision"
    NEURAL_DECISION = "neural_decision"
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
    nav_vector: NavigationVector | None = None
    temporal_state: TemporalState | None = None
    legal_move_count: int = 0
    scoped_move_count: int = 0
    censored_move_count: int = 0


class YamiEngine:
    """The full Yami infrastructure + neural chess engine."""

    def __init__(
        self,
        use_llm: bool = True,
        use_neural: bool = False,
        neural_checkpoint: str | None = None,
        neural_config: object | None = None,
        use_opening_book: bool = True,
        use_endgame_tables: bool = True,
        use_censors: bool = True,
        use_navigator: bool = True,
        use_klines: bool = False,
        use_temporal: bool = True,
        kline_db_path: str | None = None,
        max_candidates: int = 5,
    ):
        self.use_llm = use_llm
        self.use_neural = use_neural
        self.use_opening_book = use_opening_book
        self.use_endgame_tables = use_endgame_tables
        self.use_censors = use_censors
        self.use_navigator = use_navigator
        self.use_klines = use_klines
        self.use_temporal = use_temporal
        self.max_candidates = max_candidates
        self.state = GameState()

        # Learned censors (negative learning)
        from yami.negative_learning import create_default_censors
        self._learned_censors = create_default_censors()

        # Neural model
        self._neural_decider = None
        if use_neural and neural_checkpoint:
            from yami.neural.inference import NeuralDecider
            self._neural_decider = NeuralDecider(
                neural_checkpoint, config=neural_config
            )

        # Temporal controller
        self._temporal = TemporalController() if use_temporal else None

        # K-line memory
        self._klines = None
        if use_klines and kline_db_path:
            self._klines = KLineMemory(kline_db_path)

    def reset(self) -> None:
        """Reset the engine for a new game."""
        self.state = GameState()
        if self._temporal:
            self._temporal.reset()

    @property
    def board(self) -> chess.Board:
        return self.state.board

    def decide(self, board: chess.Board | None = None) -> MoveDecision:
        """Run the full Yami pipeline and decide on a move."""
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

        # Short-circuit: if checkmate exists, play it immediately
        for m in scoped:
            if "checkmate" in m.motifs:
                return MoveDecision(
                    move=m.move,
                    source=DecisionSource.INFRASTRUCTURE_FALLBACK,
                    legal_move_count=len(legal_moves),
                    scoped_move_count=len(scoped),
                )

        # Censors
        if self.use_censors:
            censored = apply_blunder_censor(scoped)
            censored = apply_tactical_censor(censored, board)
            censored = apply_repetition_censor(censored, board)
        else:
            censored = scoped

        if not censored:
            censored = scoped

        # Layer 5: Multi-Signal Coherence (Navigator + Strategy + Temporal + K-Lines)
        nav_vector = None
        if self.use_navigator:
            nav_vector = compute_navigation_vector(board)

            # Apply learned censors (negative learning)
            censored = self._learned_censors.filter_moves(
                board, censored, nav_vector
            )

            # Run coherence scoring across all signals
            from yami.coherence import compute_coherence
            coherence_result = compute_coherence(
                board,
                [m.move for m in censored[:10]],  # top 10 candidates
                nav_vector,
                temporal=self._temporal,
                klines=self._klines,
            )

            # Reorder censored moves by coherence-weighted final score
            if coherence_result.scored_moves:
                coherence_order = {
                    s.move: i
                    for i, s in enumerate(coherence_result.scored_moves)
                }
                censored = sorted(
                    censored,
                    key=lambda m: coherence_order.get(m.move, 999),
                )

        # Layers 5-6: Candidate filtering + annotation
        candidates, plan, profile = filter_and_annotate(
            board, censored, max_candidates=self.max_candidates
        )

        if not candidates:
            return MoveDecision(
                move=legal_moves[0],
                source=DecisionSource.INFRASTRUCTURE_FALLBACK,
                legal_move_count=len(legal_moves),
                scoped_move_count=len(scoped),
                censored_move_count=len(censored),
            )

        # Layer 7: Decision
        if self.use_neural and self._neural_decider is not None:
            chosen, _conf = self._neural_decider.decide(
                board, candidates, plan, profile
            )
            source = DecisionSource.NEURAL_DECISION
        elif self.use_llm:
            chosen = choose_move_sync(board, candidates, plan, profile)
            source = DecisionSource.LLM_DECISION
        else:
            chosen = candidates[0].move
            source = DecisionSource.INFRASTRUCTURE_FALLBACK

        # Layer 8: Legal move verification
        if chosen is not None and not is_legal(board, chosen):
            chosen = candidates[0].move

        return MoveDecision(
            move=chosen,
            source=source,
            candidates=candidates,
            plan=plan,
            profile=profile,
            nav_vector=nav_vector,
            legal_move_count=len(legal_moves),
            scoped_move_count=len(scoped),
            censored_move_count=len(censored),
        )

    def play_move(self, move: chess.Move | None = None) -> MoveDecision:
        """Decide on a move and play it on the internal board."""
        if move is not None:
            self.state.board.push(move)
            self.state.move_history.append(move)
            return MoveDecision(move=move, source=DecisionSource.INFRASTRUCTURE_FALLBACK)

        decision = self.decide()
        if decision.move is not None:
            # Update temporal controller before pushing
            if self._temporal and decision.nav_vector:
                move_anchors = detect_anchors(self.state.board, decision.move)
                self._temporal.update(
                    decision.nav_vector,
                    move_anchors,
                )
            self.state.board.push(decision.move)
            self.state.move_history.append(decision.move)
            if decision.plan:
                self.state.plan_history.append(decision.plan.plan_type)
        return decision

    def play_opponent_move(self, san: str) -> chess.Move | None:
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
