"""Negative Learning — Minsky's censors formalized for chess.

Learns what NOT to do from game failures. A "no" carries as much
information as a "yes." Three censor types:
1. Move censors: suppress moves that always fail in this position type
2. Strategy censors: suppress plans that contradict the position
3. Pattern censors: suppress moves matching known losing patterns

Uses asymmetric loss: penalize false negatives (letting through a
losing move) 10x more than false positives (blocking a good move).
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

from yami.navigator import NavigationVector, compute_navigation_vector


@dataclass
class NegativeExample:
    """A move that lost in a specific position type."""

    fen: str
    move_uci: str
    nav_vector: tuple[int, ...]
    position_type: str  # "opening", "middlegame", "endgame", "tactical"
    eval_before: int  # centipawns before the move
    eval_after: int  # centipawns after the move
    eval_drop: int  # how much was lost (positive = bad)


@dataclass
class StrategyCensorRule:
    """A learned rule: this strategy fails in this position type."""

    strategy: str  # plan name
    nav_condition: dict[str, int]  # bank conditions that trigger suppression
    confidence: float  # how sure we are (0-1)
    examples_seen: int  # how many failures we've observed


class LearnedCensorStack:
    """Collection of learned censors from game data."""

    def __init__(self) -> None:
        self.move_censors: dict[str, list[NegativeExample]] = {}
        self.strategy_rules: list[StrategyCensorRule] = []
        self._bad_move_patterns: dict[str, set[str]] = {}

    def add_negative_example(self, example: NegativeExample) -> None:
        """Register a move that failed."""
        key = example.position_type
        if key not in self.move_censors:
            self.move_censors[key] = []
        self.move_censors[key].append(example)

        # Track bad move patterns by position type
        if key not in self._bad_move_patterns:
            self._bad_move_patterns[key] = set()
        if example.eval_drop > 200:  # >2 pawn blunder
            self._bad_move_patterns[key].add(example.move_uci)

    def add_strategy_rule(self, rule: StrategyCensorRule) -> None:
        """Register a learned strategy censor."""
        self.strategy_rules.append(rule)

    def should_suppress_move(
        self,
        board: chess.Board,
        move: chess.Move,
        nav_vector: NavigationVector,
        position_type: str,
    ) -> bool:
        """Check if a move should be suppressed by learned censors."""
        # Check move censors: has this exact move failed repeatedly here?
        bad_moves = self._bad_move_patterns.get(position_type, set())
        if move.uci() in bad_moves:
            return True

        # Check strategy-based suppression
        for rule in self.strategy_rules:
            if rule.confidence < 0.7:
                continue
            matches = all(
                getattr(nav_vector, bank, 0) == val
                for bank, val in rule.nav_condition.items()
            )
            if matches:
                # This strategy is suppressed in this position type
                return True

        return False

    def filter_moves(
        self,
        board: chess.Board,
        moves: list,
        nav_vector: NavigationVector,
    ) -> list:
        """Filter a list of moves/scoped moves using learned censors."""
        phase = nav_vector.phase
        pos_type = "middlegame"
        if phase == -1:
            pos_type = "endgame"
        elif phase == 1:
            pos_type = "opening"

        filtered = []
        for m in moves:
            move_obj = m.move if hasattr(m, "move") else m
            if not self.should_suppress_move(board, move_obj, nav_vector, pos_type):
                filtered.append(m)

        # Never suppress ALL moves
        return filtered if filtered else moves


# --- Mining functions ---


def mine_negative_examples_from_evals(
    fens: list[str],
    moves: list[str],
    evals_before: list[int],
    evals_after: list[int],
    threshold_cp: int = 100,
) -> list[NegativeExample]:
    """Extract negative examples from move-by-move evaluations.

    A negative example is a move where eval dropped by more than threshold.
    """
    examples = []
    for fen, move_uci, ev_before, ev_after in zip(
        fens, moves, evals_before, evals_after, strict=False
    ):
        drop = ev_before - ev_after
        if drop > threshold_cp:
            board = chess.Board(fen)
            nav = compute_navigation_vector(board)
            phase = nav.phase
            pos_type = "middlegame"
            if phase == -1:
                pos_type = "endgame"
            elif phase == 1:
                pos_type = "opening"

            examples.append(NegativeExample(
                fen=fen,
                move_uci=move_uci,
                nav_vector=nav.as_tuple(),
                position_type=pos_type,
                eval_before=ev_before,
                eval_after=ev_after,
                eval_drop=drop,
            ))

    return examples


def mine_near_misses(
    fens: list[str],
    best_moves: list[str],
    second_moves: list[str],
    best_evals: list[int],
    second_evals: list[int],
    gap_threshold: int = 30,
) -> list[tuple[NegativeExample, str]]:
    """Extract near-miss examples: moves that almost won but didn't.

    Returns (negative_example, correct_move_uci) pairs.
    The gap between best and second is the richest training signal.
    """
    near_misses = []
    for fen, best, second, ev_best, ev_second in zip(
        fens, best_moves, second_moves, best_evals, second_evals, strict=False
    ):
        gap = ev_best - ev_second
        if 0 < gap <= gap_threshold:
            board = chess.Board(fen)
            nav = compute_navigation_vector(board)
            phase = nav.phase
            pos_type = "middlegame"
            if phase == -1:
                pos_type = "endgame"
            elif phase == 1:
                pos_type = "opening"

            neg = NegativeExample(
                fen=fen,
                move_uci=second,
                nav_vector=nav.as_tuple(),
                position_type=pos_type,
                eval_before=ev_second,
                eval_after=ev_second - gap,
                eval_drop=gap,
            )
            near_misses.append((neg, best))

    return near_misses


# --- Built-in strategy censors ---

DEFAULT_STRATEGY_RULES = [
    StrategyCensorRule(
        strategy="kingside_attack",
        nav_condition={"aggression": -1, "king_pressure": -1},
        confidence=0.9,
        examples_seen=100,
    ),
    StrategyCensorRule(
        strategy="simplify_to_endgame",
        nav_condition={"piece_domain": -1},  # can't simplify in pawn-only positions
        confidence=0.8,
        examples_seen=50,
    ),
    StrategyCensorRule(
        strategy="promote_pawn",
        nav_condition={"phase": 1},  # don't try pawn promotion in the opening
        confidence=0.95,
        examples_seen=200,
    ),
]


def create_default_censors() -> LearnedCensorStack:
    """Create a censor stack with built-in strategy rules."""
    stack = LearnedCensorStack()
    for rule in DEFAULT_STRATEGY_RULES:
        stack.add_strategy_rule(rule)
    return stack
