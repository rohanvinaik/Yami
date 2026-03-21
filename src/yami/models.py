"""Core data models for the Yami architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import chess

# --- Layer 5: Positional dimensions ---


class Material(Enum):
    AHEAD = "ahead"
    EQUAL = "equal"
    BEHIND = "behind"


class Structure(Enum):
    OPEN = "open"
    SEMI_OPEN = "semi_open"
    CLOSED = "closed"
    HEDGEHOG = "hedgehog"
    ISOLATED = "isolated"
    HANGING = "hanging"


class Activity(Enum):
    ACTIVE = "active"
    PASSIVE = "passive"
    CRAMPED = "cramped"


class Safety(Enum):
    SAFE = "safe"
    EXPOSED = "exposed"
    UNDER_ATTACK = "under_attack"


class Tempo(Enum):
    AHEAD = "ahead"
    EQUAL = "equal"
    BEHIND = "behind"


class PlanType(Enum):
    ATTACK_KING = "attack_king"
    IMPROVE_PIECES = "improve_pieces"
    PAWN_BREAK = "pawn_break"
    SIMPLIFY = "simplify"
    FORTIFY = "fortify"
    EXPLOIT_WEAKNESS = "exploit_weakness"
    PROPHYLAXIS = "prophylaxis"


# --- Data classes ---


@dataclass(frozen=True)
class ScopedMove:
    """A legal move tagged with tactical motifs."""

    move: chess.Move
    motifs: tuple[str, ...] = ()
    see_value: int = 0  # static exchange evaluation in centipawns


@dataclass(frozen=True)
class PositionalProfile:
    """5-dimensional positional assessment."""

    material: Material
    structure: Structure
    activity: Activity
    safety: Safety
    opponent_safety: Safety
    tempo: Tempo


@dataclass(frozen=True)
class PlanTemplate:
    """A strategic plan with activation conditions."""

    plan_type: PlanType
    name: str
    description: str
    activation_score: float = 0.0


@dataclass(frozen=True)
class RankedMove:
    """A move ranked by plan alignment."""

    scoped_move: ScopedMove
    alignment: float = 0.0


@dataclass(frozen=True)
class AnnotatedCandidate:
    """A fully annotated candidate move for the LLM."""

    move: chess.Move
    san: str
    tactical_motifs: tuple[str, ...]
    plan_alignment: float
    plan_name: str
    positional_eval: float
    risk: str
    narrative: str


@dataclass
class BookMove:
    """A move from the opening book with weight."""

    move: chess.Move
    weight: int = 1
    games: int = 0
    win_rate: float = 0.5


@dataclass
class GameState:
    """Full game state tracked by the engine."""

    board: chess.Board = field(default_factory=chess.Board)
    move_history: list[chess.Move] = field(default_factory=list)
    plan_history: list[PlanType] = field(default_factory=list)
    in_book: bool = True
