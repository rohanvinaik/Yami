"""Opponent Profiling — behavioral compass for chess opponents.

Inspired by LintGate's behavioral compass: profile the opponent along
axes that matter for risk/strategy calibration, not move prediction.

The system plays the SAME principled chess regardless of opponent.
The profile only adjusts RISK TOLERANCE:
  - Against weak/random opponents: accept more risk, play for initiative
  - Against strong opponents: play solid, avoid unnecessary complications
  - Against tactical opponents: be extra careful about forcing lines
  - Against positional opponents: seek complications to disrupt their plans

This is NOT about changing strategy. It's about calibrating how much
uncertainty the system tolerates in its move choices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import chess


class OpponentStrength(Enum):
    UNKNOWN = "unknown"
    WEAK = "weak"  # random/beginner — play freely
    MODERATE = "moderate"  # club player — play solid
    STRONG = "strong"  # engine/expert — play very solid
    ENGINE = "engine"  # Stockfish-class — maximize survival


@dataclass
class OpponentBehavior:
    """Observed behavioral axes of the opponent."""

    # Tactical tendency: does opponent find tactics?
    tactics_found: int = 0  # checks, forks, pins played
    tactics_missed: int = 0  # hanging pieces left, missed mates

    # Aggressiveness: does opponent push for initiative?
    aggressive_moves: int = 0  # advances toward our king, sacrifices
    passive_moves: int = 0  # retreats, quiet moves when attacking was possible

    # Consistency: does opponent play coherent plans?
    plan_changes: int = 0  # sudden strategy shifts
    plan_follows: int = 0  # moves consistent with apparent plan

    # Speed: how quickly does opponent's advantage grow?
    eval_gains: list[int] = field(default_factory=list)  # centipawn gains per move


@dataclass
class OpponentProfile:
    """Multi-axis behavioral profile of the current opponent.

    Each axis is a float in [-1, +1]:
      -1 = weak on this axis
       0 = unknown/average
      +1 = strong on this axis
    """

    tactical_skill: float = 0.0  # -1=misses tactics, +1=finds them
    aggressiveness: float = 0.0  # -1=passive, +1=aggressive
    consistency: float = 0.0  # -1=erratic, +1=coherent plans
    pressure_rate: float = 0.0  # -1=slow, +1=fast advantage growth
    estimated_strength: OpponentStrength = OpponentStrength.UNKNOWN

    def risk_tolerance(self) -> float:
        """How much risk should we accept? Higher = more aggressive play.

        Returns 0.0-1.0 where:
          0.0 = maximum caution (play against engines)
          0.5 = balanced (default)
          1.0 = maximum aggression (play against random)
        """
        if self.estimated_strength == OpponentStrength.WEAK:
            return 0.85
        if self.estimated_strength == OpponentStrength.ENGINE:
            return 0.15
        if self.estimated_strength == OpponentStrength.STRONG:
            return 0.25

        # Dynamic: based on observed behavior
        # Weak tactics + passive + erratic = we can take risks
        weakness_signal = (
            max(0, -self.tactical_skill)
            + max(0, -self.aggressiveness)
            + max(0, -self.consistency)
        ) / 3.0

        return min(0.9, 0.4 + weakness_signal * 0.5)

    def look_ahead_weight(self) -> float:
        """How much to weight look-ahead penalty.

        Against strong opponents: high (avoid traps).
        Against weak opponents: low (their responses won't exploit our mistakes).
        """
        if self.estimated_strength == OpponentStrength.WEAK:
            return 0.2
        if self.estimated_strength == OpponentStrength.ENGINE:
            return 2.0
        if self.estimated_strength == OpponentStrength.STRONG:
            return 1.5
        return 0.8  # moderate default


class OpponentProfiler:
    """Build and update an opponent profile from observed moves."""

    def __init__(self) -> None:
        self.behavior = OpponentBehavior()
        self.profile = OpponentProfile()
        self.moves_observed = 0

    def reset(self) -> None:
        self.behavior = OpponentBehavior()
        self.profile = OpponentProfile()
        self.moves_observed = 0

    def observe_move(
        self,
        board_before: chess.Board,
        move: chess.Move,
        eval_before: int = 0,
        eval_after: int = 0,
    ) -> None:
        """Observe an opponent's move and update profile."""
        self.moves_observed += 1

        # Tactical skill: did opponent play checks/captures when available?
        board_before.push(move)
        gave_check = board_before.is_check()
        board_before.pop()

        is_capture = board_before.is_capture(move)

        if gave_check or is_capture:
            self.behavior.tactics_found += 1
        else:
            # Were there better tactical options?
            has_tactical = False
            for legal in board_before.legal_moves:
                if board_before.is_capture(legal):
                    has_tactical = True
                    break
            if has_tactical:
                self.behavior.passive_moves += 1

        # Aggressiveness: is the move advancing toward our king?
        our_king = board_before.king(not board_before.turn)
        if our_king is not None:
            dist_before = chess.square_distance(move.from_square, our_king)
            dist_after = chess.square_distance(move.to_square, our_king)
            if dist_after < dist_before:
                self.behavior.aggressive_moves += 1
            elif dist_after > dist_before + 1:
                self.behavior.passive_moves += 1

        # Eval tracking
        if eval_before != 0 or eval_after != 0:
            gain = eval_after - eval_before
            self.behavior.eval_gains.append(gain)

        # Update profile every 5 moves
        if self.moves_observed % 5 == 0:
            self._update_profile()

    def _update_profile(self) -> None:
        """Recompute profile from accumulated behavior."""
        total_tactical = self.behavior.tactics_found + self.behavior.tactics_missed
        if total_tactical > 0:
            self.profile.tactical_skill = (
                (self.behavior.tactics_found - self.behavior.tactics_missed)
                / total_tactical
            )

        total_tempo = self.behavior.aggressive_moves + self.behavior.passive_moves
        if total_tempo > 0:
            self.profile.aggressiveness = (
                (self.behavior.aggressive_moves - self.behavior.passive_moves)
                / total_tempo
            )

        total_plan = self.behavior.plan_follows + self.behavior.plan_changes
        if total_plan > 0:
            self.profile.consistency = (
                (self.behavior.plan_follows - self.behavior.plan_changes)
                / total_plan
            )

        if self.behavior.eval_gains:
            avg_gain = sum(self.behavior.eval_gains) / len(self.behavior.eval_gains)
            self.profile.pressure_rate = min(1.0, max(-1.0, avg_gain / 50.0))

        # Estimate strength
        if self.moves_observed >= 10:
            skill = (
                self.profile.tactical_skill
                + self.profile.consistency
                + self.profile.pressure_rate
            ) / 3.0

            if skill > 0.5:
                self.profile.estimated_strength = OpponentStrength.STRONG
            elif skill > 0.0 or skill > -0.3:
                self.profile.estimated_strength = OpponentStrength.MODERATE
            else:
                self.profile.estimated_strength = OpponentStrength.WEAK

    def set_known_opponent(self, strength: OpponentStrength) -> None:
        """Set opponent strength directly (e.g., when playing vs known engine)."""
        self.profile.estimated_strength = strength
