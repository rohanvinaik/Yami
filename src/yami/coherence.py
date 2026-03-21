"""Multi-Signal Coherence Scoring — the sparse-wiki trajectory convergence
principle applied to chess move selection.

When independent signal sources agree on a move, that convergence IS the
confidence signal. When they diverge, it's a warning.

Signal sources (banks):
  1. Navigator OTP score (6-bank ternary alignment)
  2. Strategy Library match (pre-encoded multi-move patterns)
  3. Temporal Controller bias (plan persistence)
  4. K-Line Memory match (empirical winning patterns)
  5. Censor stack (negative learning — what NOT to do)

Coherence = weighted agreement across banks, inspired by:
  - sparse-wiki: Jaccard similarity across semantic banks
  - sparse-wiki: trajectory convergence for disambiguation
  - Wayfinder: multiplicative bank alignment scoring
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess

from yami.kline_memory import KLineMemory, NavigationVector
from yami.navigator import detect_anchors, otp_score_candidate
from yami.strategy_library import Strategy, query_strategies, score_move_against_strategy
from yami.temporal_controller import TemporalController


@dataclass
class MoveSignals:
    """All signal scores for a single candidate move."""

    move: chess.Move
    navigator_score: float = 0.0
    strategy_score: float = 0.0
    strategy_name: str = ""
    temporal_bias: float = 0.0
    kline_score: float = 0.0
    censor_pass: bool = True
    coherence: float = 0.0
    final_score: float = 0.0


@dataclass
class CoherenceResult:
    """Result of coherence scoring across all candidates."""

    scored_moves: list[MoveSignals] = field(default_factory=list)
    active_strategy: Strategy | None = None
    signal_agreement: float = 0.0  # 0-1, how much signals agree
    dominant_signal: str = ""  # which signal dominated


def compute_coherence(
    board: chess.Board,
    candidate_moves: list[chess.Move],
    nav_vector: NavigationVector,
    temporal: TemporalController | None = None,
    klines: KLineMemory | None = None,
) -> CoherenceResult:
    """Score candidate moves by multi-signal coherence.

    Each move gets scored by all available signal sources independently,
    then coherence (agreement between signals) is computed. Moves where
    signals converge get boosted; moves where signals diverge get suppressed.
    """
    # Gather position-level context
    all_anchors: set[str] = set()
    for move in candidate_moves[:5]:
        all_anchors |= detect_anchors(board, move)

    # Query strategy library
    strategies = query_strategies(board, nav_vector, all_anchors, top_k=2)
    active_strategy = strategies[0][0] if strategies else None

    # Get temporal bias
    temporal_bias: dict[str, float] = {}
    if temporal:
        temporal_bias = temporal.get_plan_bias()

    # K-line suggestions
    kline_move_ucis: set[str] = set()
    if klines:
        matches = klines.query(board, nav_vector, all_anchors, top_k=2)
        for kl in matches:
            if kl.move_sequence and kl.match_score > 0.4:
                # Try to parse the first suggested move
                try:
                    m = board.parse_san(kl.move_sequence[0])
                    kline_move_ucis.add(m.uci())
                except (chess.InvalidMoveError, chess.IllegalMoveError,
                        chess.AmbiguousMoveError):
                    pass

    # Score each candidate move
    results: list[MoveSignals] = []
    for move in candidate_moves:
        signals = MoveSignals(move=move)

        # Signal 1: Navigator OTP
        move_anchors = detect_anchors(board, move)
        signals.navigator_score = otp_score_candidate(
            move, move_anchors, nav_vector, board
        )

        # Signal 2: Strategy library
        if active_strategy:
            signals.strategy_score = score_move_against_strategy(
                board, move, active_strategy
            )
            signals.strategy_name = active_strategy.name

        # Signal 3: Temporal bias
        if temporal_bias:
            # Sum biases — positive means move aligns with plan
            signals.temporal_bias = sum(temporal_bias.values()) * 0.5

        # Signal 4: K-line match
        if move.uci() in kline_move_ucis:
            signals.kline_score = 3.0  # strong boost for K-line match

        results.append(signals)

    # Normalize each signal to [0, 1] range for coherence computation
    _normalize_signals(results)

    # Compute coherence: agreement between normalized signals
    for signals in results:
        scores = [
            signals.navigator_score,
            signals.strategy_score,
            signals.temporal_bias,
        ]
        # Only include K-line if any match exists
        if kline_move_ucis:
            scores.append(signals.kline_score)

        non_zero = [s for s in scores if s > 0.01]
        if len(non_zero) >= 2:
            # Coherence = product of agreement
            # When multiple signals agree (all high), coherence is high
            # When they disagree (mixed), coherence is moderate
            mean = sum(non_zero) / len(non_zero)
            variance = sum((s - mean) ** 2 for s in non_zero) / len(non_zero)
            agreement = 1.0 - min(variance, 1.0)  # low variance = high agreement
            signals.coherence = agreement * mean
        elif len(non_zero) == 1:
            signals.coherence = non_zero[0] * 0.5  # single signal, half confidence
        else:
            signals.coherence = 0.0

        # Final score: weighted combination with coherence bonus
        signals.final_score = (
            signals.navigator_score * 2.0
            + signals.strategy_score * 2.5
            + signals.temporal_bias * 1.5
            + signals.kline_score * 2.0
            + signals.coherence * 3.0  # coherence is the strongest signal
        )

        # Censor penalty
        if not signals.censor_pass:
            signals.final_score *= 0.1

    # Sort by final score
    results.sort(key=lambda s: s.final_score, reverse=True)

    # Compute overall signal agreement
    if results:
        coherences = [s.coherence for s in results[:3]]
        overall_agreement = sum(coherences) / max(len(coherences), 1)
    else:
        overall_agreement = 0.0

    # Determine dominant signal
    dominant = ""
    if results:
        top = results[0]
        signal_contribs = {
            "navigator": top.navigator_score * 2.0,
            "strategy": top.strategy_score * 2.5,
            "temporal": top.temporal_bias * 1.5,
            "kline": top.kline_score * 2.0,
            "coherence": top.coherence * 3.0,
        }
        dominant = max(signal_contribs, key=signal_contribs.get)  # type: ignore[arg-type]

    return CoherenceResult(
        scored_moves=results,
        active_strategy=active_strategy,
        signal_agreement=overall_agreement,
        dominant_signal=dominant,
    )


def _normalize_signals(results: list[MoveSignals]) -> None:
    """Normalize each signal dimension to [0, 1] across candidates."""
    if not results:
        return

    for attr in ("navigator_score", "strategy_score", "temporal_bias", "kline_score"):
        values = [getattr(s, attr) for s in results]
        max_val = max(values) if values else 1.0
        if max_val > 0:
            for s in results:
                setattr(s, attr, getattr(s, attr) / max_val)
