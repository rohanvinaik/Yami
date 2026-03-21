"""Holographic Coherence — multi-signal fusion through interference patterns.

The answer to "what move should I play?" exists as an interference pattern
across all signal sources simultaneously. Like a hologram, no single fragment
contains the full image — but the superposition reconstructs it.

Signal sources (banks):
  1. Navigator OTP score (6-bank ternary alignment)
  2. Strategy Library match (pre-encoded multi-move patterns)
  3. Temporal SoM agents (6 specialist scores + convergence)
  4. K-Line Memory match (empirical winning patterns)
  5. GM Pattern Database (grandmaster move frequencies)
  6. Censor stack (negative learning — what NOT to do)

OTP Ternary Signal Handling:
  +1 = signal supports this move (constructive interference)
   0 = signal is orthogonal (no information — NOT disagreement)
  -1 = signal opposes this move (destructive interference)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess

from yami.gm_patterns import GMPatternDB, GMSuggestion
from yami.kline_memory import KLineMemory, NavigationVector
from yami.navigator import detect_anchors, otp_score_candidate
from yami.strategy_library import Strategy, query_strategies, score_move_against_strategy
from yami.temporal_controller import TemporalController


@dataclass
class MoveSignals:
    """All signal scores for a single candidate move (OTP ternary)."""

    move: chess.Move
    # Raw scores
    navigator_score: float = 0.0
    strategy_score: float = 0.0
    strategy_name: str = ""
    temporal_scores: dict[str, float] = field(default_factory=dict)
    temporal_convergence: float = 0.0
    kline_score: float = 0.0
    gm_frequency: float = 0.0
    gm_win_rate: float = 0.5
    censor_pass: bool = True
    # OTP ternary signals (-1, 0, +1)
    ternary_navigator: int = 0
    ternary_strategy: int = 0
    ternary_temporal: int = 0
    ternary_kline: int = 0
    ternary_gm: int = 0
    # Computed
    coherence: float = 0.0
    interference: float = 0.0  # constructive (+) or destructive (-)
    final_score: float = 0.0


@dataclass
class CoherenceResult:
    """Result of holographic coherence scoring."""

    scored_moves: list[MoveSignals] = field(default_factory=list)
    active_strategy: Strategy | None = None
    signal_agreement: float = 0.0
    dominant_signal: str = ""
    trajectory_trend: float = 0.0  # positive = plan converging
    constructive_count: int = 0  # moves with 3+ agreeing signals
    destructive_count: int = 0  # moves with contradicting signals


def compute_coherence(
    board: chess.Board,
    candidate_moves: list[chess.Move],
    nav_vector: NavigationVector,
    temporal: TemporalController | None = None,
    klines: KLineMemory | None = None,
    gm_db: GMPatternDB | None = None,
) -> CoherenceResult:
    """Score candidate moves by holographic multi-signal coherence."""
    # Gather position-level context
    all_anchors: set[str] = set()
    for move in candidate_moves[:8]:
        all_anchors |= detect_anchors(board, move)

    # Query strategy library
    strategies = query_strategies(board, nav_vector, all_anchors, top_k=2)
    active_strategy = strategies[0][0] if strategies else None

    # K-line suggestions
    kline_move_ucis: set[str] = set()
    if klines:
        matches = klines.query(board, nav_vector, all_anchors, top_k=2)
        for kl in matches:
            if kl.move_sequence and kl.match_score > 0.4:
                try:
                    m = board.parse_san(kl.move_sequence[0])
                    kline_move_ucis.add(m.uci())
                except (chess.InvalidMoveError, chess.IllegalMoveError,
                        chess.AmbiguousMoveError):
                    pass

    # GM pattern suggestions
    gm_suggestions: dict[str, GMSuggestion] = {}
    if gm_db:
        gm_results = gm_db.query(board, nav_vector, top_k=5)
        for gs in gm_results:
            gm_suggestions[gs.move_uci] = gs

    # Get SoM agent scores for position context
    som_scores: dict[str, float] = {}
    convergence = 0.0
    trajectory_trend = 0.0
    if temporal:
        som_scores = temporal.score_move_by_agents(
            nav_vector, all_anchors, material_balance=0
        )
        convergence = temporal.compute_convergence(som_scores)
        trajectory_trend = temporal.get_trajectory_trend()

    # Score each candidate
    results: list[MoveSignals] = []
    for move in candidate_moves:
        signals = MoveSignals(move=move)
        move_anchors = detect_anchors(board, move)

        # Signal 1: Navigator OTP
        signals.navigator_score = otp_score_candidate(
            move, move_anchors, nav_vector, board
        )

        # Signal 2: Strategy library
        if active_strategy:
            signals.strategy_score = score_move_against_strategy(
                board, move, active_strategy
            )
            signals.strategy_name = active_strategy.name

        # Signal 3: Temporal SoM agents
        if som_scores:
            # Score this specific move through agents
            move_som = temporal.score_move_by_agents(
                nav_vector, move_anchors, material_balance=0
            ) if temporal else {}
            signals.temporal_scores = move_som
            signals.temporal_convergence = convergence

        # Signal 4: K-line match
        if move.uci() in kline_move_ucis:
            signals.kline_score = 3.0

        # Signal 5: GM pattern frequency
        gm = gm_suggestions.get(move.uci())
        if gm:
            signals.gm_frequency = gm.frequency
            signals.gm_win_rate = gm.win_rate

        # Convert to OTP ternary (-1, 0, +1)
        signals.ternary_navigator = _to_ternary(signals.navigator_score, 2.0)
        signals.ternary_strategy = _to_ternary(signals.strategy_score, 1.5)
        signals.ternary_temporal = _to_ternary(
            sum(signals.temporal_scores.values()), 3.0
        )
        signals.ternary_kline = 1 if signals.kline_score > 0 else 0
        signals.ternary_gm = _to_ternary(signals.gm_frequency, 0.1)

        results.append(signals)

    # Normalize raw scores across candidates
    _normalize_signals(results)

    # Compute holographic interference pattern for each move
    constructive = 0
    destructive = 0

    for signals in results:
        ternary_values = [
            signals.ternary_navigator,
            signals.ternary_strategy,
            signals.ternary_temporal,
            signals.ternary_kline,
            signals.ternary_gm,
        ]

        # Count support/oppose/orthogonal (OTP)
        supports = sum(1 for t in ternary_values if t == 1)
        opposes = sum(1 for t in ternary_values if t == -1)
        # orthogonal signals are explicitly "no opinion" (OTP informational zero)

        # Interference pattern
        # Constructive: 3+ signals agree → strong
        # Partial: 2 agree, rest orthogonal → moderate
        # Destructive: signals contradict → weak
        if supports >= 3:
            signals.interference = supports * 1.5
            constructive += 1
        elif supports >= 2 and opposes == 0:
            signals.interference = supports * 1.0
        elif opposes >= 2:
            signals.interference = -opposes * 1.0
            destructive += 1
        else:
            signals.interference = (supports - opposes) * 0.5

        # Coherence: weighted combination
        signals.coherence = max(0.0, signals.interference / 5.0)

        # Final score: raw signals + interference pattern + convergence
        signals.final_score = (
            signals.navigator_score * 2.0
            + signals.strategy_score * 2.5
            + sum(signals.temporal_scores.values()) * 0.3
            + signals.kline_score * 1.5
            + signals.gm_frequency * 4.0  # GM patterns carry strong weight
            + signals.gm_win_rate * 2.0  # winning moves weighted more
            + signals.interference * 3.0  # interference is key
            + signals.temporal_convergence * 2.0  # convergence bonus
        )

        if not signals.censor_pass:
            signals.final_score *= 0.1

    # Sort by final score
    results.sort(key=lambda s: s.final_score, reverse=True)

    # Overall metrics
    if results:
        coherences = [s.coherence for s in results[:3]]
        overall_agreement = sum(coherences) / max(len(coherences), 1)
    else:
        overall_agreement = 0.0

    dominant = ""
    if results:
        top = results[0]
        contribs = {
            "navigator": top.navigator_score * 2.0,
            "strategy": top.strategy_score * 2.5,
            "temporal": sum(top.temporal_scores.values()) * 0.3,
            "kline": top.kline_score * 1.5,
            "gm_pattern": top.gm_frequency * 4.0,
            "interference": top.interference * 3.0,
        }
        dominant = max(contribs, key=contribs.get)  # type: ignore[arg-type]

    return CoherenceResult(
        scored_moves=results,
        active_strategy=active_strategy,
        signal_agreement=overall_agreement,
        dominant_signal=dominant,
        trajectory_trend=trajectory_trend,
        constructive_count=constructive,
        destructive_count=destructive,
    )


def _to_ternary(value: float, threshold: float) -> int:
    """Convert a continuous score to OTP ternary {-1, 0, +1}.

    Informational Zero: 0 means orthogonal (no information),
    NOT absence. This is the OTP principle.
    """
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def _normalize_signals(results: list[MoveSignals]) -> None:
    """Normalize raw signal dimensions to [0, 1] across candidates."""
    if not results:
        return

    for attr in ("navigator_score", "strategy_score", "kline_score", "gm_frequency"):
        values = [getattr(s, attr) for s in results]
        max_val = max(values) if values else 1.0
        if max_val > 0:
            for s in results:
                setattr(s, attr, getattr(s, attr) / max_val)
