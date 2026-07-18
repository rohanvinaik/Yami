"""Signal-profile feature extraction for the fusion model.

Takes a board position, runs the full Yami infrastructure, and extracts
the complete signal profile (position signals + candidate signals) for
training the learned fusion model.

This replaces the old feature_extractor.py for the new architecture.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import chess

from yami.candidate_signals import (
    CandidateSignalVector,
    extract_candidate_signals,
)
from yami.coherence import MoveSignals, compute_coherence
from yami.navigator import (
    NavigationVector,
    compute_navigation_vector,
    detect_anchors,
    otp_score_candidate,
)
from yami.negative_learning import create_default_censors
from yami.position_signals import PositionSignals, extract_position_signals
from yami.tactical_scoper import (
    apply_blunder_censor,
    apply_repetition_censor,
    apply_tactical_censor,
    scope_moves,
)
from yami.temporal_controller import TemporalController


MAX_CANDIDATES = 5


@dataclass
class SignalProfileExample:
    """One training example with full signal profiles."""

    fen: str
    position_signals: list[float]     # PositionSignals.to_vector()
    candidate_signals: list[list[float]]  # [cand.to_vector() for each candidate]
    candidate_moves: list[str]        # UCI strings
    num_candidates: int
    # Labels (from Stockfish)
    best_candidate_idx: int
    oracle_eval_cp: int = 0
    eval_gap_cp: int = 0
    # Optional game outcome for bank profile training
    game_outcome: float = 0.0         # -1=loss, 0=draw, +1=win

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, line: str) -> SignalProfileExample:
        return cls(**json.loads(line))


def extract_signal_profile(
    board: chess.Board,
    temporal: TemporalController | None = None,
    gm_db=None,
    klines=None,
) -> tuple[PositionSignals, list[chess.Move], list[CandidateSignalVector]]:
    """Run the full infrastructure and extract signal profiles.

    Returns (position_signals, candidate_moves, candidate_signal_vectors).
    """
    # Step 1: Position signals (board-level)
    pos_signals = extract_position_signals(board)

    # Step 2: Navigator
    nav_vector = compute_navigation_vector(board)
    pos_signals.nav_aggression = nav_vector.aggression
    pos_signals.nav_piece_domain = nav_vector.piece_domain
    pos_signals.nav_complexity = nav_vector.complexity
    pos_signals.nav_initiative = nav_vector.initiative
    pos_signals.nav_king_pressure = nav_vector.king_pressure
    pos_signals.nav_phase = nav_vector.phase

    # Step 3: Scope and censor moves
    scoped = scope_moves(board)
    total_scoped = len(scoped)
    censored = apply_blunder_censor(scoped)
    censored = apply_tactical_censor(censored, board)
    censored = apply_repetition_censor(censored, board)
    if not censored:
        censored = scoped

    # Negative learning censor rate
    censors = create_default_censors()
    pre_censor = len(censored)
    censored = censors.filter_moves(board, censored, nav_vector)
    if not censored:
        censored = scoped[:MAX_CANDIDATES]
    pos_signals.censor_stack_rejection_rate = (
        1.0 - len(censored) / max(total_scoped, 1)
    )

    # Step 4: Coherence scoring (to get per-candidate signals)
    candidate_moves_raw = [m.move for m in censored[:MAX_CANDIDATES * 2]]
    coherence_result = compute_coherence(
        board,
        candidate_moves_raw,
        nav_vector,
        temporal=temporal,
        klines=klines,
        gm_db=gm_db,
    )

    # Step 5: SoM scores
    som_scores_position = {}
    convergence = 0.0
    trajectory = 0.0
    if temporal:
        all_anchors = set()
        for move in candidate_moves_raw[:5]:
            all_anchors |= detect_anchors(board, move)
        som_scores_position = temporal.score_move_by_agents(
            nav_vector, all_anchors, material_balance=0
        )
        convergence = temporal.compute_convergence(som_scores_position)
        trajectory = temporal.get_trajectory_trend()

    pos_signals.som_convergence = convergence
    pos_signals.som_trajectory_trend = trajectory

    # Step 6: Build per-candidate signal vectors
    # Use coherence result to get signal values, take top N by coherence score
    scored_moves = coherence_result.scored_moves[:MAX_CANDIDATES]

    candidate_moves = []
    candidate_vectors = []

    for sig in scored_moves:
        move = sig.move
        if move not in board.legal_moves:
            continue

        anchors = detect_anchors(board, move)

        # Per-candidate SoM scores
        move_som = {}
        if temporal:
            move_som = temporal.score_move_by_agents(
                nav_vector, anchors, material_balance=0
            )

        # Find SEE value from scoped moves
        see_val = 0
        for sm in scoped:
            if sm.move == move:
                see_val = sm.see_value
                break

        cand_sig = extract_candidate_signals(
            board,
            move,
            anchors=anchors,
            nav_vector=nav_vector,
            som_scores=move_som,
            strategy_score=sig.strategy_score,
            gm_freq=sig.gm_frequency,
            gm_winrate=sig.gm_win_rate,
            kline_score=sig.kline_score,
            look_ahead=sig.look_ahead_score,
            censor_pass=sig.censor_pass,
            interference=sig.interference,
            navigator_score=sig.navigator_score,
            navigator_ternary=sig.ternary_navigator,
            strategy_ternary=sig.ternary_strategy,
            gm_ternary=sig.ternary_gm,
            kline_ternary=sig.ternary_kline,
            see_value=see_val,
        )

        candidate_moves.append(move)
        candidate_vectors.append(cand_sig)

    return pos_signals, candidate_moves, candidate_vectors


def board_to_signal_example(
    board: chess.Board,
    best_idx: int = 0,
    oracle_eval_cp: int = 0,
    eval_gap_cp: int = 0,
    game_outcome: float = 0.0,
    temporal: TemporalController | None = None,
    gm_db=None,
    klines=None,
) -> SignalProfileExample | None:
    """Full pipeline: board → signal profile example for training."""
    pos_signals, cand_moves, cand_vectors = extract_signal_profile(
        board, temporal=temporal, gm_db=gm_db, klines=klines,
    )

    if len(cand_moves) < 2:
        return None

    # Pad candidates to MAX_CANDIDATES
    pos_vec = pos_signals.to_vector()
    cand_vecs = [cv.to_vector() for cv in cand_vectors]
    move_ucis = [m.uci() for m in cand_moves]

    # Pad
    pad_vec = [0.0] * CandidateSignalVector.vector_dim()
    while len(cand_vecs) < MAX_CANDIDATES:
        cand_vecs.append(pad_vec)
        move_ucis.append("")

    return SignalProfileExample(
        fen=board.fen(),
        position_signals=pos_vec,
        candidate_signals=cand_vecs[:MAX_CANDIDATES],
        candidate_moves=move_ucis[:MAX_CANDIDATES],
        num_candidates=len(cand_moves),
        best_candidate_idx=min(best_idx, len(cand_moves) - 1),
        oracle_eval_cp=oracle_eval_cp,
        eval_gap_cp=eval_gap_cp,
        game_outcome=game_outcome,
    )


def save_signal_dataset(
    examples: list[SignalProfileExample], path: Path
) -> None:
    """Save signal profile examples to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(ex.to_json() + "\n")


def append_signal_dataset(
    examples: list[SignalProfileExample], path: Path
) -> None:
    """Append signal profile examples to existing JSONL (crash-safe)."""
    with open(path, "a") as f:
        for ex in examples:
            f.write(ex.to_json() + "\n")


def load_signal_dataset(path: Path) -> list[SignalProfileExample]:
    """Load signal profile examples from JSONL."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(SignalProfileExample.from_json(line))
    return examples
