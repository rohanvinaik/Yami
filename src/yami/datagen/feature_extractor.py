"""Extract structured features from a board position via Yami Layers 1-6.

Runs the infrastructure pipeline and converts the output into numeric
feature vectors suitable for neural training.
"""

from __future__ import annotations

import chess

from yami.candidate_filter import filter_and_annotate
from yami.datagen.contracts import (
    MOTIF_TO_IDX,
    MOTIF_VOCAB,
    RISK_LEVELS,
    CandidateFeatures,
    ChessExample,
)
from yami.models import (
    Activity,
    Material,
    PlanType,
    Safety,
    Structure,
    Tempo,
)
from yami.navigator import compute_navigation_vector
from yami.tactical_scoper import (
    PIECE_VALUES,
    apply_blunder_censor,
    apply_repetition_censor,
    apply_tactical_censor,
    scope_moves,
)

_MATERIAL_MAP = {Material.BEHIND: 0, Material.EQUAL: 1, Material.AHEAD: 2}
_STRUCTURE_MAP = {
    Structure.OPEN: 0, Structure.SEMI_OPEN: 1, Structure.CLOSED: 2,
    Structure.HEDGEHOG: 3, Structure.ISOLATED: 4, Structure.HANGING: 5,
}
_ACTIVITY_MAP = {Activity.ACTIVE: 0, Activity.PASSIVE: 1, Activity.CRAMPED: 2}
_SAFETY_MAP = {Safety.SAFE: 0, Safety.EXPOSED: 1, Safety.UNDER_ATTACK: 2}
_TEMPO_MAP = {Tempo.AHEAD: 0, Tempo.EQUAL: 1, Tempo.BEHIND: 2}
_PLAN_MAP = {pt: i for i, pt in enumerate(PlanType)}

_PIECE_TYPE_IDX = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
}

# Max material for normalization (all pieces for both sides)
_MAX_MATERIAL = 2 * (900 + 2 * 500 + 2 * 330 + 2 * 320 + 8 * 100)  # 7880

_EMPTY_PIECE_ONEHOT = [0] * 6

# Padding candidate for positions with <5 candidates
_PAD_CANDIDATE = CandidateFeatures(
    move_uci="0000", move_san="--",
    motif_flags=[0] * len(MOTIF_VOCAB),
    plan_alignment=0.0, positional_eval=0.0,
    risk_level=0, is_capture=False, is_check=False, see_value=0.0,
    piece_type_onehot=[0] * 6, target_centrality=0.0,
    dist_to_opp_king=0.0, dist_to_own_king=0.0,
    is_castling=False, opponent_mobility=0.0, pawn_structure_change=0.0,
)


def extract_features(board: chess.Board, max_candidates: int = 5) -> tuple[
    list[CandidateFeatures],
    dict[str, int | float],
    int,
]:
    """Run Yami L1-6 on a board and emit structured features."""
    scoped = scope_moves(board)

    censored = apply_blunder_censor(scoped)
    censored = apply_tactical_censor(censored, board)
    censored = apply_repetition_censor(censored, board)
    if not censored:
        censored = scoped

    candidates, plan, profile = filter_and_annotate(
        board, censored, max_candidates=max_candidates
    )

    our_king = board.king(board.turn)
    opp_king = board.king(not board.turn)

    cand_features = []
    for c in candidates:
        motif_flags = [0] * len(MOTIF_VOCAB)
        for motif in c.tactical_motifs:
            idx = MOTIF_TO_IDX.get(motif)
            if idx is not None:
                motif_flags[idx] = 1

        # Piece type one-hot
        piece = board.piece_at(c.move.from_square)
        piece_onehot = list(_EMPTY_PIECE_ONEHOT)
        if piece:
            pt_idx = _PIECE_TYPE_IDX.get(piece.piece_type)
            if pt_idx is not None:
                piece_onehot[pt_idx] = 1

        # Target square geometry
        to_sq = c.move.to_square
        to_file = chess.square_file(to_sq)
        to_rank = chess.square_rank(to_sq)
        centrality = (4.0 - (abs(3.5 - to_file) + abs(3.5 - to_rank))) / 4.0

        dist_opp = chess.square_distance(to_sq, opp_king) / 7.0 if opp_king is not None else 1.0
        dist_own = chess.square_distance(to_sq, our_king) / 7.0 if our_king is not None else 1.0

        # Opponent mobility after this move
        board.push(c.move)
        opp_mobility = board.legal_moves.count() / 40.0
        board.pop()

        # Pawn structure change
        pawn_change = _eval_pawn_structure_change(board, c.move)

        cand_features.append(CandidateFeatures(
            move_uci=c.move.uci(),
            move_san=c.san,
            motif_flags=motif_flags,
            plan_alignment=c.plan_alignment,
            positional_eval=c.positional_eval,
            risk_level=RISK_LEVELS.get(c.risk, 0),
            is_capture="capture" in c.tactical_motifs,
            is_check="check" in c.tactical_motifs,
            see_value=c.see_value / 900.0,
            piece_type_onehot=piece_onehot,
            target_centrality=centrality,
            dist_to_opp_king=dist_opp,
            dist_to_own_king=dist_own,
            is_castling=board.is_castling(c.move),
            opponent_mobility=opp_mobility,
            pawn_structure_change=pawn_change,
        ))

    while len(cand_features) < max_candidates:
        cand_features.append(_PAD_CANDIDATE)

    profile_dict = {
        "material": _MATERIAL_MAP.get(profile.material, 1),
        "structure": _STRUCTURE_MAP.get(profile.structure, 1),
        "activity": _ACTIVITY_MAP.get(profile.activity, 0),
        "safety": _SAFETY_MAP.get(profile.safety, 0),
        "opponent_safety": _SAFETY_MAP.get(profile.opponent_safety, 0),
        "tempo": _TEMPO_MAP.get(profile.tempo, 1),
        "plan_activation": plan.activation_score,
    }

    plan_idx = _PLAN_MAP.get(plan.plan_type, 0)
    return cand_features, profile_dict, plan_idx


def _eval_pawn_structure_change(board: chess.Board, move: chess.Move) -> float:
    """Evaluate whether a move improves or worsens pawn structure."""
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return 0.0

    our_color = board.turn
    our_pawns = board.pieces(chess.PAWN, our_color)

    # Count isolated pawns before
    iso_before = _count_isolated(our_pawns)

    # Simulate move
    board.push(move)
    our_pawns_after = board.pieces(chess.PAWN, our_color)
    iso_after = _count_isolated(our_pawns_after)
    board.pop()

    diff = iso_before - iso_after  # positive = improved
    if diff > 0:
        return 1.0
    if diff < 0:
        return -1.0
    return 0.0


def _count_isolated(pawns: chess.SquareSet) -> int:
    """Count isolated pawns (no friendly pawns on adjacent files)."""
    isolated = 0
    for sq in pawns:
        f = chess.square_file(sq)
        has_neighbor = False
        for adj_f in [f - 1, f + 1]:
            if 0 <= adj_f <= 7 and (pawns & chess.BB_FILES[adj_f]):
                has_neighbor = True
                break
        if not has_neighbor:
            isolated += 1
    return isolated


def _compute_board_features(board: chess.Board) -> dict[str, float]:
    """Compute board-level continuous features."""
    total_mat = sum(
        PIECE_VALUES.get(p.piece_type, 0)
        for p in board.piece_map().values()
    )
    return {
        "game_phase": total_mat / _MAX_MATERIAL,
        "total_material": total_mat / _MAX_MATERIAL,
        "move_number": min(board.fullmove_number, 100) / 100.0,
    }


def board_to_example(
    board: chess.Board,
    best_idx: int,
    oracle_eval_cp: int,
    second_best_idx: int = -1,
    eval_gap_cp: int = 0,
) -> ChessExample:
    """Convert a board + oracle label into a full training example."""
    cand_features, profile, plan_idx = extract_features(board)
    board_feats = _compute_board_features(board)
    nav = compute_navigation_vector(board)

    return ChessExample(
        fen=board.fen(),
        material=profile["material"],
        structure=profile["structure"],
        activity=profile["activity"],
        safety=profile["safety"],
        opponent_safety=profile["opponent_safety"],
        tempo=profile["tempo"],
        plan_type=plan_idx,
        plan_activation=profile["plan_activation"],
        candidates=cand_features,
        num_candidates=min(5, sum(1 for c in cand_features if c.move_uci != "0000")),
        game_phase=board_feats["game_phase"],
        total_material=board_feats["total_material"],
        move_number=board_feats["move_number"],
        nav_vector=list(nav.as_tuple()),
        best_candidate_idx=best_idx,
        oracle_eval_cp=oracle_eval_cp,
        second_best_idx=second_best_idx,
        eval_gap_cp=eval_gap_cp,
    )
