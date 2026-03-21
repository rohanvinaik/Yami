"""Layer 5: Positional Knowledge Graph — 5-dimensional positional navigation.

The chess instantiation of GSE: meaning (the right move) arises from position
in structured geometric space, not from learned embeddings.
"""

from __future__ import annotations

from collections.abc import Callable

import chess

from yami.models import (
    Activity,
    Material,
    PlanTemplate,
    PlanType,
    PositionalProfile,
    RankedMove,
    Safety,
    ScopedMove,
    Structure,
    Tempo,
)
from yami.tactical_scoper import PIECE_VALUES

# --- Plan template definitions ---

PLAN_TEMPLATES: list[PlanTemplate] = [
    PlanTemplate(
        PlanType.ATTACK_KING,
        "Attack King",
        "Direct attack on the opponent's king — sacrifices, pawn storms, piece convergence",
    ),
    PlanTemplate(
        PlanType.IMPROVE_PIECES,
        "Improve Pieces",
        "Maneuver pieces to better squares — outpost occupation, piece coordination",
    ),
    PlanTemplate(
        PlanType.PAWN_BREAK,
        "Pawn Break",
        "Central or flank pawn advance to open lines or create weaknesses",
    ),
    PlanTemplate(
        PlanType.SIMPLIFY,
        "Simplify",
        "Exchange pieces to reach a winning endgame — trade when ahead in material",
    ),
    PlanTemplate(
        PlanType.FORTIFY,
        "Fortify",
        "Defensive consolidation — improve king shelter, block attacking lines",
    ),
    PlanTemplate(
        PlanType.EXPLOIT_WEAKNESS,
        "Exploit Weakness",
        "Apply pressure on weak pawns, squares, or structures in opponent's position",
    ),
    PlanTemplate(
        PlanType.PROPHYLAXIS,
        "Prophylaxis",
        "Prevent the opponent's plan before it materializes — Nimzowitsch-style restraint",
    ),
]

_TEMPLATE_BY_TYPE = {t.plan_type: t for t in PLAN_TEMPLATES}


# --- Positional evaluation functions ---


def evaluate_position(board: chess.Board) -> PositionalProfile:
    """Evaluate the position along 5 navigational dimensions."""
    return PositionalProfile(
        material=_eval_material(board),
        structure=_eval_pawn_structure(board),
        activity=_eval_piece_activity(board),
        safety=_eval_king_safety(board, board.turn),
        opponent_safety=_eval_king_safety(board, not board.turn),
        tempo=_eval_tempo(board),
    )


def suggest_plan(profile: PositionalProfile) -> PlanTemplate:
    """Navigate the knowledge graph to find the right plan via spreading activation."""
    scores: dict[PlanType, float] = {}
    for template in PLAN_TEMPLATES:
        scores[template.plan_type] = _activation_score(profile, template.plan_type)

    best_type = max(scores, key=lambda k: scores[k])
    best = _TEMPLATE_BY_TYPE[best_type]
    return PlanTemplate(
        plan_type=best.plan_type,
        name=best.name,
        description=best.description,
        activation_score=scores[best_type],
    )


def rank_moves(
    moves: list[ScopedMove], plan: PlanTemplate, board: chess.Board
) -> list[RankedMove]:
    """Rank moves by alignment with the active plan."""
    ranked = []
    for m in moves:
        alignment = _plan_alignment(m, plan, board)
        ranked.append(RankedMove(scoped_move=m, alignment=alignment))
    return sorted(ranked, key=lambda r: r.alignment, reverse=True)


# --- Internal evaluation functions ---


def _eval_material(board: chess.Board) -> Material:
    """Evaluate material balance from the perspective of the side to move."""
    white_material = sum(
        PIECE_VALUES.get(p.piece_type, 0)
        for p in board.piece_map().values()
        if p.color == chess.WHITE
    )
    black_material = sum(
        PIECE_VALUES.get(p.piece_type, 0)
        for p in board.piece_map().values()
        if p.color == chess.BLACK
    )

    if board.turn == chess.WHITE:
        diff = white_material - black_material
    else:
        diff = black_material - white_material

    if diff > 150:
        return Material.AHEAD
    if diff < -150:
        return Material.BEHIND
    return Material.EQUAL


def _eval_pawn_structure(board: chess.Board) -> Structure:
    """Classify the pawn structure type."""
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)

    # Count open files (no pawns from either side)
    open_files = 0
    semi_open = 0
    for f in range(8):
        file_mask = chess.BB_FILES[f]
        has_white = bool(white_pawns & file_mask)
        has_black = bool(black_pawns & file_mask)
        if not has_white and not has_black:
            open_files += 1
        elif not has_white or not has_black:
            semi_open += 1

    # Count isolated pawns
    our_pawns = white_pawns if board.turn == chess.WHITE else black_pawns
    isolated = 0
    for sq in our_pawns:
        f = chess.square_file(sq)
        has_neighbor = False
        for adj_f in [f - 1, f + 1]:
            if 0 <= adj_f <= 7:
                adj_mask = chess.BB_FILES[adj_f]
                if our_pawns & adj_mask:
                    has_neighbor = True
                    break
        if not has_neighbor:
            isolated += 1

    if isolated >= 2:
        return Structure.ISOLATED

    # Check for locked center (pawns blocking each other on d/e files)
    d_file = chess.BB_FILES[3]
    e_file = chess.BB_FILES[4]
    center_locked = (
        bool(white_pawns & d_file)
        and bool(black_pawns & d_file)
        and bool(white_pawns & e_file)
        and bool(black_pawns & e_file)
    )

    if center_locked:
        return Structure.CLOSED
    if open_files >= 3:
        return Structure.OPEN
    if semi_open >= 3:
        return Structure.SEMI_OPEN
    return Structure.SEMI_OPEN


def _eval_piece_activity(board: chess.Board) -> Activity:
    """Evaluate piece activity based on mobility."""
    # Count total legal moves as a proxy for activity
    our_mobility = board.legal_moves.count()

    # Estimate opponent mobility
    board_copy = board.copy()
    board_copy.push(chess.Move.null())
    opp_mobility = board_copy.legal_moves.count()
    board_copy.pop()

    if our_mobility > opp_mobility + 10:
        return Activity.ACTIVE
    if our_mobility < opp_mobility - 10:
        return Activity.CRAMPED
    if our_mobility < 15:
        return Activity.PASSIVE
    return Activity.ACTIVE


def _eval_king_safety(board: chess.Board, color: chess.Color) -> Safety:
    """Evaluate king safety for the given color."""
    king_sq = board.king(color)
    if king_sq is None:
        return Safety.UNDER_ATTACK

    # Check if king is in check
    if board.turn == color and board.is_check():
        return Safety.UNDER_ATTACK

    king_rank = chess.square_rank(king_sq)
    king_file = chess.square_file(king_sq)

    # King in center (not castled) is exposed
    home_rank = 0 if color == chess.WHITE else 7
    in_center = king_rank == home_rank and 2 <= king_file <= 5
    if in_center and board.fullmove_number > 10:
        return Safety.EXPOSED

    # Check pawn shield
    our_pawns = board.pieces(chess.PAWN, color)
    shield_squares = []
    direction = 1 if color == chess.WHITE else -1
    for df in [-1, 0, 1]:
        f = king_file + df
        r = king_rank + direction
        if 0 <= f <= 7 and 0 <= r <= 7:
            shield_squares.append(chess.square(f, r))

    shield_count = sum(1 for sq in shield_squares if sq in our_pawns)

    # Count attackers near the king
    opp_color = not color
    attackers = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == opp_color and piece.piece_type != chess.PAWN:
            dist = chess.square_distance(sq, king_sq)
            if dist <= 3:
                attackers += 1

    if attackers >= 3 or (attackers >= 2 and shield_count == 0):
        return Safety.UNDER_ATTACK
    if shield_count <= 1 and board.fullmove_number > 10:
        return Safety.EXPOSED
    return Safety.SAFE


def _eval_tempo(board: chess.Board) -> Tempo:
    """Evaluate development/initiative tempo."""
    white_developed = 0
    black_developed = 0

    # Count developed minor pieces (not on starting squares)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            rank = chess.square_rank(sq)
            if piece.color == chess.WHITE and rank > 0:
                white_developed += 1
            elif piece.color == chess.BLACK and rank < 7:
                black_developed += 1

    if board.turn == chess.WHITE:
        diff = white_developed - black_developed
    else:
        diff = black_developed - white_developed

    # Also consider castling rights used
    color = board.turn
    has_k = board.has_kingside_castling_rights(color)
    has_q = board.has_queenside_castling_rights(color)
    if not has_k or not has_q:
        diff += 1  # has castled

    if diff >= 2:
        return Tempo.AHEAD
    if diff <= -2:
        return Tempo.BEHIND
    return Tempo.EQUAL


def _activation_score(profile: PositionalProfile, plan_type: PlanType) -> float:
    """Spreading activation: compute how strongly a plan template should activate."""
    score = 0.0

    match plan_type:
        case PlanType.ATTACK_KING:
            if profile.opponent_safety == Safety.EXPOSED:
                score += 3.0
            if profile.opponent_safety == Safety.UNDER_ATTACK:
                score += 4.0
            if profile.activity == Activity.ACTIVE:
                score += 1.5
            if profile.tempo == Tempo.AHEAD:
                score += 1.0

        case PlanType.IMPROVE_PIECES:
            if profile.activity == Activity.PASSIVE:
                score += 3.0
            if profile.activity == Activity.CRAMPED:
                score += 2.0
            if profile.structure == Structure.CLOSED:
                score += 1.5

        case PlanType.PAWN_BREAK:
            if profile.structure == Structure.CLOSED:
                score += 3.0
            if profile.tempo == Tempo.AHEAD:
                score += 1.0
            if profile.activity == Activity.CRAMPED:
                score += 1.5

        case PlanType.SIMPLIFY:
            if profile.material == Material.AHEAD:
                score += 4.0
            if profile.activity == Activity.PASSIVE:
                score += 0.5

        case PlanType.FORTIFY:
            if profile.safety == Safety.EXPOSED:
                score += 3.0
            if profile.safety == Safety.UNDER_ATTACK:
                score += 4.0
            if profile.material == Material.BEHIND:
                score += 1.0

        case PlanType.EXPLOIT_WEAKNESS:
            if profile.structure in (Structure.ISOLATED, Structure.HANGING):
                score += 3.0
            if profile.activity == Activity.ACTIVE:
                score += 1.0
            if profile.structure == Structure.SEMI_OPEN:
                score += 1.5

        case PlanType.PROPHYLAXIS:
            if profile.opponent_safety == Safety.SAFE:
                score += 1.0
            if profile.material == Material.EQUAL:
                score += 1.0
            if profile.structure == Structure.CLOSED:
                score += 1.0

    return score


def _plan_alignment(
    move: ScopedMove, plan: PlanTemplate, board: chess.Board
) -> float:
    """Score how well a move aligns with the active plan."""
    scorer = _PLAN_SCORERS.get(plan.plan_type, lambda _m, _b: 0.0)
    score = scorer(move, board)

    # Universal bonuses
    if "material_gain" in move.motifs:
        score += 2.0
    if "promotion" in move.motifs:
        score += 5.0

    return score


def _score_attack_king(move: ScopedMove, board: chess.Board) -> float:
    score = 0.0
    if "check" in move.motifs:
        score += 3.0
    if "checkmate" in move.motifs:
        score += 10.0
    if "fork" in move.motifs:
        score += 2.0
    opp_king = board.king(not board.turn)
    if opp_king is not None:
        dist_before = chess.square_distance(move.move.from_square, opp_king)
        dist_after = chess.square_distance(move.move.to_square, opp_king)
        if dist_after < dist_before:
            score += 1.0
    return score


def _score_improve_pieces(move: ScopedMove, board: chess.Board) -> float:
    score = 0.0
    piece = board.piece_at(move.move.from_square)
    if piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
        to_file = chess.square_file(move.move.to_square)
        to_rank = chess.square_rank(move.move.to_square)
        center_dist = abs(3.5 - to_file) + abs(3.5 - to_rank)
        score += max(0, 4 - center_dist)
    if "capture" not in move.motifs:
        score += 0.5
    return score


def _score_pawn_break(move: ScopedMove, board: chess.Board) -> float:
    score = 0.0
    piece = board.piece_at(move.move.from_square)
    if piece and piece.piece_type == chess.PAWN:
        score += 2.0
        to_file = chess.square_file(move.move.to_square)
        if 2 <= to_file <= 5:
            score += 1.5
    return score


def _score_simplify(move: ScopedMove, board: chess.Board) -> float:
    score = 0.0
    if "capture" in move.motifs:
        score += 2.0
    if move.see_value > 0:
        score += 1.5
    return score


def _score_fortify(move: ScopedMove, board: chess.Board) -> float:
    score = 0.0
    piece = board.piece_at(move.move.from_square)
    our_king = board.king(board.turn)
    if piece and our_king is not None:
        dist_to_king = chess.square_distance(move.move.to_square, our_king)
        if dist_to_king <= 2:
            score += 2.0
    if "check" not in move.motifs and "capture" not in move.motifs:
        score += 0.5
    return score


def _score_exploit_weakness(move: ScopedMove, board: chess.Board) -> float:
    score = 0.0
    if "capture" in move.motifs:
        score += 1.5
    if "pin" in move.motifs:
        score += 2.0
    if move.see_value > 0:
        score += 1.0
    return score


def _score_prophylaxis(move: ScopedMove, board: chess.Board) -> float:
    score = 0.0
    board.push(move.move)
    opp_moves = board.legal_moves.count()
    board.pop()
    if opp_moves < 20:
        score += 1.5
    if "capture" not in move.motifs and "check" not in move.motifs:
        score += 0.5
    return score


_PLAN_SCORERS: dict[
    PlanType,
    Callable[[ScopedMove, chess.Board], float],
] = {
    PlanType.ATTACK_KING: _score_attack_king,
    PlanType.IMPROVE_PIECES: _score_improve_pieces,
    PlanType.PAWN_BREAK: _score_pawn_break,
    PlanType.SIMPLIFY: _score_simplify,
    PlanType.FORTIFY: _score_fortify,
    PlanType.EXPLOIT_WEAKNESS: _score_exploit_weakness,
    PlanType.PROPHYLAXIS: _score_prophylaxis,
}
