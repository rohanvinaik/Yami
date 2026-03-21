"""Layer 2: Tactical Scoping — pattern-match tactical motifs on the board.

Tags each legal move with tactical motifs and detects forcing sequences.
The chess equivalent of tactic family identification in Wayfinder.
"""

from __future__ import annotations

import chess

from yami.models import ScopedMove

# Piece values in centipawns for SEE
PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


def scope_moves(board: chess.Board) -> list[ScopedMove]:
    """Scope all legal moves with tactical motifs."""
    scoped = []
    for move in board.legal_moves:
        motifs = _detect_motifs(board, move)
        see = _static_exchange_eval(board, move)
        scoped.append(ScopedMove(move=move, motifs=tuple(motifs), see_value=see))
    return scoped


def _detect_motifs(board: chess.Board, move: chess.Move) -> list[str]:
    """Detect tactical motifs for a move."""
    motifs: list[str] = []

    # Capture detection
    if board.is_capture(move):
        motifs.append("capture")

    # Pre-move checks
    moving_piece = board.piece_at(move.from_square)

    board.push(move)
    try:
        if board.is_check():
            motifs.append("check")
        if board.is_checkmate():
            motifs.append("checkmate")
        if _creates_fork(board, move, moving_piece):
            motifs.append("fork")
        if _creates_pin(board, move):
            motifs.append("pin")
        if _creates_discovered_attack(board, move):
            motifs.append("discovery")
        if _wins_material(board, move):
            motifs.append("material_gain")
        if _hangs_piece(board, move):
            motifs.append("hangs_piece")
    finally:
        board.pop()

    # Promotion
    if move.promotion:
        motifs.append("promotion")

    return motifs


def _creates_fork(
    board: chess.Board, move: chess.Move, piece: chess.Piece | None
) -> bool:
    """Check if the moved piece attacks two or more higher-value pieces."""
    if piece is None:
        return False

    attacks = board.attacks(move.to_square)
    attacked_values = []
    piece_value = PIECE_VALUES.get(piece.piece_type, 0)

    for sq in attacks:
        target = board.piece_at(sq)
        if target and target.color != piece.color:
            target_value = PIECE_VALUES.get(target.piece_type, 0)
            if target_value > piece_value:
                attacked_values.append(target_value)

    return len(attacked_values) >= 2


def _creates_pin(board: chess.Board, move: chess.Move) -> bool:
    """Check if the move creates a pin against the opponent's king."""
    opponent = board.turn  # after push, it's opponent's turn
    king_sq = board.king(opponent)
    if king_sq is None:
        return False

    our_color = not opponent
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.color != our_color:
            continue
        if piece.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            continue
        if _is_pinning(board, sq, king_sq, piece, opponent):
            return True
    return False


def _is_pinning(
    board: chess.Board,
    sq: chess.Square,
    king_sq: chess.Square,
    piece: chess.Piece,
    opponent: chess.Color,
) -> bool:
    """Check if a sliding piece at sq pins an opponent piece to king_sq."""
    between = chess.SquareSet(chess.between(sq, king_sq))
    if not between:
        return False
    blockers = [(bsq, bp) for bsq in between if (bp := board.piece_at(bsq))]
    if len(blockers) != 1 or blockers[0][1].color != opponent:
        return False
    return _attacks_along_line(board, sq, king_sq, piece, blockers[0][0])


def _attacks_along_line(
    board: chess.Board,
    sq: chess.Square,
    king_sq: chess.Square,
    piece: chess.Piece,
    blocker_sq: chess.Square,
) -> bool:
    """Check if a sliding piece attacks along the line from sq to king_sq."""
    if piece.piece_type == chess.QUEEN:
        return True
    if piece.piece_type == chess.ROOK:
        same_file = chess.square_file(sq) == chess.square_file(king_sq)
        same_rank = chess.square_rank(sq) == chess.square_rank(king_sq)
        return same_file or same_rank
    if piece.piece_type == chess.BISHOP:
        if chess.square_distance(sq, king_sq) <= 1:
            return False
        return sq in board.attacks(blocker_sq) or king_sq in board.attacks(sq)
    return False


def _creates_discovered_attack(board: chess.Board, move: chess.Move) -> bool:
    """Check if moving a piece reveals an attack from a piece behind it."""
    our_color = not board.turn
    # Check if any of our sliding pieces now attack the opponent king
    # through the square the moved piece vacated
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.color != our_color:
            continue
        if piece.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            continue
        if sq == move.to_square:
            continue
        # Does this piece attack the opponent's king?
        opp_king = board.king(board.turn)
        if opp_king is not None and opp_king in board.attacks(sq):
            # Was the from_square between this piece and the king?
            between = chess.SquareSet(chess.between(sq, opp_king))
            if move.from_square in between:
                return True
    return False


def _wins_material(board: chess.Board, move: chess.Move) -> bool:
    """Check if the move results in a net material gain."""
    see = _static_exchange_eval_post(board, move)
    return see > 50  # at least half a pawn advantage


def _hangs_piece(board: chess.Board, move: chess.Move) -> bool:
    """Check if the move leaves a piece hanging (undefended and attacked)."""
    # After the move, is the piece on to_square attacked and undefended?
    piece = board.piece_at(move.to_square)
    if piece is None:
        return False

    our_color = piece.color
    opp_color = not our_color

    attackers = board.attackers(opp_color, move.to_square)
    defenders = board.attackers(our_color, move.to_square)

    if attackers and not defenders:
        return True

    # Check if any of our other pieces are now hanging
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None or p.color != our_color or p.piece_type == chess.KING:
            continue
        if sq == move.to_square:
            continue
        att = board.attackers(opp_color, sq)
        defs = board.attackers(our_color, sq)
        if att and not defs and PIECE_VALUES.get(p.piece_type, 0) >= 300:
            return True

    return False


def _static_exchange_eval(board: chess.Board, move: chess.Move) -> int:
    """Static Exchange Evaluation — estimate material outcome of a capture sequence."""
    if not board.is_capture(move):
        return 0

    captured = board.piece_at(move.to_square)
    if captured is None:
        # En passant
        if board.is_en_passant(move):
            return PIECE_VALUES[chess.PAWN]
        return 0

    return PIECE_VALUES.get(captured.piece_type, 0)


def _static_exchange_eval_post(board: chess.Board, move: chess.Move) -> int:
    """SEE evaluated from the position after the move was played."""
    # Simplified: just check if the piece on to_square is adequately defended
    piece = board.piece_at(move.to_square)
    if piece is None:
        return 0

    our_color = piece.color
    opp_color = not our_color

    attackers = board.attackers(opp_color, move.to_square)
    if not attackers:
        return PIECE_VALUES.get(piece.piece_type, 0)

    defenders = board.attackers(our_color, move.to_square)
    if defenders:
        return 0

    return -PIECE_VALUES.get(piece.piece_type, 0)


# --- Censors ---


def apply_blunder_censor(moves: list[ScopedMove]) -> list[ScopedMove]:
    """Suppress moves that hang material. The LLM never sees blunders."""
    return [m for m in moves if "hangs_piece" not in m.motifs]


def apply_tactical_censor(
    moves: list[ScopedMove], board: chess.Board
) -> list[ScopedMove]:
    """Suppress moves that walk into known tactical patterns."""
    result = []
    for m in moves:
        board.push(m.move)
        # Check if opponent has a forcing response
        dominated = False
        for opp_move in board.legal_moves:
            board.push(opp_move)
            if board.is_checkmate():
                dominated = True
            board.pop()
            if dominated:
                break
        board.pop()
        if not dominated:
            result.append(m)
    return result


def apply_repetition_censor(
    moves: list[ScopedMove], board: chess.Board
) -> list[ScopedMove]:
    """Suppress moves that repeat a position unless winning."""
    result = []
    for m in moves:
        board.push(m.move)
        repeats = board.is_repetition(2)
        board.pop()
        if not repeats:
            result.append(m)
    # If all moves repeat, return them all (forced repetition)
    return result if result else moves
