"""6-Bank Chess Navigator — Wayfinder-style ternary navigation for chess.

Replaces the flat 5-dim knowledge graph with 6 orthogonal navigational banks,
each producing a ternary direction {-1, 0, +1}. Combined: 3^6 = 729 coarse
navigational bins, 100x finer resolution than the original 7 plan templates.

Banks:
  AGGRESSION:    -1=defensive, 0=balanced, +1=attacking
  PIECE_DOMAIN:  -1=pawn play, 0=mixed, +1=major piece activity
  COMPLEXITY:    -1=simple/forcing, 0=standard, +1=deep combination
  INITIATIVE:    -1=responding, 0=equal, +1=dictating
  KING_PRESSURE: -1=own king danger, 0=both safe, +1=targeting opponent king
  PHASE:         -1=endgame, 0=middlegame, +1=opening
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

from yami.tactical_scoper import PIECE_VALUES

# --- Anchor Dictionary ---
# ~100 chess tactical and strategic anchors

TACTICAL_ANCHORS = frozenset({
    "fork", "pin", "skewer", "discovered-attack", "double-check",
    "back-rank-threat", "smothered-mate", "sacrifice", "zwischenzug",
    "x-ray", "deflection", "decoy", "overloading", "removal-of-guard",
    "trapped-piece", "interference", "desperado",
})

STRATEGIC_ANCHORS = frozenset({
    "outpost", "open-file", "half-open-file", "pawn-break",
    "passed-pawn", "isolated-pawn", "doubled-pawns", "backward-pawn",
    "pawn-chain", "pawn-majority", "fianchetto", "bishop-pair",
    "good-bishop", "bad-bishop", "knight-outpost", "rook-on-seventh",
    "rook-on-open-file", "connected-rooks", "battery",
    "piece-coordination", "space-advantage", "pawn-storm",
    "minority-attack", "central-control", "king-activity",
})

POSITIONAL_ANCHORS = frozenset({
    "tempo-gain", "tempo-loss", "opposition", "zugzwang",
    "fortress", "blockade", "prophylaxis", "exchange-sacrifice",
    "positional-sacrifice", "piece-exchange", "simplification",
    "king-march", "pawn-race", "corresponding-squares",
})

PHASE_ANCHORS = frozenset({
    "development", "castling", "center-control", "king-safety",
    "endgame-technique", "pawn-endgame", "rook-endgame",
    "bishop-endgame", "knight-endgame", "queen-endgame",
    "opposite-bishops", "same-color-bishops",
})

ALL_ANCHORS = TACTICAL_ANCHORS | STRATEGIC_ANCHORS | POSITIONAL_ANCHORS | PHASE_ANCHORS
ANCHOR_TO_IDX = {a: i for i, a in enumerate(sorted(ALL_ANCHORS))}


@dataclass(frozen=True)
class NavigationVector:
    """6-bank ternary navigation vector."""

    aggression: int  # {-1, 0, +1}
    piece_domain: int
    complexity: int
    initiative: int
    king_pressure: int
    phase: int

    def as_tuple(self) -> tuple[int, ...]:
        return (
            self.aggression, self.piece_domain, self.complexity,
            self.initiative, self.king_pressure, self.phase,
        )

    def hamming_distance(self, other: NavigationVector) -> int:
        """Count mismatched banks."""
        return sum(
            a != b for a, b in zip(self.as_tuple(), other.as_tuple(), strict=True)
        )


# --- Navigation computation ---

_MAX_MATERIAL = 2 * (900 + 2 * 500 + 2 * 330 + 2 * 320 + 8 * 100)


def compute_navigation_vector(board: chess.Board) -> NavigationVector:
    """Compute the 6-bank ternary navigation vector for a position."""
    return NavigationVector(
        aggression=_compute_aggression(board),
        piece_domain=_compute_piece_domain(board),
        complexity=_compute_complexity(board),
        initiative=_compute_initiative(board),
        king_pressure=_compute_king_pressure(board),
        phase=_compute_phase(board),
    )


def _compute_aggression(board: chess.Board) -> int:
    """Assess the aggressive character of the position for side to move."""
    score = 0
    our_color = board.turn
    opp_king = board.king(not our_color)

    if opp_king is None:
        return 0

    # Pieces attacking near opponent king
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == our_color and piece.piece_type != chess.KING:
            dist = chess.square_distance(sq, opp_king)
            if dist <= 2:
                score += 2
            elif dist <= 4:
                score += 1

    # Checks available
    for move in board.legal_moves:
        board.push(move)
        if board.is_check():
            score += 2
        board.pop()
        if score >= 6:
            break

    if score >= 6:
        return 1  # attacking
    if score <= 1:
        return -1  # defensive
    return 0  # balanced


def _compute_piece_domain(board: chess.Board) -> int:
    """Is this a pawn-structure game or a piece-activity game?"""
    our_color = board.turn
    our_pieces = sum(
        1 for sq in chess.SQUARES
        if (p := board.piece_at(sq)) and p.color == our_color
        and p.piece_type in (chess.ROOK, chess.QUEEN, chess.BISHOP, chess.KNIGHT)
    )
    our_pawns = len(board.pieces(chess.PAWN, our_color))

    if our_pieces >= 4 and our_pawns <= 4:
        return 1  # major piece activity
    if our_pawns >= 5 and our_pieces <= 2:
        return -1  # pawn structure play
    return 0  # mixed


def _compute_complexity(board: chess.Board) -> int:
    """Is this a simple forcing position or a deep combinational one?"""
    legal_count = board.legal_moves.count()

    # Forcing: few legal moves, checks/captures dominate
    forcing = 0
    total_checked = 0
    for move in board.legal_moves:
        total_checked += 1
        if board.is_capture(move):
            forcing += 1
        board.push(move)
        if board.is_check():
            forcing += 1
        board.pop()
        if total_checked >= 10:
            break

    forcing_ratio = forcing / max(total_checked, 1)

    if forcing_ratio > 0.5 or legal_count < 10:
        return -1  # simple/forcing
    if legal_count > 30 and forcing_ratio < 0.2:
        return 1  # deep combination territory
    return 0  # standard


def _compute_initiative(board: chess.Board) -> int:
    """Does the side to move have the initiative?"""
    our_mobility = board.legal_moves.count()

    # Estimate opponent mobility
    board_copy = board.copy()
    board_copy.push(chess.Move.null())
    opp_mobility = board_copy.legal_moves.count()
    board_copy.pop()

    diff = our_mobility - opp_mobility
    if diff > 8:
        return 1  # dictating
    if diff < -8:
        return -1  # responding
    return 0  # equal


def _compute_king_pressure(board: chess.Board) -> int:
    """Is either king under pressure?"""
    our_color = board.turn
    our_king = board.king(our_color)
    opp_king = board.king(not our_color)

    our_danger = _king_danger(board, our_king, not our_color) if our_king else 0
    opp_danger = _king_danger(board, opp_king, our_color) if opp_king else 0

    if our_danger > opp_danger + 3:
        return -1  # own king in danger
    if opp_danger > our_danger + 3:
        return 1  # targeting opponent king
    return 0  # both safe


def _king_danger(
    board: chess.Board, king_sq: chess.Square | None, attacker_color: chess.Color
) -> int:
    """Score how much danger a king is in from the attacker's pieces."""
    if king_sq is None:
        return 0

    danger = 0
    # Pieces near the king
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == attacker_color:
            dist = chess.square_distance(sq, king_sq)
            if dist <= 2:
                danger += PIECE_VALUES.get(piece.piece_type, 0) // 100
            elif dist <= 3:
                danger += 1

    # Pawn shield weakness
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    defender_color = not attacker_color
    shield_pawns = board.pieces(chess.PAWN, defender_color)
    shield_dir = 1 if defender_color == chess.WHITE else -1
    shield_count = 0
    for df in [-1, 0, 1]:
        f = king_file + df
        r = king_rank + shield_dir
        if (0 <= f <= 7 and 0 <= r <= 7
                and chess.square(f, r) in shield_pawns):
            shield_count += 1

    if shield_count == 0:
        danger += 3
    elif shield_count == 1:
        danger += 1

    return danger


def _compute_phase(board: chess.Board) -> int:
    """What phase of the game are we in?"""
    total_material = sum(
        PIECE_VALUES.get(p.piece_type, 0)
        for p in board.piece_map().values()
        if p.piece_type != chess.KING
    )
    piece_count = chess.popcount(board.occupied)

    if piece_count <= 10 or total_material < 2000:
        return -1  # endgame
    if board.fullmove_number <= 12 and piece_count >= 28:
        return 1  # opening
    return 0  # middlegame


# --- Anchor Detection ---


def detect_anchors(board: chess.Board, move: chess.Move) -> set[str]:
    """Detect which anchors a move activates."""
    anchors: set[str] = set()
    piece = board.piece_at(move.from_square)
    if piece is None:
        return anchors

    # Castling (phase-independent)
    if board.is_castling(move):
        anchors.add("castling")
        anchors.add("king-safety")

    # Phase anchors
    phase = _compute_phase(board)
    if phase == 1:
        anchors.add("development")
        if 2 <= chess.square_file(move.to_square) <= 5:
            anchors.add("center-control")
    elif phase == -1:
        anchors.add("endgame-technique")
        if piece.piece_type == chess.KING:
            anchors.add("king-march")
            anchors.add("king-activity")

    # Piece-type anchors
    if piece.piece_type == chess.ROOK:
        to_file = chess.square_file(move.to_square)
        file_mask = chess.BB_FILES[to_file]
        if not (board.pieces(chess.PAWN, chess.WHITE) & file_mask) and \
           not (board.pieces(chess.PAWN, chess.BLACK) & file_mask):
            anchors.add("rook-on-open-file")
            anchors.add("open-file")
        to_rank = chess.square_rank(move.to_square)
        seventh = 6 if piece.color == chess.WHITE else 1
        if to_rank == seventh:
            anchors.add("rook-on-seventh")

    # Pawn structure anchors
    if piece.piece_type == chess.PAWN:
        to_file = chess.square_file(move.to_square)
        if 2 <= to_file <= 5:
            anchors.add("pawn-break")

        # Check for passed pawn
        our_color = piece.color
        opp_pawns = board.pieces(chess.PAWN, not our_color)
        to_rank = chess.square_rank(move.to_square)
        is_passed = True
        for f in [to_file - 1, to_file, to_file + 1]:
            if 0 <= f <= 7:
                for r in range(to_rank, 8 if our_color == chess.WHITE else -1,
                               1 if our_color == chess.WHITE else -1):
                    if chess.square(f, r) in opp_pawns:
                        is_passed = False
                        break
            if not is_passed:
                break
        if is_passed:
            anchors.add("passed-pawn")

    # Tactical anchors (from scoped motifs)
    if board.is_capture(move):
        anchors.add("piece-exchange")
    board.push(move)
    if board.is_check():
        anchors.add("tempo-gain")
    board.pop()

    # Exchange/simplification
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        if captured and PIECE_VALUES.get(captured.piece_type, 0) >= 300:
            anchors.add("simplification")

    return anchors


# --- OTP Scoring ---


def otp_score_candidate(
    move: chess.Move,
    move_anchors: set[str],
    nav_vector: NavigationVector,
    board: chess.Board,
) -> float:
    """Score a candidate move by OTP alignment with the navigation vector.

    OTP: {+1, 0, -1} = {support, irrelevant, contradict}
    A move's bank alignment is computed per-bank, then summed.
    """
    score = 0.0
    nv = nav_vector

    piece = board.piece_at(move.from_square)
    if piece is None:
        return 0.0

    # AGGRESSION alignment
    if nv.aggression == 1:  # attacking
        if "tempo-gain" in move_anchors:
            score += 2.0
        opp_king = board.king(not board.turn)
        if opp_king is not None:
            dist = chess.square_distance(move.to_square, opp_king)
            if dist <= 2:
                score += 1.5
    elif nv.aggression == -1:  # defensive
        own_king = board.king(board.turn)
        if own_king is not None and chess.square_distance(move.to_square, own_king) <= 2:
            score += 1.5
        if "king-safety" in move_anchors:
            score += 1.0

    # PIECE_DOMAIN alignment
    if nv.piece_domain == 1:  # major piece activity
        if piece.piece_type in (chess.ROOK, chess.QUEEN, chess.BISHOP, chess.KNIGHT):
            score += 1.0
        if "rook-on-open-file" in move_anchors or "rook-on-seventh" in move_anchors:
            score += 1.5
    elif nv.piece_domain == -1:  # pawn play
        if piece.piece_type == chess.PAWN:
            score += 1.0
        if "pawn-break" in move_anchors or "passed-pawn" in move_anchors:
            score += 1.5

    # COMPLEXITY alignment
    if nv.complexity == -1:  # simple/forcing
        board.push(move)
        if board.is_check():
            score += 2.0
        board.pop()
        if board.is_capture(move):
            score += 1.0
    elif nv.complexity == 1:  # deep combination
        if "piece-coordination" in move_anchors:
            score += 1.5
        if not board.is_capture(move):
            score += 0.5  # quiet moves for deep play

    # INITIATIVE alignment
    if nv.initiative == 1:  # dictating
        if "tempo-gain" in move_anchors:
            score += 2.0
        board.push(move)
        opp_moves = board.legal_moves.count()
        board.pop()
        if opp_moves < 15:
            score += 1.0  # restricting opponent

    # KING_PRESSURE alignment
    if nv.king_pressure == 1:  # targeting opponent king
        opp_king = board.king(not board.turn)
        if opp_king is not None:
            dist = chess.square_distance(move.to_square, opp_king)
            if dist <= 3:
                score += 2.0

    # PHASE alignment
    if nv.phase == -1:  # endgame
        if "endgame-technique" in move_anchors or "king-march" in move_anchors:
            score += 1.5
        if "passed-pawn" in move_anchors:
            score += 2.0
    elif nv.phase == 1:  # opening
        if "development" in move_anchors or "castling" in move_anchors:
            score += 2.0
        if "center-control" in move_anchors:
            score += 1.5

    # Anchor richness bonus (moves that activate many anchors are strategically rich)
    score += len(move_anchors) * 0.3

    return score


def rank_candidates_by_navigation(
    board: chess.Board,
    candidates: list[tuple[chess.Move, object]],
    nav_vector: NavigationVector,
) -> list[tuple[chess.Move, float, set[str]]]:
    """Rank candidates by OTP alignment with navigation vector.

    Args:
        board: Current board position.
        candidates: List of (move, scoped_move_or_data) pairs.
        nav_vector: 6-bank navigation direction.

    Returns:
        List of (move, otp_score, anchors) sorted by score descending.
    """
    scored = []
    for move, _ in candidates:
        anchors = detect_anchors(board, move)
        otp = otp_score_candidate(move, anchors, nav_vector, board)
        scored.append((move, otp, anchors))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
