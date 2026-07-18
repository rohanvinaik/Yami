"""Per-position signal extraction — the full deterministic feature vector.

Computes every static evaluation signal for a board position.
These signals are the MODEL'S ONLY INPUT — it never sees the raw board.
The infrastructure does the understanding; the model just learns which
signals to trust.

Signals organized by classical evaluation categories:
  1. Material signals
  2. Pawn structure signals
  3. King safety signals
  4. Piece activity signals
  5. Phase / tempo signals
  6. Yami-specific signals (navigator, SoM, strategy, opponent profile)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess

# Kaufman piece values (centipawns)
PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 325, chess.BISHOP: 325,
    chess.ROOK: 500, chess.QUEEN: 975, chess.KING: 0,
}

# Center squares for control calculation
CENTER = {chess.D4, chess.D5, chess.E4, chess.E5}
EXTENDED_CENTER = CENTER | {
    chess.C3, chess.C4, chess.C5, chess.C6,
    chess.D3, chess.D6, chess.E3, chess.E6,
    chess.F3, chess.F4, chess.F5, chess.F6,
}


@dataclass
class PositionSignals:
    """Complete per-position signal vector (~60 dimensions)."""

    # --- Material (5) ---
    material_balance: float = 0.0       # (ours - theirs) / queen_value
    total_material: float = 0.0         # total on board, normalized
    material_imbalance: float = 0.0     # Kaufman-style pair interaction
    bishop_pair_ours: bool = False
    bishop_pair_theirs: bool = False

    # --- Pawn Structure (12) ---
    pawn_islands_ours: int = 0
    pawn_islands_theirs: int = 0
    isolated_pawns_ours: int = 0
    isolated_pawns_theirs: int = 0
    doubled_pawns_ours: int = 0
    doubled_pawns_theirs: int = 0
    backward_pawns_ours: int = 0
    backward_pawns_theirs: int = 0
    passed_pawns_ours: int = 0
    passed_pawns_theirs: int = 0
    passed_pawn_max_rank_ours: float = 0.0   # 0-1 normalized
    passed_pawn_max_rank_theirs: float = 0.0

    # --- King Safety (8) ---
    king_pawn_shelter_ours: float = 0.0    # 0=no shelter, 1=full
    king_pawn_shelter_theirs: float = 0.0
    king_open_files_ours: int = 0          # open files near our king
    king_open_files_theirs: int = 0
    king_attacker_count_ours: int = 0      # pieces attacking our king zone
    king_attacker_count_theirs: int = 0
    castled_ours: bool = False
    castled_theirs: bool = False

    # --- Piece Activity (8) ---
    mobility_ours: float = 0.0       # normalized legal move count
    mobility_theirs: float = 0.0
    development_ours: int = 0        # minor pieces off back rank
    development_theirs: int = 0
    good_bishop_ours: float = 0.0    # bishop on opposite color from own pawns
    good_bishop_theirs: float = 0.0
    connected_rooks_ours: bool = False
    connected_rooks_theirs: bool = False

    # --- Phase / Tempo (6) ---
    game_phase: float = 0.0          # 1=opening, 0=endgame (from material)
    move_number: float = 0.0         # normalized (move/100)
    center_control_ours: float = 0.0 # squares attacked in center
    center_control_theirs: float = 0.0
    space_advantage: float = 0.0     # safe squares beyond 4th rank
    in_check: bool = False

    # --- Castling Rights (4) ---
    can_castle_kingside_ours: bool = False
    can_castle_queenside_ours: bool = False
    can_castle_kingside_theirs: bool = False
    can_castle_queenside_theirs: bool = False

    # --- Yami-Specific (filled by engine) (15+) ---
    # Navigator 6-bank vector
    nav_aggression: int = 0     # {-1, 0, +1}
    nav_piece_domain: int = 0
    nav_complexity: int = 0
    nav_initiative: int = 0
    nav_king_pressure: int = 0
    nav_phase: int = 0
    # SoM convergence
    som_convergence: float = 0.0
    som_trajectory_trend: float = 0.0
    # Strategy
    strategy_match_strength: float = 0.0
    # Opponent profile
    opp_tactical_skill: float = 0.0
    opp_aggressiveness: float = 0.0
    opp_consistency: float = 0.0
    opp_pressure_rate: float = 0.0
    opp_risk_tolerance: float = 0.0
    # Negative learning
    censor_stack_rejection_rate: float = 0.0  # what fraction of moves were censored

    def to_vector(self) -> list[float]:
        """Flatten to a numeric vector for the neural model."""
        return [
            # Material (5)
            self.material_balance,
            self.total_material,
            self.material_imbalance,
            float(self.bishop_pair_ours),
            float(self.bishop_pair_theirs),
            # Pawn Structure (12)
            self.pawn_islands_ours / 4.0,
            self.pawn_islands_theirs / 4.0,
            self.isolated_pawns_ours / 8.0,
            self.isolated_pawns_theirs / 8.0,
            self.doubled_pawns_ours / 4.0,
            self.doubled_pawns_theirs / 4.0,
            self.backward_pawns_ours / 4.0,
            self.backward_pawns_theirs / 4.0,
            self.passed_pawns_ours / 4.0,
            self.passed_pawns_theirs / 4.0,
            self.passed_pawn_max_rank_ours,
            self.passed_pawn_max_rank_theirs,
            # King Safety (8)
            self.king_pawn_shelter_ours,
            self.king_pawn_shelter_theirs,
            self.king_open_files_ours / 3.0,
            self.king_open_files_theirs / 3.0,
            self.king_attacker_count_ours / 5.0,
            self.king_attacker_count_theirs / 5.0,
            float(self.castled_ours),
            float(self.castled_theirs),
            # Piece Activity (8)
            self.mobility_ours,
            self.mobility_theirs,
            self.development_ours / 4.0,
            self.development_theirs / 4.0,
            self.good_bishop_ours,
            self.good_bishop_theirs,
            float(self.connected_rooks_ours),
            float(self.connected_rooks_theirs),
            # Phase / Tempo (6)
            self.game_phase,
            self.move_number,
            self.center_control_ours,
            self.center_control_theirs,
            self.space_advantage,
            float(self.in_check),
            # Castling (4)
            float(self.can_castle_kingside_ours),
            float(self.can_castle_queenside_ours),
            float(self.can_castle_kingside_theirs),
            float(self.can_castle_queenside_theirs),
            # Yami-specific (15)
            self.nav_aggression / 1.0,
            self.nav_piece_domain / 1.0,
            self.nav_complexity / 1.0,
            self.nav_initiative / 1.0,
            self.nav_king_pressure / 1.0,
            self.nav_phase / 1.0,
            self.som_convergence,
            self.som_trajectory_trend,
            self.strategy_match_strength,
            self.opp_tactical_skill,
            self.opp_aggressiveness,
            self.opp_consistency,
            self.opp_pressure_rate,
            self.opp_risk_tolerance,
            self.censor_stack_rejection_rate,
        ]

    @staticmethod
    def vector_dim() -> int:
        return 58


def extract_position_signals(board: chess.Board) -> PositionSignals:
    """Compute all per-position signals from a board state."""
    our_color = board.turn
    opp_color = not our_color
    sig = PositionSignals()

    # --- Material ---
    our_mat = _material(board, our_color)
    opp_mat = _material(board, opp_color)
    total = our_mat + opp_mat
    sig.material_balance = (our_mat - opp_mat) / 975.0  # normalize by queen
    sig.total_material = min(total / 7800.0, 1.0)  # 7800 = full starting material
    sig.material_imbalance = _material_imbalance(board, our_color)
    sig.bishop_pair_ours = _has_bishop_pair(board, our_color)
    sig.bishop_pair_theirs = _has_bishop_pair(board, opp_color)

    # --- Pawn Structure ---
    our_pawns = board.pieces(chess.PAWN, our_color)
    opp_pawns = board.pieces(chess.PAWN, opp_color)
    sig.pawn_islands_ours = _pawn_islands(our_pawns)
    sig.pawn_islands_theirs = _pawn_islands(opp_pawns)
    sig.isolated_pawns_ours = _isolated_pawns(our_pawns)
    sig.isolated_pawns_theirs = _isolated_pawns(opp_pawns)
    sig.doubled_pawns_ours = _doubled_pawns(our_pawns)
    sig.doubled_pawns_theirs = _doubled_pawns(opp_pawns)
    sig.backward_pawns_ours = _backward_pawns(board, our_color)
    sig.backward_pawns_theirs = _backward_pawns(board, opp_color)
    sig.passed_pawns_ours, sig.passed_pawn_max_rank_ours = _passed_pawns(
        board, our_color
    )
    sig.passed_pawns_theirs, sig.passed_pawn_max_rank_theirs = _passed_pawns(
        board, opp_color
    )

    # --- King Safety ---
    our_king_sq = board.king(our_color)
    opp_king_sq = board.king(opp_color)
    if our_king_sq is not None:
        sig.king_pawn_shelter_ours = _pawn_shelter(board, our_king_sq, our_color)
        sig.king_open_files_ours = _king_open_files(board, our_king_sq)
        sig.king_attacker_count_ours = _king_attackers(board, our_king_sq, opp_color)
    if opp_king_sq is not None:
        sig.king_pawn_shelter_theirs = _pawn_shelter(board, opp_king_sq, opp_color)
        sig.king_open_files_theirs = _king_open_files(board, opp_king_sq)
        sig.king_attacker_count_theirs = _king_attackers(board, opp_king_sq, our_color)

    # Castling detection (heuristic: king not on starting square)
    our_king_start = chess.E1 if our_color == chess.WHITE else chess.E8
    opp_king_start = chess.E1 if opp_color == chess.WHITE else chess.E8
    sig.castled_ours = our_king_sq is not None and our_king_sq != our_king_start
    sig.castled_theirs = opp_king_sq is not None and opp_king_sq != opp_king_start

    # --- Piece Activity ---
    sig.mobility_ours = min(board.legal_moves.count() / 40.0, 1.0)
    # Approximate opponent mobility without changing turn
    sig.mobility_theirs = _approx_opponent_mobility(board, opp_color) / 40.0
    sig.development_ours = _development_count(board, our_color)
    sig.development_theirs = _development_count(board, opp_color)
    sig.good_bishop_ours = _good_bishop_score(board, our_color)
    sig.good_bishop_theirs = _good_bishop_score(board, opp_color)
    sig.connected_rooks_ours = _connected_rooks(board, our_color)
    sig.connected_rooks_theirs = _connected_rooks(board, opp_color)

    # --- Phase / Tempo ---
    sig.game_phase = _game_phase(board)
    sig.move_number = min(board.fullmove_number / 100.0, 1.0)
    sig.center_control_ours = _center_control(board, our_color)
    sig.center_control_theirs = _center_control(board, opp_color)
    sig.space_advantage = _space_advantage(board, our_color)
    sig.in_check = board.is_check()

    # --- Castling Rights ---
    sig.can_castle_kingside_ours = board.has_kingside_castling_rights(our_color)
    sig.can_castle_queenside_ours = board.has_queenside_castling_rights(our_color)
    sig.can_castle_kingside_theirs = board.has_kingside_castling_rights(opp_color)
    sig.can_castle_queenside_theirs = board.has_queenside_castling_rights(opp_color)

    return sig


# --- Helper Functions ---

def _material(board: chess.Board, color: chess.Color) -> int:
    """Total material for one side in centipawns."""
    return sum(
        PIECE_VALUES.get(p.piece_type, 0)
        for p in board.piece_map().values()
        if p.color == color
    )


def _has_bishop_pair(board: chess.Board, color: chess.Color) -> bool:
    """Does this side have both a light-squared and dark-squared bishop?"""
    bishops = board.pieces(chess.BISHOP, color)
    if len(bishops) < 2:
        return False
    colors = {chess.square_rank(sq) % 2 == chess.square_file(sq) % 2 for sq in bishops}
    return len(colors) == 2


def _material_imbalance(board: chess.Board, color: chess.Color) -> float:
    """Kaufman-style material imbalance score.

    Key interactions: knight value increases with more pawns,
    rook value decreases with more pawns, bishop pair bonus.
    """
    opp = not color
    our_pawns = len(board.pieces(chess.PAWN, color))
    opp_pawns = len(board.pieces(chess.PAWN, opp))
    our_knights = len(board.pieces(chess.KNIGHT, color))
    our_rooks = len(board.pieces(chess.ROOK, color))

    imbalance = 0.0
    # Knights gain ~5cp per pawn on the board
    imbalance += our_knights * (our_pawns + opp_pawns - 10) * 5 / 975.0
    # Rooks lose ~5cp per pawn on the board
    imbalance -= our_rooks * (our_pawns + opp_pawns - 10) * 5 / 975.0
    # Bishop pair
    if _has_bishop_pair(board, color):
        imbalance += 50 / 975.0
    return imbalance


def _pawn_islands(pawns: chess.SquareSet) -> int:
    """Count pawn islands (groups of connected files with pawns)."""
    if not pawns:
        return 0
    files = sorted({chess.square_file(sq) for sq in pawns})
    islands = 1
    for i in range(1, len(files)):
        if files[i] - files[i - 1] > 1:
            islands += 1
    return islands


def _isolated_pawns(pawns: chess.SquareSet) -> int:
    """Count pawns with no friendly pawns on adjacent files."""
    files_with_pawns = {chess.square_file(sq) for sq in pawns}
    count = 0
    for sq in pawns:
        f = chess.square_file(sq)
        if (f - 1) not in files_with_pawns and (f + 1) not in files_with_pawns:
            count += 1
    return count


def _doubled_pawns(pawns: chess.SquareSet) -> int:
    """Count files with more than one pawn."""
    from collections import Counter
    files = Counter(chess.square_file(sq) for sq in pawns)
    return sum(v - 1 for v in files.values() if v > 1)


def _backward_pawns(board: chess.Board, color: chess.Color) -> int:
    """Count backward pawns — can't be supported by adjacent pawns."""
    our_pawns = board.pieces(chess.PAWN, color)
    files_with_pawns = {chess.square_file(sq) for sq in our_pawns}
    count = 0
    for sq in our_pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        # Check if any friendly pawn on adjacent files is behind or equal
        has_support = False
        for adj_f in [f - 1, f + 1]:
            if adj_f in files_with_pawns:
                for adj_sq in our_pawns:
                    if chess.square_file(adj_sq) == adj_f:
                        adj_r = chess.square_rank(adj_sq)
                        if color == chess.WHITE and adj_r <= r:
                            has_support = True
                        elif color == chess.BLACK and adj_r >= r:
                            has_support = True
        if not has_support and ((f - 1) in files_with_pawns or (f + 1) in files_with_pawns):
            # Only backward if adjacent files HAVE pawns but they're all ahead
            count += 1
    return count


def _passed_pawns(
    board: chess.Board, color: chess.Color
) -> tuple[int, float]:
    """Count passed pawns and max advancement rank (0-1)."""
    our_pawns = board.pieces(chess.PAWN, color)
    opp_pawns = board.pieces(chess.PAWN, not color)
    count = 0
    max_rank = 0.0

    for sq in our_pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        is_passed = True

        for opp_sq in opp_pawns:
            opp_f = chess.square_file(opp_sq)
            opp_r = chess.square_rank(opp_sq)
            if abs(opp_f - f) <= 1:
                if color == chess.WHITE and opp_r > r:
                    is_passed = False
                    break
                elif color == chess.BLACK and opp_r < r:
                    is_passed = False
                    break

        if is_passed:
            count += 1
            advancement = r / 7.0 if color == chess.WHITE else (7 - r) / 7.0
            max_rank = max(max_rank, advancement)

    return count, max_rank


def _pawn_shelter(
    board: chess.Board, king_sq: chess.Square, color: chess.Color
) -> float:
    """Pawn shelter quality: how well pawns protect the king. 0=none, 1=full."""
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    our_pawns = board.pieces(chess.PAWN, color)

    shelter = 0
    checked_files = 0
    for f in [king_file - 1, king_file, king_file + 1]:
        if 0 <= f <= 7:
            checked_files += 1
            # Look for a pawn on this file near the king
            for sq in our_pawns:
                if chess.square_file(sq) == f:
                    pr = chess.square_rank(sq)
                    dist = abs(pr - king_rank)
                    if dist <= 2:
                        shelter += 1
                        break

    return shelter / max(checked_files, 1)


def _king_open_files(board: chess.Board, king_sq: chess.Square) -> int:
    """Count open files adjacent to the king."""
    king_file = chess.square_file(king_sq)
    open_count = 0
    for f in [king_file - 1, king_file, king_file + 1]:
        if 0 <= f <= 7:
            file_mask = chess.BB_FILES[f]
            white_pawns = int(board.pieces(chess.PAWN, chess.WHITE)) & file_mask
            black_pawns = int(board.pieces(chess.PAWN, chess.BLACK)) & file_mask
            if not white_pawns and not black_pawns:
                open_count += 1
    return open_count


def _king_attackers(
    board: chess.Board, king_sq: chess.Square, attacker_color: chess.Color
) -> int:
    """Count pieces of attacker_color attacking squares in the king zone."""
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    attackers = 0

    for df in [-1, 0, 1]:
        for dr in [-1, 0, 1]:
            f, r = king_file + df, king_rank + dr
            if 0 <= f <= 7 and 0 <= r <= 7:
                sq = chess.square(f, r)
                attack_mask = board.attackers_mask(attacker_color, sq)
                # Count unique attacking pieces (not pawns for efficiency)
                for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                    if attack_mask & int(board.pieces(piece_type, attacker_color)):
                        attackers += 1

    return min(attackers, 10)


def _approx_opponent_mobility(board: chess.Board, opp_color: chess.Color) -> int:
    """Approximate opponent mobility without flipping the turn.

    Counts squares attacked by opponent pieces as a proxy.
    """
    attacked = 0
    for sq in chess.SQUARES:
        if board.is_attacked_by(opp_color, sq):
            attacked += 1
    return attacked


def _development_count(board: chess.Board, color: chess.Color) -> int:
    """Count minor pieces (N, B) that have moved from the back rank."""
    back_rank = 0 if color == chess.WHITE else 7
    developed = 0
    for piece_type in [chess.KNIGHT, chess.BISHOP]:
        for sq in board.pieces(piece_type, color):
            if chess.square_rank(sq) != back_rank:
                developed += 1
    return developed


def _good_bishop_score(board: chess.Board, color: chess.Color) -> float:
    """How 'good' is the bishop? 1.0 = all own pawns on opposite color."""
    bishops = board.pieces(chess.BISHOP, color)
    if not bishops:
        return 0.0

    our_pawns = board.pieces(chess.PAWN, color)
    if not our_pawns:
        return 1.0  # no pawns = bishop is unrestricted

    total_score = 0.0
    for bsq in bishops:
        bishop_dark = (chess.square_rank(bsq) + chess.square_file(bsq)) % 2 == 0
        same_color = sum(
            1 for psq in our_pawns
            if ((chess.square_rank(psq) + chess.square_file(psq)) % 2 == 0) == bishop_dark
        )
        total_score += 1.0 - same_color / max(len(our_pawns), 1)

    return total_score / len(bishops)


def _connected_rooks(board: chess.Board, color: chess.Color) -> bool:
    """Are the two rooks on the same rank/file with nothing between?"""
    rooks = list(board.pieces(chess.ROOK, color))
    if len(rooks) < 2:
        return False

    r1, r2 = rooks[0], rooks[1]
    f1, f2 = chess.square_file(r1), chess.square_file(r2)
    rk1, rk2 = chess.square_rank(r1), chess.square_rank(r2)

    if f1 == f2:
        # Same file — check no pieces between
        lo, hi = min(rk1, rk2), max(rk1, rk2)
        for r in range(lo + 1, hi):
            if board.piece_at(chess.square(f1, r)) is not None:
                return False
        return True
    elif rk1 == rk2:
        # Same rank — check no pieces between
        lo, hi = min(f1, f2), max(f1, f2)
        for f in range(lo + 1, hi):
            if board.piece_at(chess.square(f, rk1)) is not None:
                return False
        return True
    return False


def _game_phase(board: chess.Board) -> float:
    """Game phase from material: 1.0=opening, 0.0=endgame."""
    total = sum(
        PIECE_VALUES.get(p.piece_type, 0)
        for p in board.piece_map().values()
        if p.piece_type != chess.KING
    )
    # Starting non-king material is ~7800cp
    return min(total / 6000.0, 1.0)


def _center_control(board: chess.Board, color: chess.Color) -> float:
    """Fraction of center squares attacked by this color."""
    controlled = sum(1 for sq in CENTER if board.is_attacked_by(color, sq))
    return controlled / 4.0


def _space_advantage(board: chess.Board, color: chess.Color) -> float:
    """Count safe squares on the opponent's half (space advantage)."""
    opp_color = not color
    safe_count = 0
    # Our "advanced" territory: ranks 5-8 for white, ranks 1-4 for black
    if color == chess.WHITE:
        target_ranks = range(4, 8)
    else:
        target_ranks = range(0, 4)

    for sq in chess.SQUARES:
        if chess.square_rank(sq) in target_ranks:
            # Square is "safe" if we attack it and opponent doesn't
            if board.is_attacked_by(color, sq) and not board.is_attacked_by(opp_color, sq):
                safe_count += 1

    return min(safe_count / 16.0, 1.0)
