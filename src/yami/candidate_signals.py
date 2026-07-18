"""Per-candidate signal extraction — the full feature vector for each move option.

For each candidate move, computes every signal that the fusion model needs
to decide whether this is the right move. The model sees ONLY these signals
plus the position signals — never the raw board.

Signal sources:
  - Navigator OTP alignment
  - SoM specialist agent scores (6 agents)
  - Strategy library alignment
  - GM pattern frequency + win rate
  - K-line memory match
  - Tactical motif flags
  - Look-ahead blunder detection
  - Censor pass/fail
  - Move geometry (centrality, king distance, piece type)
  - Post-move evaluation (material change, mobility, structure)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess

from yami.tactical_scoper import PIECE_VALUES

# Motif vocabulary for one-hot encoding
MOTIF_VOCAB = [
    "capture", "check", "checkmate", "fork", "pin",
    "discovery", "material_gain", "hangs_piece", "promotion",
    "center-control", "development", "pawn-break", "passed-pawn",
    "castling", "king-safety", "rook-on-open-file", "simplification",
]
MOTIF_TO_IDX = {m: i for i, m in enumerate(MOTIF_VOCAB)}


@dataclass
class CandidateSignalVector:
    """Complete signal vector for a single candidate move (~60 dimensions)."""

    # --- Navigator (2) ---
    navigator_otp_score: float = 0.0     # continuous alignment score
    navigator_ternary: int = 0           # {-1, 0, +1}

    # --- SoM Agent Scores (7) ---
    som_tactical: float = 0.0
    som_positional: float = 0.0
    som_endgame: float = 0.0
    som_attack: float = 0.0
    som_defense: float = 0.0
    som_initiative: float = 0.0
    som_convergence: float = 0.0         # how much agents agree on THIS move

    # --- Strategy (2) ---
    strategy_alignment: float = 0.0      # score against active strategy
    strategy_ternary: int = 0

    # --- GM Patterns (3) ---
    gm_frequency: float = 0.0           # how often GMs play this
    gm_win_rate: float = 0.5            # win rate when played
    gm_ternary: int = 0

    # --- K-Line Memory (2) ---
    kline_match_score: float = 0.0      # empirical pattern match
    kline_ternary: int = 0

    # --- Look-ahead (1) ---
    look_ahead_score: float = 0.0       # 2-ply blunder detection

    # --- Censor (1) ---
    censor_pass: bool = True            # did negative learning approve?

    # --- Interference (1) ---
    interference: float = 0.0           # holographic constructive/destructive

    # --- Tactical Motifs (17) ---
    motif_flags: list[int] = field(default_factory=lambda: [0] * len(MOTIF_VOCAB))

    # --- Move Geometry (8) ---
    piece_type: int = 0                 # 0-5 for P/N/B/R/Q/K
    is_capture: bool = False
    is_check: bool = False
    is_castling: bool = False
    see_value: float = 0.0             # static exchange evaluation, normalized
    target_centrality: float = 0.0     # how central is target square [0,1]
    dist_to_opp_king: float = 0.0      # manhattan distance, normalized [0,1]
    dist_to_own_king: float = 0.0

    # --- Post-Move Signals (6) ---
    material_change: float = 0.0       # net material gain/loss, normalized
    resulting_opp_mobility: float = 0.0 # opponent legal moves after
    pawn_structure_change: float = 0.0 # did structure improve/worsen?
    is_outpost: bool = False           # is target an outpost square?
    rook_on_open_file: bool = False    # does this put rook on open file?
    king_tropism: float = 0.0          # piece movement toward opp king

    def to_vector(self) -> list[float]:
        """Flatten to numeric vector for the neural model."""
        return [
            # Navigator (2)
            self.navigator_otp_score / 10.0,
            self.navigator_ternary / 1.0,
            # SoM (7)
            self.som_tactical,
            self.som_positional,
            self.som_endgame,
            self.som_attack,
            self.som_defense,
            self.som_initiative,
            self.som_convergence,
            # Strategy (2)
            self.strategy_alignment / 3.0,
            self.strategy_ternary / 1.0,
            # GM (3)
            self.gm_frequency,
            self.gm_win_rate,
            self.gm_ternary / 1.0,
            # K-line (2)
            self.kline_match_score / 3.0,
            self.kline_ternary / 1.0,
            # Look-ahead (1)
            self.look_ahead_score / 10.0,
            # Censor (1)
            float(self.censor_pass),
            # Interference (1)
            self.interference / 5.0,
            # Motifs (17)
            *[float(f) for f in self.motif_flags],
            # Geometry (8)
            self.piece_type / 5.0,
            float(self.is_capture),
            float(self.is_check),
            float(self.is_castling),
            self.see_value,
            self.target_centrality,
            self.dist_to_opp_king,
            self.dist_to_own_king,
            # Post-move (6)
            self.material_change,
            self.resulting_opp_mobility,
            self.pawn_structure_change,
            float(self.is_outpost),
            float(self.rook_on_open_file),
            self.king_tropism,
        ]

    @staticmethod
    def vector_dim() -> int:
        return 33 + len(MOTIF_VOCAB)  # 33 fixed + 17 motifs = 50


def extract_candidate_signals(
    board: chess.Board,
    move: chess.Move,
    anchors: set[str] | None = None,
    nav_vector=None,
    som_scores: dict[str, float] | None = None,
    strategy_score: float = 0.0,
    gm_freq: float = 0.0,
    gm_winrate: float = 0.5,
    kline_score: float = 0.0,
    look_ahead: float = 0.0,
    censor_pass: bool = True,
    interference: float = 0.0,
    navigator_score: float = 0.0,
    navigator_ternary: int = 0,
    strategy_ternary: int = 0,
    gm_ternary: int = 0,
    kline_ternary: int = 0,
    see_value: int = 0,
) -> CandidateSignalVector:
    """Extract the full signal vector for a single candidate move.

    Most signal values are passed in (computed by the coherence engine).
    This function adds the geometry and post-move signals.
    """
    sig = CandidateSignalVector()

    # Passed-in signals from coherence engine
    sig.navigator_otp_score = navigator_score
    sig.navigator_ternary = navigator_ternary
    sig.strategy_alignment = strategy_score
    sig.strategy_ternary = strategy_ternary
    sig.gm_frequency = gm_freq
    sig.gm_win_rate = gm_winrate
    sig.gm_ternary = gm_ternary
    sig.kline_match_score = kline_score
    sig.kline_ternary = kline_ternary
    sig.look_ahead_score = look_ahead
    sig.censor_pass = censor_pass
    sig.interference = interference

    # SoM agent scores
    if som_scores:
        sig.som_tactical = som_scores.get("tactical", 0.0)
        sig.som_positional = som_scores.get("positional", 0.0)
        sig.som_endgame = som_scores.get("endgame", 0.0)
        sig.som_attack = som_scores.get("attack", 0.0)
        sig.som_defense = som_scores.get("defense", 0.0)
        sig.som_initiative = som_scores.get("initiative", 0.0)

    # Motif flags from anchors
    if anchors:
        for anchor in anchors:
            idx = MOTIF_TO_IDX.get(anchor)
            if idx is not None and idx < len(sig.motif_flags):
                sig.motif_flags[idx] = 1

    # Move geometry
    piece = board.piece_at(move.from_square)
    if piece:
        sig.piece_type = piece.piece_type - 1  # PAWN=1 → 0, KING=6 → 5
    sig.is_capture = board.is_capture(move)
    sig.is_castling = board.is_castling(move)
    sig.see_value = max(-1.0, min(1.0, see_value / 975.0))

    # Target square analysis
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    sig.target_centrality = 1.0 - (abs(to_file - 3.5) + abs(to_rank - 3.5)) / 7.0

    # King distances
    opp_king = board.king(not board.turn)
    own_king = board.king(board.turn)
    if opp_king is not None:
        sig.dist_to_opp_king = 1.0 - chess.square_distance(
            move.to_square, opp_king
        ) / 7.0
    if own_king is not None:
        sig.dist_to_own_king = 1.0 - chess.square_distance(
            move.to_square, own_king
        ) / 7.0

    # King tropism: are we moving CLOSER to opponent king?
    if opp_king is not None and piece:
        old_dist = chess.square_distance(move.from_square, opp_king)
        new_dist = chess.square_distance(move.to_square, opp_king)
        sig.king_tropism = (old_dist - new_dist) / 7.0  # positive = closer

    # Capture detection BEFORE push (on the pre-move board)
    sig.is_capture = board.is_capture(move)

    # Check detection and post-move signals (push/pop)
    board.push(move)
    sig.is_check = board.is_check()
    sig.resulting_opp_mobility = min(board.legal_moves.count() / 40.0, 1.0)
    board.pop()

    # Material change from SEE (already computed pre-push)
    sig.material_change = sig.see_value

    # Outpost detection: target square protected by own pawn, not attackable by opp pawns
    sig.is_outpost = _is_outpost(board, move.to_square, board.turn)

    # Rook on open file
    if piece and piece.piece_type == chess.ROOK:
        file_mask = chess.BB_FILES[to_file]
        no_pawns = (
            not (int(board.pieces(chess.PAWN, chess.WHITE)) & file_mask)
            and not (int(board.pieces(chess.PAWN, chess.BLACK)) & file_mask)
        )
        sig.rook_on_open_file = no_pawns

    return sig


def _is_outpost(board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
    """Is this square an outpost? Protected by own pawn, not attackable by enemy pawns."""
    opp_color = not color
    sq_file = chess.square_file(square)
    sq_rank = chess.square_rank(square)

    # Must be on opponent's half
    if color == chess.WHITE and sq_rank < 4:
        return False
    if color == chess.BLACK and sq_rank > 3:
        return False

    # Check if protected by own pawn
    own_pawns = board.pieces(chess.PAWN, color)
    protected = False
    pawn_attack_from = []
    if color == chess.WHITE:
        pawn_attack_from = [(sq_file - 1, sq_rank - 1), (sq_file + 1, sq_rank - 1)]
    else:
        pawn_attack_from = [(sq_file - 1, sq_rank + 1), (sq_file + 1, sq_rank + 1)]

    for f, r in pawn_attack_from:
        if 0 <= f <= 7 and 0 <= r <= 7:
            if chess.square(f, r) in own_pawns:
                protected = True
                break

    if not protected:
        return False

    # Check that no enemy pawn can attack this square
    opp_pawns = board.pieces(chess.PAWN, opp_color)
    for adj_f in [sq_file - 1, sq_file + 1]:
        if 0 <= adj_f <= 7:
            for opp_sq in opp_pawns:
                if chess.square_file(opp_sq) == adj_f:
                    opp_r = chess.square_rank(opp_sq)
                    if color == chess.WHITE and opp_r > sq_rank:
                        return False  # enemy pawn can advance to attack
                    if color == chess.BLACK and opp_r < sq_rank:
                        return False

    return True
