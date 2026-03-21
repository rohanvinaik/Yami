"""Strategy Library — Pre-encoded chess strategies as multi-move patterns.

Curated from chess theory, not learned. Each strategy is a multi-move
navigation vector with activation conditions and expected move sequences.
Like K-lines but deterministic and from the chess canon.

Queryable by nav_vector + anchor match via Jaccard similarity.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

from yami.navigator import NavigationVector
from yami.tactical_scoper import PIECE_VALUES


@dataclass
class Strategy:
    """A pre-encoded multi-move chess strategy."""

    name: str
    description: str
    # Navigation profile this strategy applies to
    nav_profile: tuple[int, ...]  # 6-bank ternary vector
    # Anchors that must be active for this strategy
    required_anchors: frozenset[str]
    # Anchors that boost this strategy's score
    preferred_anchors: frozenset[str]
    # Multi-move sequence (SAN patterns, not exact moves)
    move_patterns: list[str]  # e.g., ["f4", "f5", "fxg6"] or piece-type hints
    # What piece types to prioritize
    piece_priority: list[int]  # chess piece types to move
    # Target squares (files/ranks to aim for)
    target_files: list[int]  # 0-7
    target_ranks: list[int]  # 0-7
    # Conditions
    min_material_advantage: int = -500  # can activate even when behind
    phase: str = "middlegame"  # "opening", "middlegame", "endgame"
    expected_moves: int = 5


# --- Strategy Library ---

STRATEGY_LIBRARY: list[Strategy] = [
    # === ATTACKING STRATEGIES ===
    Strategy(
        name="kingside_pawn_storm",
        description="Advance kingside pawns to open lines against castled king",
        nav_profile=(1, -1, 0, 1, 1, 0),
        required_anchors=frozenset({"pawn-break"}),
        preferred_anchors=frozenset({"king-safety", "open-file", "pawn-storm"}),
        move_patterns=["h4", "g4", "h5", "g5"],
        piece_priority=[chess.PAWN, chess.ROOK],
        target_files=[5, 6, 7],  # f, g, h files
        target_ranks=[4, 5, 6, 7] if True else [],  # ranks 5-8 (push forward)
        phase="middlegame",
        expected_moves=6,
    ),
    Strategy(
        name="queenside_minority_attack",
        description="Use fewer pawns on queenside to create weaknesses in opponent's structure",
        nav_profile=(1, -1, 0, 1, 0, 0),
        required_anchors=frozenset({"pawn-break"}),
        preferred_anchors=frozenset({"minority-attack", "open-file", "half-open-file"}),
        move_patterns=["b4", "b5", "bxa6", "bxc6"],
        piece_priority=[chess.PAWN, chess.ROOK],
        target_files=[0, 1, 2],  # a, b, c files
        target_ranks=[3, 4, 5],
        phase="middlegame",
        expected_moves=5,
    ),
    Strategy(
        name="central_breakthrough",
        description="Push d or e pawn to break open the center",
        nav_profile=(0, -1, 0, 1, 0, 0),
        required_anchors=frozenset({"center-control"}),
        preferred_anchors=frozenset({"pawn-break", "open-file", "piece-coordination"}),
        move_patterns=["d5", "e5", "d4", "e4"],
        piece_priority=[chess.PAWN, chess.KNIGHT, chess.BISHOP],
        target_files=[3, 4],  # d, e files
        target_ranks=[3, 4, 5],
        phase="middlegame",
        expected_moves=4,
    ),
    Strategy(
        name="piece_sacrifice_attack",
        description="Sacrifice a piece to open lines against the king",
        nav_profile=(1, 1, 1, 1, 1, 0),
        required_anchors=frozenset({"sacrifice"}),
        preferred_anchors=frozenset({"king-hunt", "tempo-gain", "open-file"}),
        move_patterns=[],  # no fixed pattern — any sacrifice
        piece_priority=[chess.KNIGHT, chess.BISHOP],
        target_files=[],
        target_ranks=[],
        min_material_advantage=-100,  # don't sacrifice when already behind
        phase="middlegame",
        expected_moves=4,
    ),
    Strategy(
        name="rook_lift",
        description="Bring rook to 3rd/4th rank and swing to kingside",
        nav_profile=(1, 1, 0, 1, 1, 0),
        required_anchors=frozenset(),
        preferred_anchors=frozenset({"rook-on-open-file", "piece-coordination"}),
        move_patterns=["Ra3", "Rg3", "Rh3"],
        piece_priority=[chess.ROOK],
        target_files=[5, 6, 7],
        target_ranks=[2, 3],  # 3rd and 4th ranks
        phase="middlegame",
        expected_moves=3,
    ),

    # === POSITIONAL STRATEGIES ===
    Strategy(
        name="outpost_occupation",
        description="Place a knight on a strong outpost square",
        nav_profile=(0, 1, 0, 0, 0, 0),
        required_anchors=frozenset({"outpost"}),
        preferred_anchors=frozenset({"knight-outpost", "piece-coordination"}),
        move_patterns=["Nd5", "Ne5", "Nf5", "Nc5"],
        piece_priority=[chess.KNIGHT],
        target_files=[2, 3, 4, 5],
        target_ranks=[3, 4, 5],
        phase="middlegame",
        expected_moves=3,
    ),
    Strategy(
        name="bishop_pair_exploitation",
        description="Open the position to maximize bishop pair advantage",
        nav_profile=(0, 1, 0, 1, 0, 0),
        required_anchors=frozenset({"bishop-pair"}),
        preferred_anchors=frozenset({"open-file", "pawn-break", "space-advantage"}),
        move_patterns=[],
        piece_priority=[chess.BISHOP],
        target_files=[],
        target_ranks=[],
        phase="middlegame",
        expected_moves=5,
    ),
    Strategy(
        name="double_rooks_on_file",
        description="Place both rooks on an open or semi-open file",
        nav_profile=(0, 1, -1, 1, 0, 0),
        required_anchors=frozenset({"open-file"}),
        preferred_anchors=frozenset({"rook-on-open-file", "connected-rooks", "rook-on-seventh"}),
        move_patterns=[],
        piece_priority=[chess.ROOK],
        target_files=[],
        target_ranks=[],
        phase="middlegame",
        expected_moves=4,
    ),

    # === ENDGAME STRATEGIES ===
    Strategy(
        name="passed_pawn_advance",
        description="Create and advance a passed pawn toward promotion",
        nav_profile=(0, -1, -1, 1, 0, -1),
        required_anchors=frozenset({"passed-pawn"}),
        preferred_anchors=frozenset({"pawn-race", "endgame-technique"}),
        move_patterns=[],
        piece_priority=[chess.PAWN, chess.KING],
        target_files=[],
        target_ranks=[5, 6, 7],  # push toward promotion
        phase="endgame",
        expected_moves=6,
    ),
    Strategy(
        name="king_centralization",
        description="Activate the king in the endgame by marching toward the center",
        nav_profile=(0, 0, -1, 0, 0, -1),
        required_anchors=frozenset({"king-march"}),
        preferred_anchors=frozenset({"king-activity", "opposition", "endgame-technique"}),
        move_patterns=["Kd2", "Ke3", "Kd4", "Ke4"],
        piece_priority=[chess.KING],
        target_files=[2, 3, 4, 5],  # central files
        target_ranks=[2, 3, 4, 5],  # central ranks
        phase="endgame",
        expected_moves=5,
    ),
    Strategy(
        name="rook_behind_passed_pawn",
        description="Place rook behind a passed pawn (yours or opponent's)",
        nav_profile=(0, 1, -1, 0, 0, -1),
        required_anchors=frozenset({"passed-pawn"}),
        preferred_anchors=frozenset({"rook-on-open-file", "endgame-technique"}),
        move_patterns=[],
        piece_priority=[chess.ROOK],
        target_files=[],
        target_ranks=[],
        phase="endgame",
        expected_moves=3,
    ),

    # === FAMOUS GAME PATTERNS (Initiative-Creating) ===
    Strategy(
        name="greek_gift_sacrifice",
        description="Sacrifice bishop on h7 to expose castled king (Bxh7+)",
        nav_profile=(1, 1, 1, 1, 1, 0),
        required_anchors=frozenset({"sacrifice"}),
        preferred_anchors=frozenset({"king-hunt", "tempo-gain", "back-rank-threat"}),
        move_patterns=["Bxh7+", "Ng5", "Qh5"],
        piece_priority=[chess.BISHOP, chess.KNIGHT, chess.QUEEN],
        target_files=[6, 7],  # g, h files
        target_ranks=[5, 6, 7],
        phase="middlegame",
        expected_moves=5,
    ),
    Strategy(
        name="exchange_sacrifice_on_c3",
        description="Sacrifice exchange on c3 to destroy pawn structure (Petrosian style)",
        nav_profile=(0, 1, 1, 1, 0, 0),
        required_anchors=frozenset(),
        preferred_anchors=frozenset({"sacrifice", "pawn-break", "piece-coordination"}),
        move_patterns=["Rxc3"],
        piece_priority=[chess.ROOK],
        target_files=[2],  # c file
        target_ranks=[2],  # 3rd rank
        phase="middlegame",
        expected_moves=4,
    ),
    Strategy(
        name="knight_sacrifice_f7",
        description="Sacrifice knight on f7 to fork king and queen or open lines",
        nav_profile=(1, 1, -1, 1, 1, 0),
        required_anchors=frozenset(),
        preferred_anchors=frozenset({"sacrifice", "fork", "tempo-gain"}),
        move_patterns=["Nxf7"],
        piece_priority=[chess.KNIGHT],
        target_files=[5],  # f file
        target_ranks=[6],  # 7th rank
        phase="middlegame",
        expected_moves=3,
    ),
    Strategy(
        name="queenside_expansion",
        description="Expand on queenside with a4-a5 and b4-b5 pawn chain",
        nav_profile=(0, -1, 0, 1, 0, 0),
        required_anchors=frozenset(),
        preferred_anchors=frozenset({"pawn-break", "space-advantage", "minority-attack"}),
        move_patterns=["a4", "a5", "b4", "b5"],
        piece_priority=[chess.PAWN],
        target_files=[0, 1],  # a, b files
        target_ranks=[3, 4, 5],
        phase="middlegame",
        expected_moves=5,
    ),
    Strategy(
        name="fianchetto_pressure",
        description="Fianchetto bishop to put long-diagonal pressure on enemy position",
        nav_profile=(0, 1, 0, 0, 0, 1),
        required_anchors=frozenset(),
        preferred_anchors=frozenset({"fianchetto", "center-control", "development"}),
        move_patterns=["g3", "Bg2", "b3", "Bb2"],
        piece_priority=[chess.BISHOP, chess.PAWN],
        target_files=[1, 6],  # b, g files
        target_ranks=[1, 2],
        phase="opening",
        expected_moves=3,
    ),
    Strategy(
        name="zwischenzug_tempo",
        description="Interpose a forcing move before completing an exchange",
        nav_profile=(0, 0, -1, 1, 0, 0),
        required_anchors=frozenset(),
        preferred_anchors=frozenset({"tempo-gain", "zwischenzug"}),
        move_patterns=[],
        piece_priority=[],
        target_files=[],
        target_ranks=[],
        phase="middlegame",
        expected_moves=2,
    ),
    Strategy(
        name="king_walk_attack",
        description="Use king as attacking piece in endgame (Brenin/Short style)",
        nav_profile=(1, 0, 1, 1, 0, -1),
        required_anchors=frozenset({"king-march"}),
        preferred_anchors=frozenset({"king-activity", "endgame-technique"}),
        move_patterns=[],
        piece_priority=[chess.KING],
        target_files=[3, 4, 5],
        target_ranks=[4, 5, 6],
        phase="endgame",
        expected_moves=6,
    ),

    # === DEFENSIVE STRATEGIES ===
    Strategy(
        name="fortress_construction",
        description="Build an impenetrable defensive structure",
        nav_profile=(-1, 0, -1, -1, -1, 0),
        required_anchors=frozenset({"fortress"}),
        preferred_anchors=frozenset({"blockade", "king-safety", "prophylaxis"}),
        move_patterns=[],
        piece_priority=[chess.KING, chess.PAWN],
        target_files=[],
        target_ranks=[],
        min_material_advantage=-900,  # even when way behind
        phase="middlegame",
        expected_moves=5,
    ),
    Strategy(
        name="exchange_simplification",
        description="Trade pieces to reach a drawn or winning endgame",
        nav_profile=(0, 1, -1, 0, 0, 0),
        required_anchors=frozenset({"simplification"}),
        preferred_anchors=frozenset({"piece-exchange", "endgame-technique"}),
        move_patterns=[],
        piece_priority=[chess.QUEEN, chess.ROOK],
        target_files=[],
        target_ranks=[],
        min_material_advantage=100,  # only when ahead
        phase="middlegame",
        expected_moves=3,
    ),
]


def query_strategies(
    board: chess.Board,
    nav_vector: NavigationVector,
    active_anchors: set[str],
    top_k: int = 3,
) -> list[tuple[Strategy, float]]:
    """Find strategies that match the current position.

    Returns list of (strategy, match_score) sorted by score descending.
    """
    # Material balance
    our_color = board.turn
    our_mat = sum(
        PIECE_VALUES.get(p.piece_type, 0)
        for p in board.piece_map().values()
        if p.color == our_color
    )
    opp_mat = sum(
        PIECE_VALUES.get(p.piece_type, 0)
        for p in board.piece_map().values()
        if p.color != our_color
    )
    material_balance = our_mat - opp_mat

    nv = nav_vector.as_tuple()
    scored: list[tuple[Strategy, float]] = []

    for strat in STRATEGY_LIBRARY:
        # Material condition
        if material_balance < strat.min_material_advantage:
            continue

        # Phase condition
        if strat.phase == "endgame" and nav_vector.phase != -1:
            continue
        if strat.phase == "opening" and nav_vector.phase != 1:
            continue

        score = 0.0

        # Navigation vector alignment (Hamming similarity)
        matches = sum(
            a == b for a, b in zip(nv, strat.nav_profile, strict=False)
            if b != 0  # only count non-zero entries
        )
        non_zero = sum(1 for b in strat.nav_profile if b != 0)
        if non_zero > 0:
            score += 3.0 * matches / non_zero

        # Required anchors (must all be present)
        if strat.required_anchors and not strat.required_anchors.issubset(active_anchors):
            continue

        # Preferred anchor overlap (Jaccard-like)
        if strat.preferred_anchors:
            overlap = len(active_anchors & strat.preferred_anchors)
            score += 2.0 * overlap / max(len(strat.preferred_anchors), 1)

        # Bonus for matching piece types being available
        for pt in strat.piece_priority[:2]:
            if board.pieces(pt, board.turn):
                score += 0.5

        if score > 1.0:
            scored.append((strat, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def score_move_against_strategy(
    board: chess.Board,
    move: chess.Move,
    strategy: Strategy,
) -> float:
    """Score how well a specific move aligns with a strategy."""
    piece = board.piece_at(move.from_square)
    if piece is None:
        return 0.0

    score = 0.0

    # Piece type alignment
    if piece.piece_type in strategy.piece_priority:
        idx = strategy.piece_priority.index(piece.piece_type)
        score += 2.0 - idx * 0.5  # first priority = 2.0, second = 1.5

    # Target file alignment
    to_file = chess.square_file(move.to_square)
    if strategy.target_files and to_file in strategy.target_files:
        score += 1.5

    # Target rank alignment
    to_rank = chess.square_rank(move.to_square)
    if board.turn == chess.BLACK:
        to_rank = 7 - to_rank  # normalize for black
    if strategy.target_ranks and to_rank in strategy.target_ranks:
        score += 1.0

    # Move pattern match
    try:
        san = board.san(move)
        for pattern in strategy.move_patterns:
            if pattern in san:
                score += 3.0
                break
    except (chess.InvalidMoveError, chess.IllegalMoveError):
        pass

    return score
