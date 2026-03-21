"""K-Line Memory — Minsky's knowledge-lines adapted for chess.

Stores winning position patterns as reusable templates. When the current
position matches a stored K-line, the winning move sequence is activated.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import chess

from yami.navigator import NavigationVector, compute_navigation_vector, detect_anchors


@dataclass
class KLineEntry:
    """A stored winning pattern."""

    pattern_id: int = 0
    position_type: str = ""
    nav_vector: tuple[int, ...] = (0, 0, 0, 0, 0, 0)
    anchors: list[str] = field(default_factory=list)
    move_sequence: list[str] = field(default_factory=list)  # SAN moves
    material_signature: str = ""  # e.g., "RR_vs_R"
    success_rate: float = 1.0
    example_fen: str = ""
    match_score: float = 0.0  # populated during query


class KLineMemory:
    """SQLite-backed K-line pattern database."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            self.conn = sqlite3.connect(":memory:")
        else:
            self.conn = sqlite3.connect(str(db_path))
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS klines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_type TEXT,
                nav_vector TEXT,
                anchors TEXT,
                move_sequence TEXT,
                material_signature TEXT,
                success_rate REAL DEFAULT 1.0,
                example_fen TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_nav ON klines(nav_vector);
            CREATE INDEX IF NOT EXISTS idx_type ON klines(position_type);
        """)
        self.conn.commit()

    def store(self, entry: KLineEntry) -> int:
        """Store a K-line pattern. Returns the pattern ID."""
        cur = self.conn.execute(
            """INSERT INTO klines
               (position_type, nav_vector, anchors, move_sequence,
                material_signature, success_rate, example_fen)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.position_type,
                json.dumps(list(entry.nav_vector)),
                json.dumps(entry.anchors),
                json.dumps(entry.move_sequence),
                entry.material_signature,
                entry.success_rate,
                entry.example_fen,
            ),
        )
        self.conn.commit()
        return cur.lastrowid or 0

    def query(
        self,
        board: chess.Board,
        nav_vector: NavigationVector,
        active_anchors: set[str],
        top_k: int = 3,
    ) -> list[KLineEntry]:
        """Find K-lines matching the current position."""
        nv_tuple = nav_vector.as_tuple()
        mat_sig = _material_signature(board)

        rows = self.conn.execute(
            "SELECT * FROM klines"
        ).fetchall()

        scored: list[tuple[float, KLineEntry]] = []
        for row in rows:
            entry = _row_to_entry(row)
            score = _match_score(nv_tuple, entry, active_anchors, mat_sig)
            if score > 0.1:
                entry.match_score = score
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM klines").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self.conn.close()


def _row_to_entry(row: tuple) -> KLineEntry:
    return KLineEntry(
        pattern_id=row[0],
        position_type=row[1],
        nav_vector=tuple(json.loads(row[2])),
        anchors=json.loads(row[3]),
        move_sequence=json.loads(row[4]),
        material_signature=row[5],
        success_rate=row[6],
        example_fen=row[7],
    )


def _match_score(
    query_nv: tuple[int, ...],
    entry: KLineEntry,
    query_anchors: set[str],
    query_mat_sig: str,
) -> float:
    """Score how well a K-line matches the current position."""
    # Navigation vector similarity (inverse Hamming)
    hamming = sum(a != b for a, b in zip(query_nv, entry.nav_vector, strict=False))
    nav_score = max(0, 6 - hamming) / 6.0  # 1.0 = perfect match

    # Anchor overlap (Jaccard)
    entry_anchors = set(entry.anchors)
    if query_anchors or entry_anchors:
        intersection = query_anchors & entry_anchors
        union = query_anchors | entry_anchors
        anchor_score = len(intersection) / max(len(union), 1)
    else:
        anchor_score = 0.0

    # Material signature match bonus
    mat_bonus = 0.3 if query_mat_sig == entry.material_signature else 0.0

    # Combined score
    return (
        0.4 * nav_score
        + 0.4 * anchor_score
        + 0.2 * mat_bonus
    ) * entry.success_rate


def _material_signature(board: chess.Board) -> str:
    """Create a material signature string like 'QRR_vs_QR'."""
    pieces = {chess.WHITE: [], chess.BLACK: []}
    piece_chars = {
        chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B",
        chess.ROOK: "R", chess.QUEEN: "Q",
    }
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.piece_type != chess.KING:
            pieces[p.color].append(piece_chars.get(p.piece_type, "?"))

    w = "".join(sorted(pieces[chess.WHITE], reverse=True)) or "K"
    b = "".join(sorted(pieces[chess.BLACK], reverse=True)) or "K"

    if board.turn == chess.WHITE:
        return f"{w}_vs_{b}"
    return f"{b}_vs_{w}"


def mine_kline_from_game(
    moves: list[chess.Move],
    evals: list[int],
    board: chess.Board | None = None,
) -> list[KLineEntry]:
    """Extract K-line patterns from a game with move-by-move evaluations.

    A K-line is extracted when evaluation swings decisively (>200cp gain
    over 3-5 moves), indicating a winning tactical/strategic sequence.
    """
    if board is None:
        board = chess.Board()

    entries: list[KLineEntry] = []
    window_size = 4

    for i in range(len(moves) - window_size):
        # Check for decisive evaluation swing
        if i >= len(evals) or i + window_size >= len(evals):
            break

        eval_start = evals[i]
        eval_end = evals[i + window_size]
        swing = eval_end - eval_start

        # Decisive swing for the side that moved
        side = chess.WHITE if i % 2 == 0 else chess.BLACK
        if side == chess.BLACK:
            swing = -swing

        if swing > 200:  # >2 pawn advantage gained
            # Reconstruct position at start of sequence
            replay = chess.Board()
            for m in moves[:i]:
                replay.push(m)

            nav = compute_navigation_vector(replay)
            first_move = moves[i]
            anchors = detect_anchors(replay, first_move)

            # Determine position type
            phase = nav.phase
            pos_type = "middlegame"
            if phase == -1:
                pos_type = "endgame"
            elif phase == 1:
                pos_type = "opening"
            if nav.king_pressure == 1:
                pos_type = "attack"

            move_sans = []
            temp = replay.copy()
            for m in moves[i:i + window_size]:
                move_sans.append(temp.san(m))
                temp.push(m)

            entries.append(KLineEntry(
                position_type=pos_type,
                nav_vector=nav.as_tuple(),
                anchors=sorted(anchors),
                move_sequence=move_sans,
                material_signature=_material_signature(replay),
                success_rate=1.0,
                example_fen=replay.fen(),
            ))

    return entries
