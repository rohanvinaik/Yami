"""Deterministic game → event-sentence renderer — the chess instrument's Stage-1b input authoring.

Turns a real game into the clean who-does-what prose Genesis parses, with NO chess judgment of our
own: every event verb is licensed by Yami's own deterministic machinery.
  • TACTICAL events ← `tactical_scoper.scope_moves` motifs (capture/check/checkmate/fork/pin) + SEE
    (a sacrifice is a move whose static exchange gives up material — Yami's number, not our opinion).
  • POSITIONAL events ← `position_signals.extract_position_signals` BEFORE/AFTER deltas (the same
    dimensional-delta decomposition the SOM data pipeline uses). A move that drops the opponent's
    mobility "restricts"; one that lifts our development "develops"; one that opens files on their
    king "exposes" it — read off the 58-dim signal, not authored.

Objects are kept to a single bare noun ("file", not "the open file") — the register GSE parses cleanly
(the multi-word/possessive misfire). Verbs are exactly the antecedents of chess_positional.rules /
chess.rules, so the rendered prose fires the chess universes.
"""
from __future__ import annotations

import chess
import chess.pgn

from yami.position_signals import extract_position_signals
from yami.tactical_scoper import scope_moves

_PIECE = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
          chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}

_SAC_CP = -200          # SEE at or below this = material given up = a sacrifice (Yami's number)
_MAX_POSITIONAL = 3     # cap positional events per move (raised to admit the style primitives)


def _mover_view(sig, mover_is_ours: bool) -> dict:
    """Read the 58-dim signal from the MOVER's perspective (extract_position_signals is oriented to
    board.turn, which flips after a push — so we select ours/theirs by who the mover currently is)."""
    o = "ours" if mover_is_ours else "theirs"
    t = "theirs" if mover_is_ours else "ours"
    g = lambda name: getattr(sig, name)  # noqa: E731
    return {
        "dev": g(f"development_{o}"),
        "castled": g(f"castled_{o}"),
        "connected_rooks": g(f"connected_rooks_{o}"),   # piece coordination (harmony)
        "opp_open_files": g(f"king_open_files_{t}"),
        "opp_isolated": g(f"isolated_pawns_{t}"),
        "opp_backward": g(f"backward_pawns_{t}"),
        "opp_mobility": g(f"mobility_{t}"),             # for restraint/prophylaxis
        # space_advantage is signed from board-"ours"; negate when the mover is "theirs"
        "space": sig.space_advantage if mover_is_ours else -sig.space_advantage,
    }


def _file_open(board: chess.Board, file_idx: int) -> bool:
    return not any(
        board.piece_type_at(chess.square(file_idx, r)) == chess.PAWN for r in range(8)
    )


def _tactical(board: chess.Board, move: chess.Move, name: str) -> list[str]:
    scoped = next((s for s in scope_moves(board) if s.move == move), None)
    motifs = set(scoped.motifs) if scoped else set()
    see = scoped.see_value if scoped else 0
    moved = board.piece_type_at(move.from_square)
    captured = board.piece_type_at(move.to_square) if board.is_capture(move) else None
    gives_check = board.gives_check(move)
    board.push(move)
    is_mate = board.is_checkmate()
    board.pop()

    out: list[str] = []
    # NOTE (intentional, by design): SEE catches CAPTURE-sacrifices but a quiet piece-OFFER (e.g.
    # 16.Qb8+!! — a queen left en prise, not capturing) reads only as `checks king`, not `sacrifices`.
    # We do NOT patch this. That miss IS the residual the cross-game frame-of-frames learns from — the
    # per-game frames are for winning; what they cannot see is exactly what the learning layer surfaces.
    if see <= _SAC_CP and moved:                       # gives up material = sacrifice
        out.append(f"{name} sacrifices {_PIECE[moved]}.")
    elif captured:
        out.append(f"{name} captures {_PIECE[captured]}.")
    if "fork" in motifs:
        out.append(f"{name} forks piece.")
    if "pin" in motifs:
        out.append(f"{name} pins piece.")
    if is_mate:
        out.append(f"{name} checkmates king.")
    elif gives_check:
        out.append(f"{name} checks king.")
    return out


def _positional(board: chess.Board, move: chess.Move, name: str) -> list[str]:
    """Cadence (a): DISCRETE positional inflections only — each tied to a concrete move action
    (castle, pawn break, win a file, create a weakness, infiltrate), confirmed by a real signal jump.
    The slow, cumulative registers (restriction, the bind, space squeeze) are NOT per-move events —
    they are the Tier-2 ARC that chess_plan.rules derives OVER this sparse spine. Mobility/center
    jitter is deliberately dropped: it fired every move and carried no discrete meaning."""
    mover = board.turn
    before = _mover_view(extract_position_signals(board), True)
    moved = board.piece_type_at(move.from_square)
    from_rank, to_rank = chess.square_rank(move.from_square), chess.square_rank(move.to_square)
    back_rank = 0 if mover == chess.WHITE else 7
    board.push(move)
    after = _mover_view(extract_position_signals(board), False)
    board.pop()

    events: list[str] = []
    if not before["castled"] and after["castled"]:                       # discrete: king to safety
        events.append(f"{name} defends king.")
    if moved in (chess.KNIGHT, chess.BISHOP) and from_rank == back_rank \
            and after["dev"] - before["dev"] >= 1:                       # a minor leaves the back rank
        events.append(f"{name} develops piece.")
    if moved == chess.ROOK and _file_open(board, chess.square_file(move.to_square)) \
            and chess.square_file(move.from_square) != chess.square_file(move.to_square):
        events.append(f"{name} occupies file.")                          # rook MOVED onto an open file (discrete, not shuffling along it)
    if moved == chess.PAWN and after["space"] - before["space"] >= 0.10:  # a real pawn break
        events.append(f"{name} advances pawn.")
    if after["opp_isolated"] - before["opp_isolated"] >= 1:              # created a weakness
        events.append(f"{name} isolates pawn.")
    if after["opp_backward"] - before["opp_backward"] >= 1:
        events.append(f"{name} weakens structure.")
    if after["opp_open_files"] - before["opp_open_files"] >= 1:          # opened a file on their king
        events.append(f"{name} exposes king.")
    if moved != chess.PAWN and ((mover == chess.WHITE and from_rank < 5 and to_rank >= 5)
                                or (mover == chess.BLACK and from_rank > 2 and to_rank <= 2)):
        events.append(f"{name} infiltrates rank.")                       # a NEW infiltration this move (CROSSED in, not shuffling deep)
    # --- STYLE primitives (the taxonomy's defining frames, from Yami signals) ---
    if not board.is_capture(move) and moved != chess.PAWN \
            and before["opp_mobility"] - after["opp_mobility"] >= 0.12:   # a REAL mobility-curb JUMP (discrete
        events.append(f"{name} restrains opponent.")                     # prophylaxis), not marginal per-move jitter
    if not before["connected_rooks"] and after["connected_rooks"]:       # rooks connect = coordination
        events.append(f"{name} coordinates rooks.")                     # (harmony)
    return events[:_MAX_POSITIONAL]


def render_move(board: chess.Board, move: chess.Move, name: str) -> list[str]:
    """All event-sentences a single move licenses (tactical spine first, then positional)."""
    return _tactical(board, move, name) + _positional(board, move, name)


def _outcome(game: chess.pgn.Game, white: str, black: str) -> list[str]:
    """The RESULT as a stated terminal fact — the outcome the game actually reached. `defeats` names the
    loser so a `cannot` censor can retract the loser's illusory `dominant` (the mirror of brilliancy's
    continuation retracting the naive `fallen`); a draw states the equilibrium, firing no Victory/Tragedy.
    From the PGN Result header — a fact, not a chess judgment of our own."""
    res = game.headers.get("Result", "*")
    if res == "1-0":
        return [f"{white} defeats {black}."]
    if res == "0-1":
        return [f"{black} defeats {white}."]
    if res == "1/2-1/2":
        return [f"{white} draws {black}."]
    return []


def render_game(game: chess.pgn.Game, white: str | None = None, black: str | None = None,
                append_outcome: bool = False) -> list[str]:
    """Render a full game to event-sentences. Names default to the PGN headers, then White/Black.
    `append_outcome=True` states the game's RESULT as a final fact (outcome-integration: the surprise a
    loss needs — the outcome refutes the loser's win-frame). Off by default (positional-lens behavior)."""
    white = white or game.headers.get("White", "White").split(",")[0].split()[-1] or "White"
    black = black or game.headers.get("Black", "Black").split(",")[0].split()[-1] or "Black"
    board = game.board()
    out: list[str] = []
    for move in game.mainline_moves():
        name = white if board.turn == chess.WHITE else black
        out.extend(render_move(board, move, name))
        board.push(move)
    if append_outcome:
        out.extend(_outcome(game, white, black))
    return out
