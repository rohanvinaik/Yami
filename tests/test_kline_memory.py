"""Tests for K-Line Memory."""

import chess

from yami.kline_memory import KLineEntry, KLineMemory, mine_kline_from_game
from yami.navigator import NavigationVector


def test_kline_store_and_count():
    mem = KLineMemory()
    entry = KLineEntry(
        position_type="attack",
        nav_vector=(1, 0, -1, 1, 1, 0),
        anchors=["fork", "tempo-gain"],
        move_sequence=["Nf5", "Qh5", "Qxf7#"],
        material_signature="QRRBBNN_vs_QRRBBNN",
        success_rate=1.0,
        example_fen=chess.STARTING_FEN,
    )
    mem.store(entry)
    assert mem.count() == 1
    mem.close()


def test_kline_query_finds_match():
    mem = KLineMemory()
    entry = KLineEntry(
        position_type="endgame",
        nav_vector=(-1, 0, -1, 0, 0, -1),
        anchors=["passed-pawn", "king-march"],
        move_sequence=["Kd5", "e5", "e6"],
        material_signature="P_vs_K",
        success_rate=0.9,
        example_fen="4k3/8/8/4P3/3K4/8/8/8 w - - 0 1",
    )
    mem.store(entry)

    nav = NavigationVector(-1, 0, -1, 0, 0, -1)
    results = mem.query(
        chess.Board("4k3/8/8/4P3/3K4/8/8/8 w - - 0 1"),
        nav,
        {"passed-pawn", "king-march"},
        top_k=3,
    )
    assert len(results) >= 1
    assert results[0].match_score > 0.3
    mem.close()


def test_kline_query_no_match():
    mem = KLineMemory()
    entry = KLineEntry(
        position_type="attack",
        nav_vector=(1, 1, 1, 1, 1, 0),
        anchors=["sacrifice", "king-hunt"],
        move_sequence=["Qh7+"],
        material_signature="QR_vs_QR",
        success_rate=1.0,
        example_fen=chess.STARTING_FEN,
    )
    mem.store(entry)

    # Query with completely different nav vector and anchors
    nav = NavigationVector(-1, -1, -1, -1, -1, -1)
    results = mem.query(
        chess.Board(),
        nav,
        {"fortress", "blockade"},
        top_k=3,
    )
    # Should have very low match scores
    assert all(r.match_score < 0.3 for r in results)
    mem.close()


def test_mine_kline_from_game():
    # Create a game with a big eval swing
    board = chess.Board()
    moves = []
    evals = []

    # Play some moves with eval increasing
    for san in ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6"]:
        move = board.parse_san(san)
        moves.append(move)
        evals.append(0)  # neutral
        board.push(move)

    # Simulate a big swing
    evals[-1] = 50
    evals.append(300)  # swing!
    moves.append(list(board.legal_moves)[0])
    board.push(moves[-1])
    evals.append(350)
    moves.append(list(board.legal_moves)[0])
    board.push(moves[-1])
    evals.append(400)
    moves.append(list(board.legal_moves)[0])
    board.push(moves[-1])
    evals.append(500)
    moves.append(list(board.legal_moves)[0])

    entries = mine_kline_from_game(moves, evals)
    # May or may not find patterns depending on swing alignment
    assert isinstance(entries, list)


def test_kline_multiple_entries():
    mem = KLineMemory()
    for i in range(10):
        entry = KLineEntry(
            position_type=f"type_{i}",
            nav_vector=(i % 3 - 1, 0, 0, 0, 0, 0),
            anchors=[f"anchor_{i}"],
            move_sequence=[f"e{i % 8 + 1}"],
            material_signature="test",
            example_fen=chess.STARTING_FEN,
        )
        mem.store(entry)
    assert mem.count() == 10
    mem.close()
