#!/usr/bin/env python3
"""Play exhibition games with full move-by-move commentary.

Shows Yami's reasoning at each decision point: navigator classification,
active strategy, coherence signals, and interference patterns.

Usage:
    python scripts/play_exhibition.py --elo 1200 --games 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
import chess.engine

from yami.engine import YamiEngine, DecisionSource
from yami.coherence import compute_coherence
from yami.navigator import compute_navigation_vector, detect_anchors


BANK_NAMES = ["AGG", "PIECE", "CMPLX", "INIT", "KPRES", "PHASE"]
BANK_LABELS = {
    "AGG": {-1: "defensive", 0: "balanced", 1: "attacking"},
    "PIECE": {-1: "pawn play", 0: "mixed", 1: "major pieces"},
    "CMPLX": {-1: "forcing", 0: "standard", 1: "deep combo"},
    "INIT": {-1: "responding", 0: "equal", 1: "dictating"},
    "KPRES": {-1: "king danger!", 0: "both safe", 1: "targeting opp"},
    "PHASE": {-1: "endgame", 0: "middlegame", 1: "opening"},
}


def describe_nav(nav_vector):
    """Human-readable navigation vector."""
    banks = nav_vector.as_tuple()
    parts = []
    for name, val in zip(BANK_NAMES, banks):
        label = BANK_LABELS[name].get(val, "?")
        parts.append(f"{name}={val:+d}({label})")
    return " | ".join(parts)


def piece_count(board):
    """Count material for both sides."""
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9}
    w = sum(vals.get(p.piece_type, 0) for p in board.piece_map().values()
            if p.color == chess.WHITE)
    b = sum(vals.get(p.piece_type, 0) for p in board.piece_map().values()
            if p.color == chess.BLACK)
    return w, b


def play_exhibition_game(yami, sf, sf_elo, sf_skill, yami_color, game_num):
    """Play one game with full commentary."""
    board = chess.Board()
    yami.reset()
    yami.state.board = board

    if sf_elo == 0:
        # Unconstrained mode — full power Stockfish
        sf.configure({
            "Skill Level": 20,
            "UCI_LimitStrength": False,
        })
    else:
        sf.configure({
            "Skill Level": sf_skill,
            "UCI_LimitStrength": True,
            "UCI_Elo": max(sf_elo, 1320),  # SF minimum is 1320
        })

    color_name = "White" if yami_color == chess.WHITE else "Black"
    print(f"\n{'='*70}")
    print(f"  GAME {game_num}: Yami ({color_name}) vs Stockfish ELO {sf_elo}")
    print(f"{'='*70}\n")

    move_num = 0
    moves_san = []

    for ply in range(400):
        if board.is_game_over():
            break

        if board.turn == yami_color:
            # Yami's move — full commentary
            decision = yami.decide(board)
            move = decision.move

            if move is None or move not in board.legal_moves:
                legal = list(board.legal_moves)
                move = legal[0] if legal else None

            if move is None:
                break

            san = board.san(move)

            # Build commentary
            nav = decision.nav_vector
            nav_desc = describe_nav(nav) if nav else "N/A"

            # Source info
            source = decision.source.value

            # Get coherence scores for candidates
            coherence_scores = {}
            if nav and decision.candidates:
                try:
                    coh = compute_coherence(
                        board,
                        [c.move for c in decision.candidates],
                        nav,
                    )
                    for sm in coh.scored_moves:
                        coherence_scores[sm.move.uci()] = sm
                except Exception:
                    pass

            # Candidate info
            cand_info = ""
            if decision.candidates:
                cand_strs = []
                for i, c in enumerate(decision.candidates[:4]):
                    try:
                        c_san = board.san(c.move)
                    except Exception:
                        c_san = c.move.uci()
                    marker = " <--" if c.move == move else ""
                    sig = coherence_scores.get(c.move.uci())
                    if sig:
                        ternary = f"[nav={sig.ternary_navigator:+d} strat={sig.ternary_strategy:+d} som={sig.ternary_temporal:+d} gm={sig.ternary_gm:+d}]"
                        cand_strs.append(f"    {i+1}. {c_san} (final={sig.final_score:.1f}, interf={sig.interference:.1f}) {ternary}{marker}")
                    else:
                        cand_strs.append(f"    {i+1}. {c_san} (align={c.plan_alignment:.2f}){marker}")
                cand_info = "\n".join(cand_strs)

            # Print
            if board.turn == chess.WHITE:
                move_num += 1
                prefix = f"{move_num}. {san}"
            else:
                prefix = f"{move_num}...{san}"

            w_mat, b_mat = piece_count(board)
            mat_str = f"W:{w_mat} B:{b_mat}"

            print(f"  {prefix:12s}  [{source}]  ({mat_str})")
            print(f"    Nav: {nav_desc}")
            if cand_info:
                print(f"    Candidates:")
                print(cand_info)

            # Check for tactical motifs
            anchors = detect_anchors(board, move)
            if anchors:
                print(f"    Motifs: {', '.join(sorted(anchors))}")

            print()
            moves_san.append(san)
            board.push(move)

        else:
            # Stockfish's move — just record it
            result = sf.play(board, chess.engine.Limit(time=0.1))
            move = result.move
            if move is None:
                break

            san = board.san(move)
            if board.turn == chess.WHITE:
                move_num += 1
                prefix = f"{move_num}. {san}"
            else:
                prefix = f"{move_num}...{san}"

            print(f"  {prefix:12s}  [stockfish]")
            moves_san.append(san)
            board.push(move)

    # Game result
    result = board.result()
    yami_won = ((result == "1-0" and yami_color == chess.WHITE)
                or (result == "0-1" and yami_color == chess.BLACK))
    yami_lost = ((result == "0-1" and yami_color == chess.WHITE)
                 or (result == "1-0" and yami_color == chess.BLACK))

    outcome = "WIN" if yami_won else ("LOSS" if yami_lost else "DRAW")
    w_mat, b_mat = piece_count(board)

    print(f"\n  {'='*50}")
    print(f"  Result: {result} — Yami {outcome}")
    print(f"  Moves: {(len(moves_san) + 1) // 2}")
    print(f"  Final material: W:{w_mat} B:{b_mat}")
    if board.is_checkmate():
        print(f"  Checkmate!")
    elif board.is_stalemate():
        print(f"  Stalemate")
    elif board.can_claim_threefold_repetition():
        print(f"  Threefold repetition")
    elif board.is_fifty_moves():
        print(f"  50-move rule")
    print(f"  {'='*50}")

    return outcome


def main():
    parser = argparse.ArgumentParser(description="Exhibition games with commentary")
    parser.add_argument("--elo", type=int, default=1200)
    parser.add_argument("--skill", type=int, default=None)
    parser.add_argument("--games", type=int, default=3)
    args = parser.parse_args()

    # Map ELO to skill level
    skill = args.skill
    if skill is None:
        if args.elo <= 1000:
            skill = 5
        elif args.elo <= 1200:
            skill = 8
        elif args.elo <= 1500:
            skill = 10
        elif args.elo <= 1800:
            skill = 13
        else:
            skill = 15

    yami = YamiEngine(
        use_llm=False, use_neural=False,
        use_navigator=True, use_temporal=True, use_gm_patterns=True,
        use_opening_book=True,
    )

    sf = chess.engine.SimpleEngine.popen_uci("stockfish")

    outcomes = []
    try:
        for i in range(args.games):
            color = chess.WHITE if i % 2 == 0 else chess.BLACK
            outcome = play_exhibition_game(
                yami, sf, args.elo, skill, color, i + 1,
            )
            outcomes.append(outcome)
    finally:
        sf.quit()

    wins = outcomes.count("WIN")
    draws = outcomes.count("DRAW")
    losses = outcomes.count("LOSS")
    print(f"\n{'='*70}")
    print(f"  EXHIBITION RESULTS vs ELO {args.elo}")
    print(f"  {wins}W {draws}D {losses}L across {len(outcomes)} games")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
