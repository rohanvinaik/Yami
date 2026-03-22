#!/usr/bin/env python3
"""Generate elite training data focused on WINNING at high ELO.

Four data sources, all targeting decisive moments:

1. Critical positions from Stockfish self-play (eval swings >200cp)
2. Adversarial training: Yami vs high-ELO Stockfish, learning from losses/draws
3. Endgame conversion positions (material advantage → win technique)
4. High-depth Stockfish moves at key moments (depth 20 labeling)

Usage:
    python scripts/generate_elite_data.py --output data/elite_train.jsonl
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
import chess.engine

from yami.candidate_filter import filter_and_annotate
from yami.datagen.contracts import save_dataset
from yami.datagen.feature_extractor import board_to_example
from yami.datagen.label_oracle import label_candidates
from yami.engine import YamiEngine
from yami.oracle import StockfishOracle
from yami.tactical_scoper import (
    apply_blunder_censor,
    apply_repetition_censor,
    apply_tactical_censor,
    scope_moves,
)


def generate_critical_positions(sf, num_positions, label_oracle):
    """Source 1: Positions where eval swings dramatically.

    Play Stockfish vs itself at high depth. Extract positions where
    the evaluation changes by >200cp in 3-4 moves — these are the
    turning points where one specific move creates decisive advantage.
    """
    print("  [Source 1] Critical positions from SF self-play...")
    examples = []
    games = 0

    while len(examples) < num_positions:
        board = chess.Board()
        evals = []
        moves = []
        games += 1

        # Play a game at high depth
        for _ in range(200):
            if board.is_game_over():
                break
            result = sf.play(board, chess.engine.Limit(depth=12, time=0.15))
            move = result.move
            if move is None:
                break

            # Get eval
            info = sf.analyse(board, chess.engine.Limit(depth=10))
            score = info.get("score")
            cp = 0
            if score:
                pov = score.white()
                cp = pov.score() if pov.score() is not None else 0

            evals.append(cp)
            moves.append(move)
            board.push(move)

        # Find eval swing positions (turning points)
        for i in range(3, len(evals)):
            swing = abs(evals[i] - evals[i - 3])
            if swing > 200:  # >2 pawn swing in 3 moves
                # Reconstruct position at the turning point
                replay = chess.Board()
                for m in moves[:i - 2]:
                    replay.push(m)

                ex = _extract_with_deep_label(replay, label_oracle)
                if ex is not None:
                    examples.append(ex)
                    if len(examples) >= num_positions:
                        break

        if games % 10 == 0:
            print(f"    {len(examples)}/{num_positions} (games={games})")

    return examples


def generate_adversarial_positions(sf, num_positions, label_oracle):
    """Source 2: Positions where Yami drew but could have won.

    Play Yami against high-ELO Stockfish. At every position where
    Stockfish has a significant advantage (eval > 100cp from SF's
    perspective), extract the position and label with SF's move.
    This teaches the model "here's what you missed."
    """
    print("  [Source 2] Adversarial positions (Yami vs SF 2000)...")
    examples = []
    games = 0

    yami = YamiEngine(
        use_llm=False, use_neural=False,
        use_navigator=True, use_temporal=True, use_gm_patterns=True,
        use_opening_book=True,
    )

    sf.configure({"Skill Level": 15, "UCI_LimitStrength": True, "UCI_Elo": 2000})

    while len(examples) < num_positions:
        board = chess.Board()
        yami.reset()
        yami.state.board = board
        games += 1
        color = chess.WHITE if games % 2 == 0 else chess.BLACK

        for move_num in range(200):
            if board.is_game_over():
                break

            if board.turn == color:
                # Yami's turn
                dec = yami.decide(board)
                move = dec.move
                if move is None or move not in board.legal_moves:
                    legal = list(board.legal_moves)
                    move = legal[0] if legal else None
            else:
                # Stockfish's turn
                result = sf.play(board, chess.engine.Limit(time=0.1))
                move = result.move

            if move is None:
                break

            # At Yami's moves, check if SF thinks there was a better option
            if board.turn == color and move_num >= 6 and random.random() < 0.4:
                try:
                    info = sf.analyse(board, chess.engine.Limit(depth=15))
                    score = info.get("score")
                    if score:
                        pov = score.relative
                        cp = pov.score() if pov.score() is not None else 0
                        # If position is interesting (not trivially equal)
                        if abs(cp) > 50:
                            ex = _extract_with_deep_label(
                                board, label_oracle
                            )
                            if ex is not None:
                                examples.append(ex)
                                if len(examples) >= num_positions:
                                    break
                except Exception:
                    pass

            board.push(move)

        if games % 5 == 0:
            print(f"    {len(examples)}/{num_positions} (games={games})")

    return examples


def generate_endgame_positions(sf, num_positions, label_oracle):
    """Source 3: Endgame conversion positions.

    Generate positions with clear material advantages and label with
    the winning technique. The model needs to learn CONVERSION.
    """
    print("  [Source 3] Endgame conversion positions...")
    examples = []

    # Common endgame types with material advantage
    endgame_fens = [
        # Rook endgames (most common)
        "8/5k2/8/8/4P3/8/5K2/4R3 w - - 0 1",  # R+P vs K
        "8/5k2/4p3/8/4P3/8/5K2/4R3 w - - 0 1",  # R+P vs K+P
        "8/3k4/8/8/3PK3/8/8/4R3 w - - 0 1",
        "8/8/4k3/8/3PK3/8/8/4R3 w - - 0 1",
        # Queen endgames
        "8/5k2/8/8/4P3/8/5K2/4Q3 w - - 0 1",  # Q+P vs K
        "8/5k2/4p3/8/4P3/8/5K2/4Q3 w - - 0 1",
        # Pawn endgames
        "8/5k2/8/4P3/8/8/5K2/8 w - - 0 1",  # K+P vs K
        "8/3k4/8/3PP3/8/8/3K4/8 w - - 0 1",  # K+2P vs K
        "8/3k4/8/3PP3/8/5p2/3K4/8 w - - 0 1",  # K+2P vs K+P
    ]

    # Also generate random endgame positions
    for _ in range(50):
        board = chess.Board()
        # Clear the board and set up a random endgame
        board.clear()
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))

        # Random king position for black
        black_king_sq = random.choice([
            chess.E8, chess.D7, chess.F7, chess.G8, chess.C6, chess.F6,
        ])
        board.set_piece_at(black_king_sq, chess.Piece(chess.KING, chess.BLACK))

        # Add 1-3 pieces for white
        white_pieces = random.sample(
            [chess.ROOK, chess.QUEEN, chess.BISHOP, chess.KNIGHT, chess.PAWN],
            k=random.randint(1, 2),
        )
        for pt in white_pieces:
            sq = random.choice([
                s for s in chess.SQUARES
                if board.piece_at(s) is None and s != chess.E1 and s != black_king_sq
            ])
            board.set_piece_at(sq, chess.Piece(pt, chess.WHITE))

        if board.is_valid():
            endgame_fens.append(board.fen())

    for fen in endgame_fens:
        board = chess.Board(fen)
        if not board.is_valid() or board.is_game_over():
            continue

        ex = _extract_with_deep_label(board, label_oracle)
        if ex is not None:
            examples.append(ex)

        if len(examples) >= num_positions:
            break

    # Pad with more positions from endgame play
    while len(examples) < num_positions:
        board = chess.Board(random.choice(endgame_fens))
        if not board.is_valid():
            continue

        # Play a few moves to create diverse endgame positions
        for _ in range(random.randint(2, 10)):
            if board.is_game_over():
                break
            result = sf.play(board, chess.engine.Limit(depth=15, time=0.1))
            if result.move is None:
                break
            board.push(result.move)

        if not board.is_game_over():
            ex = _extract_with_deep_label(board, label_oracle)
            if ex is not None:
                examples.append(ex)

    print(f"    {len(examples)}/{num_positions}")
    return examples


def _extract_with_deep_label(board, label_oracle):
    """Extract a training example with deep Stockfish labeling."""
    scoped = scope_moves(board)
    if len(scoped) < 2:
        return None

    censored = apply_blunder_censor(scoped)
    censored = apply_tactical_censor(censored, board)
    censored = apply_repetition_censor(censored, board)
    if not censored or len(censored) < 2:
        return None

    candidates, plan, profile = filter_and_annotate(board, censored)
    if len(candidates) < 2:
        return None

    best_idx, best_cp, second_idx, gap = label_candidates(
        board, candidates, label_oracle
    )

    return board_to_example(
        board,
        best_idx=best_idx,
        oracle_eval_cp=best_cp,
        second_best_idx=second_idx,
        eval_gap_cp=gap,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate elite training data")
    parser.add_argument("--output", default="data/elite_train.jsonl")
    parser.add_argument("--eval-output", default="data/elite_eval.jsonl")
    parser.add_argument("--critical", type=int, default=5000)
    parser.add_argument("--adversarial", type=int, default=5000)
    parser.add_argument("--endgame", type=int, default=2000)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("=== Elite Training Data Generation ===")
    print(f"Critical positions: {args.critical}")
    print(f"Adversarial positions: {args.adversarial}")
    print(f"Endgame positions: {args.endgame}")
    print()

    sf = chess.engine.SimpleEngine.popen_uci("stockfish")
    label_oracle = StockfishOracle(depth=15, time_limit=0.15)

    try:
        t0 = time.time()

        # Source 1: Critical positions
        critical = generate_critical_positions(sf, args.critical, label_oracle)

        # Source 2: Adversarial positions
        adversarial = generate_adversarial_positions(
            sf, args.adversarial, label_oracle
        )

        # Source 3: Endgame positions
        endgame = generate_endgame_positions(sf, args.endgame, label_oracle)

        # Combine and shuffle
        all_examples = critical + adversarial + endgame
        random.shuffle(all_examples)

        # Split train/eval (90/10)
        split = int(len(all_examples) * 0.9)
        train = all_examples[:split]
        eval_set = all_examples[split:]

        save_dataset(train, Path(args.output))
        save_dataset(eval_set, Path(args.eval_output))

        elapsed = time.time() - t0
        print(f"\n=== Done in {elapsed:.0f}s ===")
        print(f"Train: {len(train)} examples → {args.output}")
        print(f"Eval:  {len(eval_set)} examples → {args.eval_output}")
        print(f"Sources: {len(critical)} critical + "
              f"{len(adversarial)} adversarial + {len(endgame)} endgame")

    finally:
        sf.quit()
        label_oracle.close()


if __name__ == "__main__":
    main()
