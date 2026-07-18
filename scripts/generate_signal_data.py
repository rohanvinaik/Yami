#!/usr/bin/env python3
"""Generate training data with full signal profiles for the fusion model.

Sources:
  A) SF vs SF at different ELOs — mistake curriculum
  B) Random positions from SF self-play — broad coverage
  C) Candidate spread filter — only positions where ranking matters

All output uses the new SignalProfileExample format with full
per-position and per-candidate signal vectors.

Crash-safe: checkpoints after each batch of 100 examples.

Usage:
    python scripts/generate_signal_data.py --num 5000 --output data/signal_train.jsonl
    python scripts/generate_signal_data.py --num 5000 --mode multi-elo
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

from yami.datagen.signal_extractor import (
    SignalProfileExample,
    append_signal_dataset,
    board_to_signal_example,
    save_signal_dataset,
)
from yami.oracle import StockfishOracle
from yami.temporal_controller import TemporalController


def label_with_stockfish(
    board: chess.Board,
    candidate_moves: list[str],
    sf: chess.engine.SimpleEngine,
    depth: int = 15,
) -> tuple[int, int, int]:
    """Label candidates with Stockfish. Returns (best_idx, best_cp, gap_cp)."""
    evals = []
    for uci in candidate_moves:
        if not uci:
            evals.append(-99999)
            continue
        try:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                evals.append(-99999)
                continue
            board.push(move)
            info = sf.analyse(board, chess.engine.Limit(depth=depth, time=0.1))
            score = info.get("score")
            # Negate because we pushed the move (now it's opponent's perspective)
            cp = 0
            if score:
                pov = score.relative
                cp = -(pov.score() if pov.score() is not None else 0)
            evals.append(cp)
            board.pop()
        except Exception:
            evals.append(-99999)

    if not evals or max(evals) == -99999:
        return 0, 0, 0

    best_idx = max(range(len(evals)), key=lambda i: evals[i])
    best_cp = evals[best_idx]
    sorted_evals = sorted([e for e in evals if e > -99999], reverse=True)
    gap = sorted_evals[0] - sorted_evals[1] if len(sorted_evals) >= 2 else 0

    return best_idx, best_cp, gap


def generate_from_sf_games(
    sf: chess.engine.SimpleEngine,
    num_examples: int,
    output_path: Path,
    play_depth: int = 6,
    label_depth: int = 15,
    sample_rate: float = 0.3,
    min_spread: int = 0,
) -> int:
    """Generate data from SF self-play at a given depth.

    Lower play_depth = more mistakes = more interesting positions.
    """
    examples = []
    games = 0
    batch_size = 100

    print(f"  Generating from SF self-play (depth={play_depth})...", flush=True)

    while len(examples) < num_examples:
        board = chess.Board()
        temporal = TemporalController()
        games += 1

        for ply in range(200):
            if board.is_game_over():
                break

            # Sample this position?
            if ply >= 6 and random.random() < sample_rate:
                ex = board_to_signal_example(
                    board, temporal=temporal,
                )
                if ex and ex.num_candidates >= 2:
                    # Label with deeper SF
                    best_idx, best_cp, gap = label_with_stockfish(
                        board, ex.candidate_moves, sf, depth=label_depth,
                    )

                    # Candidate spread filter
                    if gap >= min_spread:
                        ex.best_candidate_idx = best_idx
                        ex.oracle_eval_cp = best_cp
                        ex.eval_gap_cp = gap
                        examples.append(ex)

                        # Crash-safe: flush every batch_size
                        if len(examples) % batch_size == 0:
                            append_signal_dataset(
                                examples[-batch_size:], output_path,
                            )
                            print(
                                f"    {len(examples)}/{num_examples} "
                                f"(games={games}, spread>={min_spread})",
                                flush=True,
                            )

                        if len(examples) >= num_examples:
                            break

            # Play the move
            result = sf.play(board, chess.engine.Limit(depth=play_depth, time=0.05))
            if result.move is None:
                break
            board.push(result.move)

    # Flush remaining
    remainder = len(examples) % batch_size
    if remainder > 0:
        append_signal_dataset(examples[-remainder:], output_path)

    print(f"    Done: {len(examples)} examples from {games} games", flush=True)
    return len(examples)


def generate_multi_elo(
    num_per_level: int,
    output_path: Path,
    label_depth: int = 15,
) -> int:
    """Source A: SF vs SF at different ELO levels.

    Uses different play depths as proxy for different ELO levels.
    Lower depth = weaker play = more instructive mistakes.
    """
    sf = chess.engine.SimpleEngine.popen_uci("stockfish")
    total = 0

    # Clear output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    levels = [
        (3, 50, "beginner mistakes"),    # depth 3 ≈ 1200 ELO
        (5, 30, "intermediate mistakes"), # depth 5 ≈ 1600 ELO
        (8, 20, "advanced mistakes"),     # depth 8 ≈ 2000 ELO
        (12, 10, "expert play"),          # depth 12 ≈ 2400 ELO
    ]

    for play_depth, min_spread, label in levels:
        print(f"\n  Level: {label} (depth={play_depth}, spread>={min_spread})",
              flush=True)
        n = generate_from_sf_games(
            sf, num_per_level, output_path,
            play_depth=play_depth,
            label_depth=label_depth,
            min_spread=min_spread,
        )
        total += n

    sf.quit()
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Generate signal-profile training data"
    )
    parser.add_argument("--num", type=int, default=5000,
                        help="Examples per ELO level (total = 4x this)")
    parser.add_argument("--output", default="data/signal_train.jsonl")
    parser.add_argument("--eval-output", default="data/signal_eval.jsonl")
    parser.add_argument("--label-depth", type=int, default=15)
    parser.add_argument("--mode", default="multi-elo",
                        choices=["multi-elo", "single"])
    args = parser.parse_args()

    print("=== Signal Profile Data Generation ===", flush=True)
    print(f"Mode: {args.mode}", flush=True)
    print(f"Examples per level: {args.num}", flush=True)
    print(f"Label depth: {args.label_depth}", flush=True)
    print(flush=True)

    t0 = time.time()
    output = Path(args.output)

    if args.mode == "multi-elo":
        total = generate_multi_elo(
            args.num, output, label_depth=args.label_depth,
        )
    else:
        sf = chess.engine.SimpleEngine.popen_uci("stockfish")
        if output.exists():
            output.unlink()
        total = generate_from_sf_games(
            sf, args.num, output,
            play_depth=8, label_depth=args.label_depth,
        )
        sf.quit()

    # Split train/eval from the written file
    import json
    all_lines = output.read_text().strip().split("\n")
    random.shuffle(all_lines)
    split = int(len(all_lines) * 0.9)

    with open(output, "w") as f:
        f.write("\n".join(all_lines[:split]) + "\n")
    eval_path = Path(args.eval_output)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w") as f:
        f.write("\n".join(all_lines[split:]) + "\n")

    elapsed = time.time() - t0
    print(f"\n=== Done in {elapsed:.0f}s ===", flush=True)
    print(f"Train: {split} → {output}", flush=True)
    print(f"Eval: {len(all_lines) - split} → {eval_path}", flush=True)
    print(f"Total: {total} examples", flush=True)


if __name__ == "__main__":
    main()
