#!/usr/bin/env python3
"""Wrapper for SoM data generation on remote compute nodes.

Accepts --sf for custom Stockfish path and --levels for number of difficulty levels.
Delegates to generate_som_data.py's core pipeline.

Usage:
    python scripts/run_som_gen.py --output data/som_sh.jsonl --num 5500 --levels 3 --sf ~/stockfish
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import chess
import chess.engine

from generate_som_data import (
    generate_from_sf_play,
    _append,
)


LEVEL_CONFIGS = [
    (3, 50, "beginner mistakes"),
    (6, 20, "intermediate mistakes"),
    (10, 10, "advanced play"),
    (15, 5, "expert play"),
    (20, 0, "engine-level play"),
]


def main():
    parser = argparse.ArgumentParser(description="Run SoM data generation (remote node)")
    parser.add_argument("--num", type=int, default=5000, help="Examples per difficulty level")
    parser.add_argument("--output", default="data/som_sh.jsonl", help="Output JSONL path")
    parser.add_argument("--eval-output", default=None, help="Eval split output (default: <output>.eval.jsonl)")
    parser.add_argument("--levels", type=int, default=3, help="Number of difficulty levels (1-5)")
    parser.add_argument("--sf", default="stockfish", help="Path to Stockfish binary")
    parser.add_argument("--label-depth", type=int, default=15, help="SF analysis depth for labeling")
    args = parser.parse_args()

    if args.eval_output is None:
        stem = Path(args.output).stem
        args.eval_output = str(Path(args.output).parent / f"{stem}_eval.jsonl")

    levels = LEVEL_CONFIGS[: args.levels]

    print("=== SoM Training Data Generation (Remote Node) ===", flush=True)
    print(f"Stockfish: {args.sf}", flush=True)
    print(f"Examples per level: {args.num}", flush=True)
    print(f"Levels: {len(levels)}", flush=True)
    print(f"Label depth: {args.label_depth}", flush=True)
    print(f"Output: {args.output}", flush=True)
    print(flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    sf = chess.engine.SimpleEngine.popen_uci(args.sf)
    t0 = time.time()
    total = 0

    for play_depth, min_spread, label in levels:
        print(f"\n  Level: {label} (depth={play_depth}, spread>={min_spread})", flush=True)
        n = generate_from_sf_play(
            sf,
            args.num,
            output,
            play_depth=play_depth,
            label_depth=args.label_depth,
            min_spread=min_spread,
        )
        total += n

    sf.quit()

    # Split train/eval
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
    print(f"Train: {split} -> {output}", flush=True)
    print(f"Eval: {len(all_lines) - split} -> {eval_path}", flush=True)
    print(f"Total: {total} examples with per-agent labels + PV trajectories", flush=True)


if __name__ == "__main__":
    main()
