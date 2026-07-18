#!/usr/bin/env python3
"""Generate SoM training data from the Lichess 342M position eval dataset.

5-10x faster than self-play generation because:
  1. Positions are pre-selected (no game simulation needed)
  2. Position evals are pre-computed (no SF needed for position assessment)
  3. We only run SF for candidate labeling (5 moves × depth 15)

Pipeline per position:
  1. Stream FEN from Lichess HuggingFace dataset
  2. Filter: skip trivial positions (|eval| > 500cp, < 6 legal moves)
  3. Run Yami infrastructure → 5 candidates with signal profiles
  4. SF depth 15 on each candidate → label best
  5. PV walk on best candidate → signal trajectory
  6. Compute per-agent dimensional delta labels

Usage:
    python scripts/generate_som_from_lichess.py --num 100000
    python scripts/generate_som_from_lichess.py --num 50000 --output data/som_lichess_train.jsonl
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
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from generate_som_data import (
    SoMTrainingExample,
    classify_pv_theme,
    compute_agent_labels,
    extract_pv_trajectory,
)
from yami.candidate_signals import CandidateSignalVector, extract_candidate_signals
from yami.coherence import compute_coherence
from yami.navigator import compute_navigation_vector, detect_anchors
from yami.position_signals import extract_position_signals
from yami.tactical_scoper import (
    apply_blunder_censor,
    apply_repetition_censor,
    apply_tactical_censor,
    scope_moves,
)

MAX_CANDIDATES = 5


def process_position(
    fen: str,
    sf: chess.engine.SimpleEngine,
    label_depth: int = 15,
) -> SoMTrainingExample | None:
    """Process one Lichess position into a SoM training example."""
    try:
        board = chess.Board(fen)
    except ValueError:
        return None

    if not board.is_valid() or board.is_game_over():
        return None

    legal_count = board.legal_moves.count()
    if legal_count < 3:
        return None

    # Position signals
    pre_signals = extract_position_signals(board)
    nav_vector = compute_navigation_vector(board)
    pre_signals.nav_aggression = nav_vector.aggression
    pre_signals.nav_piece_domain = nav_vector.piece_domain
    pre_signals.nav_complexity = nav_vector.complexity
    pre_signals.nav_initiative = nav_vector.initiative
    pre_signals.nav_king_pressure = nav_vector.king_pressure
    pre_signals.nav_phase = nav_vector.phase
    pre_vec = pre_signals.to_vector()

    # Scope and censor
    scoped = scope_moves(board)
    if len(scoped) < 3:
        return None
    censored = apply_blunder_censor(scoped)
    censored = apply_tactical_censor(censored, board)
    censored = apply_repetition_censor(censored, board)
    if not censored or len(censored) < 2:
        censored = scoped

    # Coherence scoring → candidates
    cand_moves_raw = [m.move for m in censored[:MAX_CANDIDATES * 2]]
    coherence_result = compute_coherence(board, cand_moves_raw, nav_vector)
    scored = coherence_result.scored_moves[:MAX_CANDIDATES]

    if len(scored) < 2:
        return None

    # Build candidate signal vectors
    cand_moves = []
    cand_vecs = []
    for sig in scored:
        if sig.move not in board.legal_moves:
            continue
        anchors = detect_anchors(board, sig.move)
        csig = extract_candidate_signals(
            board, sig.move, anchors=anchors, nav_vector=nav_vector,
            navigator_score=sig.navigator_score,
            strategy_score=sig.strategy_score,
            gm_freq=sig.gm_frequency,
            gm_winrate=sig.gm_win_rate,
            kline_score=sig.kline_score,
            look_ahead=sig.look_ahead_score,
            interference=sig.interference,
            navigator_ternary=sig.ternary_navigator,
            strategy_ternary=sig.ternary_strategy,
            gm_ternary=sig.ternary_gm,
            kline_ternary=sig.ternary_kline,
        )
        cand_moves.append(sig.move)
        cand_vecs.append(csig.to_vector())

    if len(cand_moves) < 2:
        return None

    # Pad candidates
    move_ucis = [m.uci() for m in cand_moves]
    pad_vec = [0.0] * CandidateSignalVector.vector_dim()
    while len(cand_vecs) < MAX_CANDIDATES:
        cand_vecs.append(pad_vec)
        move_ucis.append("")

    # SF labels: evaluate each candidate at depth 15
    evals = []
    best_pv = []
    for i, m in enumerate(cand_moves):
        try:
            board.push(m)
            info = sf.analyse(board, chess.engine.Limit(depth=label_depth, time=0.15))
            score = info.get("score")
            cp = 0
            if score:
                pov = score.relative
                raw = pov.score()
                cp = -(raw if raw is not None else 0)
            evals.append(cp)
            if i == 0 and "pv" in info:
                best_pv = info["pv"]
            board.pop()
        except Exception:
            evals.append(-99999)
            if board.fen() != fen:
                try:
                    board = chess.Board(fen)
                except Exception:
                    return None

    if not evals or max(evals) == -99999:
        return None

    best_idx = max(range(len(evals)), key=lambda i: evals[i])
    best_cp = evals[best_idx]
    valid_evals = sorted([e for e in evals if e > -99999], reverse=True)
    gap = valid_evals[0] - valid_evals[1] if len(valid_evals) >= 2 else 0

    # PV walk for signal trajectory
    # Use the PV from the best candidate's analysis
    if best_idx < len(cand_moves):
        board_after = board.copy()
        board_after.push(cand_moves[best_idx])
        trajectory = extract_pv_trajectory(board_after, best_pv)
    else:
        trajectory = []

    # PV theme
    pv_theme = classify_pv_theme(pre_vec, trajectory)

    # Per-agent labels
    second_idx = 1 if best_idx == 0 else 0
    second_move = cand_moves[second_idx] if second_idx < len(cand_moves) else None
    agent_labels = compute_agent_labels(
        board, cand_moves[best_idx], second_move, pre_vec, gap,
    )

    return SoMTrainingExample(
        fen=board.fen(),
        position_signals=pre_vec,
        candidate_signals=cand_vecs[:MAX_CANDIDATES],
        candidate_moves=move_ucis[:MAX_CANDIDATES],
        num_candidates=len(cand_moves),
        agent_labels=agent_labels,
        pv_theme_vector=pv_theme,
        pv_signal_trajectory=trajectory,
        pv_length=len(trajectory),
        best_candidate_idx=best_idx,
        oracle_eval_cp=best_cp,
        eval_gap_cp=gap,
        sf_multipv_evals=evals,
        game_outcome=0.0,  # no game context from Lichess positions
    )


def main():
    parser = argparse.ArgumentParser(description="Generate SoM data from Lichess positions")
    parser.add_argument("--num", type=int, default=100000)
    parser.add_argument("--output", default="data/som_lichess_train.jsonl")
    parser.add_argument("--eval-output", default="data/som_lichess_eval.jsonl")
    parser.add_argument("--label-depth", type=int, default=15)
    parser.add_argument("--min-spread", type=int, default=10,
                        help="Min cp spread between candidates to keep")
    parser.add_argument("--max-position-eval", type=int, default=500,
                        help="Skip positions with |eval| > this (too decided)")
    parser.add_argument("--sample-rate", type=float, default=0.02,
                        help="What fraction of Lichess positions to consider")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    print("=== SoM Data from Lichess 342M Positions ===", flush=True)
    print(f"Target: {args.num} examples", flush=True)
    print(f"Label depth: {args.label_depth}", flush=True)
    print(f"Min spread: {args.min_spread}cp", flush=True)
    print(f"Max position eval: {args.max_position_eval}cp", flush=True)
    print(f"Sample rate: {args.sample_rate}", flush=True)
    print(flush=True)

    # Stream Lichess dataset
    print("Loading Lichess dataset (streaming)...", flush=True)
    ds = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)

    sf = chess.engine.SimpleEngine.popen_uci("stockfish")
    t0 = time.time()
    examples = []
    positions_seen = 0
    positions_tried = 0
    batch_size = 100
    last_fen = None

    try:
        for row in ds:
            # Deduplicate: Lichess dataset has multiple depths per FEN
            row_dict = dict(row)  # type: ignore[arg-type]
            fen = str(row_dict["fen"])
            if fen == last_fen:
                continue
            last_fen = fen

            positions_seen += 1

            # Sample rate filter
            if random.random() > args.sample_rate:
                continue

            # Pre-filter: skip trivially decided positions
            cp = row_dict.get("cp")
            if cp is None:
                continue
            if abs(int(cp)) > args.max_position_eval:
                continue

            positions_tried += 1

            # Process position
            ex = process_position(fen, sf, label_depth=args.label_depth)
            if ex is None:
                continue

            # Candidate spread filter
            if ex.eval_gap_cp < args.min_spread:
                continue

            examples.append(ex)

            # Crash-safe checkpoint
            if len(examples) % batch_size == 0:
                with open(output, "a") as f:
                    for e in examples[-batch_size:]:
                        f.write(e.to_json() + "\n")
                elapsed = time.time() - t0
                rate = len(examples) / elapsed * 60
                eta = (args.num - len(examples)) / max(rate, 1)
                print(
                    f"  {len(examples)}/{args.num} "
                    f"({positions_seen:,} seen, {positions_tried:,} tried, "
                    f"{rate:.0f}/min, ETA {eta:.0f}min)",
                    flush=True,
                )

            if len(examples) >= args.num:
                break

    finally:
        sf.quit()

    # Flush remaining
    remainder = len(examples) % batch_size
    if remainder > 0:
        with open(output, "a") as f:
            for e in examples[-remainder:]:
                f.write(e.to_json() + "\n")

    # Split train/eval
    all_lines = output.read_text().strip().split("\n")
    random.shuffle(all_lines)
    split = int(len(all_lines) * 0.9)

    output.write_text("\n".join(all_lines[:split]) + "\n")
    eval_path = Path(args.eval_output)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text("\n".join(all_lines[split:]) + "\n")

    elapsed = time.time() - t0
    print(f"\n=== Done in {elapsed:.0f}s ===", flush=True)
    print(f"Train: {split} → {output}", flush=True)
    print(f"Eval: {len(all_lines) - split} → {eval_path}", flush=True)
    print(f"Positions seen: {positions_seen:,}, tried: {positions_tried:,}", flush=True)


if __name__ == "__main__":
    main()
