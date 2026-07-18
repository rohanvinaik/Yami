#!/usr/bin/env python3
"""Generate Society of Mind training data with deep PV analysis.

For each position:
  1. Extract position signals (58 dims) before each candidate move
  2. Push each candidate, extract signals after, compute dimensional delta
  3. SF depth 15 multipv=3 → PV lines + evals
  4. Walk PV[0] ply-by-ply, extract 15-step signal trajectory
  5. Classify PV theme (tactical/positional/endgame/attack/defense/initiative)
  6. Compute per-agent contrastive labels (best vs second-best dimensional delta)

Output: SoMTrainingExample with per-agent labels + PV trajectories.
Crash-safe: checkpoints every 100 examples.

Usage:
    python scripts/generate_som_data.py --num 5000 --output data/som_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
import chess.engine

from yami.position_signals import PositionSignals, extract_position_signals
from yami.candidate_signals import CandidateSignalVector, extract_candidate_signals
from yami.navigator import compute_navigation_vector, detect_anchors, otp_score_candidate
from yami.tactical_scoper import (
    scope_moves, apply_blunder_censor, apply_tactical_censor, apply_repetition_censor,
)
from yami.coherence import compute_coherence


# Agent domain → position signal dimension indices
# These map to the order in PositionSignals.to_vector()
AGENT_POSITION_DIMS = {
    "tactical": list(range(0, 5)),       # material signals
    "positional": list(range(5, 17)) + list(range(25, 33)),  # pawn structure + piece activity
    "endgame": [13, 14, 15, 16, 33],     # passed pawns + game phase
    "attack": [19, 21, 23],              # king safety theirs
    "defense": [17, 20, 22, 24],         # king safety ours + castled
    "initiative": [25, 26, 27, 28, 34, 35, 36, 37],  # mobility + development + center + space
}

MAX_CANDIDATES = 5
PV_MAX_PLY = 15


@dataclass
class SoMTrainingExample:
    """One training example with per-agent labels and PV trajectory."""

    fen: str
    position_signals: list[float]
    candidate_signals: list[list[float]]
    candidate_moves: list[str]
    num_candidates: int

    # Per-agent labels
    agent_labels: dict[str, float]
    pv_theme_vector: list[float]  # 6-dim: weight per agent theme

    # PV trajectory
    pv_signal_trajectory: list[list[float]]  # [ply][58]
    pv_length: int

    # Overall labels
    best_candidate_idx: int
    oracle_eval_cp: int = 0
    eval_gap_cp: int = 0
    sf_multipv_evals: list[int] = field(default_factory=list)
    game_outcome: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, line: str) -> SoMTrainingExample:
        return cls(**json.loads(line))


def extract_pv_trajectory(
    board: chess.Board, pv: list[chess.Move], max_ply: int = PV_MAX_PLY
) -> list[list[float]]:
    """Walk a principal variation ply-by-ply, extracting signal vectors."""
    trajectory = []
    b = board.copy()

    for i, move in enumerate(pv[:max_ply]):
        if move not in b.legal_moves:
            break
        b.push(move)
        try:
            sig = extract_position_signals(b)
            trajectory.append(sig.to_vector())
        except Exception:
            break

    return trajectory


def classify_pv_theme(
    pre_signals: list[float],
    trajectory: list[list[float]],
) -> list[float]:
    """Classify the PV's theme by measuring which dimensions change most.

    Returns a 6-dim vector (one per agent) indicating theme weight.
    """
    if not trajectory or len(trajectory) < 2:
        return [1.0 / 6] * 6  # uniform if no trajectory

    # Compute total change per dimension across the trajectory
    final = trajectory[-1]
    deltas = [abs(final[i] - pre_signals[i]) for i in range(min(len(final), len(pre_signals)))]

    # Sum deltas per agent domain
    agent_deltas = {}
    for agent, dims in AGENT_POSITION_DIMS.items():
        agent_sum = sum(deltas[d] for d in dims if d < len(deltas))
        agent_deltas[agent] = agent_sum

    total = sum(agent_deltas.values()) + 1e-8
    theme = [agent_deltas.get(a, 0.0) / total for a in
             ["tactical", "positional", "endgame", "attack", "defense", "initiative"]]

    return theme


def compute_agent_labels(
    board: chess.Board,
    best_move: chess.Move,
    second_move: chess.Move | None,
    pre_signals: list[float],
    eval_gap: int,
) -> dict[str, float]:
    """Compute per-agent contrastive labels.

    For each agent, measure how much its dimensions differ between
    the best and second-best move. The agent whose dimensions show
    the biggest difference "explains" why best > second.
    """
    # Extract post-move signals for best
    board.push(best_move)
    best_after = extract_position_signals(board).to_vector()
    board.pop()

    best_delta = [best_after[i] - pre_signals[i] for i in range(len(pre_signals))]

    # Extract post-move signals for second (if available)
    if second_move and second_move in board.legal_moves:
        board.push(second_move)
        second_after = extract_position_signals(board).to_vector()
        board.pop()
        second_delta = [second_after[i] - pre_signals[i] for i in range(len(pre_signals))]
    else:
        second_delta = [0.0] * len(pre_signals)

    # Contrastive: what's different between best and second in each agent's dims?
    labels = {}
    total_diff = 0.0

    for agent, dims in AGENT_POSITION_DIMS.items():
        agent_diff = 0.0
        for d in dims:
            if d < len(best_delta):
                agent_diff += (best_delta[d] - second_delta[d]) ** 2
        agent_diff = math.sqrt(agent_diff)
        labels[agent] = agent_diff
        total_diff += agent_diff

    # Normalize and scale by eval gap
    if total_diff > 0:
        scale = min(abs(eval_gap) / 100.0, 3.0)  # cap at 3.0
        for agent in labels:
            labels[agent] = (labels[agent] / total_diff) * scale

    return labels


def label_candidates_multipv(
    board: chess.Board,
    candidate_moves: list[str],
    sf: chess.engine.SimpleEngine,
    depth: int = 15,
) -> tuple[int, int, int, list[int], list[chess.Move]]:
    """Label candidates with SF multipv. Returns (best_idx, best_cp, gap, all_evals, pv)."""
    evals = []
    best_pv = []

    # Get multipv analysis of the position
    try:
        infos = sf.analyse(board, chess.engine.Limit(depth=depth), multipv=3)
        if isinstance(infos, dict):
            infos = [infos]

        for info in infos:
            score = info.get("score")
            cp = 0
            if score:
                pov = score.relative
                cp = pov.score() if pov.score() is not None else 0
            if not best_pv and "pv" in info:
                best_pv = info["pv"]
    except Exception:
        pass

    # Also evaluate each candidate directly
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
            info = sf.analyse(board, chess.engine.Limit(depth=depth, time=0.15))
            score = info.get("score")
            cp = 0
            if score:
                pov = score.relative
                cp = -(pov.score() if pov.score() is not None else 0)
            evals.append(cp)
            board.pop()
        except Exception:
            evals.append(-99999)

    if not evals or max(evals) == -99999:
        return 0, 0, 0, [], best_pv

    best_idx = max(range(len(evals)), key=lambda i: evals[i])
    best_cp = evals[best_idx]
    valid = sorted([e for e in evals if e > -99999], reverse=True)
    gap = valid[0] - valid[1] if len(valid) >= 2 else 0

    return best_idx, best_cp, gap, evals, best_pv


def extract_som_example(
    board: chess.Board,
    sf: chess.engine.SimpleEngine,
    depth: int = 15,
) -> SoMTrainingExample | None:
    """Full pipeline: board → SoM training example."""
    # Step 1: Pre-move position signals
    pre_signals = extract_position_signals(board)
    pre_vec = pre_signals.to_vector()

    # Step 2: Navigator
    nav_vector = compute_navigation_vector(board)
    pre_signals.nav_aggression = nav_vector.aggression
    pre_signals.nav_piece_domain = nav_vector.piece_domain
    pre_signals.nav_complexity = nav_vector.complexity
    pre_signals.nav_initiative = nav_vector.initiative
    pre_signals.nav_king_pressure = nav_vector.king_pressure
    pre_signals.nav_phase = nav_vector.phase
    pre_vec = pre_signals.to_vector()  # re-extract with nav

    # Step 3: Scope and censor
    scoped = scope_moves(board)
    if len(scoped) < 2:
        return None
    censored = apply_blunder_censor(scoped)
    censored = apply_tactical_censor(censored, board)
    censored = apply_repetition_censor(censored, board)
    if not censored or len(censored) < 2:
        censored = scoped

    # Step 4: Coherence scoring to get candidates
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

    # Pad
    move_ucis = [m.uci() for m in cand_moves]
    pad_vec = [0.0] * CandidateSignalVector.vector_dim()
    while len(cand_vecs) < MAX_CANDIDATES:
        cand_vecs.append(pad_vec)
        move_ucis.append("")

    # Step 5: SF depth 15 multipv=3 labeling
    best_idx, best_cp, gap, all_evals, pv = label_candidates_multipv(
        board, move_ucis, sf, depth=depth,
    )

    if best_idx >= len(cand_moves):
        best_idx = 0

    # Step 6: Walk PV for signal trajectory
    trajectory = extract_pv_trajectory(board, pv, max_ply=PV_MAX_PLY)

    # Step 7: Classify PV theme
    pv_theme = classify_pv_theme(pre_vec, trajectory)

    # Step 8: Per-agent contrastive labels
    second_move = cand_moves[1] if len(cand_moves) > 1 and best_idx == 0 else (
        cand_moves[0] if best_idx != 0 else None
    )
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
        sf_multipv_evals=all_evals,
        game_outcome=0.0,
    )


def generate_from_sf_play(
    sf: chess.engine.SimpleEngine,
    num_examples: int,
    output_path: Path,
    play_depth: int = 6,
    label_depth: int = 15,
    sample_rate: float = 0.25,
    min_spread: int = 0,
) -> int:
    """Generate SoM data from SF self-play."""
    examples = []
    games = 0
    batch_size = 100

    while len(examples) < num_examples:
        board = chess.Board()
        games += 1

        for ply in range(200):
            if board.is_game_over():
                break

            # Sample this position?
            if ply >= 6 and random.random() < sample_rate:
                ex = extract_som_example(board, sf, depth=label_depth)
                if ex and ex.num_candidates >= 2 and ex.eval_gap_cp >= min_spread:
                    examples.append(ex)

                    if len(examples) % batch_size == 0:
                        _append(examples[-batch_size:], output_path)
                        print(
                            f"    {len(examples)}/{num_examples} "
                            f"(games={games}, pv_avg={sum(e.pv_length for e in examples[-batch_size:])/batch_size:.1f}ply)",
                            flush=True,
                        )

                    if len(examples) >= num_examples:
                        break

            # Play move
            result = sf.play(board, chess.engine.Limit(depth=play_depth, time=0.05))
            if result.move is None:
                break
            board.push(result.move)

    # Flush remaining
    remainder = len(examples) % batch_size
    if remainder > 0:
        _append(examples[-remainder:], output_path)

    return len(examples)


def _append(examples: list[SoMTrainingExample], path: Path) -> None:
    """Append examples to JSONL (crash-safe)."""
    with open(path, "a") as f:
        for ex in examples:
            f.write(ex.to_json() + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate SoM training data")
    parser.add_argument("--num", type=int, default=5000, help="Examples per ELO level")
    parser.add_argument("--output", default="data/som_train.jsonl")
    parser.add_argument("--eval-output", default="data/som_eval.jsonl")
    parser.add_argument("--label-depth", type=int, default=15)
    args = parser.parse_args()

    print("=== SoM Training Data Generation ===", flush=True)
    print(f"Examples per level: {args.num}", flush=True)
    print(f"Label depth: {args.label_depth} (with multipv=3 + PV walk)", flush=True)
    print(flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    sf = chess.engine.SimpleEngine.popen_uci("stockfish")
    t0 = time.time()
    total = 0

    levels = [
        (3, 50, "beginner mistakes"),
        (6, 20, "intermediate mistakes"),
        (10, 10, "advanced play"),
    ]

    for play_depth, min_spread, label in levels:
        print(f"\n  Level: {label} (depth={play_depth}, spread>={min_spread})", flush=True)
        n = generate_from_sf_play(
            sf, args.num, output,
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
    print(f"Train: {split} → {output}", flush=True)
    print(f"Eval: {len(all_lines) - split} → {eval_path}", flush=True)
    print(f"Total: {total} examples with per-agent labels + PV trajectories", flush=True)


if __name__ == "__main__":
    main()
