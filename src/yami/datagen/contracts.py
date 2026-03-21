"""Training data contracts for the Yami neural Layer 7.

Each training example captures: position features + annotated candidates + Stockfish label.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Tactical motif vocabulary (fixed order for one-hot encoding)
MOTIF_VOCAB = [
    "capture", "check", "checkmate", "fork", "pin",
    "discovery", "material_gain", "hangs_piece", "promotion",
]
MOTIF_TO_IDX = {m: i for i, m in enumerate(MOTIF_VOCAB)}

RISK_LEVELS = {"low": 0, "medium": 1, "high": 2, "critical": 3}


# Total candidate feature dimensions: 30
CANDIDATE_FEAT_DIM = 30


@dataclass
class CandidateFeatures:
    """Numeric encoding of a single annotated candidate."""

    move_uci: str
    move_san: str
    motif_flags: list[int]  # one-hot, len=9
    plan_alignment: float
    positional_eval: float
    risk_level: int  # 0-3
    is_capture: bool
    is_check: bool
    see_value: float  # normalized by queen value
    # New features (v2)
    piece_type_onehot: list[int]  # one-hot, len=6 (P/N/B/R/Q/K)
    target_centrality: float  # [0, 1]
    dist_to_opp_king: float  # [0, 1]
    dist_to_own_king: float  # [0, 1]
    is_castling: bool
    opponent_mobility: float  # [0, 1]
    pawn_structure_change: float  # {-1, 0, +1}


@dataclass
class ChessExample:
    """One training example: position + candidates + oracle label."""

    fen: str
    # Positional profile (encoded as ints)
    material: int  # 0=behind, 1=equal, 2=ahead
    structure: int  # 0-5 enum index
    activity: int  # 0-2
    safety: int  # 0-2
    opponent_safety: int  # 0-2
    tempo: int  # 0-2
    # Plan
    plan_type: int  # 0-6
    plan_activation: float
    # Candidates (padded to 5)
    candidates: list[CandidateFeatures]
    num_candidates: int
    # Oracle label
    best_candidate_idx: int  # 0-4
    oracle_eval_cp: int
    # Board-level continuous features (v2)
    game_phase: float = 0.0  # 1.0=opening, 0.0=endgame
    total_material: float = 0.0  # normalized
    move_number: float = 0.0  # normalized
    # Navigation vector (v3 - Wayfinder)
    nav_vector: list[int] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0])
    # Holographic coherence features (v4)
    gm_top_move_freq: float = 0.0  # GM frequency for infrastructure's top candidate
    som_convergence: float = 0.0  # SoM agent convergence score
    interference_score: float = 0.0  # holographic interference pattern
    # Near-miss metadata
    second_best_idx: int = -1
    eval_gap_cp: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, line: str) -> ChessExample:
        d = json.loads(line)
        d["candidates"] = [CandidateFeatures(**c) for c in d["candidates"]]
        return cls(**d)


def save_dataset(examples: list[ChessExample], path: Path) -> None:
    """Save examples to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(ex.to_json() + "\n")


def load_dataset(path: Path) -> list[ChessExample]:
    """Load examples from JSONL."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(ChessExample.from_json(line))
    return examples
