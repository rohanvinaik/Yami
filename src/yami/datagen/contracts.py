"""Training data contracts for the Yami neural Layer 7.

Each training example captures: position features + annotated candidates + Stockfish label.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# Tactical motif vocabulary (fixed order for one-hot encoding)
MOTIF_VOCAB = [
    "capture", "check", "checkmate", "fork", "pin",
    "discovery", "material_gain", "hangs_piece", "promotion",
]
MOTIF_TO_IDX = {m: i for i, m in enumerate(MOTIF_VOCAB)}

RISK_LEVELS = {"low": 0, "medium": 1, "high": 2, "critical": 3}


@dataclass
class CandidateFeatures:
    """Numeric encoding of a single annotated candidate."""

    move_uci: str
    move_san: str
    motif_flags: list[int]  # one-hot, len=len(MOTIF_VOCAB)
    plan_alignment: float
    positional_eval: float
    risk_level: int  # 0-3
    is_capture: bool
    is_check: bool
    see_value: float  # normalized


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
