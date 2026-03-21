"""PyTorch dataset for chess training examples."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from yami.datagen.contracts import CANDIDATE_FEAT_DIM, ChessExample, load_dataset


class ChessDataset(Dataset):  # type: ignore[type-arg]
    """PyTorch dataset wrapping ChessExample JSONL files."""

    def __init__(self, path: Path) -> None:
        self.examples = load_dataset(path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return example_to_tensors(self.examples[idx])


def example_to_tensors(ex: ChessExample) -> dict[str, torch.Tensor]:
    """Convert a ChessExample into model-ready tensors."""
    # Profile: [6] int
    profile = torch.tensor(
        [ex.material, ex.structure, ex.activity,
         ex.safety, ex.opponent_safety, ex.tempo],
        dtype=torch.long,
    )

    # Profile continuous: [3] float
    profile_continuous = torch.tensor(
        [ex.game_phase, ex.total_material, ex.move_number],
        dtype=torch.float32,
    )

    # Plan
    plan_type = torch.tensor(ex.plan_type, dtype=torch.long)
    plan_activation = torch.tensor([ex.plan_activation], dtype=torch.float32)

    # Candidates: [5, CANDIDATE_FEAT_DIM]
    candidate_features = torch.zeros(5, CANDIDATE_FEAT_DIM)
    candidate_mask = torch.zeros(5, dtype=torch.bool)

    for i, c in enumerate(ex.candidates[:5]):
        if c.move_uci == "0000":
            continue
        candidate_mask[i] = True
        feats = (
            c.motif_flags  # 9
            + [c.plan_alignment]  # 1
            + [c.positional_eval]  # 1
            + [1.0 if c.risk_level == j else 0.0 for j in range(4)]  # 4
            + [float(c.is_capture)]  # 1
            + [float(c.is_check)]  # 1
            + [c.see_value]  # 1
            + c.piece_type_onehot  # 6
            + [c.target_centrality]  # 1
            + [c.dist_to_opp_king]  # 1
            + [c.dist_to_own_king]  # 1
            + [float(c.is_castling)]  # 1
            + [c.opponent_mobility]  # 1
            + [c.pawn_structure_change]  # 1
        )  # total = 30
        candidate_features[i] = torch.tensor(feats, dtype=torch.float32)

    # Labels
    best_idx = torch.tensor(ex.best_candidate_idx, dtype=torch.long)
    second_idx = torch.tensor(max(ex.second_best_idx, 0), dtype=torch.long)
    has_second = torch.tensor(ex.second_best_idx >= 0, dtype=torch.bool)

    return {
        "profile": profile,
        "profile_continuous": profile_continuous,
        "plan_type": plan_type,
        "plan_activation": plan_activation,
        "candidate_features": candidate_features,
        "candidate_mask": candidate_mask,
        "best_idx": best_idx,
        "second_idx": second_idx,
        "has_second": has_second,
    }
