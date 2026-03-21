"""Extract structured features from a board position via Yami Layers 1-6.

Runs the infrastructure pipeline and converts the output into numeric
feature vectors suitable for neural training.
"""

from __future__ import annotations

import chess

from yami.candidate_filter import filter_and_annotate
from yami.datagen.contracts import (
    MOTIF_TO_IDX,
    MOTIF_VOCAB,
    RISK_LEVELS,
    CandidateFeatures,
    ChessExample,
)
from yami.models import (
    Activity,
    Material,
    Safety,
    Structure,
    Tempo,
)
from yami.tactical_scoper import (
    apply_blunder_censor,
    apply_repetition_censor,
    apply_tactical_censor,
    scope_moves,
)

_MATERIAL_MAP = {Material.BEHIND: 0, Material.EQUAL: 1, Material.AHEAD: 2}
_STRUCTURE_MAP = {
    Structure.OPEN: 0, Structure.SEMI_OPEN: 1, Structure.CLOSED: 2,
    Structure.HEDGEHOG: 3, Structure.ISOLATED: 4, Structure.HANGING: 5,
}
_ACTIVITY_MAP = {Activity.ACTIVE: 0, Activity.PASSIVE: 1, Activity.CRAMPED: 2}
_SAFETY_MAP = {Safety.SAFE: 0, Safety.EXPOSED: 1, Safety.UNDER_ATTACK: 2}
_TEMPO_MAP = {Tempo.AHEAD: 0, Tempo.EQUAL: 1, Tempo.BEHIND: 2}
_PLAN_MAP = {pt: i for i, pt in enumerate(
    __import__("yami.models", fromlist=["PlanType"]).PlanType
)}

# Padding candidate for positions with <5 candidates
_PAD_CANDIDATE = CandidateFeatures(
    move_uci="0000", move_san="--",
    motif_flags=[0] * len(MOTIF_VOCAB),
    plan_alignment=0.0, positional_eval=0.0,
    risk_level=0, is_capture=False, is_check=False, see_value=0.0,
)


def extract_features(board: chess.Board, max_candidates: int = 5) -> tuple[
    list[CandidateFeatures],
    dict[str, int | float],
    int,
]:
    """Run Yami L1-6 on a board and emit structured features.

    Returns:
        (candidate_features, profile_dict, plan_type_idx)
    """
    scoped = scope_moves(board)

    # Apply censors
    censored = apply_blunder_censor(scoped)
    censored = apply_tactical_censor(censored, board)
    censored = apply_repetition_censor(censored, board)
    if not censored:
        censored = scoped

    candidates, plan, profile = filter_and_annotate(
        board, censored, max_candidates=max_candidates
    )

    # Convert candidates to feature vectors
    cand_features = []
    for c in candidates:
        motif_flags = [0] * len(MOTIF_VOCAB)
        for motif in c.tactical_motifs:
            idx = MOTIF_TO_IDX.get(motif)
            if idx is not None:
                motif_flags[idx] = 1

        cand_features.append(CandidateFeatures(
            move_uci=c.move.uci(),
            move_san=c.san,
            motif_flags=motif_flags,
            plan_alignment=c.plan_alignment,
            positional_eval=c.positional_eval,
            risk_level=RISK_LEVELS.get(c.risk, 0),
            is_capture="capture" in c.tactical_motifs,
            is_check="check" in c.tactical_motifs,
            see_value=0.0,  # TODO: pass through from ScopedMove
        ))

    # Pad to max_candidates
    while len(cand_features) < max_candidates:
        cand_features.append(_PAD_CANDIDATE)

    profile_dict = {
        "material": _MATERIAL_MAP.get(profile.material, 1),
        "structure": _STRUCTURE_MAP.get(profile.structure, 1),
        "activity": _ACTIVITY_MAP.get(profile.activity, 0),
        "safety": _SAFETY_MAP.get(profile.safety, 0),
        "opponent_safety": _SAFETY_MAP.get(profile.opponent_safety, 0),
        "tempo": _TEMPO_MAP.get(profile.tempo, 1),
        "plan_activation": plan.activation_score,
    }

    plan_idx = _PLAN_MAP.get(plan.plan_type, 0)

    return cand_features, profile_dict, plan_idx


def board_to_example(
    board: chess.Board,
    best_idx: int,
    oracle_eval_cp: int,
    second_best_idx: int = -1,
    eval_gap_cp: int = 0,
) -> ChessExample:
    """Convert a board + oracle label into a full training example."""
    cand_features, profile, plan_idx = extract_features(board)

    return ChessExample(
        fen=board.fen(),
        material=profile["material"],
        structure=profile["structure"],
        activity=profile["activity"],
        safety=profile["safety"],
        opponent_safety=profile["opponent_safety"],
        tempo=profile["tempo"],
        plan_type=plan_idx,
        plan_activation=profile["plan_activation"],
        candidates=cand_features,
        num_candidates=min(5, sum(1 for c in cand_features if c.move_uci != "0000")),
        best_candidate_idx=best_idx,
        oracle_eval_cp=oracle_eval_cp,
        second_best_idx=second_best_idx,
        eval_gap_cp=eval_gap_cp,
    )
