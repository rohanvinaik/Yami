"""Scale functions for the progressive revelation pipeline.

Each scale follows the same interface:
    (board, candidates, context) → ScaleResult

Each scale:
    1. Receives RESIDUAL candidates from the scale below
    2. Computes its own signals and scores
    3. Measures AMBIGUITY (how confident is this scale?)
    4. Returns selected move OR residual candidates for the next scale

Progressive revelation: each scale gets a SMALLER, HARDER problem.
COEC amplification: co-supporting signal pairs > sum of parts.
"""

from __future__ import annotations

from typing import Any

import chess

from yami.coherence import compute_coherence
from yami.decision_capsule import (
    DecisionType,
    build_decision_capsule,
    classify_decision,
    compute_active_agents,
)
from yami.navigator import detect_anchors, otp_score_candidate
from yami.progressive_engine import ScaleResult, ScoredCandidate
from yami.tactical_scoper import (
    apply_blunder_censor,
    apply_repetition_censor,
    apply_tactical_censor,
    scope_moves,
)


# --- COEC Compositional Amplification ---

# Signal pairs that amplify beyond addition when both are ternary +1.
# Format: (signal_a, signal_b, multiplier)
# Multiplier > 1.0 means super-additive.
COEC_PAIRS = [
    ("navigator", "strategy", 1.4),
    ("tactical", "king_pressure", 1.3),
    ("gm", "som_convergence", 1.5),
    ("navigator", "gm", 1.3),
    ("strategy", "temporal", 1.2),
]


def _compute_coec_bonus(
    candidate: ScoredCandidate, context: dict[str, Any]
) -> float:
    """Compute COEC compositional amplification for a candidate.

    When two specific signals co-support (both ternary +1), the
    combined weight exceeds their sum. This is the key insight from
    the Data Geometry Architecture: constraint pairs that co-support
    are domain-specific and amplify beyond addition.
    """
    bonus = 0.0
    ts = candidate.ternary_signals

    for sig_a, sig_b, multiplier in COEC_PAIRS:
        val_a = ts.get(sig_a, 0)
        val_b = ts.get(sig_b, 0)

        if val_a == 1 and val_b == 1:
            # Both signals support — super-additive amplification
            bonus += (multiplier - 1.0) * (candidate.total_score + 1.0)
        elif val_a == -1 and val_b == -1:
            # Both signals oppose — amplified rejection
            bonus -= (multiplier - 1.0) * 0.5

    return bonus


# --- Scale 0: Legal Moves + Censors ---


def scale_0_legal_and_censors(
    board: chess.Board,
    _candidates: list[ScoredCandidate],
    context: dict[str, Any],
) -> ScaleResult:
    """Scale 0: Generate legal moves and apply all censors.

    Input: raw board state (candidates list is empty at this scale)
    Output: censored candidates
    Ambiguity: survivor_count / legal_count
    Short-circuit: exactly 1 candidate survives
    """
    scoped = scope_moves(board)
    legal_count = len(scoped)
    context["legal_count"] = legal_count

    if legal_count == 0:
        return ScaleResult(selected=None, residual=[], ambiguity=0.0, resolved=True)

    # Apply censor stack
    censored = apply_blunder_censor(scoped)
    censored = apply_tactical_censor(censored, board)
    censored = apply_repetition_censor(censored, board)

    # Learned censors (negative learning)
    nav_vector = context.get("nav_vector")
    learned = context.get("learned_censors")
    if learned and nav_vector:
        censored = learned.filter_moves(board, censored, nav_vector)

    if not censored:
        censored = scoped  # fallback: don't censor everything

    # Track censor rejection rate
    pos_signals = context.get("pos_signals")
    if pos_signals:
        pos_signals.censor_stack_rejection_rate = (
            1.0 - len(censored) / max(legal_count, 1)
        )

    # Build ScoredCandidates
    candidates = [
        ScoredCandidate(move=m.move, scoped=m) for m in censored
    ]

    ambiguity = len(candidates) / max(legal_count, 1)
    resolved = len(candidates) == 1

    return ScaleResult(
        selected=candidates[0].move if resolved else None,
        residual=candidates,
        ambiguity=ambiguity,
        scale_signals={"legal_count": legal_count, "censored_count": len(candidates)},
        resolved=resolved,
    )


# --- Scale 1: Tactical Resolution ---


def scale_1_tactical_resolution(
    board: chess.Board,
    candidates: list[ScoredCandidate],
    context: dict[str, Any],
) -> ScaleResult:
    """Scale 1: Forced tactical sequences.

    Resolves: checkmates, forced material wins, forcing sequences.
    Ambiguity: gap between top tactical score and second.
    Short-circuit: checkmate exists, or one move is tactically dominant.
    """
    if not candidates:
        return ScaleResult(selected=None, residual=[], ambiguity=0.0, resolved=True)

    for cand in candidates:
        score = 0.0
        motifs = cand.scoped.motifs

        # Checkmate is absolute
        if "checkmate" in motifs:
            return ScaleResult(
                selected=cand.move,
                residual=[cand],
                ambiguity=0.0,
                scale_signals={"reason": "checkmate"},
                resolved=True,
            )

        # Tactical motif scoring
        if "material_gain" in motifs:
            score += max(cand.scoped.see_value / 100.0, 0.5)
        if "fork" in motifs:
            score += 2.0
        if "pin" in motifs:
            score += 1.5
        if "discovery" in motifs:
            score += 2.0
        if "check" in motifs:
            score += 0.5
        if "promotion" in motifs:
            score += 4.0
        if "hangs_piece" in motifs:
            score -= 5.0

        # Look-ahead (2-ply blunder detection)
        from yami.coherence import _compute_look_ahead
        la = _compute_look_ahead(board, cand.move)
        score += la

        cand.tactical_score = score
        cand.total_score += score

    # Sort by tactical score
    candidates.sort(key=lambda c: c.tactical_score, reverse=True)

    # Ambiguity: gap between top and second
    if len(candidates) >= 2:
        top = candidates[0].tactical_score
        second = candidates[1].tactical_score
        gap = (top - second) / (abs(top) + 1e-6)
        ambiguity = max(0.0, 1.0 - gap)
    else:
        ambiguity = 0.0

    return ScaleResult(
        selected=candidates[0].move if ambiguity < 0.01 else None,
        residual=candidates,
        ambiguity=ambiguity,
        scale_signals={"top_tactical": candidates[0].tactical_score},
        resolved=ambiguity < 0.01,
    )


# --- Scale 2: Positional Evaluation ---


def scale_2_positional_evaluation(
    board: chess.Board,
    candidates: list[ScoredCandidate],
    context: dict[str, Any],
) -> ScaleResult:
    """Scale 2: Navigator OTP + positional signals.

    Receives: candidates that tactics couldn't differentiate.
    Resolves: positional signal agreement.
    Residual: candidates within 80% of top positional score.
    """
    if not candidates:
        return ScaleResult(selected=None, residual=[], ambiguity=0.0, resolved=True)

    nav_vector = context.get("nav_vector")

    for cand in candidates:
        cand.anchors = detect_anchors(board, cand.move)

        if nav_vector:
            cand.positional_score = otp_score_candidate(
                cand.move, cand.anchors, nav_vector, board
            )
        cand.total_score += cand.positional_score

        # Build ternary signal
        cand.ternary_signals["navigator"] = (
            1 if cand.positional_score > 2.0
            else -1 if cand.positional_score < -2.0
            else 0
        )

    # Sort by total accumulated score
    candidates.sort(key=lambda c: c.total_score, reverse=True)

    # Ambiguity: spread of positional scores
    scores = [c.positional_score for c in candidates]
    if len(scores) >= 2:
        gap = (scores[0] - scores[1]) / (abs(scores[0]) + 1e-6)
        ambiguity = max(0.0, 1.0 - gap)
    else:
        ambiguity = 0.0

    # Residual: keep candidates within 80% of top score
    if candidates[0].total_score > 0:
        threshold = candidates[0].total_score * 0.8
        residual = [c for c in candidates if c.total_score >= threshold]
    else:
        residual = candidates[:5]  # keep top 5 when scores are negative

    if not residual:
        residual = candidates[:3]

    return ScaleResult(
        selected=candidates[0].move if ambiguity < 0.05 else None,
        residual=residual,
        ambiguity=ambiguity,
        scale_signals={"positional_spread": scores[0] - scores[-1] if scores else 0},
        resolved=ambiguity < 0.05,
    )


# --- Scale 3: Strategic Coherence + COEC ---


def scale_3_strategic_coherence(
    board: chess.Board,
    candidates: list[ScoredCandidate],
    context: dict[str, Any],
) -> ScaleResult:
    """Scale 3: Multi-signal coherence with COEC compositional amplification.

    This is where multiple signal banks interact. COEC pairs that
    co-support amplify beyond addition — the key insight from the
    Data Geometry Architecture.

    Signal banks: navigator (from Scale 2), strategy, GM patterns, K-lines.

    Also: computes agent suppression and decision classification.
    If the decision is infrastructure-resolved, short-circuits here
    without invoking the neural model (Wayfinder bimodal split).
    """
    if not candidates:
        return ScaleResult(selected=None, residual=[], ambiguity=0.0, resolved=True)

    nav_vector = context.get("nav_vector")
    gm_db = context.get("gm_db")
    klines = context.get("klines")
    la_weight = context.get("look_ahead_weight", 1.0)

    # Run coherence scoring on candidate moves
    cand_moves = [c.move for c in candidates]
    coherence_result = compute_coherence(
        board, cand_moves, nav_vector,
        temporal=context.get("temporal"),
        klines=klines,
        gm_db=gm_db,
        look_ahead_weight=la_weight,
    )

    # Map coherence signals back to candidates
    coh_by_move = {s.move: s for s in coherence_result.scored_moves}

    for cand in candidates:
        sig = coh_by_move.get(cand.move)
        if sig:
            cand.coherence_score = sig.final_score
            cand.total_score += sig.final_score

            # Build ternary signals for COEC
            cand.ternary_signals["strategy"] = sig.ternary_strategy
            cand.ternary_signals["gm"] = sig.ternary_gm
            cand.ternary_signals["kline"] = sig.ternary_kline

            # King pressure context for COEC pair
            if nav_vector and nav_vector.king_pressure == 1:
                cand.ternary_signals["king_pressure"] = 1
            elif nav_vector and nav_vector.king_pressure == -1:
                cand.ternary_signals["king_pressure"] = -1
            else:
                cand.ternary_signals["king_pressure"] = 0

            # Tactical ternary (from Scale 1)
            cand.ternary_signals["tactical"] = (
                1 if cand.tactical_score > 1.0
                else -1 if cand.tactical_score < -1.0
                else 0
            )

        # COEC amplification
        cand.coec_bonus = _compute_coec_bonus(cand, context)
        cand.total_score += cand.coec_bonus

    # Sort by total accumulated score
    candidates.sort(key=lambda c: c.total_score, reverse=True)

    # Ambiguity: signal agreement
    top = candidates[0]
    supports = sum(1 for v in top.ternary_signals.values() if v == 1)
    opposes = sum(1 for v in top.ternary_signals.values() if v == -1)
    total_signals = max(len(top.ternary_signals), 1)
    agreement = supports / total_signals
    ambiguity = 1.0 - agreement

    # --- Agent suppression (Wayfinder: Dr. Ducky bank filtering) ---
    pos_signals = context.get("pos_signals")
    active_agents = compute_active_agents(pos_signals) if pos_signals else {
        a: True for a in ["tactical", "positional", "endgame",
                          "attack", "defense", "initiative"]
    }
    context["active_agents"] = active_agents

    # --- Decision classification (Wayfinder: bimodal split) ---
    coherence_gap = 0.0
    if len(candidates) >= 2:
        gap = candidates[0].total_score - candidates[1].total_score
        coherence_gap = gap / (abs(candidates[0].total_score) + 1e-6)

    decision_type = classify_decision(
        coherence_gap=coherence_gap,
        signal_agreement=agreement,
        constructive_count=coherence_result.constructive_count,
        top_score=candidates[0].total_score,
        candidate_count=len(candidates),
    )
    context["decision_type"] = decision_type

    # --- Build decision capsule (Wayfinder: projector boundary) ---
    if pos_signals and coherence_result.scored_moves:
        capsule = build_decision_capsule(
            board=board,
            pos_signals=pos_signals,
            scored_moves=coherence_result.scored_moves,
            active_agents=active_agents,
            coherence_gap=coherence_gap,
            signal_agreement=agreement,
            constructive_count=coherence_result.constructive_count,
        )
        context["decision_capsule"] = capsule

    # If infrastructure-resolved, short-circuit: skip Scales 4-5
    if decision_type == DecisionType.INFRASTRUCTURE_RESOLVED:
        return ScaleResult(
            selected=candidates[0].move,
            residual=candidates[:2],
            ambiguity=ambiguity,
            scale_signals={
                "coec_bonus_top": top.coec_bonus,
                "supports": supports,
                "opposes": opposes,
                "agreement": agreement,
                "decision_type": "infrastructure_resolved",
                "suppressed_agents": sum(
                    1 for v in active_agents.values() if not v
                ),
            },
            resolved=True,
        )

    # Residual: candidates with mixed interference (ambiguous signals)
    if len(candidates) >= 2:
        if coherence_gap > 0.3:
            residual = candidates[:2]  # clear leader + runner-up
        else:
            residual = [c for c in candidates if c.total_score > candidates[0].total_score * 0.7]
            if not residual:
                residual = candidates[:3]
    else:
        residual = candidates

    return ScaleResult(
        selected=candidates[0].move if ambiguity < 0.15 else None,
        residual=residual,
        ambiguity=ambiguity,
        scale_signals={
            "coec_bonus_top": top.coec_bonus,
            "supports": supports,
            "opposes": opposes,
            "agreement": agreement,
            "decision_type": "genuine_residual",
            "suppressed_agents": sum(
                1 for v in active_agents.values() if not v
            ),
        },
        resolved=ambiguity < 0.15,
    )


# --- Scale 4: Temporal / SoM Context ---


def scale_4_temporal_context(
    board: chess.Board,
    candidates: list[ScoredCandidate],
    context: dict[str, Any],
) -> ScaleResult:
    """Scale 4: Society of Mind agent debate on remaining ambiguity.

    Six specialist agents score each candidate independently.
    Convergence across agents IS the confidence signal.

    Agent suppression (Wayfinder enhancement): only active agents
    contribute scores. Suppressed agents are structurally inert for
    this position type and would only add noise.
    """
    if not candidates:
        return ScaleResult(selected=None, residual=[], ambiguity=0.0, resolved=True)

    temporal = context.get("temporal")
    nav_vector = context.get("nav_vector")

    if temporal is None or nav_vector is None:
        return ScaleResult(
            selected=None, residual=candidates,
            ambiguity=1.0, scale_signals={}, resolved=False,
        )

    # Get agent activation state from Scale 3 (or compute fresh)
    active_agents = context.get("active_agents", {
        a: True for a in ["tactical", "positional", "endgame",
                          "attack", "defense", "initiative"]
    })

    for cand in candidates:
        agent_scores = temporal.score_move_by_agents(
            nav_vector, cand.anchors, material_balance=0
        )

        # Zero out suppressed agents — they are structurally inert
        # (Dr. Ducky: "no existential structure → don't run witness engine")
        for agent_name, is_active in active_agents.items():
            if not is_active:
                agent_scores[agent_name] = 0.0

        cand.temporal_score = sum(agent_scores.values())
        cand.total_score += cand.temporal_score * 0.3

        # Convergence only over ACTIVE agents
        active_scores = {k: v for k, v in agent_scores.items()
                         if active_agents.get(k, True)}
        convergence = temporal.compute_convergence(active_scores)

        cand.ternary_signals["temporal"] = (
            1 if convergence > 0.7
            else -1 if convergence < 0.3
            else 0
        )
        cand.ternary_signals["som_convergence"] = (
            1 if convergence > 0.6 else 0
        )

        # Late COEC: GM + SoM convergence pair
        if cand.ternary_signals.get("gm", 0) == 1 and convergence > 0.6:
            late_coec = 0.5 * cand.coherence_score
            cand.coec_bonus += late_coec
            cand.total_score += late_coec

    # Sort by total
    candidates.sort(key=lambda c: c.total_score, reverse=True)

    # Ambiguity: convergence over active agents only
    top_convergence = 0.0
    if candidates:
        top_agents = temporal.score_move_by_agents(
            nav_vector, candidates[0].anchors, material_balance=0
        )
        for agent_name, is_active in active_agents.items():
            if not is_active:
                top_agents[agent_name] = 0.0
        active_top = {k: v for k, v in top_agents.items()
                      if active_agents.get(k, True)}
        top_convergence = temporal.compute_convergence(active_top)
    ambiguity = 1.0 - top_convergence

    suppressed_count = sum(1 for v in active_agents.values() if not v)

    return ScaleResult(
        selected=candidates[0].move if ambiguity < 0.3 else None,
        residual=candidates,
        ambiguity=ambiguity,
        scale_signals={
            "convergence": top_convergence,
            "active_agents": sum(1 for v in active_agents.values() if v),
            "suppressed_agents": suppressed_count,
        },
        resolved=ambiguity < 0.3,
    )


# --- Scale 5: Neural Residual ---


def scale_5_neural_residual(
    board: chess.Board,
    candidates: list[ScoredCandidate],
    context: dict[str, Any],
) -> ScaleResult:
    """Scale 5: Neural/LLM decision on the HARD cases.

    Only reaches here if Scales 0-4 could not resolve.
    The model sees 1-3 candidates with full accumulated signals.
    Always resolves (terminal scale).
    """
    if not candidates:
        legal = list(board.legal_moves)
        return ScaleResult(
            selected=legal[0] if legal else None,
            residual=[],
            ambiguity=0.0,
            resolved=True,
        )

    fusion_model = context.get("fusion_model")

    if fusion_model is not None:
        import torch
        from yami.candidate_signals import CandidateSignalVector, extract_candidate_signals
        from yami.position_signals import PositionSignals

        pos_signals = context.get("pos_signals")
        nav_vector = context.get("nav_vector")

        pos_vec = pos_signals.to_vector() if pos_signals else [0.0] * 58

        cand_vecs = []
        for cand in candidates[:5]:
            csig = extract_candidate_signals(
                board, cand.move,
                anchors=cand.anchors,
                nav_vector=nav_vector,
                navigator_score=cand.positional_score,
                see_value=cand.scoped.see_value,
            )
            cand_vecs.append(csig.to_vector())

        # Pad to 5
        pad = [0.0] * CandidateSignalVector.vector_dim()
        while len(cand_vecs) < 5:
            cand_vecs.append(pad)

        pos_t = torch.tensor([pos_vec], dtype=torch.float32)
        cand_t = torch.tensor([cand_vecs[:5]], dtype=torch.float32)
        mask = torch.tensor([[i < len(candidates) for i in range(5)]])

        with torch.no_grad():
            out = fusion_model(pos_t, cand_t, mask)
            pred = out["candidate_logits"].argmax(dim=-1).item()

        if pred < len(candidates) and candidates[pred].move in board.legal_moves:
            selected = candidates[pred].move
        else:
            selected = candidates[0].move
    else:
        # No neural model — pick the top accumulated scorer
        selected = candidates[0].move

    return ScaleResult(
        selected=selected,
        residual=candidates,
        ambiguity=0.0,
        scale_signals={"method": "fusion_model" if fusion_model else "accumulated_score"},
        resolved=True,
    )
