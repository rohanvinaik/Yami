"""U3 increment 2 — break-conditions as SIGNAL-GUARDS.

A literature-mined break-condition (data/genesis_roster/u3_atoms.jsonl) is a *transition*: the point at
which a frame's meaning flips — `law_as_architecture`'s "break the rule at square 11". Here each guard
reads ONLY real Yami signals (never an authored chess judgment, same discipline as the renderer) and,
when its break-condition holds, emits a DISTINCT token. That token is what the precedence thermometer
(`scripts/style_composition_probe.py`) can see: a player who *converts* a structural weakness in the
endgame emits the transition token; one who only *creates* weaknesses for dynamic play does not.

Every guard cites the mined atom that licenses it. The naming is an interpretation licensed by that
grounded atom + a deterministic signal — exactly the register the renderer already uses ("restrains").
"""
from __future__ import annotations


def structural_weakness_conversion(before: dict, after: dict, phase: int, name: str) -> list[str]:
    """ATOM: "Isolated pawn": isolates -> weak, break_condition="when the game reaches the endgame".

    Generalized to the structural-weakness CLASS (isolated + backward): a static weakness is only a
    *dynamic marker* in the middlegame but becomes a REAL, convertible weakness once the game simplifies
    to the endgame — the mined break-condition. Fires when, IN THE ENDGAME, the opponent carries a
    structural weakness AND this move curbs their mobility (the pressure signal the renderer already
    uses for `restrains`). Reads: `opp_isolated`, `opp_backward`, `opp_mobility`, phase. Nothing authored.
    """
    if phase != -1:                                              # break-guard: endgame only
        return []
    opp_has_weakness = after["opp_isolated"] > 0 or after["opp_backward"] > 0
    pressuring = before["opp_mobility"] - after["opp_mobility"] >= 0.03
    return [f"{name} exploits weakness."] if (opp_has_weakness and pressuring) else []


# Registry — one entry per wired guard. Board-scan guards (passed-pawn × rook-behind, outpost) are added
# here once the flagship confirms break-guarded tokens move the thermometer.
GUARDS = [structural_weakness_conversion]


def apply_break_guards(before: dict, after: dict, phase: int, name: str) -> list[str]:
    out: list[str] = []
    for guard in GUARDS:
        out.extend(guard(before, after, phase, name))
    return out
