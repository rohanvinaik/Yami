"""Navigate-over-moves — the Scale-4 recommendation layer, ported from ModelAtlas (2026-07-14).

The problem (user): a game read derives BOTH sides' brilliance — genuine signal, learn from both — but
for the DECISION *here*, the WINNER's move must be preferred. "Learn from both AND learn what wins."

ModelAtlas already solved the shape: score = significance × bank_alignment, MULTIPLICATIVE, banks steered
by DIRECTION (+1/-1/0). Ported verbatim from `model_atlas/query_navigate.py::_bank_score_single`:
  direction 0  → 1/(1+|pos|)   : judge on own merit  (LEARN FROM BOTH — outcome ignored)
  direction +1 → aligned rewarded, opposed decays    : PREFER THE WINNER
So the two modes are one knob: outcome-direction 0 (learning) vs +1 (deciding). Significance is the
read's chain-improbability (the frame's weight); a brilliant-but-LOSING move keeps its significance
(learnable) but its DECISION score is crushed multiplicatively (not halved — the ModelAtlas property).
Auditable: every rank traces to (significance × outcome), never an opaque score."""
from __future__ import annotations


def bank_score(pos: int, direction: int) -> float:
    """ModelAtlas _bank_score_single — the direction knob. pos = signed position on the bank."""
    if direction == 0:                       # judge on own merit (learn from both)
        return 1.0 / (1.0 + abs(pos))
    alignment = pos * direction
    if alignment > 0:                        # aligned with the wanted direction → rewarded
        return 1.0
    if alignment == 0:
        return 0.5
    return 1.0 / (1.0 + abs(alignment))      # opposed → hyperbolic decay


def navigate(candidates: list[dict], outcome_dir: int) -> list[dict]:
    """Rank candidate moves/frames by significance × outcome-bank alignment (multiplicative)."""
    scored = []
    for c in candidates:
        s = c["significance"] * bank_score(c["outcome"], outcome_dir)
        scored.append({**c, "score": round(s, 2)})
    return sorted(scored, key=lambda c: -c["score"])


# The REAL read of the Karpov game (both sides reached `brilliant`; significance = chain-improbability).
# outcome: +1 = the winner's side (Karpov), -1 = the loser's side (Opponent).
CANDIDATES = [
    {"move": "Opponent: overwhelm king → brilliant", "significance": 6.36, "outcome": -1},
    {"move": "Karpov:   overwhelm king → brilliant", "significance": 5.66, "outcome": +1},
    {"move": "Opponent: hunt king",                  "significance": 5.66, "outcome": -1},
    {"move": "Karpov:   hunt king",                  "significance": 3.87, "outcome": +1},
    {"move": "Opponent: build attack",               "significance": 4.56, "outcome": -1},
    {"move": "Karpov:   build attack",               "significance": 2.77, "outcome": +1},
    {"move": "Karpov:   obtain queen (material)",    "significance": 0.00, "outcome": +1},
]


def _show(title: str, ranked: list[dict]) -> None:
    print(f"\n=== {title} ===")
    for i, c in enumerate(ranked[:5], 1):
        oc = "WON " if c["outcome"] > 0 else "lost"
        print(f"  {i}. {c['move']:42s} score={c['score']:5.2f}  (sig {c['significance']:.2f} × {oc})")


# ── Outcome-modulation over a READ (§14.8): win/draw/loss = one outcome axis; the draw is its zero-point ──
# A fact's VALENCE vs its subject's OUTCOME: a win-frame co-present with a LOSS is the illusion the result
# refuted (the surprise); with a WIN it is real. Aligned facts surface; contradictions are the surprise —
# boosted in LEARN mode (the lesson), dampened in DECIDE mode (don't play the losing move).
WIN_FRAMES = {"dominant", "ascendant", "prevailing", "brilliant", "hunt", "trap", "overwhelm",
              "build", "sustain", "seize", "initiate", "victorious"}          # valence +1 (ascendancy)
FALL_FRAMES = {"fallen", "ruined", "defeated", "shattered", "refuted", "overextended", "collapse"}  # −1
HOLD_FRAMES = {"balanced", "resilient", "impregnable", "held", "neutralize", "endures", "holds"}     # 0 (draw)


def valence(state: str) -> int | None:
    if state in WIN_FRAMES:
        return +1
    if state in FALL_FRAMES:
        return -1
    if state in HOLD_FRAMES:
        return 0
    return None                                                              # universal (obtain) → sig already 0


def _alignment(val: int | None, outcome: int) -> int:
    """+1 aligned (surface), −1 contradicts (the surprise), 0 neutral."""
    if val is None:
        return 0
    if outcome == 0:                       # DRAW: the hold-frames are the achievement; win/fall are neutral
        return +1 if val == 0 else 0
    if val == 0:                           # a hold-frame in a decisive game — not the story
        return 0
    return val * outcome                   # win-frame×won / fall-frame×lost = +1 aligned; else −1 contradicts


def navigate_read(facts: list[dict], mode: str) -> list[dict]:
    """Modulate a read's significance by outcome. mode='decide' prefers what won; 'learn' surfaces surprise."""
    out = []
    for f in facts:
        a = _alignment(valence(f["state"]), f["outcome"])
        if a > 0:
            mult, tag = 1.0, "aligned"
        elif a == 0:
            mult, tag = 0.5, "neutral"
        else:                              # contradiction = the SURPRISE
            mult, tag = (1.3, "SURPRISE") if mode == "learn" else (0.35, "illusion")
        out.append({**f, "score": round(f["significance"] * mult, 2), "tag": tag})
    return sorted(out, key=lambda c: -c["score"])


# Real reads (subject, state, significance, subject-outcome). KARPOV won; TAL lost; PETROSIAN drew.
KARPOV = [  # both reached `brilliant`; Karpov WON (+1), Opponent LOST (-1)
    {"fact": "opponent → brilliant", "state": "brilliant", "significance": 6.36, "outcome": -1},
    {"fact": "karpov → brilliant",   "state": "brilliant", "significance": 5.66, "outcome": +1},
    {"fact": "opponent → hunt king", "state": "hunt",      "significance": 5.66, "outcome": -1},
    {"fact": "karpov → hunt king",   "state": "hunt",      "significance": 3.87, "outcome": +1},
]
TAL = [  # Tal LOST (-1): his win-arc is the illusion, the fall is the lesson
    {"fact": "tal → hunt king", "state": "hunt",     "significance": 2.48, "outcome": -1},
    {"fact": "tal → ascendant", "state": "ascendant", "significance": 1.79, "outcome": -1},
    {"fact": "tal → dominant",  "state": "dominant",  "significance": 1.39, "outcome": -1},
    {"fact": "tal → ruined",    "state": "ruined",    "significance": 0.69, "outcome": -1},
    {"fact": "tal → fallen",    "state": "fallen",    "significance": 0.69, "outcome": -1},
]
PETROSIAN = [  # DRAW (0): the hold is the achievement
    {"fact": "opponent → hunt king",       "state": "hunt",       "significance": 5.26, "outcome": 0},
    {"fact": "petrosian → balanced",       "state": "balanced",   "significance": 1.61, "outcome": 0},
    {"fact": "petrosian → neutralize opp", "state": "neutralize", "significance": 1.61, "outcome": 0},
    {"fact": "petrosian → strangled",      "state": "strangled",  "significance": 1.61, "outcome": 0},
]


def _show_read(title: str, ranked: list[dict]) -> None:
    print(f"\n=== {title} ===")
    for i, c in enumerate(ranked[:4], 1):
        print(f"  {i}. {c['fact']:26s} score={c['score']:5.2f}  [{c['tag']}]  (raw sig {c['significance']:.2f})")


if __name__ == "__main__":
    print("Navigate-over-moves — the same read, two directions of the outcome knob:")
    _show("LEARNING  (outcome_dir=0)  — learn from BOTH, ranked by how instructive",
          navigate(CANDIDATES, outcome_dir=0))
    _show("DECIDING  (outcome_dir=+1) — prefer what WINS, here and now",
          navigate(CANDIDATES, outcome_dir=+1))
    learn_top = navigate(CANDIDATES, 0)[0]["move"]
    decide_top = navigate(CANDIDATES, +1)[0]["move"]
    print(f"\nEqual-and-opposite: LEARNING tops with  '{learn_top}'")
    print(f"                    DECIDING tops with  '{decide_top}'")
    print("→ The opponent's brilliancy is the most instructive single lesson (learn), but Karpov's\n"
          "  winning move is what we PLAY (decide). Both kept; the outcome knob re-ranks — auditable.")

    print("\n" + "=" * 60)
    print("OUTCOME-MODULATION over a full read (§14.8) — win/draw/loss = one axis")
    _show_read("WIN  (Karpov, DECIDE) — the winner's climax outranks the loser's",
               navigate_read(KARPOV, mode="decide"))
    _show_read("LOSS (Tal, DECIDE) — the illusion dampened, the fall surfaces",
               navigate_read(TAL, mode="decide"))
    _show_read("LOSS (Tal, LEARN)  — the SURPRISE (dominant-then-fell) is the lesson",
               navigate_read(TAL, mode="learn"))
    _show_read("DRAW (Petrosian) — the hold is the achievement (neutral middle)",
               navigate_read(PETROSIAN, mode="decide"))
