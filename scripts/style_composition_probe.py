"""Composition thermometer — does ORDER move the style needle the marginals can't?

The finding U1 made concrete: single-move frames flood or vanish (18/26 candidates fire the same
`restrains`, or 25/29 render empty). A bag-of-(verb,object) captures which atoms are PRESENT, never
which follow which. Style is the multi-turn rule ("restrain, THEN strike") — a claim about ORDER.

This is the proof-of-concept probe, NOT a resolver: does adding an order channel lift the leave-one-out
nearest-neighbour style-match ABOVE the marginal baseline on the same 19-master data? If yes, composition
carries style signal the histogram structurally cannot — the green light for U3 (richer atoms) to feed a
metric that can see their combination. If no, adjacency is the wrong composition operator and the next
probe is a wider co-occurrence window, not more atoms.

Three representations, identical NN metric, so any delta is attributable to the added structure:
  MARGINAL      TF-IDF over (verb, obj) unigrams        — the current ceiling (== action_profile.py)
  BIGRAM        TF-IDF over consecutive action pairs    — composition ALONE
  MARGINAL+BIGRAM  concatenation                        — does order ADD to presence?
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

ROSTER = Path(__file__).resolve().parent.parent / "data" / "genesis_roster"
# optional CLI arg: a stories filename under data/genesis_roster/ (e.g. roster_stories_v2.json) to probe
# a re-render; defaults to the v1 baseline.
import sys  # noqa: E402
_STORIES_FILE = sys.argv[1] if len(sys.argv) > 1 else "roster_stories.json"
stories = json.loads((ROSTER / _STORIES_FILE).read_text())
roster = json.loads((ROSTER / "roster.json").read_text())
LABELS = {k: v["style"] for k, v in roster.items()}
# roster.json stores these two with a placeholder '?'; the established baseline patched them by hand.
# Restored verbatim so this probe reproduces that baseline exactly (else both drop to 'uni').
LABELS.update({"nimzowitsch": "hypermodern", "smyslov": "harmonious"})

# Coarse taxonomy (dyn/sol/uni) — the 78% ceiling was measured on this split.
DYN = {"attacker", "deep-tactics", "initiator", "unorthodox", "counter-puncher", "swindler"}
SOL = {"technician", "defender", "harmonious", "hypermodern", "engine", "positional"}


def coarse(style: str) -> str:
    return "dyn" if style in DYN else ("sol" if style in SOL else "uni")


def player_action_stream(story: str, short: str) -> list[tuple[str, str]]:
    """The player's OWN actions in one game, in move order (drops Opponent sentences).
    Each action is (verb, last-token-object) — the same unit action_profile.py counts."""
    acts: list[tuple[str, str]] = []
    for sent in story.split("."):
        toks = sent.strip().split()
        if len(toks) >= 2 and toks[0].lower() == short.lower():
            acts.append((toks[1], toks[-1]))
    return acts


_PREC_WINDOW = 4   # "A precedes B within a few of the player's own moves" — captures restrain-THEN-strike


def build_profiles(mode: str) -> dict[str, collections.Counter]:
    """mode: 'uni' | 'bi' | 'prec' | 'uni+prec'.
      uni   — unigram marginals (the 78%-coarse baseline)
      bi    — strictly CONSECUTIVE action pairs (adjacency: does the very next move matter?)
      prec  — DIRECTED PRECEDENCE within a window: A before B, distance 1.._PREC_WINDOW (not just next).
              This is the "restrain … [gap] … strike" operator adjacency can't see.
    All pairs are within-game (never bridge games)."""
    prof: dict[str, collections.Counter] = {}
    for short, games in stories.items():
        c: collections.Counter = collections.Counter()
        for story in games:
            stream = player_action_stream(story, short)
            if mode in ("uni", "uni+prec"):
                for a in stream:
                    c[("u",) + a] += 1
            if mode == "bi":
                for a, b in zip(stream, stream[1:]):
                    c[("b",) + a + b] += 1
            if mode in ("prec", "uni+prec"):
                for i, a in enumerate(stream):
                    for b in stream[i + 1:i + 1 + _PREC_WINDOW]:   # A precedes B within the window
                        c[("p",) + a + b] += 1
        prof[short] = c
    return prof


def tfidf_vectors(prof: dict[str, collections.Counter]) -> dict[str, dict]:
    keys = {k for c in prof.values() for k in c}
    n = len(prof)
    df = {k: sum(1 for c in prof.values() if c.get(k, 0) > 0) for k in keys}
    # IDF floor 1e-6 (NOT the usual +1.0): frames present in ~every player (develops/captures/checks)
    # collapse to ~0 weight. That zeroing of the universals IS what the established 78%-coarse baseline
    # rides on — style lives in the RARE frames. Held fixed here so the bigram channel is the only change.
    idf = {k: math.log((n + 1) / (df[k] + 1)) + 1e-6 for k in keys}
    vecs: dict[str, dict] = {}
    for name, c in prof.items():
        total = sum(c.values()) or 1
        vecs[name] = {k: (v / total) * idf[k] for k, v in c.items()}
    return vecs


def cos(a: dict, b: dict) -> float:
    common = a.keys() & b.keys()
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def score(mode: str) -> tuple[int, int, int, int]:
    vecs = tfidf_vectors(build_profiles(mode))
    names = list(vecs)
    fine = coarse_hit = coarse_tot = 0
    for n in names:
        nn = max((m for m in names if m != n), key=lambda m: cos(vecs[n], vecs[m]))
        fine += LABELS[n] == LABELS[nn]
        if coarse(LABELS[n]) != "uni":
            cand = [m for m in names if m != n and coarse(LABELS[m]) != "uni"]
            best = max(cand, key=lambda m: cos(vecs[n], vecs[m]))
            coarse_hit += coarse(LABELS[n]) == coarse(LABELS[best])
            coarse_tot += 1
    return fine, len(names), coarse_hit, coarse_tot


print(f"=== Composition thermometer — {len(stories)} masters, leave-one-out NN ===\n")
print(f"{'representation':<18}{'fine-match':<16}{'coarse dyn/sol':<16}")
print("-" * 50)
for mode, label in [("uni", "MARGINAL"), ("bi", "BIGRAM(adjacent)"),
                    ("prec", f"PRECEDENCE(w={_PREC_WINDOW})"), ("uni+prec", "MARGINAL+PRECEDENCE")]:
    f, ftot, c, ctot = score(mode)
    print(f"{label:<18}{f}/{ftot} = {f/ftot:4.0%}     {c}/{ctot} = {c/ctot:4.0%}")
print("\nchance fine ≈ 1/(#styles); chance coarse = 50%. "
      "Composition earns its place only if it beats MARGINAL.")
