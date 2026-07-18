"""The NEGATIVE (surprise) lens — the win thermometer's method, ported to the mistake.

Why the raw Genesis-outcome profiler failed to discriminate Tal from Capablanca: it counted DERIVED
outcomes (`obtain`/`dominant`) with RAW counts — the exact below-chance lens the win side abandoned. The
win discrimination (style_composition_probe.py, 78% coarse) rides on three things it dropped, all ported
here: (1) the ACTION stream (the player's rendered move-verbs, not outcomes); (2) TF-IDF with the +1e-6
IDF-FLOOR that ZEROES the universals (obtain/develop/capture — "style lives in the RARE frames"); (3) the
LOO-NN clustering test as the metric.

Two measurements, same IDF machinery:
  LOSS-STYLE  — LOO-NN on the IDF-weighted action profile of each master's LOSSES (does loss-style
                discriminate at all, like win-style?).
  MISTAKE     — the SURPRISE, localized in action-space: what each master does MORE in losses than in
                their normal (roster) play = loss_vec − normal_vec (shared IDF), then LOO-NN. This is the
                SSL surprise (divergence of naive[normal] vs context-constrained[loss]) — the characteristic
                mistake, not the shared style.

Honest ceiling (user): these are the best players alive; their losing mistakes were subtle enough they
missed them in the moment — so expect a WEAKER signal than wins, and fine mistake-discrimination is
U3-gated (the subtle positional error may not be in the current vocabulary). The clustering test judges.
Pure symbolic (no Genesis fire) — the STYLE lens, exactly as the win probe."""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from yami.datagen.game_renderer import render_game  # noqa: E402

ROSTER = Path(__file__).resolve().parent.parent / "data" / "genesis_roster"
roster = json.loads((ROSTER / "roster.json").read_text())
LABELS = {k: v["style"] for k, v in roster.items()}
LABELS.update({"nimzowitsch": "hypermodern", "smyslov": "harmonious"})   # the win-probe's hand-patch, verbatim
DYN = {"attacker", "deep-tactics", "initiator", "unorthodox", "counter-puncher", "swindler"}
SOL = {"technician", "defender", "harmonious", "hypermodern", "engine", "positional"}
KEYS = list(roster)
NORMAL = json.loads((ROSTER / "roster_stories.json").read_text())   # the win/normal baseline (positive corpus)


def coarse(style: str) -> str:
    return "dyn" if style in DYN else ("sol" if style in SOL else "uni")


def player_action_stream(story: str, short: str) -> list[tuple[str, str]]:
    """The player's OWN actions in one game (drops Opponent sentences) — ported verbatim from the win probe."""
    acts: list[tuple[str, str]] = []
    for sent in story.split("."):
        toks = sent.strip().split()
        if len(toks) >= 2 and toks[0].lower() == short.lower():
            acts.append((toks[1], toks[-1]))
    return acts


def render_losses(key: str) -> list[str]:
    """Render a master's loss games to action-stream stories (no outcome fact — the stream is their MOVES)."""
    pgn = ROSTER / "pgn_losses" / f"roster_{key}.pgn"
    out: list[str] = []
    if not pgn.exists():
        return out
    with pgn.open() as f:
        while (g := chess.pgn.read_game(f)) is not None:
            ws = g.headers.get("White", "").split(",")[0].strip()
            white, black = (ws, "Opponent") if key.lower() in ws.lower() else ("Opponent", key)
            s = render_game(g, white=white, black=black)
            if s:
                out.append(" ".join(s))
    return out


def action_profile(stories: list[str], short: str) -> collections.Counter:
    c: collections.Counter = collections.Counter()
    for story in stories:
        for a in player_action_stream(story, short):
            c[a] += 1
    return c


def _idf(profs: dict[str, collections.Counter]) -> dict:
    keys = {k for c in profs.values() for k in c}
    n = len(profs)
    df = {k: sum(1 for c in profs.values() if c.get(k, 0) > 0) for k in keys}
    return {k: math.log((n + 1) / (df[k] + 1)) + 1e-6 for k in keys}   # the 1e-6 floor — zero the universals


def _vec(prof: collections.Counter, idf: dict) -> dict:
    total = sum(prof.values()) or 1
    return {k: (v / total) * idf.get(k, 1e-6) for k, v in prof.items()}


def cos(a: dict, b: dict) -> float:
    common = a.keys() & b.keys()
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def loo_nn(vecs: dict[str, dict]) -> tuple[int, int, int, int]:
    """Leave-one-out nearest-neighbour style-match — the win probe's metric, verbatim."""
    names = [n for n in vecs if vecs[n]]
    fine = coarse_hit = coarse_tot = 0
    for n in names:
        nn = max((m for m in names if m != n), key=lambda m: cos(vecs[n], vecs[m]))
        fine += LABELS[n] == LABELS[nn]
        if coarse(LABELS[n]) != "uni":
            cand = [m for m in names if m != n and coarse(LABELS[m]) != "uni"]
            if cand:
                best = max(cand, key=lambda m: cos(vecs[n], vecs[m]))
                coarse_hit += coarse(LABELS[n]) == coarse(LABELS[best])
                coarse_tot += 1
    return fine, len(names), coarse_hit, coarse_tot


def main() -> None:
    loss_prof = {k: action_profile(render_losses(k), k) for k in KEYS}
    norm_prof = {k: action_profile(NORMAL.get(k, []), k) for k in KEYS}
    loss_prof = {k: c for k, c in loss_prof.items() if c}          # drop empties (thin masters)

    # (1) LOSS-STYLE: IDF over the loss profiles, LOO-NN.
    lidf = _idf(loss_prof)
    loss_vecs = {k: _vec(c, lidf) for k, c in loss_prof.items()}

    # (2) MISTAKE = surprise: shared IDF, what they do MORE in losses than normal (loss_vec − norm_vec).
    shared = _idf({**{f"L_{k}": v for k, v in loss_prof.items()},
                   **{f"N_{k}": v for k, v in norm_prof.items() if v}})
    mistake_vecs: dict[str, dict] = {}
    for k in loss_prof:
        lv, nv = _vec(loss_prof[k], shared), _vec(norm_prof.get(k, collections.Counter()), shared)
        dev = {key: lv.get(key, 0.0) - nv.get(key, 0.0) for key in set(lv) | set(nv)}
        mistake_vecs[k] = {key: d for key, d in dev.items() if d > 0}   # positive deviation = the mistake

    print(f"=== Surprise lens — {len(loss_prof)} masters, leave-one-out NN (win baseline: 78% coarse / 0% fine) ===\n")
    print(f"{'representation':<20}{'fine-match':<16}{'coarse dyn/sol':<16}")
    print("-" * 52)
    for label, vecs in [("LOSS-STYLE", loss_vecs), ("MISTAKE (surprise)", mistake_vecs)]:
        f, ftot, c, ctot = loo_nn(vecs)
        cs = f"{c}/{ctot} = {c/ctot:.0%}" if ctot else "n/a"
        print(f"{label:<20}{f}/{ftot} = {f/ftot:.0%}     {cs}")
    print(f"\nchance fine ≈ {1/len(set(LABELS.values())):.0%}; chance coarse = 50%. "
          "The MISTAKE row is the negative lens; beating chance = the surprise discriminates style.")

    # BANK the lens: per-master loss action profile (the negative lens the downstream learning/SoM consumes)
    # + the measured LOO-NN scores. Mirrors data/genesis_roster/player_lenses_v3.json (the positive lens).
    lf, lft, lc, lct = loo_nn(loss_vecs)
    mf, mft, mc, mct = loo_nn(mistake_vecs)
    store = {
        "_scores": {"loss_style": {"fine": [lf, lft], "coarse": [lc, lct]},
                    "mistake": {"fine": [mf, mft], "coarse": [mc, mct]},
                    "win_baseline": {"coarse": 0.78, "fine": 0.0}},
        "lenses": {k: {"style": LABELS[k], "coarse": coarse(LABELS[k]),
                       "loss_profile": {f"{v}|{o}": n for (v, o), n in c.items()}}
                   for k, c in loss_prof.items()},
    }
    out = ROSTER / "player_surprise_lenses.json"
    out.write_text(json.dumps(store, indent=2))
    print(f"\nbanked negative lens → {out}  ({len(store['lenses'])} masters)")


if __name__ == "__main__":
    main()
