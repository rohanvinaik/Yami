"""Fetch each master's LOSSES (the negative-lens corpus) and DRAWS (the future equilibrium-lens corpus)
from 365chess — result-filtered, ~50/player, crash-safe. Losses and draws are SEPARATE corpora: a loss is
the style breaking; a draw is often an intentional goal (the prophylactic hold), a different story entirely.

The positive lens (build_lens_store.py) uses the roster's ALL-games PGNs (mostly wins); this pulls the
games those curated sets are too sparse in. Output:
  data/genesis_roster/pgn_losses/roster_{short}.pgn   — target lost
  data/genesis_roster/pgn_draws/roster_{short}.pgn    — target drew
"""
from pathlib import Path

from fetch_365 import get_player_pgns

ROSTER_DIR = Path(__file__).resolve().parent.parent / "data" / "genesis_roster"
N = 50

# Canonical 19-master roster (precise "Last, First" 365chess names + short key + style), matching roster.json.
ROSTER = [
    ("Tal, Mikhail", "Tal", "attacker"),
    ("Kasparov, Garry", "Kasparov", "attacker"),
    ("Morphy, Paul", "Morphy", "attacker"),
    ("Capablanca, Jose Raul", "Capablanca", "technician"),
    ("Karpov, Anatoly", "Karpov", "defender"),
    ("Petrosian, Tigran V", "Petrosian", "defender"),
    ("Fischer, Robert James", "Fischer", "universal"),
    ("Nimzowitsch, Aaron", "Nimzowitsch", "hypermodern"),   # 365chess spells it "Aaron"
    ("Alekhine, Alexander", "Alekhine", "deep-tactics"),
    ("Polgar, Judit", "Polgar", "initiator"),
    ("Korchnoi, Viktor", "Korchnoi", "counter-puncher"),
    ("Lasker, Emanuel", "Lasker", "swindler"),
    ("Marshall, Frank", "Marshall", "swindler"),
    ("Smyslov, Vassily", "Smyslov", "harmonious"),          # 365chess spells it "Vassily"
    ("Rubinstein, Akiba", "Rubinstein", "harmonious"),
    ("Nezhmetdinov, Rashid", "Nezhmetdinov", "unorthodox"),
    ("Rapport, Richard", "Rapport", "unorthodox"),
    ("Caruana, Fabiano", "Caruana", "engine"),
    ("So, Wesley", "So", "engine"),
]


def fetch(want: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {want.upper()}S → {out_dir} (target {N}/player) ===", flush=True)
    for name, short, style in ROSTER:
        f = out_dir / f"roster_{short}.pgn"
        if f.exists():                                          # crash-safe resume
            n = f.read_text().count('[White ')
            print(f"  skip (have {n:3d}): {short}", flush=True)
            continue
        try:
            pgns, target = get_player_pgns(name, short, want, N)
        except Exception as e:                                  # noqa: BLE001 — report, keep going
            pgns, target = [], f"ERR:{e}"
        if pgns:
            f.write_text("\n\n\n".join(pgns))
        print(f"  [{'OK ' if pgns else 'THIN'}] {short:14s} ({style:15s}) target={target} {want}s={len(pgns)}", flush=True)


def main() -> None:
    fetch("loss", ROSTER_DIR / "pgn_losses")
    fetch("draw", ROSTER_DIR / "pgn_draws")


if __name__ == "__main__":
    main()
