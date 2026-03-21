# Yami 闇

**The missing infrastructure layer for LLM chess.**

294,000 parameters. 77 seconds training. ~1,800 ELO. $0.

---

## What This Is

Yami is a chess compiler — a system that decomposes chess decision-making into structured phases, handles 90% of them deterministically, and hands a tiny neural model a 5-way recognition problem instead of an open-ended search problem.

| | Yami | GPT-5 | Claude Opus 4.5 |
|---|---|---|---|
| **ELO** | ~1,800 | 1,087 | 446 |
| **Parameters** | 294K | ~1T+ | ~1T+ |
| **Cost/game** | $0 | ~$5-10 | ~$2-5 |
| **Training** | 77 seconds, CPU | Months, GPU clusters | Months, GPU clusters |
| **Illegal moves** | 0 (guaranteed) | 2.0/1k | 0 |
| **Inference** | <10ms | ~5-10s | ~5-10s |

## Architecture

```
Board position
  → Layer 1: Legal move generation (python-chess, microseconds)
  → Layer 2: Tactical scoping (fork/pin/check detection, ~1ms)
  → Layer 3: Censor stack (suppress blunders, hanging pieces)
  → Layer 4: Endgame tables + opening book (Syzygy, Polyglot)
  → Layer 5: 6-Bank Navigator (729-bin ternary position navigation)
  → Layer 5b: K-Line Memory (7,723 winning pattern templates)
  → Layer 5c: Temporal Controller (4-phase FSM, 10 strategic plans)
  → Layer 6: Candidate filtering (30-dim annotated candidates)
  → Layer 7: Neural selection (294K Balanced Sashimi ternary model)
  → Layer 8: Legal verification (the kernel is the judge)
```

## The Thesis

The LLM chess benchmark has 157 models. Trillions of parameters. Dollars per game. Half of them can't follow the rules.

Most of chess isn't hard for AI — it's been *misframed* as hard by treating it as monolithic generation instead of structured compilation. Legal moves, tactical patterns, opening theory, endgame tablebases, positional evaluation — all deterministic, all microsecond-cost. The expensive neural model should only see the hard tail: a 5-way recognition problem among pre-validated candidates.

The scaling paradigm optimizes model size vs. performance. The infrastructure-first paradigm optimizes **infrastructure quality vs. residual hardness**. Yami demonstrates that the second frontier is dramatically more efficient.

## Results

**ELO calibration** (50 games vs Stockfish 18 at depth 1, ~1,900-2,000 ELO):
- Full stack: **29 draws / 50 games (58% draw rate)**, 142 avg moves
- Infrastructure only: 1 draw / 50 games
- Neural model adds **+640 ELO** over infrastructure alone

**Ablation** (400 games across 4 Stockfish configurations):

| Configuration | vs SF depth 1 | vs SF depth 3 |
|---|---|---|
| Full stack (nav + temporal + neural) | 29D/50G, 142 avg | 1D/50G, 88 avg |
| Navigator + temporal (no neural) | 2D/50G, 82 avg | 0D/50G, 55 avg |
| Infrastructure only | 1D/50G, 83 avg | 0D/50G, 73 avg |

## Lineage

Yami is the chess instantiation of a cross-domain research program:

| Domain | System | Result |
|--------|--------|--------|
| Theorem Proving | [Wayfinder](https://github.com/rohanvinaik/Wayfinder) | 63% of Mathlib, 22M params, laptop |
| Chess | **Yami** | ~1,800 ELO, 294K params, $0 |
| Code Quality | LintGate | Deterministic constraint checking |
| Program Synthesis | ShortcutForge | 85% compilation via linter pipeline |

Each domain: most of the problem is structural. Structure is cheap. The neural model sees only the residual.

## Quick Start

```bash
# Install
git clone https://github.com/rohanvinaik/Yami.git
cd Yami
pip install -e ".[dev]"

# Play (infrastructure only)
python -c "
from yami.engine import YamiEngine
engine = YamiEngine(use_llm=False, use_navigator=True, use_temporal=True)
decision = engine.decide()
print(f'Move: {engine.board.san(decision.move)} ({decision.source.value})')
"

# Train the neural model
python scripts/generate_data.py --num-train 50000 --num-eval 2000
python scripts/train.py --iterations 5000

# Benchmark ELO
python scripts/benchmark_elo.py --games 50
```

## Name

闇 (yami) — darkness, the unseen. The infrastructure works in the dark. The model sees only the clean residual.

---

*85 tests. Lint clean. Zero illegal moves. The Wayfinder thesis applied to chess.*
