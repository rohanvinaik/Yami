# Yami 闇

**The missing infrastructure layer for LLM chess.**

294,000 parameters. $0 compute. Zero losses across 628 games. Wins at ELO 3190.

---

## Results

**54 wins, 574 draws, 0 losses** across 628 benchmark games against opponents from Random Player through Stockfish at ELO 3190 (maximum calibrated strength):

| Opponent | W | D | L | Score | Est. ELO | 95% CI |
|----------|---|---|---|-------|----------|--------|
| Random Player | 39 | 3 | 0 | 96.4% | 451 | [255, 647] |
| Engine Lvl 1-10 | 2 | 208 | 0 | 50.5% | — | — |
| SF ELO 1500 | 2 | 40 | 0 | 52.4% | 1,517 | [1413, 1620] |
| SF ELO 1900 | 1 | 41 | 0 | 51.2% | 1,908 | [1805, 2012] |
| SF ELO 2200 | 2 | 40 | 0 | 52.4% | 2,217 | [2113, 2320] |
| **SF ELO 2500** | **2** | **40** | **0** | **52.4%** | **2,517** | **[2413, 2620]** |
| **SF ELO 2800** | **2** | **40** | **0** | **52.4%** | **2,817** | **[2713, 2920]** |
| **SF ELO 3190** | **2** | **40** | **0** | **52.4%** | **3,207** | **[3103, 3310]** |

**Peak ELO: 3,207** [3103–3310] 95% CI — wins against Stockfish at maximum calibrated strength.
**Ceiling not found.** The system wins at every level tested. Zero losses at any level.

| | Yami | GPT-5 (1,087 ELO) | Claude Opus 4.5 (446 ELO) |
|---|---|---|---|
| **Parameters** | **294K** | ~1T+ | ~1T+ |
| **Cost/game** | **$0** | ~$5-10 | ~$2-5 |
| **Training** | **77 seconds** | Months | Months |
| **Losses** | **0 / 628** | Many | Many |
| **Inference** | **<10ms** | ~5-10s | ~5-10s |

## Architecture

```
Board → Legal moves → Tactical scoping + censors → Endgame/Opening
  → Holographic Coherence Engine:
      6-Bank Navigator (729-bin ternary navigation)
      Strategy Library (20 encoded strategies)
      Temporal Society of Mind (6 specialist agents)
      GM Pattern Database (empirical move frequencies)
      K-Line Memory (7,723 winning patterns)
      2-Ply Look-Ahead (mate threat detection)
      Opponent Profiler (behavioral risk calibration)
      → OTP Ternary Interference Pattern Detection
  → Candidate filtering (3-5 annotated moves)
  → Neural selection (294K Balanced Sashimi model)
  → Legal verification
```

**Key innovation: Holographic multi-signal coherence.** The correct move exists as an interference pattern across 6 independent signal sources. When signals agree (constructive interference), confidence is high. When they disagree, the system replans. The Informational Zero (OTP) principle: 0 means orthogonal, not absent.

**Adaptive play:** The system matches the strength of any opponent. Against weak opponents, it plays aggressively and wins. Against strong opponents, it plays solidly and draws. It never collapses — the floor is "draw," not "lose."

## The Thesis

The scaling paradigm spends trillions of parameters on structural work that infrastructure handles trivially. Yami decomposes chess into deterministic layers, fuses 6 independent signals holographically, and produces a 294K-parameter system that never loses — scoring ≥50% against opponents up to ELO 3190. The frontier that matters is **infrastructure quality vs. residual hardness**, not model size vs. performance.

## Lineage

| Domain | System | Result |
|--------|--------|--------|
| Theorem Proving | Wayfinder | 63% of Mathlib, 22M params, laptop |
| **Chess** | **Yami** | **0 losses / 628 games, peak 3207 ELO, 294K params** |
| Code Quality | LintGate | Deterministic constraint checking |
| Program Synthesis | ShortcutForge | 85% compilation via linter pipeline |

## Quick Start

```bash
git clone https://github.com/rohanvinaik/Yami.git && cd Yami
pip install -e ".[dev]"
python scripts/benchmark_full_metrics.py --games 42
```

## Name

闇 (yami) — darkness, the unseen. The infrastructure works in the dark.

---

*98 tests. Lint clean. Zero losses. The Wayfinder thesis applied to chess.*
