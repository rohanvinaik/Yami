# Yami 闇

**The missing infrastructure layer for LLM chess.**

294,000 parameters. $0 compute. Zero losses across 210 games.

---

## Results

**45 wins, 165 draws, 0 losses** across 210 benchmark games (42 per opponent, llm_chess compatible):

| Opponent | W | D | L | Score | Est. ELO |
|----------|---|---|---|-------|----------|
| Random Player | 39 | 3 | 0 | 96.4% | ~971 |
| Stockfish Skill 0 | 1 | 41 | 0 | 51.2% | ~808 |
| Stockfish Skill 3 | 1 | 41 | 0 | 51.2% | ~1,208 |
| Stockfish Skill 5 | 1 | 41 | 0 | 51.2% | ~1,508 |
| **Stockfish Skill 8** | **3** | **39** | **0** | **53.6%** | **~1,825** |

| | Yami | GPT-5 (1,087 ELO) | Claude Opus 4.5 (446 ELO) |
|---|---|---|---|
| **Parameters** | **294K** | ~1T+ | ~1T+ |
| **Cost/game** | **$0** | ~$5-10 | ~$2-5 |
| **Training** | **77 seconds** | Months | Months |
| **Losses in benchmark** | **0** | Many | Many |
| **Inference** | **<10ms** | ~5-10s | ~5-10s |

## Architecture

Yami decomposes chess into structured layers. Each layer handles cheap work deterministically, compressing the decision space until the remaining residual is a recognition problem solvable by a tiny model.

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

**Key innovation: Holographic multi-signal coherence.** The correct move exists as an interference pattern across 6 independent signal sources. When signals agree (constructive interference), confidence is high. When they disagree (destructive interference), the system replans. No single source has the answer — it emerges from the superposition.

## The Thesis

The LLM chess benchmark has 157 models. Trillions of parameters. Dollars per game. Half can't follow the rules. Most of chess isn't hard — it's been *misframed* as hard by treating it as monolithic generation. The scaling paradigm optimizes model size vs. performance. The infrastructure-first paradigm optimizes **infrastructure quality vs. residual hardness.**

## Lineage

Yami is the chess instantiation of a cross-domain research program:

| Domain | System | Result |
|--------|--------|--------|
| Theorem Proving | [Wayfinder](https://github.com/rohanvinaik/Wayfinder) | 63% of Mathlib, 22M params, laptop |
| **Chess** | **Yami** | **0 losses / 210 games, 294K params, $0** |
| Code Quality | LintGate | Deterministic constraint checking |
| Program Synthesis | ShortcutForge | 85% compilation via linter pipeline |

Each domain: most of the problem is structural. Structure is cheap. The neural model sees only the residual.

## Quick Start

```bash
git clone https://github.com/rohanvinaik/Yami.git && cd Yami
pip install -e ".[dev]"

# Play (infrastructure only — already unbeatable)
python -c "
from yami.engine import YamiEngine
engine = YamiEngine(use_llm=False, use_navigator=True, use_temporal=True)
decision = engine.decide()
print(f'Move: {engine.board.san(decision.move)}')
"

# Benchmark
python scripts/benchmark_llm_chess.py --full-suite --games 42
```

## Name

闇 (yami) — darkness, the unseen. The infrastructure works in the dark. The model sees only the clean residual.

---

*98 tests. Lint clean. Zero losses. 20 commits.*
*The Wayfinder thesis applied to chess through holographic multi-signal coherence.*
