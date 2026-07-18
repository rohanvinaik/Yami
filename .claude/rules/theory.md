---
paths:
  - "**/*.py"
---

# Theory Rules

This file stores extracted theory signals for `Yami`.

## Facet Summaries
- Core Theory: No strong signal extracted for this facet yet.
- Problem-Solving: It handles the structural layers, reduces the decision space from ~30 legal moves to ~3-5 annotated candidates, and hands the LLM a recognition problem instead of a search problem.
- Alignment: It handles the structural layers, reduces the decision space from ~30 legal moves to ~3-5 annotated candidates, and hands the LLM a recognition problem instead of a search problem.
- Architecture: Together they say: the scaling paradigm is optimizing the wrong frontier. Not model size vs. performance.
- Anti-Patterns: No strong signal extracted for this facet yet.
- Key Abstractions: No strong signal extracted for this facet yet.

## High-Signal Anti-Patterns
- Do not try a 4th approach without first enumerating all known constraints and verifying which ones the new approach actually addresses.
- Do not discover constraints one-at-a-time through failure — enumerate the full constraint space upfront by reading before acting.
- Do not re-attempt an approach that already failed unless the conditions that caused the failure have changed.
- Do not use O(n²) algorithms when O(n) alternatives exist — quadratic membership checks on lists, re.compile inside loops, and sorted()[0] instead of min() are structural mistakes, not style issues.
- Do not treat N instances of the same root cause as N separate problems — cluster issues by shared fix before diving into individual repairs.

## Enforceable Rules
- No enforceable rules extracted yet.

## Extraction Quality
- Validity status: weak
- Docs scanned: 1
- Total claims: 3
- Missing required facets: core_theory
- Warning: Missing required facets: core_theory
- Warning: No enforceable rules found (existing or proposed).
