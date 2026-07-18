#!/usr/bin/env python3
"""Enrich SoM training data with LLM position annotations via Ollama.

Uses Qwen 3.5 to evaluate each position along dimensions that
deterministic signals miss. The LLM is a TEACHER at training time —
the trained model learns to approximate these rich annotations from
cheap deterministic signals at inference time.

Reads existing SoM JSONL, adds 12 LLM signal dimensions, writes enriched JSONL.
Crash-safe: writes incrementally, tracks progress, can resume.

Usage:
    python scripts/enrich_with_llm.py --input data/som_merged_train.jsonl --output data/som_enriched_train.jsonl
    python scripts/enrich_with_llm.py --input data/som_merged_train.jsonl --model qwen2.5:3b  # faster
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import urllib.request


PROMPT_TEMPLATE = """Chess FEN: {fen}
Rate this position as JSON: {{king_safety_w, king_safety_b, piece_activity_w, piece_activity_b, pawn_quality_w, pawn_quality_b, space(-5to5), tension, attack_w, attack_b, initiative(-5to5), complexity}} all 1-10 unless noted."""

LLM_SIGNAL_KEYS = [
    "king_safety_w", "king_safety_b",
    "piece_activity_w", "piece_activity_b",
    "pawn_quality_w", "pawn_quality_b",
    "space", "tension",
    "attack_w", "attack_b",
    "initiative", "complexity",
]


def query_ollama(fen, model="qwen3.5:9b", host="http://localhost:11434"):
    """Query Ollama for structured position evaluation."""
    prompt = PROMPT_TEMPLATE.format(fen=fen)

    try:
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_predict": 200},
        }).encode()
        req = urllib.request.Request(
            f"{host}/api/generate", data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=300)
        data = json.loads(resp.read())

        # Qwen 3.5 puts JSON in "thinking" field, Qwen 2.5 in "response"
        raw = data.get("thinking", "") or data.get("response", "")
        if not raw:
            return None

        parsed = json.loads(raw)
        # Validate and normalize
        signals = {}
        for key in LLM_SIGNAL_KEYS:
            val = parsed.get(key, 5)
            if isinstance(val, (int, float)):
                signals[key] = float(val)
            else:
                signals[key] = 5.0  # default neutral
        return signals

    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Enrich SoM data with LLM signals")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load input
    with open(input_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    # Resume support
    already_done = 0
    if args.resume and output_path.exists():
        with open(output_path) as f:
            already_done = sum(1 for l in f if l.strip())
        print(f"Resuming from {already_done}/{len(lines)}", flush=True)

    print(f"=== LLM Signal Enrichment ===", flush=True)
    print(f"Input: {len(lines)} examples", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Host: {args.host}", flush=True)
    print(flush=True)

    t0 = time.time()
    enriched = 0
    failed = 0
    mode = "a" if args.resume else "w"

    with open(output_path, mode) as out_f:
        for i, line in enumerate(lines):
            if i < already_done:
                continue

            ex = json.loads(line)
            fen = ex.get("fen", "")

            signals = query_ollama(fen, model=args.model, host=args.host)

            if signals:
                ex["llm_signals"] = signals
                enriched += 1
            else:
                ex["llm_signals"] = {k: 5.0 for k in LLM_SIGNAL_KEYS}
                failed += 1

            out_f.write(json.dumps(ex) + "\n")

            if (i + 1) % args.batch_size == 0:
                elapsed = time.time() - t0
                rate = (enriched + failed) / max(elapsed, 1) * 60
                remaining = len(lines) - i - 1
                eta = remaining / max(rate, 1)
                out_f.flush()
                print(
                    f"  {i+1}/{len(lines)} | enriched={enriched} failed={failed} | "
                    f"{rate:.1f}/min | ETA {eta:.0f}min",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"\n=== Done in {elapsed:.0f}s ===", flush=True)
    print(f"Enriched: {enriched}, Failed: {failed}, Total: {len(lines)}", flush=True)
    print(f"Output: {output_path}", flush=True)


if __name__ == "__main__":
    main()
