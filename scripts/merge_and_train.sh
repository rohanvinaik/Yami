#!/bin/bash
# Merge SoM data from both machines and kick off training.
# Run this after HomeBridge finishes its 50K batch.
#
# Usage: bash scripts/merge_and_train.sh

set -e
cd "$(dirname "$0")/.."

echo "=== Merging SoM training data ==="

# Pull HomeBridge data
echo "Pulling from HomeBridge..."
scp homebridge:~/Projects/Yami/data/som_lichess_train.jsonl data/som_lichess_hb.jsonl 2>/dev/null || echo "No HomeBridge data"

# Count what we have
HB=$(wc -l < data/som_lichess_hb.jsonl 2>/dev/null || echo 0)
LOCAL=$(wc -l < data/som_lichess_train.jsonl 2>/dev/null || echo 0)
OLD_SOM=$(wc -l < data/som_train.jsonl 2>/dev/null || echo 0)
OLD_LOCAL=$(wc -l < data/som_train_local.jsonl 2>/dev/null || echo 0)

echo "  HomeBridge Lichess: $HB"
echo "  Local Lichess:      $LOCAL"
echo "  Old self-play:      $OLD_SOM"
echo "  Old local:          $OLD_LOCAL"

# Merge all into one file
cat data/som_lichess_hb.jsonl data/som_lichess_train.jsonl data/som_train.jsonl data/som_train_local.jsonl 2>/dev/null | \
    shuf > data/som_merged_all.jsonl

TOTAL=$(wc -l < data/som_merged_all.jsonl)
SPLIT=$((TOTAL * 9 / 10))
EVAL=$((TOTAL - SPLIT))

head -n $SPLIT data/som_merged_all.jsonl > data/som_merged_train.jsonl
tail -n $EVAL data/som_merged_all.jsonl > data/som_merged_eval.jsonl

echo "  TOTAL: $TOTAL → Train: $SPLIT, Eval: $EVAL"
echo ""

# Train
echo "=== Training SoM (3-stage) ==="
.venv/bin/python scripts/train_som.py \
    --data data/som_merged_train.jsonl \
    --eval-data data/som_merged_eval.jsonl \
    --checkpoint-dir models/som_v2 \
    --iterations-1 10000 \
    --iterations-2 5000 \
    --iterations-3 2000 \
    --device mps

echo ""
echo "=== Done! Checkpoints in models/som_v2/ ==="
