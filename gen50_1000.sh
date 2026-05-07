#!/bin/bash
for i in $(seq 1 50); do
  seed=$((2000 + i))
  echo "generating set {$i}"
  python scripts/publish/publish_public_benchmark.py \
    --skip-wandb \
    --chunk-count 1000 \
    --min-hands-per-chunk 1 \
    --max-hands-per-chunk 1 \
    --seed "$seed" \
    --output-path "data/public_benchmark_1hand_1000_set5_seed_${seed}.json.gz" \
    --verbose
done
