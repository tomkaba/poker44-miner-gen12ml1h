source /home/tk/Poker44-subnet-main/.venv/bin/activate
cd /home/tk/Poker44-subnet-main
mkdir -p data/public_benchmark_multihand_1000_set1

for i in $(seq 1 20); do
  seed=$((110001 + i))
  echo "generating multihand set ${i}/1000 (seed=${seed})"
  python scripts/publish/publish_public_benchmark_custom.py \
    --skip-wandb \
    --chunk-count 1000 \
    --min-hands-per-chunk 2 \
    --max-hands-per-chunk 120 \
    --human-ratio 0.5 \
    --seed "$seed" \
    --output-path "data/public_benchmark_multihand_1000_set1_2-70/public_benchmark_multihand_1000_set1_2-70_seed_${seed}.json.gz" \
    --progress-every 1 \
    --verbose
done



