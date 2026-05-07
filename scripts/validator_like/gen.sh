cd /home/tk/Poker44-subnet-main
mkdir -p analysis/data/validator_like_200x100

total=200
for i in $(seq -w 1 $total); do
  window_id=$((120000 + 10#$i))
  out="analysis/data/validator_like_200x100/validator_like_window_${i}.json"

  echo "[$(date '+%H:%M:%S')] START iteracja ${i}/${total} | window_id=${window_id} | out=${out}"

  ./scripts/validator_like/generate_chunks.py \
    --output "$out" \
    --cache-path analysis/data/validator_provider_cache.json \
    --chunk-count 100 \
    --limit 100 \
    --human-ratio 0.5 \
    --window-id "$window_id"

  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date '+%H:%M:%S')] ERROR iteracja ${i}/${total} (exit=${rc})"
    break
  fi

  echo "[$(date '+%H:%M:%S')] DONE  iteracja ${i}/${total}"
done