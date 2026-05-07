#!/usr/bin/env bash
set -euo pipefail

# Generate 5,000 one-hand chunks for:
# 1) each single bot profile (--bot-profiles <profile>)
# 2) each preset (--bot-profile-preset <preset>)
#
# Output is written to a dedicated subdirectory under data/ with timestamp.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

CHUNK_COUNT="${CHUNK_COUNT:-5000}"
SEED_BASE="${SEED_BASE:-44}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/data/public_benchmark_custom_1hand_${CHUNK_COUNT}_${RUN_TAG}}"
LOG_FILE="$OUT_DIR/generation.log"

mkdir -p "$OUT_DIR"

echo "[start] repo=$REPO_ROOT" | tee -a "$LOG_FILE"
echo "[start] python=$PYTHON_BIN" | tee -a "$LOG_FILE"
echo "[start] out_dir=$OUT_DIR" | tee -a "$LOG_FILE"
echo "[start] chunk_count=$CHUNK_COUNT min_hands=1 max_hands=1 progress_every=500" | tee -a "$LOG_FILE"

declare -a PROFILES=(
  "balanced"
  "tight_aggressive"
  "loose_aggressive"
  "tight_passive"
  "loose_passive"
)

declare -a PRESETS=(
  "default_mix"
  "balanced_only"
  "tight_aggressive_only"
  "loose_aggressive_only"
  "tight_passive_only"
  "loose_passive_only"
  "aggressive_mix"
  "passive_mix"
  "tight_mix"
  "loose_mix"
  "no_balanced"
)

seed="$SEED_BASE"

generate_profile_dataset() {
  local profile="$1"
  local output="$OUT_DIR/public_1hand_${CHUNK_COUNT}_profile_${profile}.json.gz"

  echo "[profile] generating profile=$profile seed=$seed -> $output" | tee -a "$LOG_FILE"
  "$PYTHON_BIN" scripts/publish/publish_public_benchmark_custom.py \
    --skip-wandb \
    --output-path "$output" \
    --chunk-count "$CHUNK_COUNT" \
    --min-hands-per-chunk 1 \
    --max-hands-per-chunk 1 \
    --seed "$seed" \
    --progress-every 500 \
    --bot-profiles "$profile" \
    --verbose | tee -a "$LOG_FILE"

  seed=$((seed + 1))
}

generate_preset_dataset() {
  local preset="$1"
  local output="$OUT_DIR/public_1hand_${CHUNK_COUNT}_preset_${preset}.json.gz"

  echo "[preset] generating preset=$preset seed=$seed -> $output" | tee -a "$LOG_FILE"
  "$PYTHON_BIN" scripts/publish/publish_public_benchmark_custom.py \
    --skip-wandb \
    --output-path "$output" \
    --chunk-count "$CHUNK_COUNT" \
    --min-hands-per-chunk 1 \
    --max-hands-per-chunk 1 \
    --seed "$seed" \
    --progress-every 500 \
    --bot-profile-preset "$preset" \
    --verbose | tee -a "$LOG_FILE"

  seed=$((seed + 1))
}

for profile in "${PROFILES[@]}"; do
  generate_profile_dataset "$profile"
done

for preset in "${PRESETS[@]}"; do
  generate_preset_dataset "$preset"
done

echo "[done] generated $(( ${#PROFILES[@]} + ${#PRESETS[@]} )) datasets in $OUT_DIR" | tee -a "$LOG_FILE"
ls -lh "$OUT_DIR" | tee -a "$LOG_FILE"
