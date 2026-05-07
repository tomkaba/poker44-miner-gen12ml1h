#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/copy-for-release.sh [--src <repo_path>] [--dst <release_path>] [--clean]

Defaults:
  --src  inferred from this script's parent repo
  --dst  ~/Poker44-subnet-release

Behavior:
  - Copies only miner-release files.
  - Does not copy analysis/, scripts/, tests/, data/, logs/, or local artifacts.
  - Preserves the destination .git directory if it already exists.

Examples:
  scripts/copy-for-release.sh
  scripts/copy-for-release.sh --dst ~/Poker44-subnet-release
  scripts/copy-for-release.sh --clean
USAGE
}

SRC_REPO=""
DST_REPO="${HOME}/Poker44-subnet-release"
DO_CLEAN="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src)
      SRC_REPO="${2:-}"
      shift 2
      ;;
    --dst)
      DST_REPO="${2:-}"
      shift 2
      ;;
    --clean)
      DO_CLEAN="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SRC_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_REPO="${SRC_REPO:-$DEFAULT_SRC_REPO}"

if [[ ! -d "$SRC_REPO" ]]; then
  echo "Source repo does not exist: $SRC_REPO" >&2
  exit 1
fi

if [[ ! -f "$SRC_REPO/neurons/miner.py" || ! -f "$SRC_REPO/poker44/miner_heuristics.py" ]]; then
  echo "Source repo does not look like Poker44-subnet-main: $SRC_REPO" >&2
  exit 1
fi

mkdir -p "$DST_REPO"
mkdir -p "$DST_REPO/neurons" "$DST_REPO/weights" "$DST_REPO/models" "$DST_REPO/docs"

echo "Source:      $SRC_REPO"
echo "Destination: $DST_REPO"

MANAGED_FILES=(
  "README.md"
  "requirements.txt"
  "pyproject.toml"
  "setup.py"
  "LICENSE"
  "neurons/__init__.py"
  "neurons/miner.py"
  "docs/miner.md"
)

MANAGED_DIRS=(
  "poker44"
  "weights"
  "models"
)

if [[ "$DO_CLEAN" == "true" ]]; then
  echo "Cleaning managed paths in destination"
  for rel_path in "${MANAGED_FILES[@]}"; do
    rm -f "$DST_REPO/$rel_path"
  done
  for rel_path in "${MANAGED_DIRS[@]}"; do
    rm -rf "$DST_REPO/$rel_path"
  done
  mkdir -p "$DST_REPO/neurons" "$DST_REPO/weights" "$DST_REPO/models" "$DST_REPO/docs"
fi

copy_file() {
  local rel_path="$1"
  local src_path="$SRC_REPO/$rel_path"
  local dst_path="$DST_REPO/$rel_path"

  if [[ ! -f "$src_path" ]]; then
    echo "Missing expected file: $src_path" >&2
    exit 1
  fi

  mkdir -p "$(dirname "$dst_path")"
  cp "$src_path" "$dst_path"
}

echo "Copying top-level and miner entry files"
for rel_path in "${MANAGED_FILES[@]}"; do
  copy_file "$rel_path"
done

echo "Syncing poker44 package"
rsync -a --delete \
  --exclude='__pycache__/' \
  --exclude='assets/' \
  "$SRC_REPO/poker44/" "$DST_REPO/poker44/"

echo "Syncing weights"
rm -rf "$DST_REPO/weights"
mkdir -p "$DST_REPO/weights"

WEIGHT_FILES=(
  "ml_filtered0_model.pkl"
  "ml_filtered0_scaler.pkl"
  "ml_filtered1_nonhardcut_model.pkl"
  "ml_filtered1_nonhardcut_scaler.pkl"
  "ml_filtered1_nonhardcut_metrics.json"
)

for rel_name in "${WEIGHT_FILES[@]}"; do
  src_path="$SRC_REPO/weights/$rel_name"
  if [[ -f "$src_path" ]]; then
    cp "$src_path" "$DST_REPO/weights/$rel_name"
  else
    echo "Warning: optional weight artifact missing: $src_path" >&2
  fi
done

echo "Syncing models"
rm -rf "$DST_REPO/models"
mkdir -p "$DST_REPO/models"

MODEL_FILES=(
  "benchmark_heuristic_profile.json"
  "benchmark_lgbm_model.pkl"
  "benchmark_lgbm_profile.json"
)

for rel_name in "${MODEL_FILES[@]}"; do
  src_path="$SRC_REPO/models/$rel_name"
  if [[ -f "$src_path" ]]; then
    cp "$src_path" "$DST_REPO/models/$rel_name"
  else
    echo "Warning: optional model artifact missing: $src_path" >&2
  fi
done

echo "Done. Release snapshot updated in: $DST_REPO"
echo "Next steps:"
echo "  cd \"$DST_REPO\""
echo "  git status"
echo "  git add . && git commit -m 'Update release snapshot'"