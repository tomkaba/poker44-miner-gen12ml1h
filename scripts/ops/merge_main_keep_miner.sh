#!/usr/bin/env bash
set -euo pipefail

# Merge latest origin/main into a new local integration branch while preserving
# miner-customized files from the current branch.
#
# This script never pushes.
#
# Usage:
#   scripts/ops/merge_main_keep_miner.sh
#   scripts/ops/merge_main_keep_miner.sh --base-branch feature/pre-v0-validator-sync
#   scripts/ops/merge_main_keep_miner.sh --keep neurons/miner.py --keep poker44/miner_heuristics.py

usage() {
  cat <<'USAGE'
Usage:
  scripts/ops/merge_main_keep_miner.sh [--base-branch <branch>] [--keep <path>]...

Options:
  --base-branch <branch>  Branch to merge into (default: current branch)
  --keep <path>           File to preserve from local branch (can be repeated)
  -h, --help              Show help

Behavior:
  1) Requires a clean working tree.
  2) Fetches origin.
  3) Creates a new local branch: <base>-merge-main-<timestamp>
  4) Merges origin/main with --no-commit.
  5) Forces local versions for protected files via checkout --ours.
  6) Creates a merge commit locally.

No push is performed.
USAGE
}

BASE_BRANCH=""
KEEP_FILES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-branch)
      BASE_BRANCH="${2:-}"
      shift 2
      ;;
    --keep)
      KEEP_FILES+=("${2:-}")
      shift 2
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

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: run from inside a git repository." >&2
  exit 1
fi

if [[ -z "$BASE_BRANCH" ]]; then
  BASE_BRANCH="$(git branch --show-current)"
fi

if [[ -z "$BASE_BRANCH" ]]; then
  echo "Error: could not detect current branch; pass --base-branch." >&2
  exit 1
fi

if [[ ${#KEEP_FILES[@]} -eq 0 ]]; then
  KEEP_FILES=(
    "neurons/miner.py"
    "poker44/miner_heuristics.py"
    "start_miner.sh"
    "start_miner2.sh"
  )
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Error: working tree is not clean." >&2
  echo "Commit/stash local changes first, then rerun." >&2
  exit 1
fi

echo "[1/6] Fetching origin"
git fetch origin --prune

echo "[2/6] Switching to base branch: $BASE_BRANCH"
git switch "$BASE_BRANCH" >/dev/null

echo "[3/6] Creating integration branch"
TS="$(date +%Y%m%d_%H%M%S)"
MERGE_BRANCH="${BASE_BRANCH}-merge-main-${TS}"
git switch -c "$MERGE_BRANCH" >/dev/null

echo "[4/6] Merging origin/main (no commit yet)"
git merge --no-ff --no-commit origin/main || true

echo "[5/6] Preserving local miner customizations"
for file in "${KEEP_FILES[@]}"; do
  if [[ -e "$file" || -n "$(git ls-files -- "$file")" ]]; then
    git checkout --ours -- "$file" 2>/dev/null || true
    git add -- "$file" 2>/dev/null || true
  fi
done

if git diff --name-only --diff-filter=U | grep -q .; then
  echo "Unresolved conflicts remain:" >&2
  git diff --name-only --diff-filter=U >&2
  echo "Resolve manually on branch: $MERGE_BRANCH" >&2
  exit 2
fi

echo "[6/6] Creating local merge commit"
git commit -m "merge(origin/main): keep miner customizations"

echo
echo "Done. Local integration branch created: $MERGE_BRANCH"
echo "No push performed."
echo "Review with: git show --stat --oneline -n 1"
