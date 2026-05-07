#!/usr/bin/env bash
set -euo pipefail

# Report new commits on origin/dev since last check and flag miner-relevant
# changes, especially around validator chunk semantics and scoring interface.
#
# Usage:
#   scripts/ops/watch_dev_changes.sh
#   scripts/ops/watch_dev_changes.sh --since <sha>
#   scripts/ops/watch_dev_changes.sh --no-state

usage() {
  cat <<'USAGE'
Usage:
  scripts/ops/watch_dev_changes.sh [--since <sha>] [--no-state]

Options:
  --since <sha>  Compare from explicit commit instead of saved state.
  --no-state     Do not read/write state file.
  -h, --help     Show help.

State file:
  .git/poker44_dev_watch_last_sha

No push or branch changes are performed.
USAGE
}

SINCE_SHA=""
USE_STATE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --since)
      SINCE_SHA="${2:-}"
      shift 2
      ;;
    --no-state)
      USE_STATE=0
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

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: run from inside a git repository." >&2
  exit 1
fi

STATE_FILE="$(git rev-parse --git-dir)/poker44_dev_watch_last_sha"

echo "[1/4] Fetching origin"
git fetch origin --prune >/dev/null

HEAD_DEV="$(git rev-parse origin/dev)"

if [[ -z "$SINCE_SHA" && $USE_STATE -eq 1 && -f "$STATE_FILE" ]]; then
  SINCE_SHA="$(cat "$STATE_FILE")"
fi

if [[ -z "$SINCE_SHA" ]]; then
  SINCE_SHA="$(git merge-base origin/main origin/dev)"
fi

if ! git cat-file -e "$SINCE_SHA^{commit}" 2>/dev/null; then
  echo "Error: invalid --since sha: $SINCE_SHA" >&2
  exit 1
fi

if [[ "$SINCE_SHA" == "$HEAD_DEV" ]]; then
  echo "No new commits on origin/dev since $SINCE_SHA"
  exit 0
fi

echo "[2/4] New commits on origin/dev"
git log --oneline --decorate "$SINCE_SHA..origin/dev"

echo
echo "[3/4] Miner relevance scan"
COMMITS="$(git rev-list --reverse "$SINCE_SHA..origin/dev")"
for c in $COMMITS; do
  files="$(git show --name-only --pretty='' "$c")"
  impact="low"

  if echo "$files" | grep -Eq '^(neurons/miner.py|poker44/miner_heuristics.py)$'; then
    impact="high"
  elif echo "$files" | grep -Eq '^(neurons/validator.py|poker44/validator/|poker44/core/models.py|poker44/validator/synapse.py)'; then
    impact="medium"
  fi

  chunk_hint=""
  if git show --pretty='' "$c" | grep -Eqi 'chunkCount|minHandsPerChunk|maxHandsPerChunk|chunks|risk_scores|single.?hand|multi.?hand'; then
    chunk_hint=" | scoring/chunk semantics touched"
  fi

  echo "- $c | impact=$impact$chunk_hint"
done

echo
echo "[4/4] Summary"
echo "Compare branch tips:"
git rev-list --left-right --count HEAD...origin/dev | awk '{print "HEAD_only=" $1 " origin_dev_only=" $2}'

if [[ $USE_STATE -eq 1 ]]; then
  echo "$HEAD_DEV" > "$STATE_FILE"
  echo "Saved state: $STATE_FILE -> $HEAD_DEV"
else
  echo "State file update skipped (--no-state)."
fi
