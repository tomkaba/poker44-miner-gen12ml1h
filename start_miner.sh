#!/bin/bash

set -euo pipefail

if [[ $# -lt 1 || -z "${1:-}" ]]; then
  echo "Użycie: $0 HOTKEY_ID[,HOTKEY_ID2,...]"
  echo "Przykład: $0 214"
  echo "Przykład: $0 11,14,22"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$SCRIPT_DIR"
IDS_STRING="$1"
ENV_FILE="${ENV_FILE:-$REPO/.env}"

WALLET_NAME="${POKER44_WALLET_NAME:-sn126b}"
SESSION_PREFIX="${POKER44_SESSION_PREFIX:-sn126b_m}"
AXON_BASE_PORT="${POKER44_AXON_BASE_PORT:-12080}"
VENV_BIN="${POKER44_VENV_BIN:-$REPO/.venv/bin}"

MANIFEST_REPO_URL="${POKER44_MODEL_REPO_URL:-https://github.com/tomkaba/poker44-miner-gen12ml1h}"
MANIFEST_REPO_COMMIT="${POKER44_MODEL_REPO_COMMIT:-$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)}"
MODEL_ARTIFACT_REL="weights/ml_gen5_s123467_model.pkl"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  echo "[env] Loaded $ENV_FILE"
else
  echo "[env] File not found, skipping: $ENV_FILE"
fi

if [[ ! -x "$VENV_BIN/python" ]]; then
  echo "ERROR: Python runtime not found at $VENV_BIN/python"
  exit 1
fi

MODEL_ARTIFACT_PATH="$REPO/$MODEL_ARTIFACT_REL"
if [[ ! -f "$MODEL_ARTIFACT_PATH" ]]; then
  echo "ERROR: Missing model artifact: $MODEL_ARTIFACT_REL"
  exit 1
fi

# If the model file is still a Git LFS pointer, try to fetch real content.
if head -n 1 "$MODEL_ARTIFACT_PATH" 2>/dev/null | grep -q "^version https://git-lfs.github.com/spec/v1$"; then
  echo "[lfs] Detected Git LFS pointer in $MODEL_ARTIFACT_REL"
  if git -C "$REPO" lfs version >/dev/null 2>&1; then
    echo "[lfs] Fetching model artifact via git lfs pull..."
    git -C "$REPO" lfs pull --include "$MODEL_ARTIFACT_REL"
  else
    echo "ERROR: git-lfs is not installed on this host."
    echo "Install git-lfs, then run: git -C $REPO lfs pull --include $MODEL_ARTIFACT_REL"
    exit 1
  fi

  if head -n 1 "$MODEL_ARTIFACT_PATH" 2>/dev/null | grep -q "^version https://git-lfs.github.com/spec/v1$"; then
    echo "ERROR: Model artifact is still an LFS pointer after pull: $MODEL_ARTIFACT_REL"
    echo "Check git-lfs installation and repository authentication, then retry."
    exit 1
  fi
fi

echo "[manifest] POKER44_MODEL_REPO_URL=$MANIFEST_REPO_URL"
echo "[manifest] POKER44_MODEL_REPO_COMMIT=$MANIFEST_REPO_COMMIT"

for raw_id in $(echo "$IDS_STRING" | tr ',' '\n'); do
  I="$(echo "$raw_id" | tr -d ' ')"

  if [[ -z "$I" ]]; then
    continue
  fi
  if ! [[ "$I" =~ ^[0-9]+$ ]]; then
    echo "WARN: Invalid HOTKEY_ID '$I', skipping"
    continue
  fi

  PORT=$((AXON_BASE_PORT + I))
  SESSION="${SESSION_PREFIX}${I}"

  echo "[start] HOTKEY_ID=$I SESSION=$SESSION PORT=$PORT"

  OLD_PID=$(screen -list 2>/dev/null | grep "\.$SESSION[[:space:]]" | awk '{print $1}' | cut -d. -f1 || true)
  if [[ -n "$OLD_PID" ]]; then
    echo "[cleanup] Killed old session PID=$OLD_PID"
    screen -S "$OLD_PID" -X quit 2>/dev/null || true
  fi

  screen -dmS "$SESSION" /bin/bash -c "
    cd $REPO
    source $VENV_BIN/activate
    export PYTHONPATH=$REPO:\${PYTHONPATH:-}
    export POKER44_CHUNK_SCORER=ml1h
    export POKER44_MODEL_REPO_URL=$MANIFEST_REPO_URL
    export POKER44_MODEL_REPO_COMMIT=$MANIFEST_REPO_COMMIT
    echo '[runtime] HOTKEY_ID=$I'
    echo '[runtime] CHUNK_SCORER=ml1h'
    $VENV_BIN/python -m neurons.miner \
      --netuid 126 \
      --wallet.name $WALLET_NAME \
      --wallet.hotkey hk$I \
      --subtensor.network finney \
      --axon.port $PORT \
      --logging.debug
    echo '[miner-exit] Process ended, shell remains active'
    /bin/bash
  "

  if [[ $? -eq 0 ]]; then
    echo "[ok] Session $SESSION started"
  else
    echo "[fail] Failed to start session $SESSION"
  fi
done

echo "[done] All requested HOTKEY_ID(s) processed"
