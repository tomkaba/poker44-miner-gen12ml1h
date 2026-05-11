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

MANIFEST_REPO_URL="${POKER44_MODEL_REPO_URL:-https://github.com/tomkaba/poker44-miner-gen11lgbm}"
MANIFEST_REPO_COMMIT="${POKER44_MODEL_REPO_COMMIT:-$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)}"
MANIFEST_IMPL_FILES="models/benchmark_lgbm_model.pkl,models/benchmark_lgbm_profile.json,neurons/miner.py,poker44/base/miner.py,poker44/base/neuron.py,poker44/miner_heuristics.py,poker44/utils/config.py,poker44/utils/misc.py,poker44/utils/model_manifest.py,poker44/validator/synapse.py"

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

MANIFEST_IMPL_SHA256="$($VENV_BIN/python - <<'PY' "$REPO"
from pathlib import Path
import hashlib
import sys

repo_root = Path(sys.argv[1]).resolve()
files = [
    'neurons/miner.py',
    'poker44/miner_heuristics.py',
  'models/benchmark_lgbm_profile.json',
  'models/benchmark_lgbm_model.pkl',
]
digest = hashlib.sha256()
for rel in sorted(files):
    p = repo_root / rel
    if not p.exists():
        raise SystemExit(f"MISSING: {rel}")
    digest.update(rel.encode('utf-8'))
    with p.open('rb') as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
print(digest.hexdigest())
PY
)"

echo "[manifest] POKER44_MODEL_REPO_URL=$MANIFEST_REPO_URL"
echo "[manifest] POKER44_MODEL_REPO_COMMIT=$MANIFEST_REPO_COMMIT"
echo "[manifest] POKER44_MODEL_IMPLEMENTATION_FILES=$MANIFEST_IMPL_FILES"
echo "[manifest] POKER44_MODEL_IMPLEMENTATION_SHA256=$MANIFEST_IMPL_SHA256"

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
    export POKER44_SINGLE_HAND_MODEL_ALIAS=gen11lgbm
    export POKER44_CHUNK_SCORER=gen11lgbm
    export POKER44_GEN11LGBM_PROFILE=$REPO/models/benchmark_lgbm_profile.json
    export POKER44_GEN11LGBM_MODEL=$REPO/models/benchmark_lgbm_model.pkl
    export POKER44_MODEL_REPO_URL=$MANIFEST_REPO_URL
    export POKER44_MODEL_REPO_COMMIT=$MANIFEST_REPO_COMMIT
    export POKER44_MODEL_IMPLEMENTATION_FILES=$MANIFEST_IMPL_FILES
    export POKER44_MODEL_IMPLEMENTATION_SHA256=$MANIFEST_IMPL_SHA256
    echo '[runtime] HOTKEY_ID=$I'
    echo '[runtime] CHUNK_SCORER=gen11lgbm'
    echo '[runtime] MANIFEST_SHA256=$MANIFEST_IMPL_SHA256'
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
