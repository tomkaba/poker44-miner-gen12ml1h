#!/usr/bin/env bash
set -euo pipefail

# Compare local vs production miner-critical files, model artifacts, and Python environment.
# Usage:
#   scripts/check_prod_sync.sh --host user@prod --repo /path/to/Poker44-subnet-main
# Optional:
#   --local /home/tk/Poker44-subnet-main
#   --out /tmp/custom_output_dir

usage() {
  cat <<'USAGE'
Usage:
  scripts/check_prod_sync.sh --host <user@prod> --repo <prod_repo_path> [--local <local_repo_path>] [--out <output_dir>]

Examples:
  scripts/check_prod_sync.sh --host user@prod --repo /opt/Poker44-subnet-main
  scripts/check_prod_sync.sh --host user@prod --repo /opt/Poker44-subnet-main --local /home/tk/Poker44-subnet-main
USAGE
}

PROD_HOST=""
PROD_REPO=""
LOCAL_REPO=""
OUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      PROD_HOST="${2:-}"
      shift 2
      ;;
    --repo)
      PROD_REPO="${2:-}"
      shift 2
      ;;
    --local)
      LOCAL_REPO="${2:-}"
      shift 2
      ;;
    --out)
      OUT_DIR="${2:-}"
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

if [[ -z "$PROD_HOST" || -z "$PROD_REPO" ]]; then
  echo "Error: --host and --repo are required." >&2
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_LOCAL_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_REPO="${LOCAL_REPO:-$DEFAULT_LOCAL_REPO}"
OUT_DIR="${OUT_DIR:-/tmp/poker44_sync_check_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUT_DIR"

echo "[1/6] Local hashes"
cd "$LOCAL_REPO"

sha256sum \
  neurons/miner.py \
  poker44/miner_heuristics.py \
  poker44/base/miner.py \
  poker44/validator/synapse.py \
  poker44/core/models.py \
  > "$OUT_DIR/local_code.sha"

sha256sum \
  weights/ml_filtered0_model.pkl \
  weights/ml_filtered0_scaler.pkl \
  weights/ml_filtered1_nonhardcut_model.pkl \
  weights/ml_filtered1_nonhardcut_scaler.pkl \
  > "$OUT_DIR/local_weights.sha"

echo "[2/6] Local environment"
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
  python -V > "$OUT_DIR/local_python.txt" 2>&1
  pip freeze | sort > "$OUT_DIR/local_freeze.txt"
else
  echo "NO_VENV" > "$OUT_DIR/local_python.txt"
  echo "NO_VENV" > "$OUT_DIR/local_freeze.txt"
fi

echo "[3/6] Remote hashes + environment"
ssh "$PROD_HOST" "
set -euo pipefail
cd \"$PROD_REPO\"

sha256sum \
  neurons/miner.py \
  poker44/miner_heuristics.py \
  poker44/base/miner.py \
  poker44/validator/synapse.py \
  poker44/core/models.py

sha256sum \
  weights/ml_filtered0_model.pkl \
  weights/ml_filtered0_scaler.pkl \
  weights/ml_filtered1_nonhardcut_model.pkl \
  weights/ml_filtered1_nonhardcut_scaler.pkl

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
  python -V
  pip freeze | sort
else
  echo NO_VENV
  echo NO_VENV
fi
" > "$OUT_DIR/remote_all.txt"

echo "[4/6] Split remote output"
sed -n '1,5p' "$OUT_DIR/remote_all.txt" > "$OUT_DIR/remote_code.sha"
sed -n '6,9p' "$OUT_DIR/remote_all.txt" > "$OUT_DIR/remote_weights.sha"
sed -n '10p' "$OUT_DIR/remote_all.txt" > "$OUT_DIR/remote_python.txt"
sed -n '11,$p' "$OUT_DIR/remote_all.txt" > "$OUT_DIR/remote_freeze.txt"

echo "[5/6] Diff"
diff -u "$OUT_DIR/remote_code.sha" "$OUT_DIR/local_code.sha" > "$OUT_DIR/diff_code.txt" || true
diff -u "$OUT_DIR/remote_weights.sha" "$OUT_DIR/local_weights.sha" > "$OUT_DIR/diff_weights.txt" || true
diff -u "$OUT_DIR/remote_python.txt" "$OUT_DIR/local_python.txt" > "$OUT_DIR/diff_python.txt" || true
diff -u "$OUT_DIR/remote_freeze.txt" "$OUT_DIR/local_freeze.txt" > "$OUT_DIR/diff_freeze.txt" || true

echo "[6/6] Summary"
for f in code weights python freeze; do
  d="$OUT_DIR/diff_${f}.txt"
  if [[ -s "$d" ]]; then
    echo "DIFF: $f"
  else
    echo "OK:   $f"
  fi
done

echo
echo "Output directory: $OUT_DIR"
echo "Preview commands:"
echo "  sed -n '1,120p' $OUT_DIR/diff_code.txt"
echo "  sed -n '1,120p' $OUT_DIR/diff_weights.txt"
echo "  sed -n '1,120p' $OUT_DIR/diff_python.txt"
echo "  sed -n '1,120p' $OUT_DIR/diff_freeze.txt"
