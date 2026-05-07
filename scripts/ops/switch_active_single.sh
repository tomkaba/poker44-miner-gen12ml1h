#!/usr/bin/env bash
set -euo pipefail

# Switch active single-hand runtime artifacts by updating symlinks:
# - weights/ml_single_hand_model.pkl
# - weights/ml_single_hand_scaler.pkl
#
# Usage:
#   scripts/ops/switch_active_single.sh <model_code> [--check]
#   scripts/ops/switch_active_single.sh <model_file> <scaler_file> [--check]
#
# Examples:
#   scripts/ops/switch_active_single.sh gen5_17 --check
#   scripts/ops/switch_active_single.sh 5plus
#   scripts/ops/switch_active_single.sh ml_gen4_model.pkl ml_gen4_scaler.pkl
#   scripts/ops/switch_active_single.sh ml_gen5_17_s123467_model.pkl ml_gen5_17_s123467_scaler.pkl --check

usage() {
  cat <<'USAGE'
Usage:
  scripts/ops/switch_active_single.sh <model_code> [--check]
  scripts/ops/switch_active_single.sh <model_file> <scaler_file> [--check]

Arguments:
  model_code   Shortcut alias:
               gen4 | gen4_17 | gen5 | gen5_17 | gen5plus | 5plus
  model_file   File name in weights/ (e.g. ml_gen4_model.pkl)
  scaler_file  File name in weights/ (e.g. ml_gen4_scaler.pkl)

Options:
  --check      Run a quick python runtime load check after switching.
  -h, --help   Show this help.
USAGE
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

DO_CHECK=0

if [[ ${!#:-} == "--check" ]]; then
  DO_CHECK=1
  set -- "${@:1:$(($#-1))}"
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

resolve_alias() {
  local alias="$1"
  local key="${alias,,}"
  case "$key" in
    gen4)
      echo "ml_gen4_model.pkl ml_gen4_scaler.pkl"
      ;;
    gen4_17)
      echo "ml_gen4_17_model.pkl ml_gen4_17_scaler.pkl"
      ;;
    gen5)
      echo "ml_gen5_s123467_model.pkl ml_gen5_s123467_scaler.pkl"
      ;;
    gen5_17)
      echo "ml_gen5_17_s123467_model.pkl ml_gen5_17_s123467_scaler.pkl"
      ;;
    gen5plus|5plus)
      echo "ml_single_hand_v5plus_s123467_model.pkl ml_single_hand_v5plus_s123467_scaler.pkl"
      ;;
    *)
      return 1
      ;;
  esac
}

MODEL_FILE=""
SCALER_FILE=""
MODE_DESC=""

if [[ $# -eq 1 ]]; then
  if resolved="$(resolve_alias "$1")"; then
    MODEL_FILE="${resolved%% *}"
    SCALER_FILE="${resolved##* }"
    MODE_DESC="alias=$1"
  else
    echo "Unknown model_code: $1" >&2
    usage
    exit 1
  fi
else
  MODEL_FILE="$1"
  SCALER_FILE="$2"
  MODE_DESC="explicit_files"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WEIGHTS_DIR="$REPO_ROOT/weights"

MODEL_PATH="$WEIGHTS_DIR/$MODEL_FILE"
SCALER_PATH="$WEIGHTS_DIR/$SCALER_FILE"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Error: missing model file: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$SCALER_PATH" ]]; then
  echo "Error: missing scaler file: $SCALER_PATH" >&2
  exit 1
fi

cd "$REPO_ROOT"

ln -sfn "$MODEL_FILE" "weights/ml_single_hand_model.pkl"
ln -sfn "$SCALER_FILE" "weights/ml_single_hand_scaler.pkl"

echo "Switched active single-hand artifacts:"
echo "mode=$MODE_DESC"
ls -l "weights/ml_single_hand_model.pkl" "weights/ml_single_hand_scaler.pkl"

if [[ $DO_CHECK -eq 1 ]]; then
  if [[ ! -f ".venv/bin/activate" ]]; then
    echo "Warning: .venv not found, skipping --check" >&2
    exit 0
  fi

  # shellcheck disable=SC1091
  . ".venv/bin/activate"
  python - <<'PY'
import poker44.miner_heuristics as mh

mh.reset_ml_request_stats()
mh._load_ml_model_single_hand()
st = mh.get_ml_runtime_stats()
m = getattr(mh, "_ML_SINGLE_HAND_MODEL", None)

print("[check] loaded=", st.get("ml_single_hand_model_loaded"))
print("[check] available=", st.get("ml_single_hand_model_available"))
print("[check] last_error=", st.get("ml_single_hand_last_error"))
print("[check] model_path=", mh.ML_SINGLE_HAND_MODEL_PATH)
print("[check] scaler_path=", mh.ML_SINGLE_HAND_SCALER_PATH)
print("[check] model_name=", getattr(m, "_model_name", None))
print("[check] extractor_tag=", getattr(m, "_feature_extractor_tag", None))
print("[check] n_features=", getattr(m, "n_features_in_", None))
PY
fi
