#!/usr/bin/env bash
# Poker44 Benchmark Cron Job
# ===========================
# Runs daily to download new benchmark releases.
#
# Add to crontab with:
#   crontab -e
# Then add (runs daily at 02:30 AM):
#   30 2 * * * /home/tk/Poker44-subnet-main/scripts/fetch_benchmark_cron.sh >> /home/tk/Poker44-subnet-main/logs/benchmark_cron.log 2>&1
#
# Or for a specific user's crontab, update the paths below to match your setup.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_DIR/.venv-2/bin/python"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Poker44 Benchmark Fetch — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=========================================="

# Use venv python if available, otherwise system python3
if [[ -x "$VENV_PYTHON" ]]; then
    PYTHON="$VENV_PYTHON"
else
    PYTHON="python3"
fi

echo "Python: $PYTHON"
echo "Project: $PROJECT_DIR"
echo ""

"$PYTHON" "$SCRIPT_DIR/fetch_benchmark.py"

echo ""
echo "Finished at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
