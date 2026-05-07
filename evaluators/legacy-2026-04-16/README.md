# Legacy Evaluators (2026-04-16)

This directory archives the pre-vanilla-switch evaluator scripts exactly as they were before the active versions were updated to use validator-side modules from `~/Poker44-vanilla`.

These archived copies resolve imports from the current local repository state.

## Archived scripts

- `evaluate_validator_like.py`
- `compare_custom_1hand_suite.py`

## Historical run commands

### `evaluate_validator_like.py`

```bash
python -W ignore evaluators/legacy-2026-04-16/evaluate_validator_like.py \
  --data-dir data \
  --pattern "public_benchmark_1hand_1000_set5_seed_*.json.gz" \
  --threshold 0.5 \
  --print-every 10 \
  --progress-chunks 5000 \
  --errors-out analysis/gen5/set5_eval_gen5_t50_errors.json \
  | tee analysis/gen5/set5_eval_gen5_t50.log
```

### `compare_custom_1hand_suite.py`

```bash
python -W ignore evaluators/legacy-2026-04-16/compare_custom_1hand_suite.py \
  --suite-dir data/public_benchmark_custom_1hand_5000_20260404_092644 \
  --threshold 0.3 \
  --output-csv analysis/gen5/bench_gen5_v1_t30_rerun_fixed.csv \
  --progress-chunks 1000 \
  | tee analysis/gen5/bench_gen5_v1_t30_rerun_fixed.log
```