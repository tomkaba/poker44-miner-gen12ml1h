# Local Feature Flags / Debug Hooks

This file tracks custom environment flags we have added on top of upstream `main`, so the patches are easy to cherry-pick when rebasing.

## `SAVECHUNKS_LOG_EXPECTED`
- **Files**: `poker44/validator/forward.py`
- **Purpose**: When set to `1/true`, the validator writes every batch of chunks + expected labels to `~/Poker44-subnet/logs/validator_chunks/YYYY-MM-DD/forward_*.json.gz`, with a matching `index.jsonl` for quick browsing.
- **Key logic**: `_maybe_save_validator_chunks(...)` is invoked inside the forward loop right after chunks/labels are prepared.

## `EXPORT_HH`
- **Files**: `neurons/validator.py`
- **Purpose**: When `EXPORT_HH=1`, the validator dumps the entire pool of labeled chunks (the same list used by `GeneratedDatasetProvider`) to `~/Poker44-subnet/logs/hand_pool/hand_pool_<timestamp>.json.gz`. Override destination via `EXPORT_HH_DIR=<path>`.
- **Key logic**: `_maybe_export_hand_pool(labeled_chunks)` is called immediately after `generate_dataset_array(include_labels=True)` in the validator constructor; it records metadata (`chunk_count`, `total_hands`) alongside the full chunk payload.

> When bringing these changes to a fresh branch derived from `origin/main`, re-apply the snippets above or cherry-pick the commits touching the listed files.
