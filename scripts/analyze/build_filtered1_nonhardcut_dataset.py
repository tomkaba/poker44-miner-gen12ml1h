#!/usr/bin/env python3
"""Build filtered=1 training pool excluding hard-cutoff branch.

Selection:
- include only chunks with filtered_multi_leave == 1
- exclude chunks that trigger hard cutoff in _score_filtered_one:
  multi_joinleave > 1 or raw_multi_leave > 1

Sources:
- analysis/data/mixed_chunks*.json
- analysis/data/validator_like_window_001..270.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from poker44.miner_heuristics import _multi_leave_stats


def load_chunks(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "labeled_chunks" in payload:
        return payload["labeled_chunks"]
    return []


def collect_sources(data_dir: Path, max_window: int):
    mixed_files = sorted(data_dir.glob("mixed_chunks*.json"))
    window_files = []
    for p in sorted(data_dir.glob("validator_like_window_*.json")):
        m = re.search(r"validator_like_window_(\d+)\.json$", p.name)
        if m and 1 <= int(m.group(1)) <= max_window:
            window_files.append(p)
    return mixed_files, window_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Build filtered=1 non-hardcut dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("analysis/data"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis/data/filtered1_nonhardcut_train_mixed_validator_001_270.json"),
    )
    parser.add_argument("--max-window", type=int, default=270)
    args = parser.parse_args()

    data_dir = args.data_dir
    out_path = args.out

    mixed_files, window_files = collect_sources(data_dir, args.max_window)
    source_files = mixed_files + window_files

    rows = []
    label_counter = Counter()
    per_source = Counter()

    for idx, src in enumerate(source_files, start=1):
        chunks = load_chunks(src)
        for ch in chunks:
            if not isinstance(ch, dict):
                continue
            if "hands" not in ch or "is_bot" not in ch:
                continue

            hands = ch.get("hands") or []
            filtered_multi_leave, total_transitions, multi_joinleave, raw_multi_leave = _multi_leave_stats(hands)

            if filtered_multi_leave != 1:
                continue
            if multi_joinleave > 1 or raw_multi_leave > 1:
                continue

            row = {
                "source_file": src.name,
                "source_group": "mixed_chunks" if src.name.startswith("mixed_chunks") else "validator_like_window",
                "window_id": int(re.search(r"_(\d+)\.json$", src.name).group(1)) if src.name.startswith("validator_like_window_") else None,
                "is_bot": bool(ch.get("is_bot")),
                "hands": hands,
                "stats": {
                    "filtered_multi_leave": int(filtered_multi_leave),
                    "multi_joinleave": int(multi_joinleave),
                    "raw_multi_leave": int(raw_multi_leave),
                    "total_transitions": int(total_transitions),
                },
            }
            rows.append(row)
            label_counter[row["is_bot"]] += 1
            per_source[src.name] += 1

        if idx % 25 == 0:
            print(f"[progress] processed {idx}/{len(source_files)} files; rows={len(rows)}")

    hash_counter = Counter(
        hashlib.sha256(json.dumps(r["hands"], sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        for r in rows
    )
    unique_count = len(hash_counter)
    dup_count = len(rows) - unique_count

    payload = {
        "metadata": {
            "description": "Filtered=1 training pool excluding hard-cutoff cases (multi_joinleave>1 or raw_multi_leave>1).",
            "source_sets": {
                "mixed_chunks_files": [p.name for p in mixed_files],
                "validator_like_window_range_included": [1, args.max_window],
                "validator_files_included_count": len(window_files),
            },
            "selection_rule": {
                "filtered_multi_leave_eq": 1,
                "exclude_if": "multi_joinleave > 1 or raw_multi_leave > 1",
            },
            "counts": {
                "total_rows": len(rows),
                "bot_rows": label_counter[True],
                "human_rows": label_counter[False],
                "unique_hands_hashes": unique_count,
                "duplicate_rows_by_hands_hash": dup_count,
            },
            "top_sources": per_source.most_common(20),
        },
        "labeled_chunks": rows,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    print(f"[done] wrote {out_path}")
    print(f"[done] total={len(rows)} bot={label_counter[True]} human={label_counter[False]}")
    print(f"[done] unique={unique_count} duplicates={dup_count}")


if __name__ == "__main__":
    main()
