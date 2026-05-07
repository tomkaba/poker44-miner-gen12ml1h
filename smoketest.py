#!/usr/bin/env python3
"""Quick smoke test for Poker44 miner scoring paths.

Checks both:
- single-hand path (len(chunk) == 1)
- multi-hand path (len(chunk) > 1)

It reports runtime counters from poker44.miner_heuristics to confirm whether
ML paths are used or if fallback paths are triggered.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import poker44.miner_heuristics as mh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test single-hand and multi-hand scoring paths")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory with benchmark files (default: data)",
    )
    parser.add_argument(
        "--single-pattern",
        type=str,
        default="public_benchmark_1hand_1000_set5_seed_*.json.gz",
        help="Glob pattern for single-hand source files",
    )
    parser.add_argument(
        "--multi-pattern",
        type=str,
        default="validator_like_window_*.json",
        help="Glob pattern for multi-hand source files",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("weights/ml_single_hand_model.pkl"),
        help="Path to single-hand model artifact",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=Path("weights/ml_single_hand_scaler.pkl"),
        help="Path to single-hand scaler artifact",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Threshold for quick single-hand confusion summary",
    )
    parser.add_argument(
        "--single-rows",
        type=int,
        default=100,
        help="Number of rows for single-hand check",
    )
    parser.add_argument(
        "--multi-rows",
        type=int,
        default=100,
        help="Number of rows for multi-hand check",
    )
    parser.add_argument(
        "--synthetic-multi",
        action="store_true",
        help="If a row has a single hand, duplicate it to force multi-hand path",
    )
    parser.add_argument(
        "--single-file-index",
        type=int,
        default=0,
        help="Index of matched single-hand file to use (default: 0)",
    )
    parser.add_argument(
        "--multi-file-index",
        type=int,
        default=0,
        help="Index of matched multi-hand file to use (default: 0)",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    rows = payload.get("labeled_chunks", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported payload format in {path}")
    return rows


def ensure_single_hand_model(model_path: Path, scaler_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    with model_path.open("rb") as f:
        mh._ML_SINGLE_HAND_MODEL = pickle.load(f)
    with scaler_path.open("rb") as f:
        mh._ML_SINGLE_HAND_SCALER = pickle.load(f)

    mh._ML_SINGLE_HAND_MODEL_AVAILABLE = True
    mh._ML_SINGLE_HAND_LAST_ERROR = None


def run_single_hand(rows: List[Dict[str, Any]], threshold: float, limit: int) -> Dict[str, int]:
    tp = tn = fp = fn = 0
    checked = skipped = 0

    for row in rows[: max(0, limit)]:
        hands = row.get("hands") or []
        if not hands:
            skipped += 1
            continue

        y = 1 if row.get("is_bot") else 0
        score = mh.score_chunk_modern(hands[:1])
        pred = 1 if score >= threshold else 0

        if y == 1 and pred == 1:
            tp += 1
        elif y == 0 and pred == 0:
            tn += 1
        elif y == 0 and pred == 1:
            fp += 1
        else:
            fn += 1
        checked += 1

    return {
        "checked": checked,
        "skipped": skipped,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_multi_hand(rows: List[Dict[str, Any]], limit: int, synthetic_multi: bool) -> Dict[str, int]:
    checked = skipped = none_scores = skipped_single_only = 0

    for row in rows[: max(0, limit)]:
        hands = row.get("hands") or []
        if not hands:
            skipped += 1
            continue

        if len(hands) > 1:
            chunk = hands
        elif synthetic_multi:
            # Optional fallback: force multi-hand path for single-hand rows.
            chunk = hands + hands
        else:
            skipped_single_only += 1
            continue

        score = mh.score_chunk_modern(chunk)
        if score is None:
            none_scores += 1
        checked += 1

    return {
        "checked": checked,
        "skipped": skipped,
        "skipped_single_only": skipped_single_only,
        "none_scores": none_scores,
    }


def main() -> None:
    args = parse_args()

    single_files = sorted(args.data_dir.glob(args.single_pattern))
    if not single_files:
        raise SystemExit(f"No files match single-pattern '{args.single_pattern}' in {args.data_dir}")
    if args.single_file_index < 0 or args.single_file_index >= len(single_files):
        raise SystemExit(
            f"single-file-index out of range: {args.single_file_index} (matches={len(single_files)})"
        )

    multi_files = sorted(args.data_dir.glob(args.multi_pattern))
    if not multi_files:
        raise SystemExit(f"No files match multi-pattern '{args.multi_pattern}' in {args.data_dir}")
    if args.multi_file_index < 0 or args.multi_file_index >= len(multi_files):
        raise SystemExit(
            f"multi-file-index out of range: {args.multi_file_index} (matches={len(multi_files)})"
        )

    selected_single = single_files[args.single_file_index]
    selected_multi = multi_files[args.multi_file_index]
    single_rows = load_rows(selected_single)
    multi_rows = load_rows(selected_multi)

    ensure_single_hand_model(args.model_path, args.scaler_path)
    mh.reset_ml_request_stats()

    single = run_single_hand(single_rows, args.threshold, args.single_rows)
    multi = run_multi_hand(multi_rows, args.multi_rows, synthetic_multi=args.synthetic_multi)

    stats = mh.get_ml_runtime_stats()

    single_total = single["tp"] + single["tn"] + single["fp"] + single["fn"]
    single_acc = ((single["tp"] + single["tn"]) / single_total) if single_total else 0.0

    print("=== Smoke Test Summary ===")
    print(f"single_file={selected_single}")
    print(f"single_rows_total={len(single_rows)}")
    print(f"multi_file={selected_multi}")
    print(f"multi_rows_total={len(multi_rows)}")
    print()

    print("[single-hand]")
    print(
        {
            "checked": single["checked"],
            "skipped": single["skipped"],
            "acc": round(single_acc, 6),
            "tp": single["tp"],
            "tn": single["tn"],
            "fp": single["fp"],
            "fn": single["fn"],
        }
    )

    print("[multi-hand]")
    print(multi)

    print("[runtime counters]")
    print(
        {
            "single_ml_used": stats.get("request_single_hand_ml_used"),
            "single_f0_fallback": stats.get("request_single_hand_f0_fallback"),
            "f0_ml_used": stats.get("request_f0_ml_used"),
            "f0_heur_fallback": stats.get("request_f0_heur_fallback"),
            "f1_ml_used": stats.get("request_f1_ml_used"),
            "f1_heur_fallback": stats.get("request_f1_heur_fallback"),
            "f1_hardcut_forced_human": stats.get("request_f1_hardcut_forced_human"),
            "f2plus_forced_human": stats.get("request_f2plus_forced_human"),
            "f0_loaded": stats.get("ml_model_loaded"),
            "f1_loaded": stats.get("ml_f1_model_loaded"),
            "single_loaded": stats.get("ml_single_hand_model_loaded"),
            "single_last_error": stats.get("ml_single_hand_last_error"),
        }
    )


if __name__ == "__main__":
    main()
