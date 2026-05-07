#!/usr/bin/env python3
"""Evaluate generated 1-hand custom benchmark suite and print a comparison table.

Expected input files in suite dir:
- public_1hand_<N>_profile_<name>.json.gz
- public_1hand_<N>_preset_<name>.json.gz

This script uses the active miner scoring path (score_chunk_modern), so results
reflect current runtime behavior with currently active model weights.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from poker44 import miner_heuristics as mh


def log(message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare accuracy across 1-hand custom benchmark suite")
    parser.add_argument(
        "--suite-dir",
        type=Path,
        required=True,
        help="Directory with generated public_1hand_*_profile_*.json.gz and *_preset_*.json.gz files",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold for bot classification (default: 0.5)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path",
    )
    parser.add_argument(
        "--progress-chunks",
        type=int,
        default=1000,
        help="Print in-file progress every N chunks (0 disables, default: 1000)",
    )
    parser.add_argument(
        "--score-source",
        choices=("runtime", "raw-single-hand"),
        default="runtime",
        help=(
            "Source for classification score. 'runtime' uses score_chunk_modern and reflects live miner behavior; "
            "'raw-single-hand' uses the active single-hand model probability before runtime threshold calibration."
        ),
    )
    return parser.parse_args()


def extract_rows(path: Path) -> List[Dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        rows = payload.get("labeled_chunks", [])
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported dataset structure: {path}")
    return rows


def dataset_tag(path: Path) -> Tuple[str, str]:
    name = path.name
    marker = "_profile_"
    kind = "profile"
    if marker not in name:
        marker = "_preset_"
        kind = "preset"
    if marker not in name:
        return "other", path.stem
    suffix = name.split(marker, 1)[1]
    if suffix.endswith(".json.gz"):
        suffix = suffix[: -len(".json.gz")]
    elif suffix.endswith(".json"):
        suffix = suffix[: -len(".json")]
    return kind, suffix


def evaluate_rows(
    rows: List[Dict[str, Any]],
    threshold: float,
    score_source: str = "runtime",
    progress_label: str = "",
    progress_chunks: int = 0,
) -> Dict[str, Any]:
    tp = fp = fn = tn = 0
    skipped = 0

    total_rows = len(rows)
    for idx, row in enumerate(rows, 1):
        hands = row.get("hands") or []
        if not hands:
            skipped += 1
            if progress_chunks and idx % progress_chunks == 0:
                log(f"{progress_label} chunk {idx}/{total_rows} (skipped={skipped})")
            continue

        if score_source == "raw-single-hand":
            mh._load_ml_model_single_hand()
            raw_score = mh._score_single_hand_ml(hands)
            if raw_score is None:
                raise RuntimeError("raw-single-hand scoring requested but active single-hand model is unavailable")
            score = float(raw_score)
        else:
            score = float(mh.score_chunk_modern(hands))
        pred_bot = score >= threshold
        true_bot = bool(row.get("is_bot", False))

        if true_bot and pred_bot:
            tp += 1
        elif true_bot and not pred_bot:
            fn += 1
        elif not true_bot and pred_bot:
            fp += 1
        else:
            tn += 1

        if progress_chunks and idx % progress_chunks == 0:
            done = tp + fp + fn + tn
            acc = (tp + tn) / done if done else 0.0
            log(f"{progress_label} chunk {idx}/{total_rows} acc={acc:.4f}")

    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "n": total,
        "skipped": skipped,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def format_table(results: List[Dict[str, Any]]) -> str:
    headers = [
        "dataset",
        "type",
        "acc",
        "f1",
        "prec",
        "rec",
        "tp",
        "fp",
        "fn",
        "tn",
        "n",
    ]
    rows = []
    for r in results:
        rows.append(
            [
                r["name"],
                r["kind"],
                f"{r['accuracy']:.4f}",
                f"{r['f1']:.4f}",
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                str(r["tp"]),
                str(r["fp"]),
                str(r["fn"]),
                str(r["tn"]),
                str(r["n"]),
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, col in enumerate(row):
            widths[i] = max(widths[i], len(col))

    def _fmt_line(cols: List[str]) -> str:
        return " | ".join(col.ljust(widths[i]) for i, col in enumerate(cols))

    out = [_fmt_line(headers)]
    out.append("-+-".join("-" * w for w in widths))
    for row in rows:
        out.append(_fmt_line(row))
    return "\n".join(out)


def maybe_write_csv(path: Path, results: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["dataset,type,accuracy,f1,precision,recall,tp,fp,fn,tn,n,skipped"]
    for r in results:
        lines.append(
            ",".join(
                [
                    r["name"],
                    r["kind"],
                    f"{r['accuracy']:.6f}",
                    f"{r['f1']:.6f}",
                    f"{r['precision']:.6f}",
                    f"{r['recall']:.6f}",
                    str(r["tp"]),
                    str(r["fp"]),
                    str(r["fn"]),
                    str(r["tn"]),
                    str(r["n"]),
                    str(r["skipped"]),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    suite_dir = args.suite_dir.expanduser().resolve()
    if not suite_dir.exists():
        raise FileNotFoundError(f"Suite directory not found: {suite_dir}")

    profile_files = sorted(suite_dir.glob("public_1hand_*_profile_*.json.gz"))
    preset_files = sorted(suite_dir.glob("public_1hand_*_preset_*.json.gz"))
    files = profile_files + preset_files

    if not files:
        raise FileNotFoundError(
            f"No matching suite files in {suite_dir}. Expected public_1hand_*_profile_*.json.gz and *_preset_*.json.gz"
        )

    log(f"suite_dir={suite_dir}")
    log(f"files={len(files)} threshold={args.threshold:.3f} score_source={args.score_source}")
    if args.score_source == "raw-single-hand":
        log("loading active model via raw single-hand probability path...")
    else:
        log("loading active model via miner runtime scoring path...")

    results: List[Dict[str, Any]] = []
    for idx, path in enumerate(files, 1):
        kind, name = dataset_tag(path)
        progress_label = f"[{idx}/{len(files)}] {path.name}"
        log(f"{progress_label} start")
        rows = extract_rows(path)
        metrics = evaluate_rows(
            rows,
            args.threshold,
            score_source=args.score_source,
            progress_label=progress_label,
            progress_chunks=max(0, args.progress_chunks),
        )
        results.append({"file": str(path), "name": name, "kind": kind, **metrics})
        log(
            (
                f"{progress_label} done "
                f"acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
                f"fp={metrics['fp']} fn={metrics['fn']}"
            )
        )

    results.sort(key=lambda r: (r["accuracy"], r["f1"]), reverse=True)

    log("=== COMPARISON TABLE (sorted by accuracy desc) ===")
    print(format_table(results), flush=True)

    if args.output_csv:
        csv_path = args.output_csv.expanduser().resolve()
        maybe_write_csv(csv_path, results)
        log(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
