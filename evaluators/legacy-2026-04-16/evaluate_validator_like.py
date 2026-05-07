#!/usr/bin/env python3
"""Evaluate current miner scoring on validator-like datasets."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker44 import miner_heuristics as mh  # noqa: E402


def _runtime_capabilities() -> Dict[str, bool]:
    return {
        "has_score_chunk": hasattr(mh, "score_chunk"),
        "has_filtered_multi_leave_stats": hasattr(mh, "_filtered_multi_leave_stats"),
        "has_multi_leave_stats": hasattr(mh, "_multi_leave_stats"),
        "has_score_filtered_zero": hasattr(mh, "_score_filtered_zero"),
        "has_score_filtered_one": hasattr(mh, "_score_filtered_one"),
    }


def _install_warning_filters() -> None:
    # Scikit-learn 1.5+ may emit this warning repeatedly during parallel inference.
    # It is noisy for evaluation and does not change predicted values.
    warnings.filterwarnings(
        "ignore",
        message=(
            r"sklearn\.utils\.parallel\.delayed should be used with "
            r"sklearn\.utils\.parallel\.Parallel"
        ),
        category=UserWarning,
    )


def _force_single_thread_eval_models() -> None:
    # Avoid repeated joblib worker spin-up and sklearn parallel warnings in eval.
    for attr_name in ("_ML_FILTERED0_MODEL", "_ML_FILTERED1_MODEL", "_ML_SINGLE_HAND_MODEL"):
        model = getattr(mh, attr_name, None)
        if model is not None and hasattr(model, "n_jobs"):
            try:
                setattr(model, "n_jobs", 1)
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate miner effectiveness on validator_like_window datasets",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "analysis" / "data" / "validator_like_200x100",
        help="Directory with validator_like_window_*.json files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="validator_like_window_*.json",
        help="Glob pattern for dataset files",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of files to evaluate",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold: score >= threshold => bot",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print progress every N files",
    )
    parser.add_argument(
        "--progress-chunks",
        type=int,
        default=0,
        help="Optional progress print every N evaluated chunks (0 disables)",
    )
    parser.add_argument(
        "--errors-out",
        type=Path,
        default=None,
        help="Optional path to save detailed misclassifications as JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed debug info for each chunk (useful with --max-files 1)",
    )
    return parser.parse_args()


def _extract_chunks(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        chunks = payload.get("labeled_chunks", [])
        if isinstance(chunks, list):
            return chunks
    return []


def _filtered_stats(chunk: List[dict]) -> Tuple[Optional[int], int, int]:
    """
    Returns: (filtered_multi_leave, multi_joinleave, raw_multi_leave)
    Missing values are returned as 0 where needed by scorer, and None for unknown filtered bucket.
    """
    if hasattr(mh, "_multi_leave_stats"):
        try:
            stats = mh._multi_leave_stats(chunk)
            if isinstance(stats, tuple):
                filtered = int(stats[0]) if len(stats) > 0 else None
                mjl = int(stats[2]) if len(stats) > 2 else 0
                rml = int(stats[3]) if len(stats) > 3 else 0
                return filtered, mjl, rml
        except Exception:
            pass

    if hasattr(mh, "_filtered_multi_leave_stats"):
        try:
            stats = mh._filtered_multi_leave_stats(chunk)
            if isinstance(stats, tuple):
                filtered = int(stats[0]) if len(stats) > 0 else None
                return filtered, 0, 0
            return int(stats), 0, 0
        except Exception:
            pass

    return None, 0, 0


def _score_chunk(
    chunk: List[dict], filtered: Optional[int], mjl: int, rml: int
) -> Tuple[float, str, str]:
    # Single-hand chunks must go through score_chunk_modern so the
    # SINGLE_HAND_BOT_THRESHOLD calibration in _score_single_hand_chunk is applied,
    # matching production behaviour in neurons/miner.py.
    if len(chunk) == 1 and hasattr(mh, "score_chunk_modern"):
        return float(mh.score_chunk_modern(chunk)), "model", "single_hand->score_chunk_modern"

    caps = _runtime_capabilities()
    rich_api_available = (
        filtered is not None
        and caps["has_score_filtered_zero"]
        and caps["has_score_filtered_one"]
    )

    if rich_api_available:
        try:
            if filtered >= 2:
                return 0.0, "heuristic", "rich:f>=2->forced_human"
            if filtered == 1:
                # filtered=1 hardcut path is heuristic; non-hardcut path uses ML.
                source = "heuristic" if (mjl > 1 or rml > 1) else "model"
                reason = (
                    "rich:f1->heur_hardcut(mjl>1|rml>1)"
                    if source == "heuristic"
                    else "rich:f1->model"
                )
                return float(mh._score_filtered_one(chunk, mjl, rml)), source, reason
            return float(mh._score_filtered_zero(chunk, mjl, rml)), "model", "rich:f0->model"
        except Exception as exc:
            return float(mh.score_chunk(chunk)), "heuristic", f"fallback:rich_error:{type(exc).__name__}"

    missing = []
    if filtered is None:
        missing.append("filtered_unknown")
    if not caps["has_score_filtered_zero"]:
        missing.append("missing__score_filtered_zero")
    if not caps["has_score_filtered_one"]:
        missing.append("missing__score_filtered_one")
    reason = "fallback:" + ",".join(missing) if missing else "fallback:unknown"
    return float(mh.score_chunk(chunk)), "heuristic", reason


def _safe_div(num: int, den: int) -> float:
    return (num / den) if den else 0.0


def main() -> None:
    args = parse_args()
    _install_warning_filters()

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.exists():
        raise SystemExit(f"Data dir not found: {data_dir}")

    files = sorted(data_dir.glob(args.pattern))
    if args.max_files is not None:
        files = files[: args.max_files]

    if not files:
        raise SystemExit(f"No files matched pattern '{args.pattern}' in {data_dir}")

    caps = _runtime_capabilities()

    # Preload once and pin evaluation inference to single thread.
    mh._load_ml_model_filtered0()
    mh._load_ml_model_filtered1()
    if hasattr(mh, "_load_ml_model_single_hand"):
        mh._load_ml_model_single_hand()
    _force_single_thread_eval_models()

    print(f"[eval] data_dir={data_dir}")
    print(f"[eval] files={len(files)}")
    print(f"[eval] threshold={args.threshold:.3f} (score >= threshold => bot)")
    print("[eval] runtime capabilities:")
    for k in sorted(caps):
        print(f"  - {k}={caps[k]}")

    totals = {
        "chunks": 0,
        "correct": 0,
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "bot": 0,
        "human": 0,
    }

    bucket_totals: Dict[str, Dict[str, int]] = {
        "f0": {"chunks": 0, "correct": 0},
        "f1": {"chunks": 0, "correct": 0},
        "f2plus": {"chunks": 0, "correct": 0},
        "unknown": {"chunks": 0, "correct": 0},
    }

    detailed_errors: Dict[int, Dict[str, Dict[str, int]]] = {
        0: {
            "heuristic": {"bot": 0, "human": 0},
            "model": {"bot": 0, "human": 0},
        },
        1: {
            "heuristic": {"bot": 0, "human": 0},
            "model": {"bot": 0, "human": 0},
        },
    }

    detailed_correct: Dict[int, Dict[str, Dict[str, int]]] = {
        0: {
            "heuristic": {"bot": 0, "human": 0},
            "model": {"bot": 0, "human": 0},
        },
        1: {
            "heuristic": {"bot": 0, "human": 0},
            "model": {"bot": 0, "human": 0},
        },
    }

    route_counts: Dict[str, int] = {}

    per_file_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    
    # For verbose file loading
    if args.verbose and len(files) > 0:
        print(f"[verbose] About to load ML models...")
        t0 = time.perf_counter()
        mh._load_ml_model_filtered0()
        t1 = time.perf_counter()
        mh._load_ml_model_filtered1()
        t2 = time.perf_counter()
        if hasattr(mh, "_load_ml_model_single_hand"):
            mh._load_ml_model_single_hand()
        t3 = time.perf_counter()
        ml_stats = mh.get_ml_runtime_stats()
        print(f"[verbose] ML f0 loaded: {ml_stats['ml_model_loaded']}")
        print(f"[verbose] ML f0 load_time_s: {t1 - t0:.3f}")
        print(f"[verbose] ML f1 loaded: {ml_stats['ml_f1_model_loaded']}")
        print(f"[verbose] ML f1 load_time_s: {t2 - t1:.3f}")
        print(f"[verbose] ML single-hand loaded: {ml_stats.get('ml_single_hand_model_loaded', False)}")
        print(f"[verbose] ML single-hand available: {ml_stats.get('ml_single_hand_model_available', False)}")
        print(f"[verbose] ML single-hand load_time_s: {t3 - t2:.3f}")
        print()

    for idx, file_path in enumerate(files, 1):
        try:
            if file_path.suffix == ".gz":
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    payload = json.load(f)
            else:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[warn] failed to load {file_path.name}: {exc}")
            continue

        chunks = _extract_chunks(payload)
        file_chunks = 0
        file_correct = 0
        
        if args.verbose:
            print(f"[verbose] === File {idx}/{len(files)}: {file_path.name} ===")
            print(f"[verbose] Total chunks in file: {len(chunks)}")
            print()

        for chunk_idx, chunk_data in enumerate(chunks):
            hands = chunk_data.get("hands") or []
            is_bot = bool(chunk_data.get("is_bot", False))
            if not hands:
                continue

            filtered, mjl, rml = _filtered_stats(hands)
            score, score_source, route_reason = _score_chunk(hands, filtered, mjl, rml)
            route_counts[route_reason] = route_counts.get(route_reason, 0) + 1
            pred_bot = score >= args.threshold
            is_correct = pred_bot == is_bot
            
            if args.verbose:
                truth_label = "BOT" if is_bot else "HUM"
                pred_label = "BOT" if pred_bot else "HUM"
                status = "✓" if is_correct else "✗"
                print(f"[verbose] {status} chunk {chunk_idx:3d} | truth={truth_label} pred={pred_label} | score={score:.6f} | source={score_source} | route={route_reason}")
                print(f"[verbose]          | filtered={filtered} mjl={mjl} rml={rml} | hands={len(hands)} | chunk_data_keys={list(chunk_data.keys())}")
                print()

            totals["chunks"] += 1
            file_chunks += 1
            totals["bot" if is_bot else "human"] += 1

            if args.progress_chunks > 0 and (totals["chunks"] % args.progress_chunks == 0):
                running_acc = _safe_div(totals["correct"], totals["chunks"])
                print(
                    f"[eval] chunk_progress chunks={totals['chunks']} | files_done={idx}/{len(files)} | acc={running_acc:.4f}"
                )

            if is_correct:
                totals["correct"] += 1
                file_correct += 1

            if pred_bot and is_bot:
                totals["tp"] += 1
            elif pred_bot and (not is_bot):
                totals["fp"] += 1
            elif (not pred_bot) and is_bot:
                totals["fn"] += 1
            else:
                totals["tn"] += 1

            if filtered is None:
                bucket_name = "unknown"
            elif filtered == 0:
                bucket_name = "f0"
            elif filtered == 1:
                bucket_name = "f1"
            else:
                bucket_name = "f2plus"

            bucket_totals[bucket_name]["chunks"] += 1
            if is_correct:
                bucket_totals[bucket_name]["correct"] += 1

            if not is_correct and args.errors_out is not None:
                errors.append(
                    {
                        "file": file_path.name,
                        "chunk_index": chunk_idx,
                        "label": "bot" if is_bot else "human",
                        "prediction": "bot" if pred_bot else "human",
                        "score": round(float(score), 6),
                        "score_source": score_source,
                        "route_reason": route_reason,
                        "filtered_multi_leave": filtered,
                        "chunk_size": len(hands),
                    }
                )

            if is_correct and filtered in (0, 1):
                label_name = "bot" if is_bot else "human"
                if score_source not in ("heuristic", "model"):
                    score_source = "heuristic"
                detailed_correct[int(filtered)][score_source][label_name] += 1

            if (not is_correct) and filtered in (0, 1):
                label_name = "bot" if is_bot else "human"
                if score_source not in ("heuristic", "model"):
                    score_source = "heuristic"
                detailed_errors[int(filtered)][score_source][label_name] += 1

        if file_chunks:
            per_file_rows.append(
                {
                    "file": file_path.name,
                    "chunks": file_chunks,
                    "correct": file_correct,
                    "accuracy": _safe_div(file_correct, file_chunks),
                }
            )

        if args.print_every > 0 and (idx % args.print_every == 0 or idx == len(files)):
            running_acc = _safe_div(totals["correct"], totals["chunks"])
            print(
                f"[eval] progress {idx}/{len(files)} files | chunks={totals['chunks']} | acc={running_acc:.4f}"
            )

    if totals["chunks"] == 0:
        raise SystemExit("No chunks were evaluated (all files empty or unreadable)")

    overall_acc = _safe_div(totals["correct"], totals["chunks"])
    precision_bot = _safe_div(totals["tp"], totals["tp"] + totals["fp"])
    recall_bot = _safe_div(totals["tp"], totals["tp"] + totals["fn"])

    print("\n=== Decision Path Trace ===")
    print("conditions:")
    print("  - rich pipeline requires: filtered known AND _score_filtered_zero AND _score_filtered_one")
    print("  - fallback path: score_chunk heuristic")
    print("trigger counts:")
    for reason, count in sorted(route_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  - {reason}: {count}")

    print("\n=== Worst Files (Top 10) ===")
    for row in sorted(per_file_rows, key=lambda r: r["accuracy"])[:10]:
        print(
            f"{row['file']}: acc={row['accuracy']:.4f} ({row['correct']}/{row['chunks']})"
        )

    print("\n=== Overall ===")
    print(f"chunks:      {totals['chunks']}")
    print(f"labels:      bot={totals['bot']} human={totals['human']}")
    print(f"accuracy:    {overall_acc:.4f} ({totals['correct']}/{totals['chunks']})")
    print(f"precision_b: {precision_bot:.4f}")
    print(f"recall_b:    {recall_bot:.4f}")
    print(
        f"confusion:   TP={totals['tp']} FP={totals['fp']} FN={totals['fn']} TN={totals['tn']}"
    )

    print("\n=== Accuracy by Filter Bucket ===")
    for bucket in ("f0", "f1", "f2plus", "unknown"):
        c = bucket_totals[bucket]["chunks"]
        k = bucket_totals[bucket]["correct"]
        print(f"{bucket:8} { _safe_div(k, c):.4f} ({k}/{c})")

    print("\n=== Correct by Label and Source (filtered in {0,1}) ===")
    for filtered in (0, 1):
        print(f"filtered={filtered}")
        for source in ("heuristic", "model"):
            bot_ok = detailed_correct[filtered][source]["bot"]
            human_ok = detailed_correct[filtered][source]["human"]
            total_ok = bot_ok + human_ok
            print(
                f"  {source:9} total={total_ok:4d} | bot_correct={bot_ok:4d} human_correct={human_ok:4d}"
            )

    print("\n=== Errors by Label and Source (filtered in {0,1}) ===")
    for filtered in (0, 1):
        print(f"filtered={filtered}")
        for source in ("heuristic", "model"):
            bot_err = detailed_errors[filtered][source]["bot"]
            human_err = detailed_errors[filtered][source]["human"]
            total_err = bot_err + human_err
            print(
                f"  {source:9} total={total_err:4d} | bot_errors={bot_err:4d} human_errors={human_err:4d}"
            )

    if args.errors_out is not None:
        out_path = args.errors_out.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": {
                "data_dir": str(data_dir),
                "pattern": args.pattern,
                "files_evaluated": len(per_file_rows),
                "chunks": totals["chunks"],
                "accuracy": overall_acc,
                "tp": totals["tp"],
                "fp": totals["fp"],
                "fn": totals["fn"],
                "tn": totals["tn"],
                "threshold": args.threshold,
            },
            "errors": errors,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"\n[eval] wrote errors: {out_path} (count={len(errors)})")


if __name__ == "__main__":
    main()
