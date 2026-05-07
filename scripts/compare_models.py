#!/usr/bin/env python3
"""Compare Gen1, active miner Gen3, and Gen4 on single-hand profile suites."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

import poker44.miner_heuristics as mh


def safe_float(v: Any) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def bb(hand: Dict[str, Any]) -> float:
    metadata = hand.get("metadata") or {}
    return safe_float(metadata.get("bb")) or 0.01


def extract_chunk_features(row: Dict[str, Any], generation: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
    hands = row.get("hands") or []
    is_bot = 1 if row.get("is_bot") else 0
    if not hands:
        return None, None

    if generation in {"Gen1", "Gen3", "Gen4+"}:
        return mh._extract_ml_features_single_hand_v2([hands[0]]), is_bot

    if generation == "Gen4":
        # Use the same extractor that the miner uses at runtime for Gen4.
        return mh._extract_ml_features_gen4([hands[0]]), is_bot

    return None, None


def load_profile(profile_path: Path, generation: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    features_list: List[np.ndarray] = []
    labels_list: List[int] = []

    with gzip.open(profile_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    for row in data.get("labeled_chunks", []):
        feat, label = extract_chunk_features(row, generation)
        if feat is not None:
            features_list.append(feat)
            labels_list.append(label)

    if not features_list:
        return None, None
    return np.array(features_list, dtype=np.float32), np.array(labels_list, dtype=np.int32)


def evaluate_model(model: Any, scaler: Any, X: np.ndarray, y: np.ndarray, generation: str, profile_name: str, threshold: float) -> Dict[str, Any]:
    X_batch = scaler.transform(X) if scaler is not None else X
    proba = model.predict_proba(X_batch)[:, 1]
    pred = (proba >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "generation": generation,
        "profile": profile_name,
        "samples": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, proba)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "errors": int(fp + fn),
        "threshold": float(threshold),
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_samples = sum(r["samples"] for r in results)
    total_tp = sum(r["tp"] for r in results)
    total_tn = sum(r["tn"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)
    total_errors = total_fp + total_fn
    return {
        "profiles": len(results),
        "samples": total_samples,
        "accuracy_avg": float(np.mean([r["accuracy"] for r in results])),
        "f1_avg": float(np.mean([r["f1"] for r in results])),
        "auc_avg": float(np.mean([r["auc"] for r in results])),
        "tp": total_tp,
        "tn": total_tn,
        "fp": total_fp,
        "fn": total_fn,
        "errors": total_errors,
        "error_rate": float(total_errors / total_samples) if total_samples else 0.0,
    }


def print_profile_table(profile_names: List[str], by_profile: Dict[str, Dict[str, Dict[str, Any]]], generations: List[str]) -> None:
    print("\nPER-PROFILE BREAKDOWN")
    print("=" * 170)
    print(f"{'Profile':<30} | {'Gen':<4} | {'Acc':<8} | {'F1':<8} | {'Err':<5} | {'TP':<5} | {'TN':<5} | {'FP':<5} | {'FN':<5} | {'Thr':<5}")
    print("-" * 170)
    for profile_name in profile_names:
        for generation in generations:
            result = by_profile[profile_name].get(generation)
            if result is None:
                continue
            print(
                f"{profile_name:<30} | {generation:<4} | {result['accuracy']:.4f} | {result['f1']:.4f} | {result['errors']:<5} | "
                f"{result['tp']:<5} | {result['tn']:<5} | {result['fp']:<5} | {result['fn']:<5} | {result['threshold']:.2f}"
            )
        print("-" * 170)


def print_summary_table(all_results: Dict[str, List[Dict[str, Any]]], generations: List[str]) -> None:
    print("\nOVERALL SUMMARY")
    print("=" * 120)
    print(f"{'Gen':<4} | {'Profiles':<8} | {'Samples':<8} | {'Acc(avg)':<10} | {'F1(avg)':<10} | {'Errors':<8} | {'TP':<8} | {'TN':<8} | {'FP':<8} | {'FN':<8}")
    print("-" * 120)
    for generation in generations:
        summary = summarize(all_results[generation])
        print(
            f"{generation:<4} | {summary['profiles']:<8} | {summary['samples']:<8} | {summary['accuracy_avg']:.4f}     | "
            f"{summary['f1_avg']:.4f}    | {summary['errors']:<8} | {summary['tp']:<8} | {summary['tn']:<8} | "
            f"{summary['fp']:<8} | {summary['fn']:<8}"
        )
    print("=" * 120)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Gen1, active miner Gen3, and Gen4 on profile suites")
    parser.add_argument("--profile-dir", default="data/public_benchmark_custom_1hand_5000_20260404_092644")
    parser.add_argument("--gen1-model", default="weights/ml_single_hand_v3_search_model.pkl")
    parser.add_argument("--gen1-scaler", default="weights/ml_single_hand_v3_search_scaler.pkl")
    parser.add_argument("--gen1-threshold", type=float, default=0.35)
    parser.add_argument("--gen3-model", default="weights/ml_single_hand_model.pkl")
    parser.add_argument("--gen3-scaler", default="weights/ml_single_hand_scaler.pkl")
    parser.add_argument("--gen3-threshold", type=float, default=0.50)
    parser.add_argument("--gen4-model", default="weights/ml_gen4_model.pkl")
    parser.add_argument("--gen4-scaler", default="weights/ml_gen4_scaler.pkl")
    parser.add_argument("--gen4-threshold", type=float, default=0.30)
    parser.add_argument("--gen4plus-model", default="weights/ml_single_hand_v4plus_s12346_model.pkl")
    parser.add_argument("--gen4plus-scaler", default="weights/ml_single_hand_v4plus_s12346_scaler.pkl")
    parser.add_argument("--gen4plus-threshold", type=float, default=0.50)
    parser.add_argument("--output", help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile_dir = Path(args.profile_dir)
    if not profile_dir.exists():
        print(f"ERROR: missing profile dir: {profile_dir}")
        return 1

    profiles = sorted(profile_dir.glob("public_1hand_5000_*.json.gz"))
    if not profiles:
        print(f"ERROR: no profile files in {profile_dir}")
        return 1

    generations = {
        "Gen1": (Path(args.gen1_model), Path(args.gen1_scaler), args.gen1_threshold),
        "Gen3": (Path(args.gen3_model), Path(args.gen3_scaler), args.gen3_threshold),
        "Gen4": (Path(args.gen4_model), Path(args.gen4_scaler), args.gen4_threshold),
        "Gen4+": (Path(args.gen4plus_model), Path(args.gen4plus_scaler), args.gen4plus_threshold),
    }

    loaded: Dict[str, Tuple[Any, Any, float]] = {}
    for generation, (model_path, scaler_path, threshold) in generations.items():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        scaler = None
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        loaded[generation] = (model, scaler, threshold)

    by_profile: Dict[str, Dict[str, Dict[str, Any]]] = {}
    all_results: Dict[str, List[Dict[str, Any]]] = {generation: [] for generation in generations}

    for profile_path in profiles:
        profile_name = profile_path.stem.replace("public_1hand_5000_profile_", "").replace("public_1hand_5000_preset_", "")
        by_profile[profile_name] = {}
        for generation in generations:
            X, y = load_profile(profile_path, generation)
            if X is None or y is None:
                continue
            model, scaler, threshold = loaded[generation]
            result = evaluate_model(model, scaler, X, y, generation, profile_name, threshold)
            by_profile[profile_name][generation] = result
            all_results[generation].append(result)

    ordered_profiles = sorted(by_profile.keys())
    ordered_generations = ["Gen1", "Gen3", "Gen4", "Gen4+"]
    print_profile_table(ordered_profiles, by_profile, ordered_generations)
    print_summary_table(all_results, ordered_generations)

    print("\nDELTA VS ACTIVE MINER GEN3")
    print("=" * 90)
    gen3_summary = summarize(all_results["Gen3"])
    for generation in ["Gen1", "Gen4", "Gen4+"]:
        summary = summarize(all_results[generation])
        print(
            f"{generation} vs Gen3: errors delta={summary['errors'] - gen3_summary['errors']}, "
            f"accuracy delta={(summary['accuracy_avg'] - gen3_summary['accuracy_avg']) * 100:.3f} pp, "
            f"fp delta={summary['fp'] - gen3_summary['fp']}, fn delta={summary['fn'] - gen3_summary['fn']}"
        )
    print("=" * 90)

    if args.output:
        payload = {
            "by_profile": by_profile,
            "summary": {generation: summarize(results) for generation, results in all_results.items()},
            "thresholds": {generation: loaded[generation][2] for generation in loaded},
        }
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved JSON to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
