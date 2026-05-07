#!/usr/bin/env python3
"""Tune dedicated single-hand model.

Workflow:
1. Load 1-hand datasets split by seed ranges.
2. Run RandomForest hyperparameter grid on train split.
3. For each model, scan decision thresholds on validation split.
4. Pick best by chosen optimization metric.
5. Retrain best config on train+validation.
6. Evaluate on test split and save artifacts + metrics report.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import itertools
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from poker44.miner_heuristics import _extract_ml_features_filtered0


SEED_RE = re.compile(r"seed_(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune random forest for single-hand chunks")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--pattern", type=str, default="public_benchmark_1hand_1000_seed_*.json.gz")
    parser.add_argument("--train-seed-min", type=int, default=1001)
    parser.add_argument("--train-seed-max", type=int, default=1040)
    parser.add_argument("--val-seed-min", type=int, default=1041)
    parser.add_argument("--val-seed-max", type=int, default=1046)
    parser.add_argument("--test-seed-min", type=int, default=1047)
    parser.add_argument("--test-seed-max", type=int, default=1050)
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--model-prefix", type=str, default="ml_single_hand_tuned")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deduplicate", action="store_true")

    parser.add_argument("--n-estimators", type=str, default="300,500,800")
    parser.add_argument("--max-depth", type=str, default="None,12,18,24")
    parser.add_argument("--min-samples-leaf", type=str, default="1,2,4")
    parser.add_argument("--max-features", type=str, default="sqrt,log2,0.5")
    parser.add_argument("--thresholds", type=str, default="0.35,0.40,0.45,0.50,0.55,0.60,0.65")
    parser.add_argument(
        "--optimize",
        type=str,
        default="f1",
        choices=["accuracy", "f1", "balanced_accuracy"],
        help="Metric used to select best hyperparameters and threshold on validation split",
    )

    return parser.parse_args()


def log(message: str) -> None:
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def parse_int_list(text: str) -> List[int]:
    return [int(tok.strip()) for tok in text.split(",") if tok.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(tok.strip()) for tok in text.split(",") if tok.strip()]


def parse_max_depth_list(text: str) -> List[int | None]:
    out: List[int | None] = []
    for tok in [t.strip() for t in text.split(",") if t.strip()]:
        if tok.lower() in {"none", "null"}:
            out.append(None)
        else:
            out.append(int(tok))
    return out


def parse_max_features_list(text: str) -> List[str | float]:
    out: List[str | float] = []
    for tok in [t.strip() for t in text.split(",") if t.strip()]:
        if tok in {"sqrt", "log2"}:
            out.append(tok)
        else:
            out.append(float(tok))
    return out


def extract_seed(path: Path) -> int:
    match = SEED_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse seed from filename: {path.name}")
    return int(match.group(1))


def seed_in_range(seed: int, lo: int, hi: int) -> bool:
    return lo <= seed <= hi


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("labeled_chunks") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported dataset structure in {path}")
    return rows


def row_hash(row: Dict[str, Any]) -> str:
    hands = row.get("hands") or []
    return hashlib.sha256(
        json.dumps(hands, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def deduplicate_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = row_hash(row)
        if key not in seen:
            seen[key] = row
    return list(seen.values())


def collect_split_rows(files: Sequence[Path], deduplicate: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, path in enumerate(files, 1):
        log(f"Loading {idx}/{len(files)}: {path.name}")
        rows = load_rows(path)
        kept = 0
        for row in rows:
            hands = row.get("hands") or []
            if len(hands) != 1:
                continue
            if "is_bot" not in row:
                continue
            out.append(row)
            kept += 1
        log(f"  kept {kept}/{len(rows)} rows")

    if deduplicate:
        before = len(out)
        out = deduplicate_rows(out)
        log(f"Deduplicated rows: {before} -> {len(out)}")
    return out


def extract_xy(rows: Sequence[Dict[str, Any]], split_name: str) -> Tuple[np.ndarray, np.ndarray]:
    x_list: List[np.ndarray] = []
    y_list: List[int] = []
    log(f"Extracting features for {split_name}: {len(rows)} rows")
    for idx, row in enumerate(rows, 1):
        hands = row.get("hands") or []
        feats = _extract_ml_features_filtered0(hands)
        if feats is None:
            continue
        x_list.append(feats.astype(np.float32))
        y_list.append(1 if bool(row.get("is_bot")) else 0)
        if idx % 5000 == 0 or idx == len(rows):
            log(f"  progress {split_name}: {idx}/{len(rows)}")

    if not x_list:
        raise ValueError(f"No usable rows for split: {split_name}")
    x = np.vstack(x_list)
    y = np.array(y_list, dtype=np.int32)
    log(f"Finished {split_name}: X={x.shape} human={(y == 0).sum()} bot={(y == 1).sum()}")
    return x, y


def split_files_by_seed(files: Iterable[Path], args: argparse.Namespace) -> Dict[str, List[Path]]:
    buckets = {"train": [], "val": [], "test": []}
    for path in sorted(files):
        seed = extract_seed(path)
        if seed_in_range(seed, args.train_seed_min, args.train_seed_max):
            buckets["train"].append(path)
        elif seed_in_range(seed, args.val_seed_min, args.val_seed_max):
            buckets["val"].append(path)
        elif seed_in_range(seed, args.test_seed_min, args.test_seed_max):
            buckets["test"].append(path)
    return buckets


def ensure_non_empty_split(name: str, files: Sequence[Path]) -> None:
    if not files:
        raise ValueError(f"No files assigned to split '{name}'")


def predict_with_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    return (probs >= threshold).astype(np.int32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | int | List[List[int]]]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = 0.5 * (recall + tnr)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix_labels_0_1": cm.tolist(),
    }


def score_for_objective(metrics: Dict[str, Any], objective: str) -> float:
    return float(metrics[objective])


def build_rf(params: Dict[str, Any], seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    weights_dir = args.weights_dir.expanduser().resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)

    n_estimators_grid = parse_int_list(args.n_estimators)
    max_depth_grid = parse_max_depth_list(args.max_depth)
    min_leaf_grid = parse_int_list(args.min_samples_leaf)
    max_features_grid = parse_max_features_list(args.max_features)
    threshold_grid = parse_float_list(args.thresholds)

    if not threshold_grid:
        raise ValueError("Threshold grid cannot be empty")

    log("Starting single-hand RF tuning")
    log(f"Optimization metric: {args.optimize}")
    log(f"Threshold grid: {threshold_grid}")

    all_files = sorted(data_dir.glob(args.pattern))
    if not all_files:
        raise FileNotFoundError(f"No files matched {args.pattern} in {data_dir}")
    split_files = split_files_by_seed(all_files, args)
    ensure_non_empty_split("train", split_files["train"])
    ensure_non_empty_split("val", split_files["val"])
    ensure_non_empty_split("test", split_files["test"])

    for split in ("train", "val", "test"):
        seeds = [extract_seed(p) for p in split_files[split]]
        log(f"Split {split}: files={len(split_files[split])} seeds={min(seeds)}..{max(seeds)}")

    train_rows = collect_split_rows(split_files["train"], deduplicate=args.deduplicate)
    val_rows = collect_split_rows(split_files["val"], deduplicate=args.deduplicate)
    test_rows = collect_split_rows(split_files["test"], deduplicate=args.deduplicate)

    x_train, y_train = extract_xy(train_rows, "train")
    x_val, y_val = extract_xy(val_rows, "validation")
    x_test, y_test = extract_xy(test_rows, "test")

    grid = list(
        itertools.product(
            n_estimators_grid,
            max_depth_grid,
            min_leaf_grid,
            max_features_grid,
        )
    )
    log(f"Hyperparameter combinations: {len(grid)}")

    tuning_rows: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None

    for idx, (n_estimators, max_depth, min_leaf, max_features) in enumerate(grid, 1):
        params = {
            "n_estimators": int(n_estimators),
            "max_depth": max_depth,
            "min_samples_leaf": int(min_leaf),
            "max_features": max_features,
        }
        log(
            f"[{idx}/{len(grid)}] train RF n={params['n_estimators']} depth={params['max_depth']} "
            f"leaf={params['min_samples_leaf']} max_features={params['max_features']}"
        )

        model = build_rf(params, args.seed)
        model.fit(x_train, y_train)
        val_probs = model.predict_proba(x_val)[:, 1]

        best_thr_metrics = None
        best_thr = None
        for thr in threshold_grid:
            val_pred = predict_with_threshold(val_probs, thr)
            val_metrics = compute_metrics(y_val, val_pred)
            if best_thr_metrics is None:
                best_thr_metrics = val_metrics
                best_thr = thr
                continue

            cur = score_for_objective(val_metrics, args.optimize)
            prev = score_for_objective(best_thr_metrics, args.optimize)
            if cur > prev:
                best_thr_metrics = val_metrics
                best_thr = thr

        assert best_thr_metrics is not None and best_thr is not None

        candidate = {
            "params": params,
            "best_threshold": float(best_thr),
            "validation_metrics": best_thr_metrics,
        }
        tuning_rows.append(candidate)

        score = score_for_objective(best_thr_metrics, args.optimize)
        log(
            f"  best threshold={best_thr:.3f} "
            f"acc={best_thr_metrics['accuracy']:.4f} "
            f"f1={best_thr_metrics['f1']:.4f} bal_acc={best_thr_metrics['balanced_accuracy']:.4f} "
            f"({args.optimize}={score:.4f})"
        )

        if best is None:
            best = candidate
        else:
            best_score = score_for_objective(best["validation_metrics"], args.optimize)
            if score > best_score:
                best = candidate

    assert best is not None
    best_params = best["params"]
    best_threshold = float(best["best_threshold"])

    log("Retraining best config on train+validation")
    x_trainval = np.vstack([x_train, x_val])
    y_trainval = np.concatenate([y_train, y_val])
    final_model = build_rf(best_params, args.seed)
    final_model.fit(x_trainval, y_trainval)

    log(f"Evaluating on test split with threshold={best_threshold:.3f}")
    test_probs = final_model.predict_proba(x_test)[:, 1]
    test_pred = predict_with_threshold(test_probs, best_threshold)
    test_metrics = compute_metrics(y_test, test_pred)

    model_path = weights_dir / f"{args.model_prefix}_model.pkl"
    scaler_path = weights_dir / f"{args.model_prefix}_scaler.pkl"
    metrics_path = weights_dir / f"{args.model_prefix}_metrics.json"

    log(f"Saving model: {model_path}")
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    # Keep scaler artifact for compatibility with existing loaders.
    log(f"Saving scaler placeholder: {scaler_path}")
    with open(scaler_path, "wb") as f:
        pickle.dump(None, f)

    tuning_rows.sort(
        key=lambda row: score_for_objective(row["validation_metrics"], args.optimize),
        reverse=True,
    )

    report = {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "objective": args.optimize,
        "threshold_grid": threshold_grid,
        "grids": {
            "n_estimators": n_estimators_grid,
            "max_depth": max_depth_grid,
            "min_samples_leaf": min_leaf_grid,
            "max_features": max_features_grid,
        },
        "data": {
            "dir": str(data_dir),
            "pattern": args.pattern,
            "deduplicate": bool(args.deduplicate),
            "seed_ranges": {
                "train": [args.train_seed_min, args.train_seed_max],
                "validation": [args.val_seed_min, args.val_seed_max],
                "test": [args.test_seed_min, args.test_seed_max],
            },
            "rows": {
                "train": int(len(y_train)),
                "validation": int(len(y_val)),
                "test": int(len(y_test)),
            },
        },
        "best": {
            "params": best_params,
            "threshold": best_threshold,
            "validation_metrics": best["validation_metrics"],
            "test_metrics": test_metrics,
        },
        "top_candidates": tuning_rows[:10],
        "artifacts": {
            "model": str(model_path),
            "scaler": str(scaler_path),
            "metrics": str(metrics_path),
        },
    }

    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("=== SINGLE-HAND TUNING COMPLETE ===")
    print(f"Objective:         {args.optimize}")
    print(f"Best params:       {best_params}")
    print(f"Best threshold:    {best_threshold:.3f}")
    print(f"Validation acc:    {best['validation_metrics']['accuracy']:.4f}")
    print(f"Validation F1:     {best['validation_metrics']['f1']:.4f}")
    print(f"Validation balAcc: {best['validation_metrics']['balanced_accuracy']:.4f}")
    print(f"Test accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"Test precision:    {test_metrics['precision']:.4f}")
    print(f"Test recall:       {test_metrics['recall']:.4f}")
    print(f"Test F1:           {test_metrics['f1']:.4f}")
    print(f"Test balAcc:       {test_metrics['balanced_accuracy']:.4f}")
    print(
        "Test confusion:    "
        f"TN={test_metrics['tn']} FP={test_metrics['fp']} FN={test_metrics['fn']} TP={test_metrics['tp']}"
    )
    print(f"Saved model:       {model_path}")
    print(f"Saved scaler:      {scaler_path}")
    print(f"Saved metrics:     {metrics_path}")


if __name__ == "__main__":
    main()
