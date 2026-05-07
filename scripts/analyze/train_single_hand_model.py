#!/usr/bin/env python3
"""Train a dedicated ML model for single-hand chunks.

Expected dataset format:
- data/public_benchmark_1hand_1000_seed_<SEED>.json.gz

Default split by source file seed:
- train: 1001-1040
- validation: 1041-1046
- test: 1047-1050

This avoids mixing chunks from the same generated file across splits.
Artifacts are saved under weights/ using single-hand specific names.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from poker44.miner_heuristics import _extract_ml_features_filtered0


SEED_RE = re.compile(r"seed_(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dedicated model for 1-hand chunks")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing generated single-hand benchmark files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="public_benchmark_1hand_1000_seed_*.json.gz",
        help="Glob pattern for source files inside data-dir",
    )
    parser.add_argument(
        "--train-seed-min",
        type=int,
        default=1001,
        help="Minimum seed for train split",
    )
    parser.add_argument(
        "--train-seed-max",
        type=int,
        default=1040,
        help="Maximum seed for train split",
    )
    parser.add_argument(
        "--val-seed-min",
        type=int,
        default=1041,
        help="Minimum seed for validation split",
    )
    parser.add_argument(
        "--val-seed-max",
        type=int,
        default=1046,
        help="Maximum seed for validation split",
    )
    parser.add_argument(
        "--test-seed-min",
        type=int,
        default=1047,
        help="Minimum seed for test split",
    )
    parser.add_argument(
        "--test-seed-max",
        type=int,
        default=1050,
        help="Maximum seed for test split",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("weights"),
        help="Directory to save model artifacts",
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        default="ml_single_hand",
        help="Artifact prefix for model/scaler/metrics files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for candidate models",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Deduplicate rows within each split by canonical hand hash",
    )
    return parser.parse_args()


def log(message: str) -> None:
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


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
    all_rows: List[Dict[str, Any]] = []
    total_kept = 0
    total_seen = 0

    for idx, path in enumerate(files, 1):
        log(f"Loading file {idx}/{len(files)}: {path.name}")
        rows = load_rows(path)
        total_seen += len(rows)

        kept = 0
        for row in rows:
            hands = row.get("hands") or []
            if len(hands) != 1:
                continue
            if "is_bot" not in row:
                continue
            all_rows.append(row)
            kept += 1

        total_kept += kept
        log(f"  kept {kept}/{len(rows)} rows after single-hand + label filtering")

    if deduplicate:
        before = len(all_rows)
        all_rows = deduplicate_rows(all_rows)
        log(f"Deduplicated split rows: {before} -> {len(all_rows)}")

    log(f"Finished split load: source_rows={total_seen} usable_rows={total_kept} final_rows={len(all_rows)}")
    return all_rows


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
            log(f"  feature extraction progress for {split_name}: {idx}/{len(rows)}")

    if not x_list:
        raise ValueError(f"No usable rows after feature extraction for split: {split_name}")

    x = np.vstack(x_list)
    y = np.array(y_list, dtype=np.int32)
    log(
        f"Completed features for {split_name}: X={x.shape} "
        f"human={(y == 0).sum()} bot={(y == 1).sum()}"
    )
    return x, y


def candidate_models(seed: int) -> Dict[str, Any]:
    return {
        "logreg_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                random_state=seed,
                solver="lbfgs",
            ),
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=700,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix_labels_0_1": cm.tolist(),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def evaluate_model(name: str, model: Any, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray) -> Dict[str, Any]:
    log(f"Training candidate: {name}")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_eval)
    metrics = compute_metrics(y_eval, y_pred)
    log(
        f"  {name}: acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
        f"rec={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
    )
    return {
        "name": name,
        "validation_metrics": metrics,
        "fitted_model": model,
    }


def extract_model_and_scaler(fitted_model: Any) -> Tuple[Any, Any]:
    if hasattr(fitted_model, "named_steps"):
        scaler = fitted_model.named_steps.get("standardscaler")
        step_names = list(fitted_model.named_steps.keys())
        estimator = fitted_model.named_steps[step_names[-1]]
        return estimator, scaler
    return fitted_model, None


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


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    weights_dir = args.weights_dir.expanduser().resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)

    log("Starting single-hand model training")
    log(f"Data dir: {data_dir}")
    log(f"Pattern:  {args.pattern}")

    all_files = sorted(data_dir.glob(args.pattern))
    if not all_files:
        raise FileNotFoundError(f"No files matched {args.pattern} in {data_dir}")

    log(f"Matched files: {len(all_files)}")
    split_files = split_files_by_seed(all_files, args)
    ensure_non_empty_split("train", split_files["train"])
    ensure_non_empty_split("val", split_files["val"])
    ensure_non_empty_split("test", split_files["test"])

    for split_name in ("train", "val", "test"):
        seeds = [extract_seed(path) for path in split_files[split_name]]
        log(
            f"Split {split_name}: files={len(split_files[split_name])} "
            f"seeds={min(seeds)}..{max(seeds)}"
        )

    train_rows = collect_split_rows(split_files["train"], deduplicate=args.deduplicate)
    val_rows = collect_split_rows(split_files["val"], deduplicate=args.deduplicate)
    test_rows = collect_split_rows(split_files["test"], deduplicate=args.deduplicate)

    x_train, y_train = extract_xy(train_rows, "train")
    x_val, y_val = extract_xy(val_rows, "validation")
    x_test, y_test = extract_xy(test_rows, "test")

    models = candidate_models(args.seed)
    ranking: List[Dict[str, Any]] = []

    log("Evaluating candidate models on validation split")
    for name, model in models.items():
        result = evaluate_model(name, model, x_train, y_train, x_val, y_val)
        ranking.append(result)

    ranking.sort(key=lambda item: item["validation_metrics"]["accuracy"], reverse=True)
    best_name = ranking[0]["name"]
    log(f"Best validation model: {best_name}")

    x_trainval = np.vstack([x_train, x_val])
    y_trainval = np.concatenate([y_train, y_val])

    log(f"Retraining best model on train+validation: {best_name}")
    final_model = candidate_models(args.seed)[best_name]
    final_model.fit(x_trainval, y_trainval)

    log("Running final evaluation on held-out test split")
    test_pred = final_model.predict(x_test)
    test_metrics = compute_metrics(y_test, test_pred)
    log(
        f"Final test metrics: acc={test_metrics['accuracy']:.4f} "
        f"prec={test_metrics['precision']:.4f} rec={test_metrics['recall']:.4f} f1={test_metrics['f1']:.4f}"
    )

    model_obj, scaler_obj = extract_model_and_scaler(final_model)
    model_path = weights_dir / f"{args.model_prefix}_model.pkl"
    scaler_path = weights_dir / f"{args.model_prefix}_scaler.pkl"
    metrics_path = weights_dir / f"{args.model_prefix}_metrics.json"

    log(f"Saving model artifact: {model_path.name}")
    with open(model_path, "wb") as f:
        pickle.dump(model_obj, f)

    log(f"Saving scaler artifact: {scaler_path.name}")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_obj, f)

    report = {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "data_dir": str(data_dir),
        "pattern": args.pattern,
        "deduplicate": bool(args.deduplicate),
        "seed_ranges": {
            "train": [args.train_seed_min, args.train_seed_max],
            "validation": [args.val_seed_min, args.val_seed_max],
            "test": [args.test_seed_min, args.test_seed_max],
        },
        "files": {
            split: [str(path) for path in split_files[split]]
            for split in ("train", "val", "test")
        },
        "rows": {
            "train": int(len(y_train)),
            "validation": int(len(y_val)),
            "test": int(len(y_test)),
            "train_human": int((y_train == 0).sum()),
            "train_bot": int((y_train == 1).sum()),
            "validation_human": int((y_val == 0).sum()),
            "validation_bot": int((y_val == 1).sum()),
            "test_human": int((y_test == 0).sum()),
            "test_bot": int((y_test == 1).sum()),
        },
        "candidates": [
            {
                "name": item["name"],
                "validation_metrics": item["validation_metrics"],
            }
            for item in ranking
        ],
        "best_model": best_name,
        "test_metrics": test_metrics,
        "artifacts": {
            "model": str(model_path),
            "scaler": str(scaler_path),
            "metrics": str(metrics_path),
        },
    }

    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("=== SINGLE-HAND TRAINING COMPLETE ===")
    print(f"Best model:        {best_name}")
    print(f"Validation leader: {ranking[0]['validation_metrics']['accuracy']:.4f}")
    print(f"Test accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"Test precision:    {test_metrics['precision']:.4f}")
    print(f"Test recall:       {test_metrics['recall']:.4f}")
    print(f"Test F1:           {test_metrics['f1']:.4f}")
    print(f"Test confusion:    TN={test_metrics['tn']} FP={test_metrics['fp']} FN={test_metrics['fn']} TP={test_metrics['tp']}")
    print(f"Saved model:       {model_path}")
    print(f"Saved scaler:      {scaler_path}")
    print(f"Saved metrics:     {metrics_path}")


if __name__ == "__main__":
    main()