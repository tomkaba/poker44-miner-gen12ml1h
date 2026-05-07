#!/usr/bin/env python3
"""Train ML model for filtered=1 non-hardcut cases.

Data source:
- analysis/data/filtered1_nonhardcut_train_mixed_validator_001_270.json

This script:
1. Loads labeled chunks with hands + is_bot labels.
2. Extracts numerical features from hands.
3. Trains/evaluates multiple candidate models on a stratified split.
4. Saves best model artifacts under new filtered=1-specific names.
5. Backs up existing filtered=0 model/scaler before writing new files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from poker44.miner_heuristics import _extract_ml_features_filtered0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train filtered=1 non-hardcut model")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("analysis/data/filtered1_nonhardcut_train_mixed_validator_001_270.json"),
        help="Path to filtered=1 non-hardcut training dataset",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("weights"),
        help="Directory to save model artifacts",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Holdout fraction for final local validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Deduplicate rows by canonical hands hash before split",
    )
    return parser.parse_args()


def load_rows(dataset_path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows = payload.get("labeled_chunks") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Dataset does not contain a list of labeled chunks")
    return rows


def deduplicate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = {}
    for row in rows:
        hands = row.get("hands") or []
        key = hashlib.sha256(json.dumps(hands, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        if key not in seen:
            seen[key] = row
    return list(seen.values())


def extract_xy(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    x_list: List[np.ndarray] = []
    y_list: List[int] = []

    for row in rows:
        hands = row.get("hands") or []
        label = 1 if bool(row.get("is_bot")) else 0
        feats = _extract_ml_features_filtered0(hands)
        if feats is None:
            continue
        x_list.append(feats.astype(np.float32))
        y_list.append(label)

    if not x_list:
        raise ValueError("No usable rows after feature extraction")

    x = np.vstack(x_list)
    y = np.array(y_list, dtype=np.int32)
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


def evaluate_candidates(
    x_train: np.ndarray,
    y_train: np.ndarray,
    models: Dict[str, Any],
    seed: int,
) -> List[Dict[str, Any]]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results: List[Dict[str, Any]] = []

    for name, model in models.items():
        scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        results.append(
            {
                "name": name,
                "cv_accuracy_mean": float(np.mean(scores)),
                "cv_accuracy_std": float(np.std(scores)),
                "cv_scores": [float(s) for s in scores],
            }
        )

    results.sort(key=lambda r: r["cv_accuracy_mean"], reverse=True)
    return results


def backup_filtered0(weights_dir: Path) -> Dict[str, str]:
    backups: Dict[str, str] = {}
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    backup_dir = weights_dir / "backups" / f"filtered0_{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    for fname in ["ml_filtered0_model.pkl", "ml_filtered0_scaler.pkl"]:
        src = weights_dir / fname
        if src.exists():
            dst = backup_dir / fname
            shutil.copy2(src, dst)
            backups[fname] = str(dst)

    return backups


def extract_model_and_scaler(fitted_model: Any) -> Tuple[Any, Any]:
    # pipeline with scaler + estimator
    if hasattr(fitted_model, "named_steps"):
        scaler = fitted_model.named_steps.get("standardscaler")
        # final estimator is the last step
        step_names = list(fitted_model.named_steps.keys())
        estimator = fitted_model.named_steps[step_names[-1]]
        return estimator, scaler
    # tree model without scaler
    return fitted_model, None


def main() -> None:
    args = parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    rows = load_rows(args.dataset)
    original_rows = len(rows)
    if args.deduplicate:
        rows = deduplicate_rows(rows)

    x, y = extract_xy(rows)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    models = candidate_models(args.seed)
    ranking = evaluate_candidates(x_train, y_train, models, args.seed)
    best_name = ranking[0]["name"]

    best_model = models[best_name]
    best_model.fit(x_train, y_train)

    y_pred = best_model.predict(x_test)
    holdout_acc = float(accuracy_score(y_test, y_pred))
    holdout_precision = float(precision_score(y_test, y_pred, zero_division=0))
    holdout_recall = float(recall_score(y_test, y_pred, zero_division=0))
    holdout_f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

    weights_dir = args.weights_dir
    weights_dir.mkdir(parents=True, exist_ok=True)

    backups = backup_filtered0(weights_dir)

    model_obj, scaler_obj = extract_model_and_scaler(best_model)

    model_path = weights_dir / "ml_filtered1_nonhardcut_model.pkl"
    scaler_path = weights_dir / "ml_filtered1_nonhardcut_scaler.pkl"
    report_path = weights_dir / "ml_filtered1_nonhardcut_metrics.json"

    with open(model_path, "wb") as f:
        pickle.dump(model_obj, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_obj, f)

    report = {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "dataset": str(args.dataset),
        "dataset_rows_original": original_rows,
        "dataset_rows_used": int(len(rows)),
        "deduplicate": bool(args.deduplicate),
        "class_distribution": {
            "human_0": int(np.sum(y == 0)),
            "bot_1": int(np.sum(y == 1)),
        },
        "split": {
            "test_size": args.test_size,
            "seed": args.seed,
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
        },
        "candidates": ranking,
        "best_model": best_name,
        "holdout_metrics": {
            "accuracy": holdout_acc,
            "precision": holdout_precision,
            "recall": holdout_recall,
            "f1": holdout_f1,
            "confusion_matrix_labels_0_1": cm,
        },
        "artifacts": {
            "filtered1_model": str(model_path),
            "filtered1_scaler": str(scaler_path),
            "metrics": str(report_path),
            "filtered0_backups": backups,
        },
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== FILTERED=1 NON-HARDCUT TRAINING COMPLETE ===")
    print(f"Rows original: {original_rows}")
    print(f"Rows used:     {len(rows)}")
    print(f"Class dist:    human={int(np.sum(y == 0))} bot={int(np.sum(y == 1))}")
    print("\nCandidate ranking (CV accuracy):")
    for r in ranking:
        print(
            f"  {r['name']:<24} mean={r['cv_accuracy_mean']:.4f} std={r['cv_accuracy_std']:.4f}"
        )
    print(f"\nBest model: {best_name}")
    print(f"Holdout accuracy:  {holdout_acc:.4f}")
    print(f"Holdout precision: {holdout_precision:.4f}")
    print(f"Holdout recall:    {holdout_recall:.4f}")
    print(f"Holdout F1:        {holdout_f1:.4f}")
    print(f"Confusion matrix [0,1]: {cm}")
    print(f"\nSaved: {model_path}")
    print(f"Saved: {scaler_path}")
    print(f"Saved: {report_path}")
    if backups:
        print("Backups created for filtered0 artifacts:")
        for name, path in backups.items():
            print(f"  {name} -> {path}")


if __name__ == "__main__":
    main()
