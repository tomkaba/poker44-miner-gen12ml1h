#!/usr/bin/env python3
"""Search for a stronger single-hand model on the same split used by v2.

Goal:
- Keep exactly the same data regime as v2 for apples-to-apples comparison.
- Try broader candidate family set (sklearn + optional external boosters).
- Tune decision threshold on validation split for each candidate.
- Rank candidates and save best model + full report.
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
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED_RE = re.compile(r"seed_(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search stronger single-hand models on v2 split")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--patterns", type=str, default="public_benchmark_1hand_1000_seed_*.json.gz,public_benchmark_1hand_1000_set2_seed_*.json.gz")
    parser.add_argument("--train-seed-min", type=int, default=1001)
    parser.add_argument("--train-seed-max", type=int, default=1040)
    parser.add_argument("--val-seed-min", type=int, default=1041)
    parser.add_argument("--val-seed-max", type=int, default=1046)
    parser.add_argument("--test-seed-min", type=int, default=1047)
    parser.add_argument("--test-seed-max", type=int, default=1050)
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--model-prefix", type=str, default="ml_single_hand_v3_search")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--optimize", choices=["accuracy", "f1", "balanced_accuracy"], default="f1")
    parser.add_argument("--thresholds", type=str, default="0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="Enable additional slower candidates (kNN and larger ensembles)",
    )
    return parser.parse_args()


def log(message: str) -> None:
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def parse_float_list(text: str) -> List[float]:
    return [float(tok.strip()) for tok in text.split(",") if tok.strip()]


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


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _single_hand_bb(hand: dict) -> float:
    bb = _safe_float((hand.get("metadata") or {}).get("bb"))
    return bb if bb > 0 else 0.05


def extract_single_hand_features(hand: Dict[str, Any]) -> np.ndarray:
    players = hand.get("players") or []
    actions = hand.get("actions") or []
    outcome = hand.get("outcome") or {}
    streets = hand.get("streets") or []
    metadata = hand.get("metadata") or {}

    bb = _single_hand_bb(hand)
    max_seats = int(metadata.get("max_seats") or 6)
    max_seats = max(max_seats, 1)

    num_players = float(len(players))
    filled_ratio = num_players / float(max_seats)

    starting_stacks = [_safe_float(p.get("starting_stack")) for p in players]
    stack_mean = float(np.mean(starting_stacks)) if starting_stacks else 0.0
    stack_std = float(np.std(starting_stacks)) if starting_stacks else 0.0
    stack_cv = stack_std / (stack_mean + 1e-9)

    action_types = [str(a.get("action_type") or "").lower() for a in actions]
    total_actions = float(len(action_types))

    def _count(name: str) -> float:
        return float(sum(1 for t in action_types if t == name))

    call_c = _count("call")
    check_c = _count("check")
    fold_c = _count("fold")
    raise_c = _count("raise")
    bet_c = _count("bet")
    allin_c = float(sum(1 for t in action_types if "all_in" in t or "all-in" in t))

    meaningful = call_c + check_c + fold_c + raise_c + bet_c
    if meaningful > 0:
        call_r = call_c / meaningful
        check_r = check_c / meaningful
        fold_r = fold_c / meaningful
        raise_r = raise_c / meaningful
        bet_r = bet_c / meaningful
    else:
        call_r = check_r = fold_r = raise_r = bet_r = 0.0

    agg_ratio = (raise_c + bet_c) / (call_c + check_c + 1.0)

    amounts = [_safe_float(a.get("amount")) for a in actions]
    amounts_pos = [a for a in amounts if a > 0]
    amount_mean_bb = (float(np.mean(amounts_pos)) / bb) if amounts_pos else 0.0
    amount_max_bb = (float(np.max(amounts_pos)) / bb) if amounts_pos else 0.0
    amount_std_bb = (float(np.std(amounts_pos)) / bb) if len(amounts_pos) > 1 else 0.0

    total_pot = _safe_float(outcome.get("total_pot"))
    total_pot_bb = total_pot / bb
    showdown = 1.0 if bool(outcome.get("showdown")) else 0.0
    payouts = outcome.get("payouts") or {}
    winner_count = float(sum(1 for _, v in payouts.items() if _safe_float(v) > 0))
    winner_share = winner_count / (num_players + 1e-9)

    n_streets = float(len(streets))
    actions_per_player = total_actions / (num_players + 1e-9)
    actions_per_street = total_actions / (n_streets + 1e-9)

    feats = np.array(
        [
            num_players,
            float(max_seats),
            filled_ratio,
            stack_mean,
            stack_std,
            stack_cv,
            total_actions,
            meaningful,
            call_r,
            check_r,
            fold_r,
            raise_r,
            bet_r,
            agg_ratio,
            allin_c,
            amount_mean_bb,
            amount_max_bb,
            amount_std_bb,
            total_pot_bb,
            showdown,
            winner_count,
            winner_share,
            n_streets,
            actions_per_player,
            actions_per_street,
        ],
        dtype=np.float32,
    )
    return feats


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
        if len(hands) != 1:
            continue
        feats = extract_single_hand_features(hands[0])
        x_list.append(feats)
        y_list.append(1 if bool(row.get("is_bot")) else 0)

        if idx % 5000 == 0 or idx == len(rows):
            log(f"  progress {split_name}: {idx}/{len(rows)}")

    if not x_list:
        raise ValueError(f"No usable rows for split: {split_name}")

    x = np.vstack(x_list)
    y = np.array(y_list, dtype=np.int32)
    log(f"Finished {split_name}: X={x.shape} human={(y == 0).sum()} bot={(y == 1).sum()}")
    return x, y


def predict_with_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    return (probs >= threshold).astype(np.int32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    cm = np.array([[0, 0], [0, 0]], dtype=np.int64)
    for yt, yp in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(yt), int(yp)] += 1

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


def candidate_models(seed: int, include_slow: bool) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "hgb_v2_like": HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.05,
            max_iter=500,
            min_samples_leaf=30,
            random_state=seed,
        ),
        "hgb_deeper": HistGradientBoostingClassifier(
            max_depth=12,
            learning_rate=0.04,
            max_iter=700,
            min_samples_leaf=20,
            random_state=seed,
        ),
        "hgb_regularized": HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.03,
            max_iter=900,
            min_samples_leaf=40,
            l2_regularization=0.2,
            random_state=seed,
        ),
        "extra_trees_large": ExtraTreesClassifier(
            n_estimators=1200,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "rf_large": RandomForestClassifier(
            n_estimators=1000,
            max_depth=20,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "stack_hgb_et": StackingClassifier(
            estimators=[
                (
                    "hgb",
                    HistGradientBoostingClassifier(
                        max_depth=10,
                        learning_rate=0.05,
                        max_iter=500,
                        min_samples_leaf=25,
                        random_state=seed,
                    ),
                ),
                (
                    "et",
                    ExtraTreesClassifier(
                        n_estimators=600,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ],
            final_estimator=LogisticRegression(max_iter=4000, class_weight="balanced"),
            stack_method="predict_proba",
            n_jobs=-1,
            passthrough=False,
        ),
        "logreg_scaled": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=7000,
                class_weight="balanced",
                random_state=seed,
                solver="lbfgs",
            ),
        ),
    }

    if include_slow:
        models["knn_scaled"] = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=31, weights="distance", p=2),
        )

    # Optional external boosters (only if installed)
    try:
        from xgboost import XGBClassifier  # type: ignore

        models["xgboost"] = XGBClassifier(
            n_estimators=1200,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier  # type: ignore

        models["catboost"] = CatBoostClassifier(
            iterations=1200,
            depth=8,
            learning_rate=0.03,
            loss_function="Logloss",
            verbose=False,
            random_seed=seed,
        )
    except Exception:
        pass

    return models


def main() -> None:
    args = parse_args()
    thresholds = parse_float_list(args.thresholds)
    if not thresholds:
        raise ValueError("Threshold list is empty")

    data_dir = args.data_dir.expanduser().resolve()
    weights_dir = args.weights_dir.expanduser().resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)

    log("Starting single-hand model search (v3)")
    log(f"Optimization metric: {args.optimize}")
    log(f"Threshold grid: {thresholds}")

    pattern_list = [p.strip() for p in args.patterns.split(",") if p.strip()]
    all_files = sorted({f for pat in pattern_list for f in data_dir.glob(pat)})
    if not all_files:
        raise FileNotFoundError(f"No files matched {args.patterns!r} in {data_dir}")

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

    models = candidate_models(args.seed, include_slow=args.include_slow)
    log(f"Candidate models available: {len(models)}")

    candidates: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None

    for idx, (name, model) in enumerate(models.items(), 1):
        log(f"[{idx}/{len(models)}] train model={name}")
        model.fit(x_train, y_train)

        val_probs = model.predict_proba(x_val)[:, 1]
        best_thr = None
        best_metrics = None

        for thr in thresholds:
            pred = predict_with_threshold(val_probs, thr)
            metrics = compute_metrics(y_val, pred)
            if best_metrics is None or score_for_objective(metrics, args.optimize) > score_for_objective(best_metrics, args.optimize):
                best_metrics = metrics
                best_thr = float(thr)

        assert best_metrics is not None and best_thr is not None
        row = {
            "name": name,
            "threshold": best_thr,
            "validation_metrics": best_metrics,
            "model": model,
        }
        candidates.append(row)

        log(
            f"  best threshold={best_thr:.3f} "
            f"acc={best_metrics['accuracy']:.4f} f1={best_metrics['f1']:.4f} "
            f"bal_acc={best_metrics['balanced_accuracy']:.4f}"
        )

        if best is None or score_for_objective(best_metrics, args.optimize) > score_for_objective(best["validation_metrics"], args.optimize):
            best = row

    assert best is not None
    log(f"Best validation model: {best['name']} @ threshold={best['threshold']:.3f}")

    x_trainval = np.vstack([x_train, x_val])
    y_trainval = np.concatenate([y_train, y_val])

    final_model = candidate_models(args.seed, include_slow=args.include_slow)[best["name"]]
    final_model.fit(x_trainval, y_trainval)

    test_probs = final_model.predict_proba(x_test)[:, 1]
    test_pred = predict_with_threshold(test_probs, float(best["threshold"]))
    test_metrics = compute_metrics(y_test, test_pred)

    model_path = weights_dir / f"{args.model_prefix}_model.pkl"
    scaler_path = weights_dir / f"{args.model_prefix}_scaler.pkl"
    metrics_path = weights_dir / f"{args.model_prefix}_metrics.json"

    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    with open(scaler_path, "wb") as f:
        # compatibility placeholder for loader expectations
        pickle.dump(None, f)

    candidates_sorted = sorted(
        [
            {
                "name": c["name"],
                "threshold": c["threshold"],
                "validation_metrics": c["validation_metrics"],
            }
            for c in candidates
        ],
        key=lambda c: score_for_objective(c["validation_metrics"], args.optimize),
        reverse=True,
    )

    report = {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "objective": args.optimize,
        "threshold_grid": thresholds,
        "candidate_count": len(models),
        "rows": {
            "train": int(len(y_train)),
            "validation": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "data_split": {
            "train_seed_range": [args.train_seed_min, args.train_seed_max],
            "validation_seed_range": [args.val_seed_min, args.val_seed_max],
            "test_seed_range": [args.test_seed_min, args.test_seed_max],
        },
        "best": {
            "name": best["name"],
            "threshold": best["threshold"],
            "validation_metrics": best["validation_metrics"],
            "test_metrics": test_metrics,
        },
        "candidates": candidates_sorted,
        "artifacts": {
            "model": str(model_path),
            "scaler": str(scaler_path),
            "metrics": str(metrics_path),
        },
    }

    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("=== SINGLE-HAND MODEL SEARCH COMPLETE ===")
    print(f"Objective:         {args.optimize}")
    print(f"Best model:        {best['name']}")
    print(f"Best threshold:    {best['threshold']:.3f}")
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
