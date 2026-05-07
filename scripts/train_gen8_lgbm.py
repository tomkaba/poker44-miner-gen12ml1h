#!/usr/bin/env python3
"""
Train gen8 LightGBM model on benchmark features.

Uses the same 24 features as gen7heur1 (extracted by _gen7heur1_extract_features).
Train split:  2026-04-30 .. 2026-05-04
Holdout:      2026-05-05  (same as gen7heur1 baseline)

Output:
  models/benchmark_lgbm_model.pkl   (LGBMClassifier)
  models/benchmark_lgbm_scaler.pkl  (StandardScaler, for reference)
  models/benchmark_lgbm_profile.json (scorer profile for miner_heuristics)
"""

import json
import glob
import pickle
import sys
import os
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent.parent))
from poker44.miner_heuristics import _gen7heur1_extract_features

BENCHMARK_DIR = Path(__file__).parent.parent / "data" / "benchmark"
MODELS_DIR    = Path(__file__).parent.parent / "models"
HOLDOUT_DATE  = "2026-05-05"

# ── load data ──────────────────────────────────────────────────────────────────

def load_benchmark(files):
    X_rows, y_rows, dates = [], [], []
    for bf in sorted(files):
        date_str = Path(bf).stem.replace("benchmark_", "")
        with open(bf) as f:
            data = json.load(f)
        for outer in data.get("chunks", []):
            labels = outer.get("groundTruthLabels", [])
            inner_chunks = outer.get("chunks", [])
            for idx, hands in enumerate(inner_chunks):
                if idx >= len(labels):
                    continue
                label = labels[idx]
                if label not in ("bot", "human"):
                    continue
                feats = _gen7heur1_extract_features(hands)
                X_rows.append(list(feats.values()))
                y_rows.append(1 if label == "bot" else 0)
                dates.append(date_str)
    feat_names = list(_gen7heur1_extract_features(inner_chunks[0]).keys())
    return np.array(X_rows, dtype=np.float32), np.array(y_rows), dates, feat_names


all_files   = sorted(glob.glob(str(BENCHMARK_DIR / "benchmark_*.json")))
train_files = [f for f in all_files if HOLDOUT_DATE not in f]
hold_files  = [f for f in all_files if HOLDOUT_DATE in f]

print(f"Train files ({len(train_files)}): {[Path(f).stem for f in train_files]}")
print(f"Holdout files ({len(hold_files)}): {[Path(f).stem for f in hold_files]}")

X_tr, y_tr, _, feat_names = load_benchmark(train_files)
X_ho, y_ho, _, _          = load_benchmark(hold_files)

print(f"\nTrain: {X_tr.shape[0]} chunks  ({y_tr.sum()} bot / {(1-y_tr).sum()} human)")
print(f"Hold:  {X_ho.shape[0]} chunks  ({y_ho.sum()} bot / {(1-y_ho).sum()} human)")
print(f"Features: {len(feat_names)}")

# ── remove zero-variance features ─────────────────────────────────────────────

var = X_tr.var(axis=0)
keep_mask = var > 1e-9
removed = [feat_names[i] for i, k in enumerate(keep_mask) if not k]
if removed:
    print(f"\nRemoving zero-variance features: {removed}")
X_tr = X_tr[:, keep_mask]
X_ho = X_ho[:, keep_mask]
active_features = [f for f, k in zip(feat_names, keep_mask) if k]
print(f"Active features: {len(active_features)}")

# ── train LightGBM ─────────────────────────────────────────────────────────────

params = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

model = lgb.LGBMClassifier(**params)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_ho, y_ho)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=50)],
)

# ── evaluate ───────────────────────────────────────────────────────────────────

def evaluate(name, X, y, model):
    pred_proba = model.predict_proba(X)[:, 1]
    pred_cls   = (pred_proba >= 0.5).astype(int)
    acc    = accuracy_score(y, pred_cls)
    auc    = roc_auc_score(y, pred_proba)
    brier  = brier_score_loss(y, pred_proba)
    print(f"  {name}: acc={acc:.4f}  auc={auc:.4f}  brier={brier:.4f}")
    return pred_proba

print("\n── Evaluation ──")
evaluate("train  ", X_tr, y_tr, model)
ho_proba = evaluate("holdout", X_ho, y_ho, model)

# ── feature importance ─────────────────────────────────────────────────────────

importances = model.feature_importances_
fi_sorted = sorted(zip(active_features, importances), key=lambda x: -x[1])
print("\n── Feature importances (gain) ──")
for fname, imp in fi_sorted[:15]:
    bar = "█" * int(imp / max(importances) * 30)
    print(f"  {fname:<30} {imp:>6.0f}  {bar}")

# ── save model ─────────────────────────────────────────────────────────────────

MODELS_DIR.mkdir(exist_ok=True)
model_path = MODELS_DIR / "benchmark_lgbm_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"\nSaved model → {model_path}")

# Save metadata for miner scorer
profile = {
    "version": "gen8lgbm_v1",
    "feature_names": active_features,
    "removed_features": removed,
    "all_feature_names": feat_names,
    "keep_mask": keep_mask.tolist(),
    "n_estimators_used": model.best_iteration_ if model.best_iteration_ else params["n_estimators"],
    "train_files": [Path(f).stem for f in train_files],
    "holdout_file": [Path(f).stem for f in hold_files],
    "score_logic": {
        "chunk_size_min": 40,
        "chunk_size_max": 80,
    }
}
profile_path = MODELS_DIR / "benchmark_lgbm_profile.json"
with open(profile_path, "w") as f:
    json.dump(profile, f, indent=2)
print(f"Saved profile → {profile_path}")
print("\nDone.")
