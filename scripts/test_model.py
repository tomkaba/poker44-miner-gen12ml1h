import gzip, json, pickle, numpy as np
from pathlib import Path
import poker44.miner_heuristics as mh

DATASETS = [
    Path("data/public_miner_benchmark_1hand_1000.json.gz"),
    Path("data/public_benchmark_1hand_1000_seed_1047.json.gz"),
    Path("data/public_benchmark_1hand_1000_seed_1048.json.gz"),
    Path("data/public_benchmark_1hand_1000_seed_1049.json.gz"),
    Path("data/public_benchmark_1hand_1000_seed_1050.json.gz"),
]

MODELS = {
    "active_v2_plus_set2": Path("weights/ml_single_hand_model.pkl"),
    "stage1_s123": Path("weights/ml_single_hand_stage1_s123_2026_04_04_model.pkl"),
    "stage2_s1234": Path("weights/ml_single_hand_stage2_s1234_2026_04_04_model.pkl"),
}

def load_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["labeled_chunks"] if isinstance(payload, dict) else payload

def eval_model(model_path, rows, thr=0.5):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    X, y = [], []
    for r in rows:
        hand = (r.get("hands") or [None])[0]
        if hand is None:
            continue
        X.append(mh._extract_ml_features_single_hand_v2([hand]))
        y.append(1 if r.get("is_bot") else 0)
    X = np.vstack(X)
    y = np.array(y, dtype=np.int32)
    p = model.predict_proba(X)[:, 1]
    pred = (p >= thr).astype(np.int32)
    tn = int(((y==0)&(pred==0)).sum()); fp = int(((y==0)&(pred==1)).sum())
    fn = int(((y==1)&(pred==0)).sum()); tp = int(((y==1)&(pred==1)).sum())
    acc = (tp+tn)/len(y)
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return acc, prec, rec, f1, tp, fp, fn, tn

for ds in DATASETS:
    rows = load_rows(ds)
    print(f"\n=== {ds.name} (n={len(rows)}) ===")
    for name, mp in MODELS.items():
        acc, prec, rec, f1, tp, fp, fn, tn = eval_model(mp, rows, thr=0.5)
        print(f"{name:18s} acc={acc:.4f} f1={f1:.4f} prec={prec:.4f} rec={rec:.4f} TP={tp} FP={fp} FN={fn} TN={tn}")
