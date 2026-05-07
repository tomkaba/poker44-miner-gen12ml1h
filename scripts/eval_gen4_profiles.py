#!/usr/bin/env python3
"""
Evaluate gen4 model on hand profiles
Tests model predictions on balanced, tight_*, loose_*, etc. playing styles
"""

import gzip
import json
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def safe_float(v):
    """Safely convert value to float"""
    try:
        return float(v) if v is not None else 0.0
    except (ValueError, TypeError):
        return 0.0


def _bb(hand):
    """Get big blind from hand"""
    metadata = hand.get("metadata") or {}
    return safe_float(metadata.get("bb")) or 0.01


def extract_single_hand_features(hand):
    """Extract 16 features from a single hand - matches train_gen4_model.py exactly"""
    
    players = hand.get("players") or []
    actions = hand.get("actions") or []
    outcome = hand.get("outcome") or {}
    streets = hand.get("streets") or []
    metadata = hand.get("metadata") or {}

    bb = _bb(hand)
    max_seats = int(metadata.get("max_seats") or 6)
    max_seats = max(max_seats, 1)

    # Player/table shape
    num_players = float(len(players))
    filled_ratio = num_players / float(max_seats)
    starting_stacks = [safe_float(p.get("starting_stack")) for p in players]
    stack_mean = float(np.mean(starting_stacks)) if starting_stacks else 0.0
    stack_std = float(np.std(starting_stacks)) if starting_stacks else 0.0
    stack_cv = stack_std / (stack_mean + 1e-9)

    # Action profile
    action_types = [str(a.get("action_type") or "").lower() for a in actions]
    total_actions = float(len(action_types))

    def cnt(name: str) -> float:
        return float(sum(1 for t in action_types if t == name))

    call_c = cnt("call")
    check_c = cnt("check")
    fold_c = cnt("fold")
    raise_c = cnt("raise")
    bet_c = cnt("bet")
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

    amounts = [safe_float(a.get("amount")) for a in actions]
    amounts_pos = [a for a in amounts if a > 0]
    amount_mean_bb = (float(np.mean(amounts_pos)) / bb) if amounts_pos else 0.0
    amount_max_bb = (float(np.max(amounts_pos)) / bb) if amounts_pos else 0.0
    amount_std_bb = (float(np.std(amounts_pos)) / bb) if len(amounts_pos) > 1 else 0.0

    # Outcome / board profile
    total_pot = safe_float(outcome.get("total_pot"))
    total_pot_bb = total_pot / bb
    showdown = 1.0 if bool(outcome.get("showdown")) else 0.0
    payouts = outcome.get("payouts") or {}
    winner_count = float(sum(1 for _, v in payouts.items() if safe_float(v) > 0))
    winner_share = winner_count / (num_players + 1e-9)

    # Street count
    flop_seen = 1.0 if any(s.get("street") == "FLOP" for s in streets) else 0.0
    turn_seen = 1.0 if any(s.get("street") == "TURN" for s in streets) else 0.0
    river_seen = 1.0 if any(s.get("street") == "RIVER" for s in streets) else 0.0
    street_depth = flop_seen + turn_seen + river_seen

    # Combine into feature vector (16 features) - must match training exactly
    features = np.array([
        num_players,
        filled_ratio,
        stack_mean,
        stack_std,
        stack_cv,
        total_actions,
        call_r,
        check_r,
        fold_r,
        raise_r,
        agg_ratio,
        amount_mean_bb,
        amount_max_bb,
        total_pot_bb,
        showdown,
        street_depth,
    ], dtype=np.float32)

    return features


def extract_chunk_label(row):
    """Extract features and label from chunk - matches train_gen4_model.py exactly"""
    hands = row.get('hands') or []
    is_bot = row.get('is_bot', False)

    if not hands:
        return None, None

    features_list = []
    for hand in hands:
        try:
            feat = extract_single_hand_features(hand)
            if feat is not None:
                features_list.append(feat)
        except Exception:
            pass

    if not features_list:
        return None, None

    # Average features across all hands in chunk
    avg_features = np.mean(features_list, axis=0)
    label = int(is_bot)
    return avg_features, label


def load_profile(profile_path, max_samples=None):
    """Load chunk features and labels from gzipped file"""
    features_list = []
    labels_list = []
    
    with gzip.open(profile_path, 'rt') as f:
        data = json.load(f)
    
    chunks = data.get('labeled_chunks', [])
    for idx, row in enumerate(chunks):
        if max_samples and idx >= max_samples:
            break
        features, label = extract_chunk_label(row)
        if features is not None:
            features_list.append(features)
            labels_list.append(label)
    
    if not features_list:
        return None, None
    
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)
    return X, y


def evaluate_model(model, scaler, X, y, profile_name):
    """Evaluate model on profile data"""
    if X is None or len(X) == 0:
        print(f"  {profile_name}: SKIPPED (no data)")
        return None
    
    # Normalize
    X_scaled = scaler.transform(X)
    
    # Predict probabilities and labels
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Metrics
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_pred_proba)
    
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    result = {
        'profile': profile_name,
        'samples': len(X),
        'accuracy': acc,
        'f1': f1,
        'auc': auc,
        'recall': sensitivity,
        'specificity': specificity,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }
    
    print(f"  {profile_name:30} | Acc={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f} | n={len(X)}")
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate gen4 model on hand profiles')
    parser.add_argument('--model', default='weights/ml_gen4_model.pkl', help='Model path')
    parser.add_argument('--scaler', default='weights/ml_gen4_scaler.pkl', help='Scaler path')
    parser.add_argument('--profile-dir', 
                        default='data/public_benchmark_custom_1hand_5000_20260404_092644',
                        help='Profile directory')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples per profile')
    parser.add_argument('--output', help='Output JSON file')
    
    args = parser.parse_args()
    
    # Check files exist
    for fpath in [args.model, args.scaler]:
        if not Path(fpath).exists():
            print(f"ERROR: {fpath} not found")
            return 1
    
    profile_dir = Path(args.profile_dir)
    if not profile_dir.exists():
        print(f"ERROR: {profile_dir} not found")
        return 1
    
    # Load model and scaler
    print("Loading model and scaler...")
    with open(args.model, 'rb') as f:
        model = pickle.load(f)
    with open(args.scaler, 'rb') as f:
        scaler = pickle.load(f)
    
    # Find profile files (both *profile_* and *preset_* files)
    profile_files = sorted(profile_dir.glob('public_1hand_5000_*.json.gz'))
    if not profile_files:
        print(f"WARNING: No profile files found in {profile_dir}")
        print("Looking for pattern: public_1hand_5000_*.json.gz")
        return 1
    
    print(f"\nEvaluating on {len(profile_files)} profiles/presets...")
    print("=" * 100)
    
    results = []
    for pfile in profile_files:
        # Extract profile name from filename
        # Handles both: public_1hand_5000_profile_* and public_1hand_5000_preset_*
        profile_name = pfile.stem.replace('public_1hand_5000_profile_', '').replace('public_1hand_5000_preset_', '')
        
        # Load profile
        X, y = load_profile(pfile, max_samples=args.max_samples)
        
        # Evaluate
        result = evaluate_model(model, scaler, X, y, profile_name)
        if result:
            results.append(result)
    
    print("=" * 100)
    
    # Summary stats
    if results:
        avg_acc = np.mean([r['accuracy'] for r in results])
        avg_f1 = np.mean([r['f1'] for r in results])
        avg_auc = np.mean([r['auc'] for r in results])
        
        print(f"\nSUMMARY")
        print(f"  Profiles evaluated: {len(results)}")
        print(f"  Avg Accuracy: {avg_acc:.4f}")
        print(f"  Avg F1-Score: {avg_f1:.4f}")
        print(f"  Avg AUC: {avg_auc:.4f}")
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  Results saved to {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
