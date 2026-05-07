#!/usr/bin/env python3
"""Verify models work correctly after sklearn re-saving (1.7.2 compat)."""

import json
from pathlib import Path
from collections import defaultdict

from poker44.miner_heuristics import (
    _score_filtered_zero,
    _score_filtered_one,
    _multi_leave_stats,
    _load_ml_model_filtered0,
    get_ml_runtime_stats,
)

DATA_DIR = Path.cwd() / "analysis/data"

# Pre-load ML
print("Loading ML model...")
_load_ml_model_filtered0()
stats = get_ml_runtime_stats()
print(f"  ✓ Model loaded: {stats['ml_model_loaded']}")
print(f"  ✓ Model available: {stats['ml_model_available']}")
print()

def score_chunk(chunk):
    """Score chunk using default (ML + heuristic) system."""
    if not chunk:
        return 0.5
    
    filtered_multi_leave, _, multi_joinleave, raw_multi_leave = _multi_leave_stats(chunk)
    
    if filtered_multi_leave >= 2:
        return 0.0  # Forced human
    
    if filtered_multi_leave == 1:
        return _score_filtered_one(chunk, multi_joinleave, raw_multi_leave)
    
    # filtered=0: ML + heuristic
    return _score_filtered_zero(chunk, multi_joinleave, raw_multi_leave)


# Test range: window >= 180
print("Testing validator_like_window >= 180 (36 datasets, 2880 chunks):")
print()

all_files = sorted(DATA_DIR.glob("validator_like_window_*.json"))
test_files = [f for f in all_files if int(f.stem.split("_")[-1]) >= 180]

print(f"Found {len(test_files)} datasets in range [180, 216]")
print()

total_correct = total_chunks = 0
f0_correct = f0_chunks = 0
errors_by_class = defaultdict(lambda: {"bot": 0, "human": 0})

for fpath in test_files[:5]:  # Just first 5 to avoid slowness
    try:
        payload = json.loads(fpath.read_text())
        chunks_data = payload if isinstance(payload, list) else payload.get("labeled_chunks", [])
    except:
        continue
    
    file_correct = file_chunks = 0
    file_f0_c = file_f0_chunks = 0
    
    for chunk_data in chunks_data:
        hands = chunk_data.get("hands") or []
        is_bot = chunk_data.get("is_bot", False)
        
        if not hands:
            continue
        
        # Score
        score = score_chunk(hands)
        pred = score < 0.5  # True = bot, False = human
        is_correct = (pred == is_bot)
        
        file_chunks += 1
        total_chunks += 1
        
        if is_correct:
            file_correct += 1
            total_correct += 1
        else:
            errors_by_class[fpath.name][("bot" if is_bot else "human")] += 1
        
        # Track filtered=0
        filtered_multi_leave, _, _, _ = _multi_leave_stats(hands)
        if filtered_multi_leave == 0:
            file_f0_chunks += 1
            f0_chunks += 1
            if is_correct:
                file_f0_c += 1
                f0_correct += 1
    
    if file_chunks > 0:
        acc = file_correct / file_chunks
        f0_acc = file_f0_c / file_f0_chunks if file_f0_chunks > 0 else 0.0
        print(f"  {fpath.name:<30} | Accuracy: {acc:.4f} ({file_correct}/{file_chunks}) | f0: {f0_acc:.4f} ({file_f0_c}/{file_f0_chunks})")

print()
print("=" * 100)
if total_chunks > 0:
    overall = total_correct / total_chunks
    f0_overall = f0_correct / f0_chunks if f0_chunks > 0 else 0.0
    
    print(f"AGGREGATE (sampled {total_chunks} chunks from {len(test_files)} datasets):")
    print(f"  Overall accuracy:    {overall:.4f} ({total_correct}/{total_chunks})")
    print(f"  Filtered=0 accuracy: {f0_overall:.4f} ({f0_correct}/{f0_chunks})")
    print()
    
    if overall >= 0.99:
        print("✓ EXCELLENT: Model is working correctly!")
    elif overall >= 0.93:
        print("✓ GOOD: Model performance is as expected")
    else:
        print("⚠️  WARNING: Model accuracy seems degraded")
    
    # Show error distribution
    print()
    print("Error distribution (first 5 files):")
    for fname in sorted(errors_by_class.keys())[:5]:
        errors = errors_by_class[fname]
        if errors["bot"] > 0 or errors["human"] > 0:
            print(f"  {fname}: bot_errors={errors['bot']}, human_errors={errors['human']}")

print()
stats = get_ml_runtime_stats()
print(f"ML Runtime stats:")
print(f"  Load attempts:  {stats['ml_load_attempts']}")
print(f"  Load successes: {stats['ml_load_successes']}")
print(f"  Model loaded:   {stats['ml_model_loaded']}")
print(f"  Model available: {stats['ml_model_available']}")
