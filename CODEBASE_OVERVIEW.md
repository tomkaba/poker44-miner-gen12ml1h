# Poker44 Subnet Miner Codebase Overview

## Executive Summary

Poker44 is a **Bittensor subnet for detecting poker bots** using behavioral heuristics and machine learning. The miner architecture uses a **dual-mode scoring system** with **hybrid ML+heuristic evaluation** that achieves **99.67% accuracy on filtered=0 chunks** (vs 86.38% for pure heuristics alone).

The system:
- Classifies poker hand chunks as **human (score ≈ 0**) or **bot (score ≈ 1**) based on behavioral patterns
- Uses **two scoring modes**: Legacy (fully-sanitized data) and Modern (enhanced action details)
- Implements **three-bucket filtering strategy**: filtered_multi_leave >= 2, == 1, == 0
- Applies **ML model for filtered=0** (87% of modern chunks) with heuristic fallback
- Protects human player interests with hard false-positive penalties in reward function

---

## File Structure and Purposes

### Core Miner Architecture

| File | Purpose |
|------|---------|
| `neurons/miner.py` | **Main miner entry point**. Implements Bittensor synapse handler. Routes chunks through `score_chunk_legacy()` or `score_chunk_modern()` and returns risk scores [0-1] for each chunk. Exports predictions (≥0.5 = bot). |
| `poker44/base/miner.py` | Base class extending `BaseNeuron`. Handles axon setup, blacklist logic, priority scoring, request logging, and network synchronization. |
| `poker44/miner_heuristics.py` | **Central heuristics module** (~650 lines). Contains all scoring functions: `score_chunk_modern()`, `score_chunk_legacy()`, feature extractors, ML model loading, and heuristic weight definitions. |
| `poker44/score/scoring.py` | Reward calculation. Computes `reward(y_pred, y_true)` with human-safety penalty: zeros out reward if FPR ≥ 10%, otherwise applies `0.65 * AP_score + 0.35 * bot_recall * (1-FPR)²`. |

### Data & Configuration

| File | Purpose |
|------|---------|
| `poker44/constants.py` | Global constants. Currently minimal (just `SAMPLE_K = 256`). |
| `poker44/core/models.py` | Data models: `PlayerProfile`, `ActionEvent`, `StreetState`, `HandIntegrity`. Defines hand history structure. |
| `f0_w1.json`, `f0_w2.json`, `f1_w4.json` etc | **Weight configuration files** for `FILTERED0_WEIGHTS` and `FILTERED1_WEIGHTS` tuples (hyperparameters for heuristic scoring). |
| `LOCAL_FLAGS.md` | Documents custom environment flags: `SAVECHUNKS_LOG_EXPECTED`, `EXPORT_HH`. |

### Validator Infrastructure

| File | Purpose |
|------|---------|
| `neurons/validator.py` | Validator entry point. Initializes dataset provider, enters forward loop. |
| `poker44/validator/forward.py` | **Main evaluation loop** (~600 lines). Fetches mixed (human+bot) chunks, dispatches to all miners, computes windowed rewards, updates on-chain weights. |
| `poker44/validator/sanitization.py` | Hand sanitization for miner delivery (removes sensitive fields). |
| `poker44/validator/synapse.py` | Bittensor synapse definition: request carries `chunks` (list of hand dicts), response carries `risk_scores` and `predictions`. |

### Utilities & Analysis

| File | Purpose |
|------|---------|
| `poker44/base/utils/config.py` | Configuration parsing for miners/validators. |
| `poker44/base/validator.py` | Base validator class (network sync, weight management). |
| `hands_generator/data_generator.py` | Generates bot hand data for training/evaluation. |
| `hands_generator/mixed_dataset_provider.py` | Creates balanced chunks (human + bot). Used by validator. |

### Scripts & Tests (100+ files)

Key test/analysis scripts:
- `test_ml_hybrid.py`, `test_ml_vs_heur.py`: Compare ML vs pure heuristics
- `test_all_datasets.py`: Evaluate on multiple datasets
- `benchmark_final.py`, `benchmark_ml_fair.py`: Performance benchmarking
- `validate_ml_*.py`: ML model validation
- `scripts/analysis/evaluate_heuristics.py`: Full evaluation pipeline with metrics

---

## Heuristics System: All Metrics and Classifications

### 1. **Multi-Leave Statistics** (The 3-Bucket System)

**Purpose**: Identify bots that artificially multi-table (leave/rejoin many seats constantly). Computed first to **gate all subsequent scoring**.

**Core Metric**: `_multi_leave_stats(chunk)` returns:
- `filtered_multi_leave`: Count of hands with ≥2 **actual** player departures (excluding busted players)
- `raw_multi_leave`: Count of hands with ≥2 player departures (including busts)
- `multi_joinleave`: Count of hands with ≥2 joins OR ≥2 leaves
- `total_transitions`: Total hand-to-hand player transitions analyzed

**Bucket Logic** (from `score_chunk_modern()`):
```
if filtered_multi_leave >= 2:
    return 0.0  # Definite bot (strong multi-table pattern)
elif filtered_multi_leave == 1:
    return _score_filtered_one(...)  # Suspicious (use lighter weights)
else:      # filtered_multi_leave == 0
    return _score_filtered_zero(...)  # Normal (use main weights + ML if available)
```

**Why This Works**: Bots constantly switch tables to find profitable games; humans rarely do. Filtering out actual busts prevents false positives from normal cashout.

---

### 2. **Filtered=0 Heuristics** (Modern Chunks, ~87% of data)

**Scoring**: `_score_filtered_zero(chunk, multi_joinleave, raw_multi_leave)`

Starting score: **1.0 (human-biased)**. Penalties and bonuses accumulate.

#### A. **Street Depth Signal**
- **Metric**: `_avg_streets_per_hand(chunk)` = avg number of streets per hand
- **Interpretation**: Bots play more hands to completion; deep streets (flop, turn, river) increase for bots
- **Penalty** (from `FILTERED0_WEIGHTS`):
  ```
  if streets_avg > street_floor (≈0.759):
      penalty = (streets_avg - 0.759) / street_span (≈0.215)
      score -= 0.743 * penalty  # weight ≈ 0.743
  ```
- **Effect**: Deep-playing sessions penalized (suggest bot grinding)

#### B. **Filled Seat Ratio (Table Saturation)**
- **Metric**: `_avg_filled_seat_ratio(chunk)` = avg(players per hand / max table seats)
- **Interpretation**: Bots may play differently at short tables
- **Bonus** (if favorable):
  ```
  if filled_ratio < filled_threshold (≈0.828):
      bonus = min(0.206, (0.828 - filled_ratio) * 0.285)
      score += bonus  # Rewards tables with few players
  ```
- **Effect**: Sparse tables (short-handed games) favored

#### C. **Average Players Per Hand**
- **Metric**: `_avg_players_per_hand_chunk(chunk)` = avg player count
- **Interpretation**: Bots prefer consistent player counts; humans vary more
- **Bonus**:
  ```
  if avg_players < players_threshold (≈4.796):
      bonus = min(0.137, (4.796 - avg_players) * 0.528)
      score += bonus  # Rewards tables with fewer players
  ```
- **Effect**: Shorter tables preferred

#### D. **Action Ratios** (New metrics, action-based signals)

Computed via `_action_ratios_chunk(chunk)` across all hands in chunk:

| Action | Interpretation | Penalty Weight |
|--------|----------------|----------------|
| **Call Ratio** | High call ratio (>35%) suggests bot grinding (passive play) | 0.1 |
| **Fold Ratio** | Low fold ratio (<55%) suggests bot (folds less predictably) | 0.05 |
| **Raise/Call Ratio** (Aggressiveness) | Low aggression (<20%) suggests bot (tight, passive) | 0.1 |
| **Showdown Frequency** | Low showdown freq (<35%) suggests bot (tight play) | 0.06 |

```python
# Example calculation
aggressiveness = raise_ratio / (call_ratio + raise_ratio + eps)
if aggressiveness < 0.20:
    score -= 0.1 * clamp01((0.20 - aggressiveness) / 0.20)  # Penalize passive bots
```

#### E. **Street Variance** (Behavioral Consistency)
- **Metric**: `_street_variance(chunk)` = std dev of streets per hand
- **Interpretation**: Humans vary street depth; bots consistent
- **Bonus** (if high variance):
  ```
  if variance > 0.5:
      score += 0.04 * clamp01((variance - 0.5) / 1.0)
  ```
- **Effect**: Rewards variable play depth (human)

#### F. **Player Volatility** (Seat Churn)**
- **Metric**: `_player_volatility(chunk)` = std dev of player count
- **Interpretation**: Humans experience table churn; bots strive for consistency
- **Bonus** (if high):
  ```
  if volatility > 0.8:
      score += 0.03 * clamp01((volatility - 0.8) / 1.5)
  ```
- **Effect**: Rewards dynamic player turnover (human)

#### G. **ML Model (Filtered=0 Only)** ⭐
- **When Applied**: If `_ML_FILTERED0_MODEL` loaded (from `.ml_poc_cache.pkl`)
- **Accuracy**: **99.67%** on test data (vs 86.38% for heuristics)
- **Features** (16-dimensional):
  1. Call ratio, Check ratio, Fold ratio, Raise ratio, Bet ratio (5)
  2. Mean streets, Std dev streets, Min streets, Max streets (4)
  3. Mean players, Std dev players (2)
  4. Unique player count (1)
  5. Showdown frequency (1)
  6. Filled seat ratio (1)
  7. Mean pot, Std dev pot (2)
- **Output**: `model.predict_proba()` → probability of human (1.0 = human, 0.0 = bot)
- **Fallback**: If ML unavailable, uses pure heuristics

---

### 3. **Filtered=1 Heuristics** (One Multi-Leave Event)

**Scoring**: `_score_filtered_one(chunk, multi_joinleave, raw_multi_leave)`

Lighter weights (more conservative, assume one valid join/leave is benign):
- Same structure as filtered=0 but with different weight multipliers
- Uses `FILTERED1_WEIGHTS` instead of `FILTERED0_WEIGHTS`
- Simpler: just street depth, filled seat ratio, player count
- **No action-based signals or variance metrics** (stricter gatekeeping)

---

### 4. **Legacy (Fully-Sanitized) Scoring** (`score_chunk_legacy()`)

**Input**: Old payload format without modern action details

**Hand-Level Scoring** (`_score_hand_legacy(hand)`):

Each hand gets scored [0, 1] combining:
- Street depth (30% weight): deeper = more bot-like
- Showdown flag (22%): went to showdown = more human-like
- Call ratio (18%): high calls = bot grinding
- Check ratio (12%): checks = bot play
- Player count signal (8%): 6-player = human-normal
- Penalties for fold ratio (-18%), raise ratio (-10%)

**Chunk Aggregation**:
1. Average hand scores
2. Apply multi-leave signal (exponential decay): `exp(-0.85 * filtered_multi_leave)`
3. Bonus if chunk_size ≥ 100 and clean leaves (+15%)
4. Bonus if unique players ≤ 16 (+10%)
5. Final: `0.65 * multi_signal + 0.35 * avg_score + bonus`

---

## Scoring Architecture: How It All Fits Together

### Decision Flow Diagram

```
Input: chunk (list of hand dicts)
  ↓
[1] Detect payload type: legacy or modern?
  ├─ Legacy (old format) → score_chunk_legacy()
  └─ Modern (enhanced actions) → score_chunk_modern()
      ↓
[2] Compute multi_leave_stats()
  ├─ filtered_multi_leave >= 2 → return 0.0 (definite bot)
  ├─ filtered_multi_leave == 1 → _score_filtered_one()
  └─ filtered_multi_leave == 0 → _score_filtered_zero()
      ↓
[3a] Try ML model (if available)
  └─ success? → return ML score (99.67% accuracy)
      ↓
[3b] Fallback to heuristics
  ├─ Street depth penalty
  ├─ Filled seat bonus
  ├─ Player count bonus
  ├─ Action ratio penalties
  ├─ Variance/volatility bonuses
  └─ Return clamped [0, 1] score
      ↓
Output: risk_score (0 = human, 1 = bot, threshold 0.5)
```

### Weight Application Example (Filtered=0, Heuristic Path)

```python
FILTERED0_WEIGHTS = {
    "street_floor": 0.759,      # threshold before penalties apply
    "street_span": 0.215,       # normalize penalty range
    "street_weight": 0.743,     # strength of street penalty
    "filled_threshold": 0.828,  # below this = bonus  
    "filled_boost": 0.206,      # max bonus
    "filled_scale": 0.285,      # scale bonus by distance below threshold
    # ... etc
}

# Pseudocode:
score = 1.0  # Start human-biased
if streets_avg > 0.759:
    delta = (streets_avg - 0.759) / 0.215
    score -= 0.743 * min(1.0, delta)
if filled_ratio < 0.828:
    delta = (0.828 - filled_ratio) * 0.285
    score += min(0.206, delta)
return clamp01(score)
```

---

## Configuration & Constants

### Key Thresholds

| Constant | Value | Meaning |
|----------|-------|---------|
| `UNIQUE_PLAYER_HUMAN_THRESHOLD` | 60 | ≥60 unique IDs = definitely human (bots don't cycle players) |
| `STACK_THRESHOLD_BB_FRACTION` | 0.05 | 5% of BB = min stack to count as "in game" |
| `ZERO_STACK_EPS` | 1e-3 | Epsilon to avoid divide-by-zero |
| `LEGACY_PLAYER_PREFIX` | "seat_" | Field prefix in legacy payloads |
| `SAMPLE_K` | 256 | Target sample size (historical) |

### Scoring Thresholds

- **Bot Prediction**: score ≥ 0.5 (used as 0.5 exactly in tests)
- **Risk Score Range**: [0.0, 1.0] clamped
- **Rounding**: 6 decimal places in output

### Reward Computation (Validator Scoring)

```python
def reward(y_pred, y_true):
    # y_pred ∈ [0, 1], y_true ∈ {0,1} (0=human, 1=bot)
    preds = round(y_pred) → {0, 1}
    
    fpr = false_positives / negatives  # FPR on humans
    bot_recall = true_positives / positives
    ap_score = average_precision_score(y_true, y_pred)
    
    # Hard FPR penalty: zero reward if FPR ≥ 10%
    human_safety_penalty = max(0.0, 1.0 - fpr) ** 2 if fpr < 0.10 else 0.0
    
    base_score = 0.65 * ap_score + 0.35 * bot_recall
    final_reward = base_score * human_safety_penalty
    
    return {"fpr": fpr, "bot_recall": bot_recall, "reward": final_reward}
```

---

## Current Accuracy Metrics

### Reported Performance

| System | Filtered=0 | Filtered=1 | Filtered>=2 |
|--------|-----------|-----------|------------|
| **ML Hybrid** | 99.67% `*` | N/A | 100% (auto) |
| **Pure Heuristic** | 86.38% | N/A | 100% (auto) |
| **Improvement** | +13.29% | - | - |

`*` Conditional: only applies to filtered=0 (≈87% of modern chunks)

### Validation Logs (Sample from `valires.tail`)

```
uid=209|chunk=0|label=human|score=0.047466|pred=human  ✓
uid=209|chunk=1|label=human|score=0.039513|pred=human  ✓
uid=209|chunk=4|label=bot|score=0.777476|pred=bot      ✓
uid=209|chunk=6|label=bot|score=0.748079|pred=bot      ✓
uid=209|chunk=7|label=bot|score=0.886626|pred=bot      ✓
uid=209|chunk=28|label=human|score=0.000000|pred=human ✓
```

**Pattern**: Human chunks score very low (0.00-0.20), bot chunks score high (0.70-0.90), threshold at 0.5 separates clearly.

### Dataset Sizes

- `mixed_chunks_v0_sanitized_100.json`: 100 chunks
- `mixed_chunks_v0_sanitized_1000.json`: 1,000 chunks
- `mixed_chunks_v0_sanitized_2000_2.json`: 2,000 chunks
- `mixed_chunks_window0.json`: Historical eval window
- Public corpus: `hands_generator/human_hands/poker_hands_combined.json.gz` (human baseline)

---

## Data Flow Summary

### Miner Request-Response Cycle

1. **Validator sends** (`DetectionSynapse`):
   - `chunks`: List[List[dict]] — multiple chunks, each is a list of hand dicts
   - One chunk per batch from dataset provider

2. **Miner processes**:
   - Detects legacy vs modern format per chunk
   - Calls `score_chunk_legacy()` or `score_chunk_modern()`
   - Returns `risk_scores` (one float per chunk)
   - Computes `predictions` (boolean: `score >= 0.5`)

3. **Validator evaluates**:
   - Compares predictions to ground truth labels
   - Computes metrics: TP, FP, TN, FN → FPR, bot_recall
   - Applies reward formula with human-safety penalty
   - Updates on-chain weights (top performers)

4. **Miner logs** (if enabled):
   - Appends to `miner_<uid>.log` with full request/response details

---

## Key Implementation Details

### ML Model Lazy-Loading

```python
def _load_ml_model_filtered0():
    if _ML_MODEL_AVAILABLE or _ML_FILTERED0_MODEL is not None:
        return  # Already loaded
    
    # Load .ml_poc_cache.pkl (training data)
    cache_file = repo_root / ".ml_poc_cache.pkl"
    if not cache_file.exists():
        return  # Silent fail, will use heuristics
    
    # Train LogisticRegression on fly (StandardScaler + balanced class weights)
    # Once trained, cached in module-level globals
```

**Design Rationale**: Avoid startup overhead; only train if cache exists and filtered=0 data is encountered.

### Hand Sanitization for Miners

Validator calls `sanitize_hand_for_miner()` before dispatch:
- **Keeps**: action types, seats, amounts, stack info, streets
- **Removes**: sensitive fields, PII, exact player identities (anonymized via `player_uid`)
- **Purpose**: Prevent data leakage while preserving behavioral signals

### Multi-Leave Detection Algorithm

```python
def _compute_busted_players(hand):
    # Identify players who went all-in (final_stack ≤ threshold)
    # Used to filter out legitimate departures from suspicious multi-tabling
    return busted_uids

def _multi_leave_stats(chunk):
    # For each hand transition:
    #   prev_only = players in previous hand but not current
    #   actual_leaves = prev_only - busted_players
    #   if len(actual_leaves) >= 2 → increment filtered_multi_leave
    # Repeat for all hand pairs in chunk
```

---

## Testing & Validation Infrastructure

### Test Categories

1. **Accuracy Tests**: `test_all_datasets.py`, `test_simple_accuracy.py`
2. **ML vs Heuristic**: `test_ml_vs_heur.py`, `benchmark_ml_fair.py`
3. **Single Chunk Debug**: `test_ml_one_chunk.py`, `debug_score_execution.py`
4. **Weight Optimization**: `test_option_a.py`, `test_option_b_phases.py`
5. **Full Evaluation**: `scripts/analysis/evaluate_heuristics.py` (with precision/recall/F1)

### Integration Test: `test_ml_hybrid.py`

```bash
$ python test_ml_hybrid.py
✓ Import OK
✓ Loaded 1000 chunks from mixed_chunks_v0_sanitized_1000.json
✓ Chunk 0: score=0.0000 → pred=bot (label=BOT) ✓
✓ Chunk 1: score=0.9876 → pred=human (label=HUMAN) ✓
...
✓ Test complete
```

---

## Known Limitations & Future Work

1. **ML Model Dependency**: Performance critically depends on `.ml_poc_cache.pkl` availability
2. **Weight Tuning**: FILTERED0_WEIGHTS and FILTERED1_WEIGHTS are point estimates; no continuous optimization
3. **Action Features**: Legacy payloads (old format) lack action detail → lower accuracy
4. **Temporal Adaptation**: Static weights don't adapt to evolving bot strategies
5. **Decision Explainability**: No per-hand introspection; only chunk-level score

---

## Summary Table: All Heuristics at a Glance

| Metric | Function | Applies To | Weight | Interpretation |
|--------|----------|-----------|--------|-----------------|
| Multi-leave filtering | `_multi_leave_stats()` | All | 100% | Gates scoring logic; if ≥2 leaves → auto 0.0 (bot) |
| Streets per hand | `_avg_streets_per_hand()` | F=0, F=1 | 0.74 | Deep play = bot |
| Filled seat ratio | `_avg_filled_seat_ratio()` | F=0, F=1 | bonus | Short tables = human |
| Avg players | `_avg_players_per_hand_chunk()` | F=0, F=1 | bonus | Fewer players = human |
| Call ratio | `_action_ratios_chunk()` | F=0 | 0.10 | High calls = bot |
| Fold ratio | `_action_ratios_chunk()` | F=0 | 0.05 | Low folds = bot |
| Aggressiveness | `_action_ratios_chunk()` | F=0 | 0.10 | Low aggression = bot |
| Showdown freq | `_showdown_frequency()` | F=0 | 0.06 | Low showdown = bot |
| Street variance | `_street_variance()` | F=0 | 0.04 | High variance = human |
| Player volatility | `_player_volatility()` | F=0 | 0.03 | High volatility = human |
| **ML model** | `_score_filtered_zero_ml()` | F=0 | 100% | 99.67% accuracy (16 features) |
| **Legacy hand scores** | `_score_hand_legacy()` | Legacy | avg | Multi-signal street/action combo |

---

## Conclusion

Poker44's miner is a **sophisticated two-tier system**:

1. **Behavioral Gating** (multi-leave stats) eliminates obvious bots with hard rules
2. **ML + Heuristic Ensemble** (filtered=0) combines deep learning (99.67%) with interpretable heuristics for high-confidence detection
3. **Conservative Fallback** (filtered=1, legacy) uses simpler metrics for edge cases

This design **prioritizes human player protection** (reward zeroed if FPR ≥ 10%) while achieving strong bot recall. The validator loop rewards accurate, well-calibrated miners, driving continuous improvement in bot detection on-chain.

