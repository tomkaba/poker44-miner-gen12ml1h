#!/usr/bin/env python3
"""
Prepare chunk-level benchmark features and a deployable heuristic profile.

Outputs:
1) CSV with one row per scoring chunk (inner chunk)
2) JSON heuristic profile (feature stats + weights)
3) JSON summary with train/holdout metrics

Default split:
- train: all days except the latest day
- holdout: latest day
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple

EPS = 1e-9


@dataclass
class ChunkRow:
    source_date: str
    outer_chunk_id: str
    inner_index: int
    label: int
    hand_count: int
    features: Dict[str, float]


def _safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _safe_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(pstdev(values))


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _entropy_from_counter(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counter.values():
        p = c / total
        if p > 0:
            ent -= p * math.log(p + EPS)
    return ent


def _extract_chunk_features(hands: List[dict]) -> Dict[str, float]:
    hand_count = len(hands)

    action_counter = Counter()
    street_counter = Counter()
    actor_counter = Counter()

    actions_per_hand: List[float] = []
    raise_bb_values: List[float] = []
    bet_bb_values: List[float] = []
    pot_values: List[float] = []
    player_counts: List[float] = []
    street_depths: List[float] = []

    showdown_count = 0

    for hand in hands:
        actions = hand.get("actions") or []
        outcome = hand.get("outcome") or {}
        players = hand.get("players") or []
        streets = hand.get("streets") or []

        actions_per_hand.append(float(len(actions)))
        player_counts.append(float(len(players)))
        street_depths.append(float(len(streets)))
        pot_values.append(float(outcome.get("total_pot") or 0.0))

        if bool(outcome.get("showdown")):
            showdown_count += 1

        for action in actions:
            action_type = str(action.get("action_type") or "other")
            if action_type == "other":
                continue  # skip non-standard actions absent from production bots
            street = str(action.get("street") or "unknown")
            actor = str(action.get("actor_seat") or "?")

            action_counter[action_type] += 1
            street_counter[street] += 1
            actor_counter[actor] += 1

            bb_size = float(action.get("normalized_amount_bb") or 0.0)
            if action_type == "raise":
                raise_bb_values.append(bb_size)
            elif action_type == "bet":
                bet_bb_values.append(bb_size)

    total_actions = sum(action_counter.values())
    denom = max(1, total_actions)

    fold_ratio = action_counter.get("fold", 0) / denom
    call_ratio = action_counter.get("call", 0) / denom
    check_ratio = action_counter.get("check", 0) / denom
    raise_ratio = action_counter.get("raise", 0) / denom
    bet_ratio = action_counter.get("bet", 0) / denom

    aggression = (action_counter.get("raise", 0) + action_counter.get("bet", 0)) / max(
        1, action_counter.get("call", 0) + action_counter.get("check", 0)
    )

    preflop_ratio = street_counter.get("preflop", 0) / denom
    flop_ratio = street_counter.get("flop", 0) / denom
    turn_ratio = street_counter.get("turn", 0) / denom
    river_ratio = street_counter.get("river", 0) / denom

    features = {
        "chunk_size": float(hand_count),
        "actions_total": float(total_actions),
        "actions_per_hand_mean": _safe_mean(actions_per_hand),
        "actions_per_hand_std": _safe_std(actions_per_hand),
        "fold_ratio": fold_ratio,
        "call_ratio": call_ratio,
        "check_ratio": check_ratio,
        "raise_ratio": raise_ratio,
        "bet_ratio": bet_ratio,
        "aggression_ratio": aggression,
        "showdown_rate": showdown_count / max(1, hand_count),
        "street_depth_mean": _safe_mean(street_depths),
        "street_depth_std": _safe_std(street_depths),
        "players_mean": _safe_mean(player_counts),
        "players_std": _safe_std(player_counts),
        "pot_mean": _safe_mean(pot_values),
        "pot_std": _safe_std(pot_values),
        "raise_bb_mean": _safe_mean(raise_bb_values),
        "raise_bb_std": _safe_std(raise_bb_values),
        "bet_bb_mean": _safe_mean(bet_bb_values),
        "bet_bb_std": _safe_std(bet_bb_values),
        "action_entropy": _entropy_from_counter(action_counter),
        "street_entropy": _entropy_from_counter(street_counter),
        "actor_entropy": _entropy_from_counter(actor_counter),
    }
    return features


def load_rows(benchmark_dir: Path) -> List[ChunkRow]:
    files = sorted(glob.glob(str(benchmark_dir / "benchmark_*.json")))
    rows: List[ChunkRow] = []

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)

        source_date = str(payload.get("sourceDate") or "")
        for outer in payload.get("chunks", []):
            outer_chunk_id = str(outer.get("chunkId") or "")
            chunk_list = outer.get("chunks") or []
            labels = outer.get("groundTruth") or []

            if len(chunk_list) != len(labels):
                continue

            for idx, (inner_chunk, label) in enumerate(zip(chunk_list, labels)):
                features = _extract_chunk_features(inner_chunk)
                rows.append(
                    ChunkRow(
                        source_date=source_date,
                        outer_chunk_id=outer_chunk_id,
                        inner_index=idx,
                        label=int(label),
                        hand_count=len(inner_chunk),
                        features=features,
                    )
                )

    return rows


def _feature_names(rows: List[ChunkRow]) -> List[str]:
    if not rows:
        return []
    return sorted(rows[0].features.keys())


def _class_stats(rows: List[ChunkRow], feat_names: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {"human": {}, "bot": {}}

    human_rows = [r for r in rows if r.label == 0]
    bot_rows = [r for r in rows if r.label == 1]

    for name, subset in (("human", human_rows), ("bot", bot_rows)):
        for feat in feat_names:
            vals = [r.features[feat] for r in subset]
            out[name][feat] = {
                "mean": _safe_mean(vals),
                "std": _safe_std(vals),
            }

    return out


def build_profile(train_rows: List[ChunkRow], feat_names: List[str]) -> Dict[str, object]:
    stats = _class_stats(train_rows, feat_names)

    raw_weights: Dict[str, float] = {}
    for feat in feat_names:
        mu_h = stats["human"][feat]["mean"]
        mu_b = stats["bot"][feat]["mean"]
        sd_h = stats["human"][feat]["std"]
        sd_b = stats["bot"][feat]["std"]

        pooled = math.sqrt((sd_h * sd_h + sd_b * sd_b) / 2.0) + EPS
        sep = (mu_b - mu_h) / pooled
        raw_weights[feat] = sep

    # Normalize weights to stable scale.
    l1 = sum(abs(v) for v in raw_weights.values()) + EPS
    weights = {k: float(v / l1) for k, v in raw_weights.items()}

    return {
        "version": 1,
        "feature_names": feat_names,
        "weights": weights,
        "class_stats": stats,
        "score_logic": {
            "type": "linear_sigmoid_with_chunk_confidence",
            "chunk_confidence_min": 0.65,
            "chunk_confidence_max": 1.0,
            "chunk_size_min": 40,
            "chunk_size_max": 80,
        },
    }


def score_with_profile(features: Dict[str, float], profile: Dict[str, object]) -> float:
    weights = profile["weights"]
    stats = profile["class_stats"]

    raw = 0.0
    for feat in profile["feature_names"]:
        w = float(weights[feat])
        mu_h = float(stats["human"][feat]["mean"])
        mu_b = float(stats["bot"][feat]["mean"])
        sd_h = float(stats["human"][feat]["std"])
        sd_b = float(stats["bot"][feat]["std"])

        midpoint = 0.5 * (mu_h + mu_b)
        pooled = math.sqrt((sd_h * sd_h + sd_b * sd_b) / 2.0) + EPS
        z = (float(features[feat]) - midpoint) / pooled
        raw += w * z

    risk = _sigmoid(raw)

    # Shrink confidence for smaller chunks.
    cmin = float(profile["score_logic"]["chunk_confidence_min"])
    cmax = float(profile["score_logic"]["chunk_confidence_max"])
    smin = float(profile["score_logic"]["chunk_size_min"])
    smax = float(profile["score_logic"]["chunk_size_max"])
    size = float(features.get("chunk_size", smin))
    alpha = _clamp01((size - smin) / max(EPS, smax - smin))
    confidence = cmin + (cmax - cmin) * alpha

    return 0.5 + (risk - 0.5) * confidence


def evaluate(rows: List[ChunkRow], profile: Dict[str, object]) -> Dict[str, float]:
    if not rows:
        return {
            "n": 0,
            "accuracy@0.5": 0.0,
            "brier": 0.0,
        }

    probs: List[float] = []
    labels: List[int] = []
    correct = 0
    brier = 0.0

    for row in rows:
        p = score_with_profile(row.features, profile)
        y = row.label
        pred = 1 if p >= 0.5 else 0
        if pred == y:
            correct += 1
        brier += (p - y) ** 2
        probs.append(p)
        labels.append(y)

    return {
        "n": len(rows),
        "accuracy@0.5": correct / len(rows),
        "brier": brier / len(rows),
        "mean_score": _safe_mean(probs),
        "mean_label": _safe_mean([float(x) for x in labels]),
    }


def write_csv(rows: List[ChunkRow], profile: Dict[str, object], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    feat_names = profile["feature_names"]
    fieldnames = [
        "source_date",
        "outer_chunk_id",
        "inner_index",
        "label",
        "heuristic_score",
    ] + feat_names

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            line = {
                "source_date": row.source_date,
                "outer_chunk_id": row.outer_chunk_id,
                "inner_index": row.inner_index,
                "label": row.label,
                "heuristic_score": round(score_with_profile(row.features, profile), 6),
            }
            for feat in feat_names:
                line[feat] = row.features[feat]
            writer.writerow(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare benchmark chunk features + heuristic profile")
    parser.add_argument(
        "--benchmark-dir",
        default="data/benchmark",
        help="Directory with benchmark_YYYY-MM-DD.json files",
    )
    parser.add_argument(
        "--output-csv",
        default="analysis/benchmark_chunk_features.csv",
        help="Output CSV path with chunk-level features",
    )
    parser.add_argument(
        "--output-profile",
        default="models/benchmark_heuristic_profile.json",
        help="Output heuristic profile JSON path",
    )
    parser.add_argument(
        "--output-summary",
        default="analysis/benchmark_feature_summary.json",
        help="Output summary JSON path",
    )
    parser.add_argument(
        "--holdout-days",
        type=int,
        default=1,
        help="Number of latest sourceDate values reserved as holdout",
    )
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    out_csv = Path(args.output_csv)
    out_profile = Path(args.output_profile)
    out_summary = Path(args.output_summary)

    rows = load_rows(benchmark_dir)
    if not rows:
        raise SystemExit("No benchmark rows found. Check --benchmark-dir")

    dates = sorted({r.source_date for r in rows})
    holdout_days = max(1, int(args.holdout_days))
    holdout_set = set(dates[-holdout_days:])

    train_rows = [r for r in rows if r.source_date not in holdout_set]
    holdout_rows = [r for r in rows if r.source_date in holdout_set]

    feat_names = _feature_names(rows)
    profile = build_profile(train_rows, feat_names)

    train_metrics = evaluate(train_rows, profile)
    holdout_metrics = evaluate(holdout_rows, profile)
    all_metrics = evaluate(rows, profile)

    write_csv(rows, profile, out_csv)

    out_profile.parent.mkdir(parents=True, exist_ok=True)
    with open(out_profile, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "dates": dates,
        "holdout_dates": sorted(holdout_set),
        "row_count": len(rows),
        "train_count": len(train_rows),
        "holdout_count": len(holdout_rows),
        "feature_count": len(feat_names),
        "metrics": {
            "train": train_metrics,
            "holdout": holdout_metrics,
            "all": all_metrics,
        },
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Prepared rows: {len(rows)}")
    print(f"Train rows: {len(train_rows)} | Holdout rows: {len(holdout_rows)}")
    print(f"Features: {len(feat_names)}")
    print(f"CSV: {out_csv}")
    print(f"Profile: {out_profile}")
    print(f"Summary: {out_summary}")
    print(
        "Metrics | "
        f"train_acc={train_metrics['accuracy@0.5']:.4f}, "
        f"holdout_acc={holdout_metrics['accuracy@0.5']:.4f}, "
        f"holdout_brier={holdout_metrics['brier']:.4f}"
    )


if __name__ == "__main__":
    main()
