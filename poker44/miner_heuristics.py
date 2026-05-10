"""Heuristic scoring for Poker44 gen10heur1 release."""

from __future__ import annotations

import math
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


_GEN10HEUR1_PROFILE: Optional[dict] = None
_GEN10HEUR1_STANDARD_ACTIONS: Set[str] = {"bet", "call", "check", "fold", "raise"}
_EPS_G10 = 1e-9


def _load_gen10heur1_profile() -> dict:
    global _GEN10HEUR1_PROFILE
    if _GEN10HEUR1_PROFILE is not None:
        return _GEN10HEUR1_PROFILE

    env_path = os.getenv("POKER44_GEN10HEUR1_PROFILE", "")
    if env_path:
        profile_path = Path(env_path)
    else:
        profile_path = Path(__file__).resolve().parents[1] / "models" / "benchmark_heuristic_profile.json"

    import json as _json

    with open(profile_path, "r", encoding="utf-8") as f:
        _GEN10HEUR1_PROFILE = _json.load(f)
    return _GEN10HEUR1_PROFILE


def _gen10heur1_extract_features(chunk: List[dict]) -> Dict[str, float]:
    action_counter: Counter = Counter()
    street_counter: Counter = Counter()
    actor_counter: Counter = Counter()

    actions_per_hand: List[float] = []
    raise_bb_values: List[float] = []
    bet_bb_values: List[float] = []
    pot_values: List[float] = []
    player_counts: List[float] = []
    street_depths: List[float] = []

    showdown_count = 0

    for hand in chunk:
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
            atype = str(action.get("action_type") or "other")
            if atype not in _GEN10HEUR1_STANDARD_ACTIONS:
                continue

            street = str(action.get("street") or "unknown")
            actor = str(action.get("actor_seat") or "?")

            action_counter[atype] += 1
            street_counter[street] += 1
            actor_counter[actor] += 1

            bb_size = float(action.get("normalized_amount_bb") or 0.0)
            if atype == "raise":
                raise_bb_values.append(bb_size)
            elif atype == "bet":
                bet_bb_values.append(bb_size)

    hand_count = len(chunk)
    total_actions = sum(action_counter.values())
    denom = max(1, total_actions)

    def _smean(vals: List[float]) -> float:
        return float(sum(vals) / len(vals)) if vals else 0.0

    def _sstd(vals: List[float]) -> float:
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        return float(math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)))

    def _entropy(counter: Counter) -> float:
        t = sum(counter.values())
        if t <= 0:
            return 0.0
        return float(-sum((c / t) * math.log(c / t + _EPS_G10) for c in counter.values()))

    return {
        "action_entropy": _entropy(action_counter),
        "actions_per_hand_mean": _smean(actions_per_hand),
        "actions_per_hand_std": _sstd(actions_per_hand),
        "actions_total": float(total_actions),
        "actor_entropy": _entropy(actor_counter),
        "aggression_ratio": (
            (action_counter.get("raise", 0) + action_counter.get("bet", 0))
            / max(1, action_counter.get("call", 0) + action_counter.get("check", 0))
        ),
        "bet_bb_mean": _smean(bet_bb_values),
        "bet_bb_std": _sstd(bet_bb_values),
        "bet_ratio": action_counter.get("bet", 0) / denom,
        "call_ratio": action_counter.get("call", 0) / denom,
        "check_ratio": action_counter.get("check", 0) / denom,
        "chunk_size": float(hand_count),
        "fold_ratio": action_counter.get("fold", 0) / denom,
        "players_mean": _smean(player_counts),
        "players_std": _sstd(player_counts),
        "pot_mean": _smean(pot_values),
        "pot_std": _sstd(pot_values),
        "raise_bb_mean": _smean(raise_bb_values),
        "raise_bb_std": _sstd(raise_bb_values),
        "raise_ratio": action_counter.get("raise", 0) / denom,
        "showdown_rate": showdown_count / max(1, hand_count),
        "street_depth_mean": _smean(street_depths),
        "street_depth_std": _sstd(street_depths),
        "street_entropy": _entropy(street_counter),
    }


def _sigmoid_g10(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def score_chunk_gen10heur1(chunk: List[dict]) -> Tuple[float, str]:
    if not chunk:
        return 0.5, "gen10heur1_empty"

    try:
        profile = _load_gen10heur1_profile()
    except Exception:
        return 0.5, "gen10heur1_profile_load_error"

    features = _gen10heur1_extract_features(chunk)
    weights = profile["weights"]
    stats = profile["class_stats"]

    raw = 0.0
    for feat in profile["feature_names"]:
        w = float(weights.get(feat, 0.0))
        if w == 0.0:
            continue
        mu_h = float(stats["human"][feat]["mean"])
        mu_b = float(stats["bot"][feat]["mean"])
        sd_h = float(stats["human"][feat]["std"])
        sd_b = float(stats["bot"][feat]["std"])
        midpoint = 0.5 * (mu_h + mu_b)
        pooled = math.sqrt((sd_h * sd_h + sd_b * sd_b) / 2.0) + _EPS_G10
        z = (float(features.get(feat, midpoint)) - midpoint) / pooled
        raw += w * z

    risk = _sigmoid_g10(raw)

    smin = float(profile["score_logic"].get("chunk_size_min", 40))
    smax = float(profile["score_logic"].get("chunk_size_max", 80))
    cmin = float(profile["score_logic"].get("chunk_confidence_min", 0.65))
    cmax = float(profile["score_logic"].get("chunk_confidence_max", 1.0))
    size = float(features.get("chunk_size", smin))
    alpha = max(0.0, min(1.0, (size - smin) / max(_EPS_G10, smax - smin)))
    confidence = cmin + (cmax - cmin) * alpha

    score = 0.5 + (risk - 0.5) * confidence
    return round(max(0.0, min(1.0, score)), 6), "gen10heur1"


def score_chunk(chunk: List[dict]) -> float:
    score, _route = score_chunk_gen10heur1(chunk)
    return score


def get_chunk_scorer_startup_check(scorer: str) -> Dict[str, object]:
    scorer_norm = (scorer or "").strip().lower()
    info: Dict[str, object] = {
        "scorer": scorer_norm,
        "active": scorer_norm == "gen10heur1",
        "ok": True,
        "error": None,
        "details": {},
    }

    if scorer_norm != "gen10heur1":
        return info

    env_path = os.getenv("POKER44_GEN10HEUR1_PROFILE", "")
    profile_path = Path(env_path) if env_path else Path(__file__).resolve().parents[1] / "models" / "benchmark_heuristic_profile.json"
    info["details"] = {
        "profile_path": str(profile_path),
        "profile_exists": profile_path.exists(),
    }

    try:
        _load_gen10heur1_profile()
    except Exception as exc:
        info["ok"] = False
        info["error"] = str(exc)

    return info
