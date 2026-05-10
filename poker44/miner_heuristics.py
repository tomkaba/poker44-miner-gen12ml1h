"""LightGBM scoring for Poker44 gen11lgbm release."""

from __future__ import annotations

import os
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_GEN11LGBM_MODEL = None
_GEN11LGBM_PROFILE = None
_GEN11LGBM_KEEP_MASK = None
_GEN11LGBM_LOAD_ERROR: Optional[str] = None


def _gen11lgbm_extract_features(chunk: List[dict]) -> Dict[str, float]:
    standard_actions = {"bet", "call", "check", "fold", "raise"}
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
            if atype not in standard_actions:
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
    eps = 1e-9

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
        return float(-sum((c / t) * math.log(c / t + eps) for c in counter.values()))

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


def _load_gen11lgbm() -> bool:
    """Lazy-load LightGBM model and profile. Returns True on success."""
    global _GEN11LGBM_MODEL, _GEN11LGBM_PROFILE, _GEN11LGBM_KEEP_MASK, _GEN11LGBM_LOAD_ERROR
    if _GEN11LGBM_MODEL is not None:
        return True
    if _GEN11LGBM_LOAD_ERROR is not None:
        return False

    import json as _json
    import pickle

    import numpy as np

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    profile_path = os.environ.get(
        "POKER44_GEN11LGBM_PROFILE",
        os.path.join(base, "models", "benchmark_lgbm_profile.json"),
    )
    model_path = os.environ.get(
        "POKER44_GEN11LGBM_MODEL",
        os.path.join(base, "models", "benchmark_lgbm_model.pkl"),
    )

    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            profile = _json.load(f)
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        _GEN11LGBM_PROFILE = profile
        _GEN11LGBM_MODEL = model
        _GEN11LGBM_KEEP_MASK = np.array(profile["keep_mask"], dtype=bool)
        return True
    except Exception as exc:
        _GEN11LGBM_LOAD_ERROR = str(exc)
        return False


def score_chunk_gen11lgbm(chunk: List[dict]) -> Tuple[float, str]:
    """Score a chunk using the gen11 LightGBM model."""
    import numpy as np
    import pandas as pd

    if not _load_gen11lgbm():
        return 0.5, "gen11lgbm_load_error"

    features = _gen11lgbm_extract_features(chunk)
    all_feature_names = _GEN11LGBM_PROFILE["all_feature_names"]
    all_vals = [features[k] for k in all_feature_names]
    row = np.array([all_vals], dtype=np.float32)[:, _GEN11LGBM_KEEP_MASK]
    selected_feature_names = [
        name for name, keep in zip(all_feature_names, _GEN11LGBM_KEEP_MASK) if keep
    ]
    row_df = pd.DataFrame(row, columns=selected_feature_names)

    try:
        score = float(_GEN11LGBM_MODEL.predict_proba(row_df)[0, 1])
    except Exception:
        return 0.5, "gen11lgbm_predict_error"

    return round(max(0.0, min(1.0, score)), 6), "gen11lgbm"


def score_chunk(chunk: List[dict]) -> float:
    score, _route = score_chunk_gen11lgbm(chunk)
    return score


def get_chunk_scorer_startup_check(scorer: str) -> Dict[str, object]:
    scorer_norm = (scorer or "").strip().lower()
    info: Dict[str, object] = {
        "scorer": scorer_norm,
        "active": scorer_norm == "gen11lgbm",
        "ok": True,
        "error": None,
        "details": {},
    }

    if scorer_norm != "gen11lgbm":
        return info

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    profile_path = Path(
        os.environ.get(
            "POKER44_GEN11LGBM_PROFILE",
            os.path.join(base, "models", "benchmark_lgbm_profile.json"),
        )
    )
    model_path = Path(
        os.environ.get(
            "POKER44_GEN11LGBM_MODEL",
            os.path.join(base, "models", "benchmark_lgbm_model.pkl"),
        )
    )
    info["details"] = {
        "profile_path": str(profile_path),
        "profile_exists": profile_path.exists(),
        "model_path": str(model_path),
        "model_exists": model_path.exists(),
    }

    ok = _load_gen11lgbm()
    info["ok"] = ok
    if not ok:
        info["error"] = _GEN11LGBM_LOAD_ERROR

    return info
