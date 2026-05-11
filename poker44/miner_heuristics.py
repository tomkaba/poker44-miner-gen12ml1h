"""ML1H-only scoring for Poker44 gen12 release (fixed Gen5 artifact)."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ML1H_MODEL_PATH = REPO_ROOT / "weights" / "ml_gen5_s123467_model.pkl"
ML1H_SCALER_PATH = REPO_ROOT / "weights" / "ml_gen5_s123467_scaler.pkl"
ML1H_THRESHOLD = 0.3

_ML1H_MODEL = None
_ML1H_SCALER = None
_ML1H_AVAILABLE = False
_ML1H_LOAD_ERROR: Optional[str] = None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _extract_ml_features_gen4(chunk: List[dict]) -> Optional[np.ndarray]:
    """Extract 16-dimensional feature vector used by Gen5 single-hand model."""
    if not chunk:
        return None

    hand = chunk[0]
    players = hand.get("players") or []
    actions = hand.get("actions") or []
    outcome = hand.get("outcome") or {}
    streets = hand.get("streets") or []
    metadata = hand.get("metadata") or {}

    hand_bb_val = _safe_float(metadata.get("bb") or 0.01) or 0.01
    max_seats = max(int(metadata.get("max_seats") or 6), 1)

    num_players = float(len(players))
    filled_ratio = num_players / float(max_seats)

    starting_stacks = [_safe_float(p.get("starting_stack")) for p in players]
    # Keep the same stack scaling as the source ml1h implementation.
    stack_factor = 15.0
    starting_stacks_scaled = [s * stack_factor for s in starting_stacks]
    stack_mean = float(np.mean(starting_stacks_scaled)) if starting_stacks_scaled else 0.0
    stack_std = float(np.std(starting_stacks_scaled)) if starting_stacks_scaled else 0.0
    stack_cv = stack_std / (stack_mean + 1e-9)

    action_types = [str(a.get("action_type") or "").lower() for a in actions]
    total_actions = float(len(action_types))

    def _cnt(name: str) -> float:
        return float(sum(1 for t in action_types if t == name))

    call_c = _cnt("call")
    check_c = _cnt("check")
    fold_c = _cnt("fold")
    raise_c = _cnt("raise")
    bet_c = _cnt("bet")

    meaningful = call_c + check_c + fold_c + raise_c + bet_c
    if meaningful > 0:
        call_r = call_c / meaningful
        check_r = check_c / meaningful
        fold_r = fold_c / meaningful
        raise_r = raise_c / meaningful
    else:
        call_r = check_r = fold_r = raise_r = 0.0

    agg_ratio = (raise_c + bet_c) / (call_c + check_c + 1.0)

    amounts = [_safe_float(a.get("amount")) for a in actions]
    amounts_pos = [a for a in amounts if a > 0]
    amount_mean_bb = (float(np.mean(amounts_pos)) / hand_bb_val) if amounts_pos else 0.0
    amount_max_bb = (float(np.max(amounts_pos)) / hand_bb_val) if amounts_pos else 0.0

    total_pot = _safe_float(outcome.get("total_pot"))
    total_pot_bb = total_pot / hand_bb_val
    showdown = 1.0 if bool(outcome.get("showdown")) else 0.0

    flop_seen = 1.0 if any(s.get("street") == "FLOP" for s in streets) else 0.0
    turn_seen = 1.0 if any(s.get("street") == "TURN" for s in streets) else 0.0
    river_seen = 1.0 if any(s.get("street") == "RIVER" for s in streets) else 0.0
    street_depth = flop_seen + turn_seen + river_seen

    return np.array(
        [
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
        ],
        dtype=np.float32,
    )


def _load_ml1h_model() -> bool:
    """Load fixed Gen5 single-hand model and scaler."""
    global _ML1H_MODEL, _ML1H_SCALER, _ML1H_AVAILABLE, _ML1H_LOAD_ERROR

    if _ML1H_AVAILABLE and _ML1H_MODEL is not None and _ML1H_SCALER is not None:
        return True
    if _ML1H_LOAD_ERROR is not None:
        return False

    try:
        with open(ML1H_MODEL_PATH, "rb") as f:
            _ML1H_MODEL = pickle.load(f)
        with open(ML1H_SCALER_PATH, "rb") as f:
            _ML1H_SCALER = pickle.load(f)
        _ML1H_AVAILABLE = True
        _ML1H_LOAD_ERROR = None
        return True
    except Exception as exc:
        _ML1H_MODEL = None
        _ML1H_SCALER = None
        _ML1H_AVAILABLE = False
        _ML1H_LOAD_ERROR = str(exc)
        return False


def _predict_single_hand_probability(hand: dict) -> Optional[float]:
    if not _load_ml1h_model():
        return None

    features = _extract_ml_features_gen4([hand])
    if features is None:
        return None

    try:
        batch = features.reshape(1, -1)
        batch = _ML1H_SCALER.transform(batch)
        prob = float(_ML1H_MODEL.predict_proba(batch)[0, 1])
        return _clamp01(prob)
    except Exception:
        return None


def score_chunk_ml1h_with_route(chunk: List[dict]) -> Tuple[float, str]:
    """Score every request via ml1h model only, with no fallback scorers."""
    if not chunk:
        return 0.5, "empty_chunk"

    if not _load_ml1h_model():
        return 0.5, "ml1h_model_unavailable"

    if len(chunk) == 1:
        raw = _predict_single_hand_probability(chunk[0])
        if raw is None:
            return 0.5, "ml1h_single_error"
        calibrated = _clamp01(raw - ML1H_THRESHOLD + 0.5)
        return round(calibrated, 6), "ml1h_single"

    probs: List[float] = []
    for hand in chunk:
        raw = _predict_single_hand_probability(hand)
        if raw is not None:
            probs.append(raw)

    if not probs:
        return 0.5, "ml1h_vote_error"

    calibrated = np.clip(np.asarray(probs, dtype=np.float32) - ML1H_THRESHOLD + 0.5, 0.0, 1.0)
    bot_flags = calibrated >= 0.5
    score = float(np.mean(bot_flags.astype(np.float32)))
    return round(_clamp01(score), 6), "ml1h_vote"


def score_chunk(chunk: List[dict]) -> float:
    score, _route = score_chunk_ml1h_with_route(chunk)
    return score


def get_chunk_scorer_startup_check(scorer: str) -> Dict[str, object]:
    scorer_norm = (scorer or "").strip().lower()
    info: Dict[str, object] = {
        "scorer": scorer_norm,
        "active": scorer_norm == "ml1h",
        "ok": True,
        "error": None,
        "details": {},
    }

    if scorer_norm != "ml1h":
        return info

    info["details"] = {
        "model_path": str(ML1H_MODEL_PATH),
        "model_exists": ML1H_MODEL_PATH.exists(),
        "scaler_path": str(ML1H_SCALER_PATH),
        "scaler_exists": ML1H_SCALER_PATH.exists(),
        "threshold": ML1H_THRESHOLD,
    }

    ok = _load_ml1h_model()
    info["ok"] = ok
    if not ok:
        info["error"] = _ML1H_LOAD_ERROR

    return info
