#!/usr/bin/env python3
"""Generate validator-identical mixed chunks via TimedMixedDatasetProvider."""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List


def _ensure_bittensor_stub() -> None:
    if "bittensor" in sys.modules:
        return

    bt_module = types.ModuleType("bittensor")

    class _Logger:
        @staticmethod
        def _log(level: str, msg: str, *args: Any) -> None:
            text = msg % args if args else str(msg)
            print(f"[bt.{level}] {text}", flush=True)

        info = staticmethod(lambda msg, *a, **k: _Logger._log("info", msg, *a))
        debug = staticmethod(lambda msg, *a, **k: _Logger._log("debug", msg, *a))
        warning = staticmethod(lambda msg, *a, **k: _Logger._log("warn", msg, *a))
        error = staticmethod(lambda msg, *a, **k: _Logger._log("error", msg, *a))

    bt_module.logging = _Logger()
    sys.modules["bittensor"] = bt_module


_ensure_bittensor_stub()

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hands_generator.mixed_dataset_provider import (  # noqa: E402
    MixedDatasetConfig,
    TimedMixedDatasetProvider,
)
from poker44.validator.sanitization import sanitize_hand_for_miner  # noqa: E402
from scripts.analysis.analyze_mixed_chunks import analyze_chunk  # noqa: E402


def _load_env_path(var: str, *, required: bool = True) -> Path | None:
    value = os.getenv(var)
    if not value:
        if required:
            raise SystemExit(f"Environment variable {var} is required")
        return None
    return Path(value).expanduser().resolve()


def _build_cfg(args: argparse.Namespace) -> MixedDatasetConfig:
    human_json = _load_env_path("POKER44_HUMAN_JSON_PATH")
    refresh_seconds = int(os.getenv("POKER44_DATASET_REFRESH_SECONDS", str(12 * 60 * 60)))
    chunk_count = int(os.getenv("POKER44_CHUNK_COUNT", "40"))
    min_hands = int(os.getenv("POKER44_MIN_HANDS_PER_CHUNK", "60"))
    max_hands = int(os.getenv("POKER44_MAX_HANDS_PER_CHUNK", "120"))
    human_ratio = float(os.getenv("POKER44_HUMAN_RATIO", "0.5"))
    dataset_seed_env = os.getenv("POKER44_DATASET_SEED")
    dataset_seed = int(dataset_seed_env) if dataset_seed_env else None

    if args.chunk_count is not None:
        chunk_count = args.chunk_count
    if args.human_ratio is not None:
        human_ratio = args.human_ratio

    return MixedDatasetConfig(
        human_json_path=human_json,
        output_path=Path(args.cache_path).expanduser().resolve(),
        chunk_count=chunk_count,
        min_hands_per_chunk=min_hands,
        max_hands_per_chunk=max_hands,
        human_ratio=human_ratio,
        refresh_seconds=refresh_seconds,
        seed=dataset_seed,
    )


def _chunk_to_payload(batch) -> Dict[str, Any]:
    chunk_dicts: List[Dict[str, Any]] = []
    for hand in batch.hands:
        if isinstance(hand, dict):
            payload = hand
        else:
            payload = getattr(hand, "to_payload", None)
            if callable(payload):
                payload = payload()
            else:
                payload = getattr(hand, "__dict__", {})
        chunk_dicts.append(sanitize_hand_for_miner(dict(payload)))
    return {
        "hands": chunk_dicts,
        "is_bot": not batch.is_human,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate validator-identical mixed chunks")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "analysis" / "data" / "validator_like_chunks.json")
    parser.add_argument("--cache-path", type=Path, default=REPO_ROOT / "analysis" / "data" / "validator_provider_cache.json",
                        help="Where TimedMixedDatasetProvider should cache its dataset")
    parser.add_argument("--chunk-count", type=int, default=None, help="Override chunk count")
    parser.add_argument("--human-ratio", type=float, default=None, help="Override human ratio")
    parser.add_argument("--window-id", type=int, default=None, help="Force specific refresh window id")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of chunks saved")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[gen] Starting validator-like generation", flush=True)
    cfg = _build_cfg(args)
    print(
        "[gen] Config prepared "
        f"| chunk_count={cfg.chunk_count} "
        f"| human_ratio={cfg.human_ratio} "
        f"| hands_range=[{cfg.min_hands_per_chunk},{cfg.max_hands_per_chunk}] "
        f"| refresh_s={cfg.refresh_seconds} "
        f"| seed={cfg.seed} "
        f"| cache={cfg.output_path}",
        flush=True,
    )
    print("[gen] Initializing TimedMixedDatasetProvider...", flush=True)
    provider = TimedMixedDatasetProvider(cfg)
    print("[gen] Provider initialized", flush=True)

    if args.window_id is not None:
        print(f"[gen] Forcing refresh to window {args.window_id}", flush=True)
        provider.force_refresh(window_id=args.window_id)
    else:
        print("[gen] Refresh check: refresh_if_due()", flush=True)
        provider.refresh_if_due()

    stats = provider.stats
    print(f"[gen] Dataset window_id={stats.get('window_id')} chunk_count={stats.get('chunk_count')} hash={provider.dataset_hash[:12]}")

    limit = args.limit or cfg.chunk_count
    print(f"[gen] Fetching hand batches | limit={limit}", flush=True)
    batches = provider.fetch_hand_batch(limit=limit)
    if not batches:
        raise SystemExit("Provider returned no batches")
    print(f"[gen] Received batches: {len(batches)}", flush=True)

    labeled_chunks: List[Dict[str, Any]] = []
    bucket_counts = {"human": {0: 0, 1: 0, ">=2": 0}, "bot": {0: 0, 1: 0, ">=2": 0}}

    for idx, batch in enumerate(batches, 1):
        payload = _chunk_to_payload(batch)
        labeled_chunks.append(payload)
        stats_chunk = analyze_chunk(payload)
        fm = stats_chunk.get("filtered_multi_leave", 0)
        bucket = ">=2" if fm >= 2 else int(fm)
        label = "bot" if payload.get("is_bot") else "human"
        bucket_counts[label][bucket] += 1
        print(
            f"[chunk {idx}/{len(batches)}] label={label} hands={len(payload['hands'])} filtered={fm}",
            flush=True,
        )

    output_path = Path(args.output).expanduser().resolve()
    print(f"[gen] Writing output JSON to {output_path}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "source": "validator_like",
        "window_id": stats.get("window_id"),
        "dataset_hash": provider.dataset_hash,
        "stats": stats,
        "generated_at": stats.get("generated_at"),
        "labeled_chunks": labeled_chunks,
        "generator": {
            "chunk_count": cfg.chunk_count,
            "min_hands": cfg.min_hands_per_chunk,
            "max_hands": cfg.max_hands_per_chunk,
            "human_ratio": cfg.human_ratio,
            "refresh_seconds": cfg.refresh_seconds,
            "seed": cfg.seed,
            "window_id": stats.get("window_id"),
        },
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print("[gen] Saved validator-like dataset to", output_path)
    print(
        f"[gen] bucket summary | human: f0={bucket_counts['human'][0]} f1={bucket_counts['human'][1]} f>=2={bucket_counts['human']['>=2']} | "
        f"bot: f0={bucket_counts['bot'][0]} f1={bucket_counts['bot'][1]} f>=2={bucket_counts['bot']['>=2']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
