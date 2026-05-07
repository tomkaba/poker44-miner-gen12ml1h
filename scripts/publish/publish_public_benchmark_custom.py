#!/usr/bin/env python3
"""Build and optionally publish a public benchmark with selectable bot profile families."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hands_generator.public_benchmark_custom import (
    DEFAULT_HUMAN_JSON_PATH,
    DEFAULT_PUBLIC_BENCHMARK_PATH,
    CustomPublicBenchmarkConfig,
    available_bot_profiles,
    available_profile_presets,
    build_public_benchmark_custom,
    save_public_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-json-path", type=Path, default=DEFAULT_HUMAN_JSON_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_PUBLIC_BENCHMARK_PATH)
    parser.add_argument("--chunk-count", type=int, default=40)
    parser.add_argument("--min-hands-per-chunk", type=int, default=60)
    parser.add_argument("--max-hands-per-chunk", type=int, default=120)
    parser.add_argument("--human-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--validation-ratio", type=float, default=0.25)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--bot-profile-preset", type=str, default="default_mix")
    parser.add_argument(
        "--bot-profiles",
        type=str,
        default="",
        help="Comma-separated explicit bot profiles. Overrides --bot-profile-preset.",
    )
    parser.add_argument("--list-bot-profiles", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="poker44-miner-benchmarks")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--artifact-name", type=str, default="public-miner-benchmark")
    parser.add_argument("--artifact-type", type=str, default="dataset")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--skip-wandb", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _print_profile_help() -> None:
    print("available_bot_profiles=" + ", ".join(available_bot_profiles()))
    print("available_presets=" + ", ".join(sorted(available_profile_presets())))


def main() -> None:
    args = parse_args()
    if args.list_bot_profiles:
        _print_profile_help()
        return

    cfg = CustomPublicBenchmarkConfig(
        human_json_path=args.human_json_path,
        output_path=args.output_path,
        chunk_count=args.chunk_count,
        min_hands_per_chunk=args.min_hands_per_chunk,
        max_hands_per_chunk=args.max_hands_per_chunk,
        human_ratio=args.human_ratio,
        seed=args.seed,
        validation_ratio=args.validation_ratio,
        progress_every=args.progress_every,
        bot_profile_preset=args.bot_profile_preset,
        bot_profiles=args.bot_profiles,
    )
    payload, dataset_hash = build_public_benchmark_custom(cfg)
    save_public_benchmark(cfg.output_path, payload)
    print(f"saved={cfg.output_path}")
    print(f"dataset_hash={dataset_hash}")
    print(f"bot_profile_preset={payload['config']['bot_profile_preset']}")
    print("bot_profiles=" + ",".join(payload["config"]["bot_profiles"]))

    if args.verbose:
        stats = payload.get("stats", {})
        config = payload.get("config", {})
        print("[verbose] generation summary")
        print(
            f"[verbose] chunks={stats.get('chunk_count', 0)} "
            f"hands_total={stats.get('total_hands', 0)} "
            f"human_chunks={stats.get('human_chunks', 0)} "
            f"bot_chunks={stats.get('bot_chunks', 0)}"
        )
        print(
            f"[verbose] split train={stats.get('train_chunks', 0)} "
            f"validation={stats.get('validation_chunks', 0)}"
        )
        print(
            f"[verbose] hands_per_chunk_range="
            f"[{config.get('min_hands_per_chunk', '?')}, {config.get('max_hands_per_chunk', '?')}]"
        )
        print(
            f"[verbose] human_ratio={config.get('human_ratio', '?')} "
            f"seed={config.get('seed', '?')} "
            f"preset={config.get('bot_profile_preset', '?')}"
        )

    if args.skip_wandb:
        return

    import wandb

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB_QUIET", "true")

    init_kwargs = {
        "project": args.wandb_project,
        "job_type": "publish_public_benchmark_custom",
        "config": payload["config"],
        "notes": payload["description"],
        "settings": wandb.Settings(quiet=True),
    }
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity

    run = wandb.init(**init_kwargs)
    try:
        artifact = wandb.Artifact(
            name=args.artifact_name,
            type=args.artifact_type,
            description=payload["description"],
            metadata={
                "dataset_hash": dataset_hash,
                "source": payload["source"],
                **payload["stats"],
            },
        )
        artifact.add_file(str(cfg.output_path), name=cfg.output_path.name)
        run.log_artifact(artifact)
        run.log(
            {
                "public_benchmark/dataset_hash": dataset_hash,
                "public_benchmark/chunk_count": payload["stats"]["chunk_count"],
                "public_benchmark/train_chunks": payload["stats"]["train_chunks"],
                "public_benchmark/validation_chunks": payload["stats"]["validation_chunks"],
                "public_benchmark/shortcut_rule_accuracy": payload["stats"]["shortcut_rule_accuracy"],
                "public_benchmark/bot_profile_count": payload["stats"]["bot_profile_count"],
            }
        )
    finally:
        run.finish(quiet=True)


if __name__ == "__main__":
    main()