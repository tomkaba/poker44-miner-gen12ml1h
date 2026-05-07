#!/usr/bin/env python3
"""Count how often each UID appears at rank #2 over historical blocks.

Uses stake-weighted consensus approximation from per-validator weights.
"""

from __future__ import annotations

import os
import time
from argparse import ArgumentParser
from collections import Counter

import bittensor as bt


NETWORK = os.getenv("BT_NETWORK", "finney")
NETUID = int(os.getenv("POKER44_NETUID", "126"))
BURN_UIDS = {0, 65535}
ARCHIVE_ENDPOINTS_ENV = os.getenv("BT_ARCHIVE_ENDPOINTS", "")


def parse_args():
    parser = ArgumentParser(description="Compute historical rank-2 UID frequency")
    parser.add_argument(
        "--network",
        default=NETWORK,
        help="Bittensor network name (default: env BT_NETWORK or finney)",
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=NETUID,
        help="Subnet netuid (default: env POKER44_NETUID or 126)",
    )
    parser.add_argument(
        "--archive-endpoint",
        action="append",
        default=None,
        help=(
            "Archive RPC endpoint (can be passed multiple times). "
            "If omitted, uses env BT_ARCHIVE_ENDPOINTS (comma-separated)."
        ),
    )
    parser.add_argument("--days", type=float, default=7.0, help="Lookback window in days")
    parser.add_argument(
        "--step-blocks",
        type=int,
        default=300,
        help="Sample every N blocks (default: 300 ~= 1h at 12s/block)",
    )
    parser.add_argument(
        "--blocks-per-day",
        type=int,
        default=7200,
        help="Approximate blocks per day (12s blocks ~= 7200)",
    )
    parser.add_argument(
        "--seconds-per-block",
        type=float,
        default=12.0,
        help="Seconds per block for wall-time estimate (default: 12)",
    )
    parser.add_argument(
        "--exclude-burn",
        action="store_true",
        help="Exclude burn UIDs (0, 65535) from ranking (default keeps burn)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many UIDs to print in rank-2 summary (default: 5)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N sampled points (default: 10)",
    )
    return parser.parse_args()


def consensus_for_block(sub: bt.Subtensor, netuid: int, block: int) -> dict[int, float]:
    raw_weights = sub.weights(netuid=netuid, block=block)
    if not raw_weights:
        return {}

    metagraph = sub.metagraph(netuid=netuid, block=block, lite=False)
    stakes = {val_uid: float(metagraph.S[val_uid]) for val_uid, _ in raw_weights}
    total_stake = sum(stakes.values()) or 1.0

    consensus: dict[int, float] = {}
    for val_uid, w_list in raw_weights:
        stake = stakes.get(val_uid, 0.0)
        if stake <= 0:
            continue
        total_raw = sum(w for _, w in w_list) or 1
        stake_coeff = stake / total_stake
        for miner_uid, raw_w in w_list:
            consensus[miner_uid] = consensus.get(miner_uid, 0.0) + (raw_w / total_raw) * stake_coeff
    return consensus


def sample_rank2(
    sub: bt.Subtensor,
    netuid: int,
    start_block: int,
    current_block: int,
    step_blocks: int,
    exclude_burn: bool,
    progress_every: int,
) -> tuple[Counter[int], int]:
    rank2_counter: Counter[int] = Counter()
    sampled = 0
    blocks = list(range(start_block, current_block + 1, step_blocks))
    total_points = len(blocks)
    started = time.time()
    print(f"Sampling {total_points} historical points...")

    for idx, block in enumerate(blocks, start=1):
        consensus = consensus_for_block(sub, netuid, block)
        if exclude_burn:
            for burn_uid in BURN_UIDS:
                consensus.pop(burn_uid, None)
        if not consensus:
            if idx % progress_every == 0 or idx == total_points:
                elapsed = max(0.001, time.time() - started)
                rate = idx / elapsed
                eta = (total_points - idx) / max(0.001, rate)
                print(
                    f"  progress {idx}/{total_points} ({idx / total_points * 100:.1f}%) "
                    f"block={block} sampled={sampled} eta~{eta / 60:.1f}m"
                )
            continue

        ranking = sorted(consensus.items(), key=lambda x: x[1], reverse=True)
        if len(ranking) < 2:
            if idx % progress_every == 0 or idx == total_points:
                elapsed = max(0.001, time.time() - started)
                rate = idx / elapsed
                eta = (total_points - idx) / max(0.001, rate)
                print(
                    f"  progress {idx}/{total_points} ({idx / total_points * 100:.1f}%) "
                    f"block={block} sampled={sampled} eta~{eta / 60:.1f}m"
                )
            continue
        sampled += 1
        rank2_uid = ranking[1][0]
        rank2_counter[rank2_uid] += 1

        if idx % progress_every == 0 or idx == total_points:
            elapsed = max(0.001, time.time() - started)
            rate = idx / elapsed
            eta = (total_points - idx) / max(0.001, rate)
            print(
                f"  progress {idx}/{total_points} ({idx / total_points * 100:.1f}%) "
                f"block={block} sampled={sampled} eta~{eta / 60:.1f}m"
            )
    return rank2_counter, sampled


def main() -> None:
    args = parse_args()
    archive_endpoints = args.archive_endpoint or [
        x.strip() for x in ARCHIVE_ENDPOINTS_ENV.split(",") if x.strip()
    ]
    sub = bt.Subtensor(network=args.network, archive_endpoints=archive_endpoints or None)

    current_block = sub.get_current_block()
    lookback_blocks = int(args.days * args.blocks_per_day)
    start_block = max(0, current_block - lookback_blocks)

    rank2_counter: Counter[int] = Counter()
    sampled = 0
    effective_network = args.network

    print(f"Network={args.network} netuid={args.netuid}")
    if archive_endpoints:
        print("Archive endpoints: " + ", ".join(archive_endpoints))
    else:
        print("Archive endpoints: none")
    print(f"Current block={current_block}")
    print(f"Window: {start_block}..{current_block} (days={args.days}, step={args.step_blocks})")

    # Public finney endpoints are pruned; for long windows jump straight to archive.
    if args.network == "finney" and lookback_blocks > 300 and not archive_endpoints:
        print("Long window detected (>300 blocks). Using built-in archive network immediately...")
        effective_network = "archive"
        sub = bt.Subtensor(network="archive")

    try:
        rank2_counter, sampled = sample_rank2(
            sub=sub,
            netuid=args.netuid,
            start_block=start_block,
            current_block=current_block,
            step_blocks=args.step_blocks,
            exclude_burn=args.exclude_burn,
            progress_every=max(1, args.progress_every),
        )
    except Exception as exc:
        msg = str(exc)
        can_auto_fallback = args.network != "archive" and not archive_endpoints
        if "State discarded" in msg and can_auto_fallback:
            print("\nNode prunes old state on current endpoint. Retrying on built-in archive network...")
            effective_network = "archive"
            sub = bt.Subtensor(network="archive")
            rank2_counter, sampled = sample_rank2(
                sub=sub,
                netuid=args.netuid,
                start_block=start_block,
                current_block=current_block,
                step_blocks=args.step_blocks,
                exclude_burn=args.exclude_burn,
                progress_every=max(1, args.progress_every),
            )
        elif "State discarded" in msg:
            print("\nERROR: Node prunes old state (State discarded).")
            print("Use archive network or pass an archive endpoint.")
            print("Examples:")
            print("  python scripts/top2_history.py --network archive --days 7")
            print(
                "  python scripts/top2_history.py --days 7 --archive-endpoint wss://archive.chain.opentensor.ai:443"
            )
            return
        else:
            raise

    if sampled == 0:
        print("No blocks sampled (or no consensus data available in the selected range).")
        return

    if effective_network != args.network:
        print(f"Analysis executed on network={effective_network}.")

    print(f"\nSample points: {sampled}")
    print("Top UIDs by time at rank #2 (Yuma consensus):")
    print("UID   Hits(rank2)   Share   ~Blocks   ~Hours   ~Days")
    print("------------------------------------------------------")
    for uid, hits in rank2_counter.most_common(args.top_k):
        share = hits / sampled * 100.0
        approx_blocks = hits * args.step_blocks
        approx_seconds = approx_blocks * args.seconds_per_block
        approx_hours = approx_seconds / 3600.0
        approx_days = approx_hours / 24.0
        print(
            f"{uid:>4}   {hits:>10}   {share:>6.2f}%"
            f"   {approx_blocks:>7}   {approx_hours:>6.2f}   {approx_days:>6.2f}"
        )

    if args.exclude_burn:
        print("\nBurn UIDs were excluded from ranking.")
    else:
        print("\nBurn UIDs were included in ranking.")
    print(
        "Time and block columns are estimates based on sampling step; "
        "for exact block counts use --step-blocks 1."
    )


if __name__ == "__main__":
    main()
