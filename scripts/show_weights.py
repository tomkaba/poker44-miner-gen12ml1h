#!/usr/bin/env python3
"""Print Poker44 weight distribution: Yuma-consensus (stake-weighted) or per-validator table."""
from __future__ import annotations

import os
from argparse import ArgumentParser

import bittensor as bt

NETWORK = os.getenv("BT_NETWORK", "finney")
NETUID = int(os.getenv("POKER44_NETUID", "126"))
MY_UIDS = {5,6,7,8,10,12,13,16,20,29,33,37,38,39,45,56,59,66,67,68,69,70,81,90,92,93,99,101,106,107,109,110,111,140,151,152,155,156,169,176,182,187,199,202,215,216,219,225,227,228,229,231,246,249,251,252}
BURN_UIDS = {0, 65535}
DEFAULT_TOP_N = int(os.getenv("TOP_N", "25"))


def parse_args():
    parser = ArgumentParser(description="Show bittensor weight distribution for Poker44")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help="How many rows to display (default: %(default)s or env TOP_N)")
    parser.add_argument("--per-validator", action="store_true",
                        help="Show per-validator weight table instead of consensus view")
    return parser.parse_args()


def note_for(uid: int) -> str:
    parts = []
    if uid in BURN_UIDS:
        parts.append("<burn>")
    if uid in MY_UIDS:
        parts.append("*")
    return " ".join(parts)


def compute_consensus(val_weights: dict, stakes: dict) -> dict:
    """Stake-weighted average of per-validator normalised weights (approximates Yuma Consensus)."""
    total_stake = sum(stakes.values()) or 1
    consensus: dict[int, float] = {}
    for val_uid, weights in val_weights.items():
        stake = stakes.get(val_uid, 0.0)
        if stake == 0:
            continue
        total_raw = sum(weights.values()) or 1
        for miner_uid, raw_w in weights.items():
            consensus[miner_uid] = consensus.get(miner_uid, 0.0) + (raw_w / total_raw) * (stake / total_stake)
    return consensus


def show_consensus(val_weights: dict, stakes: dict, top_n: int) -> None:
    consensus = compute_consensus(val_weights, stakes)
    rows = sorted(consensus.items(), key=lambda x: x[1], reverse=True)

    print(f"\nYuma Consensus (stake-weighted) — top {top_n}  (* = our miner)")
    print("Lp.   UID      Weight%    Note")
    print("-" * 40)
    for idx, (uid, w) in enumerate(rows[:top_n], start=1):
        print(f"{idx:>3}. {uid:>5}  {w * 100:8.4f}%  {note_for(uid)}")

    mine = [(uid, rank, w * 100) for rank, (uid, w) in enumerate(rows, 1) if uid in MY_UIDS]
    if mine:
        print("\nOur miners summary:")
        for uid, rank, pct in sorted(mine, key=lambda x: x[1]):
            print(f"  UID {uid:>4} → rank {rank:>3}, weight {pct:8.4f}%")
    else:
        print("\n(No tracked miners found in the current snapshot)")

    missing = MY_UIDS - {uid for uid, _, _ in mine}
    if missing:
        print("Missing UIDs:", ", ".join(str(u) for u in sorted(missing)))


def show_per_validator(val_weights: dict, stakes: dict, top_n: int) -> None:
    # Sort validators by stake descending
    val_uids = sorted(val_weights.keys(), key=lambda v: stakes.get(v, 0), reverse=True)
    total_stake = sum(stakes.values()) or 1

    # Normalise per-validator weights to %
    val_norm: dict[int, dict[int, float]] = {}
    for val_uid, weights in val_weights.items():
        total_raw = sum(weights.values()) or 1
        val_norm[val_uid] = {m: w / total_raw * 100.0 for m, w in weights.items()}

    consensus = compute_consensus(val_weights, stakes)
    miners_sorted = sorted(consensus.keys(), key=lambda u: consensus[u], reverse=True)

    # Build column headers: V{uid}({stake_k}k)
    col_w = 9  # chars per validator column
    col_headers = [f"V{v}({stakes[v]/1000:.0f}k)" for v in val_uids]
    col_headers_fmt = "  ".join(f"{h:>{col_w}}" for h in col_headers)

    stake_pcts = [f"({stakes[v]/total_stake*100:.1f}%)" for v in val_uids]
    stake_pcts_fmt = "  ".join(f"{s:>{col_w}}" for s in stake_pcts)

    header = f"{'UID':>5}  {'Cons%':>8}  {col_headers_fmt}  Note"
    print(f"\nPer-validator weights — top {top_n} by consensus  (* = our miner)")
    print(header)
    print(f"{'':>5}  {'':>8}  {stake_pcts_fmt}")
    print("-" * len(header))

    for uid in miners_sorted[:top_n]:
        cons_pct = consensus[uid] * 100
        per_val = "  ".join(
            f"{val_norm[v].get(uid, 0.0):>{col_w}.3f}" for v in val_uids
        )
        print(f"{uid:>5}  {cons_pct:>8.4f}  {per_val}  {note_for(uid)}")


def main() -> None:
    args = parse_args()

    print(f"Network: {NETWORK}  netuid: {NETUID}")
    print("Fetching metagraph and weights…")

    sub = bt.Subtensor(network=NETWORK)
    mg = bt.Metagraph(netuid=NETUID, network=NETWORK, lite=False, sync=True)
    raw_weights = sub.weights(netuid=NETUID)  # [(val_uid, [(miner_uid, raw_w), ...]), ...]

    if not raw_weights:
        print("No weights returned — is the subnet active?")
        return

    val_weights = {val_uid: dict(w_list) for val_uid, w_list in raw_weights}
    stakes = {uid: mg.S[uid].item() for uid in val_weights}
    total_stake = sum(stakes.values())

    print(f"Validators ({len(val_weights)}): " +
          ", ".join(f"UID {v} ({stakes[v]/1000:.1f}k TAO)" for v in sorted(stakes, key=stakes.get, reverse=True)))
    print(f"Total validator stake: {total_stake:,.0f} TAO")

    if args.per_validator:
        show_per_validator(val_weights, stakes, args.top_n)
    else:
        show_consensus(val_weights, stakes, args.top_n)


if __name__ == "__main__":
    main()
