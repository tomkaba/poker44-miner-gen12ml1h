#!/usr/bin/env python3
"""Analyze position changes over time from my-miners snapshot CSV files."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_GLOB = "analysis/my_miners_snapshots/my_miners_*.csv"


def parse_optional_int(value: str) -> Optional[int]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_optional_float(value: str) -> Optional[float]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def extract_timestamp_from_name(path: Path) -> Optional[str]:
    stem = path.stem
    # expected: my_miners_YYYYmmdd_HHMMSS
    parts = stem.split("my_miners_")
    if len(parts) != 2:
        return None
    stamp = parts[1]
    try:
        parsed = dt.datetime.strptime(stamp, "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    return parsed.replace(tzinfo=dt.UTC).isoformat()


def load_snapshots(glob_pattern: str) -> List[Dict[str, object]]:
    paths = sorted(Path().glob(glob_pattern))
    rows: List[Dict[str, object]] = []

    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                fetched_at_utc = (row.get("fetched_at_utc") or "").strip()
                if not fetched_at_utc:
                    inferred = extract_timestamp_from_name(path)
                    fetched_at_utc = inferred or ""

                rows.append(
                    {
                        "source_file": str(path),
                        "fetched_at_utc": fetched_at_utc,
                        "host": (row.get("host") or "").strip(),
                        "session": (row.get("session") or "").strip(),
                        "miner_uid": parse_optional_int(row.get("miner_uid", "")),
                        "hotkey": (row.get("hotkey") or "").strip(),
                        "rank": parse_optional_int(row.get("rank", "")),
                        "composite": parse_optional_float(row.get("composite", "")),
                        "reward": parse_optional_float(row.get("reward", "")),
                        "POKER44_SINGLE_HAND_MODEL_ALIAS": (
                            row.get("POKER44_SINGLE_HAND_MODEL_ALIAS") or ""
                        ).strip(),
                        "ML_MAX_HANDS": (row.get("ML_MAX_HANDS") or "").strip(),
                        "REMOVE_OTHER": (row.get("REMOVE_OTHER") or "").strip(),
                    }
                )
    return rows


def fmt_opt_int(value: Optional[int]) -> str:
    return str(value) if value is not None else "N/A"


def fmt_opt_float(value: Optional[float]) -> str:
    return f"{value:.3f}" if value is not None else "N/A"


def build_summary(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        uid = row.get("miner_uid")
        host = str(row.get("host") or "")
        if uid is None:
            continue
        grouped[(host, int(uid))].append(row)

    summaries: List[Dict[str, object]] = []
    for (host, uid), points in grouped.items():
        points.sort(key=lambda x: str(x.get("fetched_at_utc") or ""))
        first = points[0]
        last = points[-1]

        ranks = [p["rank"] for p in points if p.get("rank") is not None]
        first_rank = ranks[0] if ranks else None
        last_rank = ranks[-1] if ranks else None
        rank_delta = None
        if first_rank is not None and last_rank is not None:
            rank_delta = int(last_rank) - int(first_rank)

        composites = [p["composite"] for p in points if p.get("composite") is not None]
        first_comp = composites[0] if composites else None
        last_comp = composites[-1] if composites else None
        comp_delta = None
        if first_comp is not None and last_comp is not None:
            comp_delta = float(last_comp) - float(first_comp)

        rewards = [p["reward"] for p in points if p.get("reward") is not None]
        first_reward = rewards[0] if rewards else None
        last_reward = rewards[-1] if rewards else None
        reward_delta = None
        if first_reward is not None and last_reward is not None:
            reward_delta = float(last_reward) - float(first_reward)

        rank_series = " -> ".join(fmt_opt_int(p.get("rank")) for p in points)

        summaries.append(
            {
                "host": host,
                "miner_uid": uid,
                "hotkey": str(last.get("hotkey") or ""),
                "snapshots": len(points),
                "first_seen_utc": str(first.get("fetched_at_utc") or ""),
                "last_seen_utc": str(last.get("fetched_at_utc") or ""),
                "first_rank": first_rank,
                "last_rank": last_rank,
                "rank_delta": rank_delta,
                "best_rank": min(ranks) if ranks else None,
                "worst_rank": max(ranks) if ranks else None,
                "first_composite": first_comp,
                "last_composite": last_comp,
                "composite_delta": comp_delta,
                "first_reward": first_reward,
                "last_reward": last_reward,
                "reward_delta": reward_delta,
                "rank_series": rank_series,
                "last_model_alias": str(last.get("POKER44_SINGLE_HAND_MODEL_ALIAS") or ""),
                "last_ml_max_hands": str(last.get("ML_MAX_HANDS") or ""),
                "last_remove_other": str(last.get("REMOVE_OTHER") or ""),
            }
        )

    summaries.sort(key=lambda x: (x["host"], x["miner_uid"]))
    return summaries


def print_summary(summaries: List[Dict[str, object]]) -> None:
    if not summaries:
        print("No miner history found.")
        return

    print("My miners rank change summary:")
    print(
        "host   | uid | hotkey | snapshots | first_rank -> last_rank | delta | best | worst | "
        "last_model_alias | ML_MAX_HANDS | REMOVE_OTHER"
    )
    print(
        "-------+-----+--------+-----------+-------------------------+-------+------+------+"
        "-----------------+--------------+-------------"
    )
    for s in summaries:
        first_rank = fmt_opt_int(s["first_rank"])
        last_rank = fmt_opt_int(s["last_rank"])
        delta = fmt_opt_int(s["rank_delta"])
        last_model_alias = str(s.get("last_model_alias") or "") or "N/A"
        last_ml_max_hands = str(s.get("last_ml_max_hands") or "") or "N/A"
        last_remove_other = str(s.get("last_remove_other") or "") or "N/A"
        print(
            f"{str(s['host']):<6} | "
            f"{int(s['miner_uid']):<3} | "
            f"{str(s['hotkey']):<6} | "
            f"{int(s['snapshots']):<9} | "
            f"{first_rank:>5} -> {last_rank:<5} | "
            f"{delta:>5} | "
            f"{fmt_opt_int(s['best_rank']):>4} | "
            f"{fmt_opt_int(s['worst_rank']):>5} | "
            f"{last_model_alias:<15} | "
            f"{last_ml_max_hands:<12} | "
            f"{last_remove_other}"
        )


def maybe_write_summary_csv(summaries: List[Dict[str, object]], out_csv: Optional[Path]) -> None:
    if out_csv is None:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "host",
        "miner_uid",
        "hotkey",
        "snapshots",
        "first_seen_utc",
        "last_seen_utc",
        "first_rank",
        "last_rank",
        "rank_delta",
        "best_rank",
        "worst_rank",
        "first_composite",
        "last_composite",
        "composite_delta",
        "first_reward",
        "last_reward",
        "reward_delta",
        "rank_series",
        "last_model_alias",
        "last_ml_max_hands",
        "last_remove_other",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Summary CSV: {out_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze rank/composite/reward changes from my-miners snapshot CSV files."
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default=DEFAULT_GLOB,
        help=f"Glob for snapshot files (default: {DEFAULT_GLOB})",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional output CSV path for summary table.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = load_snapshots(args.input_glob)
    if not rows:
        print(f"No snapshot files matched: {args.input_glob}")
        return 1

    summaries = build_summary(rows)
    print_summary(summaries)
    maybe_write_summary_csv(summaries, args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
