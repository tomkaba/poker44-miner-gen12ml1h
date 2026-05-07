#!/usr/bin/env python3
"""Fetch and merge paginated ranking data from https://poker44.net/dashboard."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests


BASE_URL = "https://poker44.net/dashboard"
DEFAULT_SSH_HOSTS = "tk@cx1,tk@cx2"
REMOTE_PS_ENV_CMD = "ps -aux | grep 'SCREEN -dmS sn126' | grep -v grep"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    )
}


def build_url(page: int) -> str:
    return BASE_URL if page <= 1 else f"{BASE_URL}?page={page}"


def fetch_html(session: requests.Session, page: int, timeout: int) -> str:
    url = build_url(page)
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def fetch_html_with_retries(
    session: requests.Session,
    page: int,
    timeout: int,
    retries: int,
    retry_sleep_s: float,
) -> str:
    last_exc: Optional[Exception] = None
    attempts = max(1, retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            return fetch_html(session=session, page=page, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < attempts:
                time.sleep(retry_sleep_s)
    assert last_exc is not None
    raise last_exc


def clean_html_text(raw: str) -> str:
    text = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_float_or_none(text: str) -> Optional[float]:
    try:
        return float(text.replace(",", ""))
    except (TypeError, ValueError):
        return None


def parse_int_or_none(text: str) -> Optional[int]:
    try:
        return int(text.replace(",", ""))
    except (TypeError, ValueError):
        return None


def extract_total_miners(page_html: str) -> Optional[int]:
    match = re.search(
        r"Showing\s+\d+\s*-\s*\d+\s+of\s*([\d,]+)\s+miners",
        page_html,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def extract_max_page_from_pagination(page_html: str) -> Optional[int]:
    page_values = [
        int(v)
        for v in re.findall(r"/dashboard\?page=(\d+)", page_html, flags=re.IGNORECASE)
    ]
    if not page_values:
        return None
    return max(page_values)


def parse_rows(page_html: str, page: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    tr_blocks = re.findall(r"<tr[^>]*>(.*?)</tr>", page_html, flags=re.DOTALL | re.IGNORECASE)

    for tr in tr_blocks:
        td_blocks = re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.DOTALL | re.IGNORECASE)
        if len(td_blocks) != 7:
            continue

        rank_text = clean_html_text(td_blocks[0])
        uid_hotkey_text = clean_html_text(td_blocks[1])
        model_text = clean_html_text(td_blocks[2])
        composite_text = clean_html_text(td_blocks[3])
        reward_text = clean_html_text(td_blocks[4])
        samples_text = clean_html_text(td_blocks[5])
        compliance_text = clean_html_text(td_blocks[6])

        rank_match = re.search(r"#\s*(\d+)", rank_text)
        uid_match = re.search(r"UID\s*(\d+)", uid_hotkey_text, flags=re.IGNORECASE)
        if not rank_match or not uid_match:
            continue

        hotkey_match = re.search(r"(5[\w.]+)$", uid_hotkey_text)
        version_commit_match = re.search(r"(\d[\w.-]*)\s*(?:\u00b7\s*([0-9a-f]{7,40}))?", model_text)

        # Split model line from version/commit if possible.
        model_name = model_text
        model_version = ""
        model_commit = ""
        if version_commit_match:
            model_version = version_commit_match.group(1)
            model_commit = version_commit_match.group(2) or ""
            model_name = model_text[: version_commit_match.start()].strip()

        composite_value = parse_float_or_none(composite_text)
        composite_status = ""
        if composite_value is None and composite_text:
            composite_status = composite_text

        samples_value = parse_int_or_none(samples_text)

        row = {
            "page": page,
            "rank": int(rank_match.group(1)),
            "uid": int(uid_match.group(1)),
            "hotkey": hotkey_match.group(1) if hotkey_match else "",
            "model_name": model_name,
            "model_version": model_version,
            "model_commit": model_commit,
            "composite": composite_value,
            "composite_status": composite_status,
            "reward": float(reward_text.replace(",", "")),
            "samples": samples_value,
            "compliance": compliance_text,
        }
        rows.append(row)

    return rows


def write_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        raise ValueError("No ranking rows to write")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: List[Dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def default_output_paths(prefix: str) -> tuple[Path, Path]:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    return (
        Path(f"analysis/{prefix}_{stamp}.csv"),
        Path(f"analysis/{prefix}_{stamp}.json"),
    )


def default_my_miners_snapshot_path() -> Path:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    return Path(f"analysis/my_miners_snapshots/my_miners_{stamp}.csv")


def parse_listminerinscreens_output(raw: str, host: str) -> List[Dict[str, object]]:
    miners: List[Dict[str, object]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("session") or line.startswith("-"):
            continue
        if "|" not in line:
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue

        try:
            uid_val = int(parts[1])
        except ValueError:
            continue

        miners.append(
            {
                "host": host,
                "session": parts[0],
                "miner_uid": uid_val,
                "hotkey_name": parts[2],
                "ss58_address": parts[3],
            }
        )
    return miners


def fetch_miners_from_host(host: str, timeout: int, remote_cmd: str) -> List[Dict[str, object]]:
    ssh_cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"ConnectTimeout={timeout}",
        host,
        remote_cmd,
    ]
    proc = subprocess.run(ssh_cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(
            f"SSH command failed for {host} (exit {proc.returncode}). stderr: {stderr}"
        )
    return parse_listminerinscreens_output(proc.stdout, host)


def fetch_miner_envs_from_host(host: str, timeout: int) -> Dict[str, Dict[str, str]]:
    """Return per-screen-session env config parsed from remote ps output."""

    ssh_cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"ConnectTimeout={timeout}",
        host,
        REMOTE_PS_ENV_CMD,
    ]
    proc = subprocess.run(ssh_cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(
            f"SSH command failed for {host} (exit {proc.returncode}). stderr: {stderr}"
        )

    per_session: Dict[str, Dict[str, str]] = {}
    for line in proc.stdout.splitlines():
        session_match = re.search(r"SCREEN\s+-dmS\s+([\w.-]+)", line)
        if not session_match:
            continue
        session = session_match.group(1)

        ml_max_hands_match = re.search(r"export\s+ML_MAX_HANDS=([^\s]+)", line)
        remove_other_match = re.search(r"export\s+REMOVE_OTHER=([^\s]+)", line)
        alias_match = re.search(
            r"export\s+POKER44_SINGLE_HAND_MODEL_ALIAS=([^\s]+)",
            line,
        )

        per_session[session] = {
            "ML_MAX_HANDS": ml_max_hands_match.group(1) if ml_max_hands_match else "",
            "REMOVE_OTHER": remove_other_match.group(1) if remove_other_match else "",
            "POKER44_SINGLE_HAND_MODEL_ALIAS": alias_match.group(1) if alias_match else "",
        }

    return per_session


def build_my_miners_snapshot_rows(
    all_remote_miners: List[Dict[str, object]],
    merged_rows: List[Dict[str, object]],
    fetched_at_utc: str,
    env_by_host_session: Dict[tuple[str, str], Dict[str, str]],
) -> List[Dict[str, object]]:
    row_by_uid: Dict[int, Dict[str, object]] = {int(row["uid"]): row for row in merged_rows}

    snapshot_rows: List[Dict[str, object]] = []
    for miner in sorted(all_remote_miners, key=lambda m: (str(m["host"]), int(m["miner_uid"]))):
        uid = int(miner["miner_uid"])
        row = row_by_uid.get(uid)
        rank = int(row["rank"]) if row else None
        composite = parse_float_or_none(str(row.get("composite"))) if row else None
        composite_status = str(row.get("composite_status") or "") if row else ""
        reward = float(row["reward"]) if row else None
        samples = int(row["samples"]) if row and row.get("samples") is not None else None
        compliance = str(row.get("compliance") or "") if row else ""
        env_cfg = env_by_host_session.get((str(miner["host"]), str(miner["session"])), {})

        snapshot_rows.append(
            {
                "fetched_at_utc": fetched_at_utc,
                "host": str(miner["host"]),
                "session": str(miner["session"]),
                "miner_uid": uid,
                "hotkey": str(miner["hotkey_name"]),
                "ss58_address": str(miner["ss58_address"]),
                "rank": rank,
                "composite": composite,
                "composite_status": composite_status,
                "reward": reward,
                "samples": samples,
                "compliance": compliance,
                "POKER44_SINGLE_HAND_MODEL_ALIAS": env_cfg.get(
                    "POKER44_SINGLE_HAND_MODEL_ALIAS", ""
                ),
                "ML_MAX_HANDS": env_cfg.get("ML_MAX_HANDS", ""),
                "REMOVE_OTHER": env_cfg.get("REMOVE_OTHER", ""),
            }
        )
    return snapshot_rows


def print_my_miners_report(snapshot_rows: List[Dict[str, object]]) -> None:
    print("\nMy miners in leaderboard:")
    print(
        "host   | session              | miner_uid | hotkey | rank | composite | reward | "
        "samples | compliance | composite_status | model_alias | ML_MAX_HANDS | REMOVE_OTHER"
    )
    print(
        "-------+----------------------+-----------+--------+------+-----------+--------+"
        "---------+------------+------------------+-------------+--------------+-------------"
    )

    if not snapshot_rows:
        print("(no miners found from SSH hosts)")
        return

    for row in snapshot_rows:
        uid = int(row["miner_uid"])
        rank = row["rank"]
        composite = f"{float(row['composite']):.3f}" if row["composite"] is not None else "N/A"
        composite_status = str(row.get("composite_status") or "") or "N/A"
        reward = f"{float(row['reward']):.3f}" if row["reward"] is not None else "N/A"
        samples = str(row.get("samples")) if row.get("samples") is not None else "N/A"
        compliance = str(row.get("compliance") or "") or "N/A"
        rank_str = str(rank) if rank is not None else "N/A"
        model_alias = str(row.get("POKER44_SINGLE_HAND_MODEL_ALIAS") or "") or "N/A"
        ml_max_hands = str(row.get("ML_MAX_HANDS") or "") or "N/A"
        remove_other = str(row.get("REMOVE_OTHER") or "") or "N/A"
        print(
            f"{str(row['host']):<6} | "
            f"{str(row['session']):<20} | "
            f"{uid:<9} | "
            f"{str(row['hotkey']):<6} | "
            f"{rank_str:<4} | "
            f"{composite:<9} | "
            f"{reward:<6} | "
            f"{samples:<7} | "
            f"{compliance:<10} | "
            f"{composite_status:<16} | "
            f"{model_alias:<11} | "
            f"{ml_max_hands:<12} | "
            f"{remove_other}"
        )


def write_my_miners_snapshot(snapshot_rows: List[Dict[str, object]], out_path: Path) -> None:
    if not snapshot_rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fetched_at_utc",
        "host",
        "session",
        "miner_uid",
        "hotkey",
        "ss58_address",
        "rank",
        "composite",
        "composite_status",
        "reward",
        "samples",
        "compliance",
        "POKER44_SINGLE_HAND_MODEL_ALIAS",
        "ML_MAX_HANDS",
        "REMOVE_OTHER",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(snapshot_rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and merge Poker44 dashboard ranking pages into one file."
    )
    parser.add_argument("--start-page", type=int, default=1, help="First page to fetch (default: 1)")
    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="Last page to fetch. If omitted, auto-detect from total miners.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="HTTP timeout in seconds per request (default: 20)",
    )
    parser.add_argument(
        "--sleep-ms",
        type=int,
        default=1000,
        help="Delay between dashboard page requests in milliseconds (default: 1000).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="CSV output path (default: analysis/dashboard_ranking_<utc>.csv)",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="JSON output path (default: analysis/dashboard_ranking_<utc>.json)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Disable JSON output.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable CSV output.",
    )
    parser.add_argument(
        "--ssh-hosts",
        type=str,
        default=DEFAULT_SSH_HOSTS,
        help=(
            "Comma-separated SSH hosts to fetch ./listminerinscreens from "
            f'(default: "{DEFAULT_SSH_HOSTS}"). Use empty string to disable.'
        ),
    )
    parser.add_argument(
        "--ssh-timeout",
        type=int,
        default=12,
        help="SSH connect timeout in seconds (default: 12)",
    )
    parser.add_argument(
        "--remote-cmd",
        type=str,
        default="cd ~/Poker44-subnet-main && ./listminerinscreens",
        help="Remote command used on each SSH host.",
    )
    parser.add_argument(
        "--http-retries",
        type=int,
        default=2,
        help="Number of retries per dashboard page on HTTP/network failure (default: 2).",
    )
    parser.add_argument(
        "--http-retry-sleep",
        type=float,
        default=0.8,
        help="Sleep between HTTP retries in seconds (default: 0.8).",
    )
    parser.add_argument(
        "--my-miners-out",
        type=Path,
        default=None,
        help=(
            "Path to save per-run snapshot of your miners. "
            "Default: analysis/my_miners_snapshots/my_miners_<utc>.csv"
        ),
    )
    parser.add_argument(
        "--no-my-miners-snapshot",
        action="store_true",
        help="Disable writing per-run my-miners snapshot CSV.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.start_page < 1:
        print("ERROR: --start-page must be >= 1", file=sys.stderr)
        return 2

    if args.end_page is not None and args.end_page < args.start_page:
        print("ERROR: --end-page must be >= --start-page", file=sys.stderr)
        return 2

    csv_default, json_default = default_output_paths("dashboard_ranking")
    out_csv = args.out_csv or csv_default
    out_json = args.out_json or json_default

    if args.no_csv and args.no_json:
        print("ERROR: cannot disable both outputs (--no-csv and --no-json)", file=sys.stderr)
        return 2

    session = requests.Session()
    session.headers.update(HEADERS)

    print(f"Fetching page {args.start_page}: {build_url(args.start_page)}")
    first_html = fetch_html_with_retries(
        session=session,
        page=args.start_page,
        timeout=args.timeout,
        retries=args.http_retries,
        retry_sleep_s=args.http_retry_sleep,
    )
    first_rows = parse_rows(first_html, args.start_page)
    if not first_rows:
        print(
            f"ERROR: Could not parse ranking rows from page {args.start_page}. "
            "Site markup may have changed.",
            file=sys.stderr,
        )
        return 1

    rows_per_page = len(first_rows)
    end_page = args.end_page
    if end_page is None:
        total_miners = extract_total_miners(first_html)
        if total_miners is not None:
            end_page = max(args.start_page, math.ceil(total_miners / rows_per_page))
        else:
            max_page = extract_max_page_from_pagination(first_html)
            if max_page is not None:
                end_page = max(args.start_page, max_page)
            else:
                print(
                    "WARNING: Could not auto-detect page count. Falling back to start page only.",
                    file=sys.stderr,
                )
                end_page = args.start_page

    all_rows = list(first_rows)
    if end_page > args.start_page:
        for page in range(args.start_page + 1, end_page + 1):
            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)
            print(f"Fetching page {page}: {build_url(page)}")
            try:
                page_html = fetch_html_with_retries(
                    session=session,
                    page=page,
                    timeout=args.timeout,
                    retries=args.http_retries,
                    retry_sleep_s=args.http_retry_sleep,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"WARNING: Could not fetch page {page}: {exc}. Skipping.",
                    file=sys.stderr,
                )
                continue
            page_rows = parse_rows(page_html, page)
            if not page_rows:
                print(
                    f"WARNING: Page {page} returned 0 parsed rows. Stopping at previous page.",
                    file=sys.stderr,
                )
                break
            all_rows.extend(page_rows)

    # Keep a single record per rank in case the site duplicated any rows.
    dedup_by_rank: Dict[int, Dict[str, object]] = {}
    for row in all_rows:
        dedup_by_rank[int(row["rank"])] = row

    merged_rows = [dedup_by_rank[k] for k in sorted(dedup_by_rank)]

    if not args.no_csv:
        write_csv(merged_rows, out_csv)
    if not args.no_json:
        write_json(merged_rows, out_json)

    fetched_pages = sorted({int(r["page"]) for r in merged_rows})
    print(
        "Fetched pages: "
        f"{fetched_pages[0]}-{fetched_pages[-1]} "
        f"({len(fetched_pages)} pages), rows merged: {len(merged_rows)}"
    )
    if not args.no_csv:
        print(f"CSV:  {out_csv}")
    if not args.no_json:
        print(f"JSON: {out_json}")

    ssh_hosts: Sequence[str] = []
    if args.ssh_hosts.strip():
        ssh_hosts = [h.strip() for h in args.ssh_hosts.split(",") if h.strip()]

    if ssh_hosts:
        fetched_at_utc = dt.datetime.now(dt.UTC).isoformat()
        all_remote_miners: List[Dict[str, object]] = []
        env_by_host_session: Dict[tuple[str, str], Dict[str, str]] = {}
        for host in ssh_hosts:
            try:
                host_miners = fetch_miners_from_host(
                    host=host,
                    timeout=args.ssh_timeout,
                    remote_cmd=args.remote_cmd,
                )
                all_remote_miners.extend(host_miners)

                host_envs = fetch_miner_envs_from_host(
                    host=host,
                    timeout=args.ssh_timeout,
                )
                for session, cfg in host_envs.items():
                    env_by_host_session[(host, session)] = cfg

                print(f"SSH {host}: miners found: {len(host_miners)}")
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: {exc}", file=sys.stderr)

        snapshot_rows = build_my_miners_snapshot_rows(
            all_remote_miners=all_remote_miners,
            merged_rows=merged_rows,
            fetched_at_utc=fetched_at_utc,
            env_by_host_session=env_by_host_session,
        )
        print_my_miners_report(snapshot_rows=snapshot_rows)

        if not args.no_my_miners_snapshot:
            my_miners_out = args.my_miners_out or default_my_miners_snapshot_path()
            write_my_miners_snapshot(snapshot_rows=snapshot_rows, out_path=my_miners_out)
            print(f"My miners snapshot CSV: {my_miners_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
