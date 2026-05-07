#!/usr/bin/env python3
"""
Poker44 Training Benchmark Downloader
======================================
Downloads released benchmark chunks from the Poker44 API.
Designed to run both as a one-time full download and as a daily cron job
that only fetches new releases.

Usage:
    python fetch_benchmark.py              # incremental (only new dates)
    python fetch_benchmark.py --full       # force re-download of all dates
    python fetch_benchmark.py --status     # show what's downloaded vs available
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

API_BASE = "https://api.poker44.net/api/v1/benchmark"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmark"
MANIFEST_FILE = DATA_DIR / "manifest.json"
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5  # seconds between retries


def api_get(path: str) -> dict:
    url = f"{API_BASE}{path}"
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"Accept": "application/json", "User-Agent": "poker44-benchmark-fetcher/1.0"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if not data.get("success"):
                    raise RuntimeError(f"API returned success=false for {url}: {data}")
                return data["data"]
        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code} on attempt {attempt}/{RETRY_ATTEMPTS}: {url}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY)
            else:
                raise
        except Exception as e:
            print(f"  Error on attempt {attempt}/{RETRY_ATTEMPTS}: {e}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY)
            else:
                raise


def load_manifest() -> dict:
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            return json.load(f)
    return {"downloaded_dates": {}, "last_updated": None}


def save_manifest(manifest: dict):
    manifest["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def fetch_date(source_date: str) -> dict:
    print(f"  Fetching chunks for {source_date}...")
    data = api_get(f"/chunks?sourceDate={source_date}")
    return data


def save_date_data(source_date: str, data) -> Path:
    out_file = DATA_DIR / f"benchmark_{source_date}.json"
    # data may be a list (array of chunk objects) or a dict with a list inside
    with open(out_file, "w") as f:
        json.dump(data, f)
    return out_file


def show_status():
    manifest = load_manifest()
    downloaded = manifest.get("downloaded_dates", {})

    print("Fetching available releases from API...")
    releases_data = api_get("/releases")
    releases = releases_data.get("releases", [])

    print(f"\n{'Date':<14} {'Chunks':>8} {'Hands':>10} {'Status':<12} {'File'}")
    print("-" * 70)
    total_chunks = 0
    total_hands = 0
    for r in sorted(releases, key=lambda x: x["sourceDate"]):
        date = r["sourceDate"]
        chunks = r["chunkCount"]
        hands = r["handCount"]
        if date in downloaded:
            status = "downloaded"
            file_info = downloaded[date].get("file", "")
        else:
            status = "MISSING"
            file_info = ""
        total_chunks += chunks
        total_hands += hands
        print(f"  {date:<12} {chunks:>8} {hands:>10}  {status:<12} {file_info}")

    print("-" * 70)
    print(f"  {'TOTAL':<12} {total_chunks:>8} {total_hands:>10}")
    print(f"\nDownloaded: {len(downloaded)}/{len(releases)} dates")
    print(f"Manifest last updated: {manifest.get('last_updated', 'never')}")


def run(force_full: bool = False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    downloaded = manifest.setdefault("downloaded_dates", {})

    print("Fetching available releases...")
    releases_data = api_get("/releases")
    releases = releases_data.get("releases", [])
    release_version = releases_data.get("releaseVersion", "?")

    print(f"Release version: {release_version}")
    print(f"Available dates: {len(releases)}")

    new_count = 0
    skipped_count = 0
    error_count = 0

    for r in sorted(releases, key=lambda x: x["sourceDate"]):
        date = r["sourceDate"]
        chunk_count = r["chunkCount"]
        hand_count = r["handCount"]
        released_at = r.get("releasedAt", "?")

        if not force_full and date in downloaded:
            print(f"  {date} — already downloaded ({chunk_count} chunks, {hand_count} hands), skipping")
            skipped_count += 1
            continue

        print(f"  {date} — {chunk_count} chunks, {hand_count} hands (released {released_at})")
        try:
            data = fetch_date(date)
            out_file = save_date_data(date, data)
            size_kb = out_file.stat().st_size // 1024
            downloaded[date] = {
                "file": out_file.name,
                "chunk_count": chunk_count,
                "hand_count": hand_count,
                "released_at": released_at,
                "release_version": r.get("releaseVersion", release_version),
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "file_size_kb": size_kb,
            }
            save_manifest(manifest)
            print(f"    Saved to {out_file.name} ({size_kb} KB)")
            new_count += 1
        except Exception as e:
            print(f"    ERROR: {e}", file=sys.stderr)
            error_count += 1

    print(f"\nDone. New: {new_count}, Skipped: {skipped_count}, Errors: {error_count}")
    if error_count > 0:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Poker44 Benchmark Data Fetcher")
    parser.add_argument("--full", action="store_true", help="Re-download all dates (ignore manifest)")
    parser.add_argument("--status", action="store_true", help="Show download status without downloading")
    args = parser.parse_args()

    if args.status:
        show_status()
    else:
        run(force_full=args.full)


if __name__ == "__main__":
    main()
