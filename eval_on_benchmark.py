#!/usr/bin/env python3
"""Evaluate a gen4/gen5 model on API benchmark data.

Usage:
    python eval_on_benchmark.py --model ml_gen5_s123467_OLD
    python eval_on_benchmark.py --model ml_gen5_s123467 --max-chunks 200
    python eval_on_benchmark.py --model weights/my_model_model.pkl --max-chunks 500
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from poker44 import miner_heuristics as mh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval gen4/gen5 model on API benchmark")
    p.add_argument(
        "--model",
        required=True,
        help="Model prefix (e.g. ml_gen5_s123467) or full path to _model.pkl file",
    )
    p.add_argument(
        "--weights-dir",
        type=pathlib.Path,
        default=pathlib.Path("weights"),
        help="Directory with model/scaler pkl files (default: weights/)",
    )
    p.add_argument(
        "--benchmark-dir",
        type=pathlib.Path,
        default=pathlib.Path("data/benchmark"),
        help="Directory with benchmark_*.json files",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Bot probability threshold per hand (default: 0.30)",
    )
    p.add_argument(
        "--chunk-threshold",
        type=float,
        default=0.5,
        help="Fraction of hand-level bot flags to call chunk as bot (default: 0.5)",
    )
    p.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Stop after this many scoring units (optional)",
    )
    p.add_argument(
        "--report-every",
        type=int,
        default=100,
        help="Print running stats every N chunks (default: 100)",
    )
    p.add_argument(
        "--min-players",
        type=int,
        default=4,
        help="Skip hands with fewer than this many players (default: 4)",
    )
    return p.parse_args()


def resolve_paths(args: argparse.Namespace):
    model_arg = args.model
    # If it looks like a path to a pkl file, use directly
    if model_arg.endswith(".pkl"):
        model_path = pathlib.Path(model_arg)
        scaler_path = model_path.parent / model_path.name.replace("_model.pkl", "_scaler.pkl")
    else:
        model_path = args.weights_dir / f"{model_arg}_model.pkl"
        scaler_path = args.weights_dir / f"{model_arg}_scaler.pkl"

    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not scaler_path.exists():
        print(f"ERROR: scaler not found: {scaler_path}", file=sys.stderr)
        sys.exit(1)
    return model_path, scaler_path


def print_stats(y_true, y_pred, prefix=""):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    acc = np.mean(y_true == y_pred)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr    = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    print(
        f"{prefix}n={len(y_true):4d}  bots={y_true.sum():3d}  humans={(y_true==0).sum():3d} | "
        f"acc={acc:.3f}  recall={recall:.3f}  FPR={fpr:.3f} | "
        f"TP={tp} FP={fp} FN={fn} TN={tn}"
    )


def main():
    args = parse_args()
    model_path, scaler_path = resolve_paths(args)

    print(f"Model:   {model_path}")
    print(f"Scaler:  {scaler_path}")
    print(f"Thresh:  hand>={args.threshold}  chunk>={args.chunk_threshold}  min_players>={args.min_players}")
    print()

    model  = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))

    benchmark_files = sorted(args.benchmark_dir.glob("benchmark_*.json"))
    if not benchmark_files:
        print(f"ERROR: no benchmark_*.json files in {args.benchmark_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Benchmark files: {[f.name for f in benchmark_files]}")
    print()

    y_true: list[int] = []
    y_pred: list[int] = []
    errors = 0
    done = False
    first_hand_printed = False

    for f in benchmark_files:
        if done:
            break
        d = json.loads(f.read_text())
        file_chunks = d.get("chunks", [])
        print(f"--- {f.name}  ({len(file_chunks)} chunk-items) ---")

        for chunk_item in file_chunks:
            if done:
                break
            gt_list = chunk_item.get("groundTruth", [])
            chunks  = chunk_item.get("chunks", [])

            for i, hand_list in enumerate(chunks):
                if done:
                    break
                if i >= len(gt_list):
                    continue
                label = gt_list[i]

                bot_flags: list[int] = []
                skipped = 0
                for hand in hand_list:
                    if len(hand.get("players") or []) < args.min_players:
                        skipped += 1
                        continue
                    try:
                        feat = mh._extract_ml_features_gen4([hand])
                        feat_scaled = scaler.transform(feat.reshape(1, -1))
                        prob = model.predict_proba(feat_scaled)[0][1]
                        if not first_hand_printed:
                            first_hand_printed = True
                            bb = (hand.get("metadata") or {}).get("bb", 0.02)
                            stacks_raw = [p.get("starting_stack", 0) for p in (hand.get("players") or [])]
                            print("=== FIRST HAND DEBUG ===")
                            print(f"  bb={bb}  stacks_raw={stacks_raw}")
                            print(f"  stacks_BB={[round(s/bb,1) for s in stacks_raw if s]}")
                            names = ["num_players","filled_ratio","stack_mean","stack_std","stack_cv",
                                     "total_actions","call_r","check_r","fold_r","raise_r","agg_ratio",
                                     "amount_mean_bb","amount_max_bb","total_pot_bb","showdown","street_depth"]
                            print("  raw features sent to scaler:")
                            for n, v in zip(names, feat):
                                print(f"    {n:20s} = {v:.4f}")
                            print("  scaled features (after scaler):")
                            for n, v in zip(names, feat_scaled[0]):
                                print(f"    {n:20s} = {v:.4f}")
                            print(f"  prob_bot={prob:.4f}")
                            print("========================")
                        bot_flags.append(1 if prob >= args.threshold else 0)
                    except Exception:
                        errors += 1

                if not bot_flags:
                    print(f"  chunk {len(y_true)+1}: SKIP (0 hands after filter, skipped={skipped}/{len(hand_list)})")
                    continue

                pred = 1 if np.mean(bot_flags) >= args.chunk_threshold else 0
                y_true.append(label)
                y_pred.append(pred)

                n = len(y_true)
                if skipped:
                    print(f"  chunk {n}: used={len(bot_flags)}/{len(hand_list)}  skipped={skipped}  pred={'BOT' if pred else 'HUM'}  true={'BOT' if label else 'HUM'}")
                if n % args.report_every == 0:
                    print_stats(y_true, y_pred, prefix=f"  [{n:4d}] ")

                if args.max_chunks and n >= args.max_chunks:
                    done = True

    print()
    print("=== FINAL ===")
    print_stats(y_true, y_pred)
    if errors:
        print(f"Feature extraction errors: {errors}")


if __name__ == "__main__":
    main()
