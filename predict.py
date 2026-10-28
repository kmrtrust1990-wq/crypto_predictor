#!/usr/bin/env python3
"""কমান্ড লাইন থেকে পূর্বাভাস।

উদাহরণ:
    python predict.py --symbol BTCUSDT --interval 1h
    python predict.py --source yahoo --symbol ETH-USD --interval 4h --horizon 3
    python predict.py --symbol SOLUSDT --interval 15m --watch          # প্রতিটি নতুন ক্যান্ডেলে আবার চালাবে
    python predict.py --csv my_data.csv --interval 1d --json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from predictor import INTERVALS, MODELS, Settings, analyze, fetch_ohlcv, load_csv
from predictor.data import SOURCES, interval_offset
from predictor.report import text_report


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ক্রিপ্টো ক্যান্ডেল প্রেডিক্টর")
    ap.add_argument("--source", choices=list(SOURCES), default="binance", help="ডেটার উৎস")
    ap.add_argument("--symbol", help="যেমন BTCUSDT, ETH/USDT, SOL-USD (ডিফল্ট BTCUSDT)")
    ap.add_argument("--csv", help="Binance/Yahoo-র বদলে নিজের CSV ফাইল")
    ap.add_argument("--interval", choices=INTERVALS, default="1h", help="টাইমফ্রেম")
    ap.add_argument("--limit", type=int, default=1000, help="কয়টি ক্যান্ডেল আনবে (ডিফল্ট 1000)")
    ap.add_argument("--horizon", type=int, default=1, help="কয়টি ক্যান্ডেল পরের পূর্বাভাস (ডিফল্ট 1)")
    ap.add_argument("--model", choices=list(MODELS), default="gb", help="gb / rf / linear")
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="BUY-এর জন্য ন্যূনতম সম্ভাবনা; SELL হবে 1−threshold-এর নিচে (ডিফল্ট 0.55)")
    ap.add_argument("--no-backtest", action="store_true", help="ব্যাকটেস্ট বাদ দিয়ে শুধু পূর্বাভাস")
    ap.add_argument("--min-train", type=int, default=300, help="ব্যাকটেস্টে প্রথম ট্রেনিংয়ের ক্যান্ডেল সংখ্যা")
    ap.add_argument("--retrain-every", type=int, default=50, help="ব্যাকটেস্টে কত ক্যান্ডেল পর পর মডেল নতুন করে ট্রেন")
    ap.add_argument("--fee", type=float, default=0.1, help="প্রতি ট্রেডে ফি, শতাংশে (ডিফল্ট 0.1)")
    ap.add_argument("--allow-short", action="store_true", help="SELL সিগন্যালে শর্ট পজিশন ধরবে")
    ap.add_argument("--json", action="store_true", help="রিপোর্টের বদলে JSON আউটপুট")
    ap.add_argument("--save-csv", help="আনা ডেটা এই ফাইলে সেভ করবে")
    ap.add_argument("--save-predictions", help="ব্যাকটেস্টের প্রতিটি পূর্বাভাস এই CSV-তে সেভ করবে")
    ap.add_argument("--watch", action="store_true", help="প্রতিটি নতুন ক্যান্ডেল বন্ধ হলে আবার পূর্বাভাস দেবে")
    return ap.parse_args()


def load(args) -> pd.DataFrame:
    if args.csv:
        return load_csv(args.csv).tail(args.limit)
    return fetch_ohlcv(args.source, args.symbol or "BTCUSDT", args.interval, args.limit)


def to_json(result, source: str, symbol: str) -> str:
    p = result.prediction
    out = {
        "symbol": symbol,
        "source": source,
        "interval": result.settings.interval,
        "horizon": result.settings.horizon,
        "last_candle": p.time.isoformat(),
        "target_time": p.target_time.isoformat(),
        "close": p.close,
        "prob_up": round(p.prob_up, 4),
        "direction": "UP" if p.prob_up >= 0.5 else "DOWN",
        "signal": p.signal,
        "pred_price": p.pred_price,
        "price_range_80": [p.price_low, p.price_high],
        "models_agree": bool(p.agree),
        "patterns": p.patterns,
    }
    if result.backtest is not None:
        out["backtest"] = {k: _json_value(v) for k, v in result.backtest.metrics.items()}
    return json.dumps(out, ensure_ascii=False, indent=2)


def _json_value(v):
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if pd.isna(v):
        return None
    return int(v) if isinstance(v, int) else float(v)


def run_once(args, settings: Settings) -> pd.Timestamp:
    df = load(args)
    if args.save_csv:
        df.to_csv(args.save_csv)
    result = analyze(df, settings)
    source = "CSV" if args.csv else SOURCES[args.source]
    symbol = args.symbol or (Path(args.csv).stem if args.csv else "BTCUSDT")
    print(to_json(result, source, symbol) if args.json else text_report(result, source, symbol))
    if args.save_predictions and result.backtest is not None:
        result.backtest.predictions.to_csv(args.save_predictions)
    return result.prediction.time


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # Windows টার্মিনালে বাংলা লেখার জন্য
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if not 0.5 < args.threshold < 1:
        sys.exit("--threshold 0.5 এর বেশি এবং 1 এর কম হতে হবে")
    settings = Settings(
        interval=args.interval, model=args.model, horizon=args.horizon, threshold=args.threshold,
        backtest=not args.no_backtest, min_train=args.min_train, retrain_every=args.retrain_every,
        fee=args.fee / 100, allow_short=args.allow_short, importance=False,
    )

    try:
        last = run_once(args, settings)
    except (ValueError, ConnectionError) as exc:
        print(f"ত্রুটি: {exc}", file=sys.stderr)
        return 1

    if not args.watch or args.csv:
        return 0
    settings.backtest = False  # প্রথমবার ব্যাকটেস্ট হয়েছে; পরে শুধু নতুন পূর্বাভাস
    step = interval_offset(args.interval)
    while True:
        # চলমান ক্যান্ডেল বন্ধ হওয়া পর্যন্ত অপেক্ষা (+৫ সেকেন্ড যাতে এক্সচেঞ্জ ডেটা আপডেট করে)
        wake = last + step + step + pd.Timedelta(seconds=5)
        wait = (wake - pd.Timestamp.now(tz="UTC")).total_seconds()
        print(f"\n⏳ পরের ক্যান্ডেল বন্ধ হওয়ার অপেক্ষা: {wake:%Y-%m-%d %H:%M:%S} UTC (Ctrl+C দিয়ে বন্ধ করুন)\n")
        try:
            time.sleep(max(wait, 5))
            last = run_once(args, settings)
        except KeyboardInterrupt:
            return 0
        except (ValueError, ConnectionError) as exc:
            print(f"ত্রুটি: {exc} — ৬০ সেকেন্ড পর আবার চেষ্টা", file=sys.stderr)
            time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
