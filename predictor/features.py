"""মডেলের ইনপুট (ফিচার) ও লক্ষ্য (টার্গেট) তৈরি।

সব ফিচার দামের স্কেল থেকে মুক্ত (রেশিও/শতাংশ), তাই BTC আর ছোট কয়েন — দুটোতেই একই মডেল কাজ করে।
সারি t-এর ফিচার শুধু ক্যান্ডেল t পর্যন্ত ডেটা দেখে; টার্গেট হলো t থেকে t+horizon পর্যন্ত রিটার্ন।
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import add_indicators
from .patterns import PATTERNS, detect_patterns, pattern_score


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    ind = add_indicators(df)
    pats = detect_patterns(df)
    c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
    logret = np.log(c).diff()
    rng = (h - l).replace(0, np.nan)

    f = pd.DataFrame(index=df.index)
    for n in (1, 3, 6, 12, 24):
        f[f"ret_{n}"] = np.log(c / c.shift(n))
    for lag in (1, 2, 3):
        f[f"ret_1_lag{lag}"] = logret.shift(lag)
    f["volatility_12"] = logret.rolling(12).std()
    f["volatility_48"] = logret.rolling(48).std()
    f["volatility_ratio"] = f["volatility_12"] / f["volatility_48"]

    # ক্যান্ডেলের গঠন
    f["body_pct"] = (c - o) / rng
    f["upper_wick_pct"] = (h - np.maximum(o, c)) / rng
    f["lower_wick_pct"] = (np.minimum(o, c) - l) / rng
    f["range_pct"] = (h - l) / c

    # ইন্ডিকেটর
    f["rsi_14"] = ind["rsi_14"] / 100
    f["rsi_7"] = ind["rsi_7"] / 100
    f["macd_norm"] = ind["macd"] / c
    f["macd_hist_norm"] = ind["macd_hist"] / c
    f["close_ema9"] = c / ind["ema9"] - 1
    f["close_ema21"] = c / ind["ema21"] - 1
    f["close_ema50"] = c / ind["ema50"] - 1
    f["ema9_ema21"] = ind["ema9"] / ind["ema21"] - 1
    f["ema21_ema50"] = ind["ema21"] / ind["ema50"] - 1
    f["bb_pctb"] = (c - ind["bb_lower"]) / (ind["bb_upper"] - ind["bb_lower"])
    f["bb_width"] = (ind["bb_upper"] - ind["bb_lower"]) / ind["bb_mid"]
    f["atr_pct"] = ind["atr_14"] / c
    f["stoch_k"] = ind["stoch_k"] / 100
    f["stoch_d"] = ind["stoch_d"] / 100
    f["dist_high_20"] = c / h.rolling(20).max() - 1
    f["dist_low_20"] = c / l.rolling(20).min() - 1

    # ভলিউম — ভলিউম শূন্য হলে (কিছু CSV-তে) ০ ধরা হয়
    vol_mean = v.rolling(20).mean()
    f["volume_z"] = (v - vol_mean) / v.rolling(20).std()
    f["volume_ratio"] = np.log(v / vol_mean)
    f["obv_slope"] = (ind["obv"] - ind["obv"].shift(10)) / v.rolling(10).sum()

    # ক্যান্ডেলস্টিক প্যাটার্ন
    for col in PATTERNS:
        f[f"pat_{col}"] = pats[col].astype(int)
    f["pattern_score"] = pattern_score(pats)

    f = f.replace([np.inf, -np.inf], np.nan)
    volume_cols = ["volume_z", "volume_ratio", "obv_slope"]
    f[volume_cols] = f[volume_cols].fillna(0.0)
    return f


def make_dataset(df: pd.DataFrame, horizon: int = 1) -> tuple[pd.DataFrame, list[str]]:
    """ফিচার + টার্গেট। শেষের `horizon`টি সারির টার্গেট অজানা (NaN) — সেগুলোই লাইভ পূর্বাভাসের সারি।"""
    feats = build_features(df)
    feature_cols = list(feats.columns)
    future_ret = np.log(df["close"].shift(-horizon) / df["close"])

    data = feats.copy()
    data["target_return"] = future_ret
    data["target_up"] = (future_ret > 0).astype(float).where(future_ret.notna())
    data = data.dropna(subset=feature_cols)
    return data, feature_cols
