"""ক্যান্ডেলস্টিক প্যাটার্ন শনাক্ত করা।

ক্রিপ্টো বাজার ২৪ ঘণ্টা চলে, তাই এক ক্যান্ডেলের open প্রায় সবসময় আগের close-এর সমান —
গ্যাপের শর্তগুলো সেভাবে (<=, >=) লেখা হয়েছে।
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# কলাম → (নাম, দিক)   দিক: +1 বুলিশ, -1 বেয়ারিশ, 0 নিরপেক্ষ
PATTERNS = {
    "doji": ("Doji", 0),
    "hammer": ("Hammer", 1),
    "inverted_hammer": ("Inverted Hammer", 1),
    "hanging_man": ("Hanging Man", -1),
    "shooting_star": ("Shooting Star", -1),
    "bullish_engulfing": ("Bullish Engulfing", 1),
    "bearish_engulfing": ("Bearish Engulfing", -1),
    "bullish_harami": ("Bullish Harami", 1),
    "bearish_harami": ("Bearish Harami", -1),
    "piercing_line": ("Piercing Line", 1),
    "dark_cloud_cover": ("Dark Cloud Cover", -1),
    "morning_star": ("Morning Star", 1),
    "evening_star": ("Evening Star", -1),
    "three_white_soldiers": ("Three White Soldiers", 1),
    "three_black_crows": ("Three Black Crows", -1),
    "bullish_marubozu": ("Bullish Marubozu", 1),
    "bearish_marubozu": ("Bearish Marubozu", -1),
}

DIRECTION_LABELS = {1: "বুলিশ (উপরে)", -1: "বেয়ারিশ (নিচে)", 0: "নিরপেক্ষ"}


def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    bull, bear = c > o, c < o

    # আগের ৫টি ক্যান্ডেলের ট্রেন্ড (বর্তমান ক্যান্ডেল বাদে)
    trend = c.shift(1) / c.shift(6) - 1
    down, up = trend < 0, trend > 0

    def prev(s: pd.Series, n: int = 1) -> pd.Series:
        return s.shift(n)

    p_o, p_c, p_body, p_rng = prev(o), prev(c), prev(body), prev(rng)
    p_bull, p_bear = prev(bull).eq(True), prev(bear).eq(True)
    p_mid = (p_o + p_c) / 2

    hammer_shape = (lower >= 2 * body) & (upper <= 0.2 * rng)
    inverted_shape = (upper >= 2 * body) & (lower <= 0.2 * rng)
    big = body >= 0.5 * rng
    small = body <= 0.3 * rng

    out = pd.DataFrame(index=df.index)
    out["doji"] = body <= 0.1 * rng
    out["hammer"] = hammer_shape & down
    out["inverted_hammer"] = inverted_shape & down
    out["hanging_man"] = hammer_shape & up
    out["shooting_star"] = inverted_shape & up
    out["bullish_engulfing"] = p_bear & bull & (o <= p_c) & (c >= p_o) & (body > p_body)
    out["bearish_engulfing"] = p_bull & bear & (o >= p_c) & (c <= p_o) & (body > p_body)
    out["bullish_harami"] = p_bear & bull & (o >= p_c) & (c < p_o) & (body < 0.5 * p_body)
    out["bearish_harami"] = p_bull & bear & (o <= p_c) & (c > p_o) & (body < 0.5 * p_body)
    out["piercing_line"] = (
        p_bear & bull & (p_body >= 0.5 * p_rng) & (o <= p_c) & (c > p_mid) & (c < p_o)
    )
    out["dark_cloud_cover"] = (
        p_bull & bear & (p_body >= 0.5 * p_rng) & (o >= p_c) & (c < p_mid) & (c > p_o)
    )
    first_mid = (prev(o, 2) + prev(c, 2)) / 2
    out["morning_star"] = (
        prev(bear, 2).eq(True) & prev(big, 2).eq(True) & prev(small).eq(True) & bull & (c > first_mid)
    )
    out["evening_star"] = (
        prev(bull, 2).eq(True) & prev(big, 2).eq(True) & prev(small).eq(True) & bear & (c < first_mid)
    )
    three_bull = bull & p_bull & prev(bull, 2).eq(True)
    three_bear = bear & p_bear & prev(bear, 2).eq(True)
    three_big = big & prev(big).eq(True) & prev(big, 2).eq(True)
    out["three_white_soldiers"] = three_bull & three_big & (c > p_c) & (p_c > prev(c, 2))
    out["three_black_crows"] = three_bear & three_big & (c < p_c) & (p_c < prev(c, 2))
    out["bullish_marubozu"] = bull & (body >= 0.9 * rng)
    out["bearish_marubozu"] = bear & (body >= 0.9 * rng)
    return out.fillna(False).astype(bool)


def pattern_score(patterns: pd.DataFrame) -> pd.Series:
    """বুলিশ প্যাটার্নের সংখ্যা − বেয়ারিশ প্যাটার্নের সংখ্যা।"""
    weights = pd.Series({col: d for col, (_, d) in PATTERNS.items()})
    return patterns[weights.index].astype(int).mul(weights, axis=1).sum(axis=1)


def pattern_names(row: pd.Series) -> list[str]:
    return [PATTERNS[col][0] for col in PATTERNS if row.get(col, False)]
