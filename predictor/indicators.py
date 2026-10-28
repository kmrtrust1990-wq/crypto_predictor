"""টেকনিক্যাল ইন্ডিকেটর — প্রতিটি মান শুধু ওই সময় পর্যন্ত জানা ডেটা দিয়ে হিসাব হয় (ভবিষ্যৎ দেখে না)।"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return 100 - 100 / (1 + gain / loss)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def bollinger(close: pd.Series, period: int = 20, k: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid, mid + k * std, mid - k * std


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def stochastic(df: pd.DataFrame, period: int = 14, smooth: int = 3):
    low = df["low"].rolling(period).min()
    high = df["high"].rolling(period).max()
    k = 100 * (df["close"] - low) / (high - low).replace(0, np.nan)
    return k, k.rolling(smooth).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """চার্ট ও মডেলের জন্য সব ইন্ডিকেটর একসাথে।"""
    c = df["close"]
    out = pd.DataFrame(index=df.index)
    out["ema9"] = ema(c, 9)
    out["ema21"] = ema(c, 21)
    out["ema50"] = ema(c, 50)
    out["bb_mid"], out["bb_upper"], out["bb_lower"] = bollinger(c)
    out["rsi_14"] = rsi(c, 14)
    out["rsi_7"] = rsi(c, 7)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(c)
    out["atr_14"] = atr(df)
    out["stoch_k"], out["stoch_d"] = stochastic(df)
    out["obv"] = obv(df)
    return out
