"""Binance, Yahoo Finance বা CSV থেকে OHLCV ক্যান্ডেল ডেটা আনা।

সব ফাংশন একই ফরম্যাটে DataFrame ফেরত দেয়:
index = ক্যান্ডেল খোলার সময় (UTC), কলাম = open, high, low, close, volume।
শুধু সম্পূর্ণ (closed) ক্যান্ডেল রাখা হয় — এখনো চলমান ক্যান্ডেল বাদ যায়,
কারণ অসম্পূর্ণ ক্যান্ডেলের উপর পূর্বাভাস দিলে ফলাফল ভুল হয়।
"""
from __future__ import annotations

import time

import pandas as pd
import requests

OHLCV = ["open", "high", "low", "close", "volume"]

# Binance-এর টাইমফ্রেম → সেকেন্ড
INTERVAL_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800, "12h": 43200,
    "1d": 86400, "3d": 259200, "1w": 604800, "1M": 2592000,
}
INTERVALS = list(INTERVAL_SECONDS)

SOURCES = {"binance": "Binance", "yahoo": "Yahoo Finance"}

# api.binance.com কিছু দেশে (যেমন যুক্তরাষ্ট্র) ব্লক থাকে, তখন পাবলিক মার্কেট-ডেটা মিরর ব্যবহার হয়
BINANCE_BASE_URLS = ("https://api.binance.com", "https://data-api.binance.vision")

# লম্বা কোট আগে মেলাতে হবে (FDUSD বনাম USD, USDT বনাম USD)
_QUOTES = ("FDUSD", "USDT", "USDC", "BUSD", "TUSD", "USD", "EUR", "BTC", "ETH", "BNB")
_USD_LIKE = {"USDT", "USDC", "FDUSD", "BUSD", "TUSD", "USD"}

# Binance টাইমফ্রেম → (Yahoo-র নিজস্ব টাইমফ্রেম, রিস্যাম্পল নিয়ম অথবা None)
_YAHOO_MAP = {
    "1m": ("1m", None), "3m": ("1m", "3min"), "5m": ("5m", None), "15m": ("15m", None),
    "30m": ("30m", None), "1h": ("1h", None), "2h": ("1h", "2h"), "4h": ("1h", "4h"),
    "6h": ("1h", "6h"), "8h": ("1h", "8h"), "12h": ("1h", "12h"), "1d": ("1d", None),
    "3d": ("1d", "3D"), "1w": ("1wk", None), "1M": ("1mo", None),
}
# Yahoo ইন্ট্রাডে ডেটা কত দিন পিছন পর্যন্ত দেয় (নিরাপদ সীমা)
_YAHOO_MAX_DAYS = {"1m": 7, "5m": 59, "15m": 59, "30m": 59, "1h": 729}


def split_symbol(symbol: str) -> tuple[str, str]:
    """'BTC/USDT', 'BTC-USD', 'BTCUSDT' → ('BTC', 'USDT')"""
    s = symbol.upper().replace(" ", "")
    for sep in ("/", "-", "_"):
        if sep in s:
            base, quote = s.split(sep, 1)
            return base, quote
    for quote in _QUOTES:
        if s.endswith(quote) and len(s) > len(quote):
            return s[: -len(quote)], quote
    return s, "USDT"


def to_binance_symbol(symbol: str) -> str:
    base, quote = split_symbol(symbol)
    if quote == "USD":  # Binance.com-এ USD জোড়া নেই
        quote = "USDT"
    return base + quote


def to_yahoo_symbol(symbol: str) -> str:
    base, quote = split_symbol(symbol)
    if quote in _USD_LIKE:
        quote = "USD"
    return f"{base}-{quote}"


def interval_offset(interval: str):
    if interval == "1M":
        return pd.DateOffset(months=1)
    return pd.Timedelta(seconds=INTERVAL_SECONDS[interval])


def _check_interval(interval: str) -> None:
    if interval not in INTERVAL_SECONDS:
        raise ValueError(f"অজানা টাইমফ্রেম '{interval}'। ব্যবহারযোগ্য: {', '.join(INTERVALS)}")


def _drop_incomplete(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """যে ক্যান্ডেল এখনো বন্ধ হয়নি সেটা বাদ দেয়।"""
    now = pd.Timestamp.now(tz="UTC")
    closes_at = df.index + interval_offset(interval)
    return df[closes_at <= now]


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df[OHLCV].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "time"
    return df.dropna()


def _binance_get(path: str, params: dict, timeout: float = 15):
    last_error: Exception | None = None
    for base in BINANCE_BASE_URLS:
        try:
            resp = requests.get(base + path, params=params, timeout=timeout)
        except requests.RequestException as exc:
            last_error = exc
            continue
        if resp.status_code == 400:
            try:
                msg = resp.json().get("msg", resp.text)
            except ValueError:
                msg = resp.text
            raise ValueError(f"Binance অনুরোধটি বাতিল করেছে: {msg}")
        if resp.status_code in (403, 451):  # দেশভিত্তিক ব্লক — পরের URL চেষ্টা
            last_error = RuntimeError(f"{base} → HTTP {resp.status_code}")
            continue
        resp.raise_for_status()
        return resp.json()
    raise ConnectionError(f"Binance-এ সংযোগ করা যায়নি: {last_error}")


def fetch_binance(symbol: str, interval: str = "1h", limit: int = 1000) -> pd.DataFrame:
    """Binance-এর পাবলিক API থেকে ক্যান্ডেল আনে (API key লাগে না)। ১০০০-এর বেশি হলে পেজ করে আনে।"""
    _check_interval(interval)
    pair = to_binance_symbol(symbol)
    want = limit + 1  # চলমান ক্যান্ডেলটি পরে বাদ যাবে
    rows: list = []
    end_time = None
    while len(rows) < want:
        params = {"symbol": pair, "interval": interval, "limit": min(1000, want - len(rows))}
        if end_time is not None:
            params["endTime"] = end_time
        batch = _binance_get("/api/v3/klines", params)
        if not batch:
            break
        rows = batch + rows
        end_time = batch[0][0] - 1
        if len(batch) < params["limit"]:
            break
    if not rows:
        raise ValueError(f"Binance-এ {pair} এর কোনো ডেটা পাওয়া যায়নি")

    df = pd.DataFrame([r[:7] for r in rows], columns=["open_time", *OHLCV, "close_time"])
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[df["close_time"].astype("int64") < time.time() * 1000]
    return _finalize(df).tail(limit)


def fetch_yahoo(symbol: str, interval: str = "1h", limit: int = 1000) -> pd.DataFrame:
    """Yahoo Finance থেকে ক্যান্ডেল আনে। Yahoo-তে না থাকা টাইমফ্রেম (যেমন 4h) ছোট ক্যান্ডেল জোড়া দিয়ে বানানো হয়।"""
    import yfinance as yf

    _check_interval(interval)
    ticker = to_yahoo_symbol(symbol)
    yf_interval, rule = _YAHOO_MAP[interval]

    days = (limit + 2) * INTERVAL_SECONDS[interval] / 86400 * 1.1 + 1
    max_days = _YAHOO_MAX_DAYS.get(yf_interval)
    if max_days is not None and days > max_days:
        days = max_days
    start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)

    try:
        raw = yf.Ticker(ticker).history(
            start=start.to_pydatetime(), interval=yf_interval, auto_adjust=False, actions=False
        )
    except Exception as exc:  # yfinance নেটওয়ার্ক/রেট-লিমিটের জন্য বিভিন্ন ধরনের এক্সেপশন দেয়
        raise ConnectionError(f"Yahoo Finance-এ সংযোগ করা যায়নি: {exc}") from exc
    if raw is None or raw.empty:
        raise ValueError(f"Yahoo Finance-এ '{ticker}' এর {interval} ডেটা পাওয়া যায়নি")

    df = raw.rename(columns=str.lower)
    df.index = pd.to_datetime(df.index, utc=True)
    if rule:
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        df = df[OHLCV].resample(rule, origin="epoch", label="left", closed="left").agg(agg)
    df = _drop_incomplete(_finalize(df), interval)
    return df.tail(limit)


def load_csv(path_or_buffer) -> pd.DataFrame:
    """নিজের CSV ফাইল লোড করে। দরকারি কলাম: time/timestamp/date, open, high, low, close, volume।"""
    raw = pd.read_csv(path_or_buffer)
    cols = {str(c).lower().strip(): c for c in raw.columns}
    time_col = next(
        (cols[c] for c in ("time", "timestamp", "date", "datetime", "open_time", "open time") if c in cols),
        None,
    )
    if time_col is None and str(raw.columns[0]).lower().startswith("unnamed"):
        time_col = raw.columns[0]  # pandas-এর to_csv() নামহীন index কলাম লেখে
    if time_col is None:
        raise ValueError("CSV-তে সময়ের কলাম পাওয়া যায়নি (time / timestamp / date নামে থাকতে হবে)")
    missing = [c for c in OHLCV if c not in cols]
    if missing:
        raise ValueError(f"CSV-তে এই কলামগুলো নেই: {', '.join(missing)}")

    t = raw[time_col]
    if pd.api.types.is_numeric_dtype(t):
        index = pd.to_datetime(t, unit="ms" if t.max() > 1e11 else "s", utc=True)
    else:
        index = pd.to_datetime(t, utc=True)
    df = pd.DataFrame({c: pd.to_numeric(raw[cols[c]], errors="coerce") for c in OHLCV})
    df.index = pd.DatetimeIndex(index)
    return _finalize(df)


def fetch_ohlcv(source: str, symbol: str, interval: str = "1h", limit: int = 1000) -> pd.DataFrame:
    source = source.lower()
    if source == "binance":
        return fetch_binance(symbol, interval, limit)
    if source in ("yahoo", "yfinance"):
        return fetch_yahoo(symbol, interval, limit)
    raise ValueError(f"অজানা ডেটা উৎস '{source}' — binance অথবা yahoo দিন")
