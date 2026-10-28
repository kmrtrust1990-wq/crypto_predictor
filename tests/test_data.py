"""নেটওয়ার্ক ছাড়া ডেটা-লোডার পরীক্ষা — Binance/Yahoo-র উত্তরের ফরম্যাট নকল করে।"""
import time

import numpy as np
import pandas as pd
import pytest

from predictor import data


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code, self.text = payload, status, str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def fake_binance_klines(n_total=2500, step_ms=3_600_000):
    """এখন পর্যন্ত n_total টি 1h kline; শেষেরটি এখনো চলমান।"""
    now = int(time.time() * 1000)
    last_open = now - now % step_ms
    opens = [last_open - i * step_ms for i in range(n_total)][::-1]
    klines = [[o, "100", "101", "99", "100.5", "10", o + step_ms - 1, "0", 0, "0", "0", "0"] for o in opens]

    def get(url, params, timeout):
        rows = [k for k in klines if "endTime" not in params or k[0] <= params["endTime"]]
        return FakeResponse(rows[-params["limit"]:])

    return get, last_open


def test_binance_paginates_and_drops_open_candle(monkeypatch):
    get, last_open = fake_binance_klines()
    monkeypatch.setattr(data.requests, "get", get)
    df = data.fetch_binance("BTC/USDT", "1h", limit=1500)
    assert len(df) == 1500
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert df.index[-1] == pd.Timestamp(last_open - 3_600_000, unit="ms", tz="UTC")
    assert list(df.columns) == data.OHLCV


def test_binance_falls_back_when_blocked(monkeypatch):
    get, _ = fake_binance_klines(50)
    calls = []

    def blocked_first(url, params, timeout):
        calls.append(url)
        return FakeResponse({}, 451) if "api.binance.com" in url else get(url, params, timeout)

    monkeypatch.setattr(data.requests, "get", blocked_first)
    assert len(data.fetch_binance("ETHUSDT", "1h", limit=20)) == 20
    assert any("data-api.binance.vision" in c for c in calls)


def test_binance_bad_symbol(monkeypatch):
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: FakeResponse({"msg": "Invalid symbol."}, 400))
    with pytest.raises(ValueError, match="Invalid symbol"):
        data.fetch_binance("NOPEUSDT")


def test_yahoo_resamples_to_4h(monkeypatch):
    end = pd.Timestamp.now(tz="UTC").floor("1h")
    idx = pd.date_range(end=end, periods=400, freq="1h")
    hourly = pd.DataFrame({
        "Open": np.arange(400.0), "High": np.arange(400.0) + 1, "Low": np.arange(400.0) - 1,
        "Close": np.arange(400.0) + 0.5, "Volume": np.ones(400),
    }, index=idx)

    class FakeTicker:
        def __init__(self, ticker):
            assert ticker == "BTC-USD"

        def history(self, **kwargs):
            assert kwargs["interval"] == "1h"
            return hourly

    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
    df = data.fetch_yahoo("BTCUSDT", "4h", limit=50)
    assert len(df) == 50
    assert (df.index.hour % 4 == 0).all()
    assert (df["volume"] == 4).all()  # প্রতিটি 4h ক্যান্ডেলে চারটি পূর্ণ 1h ক্যান্ডেল
    assert df.index[-1] + pd.Timedelta(hours=4) <= pd.Timestamp.now(tz="UTC")


def test_symbol_mapping():
    assert data.to_binance_symbol("btc/usdt") == "BTCUSDT"
    assert data.to_binance_symbol("ETH-USD") == "ETHUSDT"
    assert data.to_yahoo_symbol("SOLFDUSD") == "SOL-USD"
    assert data.to_yahoo_symbol("ETHBTC") == "ETH-BTC"
