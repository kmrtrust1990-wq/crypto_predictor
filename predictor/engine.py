"""সব ধাপ একসাথে: ফিচার → ব্যাকটেস্ট → চূড়ান্ত মডেল → সর্বশেষ ক্যান্ডেলের পূর্বাভাস।"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .backtest import BacktestResult, walk_forward
from .data import interval_offset
from .features import make_dataset
from .indicators import add_indicators
from .model import CandlePredictor, signal_from_prob
from .patterns import PATTERNS, detect_patterns

Z80 = 1.2816  # ৮০% রেঞ্জের জন্য (দুই পাশে ১০%)


@dataclass
class Settings:
    interval: str = "1h"
    model: str = "gb"
    horizon: int = 1           # কয়টি ক্যান্ডেল পরের পূর্বাভাস
    threshold: float = 0.55    # BUY/SELL সিগন্যালের জন্য ন্যূনতম সম্ভাবনা
    backtest: bool = True
    min_train: int = 300
    retrain_every: int = 50
    max_train: int | None = 2000
    fee: float = 0.001         # প্রতি ট্রেডে ফি (0.001 = 0.1%)
    allow_short: bool = False
    importance: bool = True


@dataclass
class Prediction:
    time: pd.Timestamp         # যে ক্যান্ডেলের পর পূর্বাভাস (শেষ ক্লোজড ক্যান্ডেল)
    target_time: pd.Timestamp  # যে ক্যান্ডেলের ক্লোজ অনুমান করা হচ্ছে
    close: float
    prob_up: float
    signal: str
    pred_return: float
    pred_price: float
    price_low: float
    price_high: float
    agree: bool                # দিক-মডেল ও দাম-মডেল একই দিকে বলছে কিনা
    patterns: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    df: pd.DataFrame
    indicators: pd.DataFrame
    patterns: pd.DataFrame
    pattern_stats: pd.DataFrame
    prediction: Prediction
    backtest: BacktestResult | None
    importance: pd.Series | None
    settings: Settings


def analyze(df: pd.DataFrame, settings: Settings | None = None) -> AnalysisResult:
    s = settings or Settings()
    data, cols = make_dataset(df, s.horizon)
    labeled = data[data["target_return"].notna()]
    if len(labeled) < 150:
        raise ValueError(f"ডেটা খুব কম ({len(df)}টি ক্যান্ডেল) — অন্তত ৫০০টি ক্যান্ডেল নিন")

    bt = None
    if s.backtest:
        bt = walk_forward(
            df, data, cols, interval=s.interval, model=s.model, horizon=s.horizon,
            threshold=s.threshold, min_train=s.min_train, retrain_every=s.retrain_every,
            max_train=s.max_train, fee=s.fee, allow_short=s.allow_short,
        )

    train = labeled.tail(s.max_train) if s.max_train else labeled
    model = CandlePredictor(s.model).fit(train[cols], train["target_up"], train["target_return"])
    last = data.iloc[[-1]]
    prob, ret = model.predict(last[cols])
    prob, ret = float(prob[0]), float(ret[0])

    close = float(df["close"].iloc[-1])
    lo, hi = _price_range(df, bt, ret, s.horizon)
    pats = detect_patterns(df)
    t = df.index[-1]
    prediction = Prediction(
        time=t,
        target_time=t + interval_offset(s.interval) * s.horizon,
        close=close,
        prob_up=prob,
        signal=signal_from_prob(prob, s.threshold),
        pred_return=ret,
        pred_price=close * np.exp(ret),
        price_low=close * np.exp(lo),
        price_high=close * np.exp(hi),
        agree=(prob >= 0.5) == (ret > 0),
        patterns=[PATTERNS[c][0] for c in PATTERNS if pats[c].iloc[-1]],
    )

    importance = _importance(labeled, cols, s) if s.importance else None
    return AnalysisResult(
        df=df,
        indicators=add_indicators(df),
        patterns=pats,
        pattern_stats=pattern_stats(df, pats, s.horizon),
        prediction=prediction,
        backtest=bt,
        importance=importance,
        settings=s,
    )


def _price_range(df, bt, pred_ret, horizon) -> tuple[float, float]:
    """৮০% সম্ভাব্য রেঞ্জ (লগ-রিটার্নে)। ব্যাকটেস্ট থাকলে আসল ভুলের পরিমাপ থেকে, নাহলে সাম্প্রতিক অস্থিরতা থেকে।"""
    if bt is not None and len(bt.predictions) >= 50:
        resid = (bt.predictions["actual_return"] - bt.predictions["pred_return"]).tail(500)
        return pred_ret + resid.quantile(0.1), pred_ret + resid.quantile(0.9)
    sigma = np.log(df["close"] / df["close"].shift(horizon)).tail(300).std()
    return pred_ret - Z80 * sigma, pred_ret + Z80 * sigma


def _importance(labeled: pd.DataFrame, cols: list[str], s: Settings) -> pd.Series:
    """Permutation importance: কোনো ফিচারের মান এলোমেলো করলে না-দেখা ডেটায় ROC-AUC কতটা কমে।
    ঋণাত্মক বা ~০ মান মানে ফিচারটি আসলে কাজে লাগছে না।"""
    split = int(len(labeled) * 0.75)
    train = labeled.iloc[: split - s.horizon + 1]
    test = labeled.iloc[split:]
    if train["target_up"].nunique() < 2 or test["target_up"].nunique() < 2:
        return pd.Series(dtype=float)
    m = CandlePredictor(s.model).fit(train[cols], train["target_up"], train["target_return"])
    r = permutation_importance(
        m.clf, test[cols], test["target_up"].astype(int), scoring="roc_auc", n_repeats=5, random_state=0
    )
    return pd.Series(r.importances_mean, index=cols).sort_values(ascending=False)


def pattern_stats(df: pd.DataFrame, pats: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """এই ডেটাতে প্রতিটি প্যাটার্নের পর আসলে কী হয়েছে — প্যাটার্ন সত্যিই কাজ করে কিনা দেখার জন্য।"""
    future = df["close"].shift(-horizon) / df["close"] - 1
    rows = []
    for col, (name, direction) in PATTERNS.items():
        mask = pats[col] & future.notna()
        hits = pats.index[pats[col]]
        rows.append({
            "pattern": name,
            "direction": direction,
            "count": int(mask.sum()),
            "up_pct": (future[mask] > 0).mean() if mask.any() else np.nan,
            "avg_return": future[mask].mean() if mask.any() else np.nan,
            "last_seen": hits[-1] if len(hits) else pd.NaT,
        })
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
