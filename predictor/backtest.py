"""Walk-forward ব্যাকটেস্ট — মডেল যেভাবে বাস্তবে চলবে ঠিক সেভাবে অতীতে চালিয়ে দেখা।

প্রতিটি টেস্ট ক্যান্ডেল t-এর জন্য মডেলকে শুধু সেই সারিগুলো দিয়ে ট্রেন করা হয় যাদের ফলাফল
t সময়ে জানা ছিল (সারি j-এর টার্গেট জানা যায় j+horizon-এ, তাই j ≤ t−horizon)।
এতে ভবিষ্যতের তথ্য ফাঁস (look-ahead leakage) হয় না।
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .data import INTERVAL_SECONDS
from .model import CandlePredictor, signal_from_prob


@dataclass
class BacktestResult:
    predictions: pd.DataFrame  # প্রতিটি টেস্ট ক্যান্ডেলের out-of-sample পূর্বাভাস ও আসল ফলাফল
    equity: pd.DataFrame       # strategy ও buy_hold — ১ থেকে শুরু
    metrics: dict


def walk_forward(
    df: pd.DataFrame,
    data: pd.DataFrame,
    feature_cols: list[str],
    *,
    interval: str = "1h",
    model: str = "gb",
    horizon: int = 1,
    threshold: float = 0.55,
    min_train: int = 300,
    retrain_every: int = 50,
    max_train: int | None = 2000,
    fee: float = 0.001,
    allow_short: bool = False,
) -> BacktestResult:
    labeled = data["target_return"].notna().to_numpy()
    n_labeled = int(labeled.sum())
    assert labeled[:n_labeled].all(), "লেবেলযুক্ত সারিগুলো ধারাবাহিক হওয়ার কথা"

    first = min_train + horizon
    if n_labeled - first < 30:
        raise ValueError(
            f"ব্যাকটেস্টের জন্য ডেটা কম: {n_labeled}টি ব্যবহারযোগ্য ক্যান্ডেল, "
            f"অন্তত {first + 30}টি দরকার — বেশি ক্যান্ডেল নিন বা min_train কমান"
        )

    X = data[feature_cols].to_numpy()
    y_up = data["target_up"].to_numpy()
    y_ret = data["target_return"].to_numpy()

    probs, rets = np.empty(0), np.empty(0)
    for start in range(first, n_labeled, retrain_every):
        stop = min(start + retrain_every, n_labeled)
        train_stop = start - horizon + 1
        train_start = max(0, train_stop - max_train) if max_train else 0
        m = CandlePredictor(model).fit(
            X[train_start:train_stop], y_up[train_start:train_stop], y_ret[train_start:train_stop]
        )
        p, r = m.predict(X[start:stop])
        probs, rets = np.concatenate([probs, p]), np.concatenate([rets, r])

    idx = data.index[first:n_labeled]
    close = df["close"].reindex(idx)
    pred = pd.DataFrame(index=idx)
    pred["close"] = close
    pred["prob_up"] = probs
    pred["signal"] = [signal_from_prob(p, threshold) for p in probs]
    pred["pred_return"] = rets
    pred["pred_price"] = close * np.exp(rets)
    pred["actual_return"] = y_ret[first:n_labeled]
    pred["actual_price"] = close * np.exp(pred["actual_return"])
    pred["actual_up"] = y_up[first:n_labeled].astype(int)
    pred["pred_up"] = (pred["prob_up"] >= 0.5).astype(int)
    pred["correct"] = (pred["pred_up"] == pred["actual_up"]).astype(int)

    # ট্রেডিং সিমুলেশন: BUY → লং, SELL → শর্ট (অনুমতি থাকলে) নাহলে বের হয়ে যাওয়া, HOLD → আগের পজিশন রাখা
    sell_pos = -1.0 if allow_short else 0.0
    pos = pred["signal"].map({"BUY": 1.0, "SELL": sell_pos, "HOLD": np.nan}).ffill().fillna(0.0)
    next_ret = (df["close"].shift(-1) / df["close"] - 1).reindex(idx).fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    strat_ret = pos * next_ret - fee * turnover
    equity = pd.DataFrame({
        "strategy": (1 + strat_ret).cumprod(),
        "buy_hold": (1 + next_ret).cumprod(),
        "position": pos,
    })

    metrics = _metrics(pred, equity, strat_ret, turnover, interval, horizon)
    return BacktestResult(pred, equity, metrics)


def _metrics(pred, equity, strat_ret, turnover, interval, horizon) -> dict:
    n = len(pred)
    up_rate = pred["actual_up"].mean()
    baseline = max(up_rate, 1 - up_rate)

    # horizon > 1 হলে পাশাপাশি টার্গেট একে অপরের উপর পড়ে (overlap), তাই তাৎপর্য পরীক্ষায়
    # শুধু প্রতি horizon-তম সারি নেওয়া হয় — নাহলে p-value ভুলভাবে ছোট আসবে
    indep = pred.iloc[::horizon]
    p_value = binomtest(int(indep["correct"].sum()), len(indep), baseline, alternative="greater").pvalue

    buys = pred[pred["signal"] == "BUY"]
    sells = pred[pred["signal"] == "SELL"]
    mae = (pred["pred_price"] - pred["actual_price"]).abs().mean()
    naive_mae = (pred["close"] - pred["actual_price"]).abs().mean()

    eq = equity["strategy"]
    periods_per_year = 365 * 86400 / INTERVAL_SECONDS[interval]
    std = strat_ret.std()
    return {
        "n_test": n,
        "start": pred.index[0],
        "end": pred.index[-1],
        "accuracy": pred["correct"].mean(),
        "baseline_accuracy": baseline,
        "up_rate": up_rate,
        "p_value": p_value,
        "n_buy": len(buys),
        "buy_hit_rate": buys["actual_up"].mean() if len(buys) else np.nan,
        "n_sell": len(sells),
        "sell_hit_rate": (1 - sells["actual_up"]).mean() if len(sells) else np.nan,
        "n_hold": int((pred["signal"] == "HOLD").sum()),
        "price_mae": mae,
        "naive_mae": naive_mae,
        "price_mape": ((pred["pred_price"] - pred["actual_price"]).abs() / pred["actual_price"]).mean(),
        "strategy_return": eq.iloc[-1] - 1,
        "buy_hold_return": equity["buy_hold"].iloc[-1] - 1,
        "max_drawdown": (eq / eq.cummax() - 1).min(),
        "buy_hold_max_drawdown": (equity["buy_hold"] / equity["buy_hold"].cummax() - 1).min(),
        "sharpe": strat_ret.mean() / std * np.sqrt(periods_per_year) if std > 0 else 0.0,
        "n_trades": int((turnover > 0).sum()),
        "exposure": (equity["position"] != 0).mean(),
    }
