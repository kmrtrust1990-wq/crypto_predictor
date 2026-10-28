"""সততা পরীক্ষা: ভবিষ্যতের তথ্য ফাঁস হচ্ছে কিনা, আর এলোমেলো দামে মডেল ভান করে ভালো ফল দেখাচ্ছে কিনা।

চালাতে:  python -m pytest tests -q
"""
import numpy as np
import pandas as pd

from predictor import Settings, analyze
from predictor.features import build_features, make_dataset
from predictor.model import signal_from_prob


def random_walk(n=1200, seed=0) -> pd.DataFrame:
    """কৃত্রিম দাম — শুধু পরীক্ষার জন্য। এখানে ভবিষ্যৎ অনুমান করা সম্ভবই নয়।"""
    rng = np.random.default_rng(seed)
    close = 30000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 0.004, (2, n))) * close
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) + wick[0],
        "low": np.minimum(open_, close) - wick[1],
        "close": close,
        "volume": rng.lognormal(10, 0.5, n),
    }, index=idx)


def test_features_do_not_look_ahead():
    df = random_walk(600)
    full = build_features(df)
    cut = build_features(df.iloc[:400])
    pd.testing.assert_frame_equal(full.iloc[:400], cut)


def test_targets_are_future_returns():
    df = random_walk(300)
    data, _ = make_dataset(df, horizon=3)
    t = data.index[10]
    expected = np.log(df.loc[:, "close"].shift(-3).loc[t] / df.loc[t, "close"])
    assert np.isclose(data.loc[t, "target_return"], expected)
    assert data["target_return"].isna().sum() == 3


def test_random_walk_has_no_edge():
    for model in ("gb", "linear"):
        result = analyze(random_walk(1500, seed=1), Settings(model=model, importance=False))
        acc = result.backtest.metrics["accuracy"]
        assert 0.42 < acc < 0.58, f"{model}: এলোমেলো দামে {acc:.1%} নির্ভুলতা — সম্ভবত তথ্য ফাঁস"


def test_signal_thresholds():
    assert signal_from_prob(0.60, 0.55) == "BUY"
    assert signal_from_prob(0.40, 0.55) == "SELL"
    assert signal_from_prob(0.50, 0.55) == "HOLD"
