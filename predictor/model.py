"""পূর্বাভাসের মডেল: দিক (উপরে/নিচে) এর জন্য ক্লাসিফায়ার, দামের জন্য রিগ্রেসর।"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = {
    "gb": "Gradient Boosting",
    "rf": "Random Forest",
    "linear": "Logistic / Ridge Regression",
}

SIGNALS = ("BUY", "SELL", "HOLD")


def make_models(kind: str = "gb", seed: int = 42):
    """বাজারের ডেটায় নয়েজ বেশি, তাই সব মডেল ইচ্ছা করে সরল ও রেগুলারাইজড রাখা হয়েছে (overfitting কমাতে)।"""
    if kind == "gb":
        params = dict(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=30,
                      l2_regularization=1.0, random_state=seed)
        return HistGradientBoostingClassifier(**params), HistGradientBoostingRegressor(**params)
    if kind == "rf":
        params = dict(n_estimators=300, max_depth=6, min_samples_leaf=20, n_jobs=-1, random_state=seed)
        return RandomForestClassifier(**params), RandomForestRegressor(**params)
    if kind == "linear":
        return (
            make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
            make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        )
    raise ValueError(f"অজানা মডেল '{kind}' — ব্যবহারযোগ্য: {', '.join(MODELS)}")


def signal_from_prob(prob_up: float, threshold: float = 0.55) -> str:
    """উপরে যাওয়ার সম্ভাবনা ≥ threshold → BUY, ≤ 1−threshold → SELL, মাঝখানে → HOLD।"""
    if prob_up >= threshold:
        return "BUY"
    if prob_up <= 1 - threshold:
        return "SELL"
    return "HOLD"


class CandlePredictor:
    def __init__(self, kind: str = "gb", seed: int = 42):
        self.kind = kind
        self.clf, self.reg = make_models(kind, seed)
        self._const_prob: float | None = None

    def fit(self, X, y_up, y_ret) -> "CandlePredictor":
        y_up = np.asarray(y_up, dtype=int)
        if len(np.unique(y_up)) < 2:  # ট্রেনিং ডেটায় শুধু এক দিক থাকলে
            self._const_prob = float(y_up[0])
        else:
            self.clf.fit(X, y_up)
        self.reg.fit(X, np.asarray(y_ret, dtype=float))
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        """(উপরে যাওয়ার সম্ভাবনা, আনুমানিক লগ-রিটার্ন)"""
        if self._const_prob is not None:
            prob = np.full(len(X), self._const_prob)
        else:
            prob = self.clf.predict_proba(X)[:, 1]
        return prob, self.reg.predict(X)
