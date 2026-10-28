"""ফলাফল মানুষের পড়ার মতো করে লেখা — টার্মিনাল রিপোর্ট ও ড্যাশবোর্ড দুই জায়গাতেই ব্যবহার হয়।"""
from __future__ import annotations

import math

from .engine import AnalysisResult

SIGNAL_TEXT = {
    "BUY": "কিনুন (BUY)",
    "SELL": "বিক্রি করুন (SELL)",
    "HOLD": "অপেক্ষা করুন (HOLD)",
}


def fmt_price(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if abs(x) >= 1000:
        return f"{x:,.2f}"
    if abs(x) >= 1:
        return f"{x:,.4f}"
    return f"{x:.8g}"


def fmt_pct(x: float, digits: int = 1, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x * 100:{'+' if sign else ''}.{digits}f}%"


def edge_verdict(metrics: dict) -> str:
    """ব্যাকটেস্টের নির্ভুলতা কি শুধু ভাগ্য, নাকি সত্যিকারের এজ — সরল ভাষায়।"""
    acc, base, p = metrics["accuracy"], metrics["baseline_accuracy"], metrics["p_value"]
    if acc <= base:
        return "মডেল সাধারণ বেসলাইনের চেয়ে ভালো করেনি — এই সেটিংসে এর সিগন্যালের উপর ভরসা করবেন না।"
    if p < 0.05:
        return "নির্ভুলতা বেসলাইনের চেয়ে বেশি এবং পরিসংখ্যানগতভাবে তাৎপর্যপূর্ণ (p < 0.05) — তবু ভবিষ্যতে একই থাকবে এমন নিশ্চয়তা নেই।"
    return "নির্ভুলতা বেসলাইনের চেয়ে সামান্য বেশি, কিন্তু পরিসংখ্যানগতভাবে তাৎপর্যপূর্ণ নয় — এটা ভাগ্যের কারণেও হতে পারে।"


def text_report(result: AnalysisResult, source: str, symbol: str) -> str:
    p, s = result.prediction, result.settings
    arrow = "▲ উপরে" if p.prob_up >= 0.5 else "▼ নিচে"
    lines = [
        "═" * 64,
        f" {symbol}  ·  {s.interval}  ·  {source}",
        f" শেষ ক্লোজড ক্যান্ডেল: {p.time:%Y-%m-%d %H:%M} UTC   →   পূর্বাভাস: {p.target_time:%Y-%m-%d %H:%M} UTC",
        "═" * 64,
        f" বর্তমান দাম (ক্লোজ)      : {fmt_price(p.close)}",
        f" পরের {s.horizon}টি ক্যান্ডেলে দিক   : {arrow}  (উপরে যাওয়ার সম্ভাবনা {fmt_pct(p.prob_up)})",
        f" সিগন্যাল                : {SIGNAL_TEXT[p.signal]}   [থ্রেশহোল্ড {fmt_pct(s.threshold, 0)}]",
        f" আনুমানিক দাম            : {fmt_price(p.pred_price)}  ({fmt_pct(p.pred_price / p.close - 1, 2, sign=True)})",
        f" ৮০% সম্ভাব্য রেঞ্জ         : {fmt_price(p.price_low)} – {fmt_price(p.price_high)}",
        f" দুই মডেল একমত?          : {'হ্যাঁ' if p.agree else 'না — সিগন্যাল দুর্বল'}",
        f" শেষ ক্যান্ডেলের প্যাটার্ন    : {', '.join(p.patterns) if p.patterns else 'কোনো প্যাটার্ন নেই'}",
    ]
    ind = result.indicators.iloc[-1]
    lines.append(
        f" ইন্ডিকেটর               : RSI14 {ind['rsi_14']:.1f} · MACD hist {ind['macd_hist']:+.4g} · "
        f"EMA21 {fmt_price(ind['ema21'])} · EMA50 {fmt_price(ind['ema50'])}"
    )

    if result.backtest is not None:
        m = result.backtest.metrics
        lines += [
            "─" * 64,
            f" ব্যাকটেস্ট (walk-forward, {m['n_test']}টি না-দেখা ক্যান্ডেল, {m['start']:%Y-%m-%d} থেকে {m['end']:%Y-%m-%d})",
            "─" * 64,
            f" দিক সঠিক               : {fmt_pct(m['accuracy'])}   (বেসলাইন {fmt_pct(m['baseline_accuracy'])}, p = {m['p_value']:.3f})",
            f" BUY সিগন্যাল সঠিক         : {fmt_pct(m['buy_hit_rate'])}  ({m['n_buy']}টি)",
            f" SELL সিগন্যাল সঠিক        : {fmt_pct(m['sell_hit_rate'])}  ({m['n_sell']}টি)",
            f" দামের গড় ভুল (MAE)       : {fmt_price(m['price_mae'])}   (শুধু আগের দাম ধরলে: {fmt_price(m['naive_mae'])})",
            f" স্ট্র্যাটেজি রিটার্ন         : {fmt_pct(m['strategy_return'], sign=True)}   (Buy & Hold: {fmt_pct(m['buy_hold_return'], sign=True)})",
            f" সর্বোচ্চ ড্রডাউন            : {fmt_pct(m['max_drawdown'])}   (Buy & Hold: {fmt_pct(m['buy_hold_max_drawdown'])})",
            f" Sharpe · ট্রেড · মার্কেটে সময় : {m['sharpe']:.2f} · {m['n_trades']} · {fmt_pct(m['exposure'], 0)}   (ফি {fmt_pct(s.fee, 2)} ধরে)",
            f" মূল্যায়ন                 : {edge_verdict(m)}",
        ]
    lines += [
        "═" * 64,
        " ⚠ এটি পরিসংখ্যানভিত্তিক অনুমান, বিনিয়োগ পরামর্শ নয়। নিজের ঝুঁকিতে সিদ্ধান্ত নিন।",
    ]
    return "\n".join(lines)
