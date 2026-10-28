"""ওয়েব ড্যাশবোর্ড — চালাতে:  streamlit run dashboard.py

সার্ভারে চালানোর সময় এনভায়রনমেন্ট ভ্যারিয়েবল দিয়ে নিয়ন্ত্রণ করা যায়:
    APP_PASSWORD   — সেট থাকলে ড্যাশবোর্ড খোলার আগে পাসওয়ার্ড চাইবে
    MAX_CANDLES    — একজন ব্যবহারকারী সর্বোচ্চ কয়টি ক্যান্ডেল চাইতে পারবে (সার্ভারের CPU বাঁচাতে)
"""
from __future__ import annotations

import hmac
import os
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from predictor import INTERVALS, MODELS, Settings, analyze, fetch_ohlcv, load_csv
from predictor.patterns import DIRECTION_LABELS, PATTERNS, pattern_names
from predictor.report import SIGNAL_TEXT, edge_verdict, fmt_pct, fmt_price

st.set_page_config(page_title="ক্রিপ্টো ক্যান্ডেল প্রেডিক্টর", page_icon="📈", layout="wide")

APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
MAX_CANDLES = max(300, int(os.environ.get("MAX_CANDLES", "10000")))

# রং: সিরিজ ১–৩ ক্রমানুসারে, উপরে/নিচে স্ট্যাটাস রং (সবসময় আইকন/লেবেলের সাথে)
PALETTES = {
    "light": dict(s1="#2a78d6", s2="#eb6834", s3="#1baf7a", up="#0ca30c", down="#d03b3b",
                  muted="#898781", ink="#0b0b0b", ink2="#52514e", grid="#e1e0d9"),
    "dark": dict(s1="#3987e5", s2="#d95926", s3="#199e70", up="#0ca30c", down="#d03b3b",
                 muted="#898781", ink="#f0efec", ink2="#c3c2b7", grid="#2c2c2a"),
}
SIGNAL_STYLE = {"BUY": ("▲", "#0ca30c"), "SELL": ("▼", "#d03b3b"), "HOLD": ("■", "#898781")}
SOURCE_KEYS = {"Binance": "binance", "Yahoo Finance": "yahoo", "CSV ফাইল": "csv"}
# ক্রিপ্টোতে Engulfing/Harami/Marubozu খুব ঘন ঘন আসে, তাই চার্টে ডিফল্টে শুধু বড় রিভার্সাল প্যাটার্ন
DEFAULT_CHART_PATTERNS = ["hammer", "shooting_star", "morning_star", "evening_star",
                          "three_white_soldiers", "three_black_crows"]


def palette() -> dict:
    theme = getattr(st.context, "theme", None)
    return PALETTES["dark" if theme is not None and theme.type == "dark" else "light"]


@st.cache_data(ttl=60, show_spinner=False)
def get_data(source: str, symbol: str, interval: str, limit: int) -> pd.DataFrame:
    return fetch_ohlcv(source, symbol, interval, limit)


@st.cache_data(ttl=300, max_entries=50, show_spinner=False)
def run_analysis(df: pd.DataFrame, settings: dict):
    """একই ডেটা ও সেটিংসে বারবার মডেল ট্রেন না করে আগের ফল ফেরত দেয় (অনেক ব্যবহারকারী থাকলে সার্ভার হালকা থাকে)।"""
    return analyze(df, Settings(**settings))


def require_password() -> None:
    """APP_PASSWORD সেট থাকলে সঠিক পাসওয়ার্ড না দেওয়া পর্যন্ত বাকি পেজ দেখায় না।"""
    if not APP_PASSWORD or st.session_state.get("authenticated"):
        return
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.title("🔒 ক্রিপ্টো ক্যান্ডেল প্রেডিক্টর")
        with st.form("login"):
            password = st.text_input("পাসওয়ার্ড", type="password")
            submitted = st.form_submit_button("প্রবেশ করুন", type="primary", width="stretch")
        if submitted:
            if hmac.compare_digest(password.encode(), APP_PASSWORD.encode()):
                st.session_state["authenticated"] = True
                st.rerun()
            time.sleep(1)  # বারবার অনুমান করে পাসওয়ার্ড ভাঙা ধীর করতে
            st.error("ভুল পাসওয়ার্ড")
    st.stop()


def base_layout(fig: go.Figure, pal: dict, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(family='system-ui, -apple-system, "Segoe UI", "Noto Sans Bengali", sans-serif'),
    )
    fig.update_xaxes(gridcolor=pal["grid"], linecolor=pal["grid"])
    fig.update_yaxes(gridcolor=pal["grid"], linecolor=pal["grid"])
    return fig


def price_chart(res, n: int, show_patterns: list[str], pal: dict) -> go.Figure:
    df = res.df.tail(n)
    ind = res.indicators.loc[df.index]
    pats = res.patterns.loc[df.index, show_patterns] if show_patterns else None
    p = res.prediction

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.62, 0.14, 0.24], subplot_titles=("", "ভলিউম", "RSI (14)"))
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="ক্যান্ডেল",
        increasing=dict(line=dict(color=pal["up"], width=1), fillcolor=pal["up"]),
        decreasing=dict(line=dict(color=pal["down"], width=1), fillcolor=pal["down"]),
    ), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=ind["bb_upper"], name="Bollinger Band", legendgroup="bb",
                             line=dict(color=pal["muted"], width=1), hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=ind["bb_lower"], name="Bollinger Band", legendgroup="bb",
                             showlegend=False, line=dict(color=pal["muted"], width=1), hoverinfo="skip",
                             fill="tonexty", fillcolor="rgba(137,135,129,0.08)"), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=ind["ema21"], name="EMA 21",
                             line=dict(color=pal["s1"], width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=ind["ema50"], name="EMA 50",
                             line=dict(color=pal["s2"], width=2)), 1, 1)

    # প্যাটার্ন মার্কার: বুলিশ ▲ ক্যান্ডেলের নিচে, বেয়ারিশ ▼ উপরে
    offset = ind["atr_14"].fillna(0) * 0.6
    for direction, symbol, y, color, label in (
        (1, "triangle-up", df["low"] - offset, pal["up"], "বুলিশ প্যাটার্ন"),
        (-1, "triangle-down", df["high"] + offset, pal["down"], "বেয়ারিশ প্যাটার্ন"),
    ):
        cols = [c for c in show_patterns if PATTERNS[c][1] == direction]
        if not cols:
            continue
        mask = pats[cols].any(axis=1)
        if mask.any():
            names = [", ".join(pattern_names(row)) for _, row in pats.loc[mask, cols].iterrows()]
            fig.add_trace(go.Scatter(
                x=df.index[mask], y=y[mask], mode="markers", name=label, text=names,
                marker=dict(symbol=symbol, size=10, color=color, line=dict(width=1, color="white")),
                hovertemplate="%{text}<extra></extra>",
            ), 1, 1)

    # পূর্বাভাস: শেষ ক্লোজ থেকে আনুমানিক দাম, ৮০% রেঞ্জসহ
    fig.add_trace(go.Scatter(
        x=[p.time, p.target_time], y=[p.close, p.pred_price], name="পূর্বাভাস (৮০% রেঞ্জ)",
        mode="lines+markers", line=dict(color=pal["s3"], width=2, dash="dot"),
        marker=dict(symbol=["circle", "diamond"], size=[6, 12], color=pal["s3"]),
        error_y=dict(type="data", symmetric=False, array=[0, p.price_high - p.pred_price],
                     arrayminus=[0, p.pred_price - p.price_low], color=pal["s3"], thickness=2, width=8),
        hovertemplate="%{y:,.6g}<extra>পূর্বাভাস</extra>",
    ), 1, 1)
    fig.add_annotation(x=p.target_time, y=p.pred_price, text=f"<b>{fmt_price(p.pred_price)}</b>",
                       showarrow=False, xanchor="left", xshift=10, font=dict(color=pal["ink"]), row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="ভলিউম", showlegend=False,
                         marker=dict(color=pal["muted"], opacity=0.55)), 2, 1)
    fig.add_trace(go.Scatter(x=df.index, y=ind["rsi_14"], name="RSI 14", showlegend=False,
                             line=dict(color=pal["s1"], width=2)), 3, 1)
    for level in (30, 70):
        fig.add_hline(y=level, line=dict(color=pal["muted"], width=1, dash="dash"), row=3, col=1)

    base_layout(fig, pal, 780)
    fig.update_layout(xaxis_rangeslider_visible=False)
    # ডান পাশে পূর্বাভাস আর তার লেবেলের জন্য জায়গা রাখা
    pad = (p.target_time - df.index[0]) * 0.08
    fig.update_xaxes(range=[df.index[0], p.target_time + pad])
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    return fig


def equity_chart(bt, pal: dict) -> go.Figure:
    eq = bt.equity
    fig = go.Figure()
    for col, name, color in (("strategy", "মডেলের স্ট্র্যাটেজি", pal["s1"]), ("buy_hold", "Buy & Hold", pal["s2"])):
        y = (eq[col] - 1) * 100
        fig.add_trace(go.Scatter(x=eq.index, y=y, name=name, line=dict(color=color, width=2),
                                 hovertemplate="%{y:+.1f}%"))
        fig.add_annotation(x=eq.index[-1], y=y.iloc[-1], text=f"{name} {y.iloc[-1]:+.1f}%", showarrow=False,
                           xanchor="left", xshift=6, font=dict(color=pal["ink2"]))
    fig.add_hline(y=0, line=dict(color=pal["muted"], width=1))
    base_layout(fig, pal, 380)
    fig.update_layout(margin=dict(r=170), yaxis=dict(ticksuffix="%", title="মোট রিটার্ন"))
    return fig


def importance_chart(imp: pd.Series, pal: dict) -> go.Figure:
    top = imp.head(15).iloc[::-1]
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker=dict(color=pal["s1"]),
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    base_layout(fig, pal, 460)
    fig.update_layout(hovermode="closest", bargap=0.45, xaxis_title="ROC-AUC কতটা কমে")
    return fig


def signal_card(signal: str) -> str:
    icon, color = SIGNAL_STYLE[signal]
    word, _, code = SIGNAL_TEXT[signal].partition(" (")
    return f"""
    <div style="min-height:100px">
    <div style="font-size:0.875rem;opacity:0.75;margin-bottom:0.25rem">সিগন্যাল</div>
    <div style="display:inline-block;background:{color};color:#fff;border-radius:8px;
                padding:0.2rem 0.75rem;font-size:1.6rem;font-weight:700">{icon} {code.rstrip(')')}</div>
    <div style="font-size:0.875rem;margin-top:0.25rem">{word}</div>
    </div>
    """


require_password()

# ফোনে সাইডবার লুকানো থাকে — কোথায় সেটিংস তা জানিয়ে দেওয়া
user_agent = st.context.headers.get("User-Agent", "") if hasattr(st.context, "headers") else ""
if any(k in user_agent for k in ("Android", "iPhone", "iPad", "Mobile")):
    st.caption("📱 কয়েন, টাইমফ্রেম বা অন্য সেটিংস বদলাতে উপরের বাম কোণের **»** চিহ্নে চাপুন।")

# ───────────────────────────── সাইডবার ─────────────────────────────
with st.sidebar:
    st.header("⚙️ সেটিংস")
    source_label = st.radio("ডেটার উৎস", list(SOURCE_KEYS), horizontal=True)
    source = SOURCE_KEYS[source_label]
    uploaded = None
    if source == "csv":
        uploaded = st.file_uploader("CSV ফাইল (time, open, high, low, close, volume)", type="csv")
        symbol = st.text_input("নাম", "MY-DATA")
    else:
        symbol = st.text_input("সিম্বল", "BTCUSDT",
                               help="যেমন BTCUSDT, ETH/USDT, SOL-USD। Yahoo-র জন্য স্বয়ংক্রিয়ভাবে BTC-USD ফরম্যাটে বদলানো হয়।")
    interval = st.selectbox("টাইমফ্রেম", INTERVALS, index=INTERVALS.index("1h"))
    limit = st.number_input("ক্যান্ডেল সংখ্যা", min_value=300, max_value=MAX_CANDLES,
                            value=min(1000, MAX_CANDLES), step=100)
    horizon = st.number_input("কয়টি ক্যান্ডেল পরের পূর্বাভাস", min_value=1, max_value=48, value=1)

    st.subheader("মডেল")
    model = st.selectbox("অ্যালগরিদম", list(MODELS), format_func=MODELS.get)
    threshold = st.slider("সিগন্যাল থ্রেশহোল্ড", 0.51, 0.75, 0.55, 0.01,
                          help="উপরে যাওয়ার সম্ভাবনা এর সমান বা বেশি হলে BUY, (1 − এই মান) বা কম হলে SELL, মাঝে HOLD")
    with st.expander("ব্যাকটেস্ট সেটিংস"):
        do_backtest = st.checkbox("ব্যাকটেস্ট চালান", value=True)
        min_train = st.number_input("প্রথম ট্রেনিং (ক্যান্ডেল)", 100, 5000, 300, 50)
        retrain_every = st.number_input("কত ক্যান্ডেল পর পর নতুন করে ট্রেন", 5, 500, 50, 5)
        fee_pct = st.number_input("প্রতি ট্রেডে ফি (%)", 0.0, 1.0, 0.1, 0.01)
        allow_short = st.checkbox("SELL সিগন্যালে শর্ট পজিশন", value=False)
    run = st.button("▶ বিশ্লেষণ চালান", type="primary", width="stretch")

# ───────────────────────────── বিশ্লেষণ ─────────────────────────────
if run or "result" not in st.session_state:
    try:
        with st.spinner("ডেটা আনা হচ্ছে ও মডেল ট্রেন হচ্ছে…"):
            if source == "csv":
                if uploaded is None:
                    st.info("বাম পাশ থেকে একটি CSV ফাইল আপলোড করে **বিশ্লেষণ চালান** চাপুন।")
                    st.stop()
                df = load_csv(uploaded).tail(int(limit))
            else:
                df = get_data(source, symbol, interval, int(limit))
            settings = Settings(
                interval=interval, model=model, horizon=int(horizon), threshold=threshold,
                backtest=do_backtest, min_train=int(min_train), retrain_every=int(retrain_every),
                fee=fee_pct / 100, allow_short=allow_short,
            )
            st.session_state["result"] = run_analysis(df, asdict(settings))
            st.session_state["meta"] = (symbol.upper(), source_label)
    except ConnectionError as exc:
        st.error(f"ডেটা সার্ভারে সংযোগ করা যায়নি।\n\n{exc}\n\nইন্টারনেট সংযোগ দেখুন, অথবা অন্য উৎস (Yahoo/Binance) বেছে নিন।")
        st.stop()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

res = st.session_state["result"]
sym, src = st.session_state["meta"]
p, s, bt = res.prediction, res.settings, res.backtest
pal = palette()

st.title("📈 ক্রিপ্টো ক্যান্ডেল প্রেডিক্টর")
st.caption(
    f"**{sym}** · {s.interval} · {src} · {MODELS[s.model]} · শেষ ক্লোজড ক্যান্ডেল "
    f"{p.time:%Y-%m-%d %H:%M} UTC → পূর্বাভাস {p.target_time:%Y-%m-%d %H:%M} UTC ({s.horizon}টি ক্যান্ডেল পর)"
)

c1, c2, c3, c4, c5 = st.columns(5)
prev_close = res.df["close"].iloc[-2]
c1.metric("বর্তমান দাম", fmt_price(p.close), fmt_pct(p.close / prev_close - 1, 2, sign=True), border=True)
with c2.container(border=True):
    st.markdown(signal_card(p.signal), unsafe_allow_html=True)
c3.metric("উপরে যাওয়ার সম্ভাবনা", fmt_pct(p.prob_up),
          ("▲ উপরে" if p.prob_up >= 0.5 else "▼ নিচে"), delta_color="normal" if p.prob_up >= 0.5 else "inverse",
          delta_arrow="off", border=True)
c4.metric("আনুমানিক দাম", fmt_price(p.pred_price), fmt_pct(p.pred_price / p.close - 1, 2, sign=True), border=True)
if bt is not None:
    m = bt.metrics
    c5.metric("ব্যাকটেস্টে দিক সঠিক", fmt_pct(m["accuracy"]),
              f"{(m['accuracy'] - m['baseline_accuracy']) * 100:+.1f} পয়েন্ট", border=True,
              help=f"বেসলাইনের ({fmt_pct(m['baseline_accuracy'])}) তুলনায় — বেসলাইন মানে সবসময় বেশি-হওয়া দিকটা অনুমান করা")
else:
    c5.metric("ব্যাকটেস্টে দিক সঠিক", "—", border=True)

st.caption(
    f"৮০% সম্ভাব্য রেঞ্জ: **{fmt_price(p.price_low)} – {fmt_price(p.price_high)}** · "
    f"দুই মডেল একমত: **{'হ্যাঁ' if p.agree else 'না (সিগন্যাল দুর্বল)'}** · "
    f"শেষ ক্যান্ডেলের প্যাটার্ন: **{', '.join(p.patterns) if p.patterns else 'নেই'}**"
)
if bt is not None:
    (st.success if m["accuracy"] > m["baseline_accuracy"] and m["p_value"] < 0.05 else st.warning)(
        "**মূল্যায়ন:** " + edge_verdict(m)
    )

tab_chart, tab_bt, tab_pat, tab_imp, tab_data = st.tabs(
    ["📊 চার্ট", "🧪 ব্যাকটেস্ট", "🕯️ প্যাটার্ন", "🔍 ফিচারের গুরুত্ব", "📄 ডেটা"]
)

with tab_chart:
    left, right = st.columns([1, 2])
    n_show = left.slider("চার্টে কয়টি ক্যান্ডেল দেখাবে", 50, len(res.df), min(150, len(res.df)), 10)
    show_patterns = right.multiselect("চার্টে যে প্যাটার্ন দেখাবে", list(PATTERNS), DEFAULT_CHART_PATTERNS,
                                      format_func=lambda c: PATTERNS[c][0])
    st.plotly_chart(price_chart(res, n_show, show_patterns, pal), width="stretch", theme="streamlit")

with tab_bt:
    if bt is None:
        st.info("ব্যাকটেস্ট বন্ধ আছে — সাইডবারের 'ব্যাকটেস্ট সেটিংস' থেকে চালু করুন।")
    else:
        st.markdown(
            f"**{m['n_test']}টি** না-দেখা ক্যান্ডেলে ({m['start']:%Y-%m-%d} থেকে {m['end']:%Y-%m-%d}) "
            f"মডেল চালানো হয়েছে — প্রতিটি পূর্বাভাসের সময় মডেল শুধু তার আগের ডেটা দেখেছে।"
        )
        a = st.columns(5)
        a[0].metric("দিক সঠিক", fmt_pct(m["accuracy"]), border=True)
        a[1].metric("বেসলাইন", fmt_pct(m["baseline_accuracy"]), border=True,
                    help="সবসময় বেশি-হওয়া দিকটা অনুমান করলে যত শতাংশ সঠিক হতো")
        a[2].metric("p-value", f"{m['p_value']:.3f}", border=True,
                    help="০.০৫-এর কম হলে ফলাফল ভাগ্যের কারণে হওয়ার সম্ভাবনা কম")
        a[3].metric("BUY সঠিক", fmt_pct(m["buy_hit_rate"]), f"{m['n_buy']}টি সিগন্যাল", delta_color="off", delta_arrow="off", border=True)
        a[4].metric("SELL সঠিক", fmt_pct(m["sell_hit_rate"]), f"{m['n_sell']}টি সিগন্যাল", delta_color="off", delta_arrow="off", border=True)
        b = st.columns(5)
        b[0].metric("স্ট্র্যাটেজি রিটার্ন", fmt_pct(m["strategy_return"], sign=True), border=True)
        b[1].metric("Buy & Hold রিটার্ন", fmt_pct(m["buy_hold_return"], sign=True), border=True)
        b[2].metric("সর্বোচ্চ ড্রডাউন", fmt_pct(m["max_drawdown"]),
                    f"Buy & Hold {fmt_pct(m['buy_hold_max_drawdown'])}", delta_color="off", delta_arrow="off", border=True)
        b[3].metric("Sharpe রেশিও", f"{m['sharpe']:.2f}", border=True)
        b[4].metric("ট্রেড সংখ্যা", f"{m['n_trades']}", f"মার্কেটে {fmt_pct(m['exposure'], 0)} সময়",
                    delta_color="off", delta_arrow="off", border=True)
        st.caption(
            f"দামের গড় ভুল (MAE): **{fmt_price(m['price_mae'])}** — শুধু আগের ক্লোজকে পূর্বাভাস ধরলে: "
            f"**{fmt_price(m['naive_mae'])}**। মডেলের ভুল এর চেয়ে কম হলে তবেই দামের পূর্বাভাস কাজের।"
        )
        st.subheader("রিটার্নের তুলনা")
        st.plotly_chart(equity_chart(bt, pal), width="stretch", theme="streamlit")
        st.caption(f"ফি {fmt_pct(s.fee, 2)} প্রতি ট্রেড ধরে · BUY = কেনা, SELL = "
                   f"{'শর্ট' if s.allow_short else 'বিক্রি করে বের হওয়া'}, HOLD = আগের পজিশন রাখা")

        with st.expander("সংখ্যাগুলো কীভাবে পড়বেন"):
            st.markdown(
                "- **দিক সঠিক** বেসলাইনের চেয়ে বেশি না হলে মডেল আসলে কিছু শিখেনি।\n"
                "- **p-value** ০.০৫-এর কম হলে ফলাফল শুধু ভাগ্যের কারণে হওয়ার সম্ভাবনা কম।\n"
                "- **স্ট্র্যাটেজি রিটার্ন** Buy & Hold-এর চেয়ে কম হলে শুধু কিনে রেখে দেওয়াই ভালো ছিল।\n"
                "- **ড্রডাউন** = সর্বোচ্চ কত শতাংশ ক্ষতিতে নেমেছিল।\n"
                "- বিভিন্ন টাইমফ্রেম, horizon ও থ্রেশহোল্ড দিয়ে চেষ্টা করুন — তবে অনেকবার চেষ্টা করে সবচেয়ে ভালোটা "
                "বেছে নিলে সেটাও এক ধরনের overfitting।"
            )

        st.subheader("প্রতিটি পূর্বাভাস")
        table = bt.predictions[["close", "prob_up", "signal", "pred_price", "actual_price", "correct"]].copy()
        table["correct"] = table["correct"].map({1: "✔ সঠিক", 0: "✘ ভুল"})
        st.dataframe(
            table.iloc[::-1].head(300), width="stretch",
            column_config={
                "close": st.column_config.NumberColumn("ক্লোজ", format="%.6g"),
                "prob_up": st.column_config.NumberColumn("উপরে যাওয়ার সম্ভাবনা", format="percent"),
                "signal": "সিগন্যাল",
                "pred_price": st.column_config.NumberColumn("আনুমানিক দাম", format="%.6g"),
                "actual_price": st.column_config.NumberColumn("আসল দাম", format="%.6g"),
                "correct": "দিক",
            },
        )
        st.download_button("⬇ সব পূর্বাভাস CSV", bt.predictions.to_csv().encode("utf-8"),
                           file_name=f"{sym}_{s.interval}_backtest.csv", mime="text/csv")

with tab_pat:
    st.markdown(f"এই ডেটাতে প্রতিটি প্যাটার্নের পর **{s.horizon}টি ক্যান্ডেলে** আসলে কী হয়েছে:")
    ps = res.pattern_stats.copy()
    ps["direction"] = ps["direction"].map(DIRECTION_LABELS)
    st.dataframe(
        ps, width="stretch", hide_index=True,
        column_config={
            "pattern": "প্যাটার্ন",
            "direction": "প্রচলিত অর্থ",
            "count": "কতবার",
            "up_pct": st.column_config.ProgressColumn("পরে উপরে গেছে", format="percent", min_value=0, max_value=1),
            "avg_return": st.column_config.NumberColumn("গড় রিটার্ন", format="percent"),
            "last_seen": st.column_config.DatetimeColumn("শেষবার", format="YYYY-MM-DD HH:mm"),
        },
    )
    st.caption("বুলিশ প্যাটার্নের পর ৫০%-এর কাছাকাছি 'উপরে গেছে' মানে এই বাজারে প্যাটার্নটি তেমন কাজ করছে না। "
               "অল্প সংখ্যক ঘটনা (যেমন ২০-এর কম) থেকে সিদ্ধান্ত নেবেন না।")

    st.subheader("সাম্প্রতিক প্যাটার্ন")
    recent = res.patterns[res.patterns.any(axis=1)].tail(30).iloc[::-1]
    st.dataframe(
        pd.DataFrame({
            "close": res.df["close"].reindex(recent.index),
            "patterns": [", ".join(pattern_names(r)) for _, r in recent.iterrows()],
        }),
        width="stretch",
        column_config={"close": st.column_config.NumberColumn("ক্লোজ", format="%.6g"), "patterns": "প্যাটার্ন"},
    )

with tab_imp:
    imp = res.importance
    if imp is None or imp.empty:
        st.info("ফিচারের গুরুত্ব হিসাব করার মতো ডেটা নেই।")
    else:
        st.markdown("কোন তথ্য মডেলের সিদ্ধান্তে সবচেয়ে বেশি কাজে লাগছে (না-দেখা ২৫% ডেটায় মাপা):")
        st.plotly_chart(importance_chart(imp, pal), width="stretch", theme="streamlit")
        st.caption("মান ~০ বা ঋণাত্মক হলে ফিচারটি আসলে সাহায্য করছে না। ret_* = আগের রিটার্ন, pat_* = ক্যান্ডেল প্যাটার্ন, "
                   "rsi/macd/ema/bb/atr/stoch = টেকনিক্যাল ইন্ডিকেটর, volume_*/obv = ভলিউম।")
        st.dataframe(imp.rename("গুরুত্ব").to_frame(), width="stretch")

with tab_data:
    full = res.df.join(res.indicators.drop(columns=["obv"]))
    st.dataframe(full.iloc[::-1], width="stretch")
    st.download_button("⬇ ডেটা CSV", full.to_csv().encode("utf-8"),
                       file_name=f"{sym}_{s.interval}.csv", mime="text/csv")

st.divider()
st.caption("⚠ এটি অতীতের ডেটাভিত্তিক পরিসংখ্যানগত অনুমান, বিনিয়োগ পরামর্শ নয়। ক্রিপ্টো বাজার অত্যন্ত ঝুঁকিপূর্ণ — "
           "যতটুকু হারাতে পারবেন তার বেশি বিনিয়োগ করবেন না।")
