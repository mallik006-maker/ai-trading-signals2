import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from strategy_engine import (
    MARKET_WATCHLISTS,
    fetch_1h_data,
    fetch_daily_data,
    fetch_ltf_data,
    run_strategy,
)

st.set_page_config(page_title="AI Live Signals | 1-5 Multi-Chart Terminal", layout="wide")

# Prevent screen dimming during auto-refresh
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.0rem; }
    [data-stale="true"] { opacity: 1 !important; transition: none !important; }
    .stApp > header { background-color: transparent; }
    .sig-buy { background: rgba(0, 200, 150, 0.14); border: 1px solid #00c896; padding: 12px; border-radius: 10px; margin-bottom: 8px; }
    .sig-sell { background: rgba(255, 75, 75, 0.14); border: 1px solid #ff4b4b; padding: 12px; border-radius: 10px; margin-bottom: 8px; }
    .sig-wait { background: rgba(140, 150, 170, 0.10); border: 1px solid #555; padding: 12px; border-radius: 10px; margin-bottom: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "sent_alerts" not in st.session_state:
    st.session_state["sent_alerts"] = set()


def _get_secret(key: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(key, default))
    except Exception:
        return default


@st.cache_data(ttl=20, show_spinner=False)
def cached_ltf(symbol: str, timeframe: str, tz_name: str):
    return fetch_ltf_data(symbol, timeframe, tz_name)


@st.cache_data(ttl=300, show_spinner=False)
def cached_1h(symbol: str, tz_name: str):
    return fetch_1h_data(symbol, tz_name)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_daily(symbol: str, tz_name: str):
    return fetch_daily_data(symbol, tz_name)


def load_and_run(symbol: str, timeframe: str, config: dict):
    tz_name = config["timezone"]
    d_ltf = cached_ltf(symbol, timeframe, tz_name)
    d_1h = cached_1h(symbol, tz_name)
    d_day = cached_daily(symbol, tz_name)
    return run_strategy(d_ltf, d_1h, d_day, config)


def send_telegram_alert(token: str, chat_id: str, msg: str):
    try:
        url = f"https://api.telegram.org/bot{token.strip()}/sendMessage"
        resp = requests.post(
            url,
            json={"chat_id": chat_id.strip(), "text": msg, "parse_mode": "HTML"},
            timeout=8,
        )
        return resp.status_code == 200, resp.text
    except Exception as e:
        return False, str(e)


def build_chart(res: dict, title: str, bars_to_show: int, chart_height: int = 460):
    df = res["df"]
    sigs = res["signals"]
    view_start = max(0, len(df) - bars_to_show)
    df_view = df.iloc[view_start:]
    x_vals = list(range(view_start, len(df)))
    x_labels = [t.strftime("%m-%d %H:%M") for t in df_view.index]

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=x_vals,
            open=df_view["Open"],
            high=df_view["High"],
            low=df_view["Low"],
            close=df_view["Close"],
            name="Price",
            increasing_line_color="#00c896",
            decreasing_line_color="#ff4b4b",
        )
    )
    fig.add_trace(go.Scatter(x=x_vals, y=df_view["PP"], name="PP", line=dict(color="gold", width=1.4, dash="dot")))
    fig.add_trace(go.Scatter(x=x_vals, y=df_view["R1"], name="R1", line=dict(color="rgba(255,90,90,0.5)", width=1)))
    fig.add_trace(go.Scatter(x=x_vals, y=df_view["S1"], name="S1", line=dict(color="rgba(0,200,150,0.5)", width=1)))

    def add_zones(zones, fill_color, border_color):
        for z in zones:
            if z["end"] < view_start:
                continue
            x0 = max(z["start"], view_start)
            x1 = z["end"]
            fig.add_shape(
                type="rect", x0=x0, x1=x1, y0=z["bot"], y1=z["top"],
                fillcolor=fill_color, line=dict(color=border_color, width=1), layer="below",
            )
            fig.add_annotation(
                x=x1, y=(z["top"] + z["bot"]) / 2, text=z["label"],
                showarrow=False, xanchor="right", font=dict(size=9, color=border_color),
            )

    add_zones(res["active_dem"], "rgba(0, 180, 160, 0.16)", "rgba(0, 220, 190, 0.85)")
    add_zones(res["active_sup"], "rgba(255, 75, 75, 0.16)", "rgba(255, 90, 90, 0.85)")
    add_zones(res["active_htf_ssl"], "rgba(40, 140, 255, 0.16)", "rgba(80, 175, 255, 0.85)")
    add_zones(res["active_htf_bsl"], "rgba(180, 70, 255, 0.16)", "rgba(200, 100, 255, 0.85)")

    for tl, col, name in [(res["active_up_tl"], "#80ff00", "Uptrend"), (res["active_dn_tl"], "#ff9900", "Downtrend")]:
        if tl is not None and tl["end"] >= view_start:
            x1_clamped = max(tl["x1"], view_start)
            slope = (tl["y2"] - tl["y1"]) / (tl["x2"] - tl["x1"]) if tl["x2"] != tl["x1"] else 0.0
            y1_clamped = tl["y1"] + slope * (x1_clamped - tl["x1"])
            y2_end = tl["y1"] + slope * (tl["end"] - tl["x1"])
            fig.add_trace(go.Scatter(x=[x1_clamped, tl["end"]], y=[y1_clamped, y2_end], mode="lines", name=name, line=dict(color=col, width=2)))

    buy_sigs = [s for s in sigs if s["side"] == "BUY" and s["idx"] >= view_start]
    sell_sigs = [s for s in sigs if s["side"] == "SELL" and s["idx"] >= view_start]

    if buy_sigs:
        fig.add_trace(go.Scatter(
            x=[s["idx"] for s in buy_sigs], y=[df["Low"].iloc[s["idx"]] * 0.9985 for s in buy_sigs],
            mode="markers+text", name="BUY", text=[f"BUY ({s['ai_score']:.0f}%)" for s in buy_sigs],
            textposition="bottom center", marker=dict(symbol="triangle-up", size=11, color="#00ffaa"), textfont=dict(size=9, color="#00ffaa"),
        ))
    if sell_sigs:
        fig.add_trace(go.Scatter(
            x=[s["idx"] for s in sell_sigs], y=[df["High"].iloc[s["idx"]] * 1.0015 for s in sell_sigs],
            mode="markers+text", name="SELL", text=[f"SELL ({s['ai_score']:.0f}%)" for s in sell_sigs],
            textposition="top center", marker=dict(symbol="triangle-down", size=11, color="#ff4b4b"), textfont=dict(size=9, color="#ff4b4b"),
        ))

    tick_step = max(1, len(x_vals) // 8)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        template="plotly_dark",
        height=chart_height,
        xaxis_rangeslider_visible=False,
        margin=dict(l=8, r=8, t=38, b=8),
        xaxis=dict(tickvals=x_vals[::tick_step], ticktext=x_labels[::tick_step]),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font=dict(size=10)),
    )
    return fig


# ==========================================
# SIDEBAR: SEPARATE INSTRUMENT SELECTION (1 TO 5)
# ==========================================
st.sidebar.title("1-5 Instrument Terminal")
st.sidebar.markdown("**Select Instruments (Max 5)**")

sel_crypto = st.sidebar.multiselect(
    "Crypto",
    list(MARKET_WATCHLISTS["Crypto"].keys()),
    default=["BTC/USD"],
)
sel_forex = st.sidebar.multiselect(
    "Forex & Gold",
    list(MARKET_WATCHLISTS["Forex & Gold"].keys()),
    default=["EUR/USD"],
)
sel_india = st.sidebar.multiselect(
    "Indian Markets (NSE)",
    list(MARKET_WATCHLISTS["Indian Markets (NSE)"].keys()),
    default=["NIFTY 50"],
)
custom_raw = st.sidebar.text_input(
    "Custom Tickers (comma-separated)",
    value="",
    placeholder="e.g. TATAPOWER.NS, SOL-USD",
)

selected_items = []
for name in sel_crypto:
    selected_items.append((name, MARKET_WATCHLISTS["Crypto"][name]))
for name in sel_forex:
    selected_items.append((name, MARKET_WATCHLISTS["Forex & Gold"][name]))
for name in sel_india:
    selected_items.append((name, MARKET_WATCHLISTS["Indian Markets (NSE)"][name]))

if custom_raw.strip():
    for tok in custom_raw.split(","):
        sym = tok.strip().upper()
        if sym:
            selected_items.append((sym, sym))

if not selected_items:
    selected_items = [("BTC/USD", "BTC-USD")]
    st.sidebar.info("Defaulted to BTC/USD. Select up to 5 instruments above.")
elif len(selected_items) > 5:
    st.sidebar.warning(f"You picked {len(selected_items)} instruments. Showing first 5 to keep updates instant.")
    selected_items = selected_items[:5]

st.sidebar.divider()
timeframe = st.sidebar.selectbox("Timeframe", ["5m", "15m", "30m", "1h"], index=1)
tz_choice = st.sidebar.selectbox("Timezone", ["Asia/Kolkata", "UTC"], index=0)
auto_refresh_choice = st.sidebar.selectbox("Auto-Refresh", ["Off", "30s", "60s", "5m"], index=0)
layout_mode = st.sidebar.radio("Chart Layout", ["Stacked (Best for Mobile)", "2-Column Grid (Desktop)"], index=0)
fresh_bars_limit = st.sidebar.slider("Fresh Alert Window (Candles)", 1, 15, 5)
bars_to_show = st.sidebar.slider("Visible Candles per Chart", 60, 250, 120, 20)

with st.sidebar.expander("Strategy & AI Settings", expanded=False):
    entry_mode = st.selectbox("Entry Mode", ["All Signals (Sweeps + Zones + TL)", "HTF / Daily Zone Confluence Only"], index=0)
    rr_ratio = st.number_input("Risk : Reward Ratio", 1.0, 5.0, 2.0, 0.25)
    sd_impulse = st.number_input("Daily S/D Impulse (x ATR)", 0.3, 2.0, 0.8, 0.1)
    tl_in_signals = st.checkbox("Trade Trendline Bounces & Breaks", value=True)
    use_pp_bias = st.checkbox("Filter by Daily PP Bias", value=False)
    use_ai = st.checkbox("Enable k-NN Self-Learning AI", value=True)
    use_ai_filter = st.checkbox("Filter by Min AI Confidence %", value=False)
    min_ai_conf = st.slider("Min AI Confidence %", 0, 90, 45, 5)
    ai_knn = st.slider("k-NN Neighbors", 3, 15, 5)

with st.sidebar.expander("Telegram Bot Alerts", expanded=True):
    default_tok = _get_secret("TG_BOT_TOKEN", "")
    default_cid = _get_secret("TG_CHAT_ID", "")
    tg_token = st.text_input("Telegram Bot Token", value=default_tok, type="password", placeholder="123456:ABC-DEF...")
    tg_chat_id = st.text_input("Telegram Chat ID", value=default_cid, placeholder="e.g. 987654321")
    auto_tg = st.checkbox("Auto-Send Fresh Alerts", value=bool(default_tok and default_cid))
    if st.button("Send Test Telegram Alert", width="stretch"):
        if tg_token and tg_chat_id:
            ok, info = send_telegram_alert(tg_token, tg_chat_id, "✅ <b>Test Alert:</b> Your AI Trading Signals App is connected!")
            if ok:
                st.success("Test message sent to Telegram!")
            else:
                st.error(f"Telegram error: {info}")
        else:
            st.warning("Enter both Bot Token and Chat ID first.")

config = {
    "timezone": tz_choice,
    "atr_len": 14,
    "sd_impulse_mult": sd_impulse,
    "max_sd_zones": 5,
    "htf_swing_len": 3,
    "max_htf_pools": 4,
    "tl_swing_len": 10,
    "tl_touch_atr": 0.30,
    "tl_in_signals": tl_in_signals,
    "swing_len": 5,
    "pin_wick_ratio": 0.50,
    "sweep_window": 2,
    "entry_mode": entry_mode,
    "use_pp_bias": use_pp_bias,
    "use_ai": use_ai,
    "use_ai_filter": use_ai_filter,
    "min_ai_conf": float(min_ai_conf),
    "ai_knn": ai_knn,
    "ai_max_memory": 50,
    "rr_ratio": float(rr_ratio),
    "sl_buffer_atr": 0.25,
}

run_every_val = None if auto_refresh_choice == "Off" else auto_refresh_choice

# ==========================================
# HEADER & MANUAL REFRESH
# ==========================================
h1, h2 = st.columns([4, 1])
with h1:
    st.title("AI Live Trading Signals (Multi-Chart)")
    st.caption(f"Tracking {len(selected_items)}/5 Instruments  |  Timeframe: {timeframe}  |  Auto-Refresh: {auto_refresh_choice}")
with h2:
    st.write("")
    if st.button("🔄 Refresh Now", type="primary", width="stretch"):
        st.cache_data.clear()
        st.rerun()


@st.fragment(run_every=run_every_val)
def live_multi_chart_fragment():
    now_str = pd.Timestamp.now(tz=tz_choice).strftime("%d %b %Y, %H:%M:%S")
    st.caption(f"Last Updated ({tz_choice}): **{now_str}**")

    results = []
    summary_rows = []

    for nm, sym in selected_items:
        try:
            r = load_and_run(sym, timeframe, config)
            if r is None:
                continue
            stt = r["stats"]
            sigs = r["signals"]
            latest = sigs[-1] if sigs else None
            last_c_str = stt["last_time"].strftime("%d %b %H:%M")
            badge = f"🟢 LIVE ({stt['mins_ago']}m)" if stt["market_live"] else f"🔴 CLOSED ({last_c_str})"

            results.append((nm, sym, r, badge, last_c_str))

            if latest is not None:
                bars_ago = int(latest["bars_ago"])
                is_fresh = bars_ago <= fresh_bars_limit
                sig_label = f"🔥 FRESH {latest['side']}" if is_fresh else latest["side"]
                summary_rows.append({
                    "Instrument": f"{nm} ({sym})",
                    "Status": badge,
                    "Last Candle": last_c_str,
                    "Live Price": round(stt["last_price"], 4 if stt["last_price"] < 200 else 2),
                    "Signal": sig_label,
                    "Bars Ago": bars_ago,
                    "Alert Time": latest["time"].strftime("%d %b %H:%M"),
                    "AI %": round(latest["ai_score"], 1),
                    "Entry": round(latest["price"], 4 if latest["price"] < 200 else 2),
                    "SL": round(latest["sl"], 4 if latest["sl"] < 200 else 2),
                    "TP": round(latest["tp"], 4 if latest["tp"] < 200 else 2),
                    "Zone": stt["zone_status"],
                    "Setup": latest["reasons"],
                })

                # Auto-send Telegram alert on fresh signals
                alert_key = f"{sym}_{timeframe}_{latest['side']}_{latest['time']}"
                if auto_tg and tg_token and tg_chat_id and bars_ago <= 1 and alert_key not in st.session_state["sent_alerts"]:
                    tg_msg = (
                        f"🚨 <b>LIVE {latest['side']} ALERT: {nm} ({sym}) [{timeframe}]</b>\n"
                        f"Entry: {latest['price']:,.4f}\nSL: {latest['sl']:,.4f}\nTP: {latest['tp']:,.4f}\n"
                        f"AI Confidence: {latest['ai_score']:.1f}%\nSetup: {latest['reasons']}\n"
                        f"Time ({tz_choice}): {latest['time'].strftime('%d %b %H:%M')}"
                    )
                    ok, _ = send_telegram_alert(tg_token, tg_chat_id, tg_msg)
                    if ok:
                        st.session_state["sent_alerts"].add(alert_key)
            else:
                summary_rows.append({
                    "Instrument": f"{nm} ({sym})",
                    "Status": badge,
                    "Last Candle": last_c_str,
                    "Live Price": round(stt["last_price"], 4 if stt["last_price"] < 200 else 2),
                    "Signal": "WAIT",
                    "Bars Ago": 9999,
                    "Alert Time": "-",
                    "AI %": round(max(stt["bull_ai"], stt["bear_ai"]), 1),
                    "Entry": None, "SL": None, "TP": None,
                    "Zone": stt["zone_status"],
                    "Setup": "-",
                })
        except Exception:
            pass

    if summary_rows:
        st.subheader("Selected Instruments — Live Signals Summary")
        sdf = pd.DataFrame(summary_rows).sort_values(by=["Bars Ago", "AI %"], ascending=[True, False])
        st.dataframe(sdf, width="stretch", hide_index=True)

    st.subheader(f"Live Charts ({len(results)} Visible)")
    use_grid = layout_mode.startswith("2-Column") and len(results) > 1
    cols = st.columns(2) if use_grid else None

    for idx, (nm, sym, r, badge, last_c_str) in enumerate(results):
        target_container = cols[idx % 2] if use_grid else st
        with target_container:
            stt = r["stats"]
            sigs = r["signals"]
            latest = sigs[-1] if sigs else None
            price_fmt = f"{stt['last_price']:,.4f}" if stt["last_price"] < 200 else f"{stt['last_price']:,.2f}"

            if latest is not None and latest["bars_ago"] <= fresh_bars_limit:
                cls = "sig-buy" if latest["side"] == "BUY" else "sig-sell"
                st.markdown(
                    f"<div class='{cls}'><b>🔥 {latest['side']} — {nm} ({sym})</b> &nbsp;|&nbsp; Price: <b>{price_fmt}</b> &nbsp;|&nbsp; {badge}<br>"
                    f"<b>Entry:</b> {latest['price']:,.4f} &nbsp;|&nbsp; <b>SL:</b> {latest['sl']:,.4f} &nbsp;|&nbsp; <b>TP:</b> {latest['tp']:,.4f} &nbsp;|&nbsp; "
                    f"<b>AI:</b> {latest['ai_score']:.1f}% ({latest['bars_ago']} bars ago)<br>"
                    f"<b>Setup:</b> {latest['reasons']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                last_txt = f"Last Alert: {latest['side']} @ {latest['price']:,.4f} ({latest['bars_ago']} bars ago)" if latest else "No recent alert"
                st.markdown(
                    f"<div class='sig-wait'><b>{nm} ({sym})</b> &nbsp;|&nbsp; Price: <b>{price_fmt}</b> &nbsp;|&nbsp; {badge} &nbsp;|&nbsp; "
                    f"Bias: {stt['bias']} &nbsp;|&nbsp; Zone: {stt['zone_status']}<br>{last_txt}</div>",
                    unsafe_allow_html=True,
                )

            fig = build_chart(r, f"{nm} ({sym}) · {timeframe}", bars_to_show, chart_height=440 if use_grid else 490)
            st.plotly_chart(fig, width="stretch", key=f"chart_{sym}_{idx}")


live_multi_chart_fragment()
