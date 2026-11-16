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

st.set_page_config(page_title="AI Live Trading Signals | India · Forex · Crypto", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.0rem; }
    .sig-buy { background: rgba(0, 200, 150, 0.14); border: 1px solid #00c896; padding: 14px; border-radius: 10px; margin-bottom: 10px; }
    .sig-sell { background: rgba(255, 75, 75, 0.14); border: 1px solid #ff4b4b; padding: 14px; border-radius: 10px; margin-bottom: 10px; }
    .sig-wait { background: rgba(140, 150, 170, 0.10); border: 1px solid #555; padding: 14px; border-radius: 10px; margin-bottom: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "sent_alerts" not in st.session_state:
    st.session_state["sent_alerts"] = set()


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
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, timeout=8)
    return resp.status_code == 200


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("AI Live Signal Terminal")

market_cat = st.sidebar.selectbox("Market Category", list(MARKET_WATCHLISTS.keys()))
preset_names = list(MARKET_WATCHLISTS[market_cat].keys())
selected_name = st.sidebar.selectbox("Select Instrument", preset_names)
custom_ticker = st.sidebar.text_input("Or Custom Ticker (Yahoo Format)", value="", placeholder="e.g. TATAPOWER.NS, ADA-USD")
active_symbol = custom_ticker.strip().upper() if custom_ticker.strip() else MARKET_WATCHLISTS[market_cat][selected_name]
display_title = active_symbol if custom_ticker.strip() else f"{selected_name} ({active_symbol})"

timeframe = st.sidebar.selectbox("Chart Timeframe", ["5m", "15m", "30m", "1h"], index=1)
tz_choice = st.sidebar.selectbox("Display Timezone", ["Asia/Kolkata", "UTC"], index=0)
auto_refresh_choice = st.sidebar.selectbox("Auto-Refresh", ["Off", "60s", "5m"], index=0)
fresh_bars_limit = st.sidebar.slider("Fresh Alert Window (Candles)", 1, 15, 5)
bars_to_show = st.sidebar.slider("Visible Candles on Chart", 80, 350, 140, 20)

with st.sidebar.expander("Strategy & Zone Settings", expanded=False):
    entry_mode = st.selectbox(
        "Entry Mode",
        ["All Signals (Sweeps + Zones + TL)", "HTF / Daily Zone Confluence Only"],
        index=0,
    )
    rr_ratio = st.number_input("Risk : Reward Ratio", 1.0, 5.0, 2.0, 0.25)
    sd_impulse = st.number_input("Daily S/D Impulse (x ATR)", 0.3, 2.0, 0.8, 0.1)
    tl_in_signals = st.checkbox("Trade Trendline Bounces & Breaks", value=True)
    use_pp_bias = st.checkbox("Filter by Daily PP Bias", value=False)

with st.sidebar.expander("Inbuilt AI Settings", expanded=False):
    use_ai = st.checkbox("Enable k-NN Self-Learning AI", value=True)
    use_ai_filter = st.checkbox("Filter by Min AI Confidence %", value=False)
    min_ai_conf = st.slider("Min AI Confidence %", 0, 90, 45, 5)
    ai_knn = st.slider("k-NN Neighbors", 3, 15, 5)

with st.sidebar.expander("Telegram Auto-Alerts", expanded=False):
    tg_token = st.text_input("Bot Token", type="password")
    tg_chat_id = st.text_input("Chat ID")
    auto_tg = st.checkbox("Auto-Send Fresh Alerts to Telegram", value=False)

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
# HEADER & TOP REFRESH BUTTON
# ==========================================
h_col1, h_col2 = st.columns([4, 1])
with h_col1:
    st.title("AI Live Trading Signals: Forex · Crypto · Indian Markets")
    now_str = pd.Timestamp.now(tz=tz_choice).strftime("%d %b %Y, %H:%M:%S")
    st.caption(f"Current Time ({tz_choice}): {now_str}  |  Auto-Refresh: {auto_refresh_choice}")
with h_col2:
    st.write("")
    if st.button("🔄 Refresh Live Now", type="primary", width="stretch"):
        st.cache_data.clear()
        st.rerun()


@st.fragment(run_every=run_every_val)
def live_terminal_fragment():
    tab_alerts, tab_chart, tab_scanner = st.tabs(
        ["Live Alerts Feed", "Live Chart & AI Analyzer", "Full Multi-Market Scan"]
    )

    # ------------------------------------------
    # TAB 1: LIVE ALERTS FEED (SORTED NEWEST FIRST)
    # ------------------------------------------
    with tab_alerts:
        st.subheader(f"Live Alerts — {market_cat} ({timeframe})")
        watch_items = list(MARKET_WATCHLISTS[market_cat].items())
        alert_rows = []
        fresh_cards = []

        for nm, sym in watch_items:
            try:
                r = load_and_run(sym, timeframe, config)
                if r is None:
                    continue
                stt = r["stats"]
                sigs = r["signals"]
                latest = sigs[-1] if sigs else None
                last_c_str = stt["last_time"].strftime("%d %b %H:%M")
                mkt_badge = f"🟢 LIVE ({stt['mins_ago']}m ago)" if stt["market_live"] else f"🔴 CLOSED ({last_c_str})"

                if latest is not None:
                    bars_ago = latest["bars_ago"]
                    is_fresh = bars_ago <= fresh_bars_limit
                    sig_tag = f"🔥 FRESH {latest['side']}" if is_fresh else f"{latest['side']}"
                    alert_rows.append({
                        "Instrument": f"{nm} ({sym})",
                        "Market Status": mkt_badge,
                        "Last Candle": last_c_str,
                        "Live Price": round(stt["last_price"], 4 if stt["last_price"] < 200 else 2),
                        "Signal": sig_tag,
                        "Bars Ago": int(bars_ago),
                        "Alert Time": latest["time"].strftime("%d %b %H:%M"),
                        "AI %": round(latest["ai_score"], 1),
                        "Entry": round(latest["price"], 4 if latest["price"] < 200 else 2),
                        "SL": round(latest["sl"], 4 if latest["sl"] < 200 else 2),
                        "TP": round(latest["tp"], 4 if latest["tp"] < 200 else 2),
                        "Zone": stt["zone_status"],
                        "Confluence": latest["reasons"],
                    })

                    if is_fresh:
                        fresh_cards.append((nm, sym, latest, mkt_badge, stt["last_price"]))
                        alert_key = f"{sym}_{timeframe}_{latest['side']}_{latest['time']}"
                        if auto_tg and tg_token and tg_chat_id and latest["bars_ago"] <= 1 and alert_key not in st.session_state["sent_alerts"]:
                            tg_msg = (
                                f"🚨 <b>LIVE {latest['side']} ALERT: {nm} ({sym}) [{timeframe}]</b>\n"
                                f"Entry: {latest['price']:,.4f}\nSL: {latest['sl']:,.4f}\nTP: {latest['tp']:,.4f}\n"
                                f"AI Confidence: {latest['ai_score']:.1f}%\nReason: {latest['reasons']}\n"
                                f"Time ({tz_choice}): {latest['time'].strftime('%d %b %H:%M')}"
                            )
                            if send_telegram_alert(tg_token, tg_chat_id, tg_msg):
                                st.session_state["sent_alerts"].add(alert_key)
                else:
                    alert_rows.append({
                        "Instrument": f"{nm} ({sym})",
                        "Market Status": mkt_badge,
                        "Last Candle": last_c_str,
                        "Live Price": round(stt["last_price"], 4 if stt["last_price"] < 200 else 2),
                        "Signal": "WAIT",
                        "Bars Ago": 9999,
                        "Alert Time": "-",
                        "AI %": round(max(stt["bull_ai"], stt["bear_ai"]), 1),
                        "Entry": None, "SL": None, "TP": None,
                        "Zone": stt["zone_status"],
                        "Confluence": "-",
                    })
            except Exception:
                pass

        if fresh_cards:
            st.markdown(f"### 🔥 Fresh Alerts (Within Last {fresh_bars_limit} Candles)")
            for nm, sym, lt, badge, cur_p in fresh_cards:
                cls = "sig-buy" if lt["side"] == "BUY" else "sig-sell"
                st.markdown(
                    f"<div class='{cls}'><b>🔥 {lt['side']} — {nm} ({sym})</b> &nbsp;|&nbsp; {badge} &nbsp;|&nbsp; <b>Current Price:</b> {cur_p:,.4f}<br>"
                    f"<b>Entry:</b> {lt['price']:,.4f} &nbsp;|&nbsp; <b>SL:</b> {lt['sl']:,.4f} &nbsp;|&nbsp; <b>TP:</b> {lt['tp']:,.4f} &nbsp;|&nbsp; "
                    f"<b>AI:</b> {lt['ai_score']:.1f}% &nbsp;|&nbsp; <b>Alert Time:</b> {lt['time'].strftime('%d %b %H:%M')} ({lt['bars_ago']} bars ago)<br>"
                    f"<b>Setup:</b> {lt['reasons']}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info(f"No signals in the last {fresh_bars_limit} candles for {market_cat}. Latest setups sorted newest-first below:")

        if alert_rows:
            adf = pd.DataFrame(alert_rows).sort_values(by=["Bars Ago", "AI %"], ascending=[True, False])
            st.dataframe(adf, width="stretch", hide_index=True)

    # ------------------------------------------
    # TAB 2: LIVE CHART & AI ANALYZER
    # ------------------------------------------
    with tab_chart:
        with st.spinner(f"Loading {display_title} ({timeframe})..."):
            res = load_and_run(active_symbol, timeframe, config)

        if res is None:
            st.error(f"Could not load sufficient market data for `{active_symbol}`.")
        else:
            stats = res["stats"]
            df = res["df"]
            sigs = res["signals"]
            latest_sig = sigs[-1] if sigs else None
            last_c_str = stats["last_time"].strftime("%d %b %H:%M")
            mkt_str = f"🟢 LIVE ({stats['mins_ago']}m ago | {last_c_str})" if stats["market_live"] else f"🔴 CLOSED ({last_c_str})"

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Last Price", f"{stats['last_price']:,.4f}" if stats["last_price"] < 200 else f"{stats['last_price']:,.2f}", mkt_str)
            c2.metric("Daily Pivot Bias", stats["bias"])
            c3.metric("Zone Status", stats["zone_status"])
            c4.metric("Live AI Score (L / S)", f"{stats['bull_ai']:.0f}% / {stats['bear_ai']:.0f}%")
            c5.metric(f"Backtest WR ({stats['total_trades']} tr)", f"{stats['win_rate']:.1f}% | AI Mem: {stats['ai_mem_wr']:.0f}%")

            if latest_sig is not None and latest_sig["bars_ago"] <= fresh_bars_limit:
                cls = "sig-buy" if latest_sig["side"] == "BUY" else "sig-sell"
                st.markdown(
                    f"<div class='{cls}'><b>🔥 FRESH {latest_sig['side']} ALERT ({latest_sig['bars_ago']} bars ago) — {display_title}</b><br>"
                    f"<b>Entry:</b> {latest_sig['price']:,.4f} &nbsp;|&nbsp; <b>SL:</b> {latest_sig['sl']:,.4f} &nbsp;|&nbsp; <b>TP:</b> {latest_sig['tp']:,.4f} &nbsp;|&nbsp; "
                    f"<b>AI:</b> {latest_sig['ai_score']:.1f}% &nbsp;|&nbsp; <b>Time:</b> {latest_sig['time'].strftime('%d %b %H:%M')}<br>"
                    f"<b>Confluence:</b> {latest_sig['reasons']}</div>",
                    unsafe_allow_html=True,
                )
            elif latest_sig is not None:
                st.markdown(
                    f"<div class='sig-wait'><b>WAIT — Monitoring {display_title} ({mkt_str})</b><br>"
                    f"Latest alert: <b>{latest_sig['side']}</b> @ {latest_sig['price']:,.4f} on {latest_sig['time'].strftime('%d %b %H:%M')} "
                    f"({latest_sig['bars_ago']} bars ago) — AI: {latest_sig['ai_score']:.1f}% ({latest_sig['reasons']})</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='sig-wait'><b>WAIT — Scanning {display_title} ({mkt_str})</b></div>",
                    unsafe_allow_html=True,
                )

            view_start = max(0, len(df) - bars_to_show)
            df_view = df.iloc[view_start:]
            x_vals = list(range(view_start, len(df)))
            x_labels = [t.strftime("%m-%d %H:%M") for t in df_view.index]

            fig = go.Figure()
            fig.add_trace(
                go.Candlestick(
                    x=x_vals, open=df_view["Open"], high=df_view["High"],
                    low=df_view["Low"], close=df_view["Close"], name="Price",
                    increasing_line_color="#00c896", decreasing_line_color="#ff4b4b",
                )
            )
            fig.add_trace(go.Scatter(x=x_vals, y=df_view["PP"], name="Daily PP", line=dict(color="gold", width=1.5, dash="dot")))
            fig.add_trace(go.Scatter(x=x_vals, y=df_view["R1"], name="Daily R1", line=dict(color="rgba(255,90,90,0.5)", width=1)))
            fig.add_trace(go.Scatter(x=x_vals, y=df_view["S1"], name="Daily S1", line=dict(color="rgba(0,200,150,0.5)", width=1)))

            def add_zones(zones, fill_color, border_color):
                for z in zones:
                    if z["end"] < view_start:
                        continue
                    x0 = max(z["start"], view_start)
                    x1 = z["end"]
                    fig.add_shape(type="rect", x0=x0, x1=x1, y0=z["bot"], y1=z["top"], fillcolor=fill_color, line=dict(color=border_color, width=1), layer="below")
                    fig.add_annotation(x=x1, y=(z["top"] + z["bot"]) / 2, text=z["label"], showarrow=False, xanchor="right", font=dict(size=9, color=border_color))

            add_zones(res["active_dem"], "rgba(0, 180, 160, 0.16)", "rgba(0, 220, 190, 0.85)")
            add_zones(res["active_sup"], "rgba(255, 75, 75, 0.16)", "rgba(255, 90, 90, 0.85)")
            add_zones(res["active_htf_ssl"], "rgba(40, 140, 255, 0.16)", "rgba(80, 175, 255, 0.85)")
            add_zones(res["active_htf_bsl"], "rgba(180, 70, 255, 0.16)", "rgba(200, 100, 255, 0.85)")

            for tl, col, name in [(res["active_up_tl"], "#80ff00", "Uptrend Line"), (res["active_dn_tl"], "#ff9900", "Downtrend Line")]:
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
                    mode="markers+text", name="BUY Signal", text=[f"BUY ({s['ai_score']:.0f}%)" for s in buy_sigs],
                    textposition="bottom center", marker=dict(symbol="triangle-up", size=12, color="#00ffaa"), textfont=dict(size=10, color="#00ffaa"),
                ))
            if sell_sigs:
                fig.add_trace(go.Scatter(
                    x=[s["idx"] for s in sell_sigs], y=[df["High"].iloc[s["idx"]] * 1.0015 for s in sell_sigs],
                    mode="markers+text", name="SELL Signal", text=[f"SELL ({s['ai_score']:.0f}%)" for s in sell_sigs],
                    textposition="top center", marker=dict(symbol="triangle-down", size=12, color="#ff4b4b"), textfont=dict(size=10, color="#ff4b4b"),
                ))

            tick_step = max(1, len(x_vals) // 10)
            fig.update_layout(
                template="plotly_dark", height=620, xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(tickvals=x_vals[::tick_step], ticktext=x_labels[::tick_step]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, width="stretch")

            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader("Recent Signals Log")
                if sigs:
                    sig_df = pd.DataFrame(sigs[::-1][:15])[["time", "bars_ago", "side", "price", "sl", "tp", "ai_score", "reasons"]]
                    sig_df["ai_score"] = sig_df["ai_score"].round(1)
                    st.dataframe(sig_df, width="stretch", hide_index=True)
                else:
                    st.info("No signals triggered with current filters.")
            with col_right:
                st.subheader("Closed Trades (AI Training History)")
                if res["trades"]:
                    tr_df = pd.DataFrame(res["trades"][::-1][:15])[["entry_time", "side", "entry", "sl", "tp", "result", "ai_score", "pnl_r"]]
                    tr_df["ai_score"] = tr_df["ai_score"].round(1)
                    st.dataframe(tr_df, width="stretch", hide_index=True)
                else:
                    st.info("No closed trades yet in this lookback window.")

    # ------------------------------------------
    # TAB 3: FULL MULTI-MARKET SCANNER
    # ------------------------------------------
    with tab_scanner:
        st.subheader("Full Multi-Market Scan (Crypto + Forex + India)")
        scan_cats = st.multiselect("Markets to Scan", list(MARKET_WATCHLISTS.keys()), default=list(MARKET_WATCHLISTS.keys()))
        if st.button("Run Full Market Scan Now", type="primary"):
            items = [(cat, nm, sym) for cat in scan_cats for nm, sym in MARKET_WATCHLISTS[cat].items()]
            rows = []
            prog = st.progress(0)
            for idx, (cat, nm, sym) in enumerate(items):
                try:
                    r = load_and_run(sym, timeframe, config)
                    if r is not None:
                        stt = r["stats"]
                        latest = r["signals"][-1] if r["signals"] else None
                        last_c_str = stt["last_time"].strftime("%d %b %H:%M")
                        badge = f"🟢 LIVE ({stt['mins_ago']}m)" if stt["market_live"] else f"🔴 CLOSED ({last_c_str})"
                        rows.append({
                            "Market": cat,
                            "Instrument": f"{nm} ({sym})",
                            "Status": badge,
                            "Last Candle": last_c_str,
                            "Live Price": round(stt["last_price"], 4 if stt["last_price"] < 200 else 2),
                            "Latest Signal": latest["side"] if latest else "WAIT",
                            "Bars Ago": int(latest["bars_ago"]) if latest else 9999,
                            "Alert Time": latest["time"].strftime("%d %b %H:%M") if latest else "-",
                            "AI %": round(latest["ai_score"], 1) if latest else round(max(stt["bull_ai"], stt["bear_ai"]), 1),
                            "Entry": round(latest["price"], 4 if latest["price"] < 200 else 2) if latest else None,
                            "SL": round(latest["sl"], 4 if latest["sl"] < 200 else 2) if latest else None,
                            "TP": round(latest["tp"], 4 if latest["tp"] < 200 else 2) if latest else None,
                            "Zone": stt["zone_status"],
                            "Setup": latest["reasons"] if latest else "-",
                        })
                except Exception:
                    pass
                prog.progress((idx + 1) / max(1, len(items)))

            if rows:
                scan_df = pd.DataFrame(rows).sort_values(by=["Bars Ago", "AI %"], ascending=[True, False])
                st.dataframe(scan_df, width="stretch", hide_index=True)
            else:
                st.warning("No instruments returned data.")


live_terminal_fragment()
