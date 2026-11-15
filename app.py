import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from strategy_engine import MARKET_WATCHLISTS, fetch_market_data, run_strategy

st.set_page_config(page_title="AI Trading Signals | India · Forex · Crypto", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; }
    .sig-buy { background: rgba(0, 200, 150, 0.14); border: 1px solid #00c896; padding: 14px; border-radius: 10px; }
    .sig-sell { background: rgba(255, 75, 75, 0.14); border: 1px solid #ff4b4b; padding: 14px; border-radius: 10px; }
    .sig-wait { background: rgba(140, 150, 170, 0.10); border: 1px solid #555; padding: 14px; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=120, show_spinner=False)
def cached_fetch(symbol: str, timeframe: str):
    return fetch_market_data(symbol, timeframe)


def send_telegram_alert(token: str, chat_id: str, msg: str):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, timeout=8)
    return resp.status_code == 200


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("AI Signal Terminal")

market_cat = st.sidebar.selectbox("Market Category", list(MARKET_WATCHLISTS.keys()))
preset_names = list(MARKET_WATCHLISTS[market_cat].keys())
selected_name = st.sidebar.selectbox("Select Instrument", preset_names)
custom_ticker = st.sidebar.text_input("Or Custom Ticker (Yahoo Format)", value="", placeholder="e.g. TATAPOWER.NS, ADA-USD")
active_symbol = custom_ticker.strip().upper() if custom_ticker.strip() else MARKET_WATCHLISTS[market_cat][selected_name]
display_title = active_symbol if custom_ticker.strip() else f"{selected_name} ({active_symbol})"

timeframe = st.sidebar.selectbox("Chart Timeframe", ["5m", "15m", "30m", "1h"], index=1)
bars_to_show = st.sidebar.slider("Visible Candles on Chart", 80, 400, 180, 20)

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
    use_ai_filter = st.checkbox("Filter by Min AI Confidence %", value=True)
    min_ai_conf = st.slider("Min AI Confidence %", 0, 90, 50, 5)
    ai_knn = st.slider("k-NN Neighbors", 3, 15, 5)

with st.sidebar.expander("Telegram Alerts (Optional)", expanded=False):
    tg_token = st.text_input("Bot Token", type="password")
    tg_chat_id = st.text_input("Chat ID")

config = {
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

if st.sidebar.button("Refresh Market Data", use_container_width=True):
    st.cache_data.clear()

# ==========================================
# MAIN HEADER & TABS
# ==========================================
st.title("AI Signals App: Forex · Crypto · Indian Markets")
st.caption("Candles + Liquidity Sweeps + Daily Pivots + Daily Supply/Demand + 4H HTF Liquidity + Dynamic Trendlines + Self-Learning k-NN AI")

tab_chart, tab_scanner = st.tabs(["Live Chart & AI Analyzer", "Multi-Market Scanner (India + Forex + Crypto)"])

with tab_chart:
    with st.spinner(f"Fetching {display_title} ({timeframe}) & running AI engine..."):
        df_ltf, df_1h, df_daily = cached_fetch(active_symbol, timeframe)
        res = run_strategy(df_ltf, df_1h, df_daily, config)

    if res is None:
        st.error(f"Could not load sufficient market data for `{active_symbol}`. Verify the Yahoo ticker format.")
    else:
        stats = res["stats"]
        df = res["df"]
        open_tr = res["open_trade"]
        sigs = res["signals"]

        # Top KPI Metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Last Price", f"{stats['last_price']:,.4f}" if stats["last_price"] < 200 else f"{stats['last_price']:,.2f}")
        c2.metric("Daily Pivot Bias", stats["bias"])
        c3.metric("Zone Status", stats["zone_status"])
        c4.metric("Live AI Score (L / S)", f"{stats['bull_ai']:.0f}% / {stats['bear_ai']:.0f}%")
        c5.metric(f"Backtest WR ({stats['total_trades']} tr)", f"{stats['win_rate']:.1f}% | AI Mem: {stats['ai_mem_wr']:.0f}%")

        # Active Trade / Latest Signal Card
        st.subheader("Live Signal Status")
        if open_tr is not None:
            is_long = open_tr["side"] == "LONG"
            box_cls = "sig-buy" if is_long else "sig-sell"
            msg_text = (
                f"<b>{open_tr['side']} SIGNAL ACTIVE — {display_title} ({timeframe})</b><br>"
                f"<b>Entry:</b> {open_tr['entry']:,.4f} &nbsp;|&nbsp; "
                f"<b>Stop Loss:</b> {open_tr['sl']:,.4f} &nbsp;|&nbsp; "
                f"<b>Take Profit:</b> {open_tr['tp']:,.4f} &nbsp;|&nbsp; "
                f"<b>AI Confidence:</b> {open_tr['ai_score']:.1f}%<br>"
                f"<b>Confluence:</b> {open_tr['reasons']} &nbsp;|&nbsp; <b>Triggered:</b> {open_tr['entry_time']}"
            )
            st.markdown(f"<div class='{box_cls}'>{msg_text}</div>", unsafe_allow_html=True)

            if tg_token and tg_chat_id:
                if st.button("Send Active Signal to Telegram"):
                    tg_msg = (
                        f"🚨 <b>{open_tr['side']} SIGNAL: {display_title} ({timeframe})</b>\n"
                        f"Entry: {open_tr['entry']:,.4f}\nSL: {open_tr['sl']:,.4f}\nTP: {open_tr['tp']:,.4f}\n"
                        f"AI Confidence: {open_tr['ai_score']:.1f}%\nReason: {open_tr['reasons']}"
                    )
                    if send_telegram_alert(tg_token, tg_chat_id, tg_msg):
                        st.success("Signal sent to Telegram!")
                    else:
                        st.error("Failed to send Telegram alert. Check Bot Token & Chat ID.")
        else:
            last_sig = sigs[-1] if sigs else None
            extra = f"Last closed signal: {last_sig['side']} @ {last_sig['price']:,.4f} ({last_sig['time']})" if last_sig else "No signals in window."
            st.markdown(
                f"<div class='sig-wait'><b>WAIT — Scanning for next Candlestick + Liquidity + Zone confirmation.</b><br>{extra}</div>",
                unsafe_allow_html=True,
            )

        # Build Interactive Plotly Chart
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

        # Pivots
        fig.add_trace(go.Scatter(x=x_vals, y=df_view["PP"], name="Daily PP", line=dict(color="gold", width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(x=x_vals, y=df_view["R1"], name="Daily R1", line=dict(color="rgba(255,90,90,0.5)", width=1)))
        fig.add_trace(go.Scatter(x=x_vals, y=df_view["S1"], name="Daily S1", line=dict(color="rgba(0,200,150,0.5)", width=1)))

        # Draw Zones (Daily Buy/Sell + 4H HTF Liquidity)
        def add_zones(zones, fill_color, border_color):
            for z in zones:
                if z["end"] < view_start:
                    continue
                x0 = max(z["start"], view_start)
                x1 = z["end"]
                fig.add_shape(
                    type="rect", x0=x0, x1=x1, y0=z["bot"], y1=z["top"],
                    fillcolor=fill_color, line=dict(color=border_color, width=1),
                    layer="below",
                )
                fig.add_annotation(
                    x=x1, y=(z["top"] + z["bot"]) / 2, text=z["label"],
                    showarrow=False, xanchor="right", font=dict(size=9, color=border_color),
                )

        add_zones(res["active_dem"], "rgba(0, 180, 160, 0.16)", "rgba(0, 220, 190, 0.85)")
        add_zones(res["active_sup"], "rgba(255, 75, 75, 0.16)", "rgba(255, 90, 90, 0.85)")
        add_zones(res["active_htf_ssl"], "rgba(40, 140, 255, 0.16)", "rgba(80, 175, 255, 0.85)")
        add_zones(res["active_htf_bsl"], "rgba(180, 70, 255, 0.16)", "rgba(200, 100, 255, 0.85)")

        # Draw Active Trendlines
        for tl, col, name in [
            (res["active_up_tl"], "#80ff00", "Uptrend Line"),
            (res["active_dn_tl"], "#ff9900", "Downtrend Line"),
        ]:
            if tl is not None and tl["end"] >= view_start:
                x1_clamped = max(tl["x1"], view_start)
                slope = (tl["y2"] - tl["y1"]) / (tl["x2"] - tl["x1"]) if tl["x2"] != tl["x1"] else 0.0
                y1_clamped = tl["y1"] + slope * (x1_clamped - tl["x1"])
                y2_end = tl["y1"] + slope * (tl["end"] - tl["x1"])
                fig.add_trace(
                    go.Scatter(
                        x=[x1_clamped, tl["end"]], y=[y1_clamped, y2_end],
                        mode="lines", name=name, line=dict(color=col, width=2),
                    )
                )

        # Plot Buy & Sell Signals
        buy_sigs = [s for s in sigs if s["side"] == "BUY" and s["idx"] >= view_start]
        sell_sigs = [s for s in sigs if s["side"] == "SELL" and s["idx"] >= view_start]

        if buy_sigs:
            fig.add_trace(
                go.Scatter(
                    x=[s["idx"] for s in buy_sigs],
                    y=[df["Low"].iloc[s["idx"]] * 0.9985 for s in buy_sigs],
                    mode="markers+text",
                    name="BUY Signal",
                    text=[f"BUY ({s['ai_score']:.0f}%)" for s in buy_sigs],
                    textposition="bottom center",
                    marker=dict(symbol="triangle-up", size=12, color="#00ffaa"),
                    textfont=dict(size=10, color="#00ffaa"),
                )
            )
        if sell_sigs:
            fig.add_trace(
                go.Scatter(
                    x=[s["idx"] for s in sell_sigs],
                    y=[df["High"].iloc[s["idx"]] * 1.0015 for s in sell_sigs],
                    mode="markers+text",
                    name="SELL Signal",
                    text=[f"SELL ({s['ai_score']:.0f}%)" for s in sell_sigs],
                    textposition="top center",
                    marker=dict(symbol="triangle-down", size=12, color="#ff4b4b"),
                    textfont=dict(size=10, color="#ff4b4b"),
                )
            )

        # Plot Active SL/TP Lines if in open trade
        if open_tr is not None:
            x_start = max(open_tr["entry_idx"], view_start)
            fig.add_trace(go.Scatter(x=[x_start, len(df) - 1], y=[open_tr["sl"], open_tr["sl"]], mode="lines", name="Active SL", line=dict(color="#ff4b4b", dash="dash")))
            fig.add_trace(go.Scatter(x=[x_start, len(df) - 1], y=[open_tr["tp"], open_tr["tp"]], mode="lines", name="Active TP", line=dict(color="#00ffaa", dash="dash")))

        tick_step = max(1, len(x_vals) // 10)
        fig.update_layout(
            template="plotly_dark",
            height=640,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(tickvals=x_vals[::tick_step], ticktext=x_labels[::tick_step]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Recent Signals & Closed Trades Table
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Recent Signals Log")
            if sigs:
                sig_df = pd.DataFrame(sigs[::-1][:15])[["time", "side", "price", "sl", "tp", "ai_score", "reasons"]]
                sig_df["ai_score"] = sig_df["ai_score"].round(1)
                st.dataframe(sig_df, use_container_width=True, hide_index=True)
            else:
                st.info("No signals triggered with current filters.")

        with col_right:
            st.subheader("Closed Trades (AI Training History)")
            if res["trades"]:
                tr_df = pd.DataFrame(res["trades"][::-1][:15])[
                    ["entry_time", "side", "entry", "sl", "tp", "result", "ai_score", "pnl_r"]
                ]
                tr_df["ai_score"] = tr_df["ai_score"].round(1)
                st.dataframe(tr_df, use_container_width=True, hide_index=True)
            else:
                st.info("No closed trades yet in this lookback window.")

with tab_scanner:
    st.subheader("Multi-Market Live Signal Scanner")
    st.write("Scans **Indian Markets (NSE)**, **Forex & Gold**, and **Crypto** simultaneously using your AI & Zone settings.")

    scan_cats = st.multiselect(
        "Markets to Scan",
        list(MARKET_WATCHLISTS.keys()),
        default=list(MARKET_WATCHLISTS.keys()),
    )

    if st.button("Run Full Market Scan Now", type="primary"):
        items = []
        for cat in scan_cats:
            for nm, sym in MARKET_WATCHLISTS[cat].items():
                items.append((cat, nm, sym))

        rows = []
        prog = st.progress(0)
        for idx, (cat, nm, sym) in enumerate(items):
            try:
                d_ltf, d_1h, d_day = cached_fetch(sym, timeframe)
                r = run_strategy(d_ltf, d_1h, d_day, config)
                if r is not None:
                    stt = r["stats"]
                    op = r["open_trade"]
                    last_sig = r["signals"][-1] if r["signals"] else None

                    if op is not None:
                        sig_status = f"ACTIVE {op['side']}"
                        entry_p, sl_p, tp_p = op["entry"], op["sl"], op["tp"]
                        ai_sc = op["ai_score"]
                        reason = op["reasons"]
                    elif last_sig is not None:
                        sig_status = f"LAST: {last_sig['side']} ({last_sig['time'].strftime('%m-%d %H:%M')})"
                        entry_p, sl_p, tp_p = last_sig["price"], last_sig["sl"], last_sig["tp"]
                        ai_sc = last_sig["ai_score"]
                        reason = last_sig["reasons"]
                    else:
                        sig_status = "WAIT"
                        entry_p = sl_p = tp_p = None
                        ai_sc = max(stt["bull_ai"], stt["bear_ai"])
                        reason = "-"

                    rows.append({
                        "Market": cat,
                        "Instrument": f"{nm} ({sym})",
                        "Price": round(stt["last_price"], 4 if stt["last_price"] < 200 else 2),
                        "Signal": sig_status,
                        "AI Score %": round(ai_sc, 1),
                        "Entry": round(entry_p, 4 if entry_p and entry_p < 200 else 2) if entry_p else None,
                        "SL": round(sl_p, 4 if sl_p and sl_p < 200 else 2) if sl_p else None,
                        "TP": round(tp_p, 4 if tp_p and tp_p < 200 else 2) if tp_p else None,
                        "Zone": stt["zone_status"],
                        "Bias": stt["bias"],
                        "Win Rate %": round(stt["win_rate"], 1),
                        "Setup": reason,
                    })
            except Exception:
                pass
            prog.progress((idx + 1) / max(1, len(items)))

        if rows:
            scan_df = pd.DataFrame(rows).sort_values(by=["Signal", "AI Score %"], ascending=[True, False])
            st.dataframe(scan_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No instruments returned data.")
