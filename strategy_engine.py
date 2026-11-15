import numpy as np
import pandas as pd
import yfinance as yf


MARKET_WATCHLISTS = {
    "Indian Markets (NSE)": {
        "NIFTY 50": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "RELIANCE": "RELIANCE.NS",
        "HDFC BANK": "HDFCBANK.NS",
        "TCS": "TCS.NS",
        "INFOSYS": "INFY.NS",
        "ICICI BANK": "ICICIBANK.NS",
        "SBIN": "SBIN.NS",
        "ITC": "ITC.NS",
        "L&T": "LT.NS",
    },
    "Forex & Gold": {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "USDJPY=X",
        "USD/INR": "USDINR=X",
        "AUD/USD": "AUDUSD=X",
        "USD/CAD": "USDCAD=X",
        "GOLD (Futures)": "GC=F",
    },
    "Crypto": {
        "BTC/USD": "BTC-USD",
        "ETH/USD": "ETH-USD",
        "SOL/USD": "SOL-USD",
        "XRP/USD": "XRP-USD",
        "BNB/USD": "BNB-USD",
        "DOGE/USD": "DOGE-USD",
    },
}


def _flatten_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def fetch_market_data(symbol: str, timeframe: str = "15m"):
    """Fetch LTF intraday data, 1H data (for 4H HTF zones), and Daily data."""
    period_map = {"5m": "30d", "15m": "60d", "30m": "60d", "1h": "60d"}
    ltf_period = period_map.get(timeframe, "60d")

    df_ltf = _flatten_yf_df(
        yf.download(symbol, period=ltf_period, interval=timeframe, progress=False, auto_adjust=True)
    )
    df_1h = _flatten_yf_df(
        yf.download(symbol, period="60d", interval="1h", progress=False, auto_adjust=True)
    )
    df_daily = _flatten_yf_df(
        yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
    )
    return df_ltf, df_1h, df_daily


def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _line_price_at(x1: int, y1: float, x2: int, y2: float, x: int) -> float:
    if x2 == x1:
        return y2
    slope = (y2 - y1) / (x2 - x1)
    return y1 + slope * (x - x1)


def run_strategy(
    df_ltf: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_daily: pd.DataFrame,
    config: dict,
):
    if df_ltf.empty or df_daily.empty or len(df_ltf) < 50:
        return None

    df = df_ltf.copy()
    df["ATR"] = compute_atr(df, config["atr_len"])
    df["RSI"] = compute_rsi(df["Close"], 14).fillna(50.0)
    df["VolSMA"] = df["Volume"].rolling(20).mean().fillna(0.0)

    # -------------------------------------------------
    # 1. DAILY PIVOTS & DAILY SUPPLY/DEMAND (NON-REPAINTING)
    # -------------------------------------------------
    d = df_daily.copy()
    d["dATR"] = compute_atr(d, 14)
    d["PP"] = (d["High"] + d["Low"] + d["Close"]) / 3.0
    d["R1"] = 2.0 * d["PP"] - d["Low"]
    d["S1"] = 2.0 * d["PP"] - d["High"]
    d["R2"] = d["PP"] + (d["High"] - d["Low"])
    d["S2"] = d["PP"] - (d["High"] - d["Low"])

    # Impulse checks on completed daily bars
    d["PrevO"] = d["Open"].shift(1)
    d["PrevH"] = d["High"].shift(1)
    d["PrevL"] = d["Low"].shift(1)
    d["PrevC"] = d["Close"].shift(1)

    d["BullImp"] = (
        (d["Close"] > d["Open"])
        & ((d["Close"] - d["Open"]) >= d["dATR"] * config["sd_impulse_mult"])
        & (d["Close"] > d["PrevH"])
    )
    d["BearImp"] = (
        (d["Close"] < d["Open"])
        & ((d["Open"] - d["Close"]) >= d["dATR"] * config["sd_impulse_mult"])
        & (d["Close"] < d["PrevL"])
    )

    d["date_norm"] = pd.to_datetime(d.index).normalize()
    ltf_dates = pd.DataFrame({"date_norm": pd.to_datetime(df.index).normalize()})
    merged_d = pd.merge_asof(
        ltf_dates,
        d.reset_index(drop=True)[
            [
                "date_norm", "PP", "R1", "S1", "R2", "S2",
                "Open", "High", "Low", "Close",
                "PrevO", "PrevH", "PrevL", "PrevC", "dATR", "BullImp", "BearImp",
            ]
        ].sort_values("date_norm"),
        on="date_norm",
        direction="backward",
        allow_exact_matches=False,  # Strictly prior completed trading day
    )

    for col in ["PP", "R1", "S1", "R2", "S2"]:
        df[col] = merged_d[col].values

    # -------------------------------------------------
    # 2. 4H HTF LIQUIDITY ZONES (NON-REPAINTING)
    # -------------------------------------------------
    htf_events = {}
    if not df_1h.empty:
        h4 = df_1h.resample("4h").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        ).dropna()
        h4["hATR"] = compute_atr(h4, 14)
        hlen = config["htf_swing_len"]
        highs_4h = h4["High"].values
        lows_4h = h4["Low"].values
        opens_4h = h4["Open"].values
        closes_4h = h4["Close"].values
        atrs_4h = h4["hATR"].values
        times_4h = h4.index

        for j in range(hlen, len(h4) - hlen):
            confirm_time = times_4h[j + hlen] + pd.Timedelta(hours=4)
            atr_j = atrs_4h[j] if not np.isnan(atrs_4h[j]) else 0.0
            if atr_j <= 0:
                continue
            win_h = highs_4h[j - hlen : j + hlen + 1]
            win_l = lows_4h[j - hlen : j + hlen + 1]
            if highs_4h[j] == np.max(win_h):
                z_top = float(highs_4h[j])
                z_bot = float(min(max(opens_4h[j], closes_4h[j]), z_top - atr_j * 0.15))
                htf_events.setdefault(confirm_time, []).append(("BSL", z_top, z_bot))
            if lows_4h[j] == np.min(win_l):
                z_bot = float(lows_4h[j])
                z_top = float(max(min(opens_4h[j], closes_4h[j]), z_bot + atr_j * 0.15))
                htf_events.setdefault(confirm_time, []).append(("SSL", z_top, z_bot))

    sorted_htf_times = sorted(htf_events.keys())
    htf_ptr = 0

    # -------------------------------------------------
    # 3. BAR-BY-BAR SIMULATION (ZONES, TRENDLINES, AI, SIGNALS)
    # -------------------------------------------------
    n = len(df)
    times = df.index
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    vols = df["Volume"].values
    vol_smas = df["VolSMA"].values
    atrs = df["ATR"].values
    rsis = df["RSI"].values
    pps = df["PP"].values
    r1s, s1s = df["R1"].values, df["S1"].values
    r2s, s2s = df["R2"].values, df["S2"].values

    # Zone trackers (store dicts with start_idx, end_idx, top, bot, status, type)
    active_dem, closed_dem = [], []
    active_sup, closed_sup = [], []
    active_htf_bsl, closed_htf_bsl = [], []
    active_htf_ssl, closed_htf_ssl = [], []

    # Chart swing liquidity pools
    bsl_pools, ssl_pools = [], []

    # Trendline trackers
    prev_pl, last_pl, active_up_tl, closed_tls = None, None, None, []
    prev_ph, last_ph, active_dn_tl = None, None, None

    # Sweep window trackers
    last_bull_sweep_bar, last_bull_sweep_low = None, None
    last_bear_sweep_bar, last_bear_sweep_high = None, None

    # AI Memory & Trade Simulator
    ai_memory = []  # tuples of (f1, f2, f3, f4, f5, win_float)
    open_trade = None
    trades = []
    signals_log = []

    sw_len = config["swing_len"]
    tl_len = config["tl_swing_len"]

    prev_day_key = None
    bull_sweeps_idx, bear_sweeps_idx = [], []
    last_bull_ai, last_bear_ai = 50.0, 50.0
    last_zone_status = "NEUTRAL"

    def compute_ai_score(f1, f2, f3, f4, f5, is_long):
        zone_pts = 10.0 + f4 * 25.0
        tl_pts = f5 * 15.0
        cnd_pts = min(f3 * 30.0, 25.0)
        vol_pts = f2 * 15.0
        rsi_pts = (1.0 - f1) * 10.0 if is_long else f1 * 10.0
        struct_score = min(zone_pts + tl_pts + cnd_pts + vol_pts + rsi_pts, 100.0)

        if config["use_ai"] and len(ai_memory) >= config["ai_knn"]:
            cur = np.array([f1, f2, f3, f4, f5])
            mem_feats = np.array([m[:5] for m in ai_memory])
            mem_outs = np.array([m[5] for m in ai_memory])
            dists = np.linalg.norm(mem_feats - cur, axis=1)
            k_idx = np.argsort(dists)[: config["ai_knn"]]
            knn_prob = float(np.mean(mem_outs[k_idx]) * 100.0)
            return 0.40 * struct_score + 0.60 * knn_prob
        return struct_score

    for i in range(n):
        t = times[i]
        o, h, l, c = float(opens[i]), float(highs[i]), float(lows[i]), float(closes[i])
        atr = float(atrs[i]) if not np.isnan(atrs[i]) and atrs[i] > 0 else (h - l + 1e-6)
        pp, r1, s1, r2, s2 = pps[i], r1s[i], s1s[i], r2s[i], s2s[i]

        # 0. Check open trade SL / TP exit
        if open_trade is not None:
            hit_sl = (l <= open_trade["sl"]) if open_trade["side"] == "LONG" else (h >= open_trade["sl"])
            hit_tp = (h >= open_trade["tp"]) if open_trade["side"] == "LONG" else (l <= open_trade["tp"])
            if hit_sl or hit_tp:
                win = 1.0 if (hit_tp and not hit_sl) else 0.0
                pnl_r = config["rr_ratio"] if win == 1.0 else -1.0
                open_trade["exit_time"] = t
                open_trade["exit_idx"] = i
                open_trade["result"] = "WIN (TP)" if win == 1.0 else "LOSS (SL)"
                open_trade["pnl_r"] = pnl_r
                trades.append(open_trade)
                ai_memory.append((*open_trade["features"], win))
                if len(ai_memory) > config["ai_max_memory"]:
                    ai_memory.pop(0)
                open_trade = None

        # A. Daily S/D Zone creation on new day
        day_key = merged_d["date_norm"].iloc[i]
        d_atr = merged_d["dATR"].iloc[i]
        if day_key != prev_day_key and pd.notna(d_atr) and d_atr > 0:
            prev_day_key = day_key
            if bool(merged_d["BullImp"].iloc[i]):
                d1o, d1l, d1c = float(merged_d["Open"].iloc[i]), float(merged_d["Low"].iloc[i]), float(merged_d["Close"].iloc[i])
                d2o, d2l, d2c = float(merged_d["PrevO"].iloc[i]), float(merged_d["PrevL"].iloc[i]), float(merged_d["PrevC"].iloc[i])
                z_top = d2o if d2c < d2o else min(d1c, max(d1o, d1l + float(d_atr) * 0.25))
                z_bot = d2l if d2c < d2o else d1l
                if z_top > z_bot:
                    active_dem.append({"start": i, "end": i, "top": z_top, "bot": z_bot, "label": "BUY ZONE (D)", "active": True})
                    if len(active_dem) > config["max_sd_zones"]:
                        active_dem.pop(0)
            if bool(merged_d["BearImp"].iloc[i]):
                d1o, d1h, d1c = float(merged_d["Open"].iloc[i]), float(merged_d["High"].iloc[i]), float(merged_d["Close"].iloc[i])
                d2o, d2h, d2c = float(merged_d["PrevO"].iloc[i]), float(merged_d["PrevH"].iloc[i]), float(merged_d["PrevC"].iloc[i])
                z_top = d2h if d2c > d2o else d1h
                z_bot = d2o if d2c > d2o else max(d1c, min(d1o, d1h - float(d_atr) * 0.25))
                if z_top > z_bot:
                    active_sup.append({"start": i, "end": i, "top": z_top, "bot": z_bot, "label": "SELL ZONE (D)", "active": True})
                    if len(active_sup) > config["max_sd_zones"]:
                        active_sup.pop(0)

        in_buy_zone, in_sell_zone = False, False
        for z in active_dem[:]:
            if c < z["bot"]:
                z["end"] = i
                z["active"] = False
                closed_dem.append(z)
                active_dem.remove(z)
            else:
                z["end"] = i
                if l <= z["top"] and c >= z["bot"]:
                    in_buy_zone = True

        for z in active_sup[:]:
            if c > z["top"]:
                z["end"] = i
                z["active"] = False
                closed_sup.append(z)
                active_sup.remove(z)
            else:
                z["end"] = i
                if h >= z["bot"] and c <= z["top"]:
                    in_sell_zone = True

        # B. 4H HTF Liquidity Zones unlock
        while htf_ptr < len(sorted_htf_times) and sorted_htf_times[htf_ptr] <= t:
            for z_type, z_top, z_bot in htf_events[sorted_htf_times[htf_ptr]]:
                if z_type == "BSL":
                    active_htf_bsl.append({"start": i, "end": i, "top": z_top, "bot": z_bot, "label": "HTF BSL (4H)", "active": True})
                    if len(active_htf_bsl) > config["max_htf_pools"]:
                        active_htf_bsl.pop(0)
                else:
                    active_htf_ssl.append({"start": i, "end": i, "top": z_top, "bot": z_bot, "label": "HTF SSL (4H)", "active": True})
                    if len(active_htf_ssl) > config["max_htf_pools"]:
                        active_htf_ssl.pop(0)
            htf_ptr += 1

        in_htf_bsl, in_htf_ssl = False, False
        htf_bear_sweep, htf_bull_sweep = False, False
        for z in active_htf_bsl[:]:
            if c > z["top"]:
                z["end"] = i
                z["active"] = False
                closed_htf_bsl.append(z)
                active_htf_bsl.remove(z)
            else:
                z["end"] = i
                if h >= z["bot"]:
                    in_htf_bsl = True
                    if h > z["top"] and c <= z["top"]:
                        htf_bear_sweep = True

        for z in active_htf_ssl[:]:
            if c < z["bot"]:
                z["end"] = i
                z["active"] = False
                closed_htf_ssl.append(z)
                active_htf_ssl.remove(z)
            else:
                z["end"] = i
                if l <= z["top"]:
                    in_htf_ssl = True
                    if l < z["bot"] and c >= z["bot"]:
                        htf_bull_sweep = True

        # C. Dynamic Trendlines
        if i >= 2 * tl_len:
            p_idx = i - tl_len
            if lows[p_idx] == np.min(lows[p_idx - tl_len : p_idx + tl_len + 1]):
                prev_pl = last_pl
                last_pl = (p_idx, float(lows[p_idx]))
                if prev_pl is not None and last_pl[1] > prev_pl[1]:
                    if active_up_tl is not None:
                        active_up_tl["end"] = i
                        active_up_tl["active"] = False
                        closed_tls.append(active_up_tl)
                    active_up_tl = {"x1": prev_pl[0], "y1": prev_pl[1], "x2": last_pl[0], "y2": last_pl[1], "created": i, "end": i, "type": "UP", "active": True}

            if highs[p_idx] == np.max(highs[p_idx - tl_len : p_idx + tl_len + 1]):
                prev_ph = last_ph
                last_ph = (p_idx, float(highs[p_idx]))
                if prev_ph is not None and last_ph[1] < prev_ph[1]:
                    if active_dn_tl is not None:
                        active_dn_tl["end"] = i
                        active_dn_tl["active"] = False
                        closed_tls.append(active_dn_tl)
                    active_dn_tl = {"x1": prev_ph[0], "y1": prev_ph[1], "x2": last_ph[0], "y2": last_ph[1], "created": i, "end": i, "type": "DN", "active": True}

        up_tl_bounce, up_tl_break = False, False
        if active_up_tl is not None:
            tl_p = _line_price_at(active_up_tl["x1"], active_up_tl["y1"], active_up_tl["x2"], active_up_tl["y2"], i)
            if c < tl_p:
                if i > active_up_tl["created"]:
                    up_tl_break = True
                active_up_tl["end"] = i
                active_up_tl["active"] = False
                closed_tls.append(active_up_tl)
                active_up_tl = None
            else:
                active_up_tl["end"] = i
                if l <= tl_p + atr * config["tl_touch_atr"] and c > tl_p:
                    up_tl_bounce = True

        dn_tl_bounce, dn_tl_break = False, False
        if active_dn_tl is not None:
            tl_p = _line_price_at(active_dn_tl["x1"], active_dn_tl["y1"], active_dn_tl["x2"], active_dn_tl["y2"], i)
            if c > tl_p:
                if i > active_dn_tl["created"]:
                    dn_tl_break = True
                active_dn_tl["end"] = i
                active_dn_tl["active"] = False
                closed_tls.append(active_dn_tl)
                active_dn_tl = None
            else:
                active_dn_tl["end"] = i
                if h >= tl_p - atr * config["tl_touch_atr"] and c < tl_p:
                    dn_tl_bounce = True

        # D. Chart Swing Liquidity & Pivot Sweeps
        if i >= 2 * sw_len:
            sp = i - sw_len
            if highs[sp] == np.max(highs[sp - sw_len : sp + sw_len + 1]):
                bsl_pools.append(float(highs[sp]))
                if len(bsl_pools) > 8:
                    bsl_pools.pop(0)
            if lows[sp] == np.min(lows[sp - sw_len : sp + sw_len + 1]):
                ssl_pools.append(float(lows[sp]))
                if len(ssl_pools) > 8:
                    ssl_pools.pop(0)

        swept_bsl, swept_ssl = False, False
        for lvl in bsl_pools[:]:
            if h > lvl:
                if c < lvl:
                    swept_bsl = True
                bsl_pools.remove(lvl)
        for lvl in ssl_pools[:]:
            if l < lvl:
                if c > lvl:
                    swept_ssl = True
                ssl_pools.remove(lvl)

        prev_c = float(closes[i - 1]) if i > 0 else c
        bull_piv_sweep = any(
            pd.notna(lvl) and l < lvl and c > lvl and (o > lvl or prev_c > lvl)
            for lvl in [pp, s1, s2, r1, r2]
        )
        bear_piv_sweep = any(
            pd.notna(lvl) and h > lvl and c < lvl and (o < lvl or prev_c < lvl)
            for lvl in [pp, r1, r2, s1, s2]
        )

        bull_sweep_ev = swept_ssl or bull_piv_sweep or htf_bull_sweep
        bear_sweep_ev = swept_bsl or bear_piv_sweep or htf_bear_sweep

        if bull_sweep_ev:
            bull_sweeps_idx.append(i)
            last_bull_sweep_bar, last_bull_sweep_low = i, l
        elif last_bull_sweep_bar is not None and c < last_bull_sweep_low:
            last_bull_sweep_bar, last_bull_sweep_low = None, None

        if bear_sweep_ev:
            bear_sweeps_idx.append(i)
            last_bear_sweep_bar, last_bear_sweep_high = i, h
        elif last_bear_sweep_bar is not None and c > last_bear_sweep_high:
            last_bear_sweep_bar, last_bear_sweep_high = None, None

        # E. Candlestick Confirmation
        c_range = h - l
        c_body = abs(c - o)
        up_wick = h - max(o, c)
        lo_wick = min(o, c) - l
        has_rng = c_range > 0
        pw = config["pin_wick_ratio"]

        bull_pin = has_rng and lo_wick >= c_range * pw and c_body <= c_range * (1 - pw) and c >= l + c_range * 0.5
        bear_pin = has_rng and up_wick >= c_range * pw and c_body <= c_range * (1 - pw) and c <= h - c_range * 0.5
        prev_o = float(opens[i - 1]) if i > 0 else o
        bull_eng = i > 0 and c > o and prev_c < prev_o and c >= prev_o and o <= prev_c and c_body > abs(prev_c - prev_o)
        bear_eng = i > 0 and c < o and prev_c > prev_o and c <= prev_o and o >= prev_c and c_body > abs(prev_c - prev_o)

        bull_cnd_ok = bull_pin or bull_eng
        bear_cnd_ok = bear_pin or bear_eng

        # F. AI Features & Scoring
        norm_rsi = float(rsis[i]) / 100.0
        rel_vol = min(float(vols[i]) / float(vol_smas[i]), 3.0) / 3.0 if vol_smas[i] > 0 else 0.5
        cnd_qual = max(up_wick, lo_wick) / c_range if has_rng else 0.0
        bull_zn_f = (0.5 if in_buy_zone else 0.0) + (0.5 if in_htf_ssl else 0.0)
        bear_zn_f = (0.5 if in_sell_zone else 0.0) + (0.5 if in_htf_bsl else 0.0)
        bull_tl_f = 1.0 if (up_tl_bounce or dn_tl_break) else 0.0
        bear_tl_f = 1.0 if (dn_tl_bounce or up_tl_break) else 0.0

        bull_ai = compute_ai_score(norm_rsi, rel_vol, cnd_qual, bull_zn_f, bull_tl_f, True)
        bear_ai = compute_ai_score(norm_rsi, rel_vol, cnd_qual, bear_zn_f, bear_tl_f, False)

        # G. Entry Logic
        bull_win = last_bull_sweep_bar is not None and (i - last_bull_sweep_bar < config["sweep_window"]) and c > last_bull_sweep_low
        bear_win = last_bear_sweep_bar is not None and (i - last_bear_sweep_bar < config["sweep_window"]) and c < last_bear_sweep_high

        bull_tl_sig = config["tl_in_signals"] and (up_tl_bounce or dn_tl_break)
        bear_tl_sig = config["tl_in_signals"] and (dn_tl_bounce or up_tl_break)

        if config["entry_mode"] == "HTF / Daily Zone Confluence Only":
            bull_setup = (bull_win or bull_tl_sig) and (in_buy_zone or in_htf_ssl)
            bear_setup = (bear_win or bear_tl_sig) and (in_sell_zone or in_htf_bsl)
        else:
            bull_setup = bull_win or in_buy_zone or in_htf_ssl or bull_tl_sig
            bear_setup = bear_win or in_sell_zone or in_htf_bsl or bear_tl_sig

        bull_bias_ok = (not config["use_pp_bias"]) or (pd.notna(pp) and c > pp)
        bear_bias_ok = (not config["use_pp_bias"]) or (pd.notna(pp) and c < pp)

        bull_ai_ok = (not config["use_ai_filter"]) or (bull_ai >= config["min_ai_conf"])
        bear_ai_ok = (not config["use_ai_filter"]) or (bear_ai >= config["min_ai_conf"])

        long_sig = bull_setup and (bull_cnd_ok or dn_tl_break) and bull_bias_ok and bull_ai_ok
        short_sig = bear_setup and (bear_cnd_ok or up_tl_break) and bear_bias_ok and bear_ai_ok

        def _build_reasons(is_long):
            r = []
            if is_long:
                if bull_win: r.append("Liquidity/Pivot Sweep")
                if in_buy_zone: r.append("Daily Buy Zone")
                if in_htf_ssl: r.append("4H HTF SSL Zone")
                if up_tl_bounce: r.append("Uptrend Bounce")
                if dn_tl_break: r.append("Downtrend Break")
                if bull_pin: r.append("Bull Pin Bar")
                if bull_eng: r.append("Bull Engulfing")
            else:
                if bear_win: r.append("Liquidity/Pivot Sweep")
                if in_sell_zone: r.append("Daily Sell Zone")
                if in_htf_bsl: r.append("4H HTF BSL Zone")
                if dn_tl_bounce: r.append("Downtrend Reject")
                if up_tl_break: r.append("Uptrend Break")
                if bear_pin: r.append("Bear Pin Bar")
                if bear_eng: r.append("Bear Engulfing")
            return " + ".join(r)

        if long_sig and open_trade is None:
            sl_anchor = last_bull_sweep_low if bull_win else float(np.min(lows[max(0, i - 2) : i + 1]))
            sl_p = sl_anchor - atr * config["sl_buffer_atr"]
            risk = c - sl_p
            if risk > 0:
                tp_p = c + risk * config["rr_ratio"]
                reasons = _build_reasons(True)
                open_trade = {
                    "side": "LONG", "entry_time": t, "entry_idx": i,
                    "entry": c, "sl": sl_p, "tp": tp_p, "ai_score": bull_ai,
                    "reasons": reasons,
                    "features": (norm_rsi, rel_vol, cnd_qual, bull_zn_f, bull_tl_f),
                }
                signals_log.append({
                    "time": t, "idx": i, "side": "BUY", "price": c,
                    "sl": sl_p, "tp": tp_p, "ai_score": bull_ai, "reasons": reasons,
                })
                last_bull_sweep_bar, last_bull_sweep_low = None, None

        elif short_sig and open_trade is None:
            sl_anchor = last_bear_sweep_high if bear_win else float(np.max(highs[max(0, i - 2) : i + 1]))
            sl_p = sl_anchor + atr * config["sl_buffer_atr"]
            risk = sl_p - c
            if risk > 0:
                tp_p = c - risk * config["rr_ratio"]
                reasons = _build_reasons(False)
                open_trade = {
                    "side": "SHORT", "entry_time": t, "entry_idx": i,
                    "entry": c, "sl": sl_p, "tp": tp_p, "ai_score": bear_ai,
                    "reasons": reasons,
                    "features": (norm_rsi, rel_vol, cnd_qual, bear_zn_f, bear_tl_f),
                }
                signals_log.append({
                    "time": t, "idx": i, "side": "SELL", "price": c,
                    "sl": sl_p, "tp": tp_p, "ai_score": bear_ai, "reasons": reasons,
                })
                last_bear_sweep_bar, last_bear_sweep_high = None, None

        if i == n - 1:
            last_bull_ai, last_bear_ai = bull_ai, bear_ai
            if in_buy_zone: last_zone_status = "D-BUY ZONE"
            elif in_sell_zone: last_zone_status = "D-SELL ZONE"
            elif in_htf_ssl: last_zone_status = "4H HTF SSL"
            elif in_htf_bsl: last_zone_status = "4H HTF BSL"
            else: last_zone_status = "NEUTRAL"

    wins_cnt = sum(1 for tr in trades if tr["pnl_r"] > 0)
    win_rate = (wins_cnt / len(trades) * 100.0) if trades else 0.0
    ai_mem_wr = (sum(m[5] for m in ai_memory) / len(ai_memory) * 100.0) if ai_memory else 0.0

    return {
        "df": df,
        "active_dem": active_dem, "closed_dem": closed_dem,
        "active_sup": active_sup, "closed_sup": closed_sup,
        "active_htf_bsl": active_htf_bsl, "closed_htf_bsl": closed_htf_bsl,
        "active_htf_ssl": active_htf_ssl, "closed_htf_ssl": closed_htf_ssl,
        "active_up_tl": active_up_tl, "active_dn_tl": active_dn_tl, "closed_tls": closed_tls,
        "bull_sweeps_idx": bull_sweeps_idx, "bear_sweeps_idx": bear_sweeps_idx,
        "signals": signals_log, "trades": trades, "open_trade": open_trade,
        "stats": {
            "last_price": float(closes[-1]),
            "last_time": times[-1],
            "pp": float(pps[-1]) if pd.notna(pps[-1]) else np.nan,
            "r1": float(r1s[-1]) if pd.notna(r1s[-1]) else np.nan,
            "s1": float(s1s[-1]) if pd.notna(s1s[-1]) else np.nan,
            "bias": "BULLISH (>PP)" if (pd.notna(pps[-1]) and closes[-1] > pps[-1]) else "BEARISH (<PP)",
            "zone_status": last_zone_status,
            "bull_ai": last_bull_ai, "bear_ai": last_bear_ai,
            "total_trades": len(trades), "win_rate": win_rate,
            "ai_mem_len": len(ai_memory), "ai_mem_wr": ai_mem_wr,
        },
    }
