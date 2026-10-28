"""
NSE Stock Analysis API  v2.0
================================================================================
FastAPI backend for the "Financial Data Analytics & Visualisation" project.
Data source: Yahoo Finance via yfinance (100% free, no API key).

ORIGINAL ENDPOINTS (unchanged, your current frontend keeps working)
    GET /                       health check + list of endpoints
    GET /analyze/{symbol}       technical + fundamental analysis
    GET /chart/{symbol}         OHLC + RSI/MACD/MA series
    GET /watchlist?symbols=     multi-stock snapshot

NEW FEATURES IN v2
    1. GET /signal/{symbol}     Buy / Hold / Sell score with plain-English reasons
    2. GET /risk/{symbol}       volatility, Sharpe, Sortino, max drawdown, VaR, beta vs Nifty,
                                return histogram, monthly-returns heatmap data
    3. GET /compare             compare 2-5 stocks: rebased performance + correlation matrix
    4. GET /backtest/{symbol}   test SMA-cross / RSI / MACD strategies vs buy-and-hold
    5. GET /portfolio           analyse a portfolio (weights, risk, contributions, vs Nifty)
    6. GET /optimize            Monte-Carlo efficient frontier (max-Sharpe & min-volatility mix)
    7. GET /market              indices, sector heatmap, top gainers/losers, market breadth
       GET /forecast/{symbol}   Monte-Carlo price simulation (probability cone)
       GET /levels/{symbol}     support/resistance, pivot points, ATR, stochastic, volume spike

REQUIREMENTS   (see requirements.txt)
    pip install fastapi uvicorn yfinance pandas numpy

RUN LOCALLY
    uvicorn nse_analysis_api:app --reload --port 8000
    open http://127.0.0.1:8000/docs      <- interactive Swagger page to test every endpoint

RUN ON RENDER
    Build command : pip install -r requirements.txt
    Start command : uvicorn nse_analysis_api:app --host 0.0.0.0 --port $PORT

NOTE: Everything here is for education only. It is NOT investment advice.
"""

import math
import time
from datetime import datetime
from functools import wraps

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="NSE Stock Analysis API",
    version="2.0",
    description="Free NSE analytics: indicators, risk, backtesting, portfolio optimisation, market overview.",
)

# Allow any frontend (v0.dev, Lovable, Vercel...) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CONSTANTS ──────────────────────────────────────────────────────────────
TRADING_DAYS = 252
RISK_FREE_RATE = 0.065          # ~ Indian 10-yr government bond yield, used for Sharpe/Sortino
BENCHMARK = "^NSEI"             # Nifty 50 index on Yahoo Finance
CACHE_TTL_SECONDS = 300         # 5 min cache -> faster responses and fewer Yahoo rate-limit errors
DISCLAIMER = "Educational analysis only. Not investment advice. Past performance does not predict future results."

INDICES = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANK NIFTY": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
}

# Sector indices (Yahoo symbols). If Yahoo has no data for one, it is skipped automatically.
SECTOR_INDICES = {
    "Bank": "^NSEBANK",
    "IT": "^CNXIT",
    "Auto": "^CNXAUTO",
    "Pharma": "^CNXPHARMA",
    "FMCG": "^CNXFMCG",
    "Metal": "^CNXMETAL",
    "Energy": "^CNXENERGY",
    "Realty": "^CNXREALTY",
    "Media": "^CNXMEDIA",
    "PSU Bank": "^CNXPSUBANK",
}

# Approximate Nifty-50 constituents. Index membership changes, so edit freely.
# Any symbol Yahoo cannot find is skipped automatically.
NIFTY50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO",
    "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
    "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY", "ITC", "JIOFIN",
    "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM",
    "TATASTEEL", "TCS", "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]

# ── SMALL UTILITIES ────────────────────────────────────────────────────────

_CACHE = {}


def cached(key, ttl, fn):
    """Tiny in-memory TTL cache so repeated calls don't hammer Yahoo."""
    now = time.time()
    if key in _CACHE and now - _CACHE[key][0] < ttl:
        return _CACHE[key][1]
    value = fn()
    _CACHE[key] = (now, value)
    return value


def safe_round(val, decimals=2):
    if val is None:
        return None
    try:
        if pd.isna(val) or (isinstance(val, float) and math.isinf(val)):
            return None
        return round(float(val), decimals)
    except (TypeError, ValueError):
        return None


def to_ticker(symbol):
    s = symbol.upper().strip()
    if s.startswith("^") or s.endswith(".NS") or s.endswith(".BO"):
        return s
    return f"{s}.NS"


def label(ticker):
    names = {"^NSEI": "NIFTY 50", "^BSESN": "SENSEX", "^NSEBANK": "BANK NIFTY", "^CNXIT": "NIFTY IT"}
    return names.get(ticker, ticker.replace(".NS", "").replace(".BO", ""))


def parse_symbols(symbols, min_n, max_n):
    out = []
    for s in symbols.split(","):
        s = s.strip()
        if s:
            t = to_ticker(s)
            if t not in out:
                out.append(t)
    if len(out) < min_n or len(out) > max_n:
        raise HTTPException(
            status_code=400,
            detail=f"Provide between {min_n} and {max_n} comma-separated symbols, e.g. ?symbols=RELIANCE,TCS,INFY "
                   f"(use M%26M for M&M).",
        )
    return out


def api_errors(fn):
    """Turn unexpected exceptions into clean JSON errors instead of crashing."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{fn.__name__} failed: {str(e)}")
    return wrapper


def get_history(ticker, period="2y"):
    """Cached price history. Index is made timezone-naive so different stocks align cleanly."""
    def fetch():
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if hist is not None and not hist.empty:
            hist.index = pd.DatetimeIndex(hist.index).tz_localize(None).normalize()
        return hist

    hist = cached(("hist", ticker, period), CACHE_TTL_SECONDS, fetch)
    if hist is None or hist.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}. Check the symbol is correct.")
    return hist


def get_closes(tickers, period="5y"):
    """Aligned close-price DataFrame for several tickers (common trading dates only)."""
    frames = {t: get_history(t, period)["Close"] for t in tickers}
    df = pd.concat(frames, axis=1).dropna()
    if len(df) < 60:
        raise HTTPException(status_code=400, detail="Not enough overlapping price history for these symbols.")
    return df


def downsample(obj, max_points=250):
    """Keep charts light: thin a Series/DataFrame to about max_points rows (always keeps the last row)."""
    n = len(obj)
    step = max(1, n // max_points)
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return obj.iloc[idx]


def download_closes(tickers, period):
    """One batched Yahoo call for many tickers -> {ticker: close Series}. Failed tickers are skipped."""
    def fetch():
        raw = yf.download(tickers, period=period, group_by="ticker", progress=False,
                          threads=True, auto_adjust=True)
        out = {}
        for t in tickers:
            try:
                s = raw[t]["Close"].dropna() if len(tickers) > 1 else raw["Close"].dropna()
                if len(s) >= 2:
                    out[t] = s
            except Exception:
                continue
        return out

    return cached(("dl", tuple(tickers), period), CACHE_TTL_SECONDS, fetch)


# ── INDICATOR FUNCTIONS (original) ─────────────────────────────────────────

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_roc(prices, period=12):
    return prices.pct_change(period) * 100


def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def calculate_moving_averages(prices, short_window=50, long_window=200):
    return prices.rolling(short_window).mean(), prices.rolling(long_window).mean()


def calculate_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm - minus_dm) < 0] = 0
    minus_dm[(minus_dm - plus_dm) < 0] = 0
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def calculate_52w_high_proximity(prices, window=252):
    return (prices / prices.rolling(window).max()) * 100


def calculate_bollinger_bands(prices, window=20, num_std=2):
    middle = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    percent_b = (prices - lower) / (upper - lower)
    bandwidth = (upper - lower) / middle * 100
    return percent_b, bandwidth


# ── INDICATOR FUNCTIONS (new) ──────────────────────────────────────────────

def calculate_atr(high, low, close, period=14):
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = 100 * (close - lowest) / (highest - lowest)
    return k, k.rolling(d_period).mean()


def calculate_obv(close, volume):
    return (np.sign(close.diff()).fillna(0) * volume).cumsum()


# ── FUNDAMENTALS (original, unchanged logic) ───────────────────────────────

def get_yoy_earnings_growth(net_income, tolerance_days=45):
    """Date-matched YoY growth (robust to irregularly reported quarters)."""
    if len(net_income) < 2:
        return None, None
    latest_date = net_income.index[0]
    latest_val = net_income.iloc[0]
    target_date = latest_date - pd.DateOffset(months=12)
    date_diffs = (net_income.index - target_date).days.to_series(index=net_income.index).abs()
    closest_idx = date_diffs.idxmin()
    if date_diffs[closest_idx] > tolerance_days:
        return None, None
    base_val = net_income.loc[closest_idx]
    if base_val == 0 or pd.isna(base_val):
        return None, None
    return round(((latest_val / base_val) - 1) * 100, 2), str(closest_idx.date())


def get_yoy_revenue_growth(revenue, tolerance_days=45):
    return get_yoy_earnings_growth(revenue, tolerance_days)


def get_fundamentals(stock):
    """Fundamentals calculated from income statement + balance sheet.
    Deliberately avoids stock.info, which Yahoo blocks on many cloud hosts (Render, Railway...)."""
    fundamentals = {
        "yoy_earnings_growth_pct": None,
        "earnings_base_quarter": None,
        "yoy_revenue_growth_pct": None,
        "revenue_base_quarter": None,
        "operating_margin_pct": None,
        "net_margin_pct": None,
        "roe_pct": None,
        "debt_to_equity": None,
        "market_cap_cr": None,
    }

    latest_revenue = None
    latest_net_income = None
    latest_operating_income = None

    try:
        income_stmt = stock.quarterly_income_stmt
        if income_stmt is not None:
            if "Net Income" in income_stmt.index:
                net_income = income_stmt.loc["Net Income"].dropna()
                growth, base_date = get_yoy_earnings_growth(net_income)
                fundamentals["yoy_earnings_growth_pct"] = growth
                fundamentals["earnings_base_quarter"] = base_date
                if len(net_income) > 0:
                    latest_net_income = net_income.iloc[0]

            if "Total Revenue" in income_stmt.index:
                revenue = income_stmt.loc["Total Revenue"].dropna()
                growth, base_date = get_yoy_revenue_growth(revenue)
                fundamentals["yoy_revenue_growth_pct"] = growth
                fundamentals["revenue_base_quarter"] = base_date
                if len(revenue) > 0:
                    latest_revenue = revenue.iloc[0]

            if "Operating Income" in income_stmt.index:
                op_income = income_stmt.loc["Operating Income"].dropna()
                if len(op_income) > 0:
                    latest_operating_income = op_income.iloc[0]

        if latest_revenue and latest_revenue != 0:
            if latest_net_income is not None:
                fundamentals["net_margin_pct"] = safe_round((latest_net_income / latest_revenue) * 100)
            if latest_operating_income is not None:
                fundamentals["operating_margin_pct"] = safe_round((latest_operating_income / latest_revenue) * 100)

    except Exception as e:
        fundamentals["_debug_income_stmt_error"] = str(e)

    try:
        balance_sheet = stock.quarterly_balance_sheet
        if balance_sheet is not None:
            equity = None
            total_debt = None

            for equity_label in ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]:
                if equity_label in balance_sheet.index:
                    equity_series = balance_sheet.loc[equity_label].dropna()
                    if len(equity_series) > 0:
                        equity = equity_series.iloc[0]
                        break

            for debt_label in ["Total Debt", "Net Debt"]:
                if debt_label in balance_sheet.index:
                    debt_series = balance_sheet.loc[debt_label].dropna()
                    if len(debt_series) > 0:
                        total_debt = debt_series.iloc[0]
                        break

            if equity and equity != 0:
                if total_debt is not None:
                    fundamentals["debt_to_equity"] = safe_round(total_debt / equity)
                if latest_net_income is not None:
                    fundamentals["roe_pct"] = safe_round(((latest_net_income * 4) / equity) * 100)

    except Exception as e:
        fundamentals["_debug_balance_sheet_error"] = str(e)

    try:
        fast_info = stock.fast_info
        if fast_info and fast_info.get("marketCap"):
            fundamentals["market_cap_cr"] = safe_round(fast_info["marketCap"] / 1e7)
    except Exception:
        pass

    return fundamentals


def get_technical_snapshot(hist):
    """Core technical calculation, reused by /analyze, /watchlist and /signal."""
    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]

    daily_pct_change = close.pct_change()
    corporate_action_flag = bool((daily_pct_change.abs() > 0.15).any())

    rsi = calculate_rsi(close)
    roc = calculate_roc(close)
    macd_line, signal_line, macd_hist = calculate_macd(close)
    ma50, ma200 = calculate_moving_averages(close)
    adx = calculate_adx(high, low, close)
    high52 = calculate_52w_high_proximity(close)
    percent_b, bandwidth = calculate_bollinger_bands(close)

    ma_trend = None
    if not pd.isna(ma50.iloc[-1]) and not pd.isna(ma200.iloc[-1]):
        ma_trend = "bullish" if ma50.iloc[-1] > ma200.iloc[-1] else "bearish"

    technical = {
        "current_price": safe_round(close.iloc[-1]),
        "rsi_14": safe_round(rsi.iloc[-1]),
        "roc_12": safe_round(roc.iloc[-1]),
        "macd_line": safe_round(macd_line.iloc[-1]),
        "macd_signal": safe_round(signal_line.iloc[-1]),
        "macd_histogram": safe_round(macd_hist.iloc[-1]),
        "ma_50": safe_round(ma50.iloc[-1]),
        "ma_200": safe_round(ma200.iloc[-1]),
        "ma_trend": ma_trend,
        "adx_14": safe_round(adx.iloc[-1]),
        "pct_of_52w_high": safe_round(high52.iloc[-1]),
        "bollinger_percent_b": safe_round(percent_b.iloc[-1], 3),
        "bollinger_bandwidth_pct": safe_round(bandwidth.iloc[-1]),
    }
    return technical, corporate_action_flag


# ── RISK MATH (shared by /risk, /compare, /portfolio, /backtest) ───────────

def risk_stats(returns, rf=RISK_FREE_RATE):
    """Standard risk/return statistics from a Series of daily simple returns."""
    r = returns.dropna()
    n = len(r)
    if n < 20:
        return {}
    daily_rf = rf / TRADING_DAYS
    std = r.std()
    total = float((1 + r).prod() - 1)
    years = n / TRADING_DAYS
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else None

    sharpe = ((r.mean() - daily_rf) / std) * math.sqrt(TRADING_DAYS) if std > 0 else None
    downside = np.sqrt(np.mean(np.minimum(r - daily_rf, 0) ** 2))
    sortino = ((r.mean() - daily_rf) / downside) * math.sqrt(TRADING_DAYS) if downside > 0 else None

    equity = (1 + r).cumprod()
    drawdown = equity / equity.cummax() - 1
    trough = drawdown.idxmin()
    peak = equity.loc[:trough].idxmax()

    var95 = float(np.percentile(r, 5))
    tail = r[r <= var95]
    cvar95 = float(tail.mean()) if len(tail) else None

    return {
        "total_return_pct": safe_round(total * 100),
        "cagr_pct": safe_round(cagr * 100) if cagr is not None else None,
        "annual_volatility_pct": safe_round(std * math.sqrt(TRADING_DAYS) * 100),
        "sharpe_ratio": safe_round(sharpe),
        "sortino_ratio": safe_round(sortino),
        "max_drawdown_pct": safe_round(drawdown.min() * 100),
        "max_drawdown_peak_date": str(peak.date()),
        "max_drawdown_trough_date": str(trough.date()),
        "var_95_daily_pct": safe_round(var95 * 100),
        "cvar_95_daily_pct": safe_round(cvar95 * 100) if cvar95 is not None else None,
        "best_day_pct": safe_round(r.max() * 100),
        "worst_day_pct": safe_round(r.min() * 100),
        "positive_days_pct": safe_round((r > 0).mean() * 100),
        "trading_days": int(n),
    }


def drawdown_series(returns):
    equity = (1 + returns.dropna()).cumprod()
    return equity / equity.cummax() - 1


# ═══════════════════════════════════════════════════════════════════════════
#  ORIGINAL ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "NSE Stock Analysis API v2 is running",
        "docs": "/docs",
        "endpoints": [
            "/analyze/{symbol}", "/chart/{symbol}", "/watchlist?symbols=",
            "/signal/{symbol}", "/risk/{symbol}", "/compare?symbols=",
            "/backtest/{symbol}", "/portfolio?holdings=", "/optimize?symbols=",
            "/market", "/forecast/{symbol}", "/levels/{symbol}",
        ],
    }


@app.get("/analyze/{symbol}")
@api_errors
def analyze_stock(symbol: str):
    """Full technical + fundamental analysis for one NSE-listed company."""
    ticker = to_ticker(symbol)
    hist = get_history(ticker, "2y")
    technical, corporate_action_flag = get_technical_snapshot(hist)
    fundamentals = get_fundamentals(yf.Ticker(ticker))

    return {
        "symbol": ticker,
        "company_name": ticker,
        "timestamp": datetime.now().isoformat(),
        "corporate_action_flag": corporate_action_flag,
        "corporate_action_note": (
            "A >15% single-day price move was detected in the last 2 years — "
            "moving-average and Bollinger figures may be distorted by a demerger, "
            "split, or similar event. Verify before trusting the technical signals."
            if corporate_action_flag else None
        ),
        "technical": technical,
        "fundamentals": fundamentals,
    }


@app.get("/chart/{symbol}")
@api_errors
def chart_data(symbol: str, months: int = 6):
    """OHLC + indicator series for charting. `months` = 1..24."""
    ticker = to_ticker(symbol)
    months = min(max(months, 1), 24)
    hist = get_history(ticker, "2y")

    close = hist["Close"]
    rsi = calculate_rsi(close)
    macd_line, signal_line, macd_hist = calculate_macd(close)
    ma50, ma200 = calculate_moving_averages(close)
    upper_pb, _ = calculate_bollinger_bands(close)  # kept for future use
    mid = close.rolling(20).mean()
    sd = close.rolling(20).std()

    recent = hist.tail(months * 21)
    rows = []
    for date, row in recent.iterrows():
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": safe_round(row["Open"]),
            "high": safe_round(row["High"]),
            "low": safe_round(row["Low"]),
            "close": safe_round(row["Close"]),
            "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else None,
            "rsi": safe_round(rsi.loc[date]),
            "macd_line": safe_round(macd_line.loc[date]),
            "macd_signal": safe_round(signal_line.loc[date]),
            "macd_histogram": safe_round(macd_hist.loc[date]),
            "ma_50": safe_round(ma50.loc[date]),
            "ma_200": safe_round(ma200.loc[date]),
            "bb_upper": safe_round(mid.loc[date] + 2 * sd.loc[date]),
            "bb_lower": safe_round(mid.loc[date] - 2 * sd.loc[date]),
        })
    return {"symbol": ticker, "months": months, "data": rows}


@app.get("/watchlist")
@api_errors
def watchlist(symbols: str):
    """Lightweight multi-stock snapshot: /watchlist?symbols=RELIANCE,TCS,INFY (max 10)."""
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:10]
    if not symbol_list:
        raise HTTPException(status_code=400, detail="Provide at least one symbol, e.g. ?symbols=RELIANCE,TCS")

    results = []
    for symbol in symbol_list:
        ticker = to_ticker(symbol)
        try:
            hist = get_history(ticker, "2y")
            technical, flag = get_technical_snapshot(hist)
            close = hist["Close"]
            day_change = (close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(close) > 1 else None
            results.append({
                "symbol": ticker,
                "company_name": ticker,
                "current_price": technical["current_price"],
                "day_change_pct": safe_round(day_change),
                "rsi_14": technical["rsi_14"],
                "macd_histogram": technical["macd_histogram"],
                "ma_trend": technical["ma_trend"],
                "pct_of_52w_high": technical["pct_of_52w_high"],
                "corporate_action_flag": flag,
            })
        except HTTPException as e:
            results.append({"symbol": ticker, "error": e.detail})
        except Exception as e:
            results.append({"symbol": ticker, "error": str(e)})
    return {"count": len(results), "stocks": results}


# ═══════════════════════════════════════════════════════════════════════════
#  NEW FEATURE 1 — BUY / HOLD / SELL SIGNAL
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/signal/{symbol}")
@api_errors
def signal(symbol: str):
    """Rule-based scorecard combining RSI, MACD, moving averages, ADX, Bollinger and 52-week position."""
    ticker = to_ticker(symbol)
    hist = get_history(ticker, "2y")
    t, flag = get_technical_snapshot(hist)

    score = 0
    reasons = []

    def add(points, text):
        nonlocal score
        score += points
        reasons.append({"points": points, "reason": text})

    price, rsi, hist_macd = t["current_price"], t["rsi_14"], t["macd_histogram"]

    if rsi is not None:
        if rsi < 30:
            add(2, f"RSI {rsi} is below 30: oversold, a rebound is more likely")
        elif rsi < 40:
            add(1, f"RSI {rsi} is low (under 40): selling pressure is easing")
        elif rsi > 70:
            add(-2, f"RSI {rsi} is above 70: overbought, a pullback is more likely")
        elif rsi > 65:
            add(-1, f"RSI {rsi} is getting stretched (above 65)")
        else:
            add(0, f"RSI {rsi} is in the neutral zone")

    if hist_macd is not None:
        if hist_macd > 0:
            add(1, "MACD histogram is positive: bullish momentum")
        else:
            add(-1, "MACD histogram is negative: bearish momentum")

    if t["ma_trend"] == "bullish":
        add(1, "50-day MA is above 200-day MA (golden-cross territory): long-term uptrend")
    elif t["ma_trend"] == "bearish":
        add(-1, "50-day MA is below 200-day MA (death-cross territory): long-term downtrend")

    if price is not None and t["ma_50"] is not None:
        if price > t["ma_50"]:
            add(1, "Price is above its 50-day moving average")
        else:
            add(-1, "Price is below its 50-day moving average")

    adx = t["adx_14"]
    if adx is not None and adx > 25 and hist_macd is not None and price is not None and t["ma_50"] is not None:
        if hist_macd > 0 and price > t["ma_50"]:
            add(1, f"ADX {adx} shows a strong trend and it points up")
        elif hist_macd < 0 and price < t["ma_50"]:
            add(-1, f"ADX {adx} shows a strong trend and it points down")

    pb = t["bollinger_percent_b"]
    if pb is not None:
        if pb < 0.05:
            add(1, "Price is hugging the lower Bollinger Band: possible oversold bounce")
        elif pb > 0.95:
            add(-1, "Price is hugging the upper Bollinger Band: possible overextension")

    p52 = t["pct_of_52w_high"]
    if p52 is not None:
        if p52 >= 95:
            add(1, f"Trading at {p52}% of its 52-week high: strong relative strength")
        elif p52 < 60:
            add(-1, f"Trading at only {p52}% of its 52-week high: weak relative strength")

    if score >= 5:
        verdict = "STRONG BUY"
    elif score >= 2:
        verdict = "BUY"
    elif score <= -5:
        verdict = "STRONG SELL"
    elif score <= -2:
        verdict = "SELL"
    else:
        verdict = "HOLD"

    return {
        "symbol": ticker,
        "verdict": verdict,
        "score": score,
        "score_range": [-8, 8],
        "confidence_pct": safe_round(min(abs(score) / 8, 1) * 100, 0),
        "reasons": reasons,
        "technical": t,
        "corporate_action_flag": flag,
        "disclaimer": DISCLAIMER,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  NEW FEATURE 2 — RISK DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/risk/{symbol}")
@api_errors
def risk(symbol: str, years: int = 3):
    """Volatility, Sharpe, Sortino, drawdown, VaR, beta vs Nifty, plus chart-ready series."""
    ticker = to_ticker(symbol)
    years = min(max(years, 1), 5)
    close = get_history(ticker, "5y")["Close"].tail(years * TRADING_DAYS)
    rets = close.pct_change().dropna()
    stats = risk_stats(rets)

    # Beta / alpha / correlation vs Nifty 50
    market = {"beta": None, "alpha_annual_pct": None, "correlation": None, "r_squared": None}
    try:
        bench = get_history(BENCHMARK, "5y")["Close"].pct_change()
        both = pd.concat([rets.rename("s"), bench.rename("b")], axis=1, join="inner").dropna()
        if len(both) > 60 and both["b"].var() > 0:
            beta = both["s"].cov(both["b"]) / both["b"].var()
            d_rf = RISK_FREE_RATE / TRADING_DAYS
            alpha = (both["s"].mean() - d_rf - beta * (both["b"].mean() - d_rf)) * TRADING_DAYS
            corr = both["s"].corr(both["b"])
            market = {
                "beta": safe_round(beta),
                "alpha_annual_pct": safe_round(alpha * 100),
                "correlation": safe_round(corr, 3),
                "r_squared": safe_round(corr ** 2, 3),
            }
    except Exception:
        pass

    beta = market["beta"]
    if beta is None:
        beta_note = None
    elif beta > 1.2:
        beta_note = "Aggressive: tends to move MORE than the market"
    elif beta < 0.8:
        beta_note = "Defensive: tends to move LESS than the market"
    else:
        beta_note = "Moves roughly in line with the market"

    # chart series
    dd = downsample(drawdown_series(rets) * 100)
    roll_vol = downsample((rets.rolling(30).std() * math.sqrt(TRADING_DAYS) * 100).dropna())
    counts, edges = np.histogram(rets * 100, bins=30)
    histogram = [
        {"bin_start": safe_round(edges[i], 2), "bin_end": safe_round(edges[i + 1], 2), "count": int(counts[i])}
        for i in range(len(counts))
    ]

    # monthly returns heatmap (year -> month -> %)
    month_end = close.groupby([close.index.year, close.index.month]).last()
    month_ret = month_end.pct_change() * 100
    monthly = {}
    for (yr, mo), v in month_ret.items():
        monthly.setdefault(str(yr), {})[str(mo)] = safe_round(v)

    return {
        "symbol": ticker,
        "years": years,
        "risk_free_rate_pct": RISK_FREE_RATE * 100,
        "stats": stats,
        "market": {**market, "benchmark": "NIFTY 50", "beta_note": beta_note},
        "drawdown_series": [{"date": str(d.date()), "drawdown_pct": safe_round(v)} for d, v in dd.items()],
        "rolling_volatility_30d": [{"date": str(d.date()), "vol_pct": safe_round(v)} for d, v in roll_vol.items()],
        "return_histogram": histogram,
        "monthly_returns_pct": monthly,
        "disclaimer": DISCLAIMER,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  NEW FEATURE 3 — COMPARE STOCKS + CORRELATION
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/compare")
@api_errors
def compare(symbols: str, months: int = 12, include_nifty: bool = True):
    """Compare 2-5 stocks: performance rebased to 100, risk stats and correlation matrix."""
    tickers = parse_symbols(symbols, 2, 5)
    months = min(max(months, 1), 60)
    if include_nifty and BENCHMARK not in tickers:
        tickers.append(BENCHMARK)

    closes = get_closes(tickers, "5y").tail(months * 21)
    rebased = closes / closes.iloc[0] * 100
    rets = closes.pct_change().dropna()
    names = [label(t) for t in tickers]

    series = []
    for date, row in downsample(rebased).iterrows():
        point = {"date": str(date.date())}
        for t in tickers:
            point[label(t)] = safe_round(row[t])
        series.append(point)

    summary = []
    for t in tickers:
        s = risk_stats(rets[t])
        summary.append({
            "symbol": label(t),
            "return_pct": s.get("total_return_pct"),
            "annual_volatility_pct": s.get("annual_volatility_pct"),
            "sharpe_ratio": s.get("sharpe_ratio"),
            "max_drawdown_pct": s.get("max_drawdown_pct"),
        })

    corr = rets.corr()
    matrix = [[safe_round(corr.loc[a, b], 2) for b in tickers] for a in tickers]

    ranked = sorted([x for x in summary if x["return_pct"] is not None], key=lambda x: x["return_pct"], reverse=True)
    return {
        "months": months,
        "labels": names,
        "performance_rebased_100": series,
        "summary": summary,
        "best_performer": ranked[0]["symbol"] if ranked else None,
        "worst_performer": ranked[-1]["symbol"] if ranked else None,
        "correlation": {"labels": names, "matrix": matrix},
    }


# ═══════════════════════════════════════════════════════════════════════════
#  NEW FEATURE 4 — STRATEGY BACKTESTER
# ═══════════════════════════════════════════════════════════════════════════

def _backtest_metrics(strat_ret, position, capital):
    equity = (1 + strat_ret).cumprod()

    # per-trade results
    entries = position.diff().fillna(position.iloc[0]) == 1
    trade_id = entries.cumsum() * (position == 1)
    in_trade = trade_id > 0
    trade_returns = strat_ret[in_trade].groupby(trade_id[in_trade]).apply(lambda r: (1 + r).prod() - 1)

    s = risk_stats(strat_ret)
    return {
        "final_value": safe_round(capital * equity.iloc[-1], 0),
        "total_return_pct": s.get("total_return_pct"),
        "cagr_pct": s.get("cagr_pct"),
        "annual_volatility_pct": s.get("annual_volatility_pct"),
        "sharpe_ratio": s.get("sharpe_ratio"),
        "max_drawdown_pct": s.get("max_drawdown_pct"),
        "trades": int(len(trade_returns)),
        "win_rate_pct": safe_round((trade_returns > 0).mean() * 100) if len(trade_returns) else None,
        "avg_trade_pct": safe_round(trade_returns.mean() * 100) if len(trade_returns) else None,
        "time_in_market_pct": safe_round(position.mean() * 100),
    }, equity


@app.get("/backtest/{symbol}")
@api_errors
def backtest(symbol: str, strategy: str = "sma_cross", fast: int = 50, slow: int = 200,
             rsi_buy: int = 30, rsi_sell: int = 55, years: int = 5,
             capital: float = 100000, cost_pct: float = 0.1):
    """
    Test a simple strategy against buy-and-hold on real history.
      strategy = sma_cross  -> long when fast SMA > slow SMA          (params: fast, slow)
      strategy = rsi        -> buy when RSI < rsi_buy, sell > rsi_sell
      strategy = macd       -> long when MACD line > signal line
    Signals are executed on the NEXT day's return (no look-ahead cheating).
    cost_pct is charged on every buy or sell (0.1% default ~ brokerage + taxes + slippage).
    """
    ticker = to_ticker(symbol)
    years = min(max(years, 1), 5)
    cost = max(cost_pct, 0) / 100
    close = get_history(ticker, "5y")["Close"].tail(years * TRADING_DAYS + 250)

    if strategy == "sma_cross":
        if fast >= slow or fast < 2:
            raise HTTPException(status_code=400, detail="fast must be smaller than slow (and at least 2).")
        f, s_ = close.rolling(fast).mean(), close.rolling(slow).mean()
        raw = (f > s_).astype(float)
        valid_start = s_.first_valid_index()
        desc = f"Long when {fast}-day SMA is above {slow}-day SMA, otherwise cash"
    elif strategy == "rsi":
        rsi_vals = calculate_rsi(close)
        pos, holding = [], False
        for v in rsi_vals.values:
            if not np.isnan(v):
                if not holding and v < rsi_buy:
                    holding = True
                elif holding and v > rsi_sell:
                    holding = False
            pos.append(1.0 if holding else 0.0)
        raw = pd.Series(pos, index=close.index)
        valid_start = rsi_vals.first_valid_index()
        desc = f"Buy when RSI < {rsi_buy}, sell when RSI > {rsi_sell}"
    elif strategy == "macd":
        m, sg, _ = calculate_macd(close)
        raw = (m > sg).astype(float)
        valid_start = close.index[34] if len(close) > 34 else close.index[0]
        desc = "Long when MACD line is above its signal line, otherwise cash"
    else:
        raise HTTPException(status_code=400, detail="strategy must be one of: sma_cross, rsi, macd")

    position = raw.shift(1).fillna(0)           # trade on next day -> no look-ahead
    position = position.loc[valid_start:]
    px = close.loc[valid_start:]
    if len(px) < 60:
        raise HTTPException(status_code=400, detail="Not enough history for this strategy. Try fewer years or a shorter slow window.")

    daily = px.pct_change().fillna(0)
    trades_cost = position.diff().abs().fillna(position.iloc[0]) * cost
    strat_ret = position * daily - trades_cost

    strat_metrics, strat_equity = _backtest_metrics(strat_ret, position, capital)
    bh_position = pd.Series(1.0, index=px.index)
    bh_ret = daily.copy()
    bh_ret.iloc[0] = 0
    bh_metrics, bh_equity = _backtest_metrics(bh_ret, bh_position, capital)

    curve = pd.DataFrame({"strategy": strat_equity * capital, "buy_hold": bh_equity * capital, "price": px})
    curve_rows = [
        {"date": str(d.date()), "strategy": safe_round(r["strategy"], 0),
         "buy_hold": safe_round(r["buy_hold"], 0), "in_position": bool(position.loc[d] == 1)}
        for d, r in downsample(curve).iterrows()
    ]

    beat = (strat_metrics["total_return_pct"] or 0) > (bh_metrics["total_return_pct"] or 0)
    return {
        "symbol": ticker,
        "strategy": strategy,
        "description": desc,
        "period": {"start": str(px.index[0].date()), "end": str(px.index[-1].date())},
        "starting_capital": capital,
        "cost_per_trade_pct": cost_pct,
        "strategy_result": strat_metrics,
        "buy_and_hold_result": bh_metrics,
        "strategy_beat_buy_and_hold": beat,
        "equity_curve": curve_rows,
        "disclaimer": DISCLAIMER + " Backtests ignore many real-world factors and can be overfit.",
    }


# ═══════════════════════════════════════════════════════════════════════════
#  NEW FEATURE 5 — PORTFOLIO ANALYSER
# ═══════════════════════════════════════════════════════════════════════════

def parse_holdings(holdings):
    """'RELIANCE:40,TCS:30,INFY:30' -> ([tickers], np.array of weights summing to 1). Weight is optional."""
    tickers, weights = [], []
    for part in holdings.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            sym, w = part.rsplit(":", 1)
            try:
                w = float(w)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Bad weight in '{part}'. Use SYMBOL:WEIGHT, e.g. TCS:30")
        else:
            sym, w = part, 1.0
        t = to_ticker(sym)
        if t in tickers:
            continue
        tickers.append(t)
        weights.append(max(w, 0))
    if len(tickers) < 2 or len(tickers) > 10:
        raise HTTPException(status_code=400, detail="Provide 2 to 10 holdings, e.g. ?holdings=RELIANCE:40,TCS:30,INFY:30")
    w = np.array(weights, dtype=float)
    if w.sum() <= 0:
        raise HTTPException(status_code=400, detail="Weights must add up to more than zero.")
    return tickers, w / w.sum()


@app.get("/portfolio")
@api_errors
def portfolio(holdings: str, amount: float = 100000, years: int = 3):
    """
    Analyse a portfolio. holdings = SYMBOL:WEIGHT,... (weights are normalised to 100%).
    Assumes constant weights (rebalanced daily), which is the standard textbook simplification.
    """
    tickers, w = parse_holdings(holdings)
    years = min(max(years, 1), 5)
    closes = get_closes(tickers + [BENCHMARK], "5y").tail(years * TRADING_DAYS)
    stock_closes = closes[tickers]
    rets = stock_closes.pct_change().dropna()
    bench_rets = closes[BENCHMARK].pct_change().dropna()

    port_ret = (rets * w).sum(axis=1)
    stats = risk_stats(port_ret)
    bench_stats = risk_stats(bench_rets)

    cov = rets.cov() * TRADING_DAYS
    port_var = float(w @ cov.values @ w)
    port_vol = math.sqrt(port_var)
    indiv_vol = np.sqrt(np.diag(cov.values))
    diversification_ratio = float((w * indiv_vol).sum() / port_vol) if port_vol > 0 else None
    risk_contrib = (w * (cov.values @ w)) / port_var if port_var > 0 else np.zeros(len(w))

    corr = rets.corr()
    n = len(tickers)
    avg_corr = (corr.values.sum() - n) / (n * (n - 1))

    holdings_out = []
    for i, t in enumerate(tickers):
        s = risk_stats(rets[t])
        holdings_out.append({
            "symbol": label(t),
            "weight_pct": safe_round(w[i] * 100),
            "amount_inr": safe_round(amount * w[i], 0),
            "return_pct": s.get("total_return_pct"),
            "annual_volatility_pct": s.get("annual_volatility_pct"),
            "sharpe_ratio": s.get("sharpe_ratio"),
            "contribution_to_return_pct": safe_round(
                (w[i] * ((1 + rets[t]).prod() - 1)) * 100),
            "contribution_to_risk_pct": safe_round(risk_contrib[i] * 100),
        })

    warnings = []
    if w.max() > 0.4:
        warnings.append(f"{label(tickers[int(w.argmax())])} is {w.max() * 100:.0f}% of the portfolio: concentration risk.")
    if avg_corr > 0.7:
        warnings.append("Holdings are highly correlated (avg > 0.7): limited diversification benefit.")
    top_risk = int(np.argmax(risk_contrib))
    if risk_contrib[top_risk] > 0.5:
        warnings.append(f"{label(tickers[top_risk])} drives {risk_contrib[top_risk] * 100:.0f}% of total portfolio risk.")

    curve = pd.DataFrame({
        "portfolio": (1 + port_ret).cumprod() * amount,
        "nifty50": (1 + bench_rets.reindex(port_ret.index).fillna(0)).cumprod() * amount,
    })
    curve_rows = [{"date": str(d.date()), "portfolio": safe_round(r["portfolio"], 0), "nifty50": safe_round(r["nifty50"], 0)}
                  for d, r in downsample(curve).iterrows()]

    return {
        "years": years,
        "starting_amount": amount,
        "ending_value": safe_round(curve["portfolio"].iloc[-1], 0),
        "portfolio_stats": stats,
        "nifty50_stats": bench_stats,
        "diversification_ratio": safe_round(diversification_ratio),
        "average_pairwise_correlation": safe_round(avg_corr, 3),
        "holdings": holdings_out,
        "correlation": {"labels": [label(t) for t in tickers],
                        "matrix": [[safe_round(corr.loc[a, b], 2) for b in tickers] for a in tickers]},
        "warnings": warnings,
        "equity_curve": curve_rows,
        "disclaimer": DISCLAIMER,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  NEW FEATURE 6 — PORTFOLIO OPTIMISER (EFFICIENT FRONTIER)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/optimize")
@api_errors
def optimize(symbols: str, years: int = 3, simulations: int = 3000):
    """
    Monte-Carlo efficient frontier: tries thousands of random weight mixes and returns the
    max-Sharpe and min-volatility portfolios plus a scatter sample for charting.
    (Uses past returns, so treat results as illustrative.)
    """
    tickers = parse_symbols(symbols, 2, 10)
    years = min(max(years, 1), 5)
    simulations = min(max(simulations, 500), 10000)

    rets = get_closes(tickers, "5y").tail(years * TRADING_DAYS).pct_change().dropna()
    mean = rets.mean().values * TRADING_DAYS
    cov = rets.cov().values * TRADING_DAYS

    rng = np.random.default_rng(42)
    W = rng.dirichlet(np.ones(len(tickers)), size=simulations)
    p_ret = W @ mean
    p_vol = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    p_sharpe = (p_ret - RISK_FREE_RATE) / p_vol

    def pack(i):
        return {
            "expected_return_pct": safe_round(p_ret[i] * 100),
            "volatility_pct": safe_round(p_vol[i] * 100),
            "sharpe_ratio": safe_round(p_sharpe[i]),
            "weights_pct": {label(t): safe_round(W[i][k] * 100, 1) for k, t in enumerate(tickers)},
        }

    eq = np.ones(len(tickers)) / len(tickers)
    eq_vol = math.sqrt(float(eq @ cov @ eq))
    eq_ret = float(eq @ mean)

    sample_idx = rng.choice(simulations, size=min(500, simulations), replace=False)
    scatter = [{"volatility_pct": safe_round(p_vol[i] * 100), "return_pct": safe_round(p_ret[i] * 100),
                "sharpe": safe_round(p_sharpe[i])} for i in sample_idx]

    return {
        "symbols": [label(t) for t in tickers],
        "years": years,
        "simulations": simulations,
        "max_sharpe_portfolio": pack(int(np.argmax(p_sharpe))),
        "min_volatility_portfolio": pack(int(np.argmin(p_vol))),
        "equal_weight_portfolio": {
            "expected_return_pct": safe_round(eq_ret * 100),
            "volatility_pct": safe_round(eq_vol * 100),
            "sharpe_ratio": safe_round((eq_ret - RISK_FREE_RATE) / eq_vol),
        },
        "frontier_scatter": scatter,
        "disclaimer": DISCLAIMER + " Optimisers tend to overfit past data.",
    }


# ═══════════════════════════════════════════════════════════════════════════
#  NEW FEATURE 7 — MARKET OVERVIEW (INDICES, SECTORS, MOVERS, BREADTH)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/market")
@api_errors
def market_overview(top_n: int = 5):
    """Indices, sector heatmap, top gainers/losers (Nifty-50 sample) and advance/decline breadth."""
    top_n = min(max(top_n, 3), 10)

    idx_data = download_closes(list(INDICES.values()), "1mo")
    indices = []
    for name, t in INDICES.items():
        s = idx_data.get(t)
        if s is not None and len(s) >= 2:
            indices.append({
                "name": name, "symbol": t, "price": safe_round(s.iloc[-1]),
                "day_change_pct": safe_round((s.iloc[-1] / s.iloc[-2] - 1) * 100),
                "month_change_pct": safe_round((s.iloc[-1] / s.iloc[0] - 1) * 100),
            })

    sec_data = download_closes(list(SECTOR_INDICES.values()), "1mo")
    sectors = []
    for name, t in SECTOR_INDICES.items():
        s = sec_data.get(t)
        if s is not None and len(s) >= 2:
            sectors.append({
                "sector": name,
                "day_change_pct": safe_round((s.iloc[-1] / s.iloc[-2] - 1) * 100),
                "month_change_pct": safe_round((s.iloc[-1] / s.iloc[0] - 1) * 100),
            })
    sectors.sort(key=lambda x: (x["day_change_pct"] is None, -(x["day_change_pct"] or 0)))

    tickers = [f"{s}.NS" for s in NIFTY50]
    stock_data = download_closes(tickers, "5d")
    movers = []
    for t, s in stock_data.items():
        movers.append({"symbol": label(t), "price": safe_round(s.iloc[-1]),
                       "day_change_pct": safe_round((s.iloc[-1] / s.iloc[-2] - 1) * 100)})
    movers = [m for m in movers if m["day_change_pct"] is not None]
    movers.sort(key=lambda m: m["day_change_pct"], reverse=True)

    advances = sum(1 for m in movers if m["day_change_pct"] > 0)
    declines = sum(1 for m in movers if m["day_change_pct"] < 0)
    mood = None
    if movers:
        ratio = advances / len(movers)
        mood = "Bullish" if ratio >= 0.6 else "Bearish" if ratio <= 0.4 else "Mixed"

    return {
        "timestamp": datetime.now().isoformat(),
        "indices": indices,
        "sectors": sectors,
        "top_gainers": movers[:top_n],
        "top_losers": list(reversed(movers[-top_n:])),
        "breadth": {"advances": advances, "declines": declines,
                    "unchanged": len(movers) - advances - declines,
                    "stocks_counted": len(movers), "market_mood": mood},
        "note": "Gainers/losers are computed from a Nifty-50 sample list; data is delayed (Yahoo Finance).",
    }


# ═══════════════════════════════════════════════════════════════════════════
#  BONUS A — MONTE-CARLO PRICE FORECAST (PROBABILITY CONE)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/forecast/{symbol}")
@api_errors
def forecast(symbol: str, days: int = 60, simulations: int = 1000):
    """
    Simulates thousands of possible future price paths (geometric Brownian motion) using the
    stock's own last-year drift and volatility. Not a prediction: it shows the RANGE of outcomes.
    """
    ticker = to_ticker(symbol)
    days = min(max(days, 5), 252)
    simulations = min(max(simulations, 200), 3000)

    close = get_history(ticker, "5y")["Close"]
    log_ret = np.log(close / close.shift()).dropna().tail(TRADING_DAYS)
    mu, sigma = float(log_ret.mean()), float(log_ret.std())
    s0 = float(close.iloc[-1])

    rng = np.random.default_rng(42)
    shocks = rng.normal(mu - 0.5 * sigma ** 2, sigma, size=(simulations, days))
    paths = s0 * np.exp(np.cumsum(shocks, axis=1))

    pct = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    dates = pd.bdate_range(start=close.index[-1] + pd.Timedelta(days=1), periods=days)
    step = max(1, days // 60)
    cone = []
    for i in list(range(0, days, step)) + ([days - 1] if (days - 1) % step else []):
        cone.append({"date": str(dates[i].date()), "p5": safe_round(pct[0][i]), "p25": safe_round(pct[1][i]),
                     "median": safe_round(pct[2][i]), "p75": safe_round(pct[3][i]), "p95": safe_round(pct[4][i])})

    final = paths[:, -1]
    return {
        "symbol": ticker,
        "current_price": safe_round(s0),
        "horizon_trading_days": days,
        "annualised_drift_pct": safe_round(mu * TRADING_DAYS * 100),
        "annualised_volatility_pct": safe_round(sigma * math.sqrt(TRADING_DAYS) * 100),
        "expected_price": safe_round(final.mean()),
        "probability_of_gain_pct": safe_round((final > s0).mean() * 100),
        "probability_of_10pct_loss_pct": safe_round((final < s0 * 0.9).mean() * 100),
        "probability_of_10pct_gain_pct": safe_round((final > s0 * 1.1).mean() * 100),
        "range_90pct_confidence": [safe_round(np.percentile(final, 5)), safe_round(np.percentile(final, 95))],
        "cone": cone,
        "disclaimer": DISCLAIMER + " Simulation assumes past volatility continues.",
    }


# ═══════════════════════════════════════════════════════════════════════════
#  BONUS B — SUPPORT / RESISTANCE, PIVOTS, ATR, STOCHASTIC, VOLUME
# ═══════════════════════════════════════════════════════════════════════════

def _cluster(levels, tol=0.01):
    """Merge levels that are within 1% of each other so the list isn't cluttered."""
    out = []
    for v in sorted(levels):
        if not out or abs(v - out[-1]) / out[-1] > tol:
            out.append(v)
    return out


@app.get("/levels/{symbol}")
@api_errors
def levels(symbol: str):
    """Key price levels and extra indicators for trading-style analysis."""
    ticker = to_ticker(symbol)
    hist = get_history(ticker, "2y")
    close, high, low, vol = hist["Close"], hist["High"], hist["Low"], hist["Volume"]
    price = float(close.iloc[-1])

    # swing highs / lows over the last ~9 months
    w = 5
    recent = hist.tail(180)
    hi, lo = recent["High"], recent["Low"]
    swing_hi = hi[hi == hi.rolling(2 * w + 1, center=True).max()]
    swing_lo = lo[lo == lo.rolling(2 * w + 1, center=True).min()]
    resistances = _cluster([float(v) for v in swing_hi if v > price])[:3]
    supports = sorted(_cluster([float(v) for v in swing_lo if v < price]), reverse=True)[:3]

    # classic pivot points from the latest candle (levels for the next session)
    h, l, c = float(high.iloc[-1]), float(low.iloc[-1]), price
    p = (h + l + c) / 3
    pivots = {"pivot": p, "r1": 2 * p - l, "r2": p + (h - l), "r3": h + 2 * (p - l),
              "s1": 2 * p - h, "s2": p - (h - l), "s3": l - 2 * (h - p)}

    atr = calculate_atr(high, low, close)
    k, d = calculate_stochastic(high, low, close)
    obv = calculate_obv(close, vol)
    obv_trend = "accumulation (rising)" if obv.iloc[-1] > obv.rolling(20).mean().iloc[-1] else "distribution (falling)"
    avg_vol = float(vol.tail(20).mean()) if len(vol) >= 20 else None
    vol_ratio = float(vol.iloc[-1]) / avg_vol if avg_vol else None

    k_last = k.iloc[-1]
    stoch_state = None if pd.isna(k_last) else "overbought" if k_last > 80 else "oversold" if k_last < 20 else "neutral"

    return {
        "symbol": ticker,
        "current_price": safe_round(price),
        "supports": [safe_round(x) for x in supports],
        "resistances": [safe_round(x) for x in resistances],
        "nearest_support_distance_pct": safe_round((price / supports[0] - 1) * 100) if supports else None,
        "nearest_resistance_distance_pct": safe_round((resistances[0] / price - 1) * 100) if resistances else None,
        "pivot_points": {k_: safe_round(v) for k_, v in pivots.items()},
        "atr_14": safe_round(atr.iloc[-1]),
        "atr_pct_of_price": safe_round(atr.iloc[-1] / price * 100),
        "suggested_stop_loss_2atr": safe_round(price - 2 * atr.iloc[-1]),
        "stochastic_k": safe_round(k_last),
        "stochastic_d": safe_round(d.iloc[-1]),
        "stochastic_state": stoch_state,
        "obv_trend": obv_trend,
        "volume_vs_20d_avg": safe_round(vol_ratio),
        "volume_spike": bool(vol_ratio is not None and vol_ratio > 1.5),
        "disclaimer": DISCLAIMER,
    }
