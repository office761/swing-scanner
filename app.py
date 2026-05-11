# -*- coding: utf-8 -*-
"""
Swing Scanner 10/10 - one-button Streamlit app
Scans S&P 500 stocks for swing-trade candidates using:
- market regime filter
- relative strength vs SPY
- sector strength
- confirmed breakout / successful retest / healthy pullback
- extension filter
- liquidity filter
- risk/reward filter
- compact backtest filter

Important: Research tool only, not investment advice.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="סורק סווינג S&P 500",
    page_icon="📈",
    layout="wide",
)

# -----------------------------
# Defaults: intentionally hidden from the UI.
# -----------------------------
MIN_PRICE = 20.0
MIN_AVG_DOLLAR_VOLUME = 50_000_000
MAX_RISK_PCT = 10.5
MIN_BARS = 260
SCAN_PERIOD = "3y"
TOP_N = 5

# Risk/backtest defaults
BACKTEST_HOLD_DAYS = 20
BACKTEST_LOOKBACK_START = 230
BACKTEST_MIN_TRADES_FOR_FILTER = 3

# Fallback list. The app first tries to pull the current S&P 500 table from Wikipedia.
FALLBACK_SP500 = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","LLY","AVGO","TSLA","JPM","V","UNH","XOM","MA","COST","PG","JNJ","HD","ABBV","BAC","KO","NFLX","CRM","ORCL","WMT","AMD","MRK","CVX","PEP","ADBE","TMO","LIN","MCD","ACN","CSCO","ABT","QCOM","GE","INTU","IBM","AMAT","TXN","DHR","CAT","VZ","NOW","DIS","PFE","PM","ISRG","CMCSA","RTX","NEE","SPGI","UBER","UNP","LOW","HON","AXP","GS","BKNG","PGR","T","TJX","COP","SYK","ELV","BLK","MS","ETN","VRTX","LMT","REGN","MDT","ADP","ADI","CB","PANW","MU","KLAC","PLD","AMGN","MMC","BMY","DE","SCHW","GILD","LRCX","CI","FI","BSX","SO","C","MDLZ","ANET","ICE","SBUX","MO","SNPS","DUK","ZTS","CL","EQIX","SHW","CDNS","TT","WM","APH","HCA","MCO","PYPL","PH","CMG","CME","AON","NOC","ITW","USB","PNC","EOG","TDG","MAR","WELL","MCK","APD","ORLY","MSI","CTAS","GD","MMM","FDX","EMR","ROP","COF","TGT","BDX","CSX","NSC","NXPI","SLB","FCX","ECL","PSX","AJG","AFL","HLT","AZO","TRV","WMB","GM","O","OKE","ADSK","SRE","CCI","PCAR","TFC","DLR","SPG","ROST","MPC","KMI","TEL","D","PSA","BK","ALL","IDXX","AEP","F","MET","DHI","MNST","URI","NEM","AIG","LULU","KMB","AMP","GWW","COR","EW","PAYX","A","FAST","FIS","KVUE","ROK","RSG","VLO","KDP","AME","MSCI","CHTR","HUM","PRU","OXY","CTVA","HES","YUM","CTSH","IQV","LEN","ODFL","OTIS","VRSK","IR","EXC","GIS","PEG","ED","PCG","KR","EXR","CBRE","GEHC","ACGL","RCL","FANG","VMC","MLM","MPWR","DAL","EA","BKR","HPQ","XYL","IT","EFX","DD","KEYS","GRMN","MTD","ON","FTNT","HIG","EIX","RMD","WAB","PPG","WEC","DFS","ANSS","HWM","TSCO","CDW","NDAQ","WTW","BIIB","AWK","EQR","CAH","GLW","AVB","DOV","FITB","CHD","EBAY","TROW","BRO","LYB","PHM","GPN","BR","MTB","VLTO","DTE","IRM","HPE","WST","NTAP","STE","IFF","RJF","WY","PPL","LYV","WDC","FE","ES","STT","ETR","TER","SBAC","CTRA","HBAN","TYL","ZBH","VTR","HUBB","CBOE","PTC","LDOS","CINF","WAT","ARE","CMS","MKC","INVH","WBD","CCL","BALL","PFG","LH","ATO","TRGP","OMC","LVS","EXPE","DG","CLX","HOLX","EXPD","MOH","JBL","DRI","MAS","CF","COO","APTV","FDS","BAX","JBHT","NVR","SYF","TXT","MRO","ALGN","WRB","IEX","MAA","TSN","BBY","LUV","AKAM","EG","ESS","SNA","SWKS","AVY","EPAM","DPZ","POOL","CE","PKG","K","CAG","BG","UAL","NDSN","KEY","SWK","VRSN","PNR","DGX","UDR","TRMB","LKQ","GEN","IP","L","KIM","NI","HST","AES","EVRG","JKHY","ROL","AOS","ALLE","INCY","REG","IPG","LW","SJM","FFIV","TFX","CRL","JNPR","TECH","QRVO","MKTX","PNW","WYNN","EMN","UHS","TPR","CHRW","CPT","BXP","HSIC","AIZ","FOXA","FOX","NWSA","NWS","PAYC","AAL","MOS","APA","MTCH","GL","BWA","GNRC","HAS","MHK","FMC","PARA","ETSY","IVZ","CZR","BEN","WBA","VFC","ENPH","MRNA","BIO","ZION","BBWI","DXC"
]

SECTOR_ETF = {
    "Information Technology": "XLK",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}


def normalize_symbol(sym: str) -> str:
    # Yahoo uses BRK-B instead of BRK.B
    return str(sym).strip().upper().replace(".", "-")


@dataclass
class SignalResult:
    symbol: str
    company: str
    sector: str
    signal_type: str
    score: float
    close: float
    entry: float
    stop: float
    target_2r: float
    target_3r: float
    risk_pct: float
    rs_percentile: float
    sector_rel_63: Optional[float]
    volume_ratio: float
    close_location: float
    distance_sma20_pct: float
    distance_sma50_pct: float
    breakout_level: Optional[float]
    market_note: str
    reasons: List[str]
    avoid_if: List[str]
    backtest: dict
    next_earnings: Optional[str] = None


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(ttl=60 * 60 * 24)
def load_sp500_table() -> pd.DataFrame:
    """Return current S&P 500 constituents with sectors when possible."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0].copy()
        df["Symbol"] = df["Symbol"].map(normalize_symbol)
        df = df.rename(columns={"Security": "Company", "GICS Sector": "Sector"})
        return df[["Symbol", "Company", "Sector"]].drop_duplicates("Symbol")
    except Exception:
        return pd.DataFrame({
            "Symbol": FALLBACK_SP500,
            "Company": FALLBACK_SP500,
            "Sector": ["Unknown"] * len(FALLBACK_SP500),
        })


def _extract_downloaded(raw: pd.DataFrame, tickers: List[str]) -> Dict[str, pd.DataFrame]:
    result = {}
    if raw is None or raw.empty:
        return result

    tickers = [normalize_symbol(t) for t in tickers]

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = set(map(str, raw.columns.get_level_values(0)))
        lvl1 = set(map(str, raw.columns.get_level_values(1)))
        for t in tickers:
            df = None
            if t in lvl0:
                try:
                    df = raw[t].copy()
                except Exception:
                    df = None
            elif t in lvl1:
                try:
                    df = raw.xs(t, level=1, axis=1).copy()
                except Exception:
                    df = None
            if df is not None and not df.empty:
                result[t] = clean_ohlcv(df)
    else:
        if len(tickers) == 1:
            result[tickers[0]] = clean_ohlcv(raw.copy())

    return {k: v for k, v in result.items() if len(v) >= MIN_BARS}


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: str(c).title() for c in df.columns}
    df = df.rename(columns=rename)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for col in needed:
        if col not in df.columns:
            return pd.DataFrame()
    out = df[needed].copy()
    out = out.dropna()
    out = out[out["Volume"] > 0]
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out


@st.cache_data(ttl=60 * 60 * 4, show_spinner=False)
def download_prices(tickers: Tuple[str, ...], period: str = SCAN_PERIOD) -> Dict[str, pd.DataFrame]:
    """Download OHLCV data in chunks for reliability."""
    tickers_list = [normalize_symbol(t) for t in tickers]
    all_data: Dict[str, pd.DataFrame] = {}
    chunk_size = 70
    for i in range(0, len(tickers_list), chunk_size):
        chunk = tickers_list[i:i + chunk_size]
        try:
            raw = yf.download(
                tickers=" ".join(chunk),
                period=period,
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            all_data.update(_extract_downloaded(raw, chunk))
        except Exception:
            # Retry one by one for failed chunk, but keep it quiet.
            for t in chunk:
                try:
                    raw = yf.download(t, period=period, interval="1d", auto_adjust=True, progress=False)
                    d = _extract_downloaded(raw, [t])
                    all_data.update(d)
                except Exception:
                    pass
    return all_data


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def get_next_earnings(symbol: str) -> Optional[str]:
    """Best-effort earnings date check. If unavailable, return None and do not block the stock."""
    try:
        tk = yf.Ticker(symbol)
        ed = tk.get_earnings_dates(limit=4)
        if ed is None or len(ed) == 0:
            return None
        dates = pd.to_datetime(ed.index).tz_localize(None)
        future = [d for d in dates if d.date() >= datetime.utcnow().date()]
        if not future:
            return None
        return min(future).strftime("%Y-%m-%d")
    except Exception:
        return None


# -----------------------------
# Indicator helpers
# -----------------------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def pct_change_n(s: pd.Series, n: int) -> float:
    if len(s) <= n or s.iloc[-n] == 0:
        return np.nan
    return (s.iloc[-1] / s.iloc[-n] - 1) * 100


def close_location(row: pd.Series) -> float:
    rng = row["High"] - row["Low"]
    if rng <= 0:
        return 0.5
    return float((row["Close"] - row["Low"]) / rng)


def safe_float(x, default=np.nan) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


# -----------------------------
# Market / relative strength
# -----------------------------
def market_regime(data: Dict[str, pd.DataFrame]) -> Tuple[bool, str, float]:
    notes = []
    score = 0.0
    ok_parts = 0
    total_parts = 0

    for ticker, name in [("SPY", "S&P 500"), ("QQQ", "Nasdaq 100")]:
        df = data.get(ticker)
        if df is None or len(df) < 220:
            notes.append(f"לא הצלחתי לבדוק {ticker}")
            continue
        total_parts += 1
        c = df["Close"]
        s50 = sma(c, 50).iloc[-1]
        s200 = sma(c, 200).iloc[-1]
        ret5 = pct_change_n(c, 5)
        ret20 = pct_change_n(c, 20)
        trend_ok = c.iloc[-1] > s50 and s50 > s200
        not_crashing = ret5 > -3.5 and ret20 > -6.5
        if trend_ok and not_crashing:
            ok_parts += 1
            score += 15
            notes.append(f"{name} במגמה תומכת")
        elif c.iloc[-1] > s200 and not_crashing:
            score += 7
            notes.append(f"{name} ניטרלי-חיובי")
        else:
            notes.append(f"{name} חלש/לא תומך")

    # If we cannot check market, do not hard-block; but score stays lower.
    if total_parts == 0:
        return True, "לא הצלחתי לבדוק את מצב השוק — הסריקה תמשיך בזהירות.", 0

    market_ok = ok_parts >= 1 and score >= 12
    return market_ok, " | ".join(notes), score


def compute_rs_table(price_data: Dict[str, pd.DataFrame], spy: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    spy_close = spy["Close"]
    rows = []
    for sym in symbols:
        df = price_data.get(sym)
        if df is None or len(df) < MIN_BARS:
            continue
        c = df["Close"]
        ret21 = pct_change_n(c, 21)
        ret63 = pct_change_n(c, 63)
        ret126 = pct_change_n(c, 126)
        ret252 = pct_change_n(c, 252)
        spy21 = pct_change_n(spy_close, 21)
        spy63 = pct_change_n(spy_close, 63)
        spy126 = pct_change_n(spy_close, 126)
        spy252 = pct_change_n(spy_close, 252)
        weighted = (
            0.15 * (ret21 - spy21) +
            0.35 * (ret63 - spy63) +
            0.30 * (ret126 - spy126) +
            0.20 * (ret252 - spy252)
        )
        rows.append({
            "Symbol": sym,
            "RSScoreRaw": weighted,
            "Rel21": ret21 - spy21,
            "Rel63": ret63 - spy63,
            "Rel126": ret126 - spy126,
            "Rel252": ret252 - spy252,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["RSPercentile"] = out["RSScoreRaw"].rank(pct=True) * 100
    return out


def sector_strength(sector: str, aux_data: Dict[str, pd.DataFrame], spy: pd.DataFrame) -> Optional[float]:
    etf = SECTOR_ETF.get(sector)
    if not etf:
        return None
    df = aux_data.get(etf)
    if df is None or len(df) < 130:
        return None
    return pct_change_n(df["Close"], 63) - pct_change_n(spy["Close"], 63)


# -----------------------------
# Signal detection
# -----------------------------
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["SMA20"] = sma(d["Close"], 20)
    d["SMA50"] = sma(d["Close"], 50)
    d["SMA200"] = sma(d["Close"], 200)
    d["ATR14"] = atr(d, 14)
    d["AvgVol20"] = d["Volume"].rolling(20).mean()
    d["High20Prev"] = d["High"].shift(1).rolling(20).max()
    d["High55Prev"] = d["High"].shift(1).rolling(55).max()
    d["Low10"] = d["Low"].rolling(10).min()
    d["Low20"] = d["Low"].rolling(20).min()
    d["CloseLocation"] = d.apply(close_location, axis=1)
    return d


def detect_latest_signal(sym: str, df_raw: pd.DataFrame, meta: dict, rs_row: pd.Series, spy: pd.DataFrame, aux_data: Dict[str, pd.DataFrame], market_note: str, market_score: float) -> Optional[SignalResult]:
    df = enrich(df_raw)
    if len(df) < MIN_BARS:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]

    close = safe_float(last["Close"])
    if close < MIN_PRICE:
        return None

    avg_vol20 = safe_float(last["AvgVol20"])
    if avg_vol20 * close < MIN_AVG_DOLLAR_VOLUME:
        return None

    # Trend filter
    s20 = safe_float(last["SMA20"])
    s50 = safe_float(last["SMA50"])
    s200 = safe_float(last["SMA200"])
    a14 = safe_float(last["ATR14"])
    if not all(np.isfinite(x) for x in [s20, s50, s200, a14]):
        return None

    if not (close > s50 and s50 > s200):
        return None

    # Relative strength filter - top 30% only, with top 20% heavily rewarded.
    rs_pct = safe_float(rs_row.get("RSPercentile"))
    rel63 = safe_float(rs_row.get("Rel63"))
    rel126 = safe_float(rs_row.get("Rel126"))
    if rs_pct < 70 or rel63 < -1.0 or rel126 < -2.0:
        return None

    sector = str(meta.get("Sector", "Unknown"))
    company = str(meta.get("Company", sym))
    sec_rel_63 = sector_strength(sector, aux_data, spy)
    # Do not hard-block unknown sector, but block clearly weak known sectors.
    if sec_rel_63 is not None and sec_rel_63 < -4.0:
        return None

    # Avoid extended names.
    dist_s20 = (close / s20 - 1) * 100
    dist_s50 = (close / s50 - 1) * 100
    if dist_s20 > 8.5 or dist_s50 > 18.0:
        return None

    vol_ratio = safe_float(last["Volume"]) / avg_vol20 if avg_vol20 > 0 else np.nan
    cloc = safe_float(last["CloseLocation"])
    high20_prev = safe_float(last["High20Prev"])
    high55_prev = safe_float(last["High55Prev"])

    signal_type = None
    entry = None
    stop = None
    breakout_level = None
    reasons: List[str] = []
    avoid_if: List[str] = []

    # A. Confirmed breakout: daily close above 20d/55d resistance, with volume and strong close.
    confirmed_breakout = (
        close > high20_prev * 1.003 and
        close >= high55_prev * 0.995 and
        vol_ratio >= 1.25 and
        cloc >= 0.62 and
        close > s20
    )

    if confirmed_breakout:
        signal_type = "פריצה מאושרת"
        entry = max(close, safe_float(last["High"]) * 1.001)
        # stop below breakout level or recent low, no wider than max risk
        structural_stop = min(high20_prev * 0.985, safe_float(last["Low"]) - 0.25 * a14, safe_float(last["Low10"]))
        stop = max(structural_stop, entry * (1 - MAX_RISK_PCT / 100))
        breakout_level = high20_prev
        reasons.extend([
            "סגירה יומית מעל התנגדות 20 יום",
            "מחזור גבוה מהממוצע",
            "הנר סגר בחלק העליון של הטווח",
        ])
        avoid_if.extend([
            "המחיר חוזר מתחת לרמת הפריצה בסגירה יומית",
            "הכניסה מתבצעת בגאפ גבוה מדי מעל הטריגר",
        ])

    # B. Successful retest after breakout: breakout occurred recently, price came back near level and held.
    if signal_type is None:
        recent = df.iloc[-12:-1].copy()
        breakout_candidates = []
        for idx, row in recent.iterrows():
            if pd.isna(row["High20Prev"]) or pd.isna(row["AvgVol20"]):
                continue
            rv = row["Volume"] / row["AvgVol20"] if row["AvgVol20"] > 0 else np.nan
            if row["Close"] > row["High20Prev"] * 1.003 and rv >= 1.15:
                breakout_candidates.append((idx, float(row["High20Prev"]), float(row["Close"])))
        if breakout_candidates:
            _, level, bo_close = breakout_candidates[-1]
            held_level = last["Low"] <= level * 1.025 and close >= level * 0.995
            positive_reversal = close > last["Open"] and cloc >= 0.55
            trend_intact = close > s20 * 0.985 and close > s50
            if held_level and positive_reversal and trend_intact:
                signal_type = "ריטסט מוצלח אחרי פריצה"
                entry = safe_float(last["High"]) * 1.002
                structural_stop = min(safe_float(last["Low"]) * 0.992, level * 0.985, safe_float(last["Low10"]))
                stop = max(structural_stop, entry * (1 - MAX_RISK_PCT / 100))
                breakout_level = level
                reasons.extend([
                    "הייתה פריצה בימים האחרונים",
                    "המחיר חזר לבדוק את אזור הפריצה והחזיק מעליו",
                    "נר ההיפוך נסגר חיובי",
                ])
                avoid_if.extend([
                    "המחיר סוגר מתחת לאזור הפריצה שנבדק מחדש",
                    "השוק הכללי מתהפך לירידות חדות",
                ])

    # C. Healthy pullback in strong trend - lower priority, only if very strong RS.
    if signal_type is None:
        near_s20 = abs(close / s20 - 1) * 100 <= 3.0
        pullback_ok = (
            rs_pct >= 82 and
            near_s20 and
            close > s50 and
            safe_float(prev["Close"]) <= safe_float(prev["SMA20"]) * 1.015 and
            close > last["Open"] and
            cloc >= 0.58
        )
        if pullback_ok:
            signal_type = "תיקון בריא במגמת עלייה"
            entry = max(safe_float(last["High"]), safe_float(prev["High"])) * 1.002
            structural_stop = min(safe_float(last["Low10"]), s20 - 0.7 * a14)
            stop = max(structural_stop, entry * (1 - MAX_RISK_PCT / 100))
            breakout_level = None
            reasons.extend([
                "המניה חזקה מאוד יחסית למדד",
                "המחיר נח קרוב לממוצע 20 ולא רדף רחוק מדי",
                "נר חיובי לאחר תיקון קצר",
            ])
            avoid_if.extend([
                "אין מעבר מעל הגבוה האחרון",
                "המחיר שובר את אזור התמיכה/הממוצע הקצר",
            ])

    if signal_type is None or entry is None or stop is None:
        return None

    if stop >= entry:
        return None
    risk = entry - stop
    risk_pct = risk / entry * 100
    if risk_pct <= 0 or risk_pct > MAX_RISK_PCT:
        return None

    target_2r = entry + 2 * risk
    target_3r = entry + 3 * risk

    # Backtest filter
    bt = run_backtest(df_raw, spy, signal_type)
    if bt["trades"] >= BACKTEST_MIN_TRADES_FOR_FILTER:
        if bt["avg_r"] <= 0:
            return None
        if bt["win_rate"] < 35 and bt["avg_r"] < 0.50:
            return None

    # Score
    score = 0.0
    score += market_score
    score += min(25, max(0, (rs_pct - 60) * 0.75))  # top RS matters
    score += 12 if sec_rel_63 is not None and sec_rel_63 > 0 else (5 if sec_rel_63 is None else 0)
    score += 18 if signal_type == "ריטסט מוצלח אחרי פריצה" else 16 if signal_type == "פריצה מאושרת" else 12
    score += min(10, max(0, (vol_ratio - 1.0) * 8))
    score += 6 if cloc >= 0.70 else 3 if cloc >= 0.58 else 0
    score += 8 if risk_pct <= 7 else 4 if risk_pct <= 9 else 0
    if bt["trades"] >= 3:
        score += min(12, max(0, bt["avg_r"] * 5 + (bt["win_rate"] - 40) * 0.12))
    else:
        score += 2

    # Best-effort earnings check for top candidates later; placeholder here.
    reasons.extend([
        "המניה נמצאת במגמת עלייה: מחיר מעל 50 ו-50 מעל 200",
        f"חוזק יחסי גבוה מול SPY: אחוזון {rs_pct:.0f}",
    ])
    if sec_rel_63 is not None:
        reasons.append(f"הסקטור {'חזק' if sec_rel_63 > 0 else 'לא מוביל'} מול SPY ב-63 ימים האחרונים")
    reasons.append("המרחק מהממוצעים אינו קיצוני, ולכן אין כאן רדיפה מאוחרת מדי")

    avoid_if.append("יש דוח קרוב מאוד או חדשות מהותיות שלא נבדקו באפליקציה")

    return SignalResult(
        symbol=sym,
        company=company,
        sector=sector,
        signal_type=signal_type,
        score=score,
        close=close,
        entry=entry,
        stop=stop,
        target_2r=target_2r,
        target_3r=target_3r,
        risk_pct=risk_pct,
        rs_percentile=rs_pct,
        sector_rel_63=sec_rel_63,
        volume_ratio=vol_ratio,
        close_location=cloc,
        distance_sma20_pct=dist_s20,
        distance_sma50_pct=dist_s50,
        breakout_level=breakout_level,
        market_note=market_note,
        reasons=reasons,
        avoid_if=avoid_if,
        backtest=bt,
    )


# -----------------------------
# Backtest
# -----------------------------
def historical_signal_at(df: pd.DataFrame, i: int, desired_type: str) -> Optional[dict]:
    if i < BACKTEST_LOOKBACK_START or i >= len(df) - BACKTEST_HOLD_DAYS - 1:
        return None
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    close = safe_float(row["Close"])
    s20 = safe_float(row["SMA20"])
    s50 = safe_float(row["SMA50"])
    s200 = safe_float(row["SMA200"])
    a14 = safe_float(row["ATR14"])
    avg_vol20 = safe_float(row["AvgVol20"])
    if not all(np.isfinite(x) for x in [close, s20, s50, s200, a14, avg_vol20]):
        return None
    if not (close > s50 and s50 > s200 and close >= MIN_PRICE and close * avg_vol20 >= MIN_AVG_DOLLAR_VOLUME):
        return None
    dist_s20 = (close / s20 - 1) * 100
    dist_s50 = (close / s50 - 1) * 100
    if dist_s20 > 8.5 or dist_s50 > 18:
        return None
    vol_ratio = row["Volume"] / avg_vol20 if avg_vol20 > 0 else np.nan
    cloc = safe_float(row["CloseLocation"])
    high20_prev = safe_float(row["High20Prev"])
    high55_prev = safe_float(row["High55Prev"])

    if desired_type == "פריצה מאושרת":
        if close > high20_prev * 1.003 and close >= high55_prev * 0.995 and vol_ratio >= 1.25 and cloc >= 0.62 and close > s20:
            entry = max(close, row["High"] * 1.001)
            structural_stop = min(high20_prev * 0.985, row["Low"] - 0.25 * a14, row["Low10"])
            stop = max(structural_stop, entry * (1 - MAX_RISK_PCT / 100))
            return {"entry": entry, "stop": stop, "date": df.index[i]}

    if desired_type == "ריטסט מוצלח אחרי פריצה":
        recent = df.iloc[max(0, i - 12):i].copy()
        breakout_candidates = []
        for _, r in recent.iterrows():
            if pd.isna(r["High20Prev"]) or pd.isna(r["AvgVol20"]):
                continue
            rv = r["Volume"] / r["AvgVol20"] if r["AvgVol20"] > 0 else np.nan
            if r["Close"] > r["High20Prev"] * 1.003 and rv >= 1.15:
                breakout_candidates.append(float(r["High20Prev"]))
        if breakout_candidates:
            level = breakout_candidates[-1]
            held = row["Low"] <= level * 1.025 and close >= level * 0.995
            reversal = close > row["Open"] and cloc >= 0.55
            if held and reversal and close > s20 * 0.985 and close > s50:
                entry = row["High"] * 1.002
                structural_stop = min(row["Low"] * 0.992, level * 0.985, row["Low10"])
                stop = max(structural_stop, entry * (1 - MAX_RISK_PCT / 100))
                return {"entry": entry, "stop": stop, "date": df.index[i]}

    if desired_type == "תיקון בריא במגמת עלייה":
        near_s20 = abs(close / s20 - 1) * 100 <= 3.0
        if near_s20 and close > s50 and prev["Close"] <= prev["SMA20"] * 1.015 and close > row["Open"] and cloc >= 0.58:
            entry = max(row["High"], prev["High"]) * 1.002
            structural_stop = min(row["Low10"], s20 - 0.7 * a14)
            stop = max(structural_stop, entry * (1 - MAX_RISK_PCT / 100))
            return {"entry": entry, "stop": stop, "date": df.index[i]}

    return None


def run_backtest(df_raw: pd.DataFrame, spy: pd.DataFrame, signal_type: str) -> dict:
    df = enrich(df_raw)
    trades = []
    i = BACKTEST_LOOKBACK_START
    while i < len(df) - BACKTEST_HOLD_DAYS - 2:
        sig = historical_signal_at(df, i, signal_type)
        if not sig:
            i += 1
            continue
        entry = float(sig["entry"])
        stop = float(sig["stop"])
        if stop >= entry:
            i += 1
            continue
        risk = entry - stop
        target2 = entry + 2 * risk
        target3 = entry + 3 * risk

        # Enter only if next 3 trading days trigger the entry price.
        entry_idx = None
        for j in range(i + 1, min(len(df), i + 4)):
            if df.iloc[j]["High"] >= entry:
                entry_idx = j
                break
        if entry_idx is None:
            i += 3
            continue

        exit_price = df.iloc[min(len(df) - 1, entry_idx + BACKTEST_HOLD_DAYS)]["Close"]
        exit_reason = "20 ימי מסחר"
        held = BACKTEST_HOLD_DAYS
        for j in range(entry_idx, min(len(df), entry_idx + BACKTEST_HOLD_DAYS + 1)):
            bar = df.iloc[j]
            # Conservative: if both stop and target touch in the same day, assume stop first.
            if bar["Low"] <= stop:
                exit_price = stop
                exit_reason = "סטופ"
                held = j - entry_idx
                break
            if bar["High"] >= target3:
                exit_price = target3
                exit_reason = "יעד 3R"
                held = j - entry_idx
                break
            if bar["High"] >= target2:
                exit_price = target2
                exit_reason = "יעד 2R"
                held = j - entry_idx
                break

        r_mult = (exit_price - entry) / risk
        trades.append({
            "date": sig["date"].strftime("%Y-%m-%d") if hasattr(sig["date"], "strftime") else str(sig["date"]),
            "entry": entry,
            "exit": exit_price,
            "r": r_mult,
            "reason": exit_reason,
            "held": held,
        })
        i = entry_idx + 7

    if not trades:
        return {
            "trades": 0,
            "win_rate": None,
            "avg_r": None,
            "best_r": None,
            "worst_r": None,
            "last_trade": None,
        }

    rs = [t["r"] for t in trades]
    wins = [r for r in rs if r > 0]
    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100,
        "avg_r": float(np.mean(rs)),
        "best_r": float(np.max(rs)),
        "worst_r": float(np.min(rs)),
        "last_trade": trades[-1],
    }


# -----------------------------
# Main scan
# -----------------------------
def scan() -> Tuple[List[SignalResult], str, bool]:
    sp500 = load_sp500_table()
    symbols = sp500["Symbol"].map(normalize_symbol).dropna().unique().tolist()
    meta = sp500.set_index("Symbol").to_dict("index")
    sector_etfs = sorted(set(SECTOR_ETF.values()))
    aux_tickers = ["SPY", "QQQ"] + sector_etfs

    progress = st.progress(0, text="טוען נתונים מהשוק...")

    aux_data = download_prices(tuple(aux_tickers), SCAN_PERIOD)
    if "SPY" not in aux_data:
        raise RuntimeError("לא הצלחתי למשוך נתוני SPY. נסה שוב בעוד כמה דקות.")

    market_ok, market_note, market_score = market_regime(aux_data)
    progress.progress(10, text="בודק מצב שוק וחוזק יחסי...")

    stock_data = download_prices(tuple(symbols), SCAN_PERIOD)
    progress.progress(60, text="מחשב חוזק יחסי, סקטורים ותבניות כניסה...")

    rs = compute_rs_table(stock_data, aux_data["SPY"], symbols)
    if rs.empty:
        raise RuntimeError("לא הצלחתי לחשב חוזק יחסי למניות. נסה שוב.")
    rs_map = rs.set_index("Symbol")

    candidates: List[SignalResult] = []
    for idx, sym in enumerate(symbols):
        df = stock_data.get(sym)
        if df is None or sym not in rs_map.index:
            continue
        res = detect_latest_signal(sym, df, meta.get(sym, {"Company": sym, "Sector": "Unknown"}), rs_map.loc[sym], aux_data["SPY"], aux_data, market_note, market_score)
        if res:
            candidates.append(res)
        if idx % 50 == 0:
            progress.progress(min(95, 60 + int(35 * idx / max(1, len(symbols)))), text=f"נסרקו {idx}/{len(symbols)} מניות...")

    # Earnings check only for preliminary top 12, to keep the app fast.
    candidates.sort(key=lambda x: x.score, reverse=True)
    checked: List[SignalResult] = []
    for c in candidates[:15]:
        ed = get_next_earnings(c.symbol)
        c.next_earnings = ed
        if ed:
            try:
                days = (pd.to_datetime(ed).date() - datetime.utcnow().date()).days
                if 0 <= days <= 10:
                    c.score -= 18
                    c.avoid_if.append(f"דוח קרוב בתאריך {ed} — עדיף להיזהר/להימנע לפני דוח")
            except Exception:
                pass
        checked.append(c)

    # Keep also candidates not checked; sort all again.
    final_candidates = checked + candidates[15:]
    final_candidates.sort(key=lambda x: x.score, reverse=True)

    # In a clearly weak market, show a defensive result: no new entries unless a candidate is exceptional.
    if not market_ok:
        exceptional = [c for c in final_candidates if c.score >= 75 and c.backtest.get("avg_r") not in [None] and c.backtest.get("avg_r", 0) > 0.7]
        final_candidates = exceptional

    progress.progress(100, text="הסריקה הושלמה")
    return final_candidates[:TOP_N], market_note, market_ok


# -----------------------------
# UI
# -----------------------------
def fmt_money(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"${x:,.2f}"


def fmt_pct(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.1f}%"


def fmt_r(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.2f}R"


st.markdown(
    """
    <style>
    .main {direction: rtl; text-align: right;}
    .stButton button {font-size: 1.15rem; font-weight: 700; border-radius: 16px; padding: 0.75rem 1.25rem; width: 100%;}
    div[data-testid="stMetric"] {background: #ffffff; padding: 16px; border-radius: 18px; border: 1px solid #eee;}
    .small-note {font-size: 0.9rem; color: #666; line-height: 1.7;}
    .stock-card {border: 1px solid #e6e6e6; padding: 18px; border-radius: 20px; background: white; margin-bottom: 14px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 סורק סווינג חכם — S&P 500")
st.write("כפתור אחד. עד 5 מניות בלבד. לכל מניה: כניסה, סטופ, יעד והסבר פשוט.")

st.warning(
    "הכלי מיועד למחקר וסינון בלבד. הוא אינו ייעוץ השקעות, אינו המלצה לקנות או למכור, ואינו מחליף בדיקה עצמאית של חדשות, דוחות, מצב שוק וגודל פוזיציה."
)

if "last_results" not in st.session_state:
    st.session_state.last_results = None
    st.session_state.market_note = ""
    st.session_state.market_ok = None

if st.button("🔎 סרוק עכשיו", type="primary"):
    try:
        with st.spinner("סורק את מניות ה-S&P 500. זה יכול לקחת דקה-שתיים..."):
            results, market_note, market_ok = scan()
        st.session_state.last_results = results
        st.session_state.market_note = market_note
        st.session_state.market_ok = market_ok
        st.rerun()
    except Exception as e:
        st.error(f"אירעה שגיאה בסריקה: {e}")
        st.info("נסה לרענן את האפליקציה ולהריץ שוב. לפעמים מקור הנתונים החינמי מגביל זמנית את המשיכה.")

results: Optional[List[SignalResult]] = st.session_state.last_results

if results is not None:
    market_ok = st.session_state.market_ok
    market_note = st.session_state.market_note
    if market_ok:
        st.success(f"מצב שוק: תומך בזהירות. {market_note}")
    else:
        st.error(f"מצב שוק: לא אידיאלי לכניסות סווינג חדשות. {market_note}")

    if not results:
        st.info("לא נמצאו כרגע עד 5 מניות שעוברות את כל שכבות הסינון. זה גם איתות חשוב: אין עסקה נקייה מספיק לפי המנגנון.")
    else:
        st.subheader("5 המועמדות הטובות ביותר כרגע")
        for idx, r in enumerate(results, start=1):
            with st.expander(f"{idx}. {r.symbol} — {r.signal_type} — ציון {r.score:.0f}/100", expanded=(idx == 1)):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("מחיר נוכחי", fmt_money(r.close))
                c2.metric("כניסה / טריגר", fmt_money(r.entry))
                c3.metric("סטופ לוס", fmt_money(r.stop))
                c4.metric("סיכון לסטופ", fmt_pct(r.risk_pct))

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("יעד ראשון 2R", fmt_money(r.target_2r))
                c6.metric("יעד שני 3R", fmt_money(r.target_3r))
                c7.metric("חוזק יחסי", f"אחוזון {r.rs_percentile:.0f}")
                c8.metric("נפח מול ממוצע", f"x{r.volume_ratio:.2f}")

                st.markdown("### מתי להיכנס")
                st.write(
                    f"כניסה טכנית רק מעל **{fmt_money(r.entry)}**. עדיף לא לרדוף אם הפתיחה/המחיר כבר גבוהים משמעותית מהטריגר."
                )

                st.markdown("### איפה למקם סטופ")
                st.write(
                    f"סטופ טכני באזור **{fmt_money(r.stop)}**. אם המחיר מגיע לשם, הרעיון הטכני של העסקה כבר נפגע."
                )

                st.markdown("### מתי לצאת / יעד")
                st.write(
                    f"יעד ראשון: **{fmt_money(r.target_2r)}**. יעד שני: **{fmt_money(r.target_3r)}**. לאחר מהלך של כ-1R לטובתך, הגיוני לשקול הגנה על העסקה או קידום סטופ לפי שיקול דעת."
                )

                st.markdown("### למה היא נכנסה לסריקה")
                for reason in r.reasons:
                    st.write(f"• {reason}")

                st.markdown("### מתי לא להיכנס / מתי להיזהר")
                for avoid in r.avoid_if:
                    st.write(f"• {avoid}")

                bt = r.backtest
                st.markdown("### בדיקה היסטורית של איתותים דומים")
                if bt.get("trades", 0) == 0:
                    st.write("לא נמצאו מספיק איתותים דומים בעבר הקרוב. לכן רמת הביטחון ההיסטורית נמוכה יותר.")
                else:
                    st.write(
                        f"נמצאו **{bt['trades']}** איתותים דומים. אחוז הצלחה: **{fmt_pct(bt['win_rate'])}**. ממוצע תוצאה: **{fmt_r(bt['avg_r'])}**. העסקה הגרועה ביותר: **{fmt_r(bt['worst_r'])}**."
                    )
                    if bt.get("last_trade"):
                        lt = bt["last_trade"]
                        st.caption(f"איתות דומה אחרון: {lt['date']} | תוצאה: {lt['r']:.2f}R | יציאה: {lt['reason']}")

                st.caption(
                    f"סקטור: {r.sector} | מרחק ממוצע 20: {fmt_pct(r.distance_sma20_pct)} | מרחק ממוצע 50: {fmt_pct(r.distance_sma50_pct)} | סגירת נר: {r.close_location:.2f}"
                )

st.markdown("---")
st.markdown(
    "<div class='small-note'>"
    "המנגנון מאחורי הקלעים: מצב שוק SPY/QQQ, חוזק יחסי מול SPY, חוזק סקטור, מגמה 50/200, פריצה מאושרת או ריטסט, נפח, מניעת רדיפה אחרי מניות מתוחות מדי, יחס סיכון-סיכוי ו-Backtest קצר."
    "</div>",
    unsafe_allow_html=True,
)
