from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="סורק סווינג S&P 500",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] { direction: rtl; text-align: right; }
    .stButton > button { width: 100%; height: 3.4rem; border-radius: 18px; font-size: 1.15rem; font-weight: 800; }
    div[data-testid="stMetricValue"] { direction: ltr; text-align: right; }
    .stock-card {border:1px solid #e5e7eb; border-radius:24px; padding:18px; margin-bottom:16px; background:#ffffff; box-shadow:0 1px 4px rgba(0,0,0,0.04);} 
    .ticker {font-size:1.5rem; font-weight:900; direction:ltr; display:inline-block;}
    .pill {display:inline-block; padding:4px 10px; border-radius:999px; background:#f1f5f9; margin:2px; font-size:0.85rem;}
    .warn {background:#fff7ed; border:1px solid #fed7aa; padding:12px 16px; border-radius:18px; color:#7c2d12;}
    </style>
    """,
    unsafe_allow_html=True,
)

FALLBACK_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","BRK-B","JPM","LLY","V","MA","UNH","XOM","COST","WMT","HD","PG",
    "NFLX","JNJ","BAC","CRM","ABBV","KO","ORCL","AMD","CVX","MRK","PEP","TMO","LIN","MCD","CSCO","ACN","ADBE","ABT","IBM","QCOM",
    "GE","TXN","AMAT","DHR","NOW","CAT","VZ","DIS","PFE","NEE","PM","RTX","SPGI","UBER","HON","UNP","LOW","GS","AXP","BKNG",
    "T","BLK","TJX","PGR","SYK","VRTX","ISRG","LMT","MS","ETN","PANW","ADP","MU","PLD","KLAC","ADI","MDT","CB","GILD","AMGN",
    "MMC","LRCX","DE","ANET","INTU","C","CI","FI","SO","MO","SHW","CDNS","SNPS","ICE","DUK","HCA","MCO","ZTS","CMG","WM",
    "EQIX","TT","PH","CME","AON","NOC","ITW","USB","PNC","MAR","EOG","APD","ORLY","MCK","GD","MMM","FDX","EMR","ROP","COF",
    "TGT","BDX","CSX","NSC","NXPI","SLB","FCX","PSX","ECL","AJG","AFL","HLT","AZO","TRV","WMB","GM","O","OKE","ADSK","SRE",
    "MPC","SPG","CCI","PCAR","TFC","DLR","KMI","TEL","PSA","BK","ALL","D","IDXX","AEP","F","MET","DHI","MNST","URI","NEM",
    "AIG","LULU","KMB","AMP","GWW","COR","EW","PAYX","A","FAST","FIS","KVUE","ROK","RSG","VLO","KDP","AME","MSCI","CHTR","HUM",
    "PRU","OXY","CTVA","HES","YUM","CTSH","IQV","LEN","ODFL","OTIS","VRSK","IR","EXC","GIS","PEG","ED","PCG","KR","EXR","CBRE",
    "GEHC","ACGL","RCL","FANG","VMC","MLM","MPWR","DAL","EA","BKR","HPQ","XYL","IT","EFX","DD","KEYS","GRMN","MTD","ON","FTNT"
]

@dataclass
class Candidate:
    ticker: str
    company: str
    sector: str
    score: int
    setup: str
    close: float
    entry: float
    stop: float
    target2: float
    target3: float
    risk_pct: float
    rs3m: float
    rs6m: float
    volume_ratio: float
    dist_high: float
    why: str
    entry_rule: str
    exit_rule: str
    bt_trades: int
    bt_win_rate: Optional[float]
    bt_avg_r: Optional[float]
    bt_best_r: Optional[float]
    bt_worst_r: Optional[float]


def to_yahoo_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_sp500() -> pd.DataFrame:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0].copy()
        df["ticker"] = df["Symbol"].map(to_yahoo_symbol)
        df = df.rename(columns={"Security": "company", "GICS Sector": "sector"})
        return df[["ticker", "company", "sector"]].drop_duplicates("ticker").reset_index(drop=True)
    except Exception:
        return pd.DataFrame({"ticker": FALLBACK_TICKERS, "company": FALLBACK_TICKERS, "sector": "S&P 500 fallback"})


@st.cache_data(ttl=55 * 60, show_spinner=False)
def download_batch(tickers: Tuple[str, ...], period: str = "3y") -> Dict[str, pd.DataFrame]:
    raw = yf.download(
        list(tickers),
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    out: Dict[str, pd.DataFrame] = {}
    if raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if t in raw.columns.get_level_values(0):
                df = raw[t].copy()
                out[t] = clean(df)
    else:
        if len(tickers) == 1:
            out[tickers[0]] = clean(raw)
    return out


def download_universe(tickers: List[str], batch_size: int = 70) -> Dict[str, pd.DataFrame]:
    all_data: Dict[str, pd.DataFrame] = {}
    batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
    bar = st.progress(0, text="מוריד נתוני שוק...")
    for i, batch in enumerate(batches, start=1):
        data = download_batch(tuple(batch), "3y")
        all_data.update(data)
        bar.progress(i / len(batches), text=f"מוריד נתונים: קבוצה {i}/{len(batches)}")
    bar.empty()
    return all_data


def clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c).title().strip() for c in out.columns]
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in out.columns]
    if len(keep) < 5:
        return pd.DataFrame()
    out = out[keep].apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out["Volume"] = out["Volume"].fillna(0)
    out = out.sort_index()
    return out


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = clean(df)
    if x.empty:
        return x
    c, h, l, v = x["Close"], x["High"], x["Low"], x["Volume"]
    x["SMA20"] = c.rolling(20).mean()
    x["SMA50"] = c.rolling(50).mean()
    x["SMA200"] = c.rolling(200).mean()
    prev = c.shift(1)
    tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    x["ATR14"] = tr.rolling(14).mean()
    x["ATR_PCT"] = x["ATR14"] / c
    x["VOL20"] = v.rolling(20).mean()
    x["VOL50"] = v.rolling(50).mean()
    x["VOL_RATIO"] = v / x["VOL20"]
    x["DOLLAR_VOL50"] = x["VOL50"] * c
    x["HIGH20"] = h.shift(1).rolling(20).max()
    x["HIGH55"] = h.shift(1).rolling(55).max()
    x["LOW10"] = l.shift(1).rolling(10).min()
    x["LOW20"] = l.shift(1).rolling(20).min()
    x["HIGH252"] = h.rolling(252).max()
    x["RET63"] = c / c.shift(63) - 1
    x["RET126"] = c / c.shift(126) - 1
    x["RET252"] = c / c.shift(252) - 1
    return x


def safe(v, default=np.nan) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def build_signal(df: pd.DataFrame, spy: pd.DataFrame, idx: int, meta: dict, strict: bool = True) -> Optional[dict]:
    if idx < 252 or idx >= len(df):
        return None
    row = df.iloc[idx]
    spy_row = spy.iloc[min(idx, len(spy)-1)] if len(spy) > idx else spy.iloc[-1]

    close = safe(row["Close"])
    sma20 = safe(row["SMA20"])
    sma50 = safe(row["SMA50"])
    sma200 = safe(row["SMA200"])
    atr = safe(row["ATR14"])
    atr_pct = safe(row["ATR_PCT"])
    dollar_vol = safe(row["DOLLAR_VOL50"], 0)
    vol_ratio = safe(row["VOL_RATIO"], 1)
    high20 = safe(row["HIGH20"])
    high55 = safe(row["HIGH55"])
    low10 = safe(row["LOW10"])
    low20 = safe(row["LOW20"])
    high252 = safe(row["HIGH252"])
    ret63 = safe(row["RET63"], 0)
    ret126 = safe(row["RET126"], 0)
    spy63 = safe(spy_row.get("RET63"), 0)
    spy126 = safe(spy_row.get("RET126"), 0)
    rs3 = ret63 - spy63
    rs6 = ret126 - spy126

    if not np.isfinite(close) or close < 20:
        return None
    if dollar_vol < 30_000_000:
        return None
    if not (close > sma50 > sma200):
        return None
    if strict and not (rs3 > 0 and rs6 > 0):
        return None
    if np.isfinite(atr_pct) and atr_pct > 0.12:
        return None
    if np.isfinite(high252) and close < high252 * 0.75:
        return None

    setup = None
    entry = None
    setup_score = 0
    entry_rule = ""
    if np.isfinite(high55) and close > high55 and vol_ratio >= 1.12:
        setup = "פריצה בפועל עם נפח"
        entry = max(close, row["High"]) * 1.002
        setup_score = 25
        entry_rule = "כניסה רק אם המחיר ממשיך מעל אזור הפריצה ולא חוזר מתחתיו. עדיף אישור של סגירה יומית או החזקה תוך-יומית מעל הגבוה האחרון."
    elif np.isfinite(high55) and high55 * 0.975 <= close <= high55 * 1.01 and close > sma20:
        setup = "קרובה לפריצה"
        entry = high55 * 1.002
        setup_score = 22
        entry_rule = "כניסה רק בפריצה מעל השיא האחרון. לא להיכנס לפני שהטריגר מופעל."
    elif close > sma50 and np.isfinite(sma20) and abs(close / sma20 - 1) <= 0.04 and rs3 > 0:
        setup = "תיקון בריא במגמת עלייה"
        entry = max(row["High"], high20 if np.isfinite(high20) else row["High"]) * 1.002
        setup_score = 18
        entry_rule = "כניסה רק מעל הגבוה האחרון אחרי התיקון, כסימן שהקונים חוזרים."
    else:
        return None

    atr_stop = entry - 1.6 * atr if np.isfinite(atr) else entry * 0.93
    structure_stop = np.nanmax([low10, low20]) if np.isfinite(low10) or np.isfinite(low20) else entry * 0.93
    stop_candidates = [x for x in [atr_stop, structure_stop] if np.isfinite(x) and x < entry]
    stop = max(stop_candidates) if stop_candidates else entry * 0.93
    risk = entry - stop
    if risk <= 0:
        return None
    risk_pct = risk / entry
    if risk_pct > 0.12:
        return None
    target2 = entry + 2 * risk
    target3 = entry + 3 * risk

    score = 0
    score += 22 if close > sma50 else 0
    score += 18 if close > sma200 else 0
    score += 10 if sma50 > sma200 else 0
    score += min(20, max(0, int(rs3 * 100)))
    score += min(10, max(0, int(rs6 * 50)))
    score += 8 if vol_ratio > 1.1 else 0
    score += setup_score
    if risk_pct <= 0.08:
        score += 8
    dist_high = close / high252 - 1 if np.isfinite(high252) and high252 else np.nan

    why_parts = []
    why_parts.append("המניה מעל ממוצע 50 ו-200, כלומר המגמה הראשית עדיין חיובית")
    if rs3 > 0:
        why_parts.append("היא חזקה מה-SPY בשלושת החודשים האחרונים")
    if rs6 > 0:
        why_parts.append("גם בחצי השנה האחרונה היא מציגה עדיפות יחסית על השוק")
    why_parts.append(f"התבנית הנוכחית היא {setup}")
    if vol_ratio > 1.1:
        why_parts.append("יש עלייה במחזור ביחס לממוצע")

    return {
        "setup": setup,
        "entry": float(entry),
        "stop": float(stop),
        "target2": float(target2),
        "target3": float(target3),
        "risk_pct": float(risk_pct),
        "score": int(score),
        "rs3m": float(rs3),
        "rs6m": float(rs6),
        "vol_ratio": float(vol_ratio),
        "dist_high": float(dist_high) if np.isfinite(dist_high) else np.nan,
        "why": "; ".join(why_parts) + ".",
        "entry_rule": entry_rule,
        "exit_rule": "יציאה חלקית/מלאה באזור יעד 2R, או ניהול המשך עד 3R. אם המחיר יורד לסטופ — יציאה בלי ויכוח. אם אין התקדמות אחרי כ-20 ימי מסחר, לשקול יציאה/הקטנה.",
        "close": float(close),
    }


def backtest(df: pd.DataFrame, spy: pd.DataFrame) -> dict:
    trades = []
    for i in range(252, len(df) - 26):
        sig = build_signal(df, spy, i, {}, strict=True)
        if not sig:
            continue
        entry = sig["entry"]
        stop = sig["stop"]
        risk = entry - stop
        target = sig["target2"]
        entered_day = None
        for j in range(i + 1, min(i + 6, len(df))):
            if df.iloc[j]["High"] >= entry:
                entered_day = j
                break
        if entered_day is None:
            continue
        exit_price = None
        exit_day = None
        for j in range(entered_day, min(entered_day + 21, len(df))):
            bar = df.iloc[j]
            # Conservative assumption: if both target and stop are hit the same day, stop first.
            if bar["Low"] <= stop:
                exit_price = stop
                exit_day = j
                break
            if bar["High"] >= target:
                exit_price = target
                exit_day = j
                break
        if exit_price is None:
            exit_day = min(entered_day + 20, len(df) - 1)
            exit_price = df.iloc[exit_day]["Close"]
        r = (exit_price - entry) / risk
        trades.append(r)
        # Avoid counting the same setup every day.
        i += 7
    if not trades:
        return {"trades": 0, "win_rate": None, "avg_r": None, "best_r": None, "worst_r": None}
    arr = np.array(trades, dtype=float)
    return {
        "trades": int(len(arr)),
        "win_rate": float((arr > 0).mean()),
        "avg_r": float(arr.mean()),
        "best_r": float(arr.max()),
        "worst_r": float(arr.min()),
    }


def analyze_all(data: Dict[str, pd.DataFrame], meta_df: pd.DataFrame) -> List[Candidate]:
    spy = indicators(data.get("SPY", pd.DataFrame()))
    if spy.empty or len(spy) < 260:
        raise RuntimeError("לא הצלחתי לטעון SPY — אין בסיס להשוואה מול ה-S&P 500.")
    meta = meta_df.set_index("ticker").to_dict("index")
    rows: List[Candidate] = []
    progress = st.progress(0, text="מחשב איתותים ובדיקה היסטורית...")
    tickers = [t for t in data.keys() if t != "SPY"]
    for n, t in enumerate(tickers, start=1):
        df = indicators(data[t])
        if df.empty or len(df) < 280:
            continue
        # align by date to SPY for fairer relative strength
        joined_spy = spy.reindex(df.index).ffill().dropna()
        if joined_spy.empty:
            continue
        sig = build_signal(df, joined_spy, len(df)-1, meta.get(t, {}), strict=True)
        if not sig or sig["score"] < 72:
            continue
        bt = backtest(df, joined_spy)
        # Small penalty if no historical confirmation.
        adjusted_score = sig["score"] + (8 if bt["avg_r"] is not None and bt["avg_r"] > 0 else 0) + (4 if bt["trades"] >= 3 else 0)
        info = meta.get(t, {})
        rows.append(Candidate(
            ticker=t,
            company=str(info.get("company", t)),
            sector=str(info.get("sector", "")),
            score=int(adjusted_score),
            setup=sig["setup"],
            close=sig["close"],
            entry=sig["entry"],
            stop=sig["stop"],
            target2=sig["target2"],
            target3=sig["target3"],
            risk_pct=sig["risk_pct"],
            rs3m=sig["rs3m"],
            rs6m=sig["rs6m"],
            volume_ratio=sig["vol_ratio"],
            dist_high=sig["dist_high"],
            why=sig["why"],
            entry_rule=sig["entry_rule"],
            exit_rule=sig["exit_rule"],
            bt_trades=bt["trades"],
            bt_win_rate=bt["win_rate"],
            bt_avg_r=bt["avg_r"],
            bt_best_r=bt["best_r"],
            bt_worst_r=bt["worst_r"],
        ))
        if n % 25 == 0:
            progress.progress(n / max(1, len(tickers)), text=f"מחשב איתותים: {n}/{len(tickers)}")
    progress.empty()
    rows.sort(key=lambda x: (x.score, -x.risk_pct, x.bt_avg_r if x.bt_avg_r is not None else -99), reverse=True)
    return rows[:5]


def money(v: float) -> str:
    return f"${v:,.2f}"


def pct(v: Optional[float], digits: int = 1) -> str:
    if v is None or not np.isfinite(v):
        return "אין מספיק נתונים"
    return f"{v * 100:.{digits}f}%"


def rfmt(v: Optional[float]) -> str:
    if v is None or not np.isfinite(v):
        return "אין מספיק נתונים"
    return f"{v:.2f}R"


st.title("📈 סורק סווינג בלחיצה אחת")
st.caption("S&P 500 בלבד · עד 5 מניות · כניסה, סטופ, יעד והסבר קצר לכל מניה")

st.markdown(
    """
    <div class="warn">
    הכלי מיועד למחקר וללמידה בלבד. הוא לא ייעוץ השקעות ולא תחליף לבדיקה עצמאית של דוחות, חדשות, מצב שוק וגודל פוזיציה.
    </div>
    """,
    unsafe_allow_html=True,
)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("סוג מניות", "S&P 500 בלבד")
with col_b:
    st.metric("פני סטוק", "מסוננות")
with col_c:
    st.metric("תוצאה", "עד 5 מניות")

if st.button("סרוק עכשיו", type="primary"):
    try:
        sp500 = load_sp500()
        tickers = sp500["ticker"].dropna().astype(str).unique().tolist()
        if "SPY" not in tickers:
            tickers = ["SPY"] + tickers
        else:
            tickers = ["SPY"] + [t for t in tickers if t != "SPY"]
        data = download_universe(tickers)
        st.success(f"נטענו נתונים עבור {max(0, len(data)-1)} מניות מתוך ה-S&P 500")
        top = analyze_all(data, sp500)
        if not top:
            st.warning("לא נמצאו כרגע עד 5 מניות שעומדות בתנאים המחמירים. זה עדיף מאיתות חלש — הסורק אמור להחזיר רק מצבים נקיים יחסית.")
            st.stop()
        st.subheader("5 המועמדות המובילות כרגע")
        for i, c in enumerate(top, start=1):
            st.markdown("<div class='stock-card'>", unsafe_allow_html=True)
            cols = st.columns([1, 2, 2, 2])
            with cols[0]:
                st.markdown(f"<span class='ticker'>{c.ticker}</span><br><b>ציון:</b> {c.score}", unsafe_allow_html=True)
            with cols[1]:
                st.metric("כניסה מעל", money(c.entry))
            with cols[2]:
                st.metric("סטופ לוס", money(c.stop), delta=f"סיכון {pct(c.risk_pct)}", delta_color="inverse")
            with cols[3]:
                st.metric("יעד ראשון", money(c.target2))
            with st.expander(f"לחץ להסבר מלא על {c.ticker}", expanded=(i == 1)):
                st.markdown(f"**חברה:** {c.company}  ")
                st.markdown(f"**סקטור:** {c.sector}  ")
                st.markdown(f"**תבנית:** {c.setup}  ")
                st.markdown(f"**מתי להיכנס:** {c.entry_rule}  ")
                st.markdown(f"**מחיר כניסה טכני:** {money(c.entry)}. לא לפני שהמחיר מגיע לטריגר.")
                st.markdown(f"**סטופ לוס:** {money(c.stop)} — מתחת למבנה/ATR. הסיכון המשוער הוא {pct(c.risk_pct)}.")
                st.markdown(f"**יעד יציאה:** יעד ראשון {money(c.target2)} באזור 2R; יעד המשך {money(c.target3)} באזור 3R.")
                st.markdown(f"**מדוע היא עלתה בסריקה:** {c.why}")
                st.markdown(f"**מתי לצאת:** {c.exit_rule}")
                st.markdown(
                    f"**בדיקה היסטורית לאיתותים דומים:** {c.bt_trades} עסקאות; "
                    f"אחוז הצלחה {pct(c.bt_win_rate)}; ממוצע {rfmt(c.bt_avg_r)}; "
                    f"הכי טוב {rfmt(c.bt_best_r)}; הכי גרוע {rfmt(c.bt_worst_r)}."
                )
                st.markdown(
                    f"<span class='pill'>חוזק 3 חודשים מול SPY: {pct(c.rs3m)}</span> "
                    f"<span class='pill'>חוזק 6 חודשים מול SPY: {pct(c.rs6m)}</span> "
                    f"<span class='pill'>נפח ביחס לממוצע: {c.volume_ratio:.2f}x</span>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"הסריקה נכשלה: {e}")
        st.info("אם זה קורה בפריסה ראשונה, לרוב מדובר בחיבור זמני ל-Yahoo Finance או בהתקנת תלויות. נסה להריץ שוב לאחר דקה.")
else:
    st.info("לחץ על הכפתור. אין הגדרות ואין טבלאות מסובכות — הסורק יחזיר עד 5 מניות בלבד עם הסבר לכל אחת.")

st.markdown("---")
st.caption("מקור נתונים: yfinance/Yahoo Finance לצורכי מחקר. התוצאה אינה המלצה לקנייה או מכירה.")
