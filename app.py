
import re
import math
from datetime import datetime, timedelta, date

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
    html, body, [data-testid="stAppViewContainer"] {
        direction: rtl;
        text-align: right;
        font-family: Arial, "Noto Sans Hebrew", sans-serif;
    }
    .ltr {
        direction: ltr;
        unicode-bidi: isolate;
        display: inline-block;
    }
    .small-note {
        color: #64748b;
        font-size: 0.92rem;
        line-height: 1.75;
    }
    .metric-card {
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 14px 16px;
        background: white;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        min-height: 95px;
    }
    .stock-card {
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 16px;
        background: #ffffff;
        margin-bottom: 12px;
    }
    .pill {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 999px;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        margin: 2px 3px;
        font-size: 0.85rem;
    }
    .good { color: #166534; font-weight: 700; }
    .warn { color: #92400e; font-weight: 700; }
    .bad { color: #991b1b; font-weight: 700; }
    .section-title {
        font-weight: 800;
        margin-top: 12px;
        margin-bottom: 4px;
    }
    div[data-testid="stExpander"] details summary {
        direction: rtl;
        text-align: right;
        font-size: 1.05rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


SECTOR_ETF = {
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}

SECTOR_HE = {
    "Information Technology": "טכנולוגיה",
    "Health Care": "בריאות",
    "Financials": "פיננסים",
    "Consumer Discretionary": "צריכה מחזורית",
    "Communication Services": "תקשורת ושירותי מדיה",
    "Industrials": "תעשייה",
    "Consumer Staples": "צריכה בסיסית",
    "Energy": "אנרגיה",
    "Utilities": "תשתיות",
    "Real Estate": "נדל״ן",
    "Materials": "חומרי גלם",
    "Unknown": "לא זמין",
}

FALLBACK_UNIVERSE = [
    ("AAPL","Apple Inc.","Information Technology","Consumer Electronics"),
    ("MSFT","Microsoft Corporation","Information Technology","Systems Software"),
    ("NVDA","NVIDIA Corporation","Information Technology","Semiconductors"),
    ("AMZN","Amazon.com Inc.","Consumer Discretionary","Broadline Retail"),
    ("META","Meta Platforms Inc.","Communication Services","Interactive Media"),
    ("GOOGL","Alphabet Inc.","Communication Services","Interactive Media"),
    ("AVGO","Broadcom Inc.","Information Technology","Semiconductors"),
    ("TSLA","Tesla Inc.","Consumer Discretionary","Automobile Manufacturers"),
    ("LLY","Eli Lilly and Company","Health Care","Pharmaceuticals"),
    ("JPM","JPMorgan Chase & Co.","Financials","Diversified Banks"),
    ("V","Visa Inc.","Financials","Transaction & Payment Processing"),
    ("MA","Mastercard Incorporated","Financials","Transaction & Payment Processing"),
    ("XOM","Exxon Mobil Corporation","Energy","Integrated Oil & Gas"),
    ("COST","Costco Wholesale Corporation","Consumer Staples","Consumer Staples Merchandise Retail"),
    ("UNH","UnitedHealth Group Incorporated","Health Care","Managed Health Care"),
    ("NFLX","Netflix Inc.","Communication Services","Movies & Entertainment"),
    ("WMT","Walmart Inc.","Consumer Staples","Consumer Staples Merchandise Retail"),
    ("AMD","Advanced Micro Devices Inc.","Information Technology","Semiconductors"),
    ("ORCL","Oracle Corporation","Information Technology","Application Software"),
    ("CRM","Salesforce Inc.","Information Technology","Application Software"),
    ("MU","Micron Technology Inc.","Information Technology","Semiconductors"),
    ("MRVL","Marvell Technology Inc.","Information Technology","Semiconductors"),
    ("SLB","Schlumberger Limited","Energy","Oil & Gas Equipment & Services"),
    ("BKR","Baker Hughes Company","Energy","Oil & Gas Equipment & Services"),
    ("TRGP","Targa Resources Corp.","Energy","Oil & Gas Storage & Transportation"),
    ("QCOM","QUALCOMM Incorporated","Information Technology","Semiconductors"),
    ("AMAT","Applied Materials Inc.","Information Technology","Semiconductor Materials & Equipment"),
    ("LRCX","Lam Research Corporation","Information Technology","Semiconductor Materials & Equipment"),
    ("KLAC","KLA Corporation","Information Technology","Semiconductor Materials & Equipment"),
    ("PANW","Palo Alto Networks Inc.","Information Technology","Systems Software"),
    ("ANET","Arista Networks Inc.","Information Technology","Communications Equipment"),
    ("NOW","ServiceNow Inc.","Information Technology","Systems Software"),
    ("ADBE","Adobe Inc.","Information Technology","Application Software"),
    ("INTU","Intuit Inc.","Information Technology","Application Software"),
    ("CAT","Caterpillar Inc.","Industrials","Construction Machinery"),
    ("GE","GE Aerospace","Industrials","Aerospace & Defense"),
    ("ETN","Eaton Corporation plc","Industrials","Electrical Components"),
    ("PH","Parker-Hannifin Corporation","Industrials","Industrial Machinery"),
    ("DE","Deere & Company","Industrials","Agricultural Machinery"),
    ("JPM","JPMorgan Chase & Co.","Financials","Diversified Banks"),
    ("GS","Goldman Sachs Group Inc.","Financials","Investment Banking"),
    ("MS","Morgan Stanley","Financials","Investment Banking"),
    ("C","Citigroup Inc.","Financials","Diversified Banks"),
    ("BAC","Bank of America Corporation","Financials","Diversified Banks"),
    ("HD","Home Depot Inc.","Consumer Discretionary","Home Improvement Retail"),
    ("LOW","Lowe's Companies Inc.","Consumer Discretionary","Home Improvement Retail"),
    ("UBER","Uber Technologies Inc.","Industrials","Passenger Ground Transportation"),
    ("BKNG","Booking Holdings Inc.","Consumer Discretionary","Hotels Restaurants & Leisure"),
    ("CMG","Chipotle Mexican Grill Inc.","Consumer Discretionary","Restaurants"),
    ("LLY","Eli Lilly and Company","Health Care","Pharmaceuticals"),
    ("ABBV","AbbVie Inc.","Health Care","Biotechnology"),
    ("TMO","Thermo Fisher Scientific Inc.","Health Care","Life Sciences Tools"),
    ("ISRG","Intuitive Surgical Inc.","Health Care","Health Care Equipment"),
    ("VRTX","Vertex Pharmaceuticals Inc.","Health Care","Biotechnology"),
    ("REGN","Regeneron Pharmaceuticals Inc.","Health Care","Biotechnology"),
    ("PFE","Pfizer Inc.","Health Care","Pharmaceuticals"),
    ("MRK","Merck & Co. Inc.","Health Care","Pharmaceuticals"),
    ("NEE","NextEra Energy Inc.","Utilities","Electric Utilities"),
    ("CEG","Constellation Energy Corporation","Utilities","Electric Utilities"),
    ("VST","Vistra Corp.","Utilities","Electric Utilities"),
    ("LIN","Linde plc","Materials","Industrial Gases"),
    ("FCX","Freeport-McMoRan Inc.","Materials","Copper"),
    ("NEM","Newmont Corporation","Materials","Gold"),
    ("PLD","Prologis Inc.","Real Estate","Industrial REITs"),
    ("EQIX","Equinix Inc.","Real Estate","Data Center REITs"),
    ("DLR","Digital Realty Trust Inc.","Real Estate","Data Center REITs"),
]


def yahoo_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def money(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f'<span class="ltr">${x:,.2f}</span>'


def pct_text(x, digits=1):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f'<span class="ltr">{x:.{digits}f}%</span>'


def num_text(x, digits=2):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f'<span class="ltr">{x:.{digits}f}</span>'


def safe_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def sma(series, n):
    return series.rolling(n).mean()


def atr(df, n=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def close_location_value(row):
    rng = row["High"] - row["Low"]
    if rng <= 0:
        return 0.5
    return (row["Close"] - row["Low"]) / rng


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def get_universe():
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0].copy()
        df = df.rename(columns={
            "Symbol": "symbol",
            "Security": "company",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "industry",
        })
        df["symbol"] = df["symbol"].apply(yahoo_symbol)
        df = df[["symbol", "company", "sector", "industry"]].drop_duplicates("symbol")
        if len(df) > 450:
            return df
    except Exception:
        pass

    df = pd.DataFrame(FALLBACK_UNIVERSE, columns=["symbol", "company", "sector", "industry"])
    return df.drop_duplicates("symbol")


def clean_ohlcv(df):
    if df is None or df.empty:
        return None
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if len(cols) < 5:
        return None
    out = df[cols].copy()
    out = out.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    out = out[out["Volume"] > 0]
    if len(out) < 260:
        return None
    return out


@st.cache_data(ttl=60 * 60, show_spinner=False)
def download_data(symbols):
    data = {}
    symbols = list(dict.fromkeys([yahoo_symbol(s) for s in symbols if s]))
    chunk_size = 90

    for start in range(0, len(symbols), chunk_size):
        chunk = symbols[start:start+chunk_size]
        try:
            raw = yf.download(
                tickers=" ".join(chunk),
                period="3y",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        except Exception:
            continue

        if raw is None or raw.empty:
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            for sym in chunk:
                try:
                    if sym in raw.columns.get_level_values(0):
                        df = raw[sym].copy()
                        cleaned = clean_ohlcv(df)
                        if cleaned is not None:
                            data[sym] = cleaned
                except Exception:
                    pass
        else:
            if len(chunk) == 1:
                cleaned = clean_ohlcv(raw.copy())
                if cleaned is not None:
                    data[chunk[0]] = cleaned
    return data


def latest_metrics(df):
    if df is None or len(df) < 260:
        return None
    d = df.copy()
    d["SMA10"] = sma(d["Close"], 10)
    d["SMA20"] = sma(d["Close"], 20)
    d["SMA50"] = sma(d["Close"], 50)
    d["SMA200"] = sma(d["Close"], 200)
    d["ATR14"] = atr(d, 14)
    d["AvgVol20"] = d["Volume"].rolling(20).mean()
    d["Hi20Prev"] = d["High"].rolling(20).max().shift(1)
    d["Hi55Prev"] = d["High"].rolling(55).max().shift(1)
    d["Lo10"] = d["Low"].rolling(10).min()
    d["Lo20"] = d["Low"].rolling(20).min()
    d["CloseLoc"] = d.apply(close_location_value, axis=1)
    row = d.iloc[-1]
    i = len(d)-1
    values = {
        "close": safe_float(row["Close"]),
        "high": safe_float(row["High"]),
        "low": safe_float(row["Low"]),
        "open": safe_float(row["Open"]),
        "volume": safe_float(row["Volume"]),
        "sma10": safe_float(row["SMA10"]),
        "sma20": safe_float(row["SMA20"]),
        "sma50": safe_float(row["SMA50"]),
        "sma200": safe_float(row["SMA200"]),
        "atr14": safe_float(row["ATR14"]),
        "avgvol20": safe_float(row["AvgVol20"]),
        "hi20prev": safe_float(row["Hi20Prev"]),
        "hi55prev": safe_float(row["Hi55Prev"]),
        "lo10": safe_float(row["Lo10"]),
        "lo20": safe_float(row["Lo20"]),
        "close_loc": safe_float(row["CloseLoc"]),
        "ret21": None,
        "ret63": None,
        "ret126": None,
        "ret252": None,
    }
    for n, key in [(21, "ret21"), (63, "ret63"), (126, "ret126"), (252, "ret252")]:
        if i-n >= 0:
            prev = safe_float(d["Close"].iloc[i-n])
            if prev and values["close"]:
                values[key] = (values["close"] / prev - 1) * 100
    return values, d


def market_regime(data):
    spy = data.get("SPY")
    qqq = data.get("QQQ")
    vix = data.get("^VIX")
    result = {
        "score": 0,
        "status": "לא זמין",
        "message": "לא הצלחתי לקבל תמונת שוק מלאה.",
        "risk_flag": "unknown",
        "spy_ret63": None,
        "spy_ret126": None,
    }
    if spy is None or qqq is None:
        return result

    spy_m, _ = latest_metrics(spy)
    qqq_m, _ = latest_metrics(qqq)
    if not spy_m or not qqq_m:
        return result

    spy_ok = spy_m["close"] > spy_m["sma20"] and spy_m["close"] > spy_m["sma50"] and spy_m["sma50"] > spy_m["sma200"]
    qqq_ok = qqq_m["close"] > qqq_m["sma20"] and qqq_m["close"] > qqq_m["sma50"] and qqq_m["sma50"] > qqq_m["sma200"]

    score = 0
    if spy_ok:
        score += 45
    elif spy_m["close"] > spy_m["sma50"]:
        score += 25
    if qqq_ok:
        score += 35
    elif qqq_m["close"] > qqq_m["sma50"]:
        score += 20

    vix_msg = "VIX לא זמין"
    vix_flag = False
    if vix is not None and len(vix) > 10:
        vclose = float(vix["Close"].iloc[-1])
        vprev5 = float(vix["Close"].iloc[-6]) if len(vix) >= 6 else vclose
        vchg = (vclose / vprev5 - 1) * 100 if vprev5 else 0
        if vclose < 20 and vchg < 15:
            score += 20
            vix_msg = f"VIX רגוע יחסית: {vclose:.1f}"
        elif vclose < 25 and vchg < 25:
            score += 10
            vix_msg = f"VIX בינוני: {vclose:.1f}"
        else:
            vix_flag = True
            vix_msg = f"VIX גבוה/קופץ: {vclose:.1f}, שינוי 5 ימים {vchg:.1f}%"

    if score >= 75:
        status = "שוק תומך בלונגים"
        flag = "good"
        msg = "SPY/QQQ במצב חיובי יחסית, ולכן סריקת לונגים לסווינג מקבלת רוח גבית."
    elif score >= 45:
        status = "שוק מעורב"
        flag = "neutral"
        msg = "השוק אינו מושלם, אבל אפשר לחפש רק עסקאות איכותיות ובררניות."
    else:
        status = "שוק חלש / מסוכן"
        flag = "bad"
        msg = "השוק כרגע פחות תומך בכניסות סווינג חדשות; אם מוצגות מניות, יש להתייחס אליהן בזהירות."

    if vix_flag and flag != "bad":
        flag = "neutral"
        msg += " בנוסף, ה-VIX מזהיר מפני תנודתיות גבוהה."

    result.update({
        "score": score,
        "status": status,
        "message": msg + " " + vix_msg,
        "risk_flag": flag,
        "spy_ret63": spy_m["ret63"],
        "spy_ret126": spy_m["ret126"],
    })
    return result


def classify_stretch(m):
    close = m["close"]
    s20 = m["sma20"]
    s50 = m["sma50"]
    if not close or not s20 or not s50:
        return {
            "level": "unknown", "label": "לא זמין", "dist20": None, "dist50": None,
            "penalty": 0, "text": "לא ניתן לחשב מרחק מהממוצעים."
        }
    dist20 = (close / s20 - 1) * 100
    dist50 = (close / s50 - 1) * 100

    if dist20 <= 8 and dist50 <= 15:
        return {
            "level": "normal", "label": "לא מתוחה", "dist20": dist20, "dist50": dist50,
            "penalty": 0,
            "text": "המניה אינה רחוקה מדי מהממוצעים, ולכן אפשר לבחון כניסה רגילה אם שאר התנאים מתקיימים."
        }
    if dist20 <= 15 and dist50 <= 25:
        return {
            "level": "medium", "label": "מתוחה בינונית", "dist20": dist20, "dist50": dist50,
            "penalty": -5,
            "text": "המניה חזקה, אך כבר התרחקה מהממוצעים. עדיף לא לרדוף; הכניסה צריכה להיות מותנית באישור המשך חזק או בריטסט מסודר."
        }
    return {
        "level": "high", "label": "מתוחה מאוד", "dist20": dist20, "dist50": dist50,
        "penalty": -12,
        "text": "המניה במומנטום חריג אך הכניסה במחיר הנוכחי פחות נוחה. עדיף להמתין לריטסט, דשדוש קצר או נר היפוך באזור תמיכה."
    }


def identify_pattern(d, m):
    close = m["close"]
    high = m["high"]
    hi20 = m["hi20prev"]
    hi55 = m["hi55prev"]
    vol_ratio = m["volume"] / m["avgvol20"] if m["volume"] and m["avgvol20"] else None
    close_loc = m["close_loc"] or 0.5
    s20 = m["sma20"]
    s50 = m["sma50"]
    atr14 = m["atr14"]

    if not close or not hi20 or not s20 or not s50:
        return {"type": "אין תבנית נקייה", "score": 0, "level": None, "vol_ratio": vol_ratio, "quality": "weak"}

    confirmed_breakout = (
        close > hi20 * 1.002 and
        close > s20 and
        (vol_ratio is not None and vol_ratio >= 1.25) and
        close_loc >= 0.58
    )

    if confirmed_breakout:
        return {
            "type": "פריצה מאושרת",
            "score": 22,
            "level": hi20,
            "vol_ratio": vol_ratio,
            "quality": "strong",
            "description": "סגירה מעל שיא 20 ימים, נפח גבוה מהממוצע וסגירה יחסית חזקה בתוך הנר."
        }

    # Retest after breakout in recent days
    recent = d.iloc[-13:-1].copy()
    retest_candidate = None
    for idx in range(len(recent)-1, -1, -1):
        row = recent.iloc[idx]
        pos = d.index.get_loc(row.name)
        if pos < 60:
            continue
        prior_hi20 = d["High"].iloc[pos-20:pos].max()
        vol20 = d["Volume"].iloc[pos-20:pos].mean()
        row_vol_ratio = row["Volume"] / vol20 if vol20 else 0
        if row["Close"] > prior_hi20 * 1.002 and row_vol_ratio >= 1.15:
            retest_candidate = (row.name, prior_hi20)
            break

    if retest_candidate:
        _, level = retest_candidate
        recent_lows = d["Low"].iloc[-6:]
        held = recent_lows.min() >= level * 0.975 and close > level
        positive_today = close > m["open"] and close > d["High"].iloc[-2] * 0.995
        if held and positive_today:
            return {
                "type": "ריטסט מוצלח אחרי פריצה",
                "score": 24,
                "level": level,
                "vol_ratio": vol_ratio,
                "quality": "strong",
                "description": "המניה פרצה לאחרונה, חזרה לבדוק את אזור הפריצה והצליחה להחזיק מעליו."
            }

    pullback = (
        close > s50 and
        m["sma50"] > m["sma200"] and
        abs((close / s20 - 1) * 100) <= 4.0 and
        close > m["open"] and
        (m["ret63"] or 0) > 0
    )
    if pullback:
        return {
            "type": "תיקון בריא במגמת עלייה",
            "score": 18,
            "level": s20,
            "vol_ratio": vol_ratio,
            "quality": "medium",
            "description": "המניה במגמת עלייה, נרגעה קרוב לממוצע 20 ומראה סימני קונים."
        }

    near_breakout = (
        close < hi20 and
        ((hi20 / close - 1) * 100) <= 2.5 and
        close > s20 and
        close > s50
    )
    if near_breakout:
        return {
            "type": "קרובה לפריצה",
            "score": 14,
            "level": hi20,
            "vol_ratio": vol_ratio,
            "quality": "watch",
            "description": "המניה קרובה לרמת התנגדות. אין אישור מלא, אך יש רמת טריגר ברורה."
        }

    return {
        "type": "אין תבנית נקייה",
        "score": 0,
        "level": None,
        "vol_ratio": vol_ratio,
        "quality": "weak",
        "description": "לא נמצאה כרגע תבנית כניסה מספיק נקייה."
    }


def entry_plan(m, pattern, stretch):
    close = m["close"]
    high = m["high"]
    atr14 = m["atr14"] or close * 0.025
    level = pattern.get("level") or m["hi20prev"] or m["sma20"] or close
    lo10 = m["lo10"] or close - 1.5 * atr14
    s20 = m["sma20"] or close - atr14

    if stretch["level"] == "normal":
        entry = max(high * 1.002, level * 1.003)
        stop = min(lo10, level - 0.65 * atr14, entry - 1.15 * atr14)
        entry_mode = "כניסה רגילה"
        entry_text = (
            f"כניסה אפשרית מעל {money(entry)}, בתנאי שהמחיר מחזיק מעל אזור הפריצה/התבנית ולא חוזר מהר למטה."
        )
    elif stretch["level"] == "medium":
        # Do not chase. Give two numeric routes, but mark pullback as preferred.
        pull_low = min(level, s20) - 0.25 * atr14
        pull_high = max(level, s20) + 0.35 * atr14
        pull_low = max(0.01, pull_low)
        pull_high = max(pull_low, pull_high)
        entry = pull_high * 1.003
        stop = pull_low - 0.85 * atr14
        momentum_entry = high * 1.004
        entry_mode = "כניסה מותנית — לא לרדוף"
        entry_text = (
            f"אזור הכניסה המועדף הוא ריטסט ל-{money(pull_low)}–{money(pull_high)}. "
            f"טריגר כניסה: חזרה מעל {money(entry)} לאחר הופעת קונים. "
            f"כניסה אגרסיבית יותר אפשרית רק מעל {money(momentum_entry)} אם יש המשך חזק ונפח חריג."
        )
    elif stretch["level"] == "high":
        pull_low = min(level, s20) - 0.35 * atr14
        pull_high = max(level, s20) + 0.25 * atr14
        pull_low = max(0.01, pull_low)
        pull_high = max(pull_low, pull_high)
        entry = pull_high * 1.004
        stop = pull_low - 0.9 * atr14
        entry_mode = "מעקב בלבד / כניסה עתידית"
        entry_text = (
            f"אין כניסה אידיאלית במחיר הנוכחי. מחיר כניסה עתידי מועדף: רק אם המניה חוזרת לאזור "
            f"{money(pull_low)}–{money(pull_high)} ואז מתאוששת מעל {money(entry)}. "
            f"לחלופין, להמתין לדשדוש של 2–3 ימי מסחר ולפריצת המשך חדשה."
        )
    else:
        entry = max(high * 1.002, level * 1.003)
        stop = min(lo10, entry - 1.2 * atr14)
        entry_mode = "כניסה מותנית"
        entry_text = f"כניסה אפשרית מעל {money(entry)}, בכפוף לאישור מחיר ונפח."

    if stop >= entry:
        stop = entry - 1.5 * atr14
    risk = entry - stop
    if risk <= 0:
        risk = entry * 0.06
        stop = entry - risk

    target1 = entry + 2 * risk
    target2 = entry + 3 * risk
    risk_pct = (risk / entry) * 100 if entry else None

    return {
        "entry": entry,
        "stop": stop,
        "target1": target1,
        "target2": target2,
        "risk_pct": risk_pct,
        "mode": entry_mode,
        "text": entry_text,
    }


def signal_for_backtest(d, idx, spy_d):
    if idx < 260 or idx > len(d) - 22:
        return None
    window = d.iloc[:idx+1].copy()
    m, dd = latest_metrics(window)
    if not m or not all([m["close"], m["sma50"], m["sma200"], m["avgvol20"]]):
        return None
    if m["close"] < 20:
        return None
    if m["close"] * m["avgvol20"] < 30_000_000:
        return None
    if not (m["close"] > m["sma50"] and m["sma50"] > m["sma200"]):
        return None

    # Relative strength to SPY using same date index if available
    if spy_d is not None and len(spy_d) > idx:
        try:
            spy_close = spy_d["Close"].iloc[idx]
            spy_close_63 = spy_d["Close"].iloc[idx-63]
            spy_ret63 = (spy_close / spy_close_63 - 1) * 100
            if (m["ret63"] or -999) < spy_ret63:
                return None
        except Exception:
            pass

    pattern = identify_pattern(dd, m)
    if pattern["score"] < 14:
        return None
    stretch = classify_stretch(m)
    plan = entry_plan(m, pattern, stretch)
    if plan["risk_pct"] is None or plan["risk_pct"] > 14:
        return None

    return {
        "entry": plan["entry"],
        "stop": plan["stop"],
        "target1": plan["target1"],
        "target2": plan["target2"],
        "date": d.index[idx],
        "type": pattern["type"],
    }


def backtest(df, spy_df=None):
    trades = []
    if df is None or len(df) < 320:
        return {"trades": 0}

    d = df.copy()
    # Ensure calculations warmed up in copied df
    for idx in range(260, len(d) - 22):
        sig = signal_for_backtest(d, idx, spy_df)
        if not sig:
            continue
        entry = sig["entry"]
        stop = sig["stop"]
        target1 = sig["target1"]
        target2 = sig["target2"]
        risk = entry - stop
        if risk <= 0:
            continue

        entered = False
        exit_price = None
        exit_type = "סוף תקופה"
        days = 20

        for j in range(idx+1, min(idx+21, len(d))):
            bar = d.iloc[j]
            if not entered:
                if bar["High"] >= entry:
                    entered = True
                else:
                    continue

            # conservative: if stop and target same day, assume stop first
            if bar["Low"] <= stop:
                exit_price = stop
                exit_type = "סטופ"
                days = j - idx
                break
            if bar["High"] >= target2:
                exit_price = target2
                exit_type = "יעד 3R"
                days = j - idx
                break
            if bar["High"] >= target1:
                exit_price = target1
                exit_type = "יעד 2R"
                days = j - idx
                break

        if entered:
            if exit_price is None:
                exit_price = float(d["Close"].iloc[min(idx+20, len(d)-1)])
            ret_pct = (exit_price / entry - 1) * 100
            r_mult = (exit_price - entry) / risk
            trades.append({
                "date": sig["date"],
                "entry": entry,
                "exit": exit_price,
                "ret_pct": ret_pct,
                "r": r_mult,
                "exit_type": exit_type,
                "days": days,
            })
            idx += 7

    if not trades:
        return {"trades": 0}

    rets = np.array([t["ret_pct"] for t in trades], dtype=float)
    rs = np.array([t["r"] for t in trades], dtype=float)
    wins = rets > 0
    losses = rets <= 0

    max_losing_streak = 0
    cur = 0
    for w in wins:
        if not w:
            cur += 1
            max_losing_streak = max(max_losing_streak, cur)
        else:
            cur = 0

    recent = trades[-8:]
    recent_avg_r = float(np.mean([t["r"] for t in recent])) if recent else None

    return {
        "trades": len(trades),
        "win_rate": float(np.mean(wins) * 100),
        "avg_return": float(np.mean(rets)),
        "avg_r": float(np.mean(rs)),
        "avg_winner_return": float(np.mean(rets[wins])) if np.any(wins) else None,
        "avg_winner_r": float(np.mean(rs[wins])) if np.any(wins) else None,
        "avg_loser_return": float(np.mean(rets[losses])) if np.any(losses) else None,
        "avg_loser_r": float(np.mean(rs[losses])) if np.any(losses) else None,
        "worst_r": float(np.min(rs)),
        "best_r": float(np.max(rs)),
        "max_losing_streak": max_losing_streak,
        "recent_avg_r": recent_avg_r,
    }


POSITIVE_KEYWORDS = [
    "beat", "beats", "raises", "raised", "upgrade", "upgraded", "outperform", "buy rating",
    "price target raised", "record", "strong demand", "contract", "partnership", "approval",
    "surges", "rises", "growth", "guidance raised", "profit jumps", "revenue growth"
]
NEGATIVE_KEYWORDS = [
    "miss", "misses", "cuts", "cut", "downgrade", "downgraded", "underperform", "sell rating",
    "price target cut", "lawsuit", "probe", "investigation", "warning", "weak demand",
    "falls", "drops", "slumps", "guidance cut", "revenue decline", "loss widens"
]


@st.cache_data(ttl=2 * 60 * 60, show_spinner=False)
def get_news_and_earnings(symbol):
    out = {
        "news_score": 0,
        "news_summary": "לא נמצאו חדשות מהותיות זמינות במקור הנתונים.",
        "news_items": [],
        "earnings_warning": None,
    }
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news or []
        items = []
        score = 0
        for n in news[:8]:
            title = (n.get("title") or "").strip()
            if not title:
                continue
            link = n.get("link") or ""
            provider = ""
            if isinstance(n.get("publisher"), str):
                provider = n.get("publisher")
            text = title.lower()
            s = 0
            if any(k in text for k in POSITIVE_KEYWORDS):
                s += 1
            if any(k in text for k in NEGATIVE_KEYWORDS):
                s -= 1
            score += s
            items.append({"title": title, "score": s, "provider": provider, "link": link})
        if items:
            if score > 0:
                summary = "החדשות האחרונות נוטות חיוביות ומחזקות את האיתות."
            elif score < 0:
                summary = "יש כותרות שליליות/זהירות, ולכן נדרש משנה זהירות."
            else:
                summary = "לא נמצאה הטיה חדשותית ברורה; האיתות מבוסס בעיקר על מחיר ונפח."
            out.update({"news_score": max(-8, min(8, score * 2)), "news_summary": summary, "news_items": items[:3]})
    except Exception:
        pass

    # Earnings date warning only, without excluding the stock
    try:
        ticker = yf.Ticker(symbol)
        cal = ticker.calendar
        earnings_dates = []
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            # yfinance formats vary; search for date-like values
            for val in cal.values.flatten():
                if isinstance(val, (pd.Timestamp, datetime, date)):
                    earnings_dates.append(pd.Timestamp(val).date())
        elif isinstance(cal, dict):
            for val in cal.values():
                if isinstance(val, (list, tuple)):
                    for x in val:
                        if isinstance(x, (pd.Timestamp, datetime, date)):
                            earnings_dates.append(pd.Timestamp(x).date())
                elif isinstance(val, (pd.Timestamp, datetime, date)):
                    earnings_dates.append(pd.Timestamp(val).date())

        today = datetime.utcnow().date()
        future = sorted([d for d in earnings_dates if d >= today])
        if future:
            days = (future[0] - today).days
            if days <= 10:
                out["earnings_warning"] = f"דוח קרוב בעוד {days} ימים בערך — לא פוסל את המניה, אך מעלה סיכון לגאפ."
    except Exception:
        pass

    return out


def score_backtest(bt):
    if not bt or bt.get("trades", 0) == 0:
        return 0
    trades = bt.get("trades", 0)
    avg_r = bt.get("avg_r") or 0
    recent = bt.get("recent_avg_r")
    score = 0
    if trades >= 5:
        score += 4
    if trades >= 10:
        score += 2
    if avg_r > 0.15:
        score += 5
    elif avg_r > 0:
        score += 2
    if recent is not None and recent > 0:
        score += 2
    if bt.get("max_losing_streak", 0) >= 4:
        score -= 2
    return max(-4, min(10, score))


def analyze_stock(symbol, company, sector, industry, df, data, market, percentile_lookup):
    try:
        m, d = latest_metrics(df)
        if not m:
            return None
        close = m["close"]
        avg_dollar_vol = close * m["avgvol20"] if close and m["avgvol20"] else 0
        if not close or close < 20 or avg_dollar_vol < 30_000_000:
            return None

        trend_ok = close > m["sma50"] and m["sma50"] > m["sma200"]
        if not trend_ok:
            return None

        pattern = identify_pattern(d, m)
        if pattern["score"] <= 0:
            return None

        stretch = classify_stretch(m)
        plan = entry_plan(m, pattern, stretch)

        if plan["risk_pct"] is None or plan["risk_pct"] > 16:
            return None

        rel63 = (m["ret63"] or 0) - (market.get("spy_ret63") or 0)
        rel126 = (m["ret126"] or 0) - (market.get("spy_ret126") or 0)

        rs_rank = percentile_lookup.get(symbol, 50)

        etf = SECTOR_ETF.get(sector)
        sector_strength = None
        sector_msg = "לא זמין"
        sector_score = 0
        if etf and data.get(etf) is not None:
            em, _ = latest_metrics(data.get(etf))
            if em:
                s63 = (em["ret63"] or 0) - (market.get("spy_ret63") or 0)
                s126 = (em["ret126"] or 0) - (market.get("spy_ret126") or 0)
                sector_strength = (s63 + s126) / 2
                if sector_strength > 3:
                    sector_score = 12
                    sector_msg = "הסקטור חזק מהשוק"
                elif sector_strength > 0:
                    sector_score = 7
                    sector_msg = "הסקטור מעט חזק מהשוק"
                else:
                    sector_score = -2
                    sector_msg = "הסקטור חלש יחסית ל-SPY"

        vol_ratio = pattern.get("vol_ratio")
        vol_score = 0
        if vol_ratio is not None:
            if vol_ratio >= 1.6:
                vol_score = 10
            elif vol_ratio >= 1.25:
                vol_score = 7
            elif vol_ratio >= 1.0:
                vol_score = 3

        score = 0
        score += 15 if trend_ok else 0
        score += min(20, max(0, (rs_rank - 50) / 50 * 20))  # top ranks get more points
        if rel63 > 0:
            score += 5
        if rel126 > 0:
            score += 5
        score += sector_score
        score += pattern["score"]
        score += vol_score
        score += 6 if market["risk_flag"] == "good" else (2 if market["risk_flag"] == "neutral" else -5)
        score += stretch["penalty"]
        if plan["risk_pct"] <= 8:
            score += 6
        elif plan["risk_pct"] <= 12:
            score += 3

        return {
            "symbol": symbol,
            "company": company,
            "sector": sector or "Unknown",
            "sector_he": SECTOR_HE.get(sector, sector or "לא זמין"),
            "industry": industry or "לא זמין",
            "sector_etf": etf or "לא זמין",
            "sector_msg": sector_msg,
            "sector_strength": sector_strength,
            "metrics": m,
            "pattern": pattern,
            "stretch": stretch,
            "plan": plan,
            "score_base": score,
            "score": score,
            "rel63": rel63,
            "rel126": rel126,
            "rs_rank": rs_rank,
            "avg_dollar_vol": avg_dollar_vol,
            "df": df,
        }
    except Exception:
        return None


def company_blurb(row):
    company = row["company"]
    sector_he = row["sector_he"]
    industry = row["industry"]
    return (
        f"{company} היא חברה הנכללת במדד S&P 500 ופועלת בתחום {sector_he}. "
        f"תחום הפעילות המדווח שלה: {industry}. ההקשר העסקי הזה חשוב כי הסורק בודק לא רק את המניה, "
        f"אלא גם האם הסקטור שלה תומך במהלך."
    )


def render_result(row, idx):
    """Render one result using native Streamlit elements only.

    Previous versions used a large raw-HTML block. On mobile, Streamlit sometimes
    escaped that block and displayed the HTML tags as text. This renderer avoids
    raw HTML entirely inside the result card, so Hebrew/English/numbers stay
    readable and no tags are shown to the user.
    """
    symbol = row["symbol"]
    m = row["metrics"]
    p = row["pattern"]
    s = row["stretch"]
    plan = row["plan"]
    bt = row.get("backtest", {}) or {}
    news = row.get("news", {}) or {}

    sector_label = row.get("sector_he") or row.get("sector") or "לא ידוע"
    title = f"{idx}. {symbol} — {p.get('type', 'איתות')} — {sector_label} — ציון {int(round(row['score']))}/100"

    with st.expander(title, expanded=(idx == 1)):
        st.subheader(f"{symbol} — {row.get('company', symbol)}")
        st.caption(f"סוג איתות: {p.get('type', 'לא ידוע')} | ציון: {int(round(row['score']))}/100")

        st.markdown("### הסבר קצר על החברה")
        st.write(company_blurb(row))

        st.markdown("### נתוני חברה וסקטור")
        info_rows = [
            ("חברה", f"{row.get('company', symbol)} ({symbol})"),
            ("סקטור", f"{row.get('sector_he', 'לא ידוע')} / {row.get('sector', 'Unknown')}"),
            ("תחום פעילות", row.get("industry") or "לא ידוע"),
            ("ETF סקטוריאלי", row.get("sector_etf") or "—"),
            ("חוזק הסקטור מול SPY", row.get("sector_msg") or "לא זמין"),
        ]
        for label, value in info_rows:
            st.write(f"**{label}:** {value}")

        st.markdown("### נתוני מחיר מרכזיים")
        c1, c2, c3 = st.columns(3)
        c1.metric("מחיר נוכחי", money(m.get("close")))
        c2.metric("כניסה / טריגר", money(plan.get("entry")))
        c3.metric("סוג המלצה", plan.get("mode", "—"))

        st.markdown("### מתי להיכנס")
        if s.get("level") == "high":
            st.warning(plan.get("text", "אין הנחיית כניסה זמינה."))
        elif s.get("level") == "medium":
            st.info(plan.get("text", "אין הנחיית כניסה זמינה."))
        else:
            st.success(plan.get("text", "אין הנחיית כניסה זמינה."))

        st.markdown("### סטופ ויעדים")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("סטופ לוס", money(plan.get("stop")))
        s2.metric("יעד ראשון", money(plan.get("target1")))
        s3.metric("יעד שני", money(plan.get("target2")))
        s4.metric("סיכון עד הסטופ", pct_text(plan.get("risk_pct")))

        st.markdown("### מדוע היא נכנסה לסריקה")
        st.write(p.get("description", "לא קיים הסבר זמין."))
        st.write(f"**חוזק יחסי מול SPY:** 63 ימים {pct_text(row.get('rel63'))}, 126 ימים {pct_text(row.get('rel126'))}.")
        st.write(f"**דירוג חוזק יחסי בתוך הסריקה:** {num_text(row.get('rs_rank'), 0)} מתוך 100.")

        st.markdown("### סטטוס מתיחה מהממוצעים")
        stretch_msg = (
            f"{s.get('label', 'לא ידוע')} — מרחק ממוצע 20: {pct_text(s.get('dist20'))}, "
            f"מרחק ממוצע 50: {pct_text(s.get('dist50'))}. {s.get('text', '')}"
        )
        if s.get("level") == "high":
            st.warning(stretch_msg)
        elif s.get("level") == "medium":
            st.info(stretch_msg)
        else:
            st.success(stretch_msg)

        st.markdown("### חדשות ודוחות")
        earnings_warning = news.get("earnings_warning")
        if earnings_warning:
            st.warning(earnings_warning)
        st.write(news.get("news_summary", "לא נמצאו חדשות מהותיות זמינות במקור הנתונים."))
        items = news.get("news_items") or []
        if items:
            for item in items:
                sign = "חיובי" if item.get("score", 0) > 0 else ("שלילי" if item.get("score", 0) < 0 else "נייטרלי")
                st.markdown(f"- **{sign}:** {item.get('title', '')}")

        st.markdown("### בדיקה היסטורית של איתותים דומים")
        if bt.get("trades", 0) == 0:
            st.info("לא נמצאו מספיק איתותים דומים בעבר לצורך מדידה היסטורית משמעותית.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("מספר עסקאות דומות", f"{bt.get('trades')}")
            c2.metric("אחוז הצלחה", f"{bt.get('win_rate', 0):.1f}%")
            c3.metric("ממוצע תשואה כולל", f"{bt.get('avg_return', 0):.2f}%")
            c4.metric("תוחלת ממוצעת", f"{bt.get('avg_r', 0):.2f}R")
            c5, c6, c7, c8 = st.columns(4)
            aw = bt.get("avg_winner_return")
            ar = bt.get("avg_winner_r")
            al = bt.get("avg_loser_return")
            worst = bt.get("worst_r")
            c5.metric("ממוצע תשואה כשהעסקה הצליחה", "—" if aw is None else f"{aw:.2f}%")
            c6.metric("ממוצע R בעסקאות מצליחות", "—" if ar is None else f"{ar:.2f}R")
            c7.metric("ממוצע הפסד בעסקאות כושלות", "—" if al is None else f"{al:.2f}%")
            c8.metric("העסקה הגרועה ביותר", "—" if worst is None else f"{worst:.2f}R")
            st.caption(
                "הבדיקה ההיסטורית אינה הבטחה לתוצאה עתידית. היא בודקת מה קרה בעבר כאשר הופיעו תנאים דומים של מגמה, פריצה/ריטסט, חוזק יחסי וסיכון."
            )

        st.markdown("### מתי לא להיכנס")
        if s.get("level") == "high":
            st.write("לא להיכנס במרדף אחרי המחיר הנוכחי. להמתין לריטסט, דשדוש קצר, או פריצת המשך חדשה לאחר מנוחה.")
        elif s.get("level") == "medium":
            st.write("לא להיכנס אם המחיר פותח בגאפ חד למעלה בלי יכולת להציב סטופ סביר. עדיף להמתין לאישור או ריטסט.")
        else:
            st.write("לא להיכנס אם המחיר חוזר מתחת לרמת הפריצה/התבנית, אם השוק הכללי נשבר, או אם הסיכון עד הסטופ גדול מדי.")

def main():
    st.title("📈 סורק סווינג — S&P 500")
    st.markdown(
        """
        <div class="small-note">
        כפתור אחד. הסורק מחפש עד 5 מניות חזקות לסווינג לפי מגמה, חוזק מול SPY, חוזק סקטור, פריצה/ריטסט, נפח, VIX, חדשות, דוחות קרובים ובדיקה היסטורית.
        זהו כלי מחקר בלבד ואינו ייעוץ השקעות.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "results" not in st.session_state:
        st.session_state.results = None
        st.session_state.market = None

    if st.button("סרוק עכשיו", type="primary", use_container_width=True):
        st.session_state.results = None
        st.session_state.market = None

        universe = get_universe()
        sector_etfs = sorted(set(SECTOR_ETF.values()))
        all_symbols = list(universe["symbol"].unique()) + ["SPY", "QQQ", "^VIX"] + sector_etfs

        progress = st.progress(0)
        status = st.empty()

        status.write("מוריד נתוני שוק ומניות S&P 500...")
        data = download_data(all_symbols)
        progress.progress(25)

        status.write("בודק מצב שוק, SPY / QQQ / VIX...")
        market = market_regime(data)
        progress.progress(35)

        # Compute relative strength ranks from available stocks
        rows_for_rank = []
        for _, r in universe.iterrows():
            sym = r["symbol"]
            if sym in data:
                lm = latest_metrics(data[sym])
                if lm:
                    m, _ = lm
                    score = 0
                    for key, w in [("ret21", 0.2), ("ret63", 0.35), ("ret126", 0.3), ("ret252", 0.15)]:
                        score += (m.get(key) or 0) * w
                    rows_for_rank.append((sym, score))
        if rows_for_rank:
            rank_df = pd.DataFrame(rows_for_rank, columns=["symbol", "rs_score"])
            rank_df["percentile"] = rank_df["rs_score"].rank(pct=True) * 100
            percentile_lookup = dict(zip(rank_df["symbol"], rank_df["percentile"]))
        else:
            percentile_lookup = {}

        progress.progress(45)
        status.write("מסנן מניות לפי מגמה, חוזק יחסי, תבנית, סקטור וסיכון...")
        candidates = []
        for _, r in universe.iterrows():
            sym = r["symbol"]
            if sym not in data:
                continue
            analyzed = analyze_stock(
                sym,
                r.get("company", sym),
                r.get("sector", "Unknown"),
                r.get("industry", "לא זמין"),
                data[sym],
                data,
                market,
                percentile_lookup,
            )
            if analyzed:
                candidates.append(analyzed)

        if not candidates:
            progress.progress(100)
            st.session_state.results = []
            st.session_state.market = market
            status.empty()
            st.warning("לא נמצאו מניות שעברו את הסינון כרגע. ייתכן שהשוק חלש או שמקור הנתונים לא החזיר מספיק מידע.")
            return

        # Preselect top 30 for expensive enrichment
        candidates = sorted(candidates, key=lambda x: x["score_base"], reverse=True)[:30]
        progress.progress(62)

        status.write("מבצע Backtest ומוסיף חדשות/דוחות למועמדות המובילות...")
        enriched = []
        spy_df = data.get("SPY")
        for i, c in enumerate(candidates):
            bt = backtest(c["df"], spy_df)
            news = get_news_and_earnings(c["symbol"])
            c["backtest"] = bt
            c["news"] = news
            c["score"] = c["score_base"] + score_backtest(bt) + news.get("news_score", 0)
            if news.get("earnings_warning"):
                c["score"] -= 2  # warning only, not exclusion
            enriched.append(c)
            progress.progress(62 + int((i+1) / len(candidates) * 30))

        # Backtest as filter-lite: do not eliminate aggressively, but poor history lowers ranking.
        results = sorted(enriched, key=lambda x: x["score"], reverse=True)[:5]
        progress.progress(100)
        status.empty()
        st.session_state.results = results
        st.session_state.market = market

    if st.session_state.market:
        m = st.session_state.market
        if m["risk_flag"] == "good":
            st.success(f"{m['status']} — {m['message']}")
        elif m["risk_flag"] == "bad":
            st.error(f"{m['status']} — {m['message']}")
        else:
            st.warning(f"{m['status']} — {m['message']}")

    results = st.session_state.results
    if results is not None:
        if not results:
            st.info("אין כרגע 5 מניות איכותיות להצגה לפי תנאי הסריקה.")
        else:
            st.subheader("5 המניות המובילות כרגע")
            for i, row in enumerate(results, start=1):
                render_result(row, i)

    st.markdown(
        """
        ---
        <div class="small-note">
        הערה: הסורק מסתמך על נתוני Yahoo Finance דרך yfinance ועל מידע זמין באינטרנט. ייתכנו עיכובים, חוסרים או טעויות בנתונים.
        אין לראות בתוצאה המלצה לקנייה או מכירה. יש לבדוק ידנית גרף, חדשות, דוחות, גודל פוזיציה וסיכון אישי.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
