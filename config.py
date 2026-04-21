"""
config.py
─────────────────────────────────────────────────────────────────────────────
Central configuration for the Nifty-50 Stock Predictor.
Edit ONLY this file to change stocks, API keys, date ranges,
model thresholds, or feature lists.
─────────────────────────────────────────────────────────────────────────────
"""

from datetime import datetime

# ── API Keys ──────────────────────────────────────────────────────────────────
import os
import streamlit as st

NEWS_API_KEY: str = (
    st.secrets.get("NEWS_API_KEY")          # Streamlit Cloud
    or os.getenv("NEWS_API_KEY")            # local terminal / train.py
    or ""
)
# ── Date Range ────────────────────────────────────────────────────────────────
START_DATE: str = "2020-01-01"
END_DATE:   str = datetime.today().strftime("%Y-%m-%d")

# ── Nifty 50 Universe ─────────────────────────────────────────────────────────
STOCKS: list[str] = [
    "RELIANCE.NS", "TCS.NS",       "HDFCBANK.NS",   "INFY.NS",
    "ICICIBANK.NS","HINDUNILVR.NS","ITC.NS",         "SBIN.NS",
    "BHARTIARTL.NS","LT.NS",       "KOTAKBANK.NS",   "AXISBANK.NS",
    "ASIANPAINT.NS","MARUTI.NS",   "SUNPHARMA.NS",   "TITAN.NS",
    "ULTRACEMCO.NS","BAJFINANCE.NS","WIPRO.NS",       "NTPC.NS",
    "POWERGRID.NS", "ONGC.NS",     "HCLTECH.NS",     "ADANIENT.NS",
    "ADANIPORTS.NS","NESTLEIND.NS","JSWSTEEL.NS",    "TATAMOTORS.NS",
    "INDUSINDBK.NS","BAJAJFINSV.NS","COALINDIA.NS",  "TECHM.NS",
    "GRASIM.NS",    "DRREDDY.NS",  "HINDALCO.NS",    "CIPLA.NS",
    "TATASTEEL.NS", "DIVISLAB.NS", "BPCL.NS",        "EICHERMOT.NS",
    "HEROMOTOCO.NS","BRITANNIA.NS","APOLLOHOSP.NS",  "BAJAJ-AUTO.NS",
    "SHRIRAMFIN.NS","SBILIFE.NS",  "HDFCLIFE.NS",    "UPL.NS",
    "TATACONSUM.NS","M&M.NS",
]

STOCK_NAME_MAP: dict[str, str] = {
    "RELIANCE.NS":   "Reliance Industries",
    "TCS.NS":        "Tata Consultancy Services",
    "INFY.NS":       "Infosys",
    "HDFCBANK.NS":   "HDFC Bank",
    "ICICIBANK.NS":  "ICICI Bank",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "SBIN.NS":       "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "ITC.NS":        "ITC Limited",
    "KOTAKBANK.NS":  "Kotak Mahindra Bank",
    "LT.NS":         "Larsen and Toubro",
    "AXISBANK.NS":   "Axis Bank",
    "ASIANPAINT.NS": "Asian Paints",
    "MARUTI.NS":     "Maruti Suzuki",
    "SUNPHARMA.NS":  "Sun Pharma",
    "TITAN.NS":      "Titan Company",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "NESTLEIND.NS":  "Nestle India",
    "WIPRO.NS":      "Wipro",
    "BAJFINANCE.NS": "Bajaj Finance",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "ADANIENT.NS":   "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports",
    "HCLTECH.NS":    "HCL Technologies",
    "NTPC.NS":       "NTPC",
    "POWERGRID.NS":  "Power Grid Corporation",
    "ONGC.NS":       "Oil and Natural Gas Corporation",
    "TECHM.NS":      "Tech Mahindra",
    "TATAMOTORS.NS": "Tata Motors",
    "TATASTEEL.NS":  "Tata Steel",
    "TATACONSUM.NS": "Tata Consumer Products",
    "M&M.NS":        "Mahindra and Mahindra",
    "JSWSTEEL.NS":   "JSW Steel",
    "HINDALCO.NS":   "Hindalco Industries",
    "DRREDDY.NS":    "Dr Reddy's Laboratories",
    "CIPLA.NS":      "Cipla",
    "DIVISLAB.NS":   "Divi's Laboratories",
    "EICHERMOT.NS":  "Eicher Motors",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "BRITANNIA.NS":  "Britannia Industries",
    "APOLLOHOSP.NS": "Apollo Hospitals",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "SHRIRAMFIN.NS": "Shriram Finance",
    "SBILIFE.NS":    "SBI Life Insurance",
    "HDFCLIFE.NS":   "HDFC Life Insurance",
    "UPL.NS":        "UPL Limited",
    "INDUSINDBK.NS": "IndusInd Bank",
    "COALINDIA.NS":  "Coal India",
    "GRASIM.NS":     "Grasim Industries",
    "BPCL.NS":       "Bharat Petroleum",
}

# ── Feature List (DO NOT CHANGE — must match training columns exactly) ─────────
FEATURES: list[str] = [
    "Open", "High", "Low", "Close", "Volume",
    "EMA", "SMA", "momentum", "breakout",
    "RSI", "ema_diff", "acceleration", "vol_spike",
    "MACD", "MACD_signal", "MACD_diff",
    "bb_width", "volatility", "trend_strength",
    "returns", "lag1", "lag2", "lag3",
    "sentiment",
]

# ── Prediction Thresholds (DO NOT CHANGE — same as original notebook) ──────────
BUY_THRESHOLD:   float = 0.62
SELL_THRESHOLD:  float = 0.38
FINAL_THRESHOLD: float = 0.54   # hard decision threshold for evaluate()

# ── Labeling (DO NOT CHANGE) ──────────────────────────────────────────────────
LABEL_HORIZON:    int   = 5      # forward days for future_return
VOLATILITY_MULT:  float = 0.8   # multiplier on 20-day vol for threshold

# ── Backtesting / Filtering ────────────────────────────────────────────────────
WEAK_SIGNAL_FILTER: float = 0.005   # ignore |future_return| below this

# ── Optuna / Training ─────────────────────────────────────────────────────────
OPTUNA_TRIALS:  int = 30
TSCV_SPLITS:    int = 5
RANDOM_STATE:   int = 42

# ── Persistence ───────────────────────────────────────────────────────────────
MODEL_DIR:      str = "models"
MODEL_PATH:     str = f"{MODEL_DIR}/xgb_model.pkl"
PARAMS_PATH:    str = f"{MODEL_DIR}/best_params.pkl"

# ── FinBERT ───────────────────────────────────────────────────────────────────
FINBERT_MODEL:  str = "ProsusAI/finbert"
SENTIMENT_DAYS: int = 30          # only fetch live news for last N days

# ── News API ──────────────────────────────────────────────────────────────────
NEWS_PAGE_SIZE: int = 5
NEWS_LANGUAGE:  str = "en"
