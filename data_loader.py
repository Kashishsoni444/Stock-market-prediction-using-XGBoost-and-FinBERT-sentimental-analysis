"""
data_loader.py
─────────────────────────────────────────────────────────────────────────────
Handles:
  • Parallel yfinance OHLCV download
  • Technical-indicator feature engineering  (IDENTICAL to notebook logic)
  • NewsAPI fetching with in-memory + disk cache
  • FinBERT sentiment scoring  (batched for speed)
  • Label creation  (identical to notebook)
  • Full dataset assembly

Optimisations vs original notebook
  • Parallel stock downloads via ThreadPoolExecutor
  • Batched FinBERT inference (all headlines in one forward pass per date)
  • Persistent news cache backed to disk (JSON) so restarts are free
  • No repeated DataFrame copies; all mutations done in-place
  • MultiIndex column flattening handled once at download time
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import torch
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.volatility import BollingerBands
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import (
    NEWS_API_KEY, START_DATE, END_DATE,
    STOCKS, STOCK_NAME_MAP, FEATURES,
    LABEL_HORIZON, VOLATILITY_MULT, WEAK_SIGNAL_FILTER,
    FINBERT_MODEL, SENTIMENT_DAYS,
    NEWS_PAGE_SIZE, NEWS_LANGUAGE,
)

logger = logging.getLogger(__name__)

# ── Disk-backed news cache ────────────────────────────────────────────────────
_CACHE_FILE = Path("news_cache.json")

def _load_disk_cache() -> dict:
    if _CACHE_FILE.exists():
        try:
            return json.loads(_CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}

def _save_disk_cache(cache: dict) -> None:
    try:
        _CACHE_FILE.write_text(json.dumps(cache))
    except Exception:
        pass

_news_cache: dict = _load_disk_cache()


# ─────────────────────────────────────────────────────────────────────────────
# NEWS FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_news(company: str, date: str) -> list[str]:
    """
    Fetch up to NEWS_PAGE_SIZE headlines for *company* on *date*.
    Results are cached in memory and on disk to avoid repeated API calls.
    """
    key = f"{company}_{date}"
    if key in _news_cache:
        return _news_cache[key]

    url = "https://newsapi.org/v2/everything"
    params = {
        "q":        company,
        "from":     date,
        "to":       date,
        "language": NEWS_LANGUAGE,
        "sortBy":   "relevancy",
        "pageSize": NEWS_PAGE_SIZE,
        "apiKey":   NEWS_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        articles = resp.json().get("articles", [])
    except Exception as exc:
        logger.warning("NewsAPI error for %s on %s: %s", company, date, exc)
        articles = []

    texts = [
        (a["title"] + " " + str(a.get("description", "")))
        for a in articles if a.get("title")
    ]

    _news_cache[key] = texts
    _save_disk_cache(_news_cache)
    return texts


# ─────────────────────────────────────────────────────────────────────────────
# FINBERT SENTIMENT ANALYSER
# ─────────────────────────────────────────────────────────────────────────────

class SentimentAnalyzer:
    """
    Lazy-loaded FinBERT wrapper.
    Call .get_score(texts) for a list of strings → numpy array of scores.
    Scores = P(positive) - P(negative), identical to original notebook.
    """

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        self.model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
        self.model.eval()

    @torch.no_grad()
    def get_score(self, texts: list[str]) -> np.ndarray:
        """Batch inference — all texts in a single forward pass."""
        inputs  = self.tokenizer(
            texts, return_tensors="pt",
            padding=True, truncation=True, max_length=512,
        )
        logits  = self.model(**inputs).logits
        probs   = torch.nn.functional.softmax(logits, dim=-1)
        # probs[:,0] = negative  probs[:,1] = neutral  probs[:,2] = positive
        scores  = (probs[:, 2] - probs[:, 0]).numpy()
        return scores


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (logic IDENTICAL to notebook)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_stock_data(ticker: str) -> pd.DataFrame:
    """
    Download OHLCV from yfinance and compute all technical indicators.
    The computation order and formulas are byte-for-byte identical to
    the original notebook cell 6.
    """
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)

    # Flatten MultiIndex columns that yfinance sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # --- Trend indicators ---
    df["EMA"]   = EMAIndicator(close=df["Close"], window=10).ema_indicator()
    df["SMA"]   = SMAIndicator(close=df["Close"], window=10).sma_indicator()
    df["ema_diff"]       = df["EMA"] - df["SMA"]
    df["trend_strength"] = df["EMA"] - df["SMA"]

    # --- Momentum / breakout ---
    df["momentum"]   = df["Close"] - df["Close"].shift(5)
    df["acceleration"] = df["Close"].diff(2)
    df["breakout"]   = (df["Close"] > df["High"].rolling(10).max().shift(1)).astype(int)

    # --- RSI ---
    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()

    # --- Volume spike ---
    df["vol_spike"] = df["Volume"] / df["Volume"].rolling(10).mean()

    # --- MACD ---
    macd = MACD(close=df["Close"])
    df["MACD"]        = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_diff"]   = macd.macd_diff()

    # --- Bollinger Bands ---
    bb = BollingerBands(close=df["Close"], window=20)
    df["bb_high"]  = bb.bollinger_hband()
    df["bb_low"]   = bb.bollinger_lband()
    df["bb_width"] = df["bb_high"] - df["bb_low"]

    # --- Volatility & returns ---
    df["volatility"] = df["Close"].pct_change().rolling(10).std()
    df["returns"]    = df["Close"].pct_change()
    df["lag1"]       = df["returns"].shift(1)
    df["lag2"]       = df["returns"].shift(2)
    df["lag3"]       = df["returns"].shift(3)

    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT ENRICHMENT  (logic IDENTICAL to notebook)
# ─────────────────────────────────────────────────────────────────────────────

def add_sentiment(
    df: pd.DataFrame,
    stock: str,
    sentiment_model: SentimentAnalyzer,
) -> pd.DataFrame:
    """
    Fetch FinBERT sentiment for the last SENTIMENT_DAYS days.
    Older rows get NaN → forward-filled → 0.
    Logic is identical to notebook cell 7.
    """
    company = STOCK_NAME_MAP.get(stock, stock)
    cutoff  = datetime.today() - timedelta(days=SENTIMENT_DAYS)

    sentiments: list[Optional[float]] = []

    for date in df.index:
        if date.to_pydatetime().replace(tzinfo=None) < cutoff:
            sentiments.append(np.nan)
            continue

        d    = date.strftime("%Y-%m-%d")
        news = fetch_news(company, d)
        if not news:
            sentiments.append(np.nan)
        else:
            # batch inference: all headlines for this date in one pass
            sentiments.append(float(np.mean(sentiment_model.get_score(news))))

    df["sentiment"] = sentiments
    df["sentiment"] = df["sentiment"].ffill().fillna(0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LABEL CREATION  (logic IDENTICAL to notebook)
# ─────────────────────────────────────────────────────────────────────────────

def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates binary BUY/SELL signal using volatility-adjusted future return.
    Logic is identical to notebook cell 8.
    """
    df["future_return"] = df["Close"].shift(-LABEL_HORIZON) / df["Close"] - 1
    df["volatility_20"] = df["Close"].pct_change().rolling(20).std()

    threshold   = VOLATILITY_MULT * df["volatility_20"]
    df["signal"] = np.where(df["future_return"] >= threshold, 1, 0)

    df.dropna(inplace=True)
    return df


def filter_weak_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where |future_return| is too small to be meaningful."""
    return df[df["future_return"].abs() > WEAK_SIGNAL_FILTER]


# ─────────────────────────────────────────────────────────────────────────────
# FULL DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _process_one_stock(
    stock: str,
    sentiment_model: SentimentAnalyzer,
) -> Optional[pd.DataFrame]:
    """Process a single stock end-to-end; returns None on failure."""
    try:
        df = fetch_stock_data(stock)
        df = add_sentiment(df, stock, sentiment_model)
        df = create_labels(df)
        df = filter_weak_signals(df)
        df["stock"] = stock
        return df
    except Exception as exc:
        logger.error("Failed to process %s: %s", stock, exc)
        return None


def build_dataset(
    stocks: list[str] = STOCKS,
    max_workers: int  = 4,
) -> pd.DataFrame:
    """
    Download and process all stocks in parallel (IO-bound download phase),
    then combine into one DataFrame.

    The per-stock logic is identical to notebook cell 9; only the
    outer loop is parallelised for speed.
    """
    sentiment_model = SentimentAnalyzer()
    results: list[pd.DataFrame] = []

    # yfinance network calls benefit from thread parallelism
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process_one_stock, s, sentiment_model): s
            for s in stocks
        }
        for fut in as_completed(futures):
            stock = futures[fut]
            df    = fut.result()
            if df is not None:
                logger.info("✓  %s processed (%d rows)", stock, len(df))
                results.append(df)
            else:
                logger.warning("✗  %s skipped", stock)

    if not results:
        raise RuntimeError("No stock data could be loaded.")

    return pd.concat(results, ignore_index=False)
