"""
predictor.py
─────────────────────────────────────────────────────────────────────────────
Handles:
  • Live BUY / SELL / NO TRADE signal for a single stock
  • Confidence probability output
  • Recent technical-indicator summary (last row of df)
  • Per-stock evaluation metrics (buy_precision, sell_precision, accuracy)

The prediction threshold logic is IDENTICAL to notebook cell 16 & 17.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from config import (
    FEATURES,
    BUY_THRESHOLD, SELL_THRESHOLD, FINAL_THRESHOLD,
)
from data_loader import (
    SentimentAnalyzer,
    fetch_stock_data,
    add_sentiment,
    create_labels,
    filter_weak_signals,
)
from model_utils import (
    backtest,
    evaluate,
    prepare_data,
    balance_data,
    tune_xgb,
    train_model,
    EvalMetrics,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RESULT TYPE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    ticker:         str
    signal:         str          # "BUY" | "SELL" | "NO TRADE"
    probability:    float        # P(BUY)  0–1
    accuracy:       float
    f1:             float
    buy_precision:  float
    sell_precision: float
    total_return:   float
    total_trades:   int
    sentiment:      float
    latest_indicators: dict[str, float]


# ─────────────────────────────────────────────────────────────────────────────
# LIVE SIGNAL  (logic IDENTICAL to notebook cell 16)
# ─────────────────────────────────────────────────────────────────────────────

def live_prediction(
    model: XGBClassifier,
    df: pd.DataFrame,
) -> tuple[str, float]:
    """
    Predict BUY / SELL / NO TRADE for the latest row of df.
    Threshold logic is identical to notebook cell 16.

    Returns (signal_label, probability_of_BUY).
    """
    latest = df[FEATURES].iloc[-1:]
    prob   = float(model.predict_proba(latest)[0][1])

    if prob > BUY_THRESHOLD:
        signal = "BUY"
    elif prob < SELL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "NO TRADE"

    return signal, prob


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATOR SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def get_indicator_summary(df: pd.DataFrame) -> dict[str, float]:
    """
    Return the most recent values of key technical indicators as a dict.
    These are read directly from the last row — no recomputation.
    """
    row = df.iloc[-1]
    summary_cols = [
        "Close", "EMA", "SMA", "RSI", "MACD", "MACD_signal",
        "bb_width", "volatility", "momentum", "sentiment",
    ]
    return {
        col: round(float(row[col]), 4)
        for col in summary_cols
        if col in df.columns
    }


# ─────────────────────────────────────────────────────────────────────────────
# FULL PER-STOCK PREDICTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def predict_stock(
    ticker: str,
    model: XGBClassifier,
    sentiment_model: Optional[SentimentAnalyzer] = None,
) -> PredictionResult:
    """
    End-to-end prediction for a single ticker using an already-trained
    global model.

    Steps:
      1. Fetch fresh OHLCV + features for this ticker
      2. Add FinBERT sentiment
      3. Create labels (needed for backtest reference)
      4. Run live signal from the final row
      5. Backtest on this stock's data
      6. Return consolidated PredictionResult
    """
    if sentiment_model is None:
        sentiment_model = SentimentAnalyzer()

    # Fresh data for this ticker
    df = fetch_stock_data(ticker)
    df = add_sentiment(df, ticker, sentiment_model)
    df = create_labels(df)
    df = filter_weak_signals(df)

    # Live signal
    signal, prob = live_prediction(model, df)

    # Backtest on this single stock
    df_clean     = df[df["signal"] != -1]
    X_stock      = df_clean[FEATURES]

    # Use FINAL_THRESHOLD to generate preds for backtest
    y_probs_all  = model.predict_proba(X_stock)[:, 1]
    y_preds_all  = (y_probs_all >= FINAL_THRESHOLD).astype(int)

    from sklearn.metrics import accuracy_score, f1_score
    y_true = df_clean["signal"].values
    acc    = float(accuracy_score(y_true, y_preds_all))
    f1     = float(f1_score(y_true, y_preds_all, average="weighted", zero_division=0))

    _, metrics = backtest(df_clean, y_preds_all)
    metrics.accuracy = acc
    metrics.f1       = f1

    indicators = get_indicator_summary(df)
    sentiment  = float(df["sentiment"].iloc[-1])

    return PredictionResult(
        ticker         = ticker,
        signal         = signal,
        probability    = prob,
        accuracy       = acc,
        f1             = f1,
        buy_precision  = metrics.buy_precision,
        sell_precision = metrics.sell_precision,
        total_return   = metrics.total_return,
        total_trades   = metrics.total_trades,
        sentiment      = sentiment,
        latest_indicators = indicators,
    )
