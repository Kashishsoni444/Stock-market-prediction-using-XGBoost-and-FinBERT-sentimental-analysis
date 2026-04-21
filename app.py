"""
app.py
─────────────────────────────────────────────────────────────────────────────
Streamlit frontend for the Nifty-50 Stock Predictor.

Optimisations
  • @st.cache_resource  → FinBERT loaded once per session
  • @st.cache_resource  → XGBoost model loaded once per session
  • @st.cache_data      → Per-stock prediction cached for 1 hour
  • Model is NEVER retrained on app launch; it loads from disk (.pkl)
  • All heavy work happens inside cached functions
  • Lazy FinBERT import (only when needed for live prediction)

Run with:
    streamlit run app.py
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Page config must be FIRST Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="Nifty-50 Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import STOCKS, STOCK_NAME_MAP, MODEL_PATH
from model_utils import load_model, full_train_pipeline
from data_loader import SentimentAnalyzer, build_dataset
from predictor import predict_stock, PredictionResult


# ─────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCE LOADERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading XGBoost model …")
def get_model():
    """
    Load the pre-trained model from disk.
    Cached for the entire session — never reloaded unless the app restarts.
    """
    try:
        model, params = load_model()
        return model
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner="Loading FinBERT sentiment model …")
def get_sentiment_model():
    """
    Lazy-load FinBERT once per session.
    Heavy (~500 MB); cached so it isn't re-instantiated on every rerun.
    """
    return SentimentAnalyzer()


@st.cache_data(ttl=3600, show_spinner="Generating prediction …")
def cached_predict(ticker: str) -> PredictionResult:
    """
    Cache per-stock predictions for 1 hour so repeated dropdown
    selections don't re-download data or re-run inference.
    """
    model           = get_model()
    sentiment_model = get_sentiment_model()
    return predict_stock(ticker, model, sentiment_model)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL COLOUR HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def signal_color(signal: str) -> str:
    return {"BUY": "#00C853", "SELL": "#D50000", "NO TRADE": "#FF6D00"}.get(signal, "#FFFFFF")


def signal_emoji(signal: str) -> str:
    return {"BUY": "🟢", "SELL": "🔴", "NO TRADE": "🟠"}.get(signal, "⚪")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> str:
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/NSE_logo.svg/220px-NSE_logo.svg.png",
            width=120,
        )
        st.title("⚙️ Controls")
        st.markdown("---")

        # Stock selection dropdown
        display_names = [
            f"{STOCK_NAME_MAP.get(s, s)}  ({s})" for s in STOCKS
        ]
        name_to_ticker = {
            f"{STOCK_NAME_MAP.get(s, s)}  ({s})": s for s in STOCKS
        }

        selected_display = st.selectbox(
            "Select Nifty 50 Stock",
            options=display_names,
            index=0,
            help="Choose a stock to analyse.",
        )
        ticker = name_to_ticker[selected_display]

        st.markdown("---")
        st.caption("Model: XGBoost + FinBERT + Optuna")
        st.caption("Data: yfinance (5-year OHLCV)")
        st.caption("Refresh cache: reload the page")

        st.markdown("---")
        if st.button("🔄 Retrain Model (all stocks)", use_container_width=True):
            _retrain_flow()

    return ticker


def _retrain_flow() -> None:
    """Full retrain triggered from the sidebar button."""
    with st.spinner("Building full Nifty-50 dataset … this may take 30–60 min"):
        df = build_dataset()

    with st.spinner("Training XGBoost model …"):
        model, metrics = full_train_pipeline(df)

    # Clear all cached predictions so new model is used
    cached_predict.clear()
    get_model.clear()

    st.success(
        f"✅ Model retrained!  "
        f"Accuracy: {metrics.accuracy:.2%}   "
        f"F1: {metrics.f1:.2%}   "
        f"Backtest return: {metrics.total_return:.2f}%"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def render_dashboard(result: PredictionResult) -> None:
    """Render all 10 required output sections."""

    ticker      = result.ticker
    stock_name  = STOCK_NAME_MAP.get(ticker, ticker)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        f"<h1 style='margin-bottom:0'>📈 {stock_name}</h1>"
        f"<p style='color:#888;font-size:1rem;margin-top:0'>{ticker}</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Row 1: Signal + Confidence ─────────────────────────────────────────────
    col_sig, col_prob, col_sent = st.columns(3)

    with col_sig:
        color = signal_color(result.signal)
        emoji = signal_emoji(result.signal)
        st.markdown(
            f"""
            <div style="
                background:{color}22;
                border:2px solid {color};
                border-radius:12px;
                padding:20px;
                text-align:center;">
              <div style="font-size:2.5rem">{emoji}</div>
              <div style="font-size:1.8rem;font-weight:700;color:{color}">
                {result.signal}
              </div>
              <div style="color:#aaa;font-size:0.85rem">Latest Signal</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_prob:
        st.metric(
            label="Confidence (P(BUY))",
            value=f"{result.probability:.1%}",
            delta=f"{'Above' if result.probability > 0.5 else 'Below'} 50% baseline",
        )
        st.progress(min(result.probability, 1.0))

    with col_sent:
        sent_label = (
            "Positive 😊" if result.sentiment > 0.1
            else ("Negative 😟" if result.sentiment < -0.1 else "Neutral 😐")
        )
        st.metric(
            label="Latest Sentiment Score",
            value=f"{result.sentiment:+.4f}",
            delta=sent_label,
        )

    st.markdown("---")

    # ── Row 2: Model Metrics ────────────────────────────────────────────────────
    st.subheader("📊 Model Performance Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Model Accuracy",    f"{result.accuracy:.1%}")
    c2.metric("Buy Precision",     f"{result.buy_precision:.1f}%")
    c3.metric("Sell Precision",    f"{result.sell_precision:.1f}%")
    c4.metric("Backtest Return",   f"{result.total_return:.2f}%",
              delta="vs buy-and-hold")
    c5.metric("Total Trades",      f"{result.total_trades:,}")

    st.markdown("---")

    # ── Row 3: Technical Indicators ─────────────────────────────────────────────
    st.subheader("🔬 Recent Technical Indicators (latest bar)")

    ind = result.latest_indicators
    cols = st.columns(5)
    indicator_items = [
        ("Close Price",       ind.get("Close",      "N/A"), "₹"),
        ("EMA (10)",          ind.get("EMA",        "N/A"), "₹"),
        ("SMA (10)",          ind.get("SMA",        "N/A"), "₹"),
        ("RSI (14)",          ind.get("RSI",        "N/A"), ""),
        ("MACD",              ind.get("MACD",       "N/A"), ""),
        ("MACD Signal",       ind.get("MACD_signal","N/A"), ""),
        ("BB Width",          ind.get("bb_width",   "N/A"), ""),
        ("Volatility (10d)",  ind.get("volatility", "N/A"), ""),
        ("Momentum (5d)",     ind.get("momentum",   "N/A"), ""),
        ("Sentiment",         ind.get("sentiment",  "N/A"), ""),
    ]

    for i, (label, val, unit) in enumerate(indicator_items):
        col_idx = i % 5
        with cols[col_idx]:
            display_val = f"{unit}{val:.4f}" if isinstance(val, float) else str(val)
            st.metric(label=label, value=display_val)

    st.markdown("---")

    # ── Row 4: RSI Gauge ────────────────────────────────────────────────────────
    rsi_val = ind.get("RSI", 50.0)
    if isinstance(rsi_val, float):
        st.subheader("📡 RSI Zone")
        rsi_col, _ = st.columns([1, 2])
        with rsi_col:
            zone = (
                "🔴 Overbought (>70)" if rsi_val > 70
                else ("🟢 Oversold (<30)" if rsi_val < 30
                      else "⚪ Neutral (30–70)")
            )
            st.metric("RSI (14)", f"{rsi_val:.1f}", delta=zone)
            st.progress(int(min(rsi_val, 100)) / 100)

    st.markdown("---")

    # ── Footer ──────────────────────────────────────────────────────────────────
    st.caption(
        "⚠️  This tool is for educational / research purposes only. "
        "It does not constitute financial advice. "
        "Past backtest performance does not guarantee future results."
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ticker = render_sidebar()

    model = get_model()

    # ── No model on disk yet ──────────────────────────────────────────────────
    if model is None:
        st.warning(
            "⚠️ No trained model found.  "
            "Click **Retrain Model** in the sidebar to train and save a model first.",
            icon="⚠️",
        )
        st.info(
            "First-time setup:\n"
            "1. Click **Retrain Model (all stocks)** in the left sidebar.\n"
            "2. Wait for training to complete (≈ 30–60 min depending on hardware).\n"
            "3. The model is saved to `models/xgb_model.pkl` and loaded automatically."
        )
        return

    # ── Prediction ──────────────────────────────────────────────────────────────
    try:
        with st.spinner(f"Fetching data and running model for **{ticker}** …"):
            result = cached_predict(ticker)
        render_dashboard(result)

    except Exception as exc:
        st.error(f"❌ Prediction failed for {ticker}: {exc}")
        logger.exception("Prediction error for %s", ticker)


if __name__ == "__main__":
    main()
