# 📈 Nifty-50 Stock Predictor

XGBoost + FinBERT + Optuna — production-ready Streamlit deployment.

---

## Project Structure

```
stock_predictor/
├── config.py         ← All constants: stocks, API keys, features, thresholds
├── data_loader.py    ← yfinance download, feature engineering, FinBERT sentiment
├── model_utils.py    ← Optuna tuning, XGBoost training, evaluation, backtesting
├── predictor.py      ← Live BUY/SELL signal, probability, indicator summary
├── app.py            ← Streamlit frontend (fully cached)
├── train.py          ← One-time training script (run before the app)
├── requirements.txt
└── models/           ← Auto-created after training
    ├── xgb_model.pkl
    └── best_params.pkl
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your NewsAPI key

Open `config.py` and replace:
```python
NEWS_API_KEY: str = "YOUR_KEY_HERE"
```

### 3. Train the model (one-time, ~30–60 min)

```bash
python train.py
```

This downloads 5 years of OHLCV for all 50 stocks, runs Optuna tuning,
trains XGBoost, and saves `models/xgb_model.pkl`.

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Streamlit App Output

| # | Output |
|---|--------|
| 1 | Stock selection dropdown (all Nifty 50) |
| 2 | Latest BUY / SELL / NO TRADE signal |
| 3 | Confidence probability |
| 4 | Buy precision % |
| 5 | Sell precision % |
| 6 | Model accuracy % |
| 7 | Backtest return % |
| 8 | Total trades |
| 9 | Latest sentiment score |
| 10 | Recent technical indicator summary |

---

## Architecture

```
User selects stock
        │
        ▼
@st.cache_resource  ──► XGBoost model (.pkl)  loaded once
@st.cache_resource  ──► FinBERT model         loaded once
        │
        ▼
@st.cache_data(ttl=1h)
  fetch_stock_data()   ◄── yfinance (5yr OHLCV)
  add_sentiment()      ◄── NewsAPI + FinBERT (last 30 days)
  create_labels()
  live_prediction()    ◄── threshold: BUY >0.62 | SELL <0.38
  backtest()
        │
        ▼
      Dashboard
```

---

## Retraining

Click **Retrain Model** in the sidebar to trigger a full retrain. This:
1. Downloads fresh data for all 50 stocks
2. Re-runs Optuna (30 trials)
3. Re-trains XGBoost
4. Saves new `.pkl` files
5. Clears all cached predictions

---

## Key Design Decisions

- **No retraining on app launch** — model loads from `.pkl` in milliseconds.
- **`@st.cache_data(ttl=3600)`** — per-stock predictions cached for 1 hour.
- **`@st.cache_resource`** — FinBERT (~500 MB) loaded once, reused across all predictions.
- **Parallel downloads** — all 50 stocks fetched concurrently via `ThreadPoolExecutor`.
- **Batched FinBERT inference** — all headlines for a date processed in one forward pass.
- **Disk-backed news cache** — `news_cache.json` persists between app restarts.
- **Single `predict_proba` call** — original notebook called it twice; now called once.
- **ML logic unchanged** — thresholds (0.62 / 0.38 / 0.54), labeling (0.8 × vol_20), SMOTE, class weights, Optuna search space all identical to original notebook.
