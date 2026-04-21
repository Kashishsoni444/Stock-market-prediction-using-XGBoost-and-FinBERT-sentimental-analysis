"""
train.py
─────────────────────────────────────────────────────────────────────────────
Standalone script to run the full training pipeline once and save the model.
Run this BEFORE launching the Streamlit app.

Usage:
    python train.py

The script will:
  1. Download 5 years of OHLCV for all Nifty-50 stocks in parallel
  2. Engineer all technical + sentiment features
  3. Run Optuna hyperparameter tuning (30 trials)
  4. Train the final XGBoost model with TimeSeriesSplit + class weights
  5. Evaluate on held-out test set
  6. Run backtesting
  7. Save model to  models/xgb_model.pkl
            params to  models/best_params.pkl

After this script completes, launch the app with:
    streamlit run app.py
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from data_loader  import build_dataset
from model_utils  import full_train_pipeline


def main() -> None:
    t0 = time.time()

    logger.info("=" * 60)
    logger.info("  Nifty-50 Stock Predictor – Training Pipeline")
    logger.info("=" * 60)

    logger.info("Step 1/2  Building dataset (parallel download + FinBERT) …")
    df = build_dataset()
    logger.info("Dataset ready: %d rows × %d columns", *df.shape)

    logger.info("Step 2/2  Training XGBoost (Optuna + TimeSeriesSplit) …")
    model, metrics = full_train_pipeline(df)

    logger.info("=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info("  Accuracy       : %.2f%%", metrics.accuracy  * 100)
    logger.info("  F1 (weighted)  : %.2f%%", metrics.f1        * 100)
    logger.info("  Buy Precision  : %.2f%%", metrics.buy_precision)
    logger.info("  Sell Precision : %.2f%%", metrics.sell_precision)
    logger.info("  Backtest Return: %.2f%%", metrics.total_return)
    logger.info("  Total Trades   : %d",     metrics.total_trades)
    logger.info("  Elapsed        : %.1f min", (time.time() - t0) / 60)
    logger.info("=" * 60)
    logger.info("Model saved.  Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
