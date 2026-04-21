"""
model_utils.py
─────────────────────────────────────────────────────────────────────────────
Handles:
  • Data preparation & SMOTE balancing
  • Optuna Bayesian hyperparameter search
  • XGBoost training with TimeSeriesSplit + class weights
  • Model evaluation (accuracy, F1, precision, RMSE, MAE, R²)
  • Backtesting (buy/sell precision, total return, win rate)
  • Model serialisation / deserialisation via joblib

ALL ML logic, thresholds, and computation formulas are IDENTICAL
to the original notebook cells 10–15.

Optimisations vs original notebook
  • predict_proba called ONCE per evaluation pass (not twice)
  • Backtest avoids redundant .copy() inside the loop
  • Model saved to disk with joblib after each successful training run
  • Optuna verbosity set to WARNING to reduce console noise
  • Class-weight computation vectorised (single call to compute_class_weight)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from config import (
    FEATURES, FINAL_THRESHOLD, OPTUNA_TRIALS, RANDOM_STATE,
    TSCV_SPLITS, MODEL_DIR, MODEL_PATH, PARAMS_PATH,
)

logger = logging.getLogger(__name__)

# Silence Optuna trial-level logs; keep WARNING+ only
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# RETURN TYPE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalMetrics:
    accuracy:       float
    f1:             float
    buy_precision:  float
    sell_precision: float
    total_return:   float
    avg_return:     float
    win_rate:       float
    total_trades:   int
    report:         str


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION  (identical to notebook cells 10-11)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Remove HOLD rows, select feature columns, time-ordered 80/20 split.
    Identical to notebook cell 10.
    """
    df = df[df["signal"] != -1]
    X  = df[FEATURES]
    y  = df["signal"]
    return train_test_split(X, y, test_size=0.2, shuffle=False)


def balance_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """SMOTE oversampling — identical to notebook cell 11."""
    sm         = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER TUNING  (identical to notebook cell 12)
# ─────────────────────────────────────────────────────────────────────────────

def tune_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = OPTUNA_TRIALS,
) -> dict[str, Any]:
    """
    Bayesian optimisation over XGBoost hyper-params using TimeSeriesSplit.
    Objective and parameter space are IDENTICAL to notebook cell 12.
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 800),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0, 5),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0, 2),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0, 2),
            "objective":         "binary:logistic",
            "eval_metric":       "logloss",
            "random_state":      RANDOM_STATE,
        }

        model = XGBClassifier(**params)
        tscv  = TimeSeriesSplit(n_splits=TSCV_SPLITS)
        scores: list[float] = []

        for train_idx, val_idx in tscv.split(X_train):
            X_tr  = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_tr  = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]

            model.fit(X_tr, y_tr)
            preds  = model.predict(X_val)
            scores.append(balanced_accuracy_score(y_val, preds))

        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info("Best Optuna params: %s", study.best_params)
    return study.best_params


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING  (identical to notebook cell 13)
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict[str, Any],
) -> XGBClassifier:
    """
    Weighted XGBoost with TimeSeriesSplit CV.
    Class-weight computation and BUY-weight override are IDENTICAL
    to notebook cell 13.
    """
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weights: dict[int, float] = dict(zip(classes.tolist(), weights.tolist()))
    class_weights[1] = 1.0          # stronger BUY importance (identical to notebook)

    logger.info("Adjusted class weights: %s", class_weights)

    sample_weights = np.array([class_weights[int(i)] for i in y])

    model = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )

    tscv   = TimeSeriesSplit(n_splits=TSCV_SPLITS)
    scores: list[float] = []

    for train_idx, val_idx in tscv.split(X):
        X_tr  = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_tr  = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        w_tr  = sample_weights[train_idx]

        model.fit(X_tr, y_tr, sample_weight=w_tr)

        # Identical threshold to notebook cell 13
        probs = model.predict_proba(X_val)[:, 1]
        preds = (probs > 0.62).astype(int)
        scores.append(f1_score(y_val, preds))

    logger.info("CV F1 Score: %.4f", float(np.mean(scores)))

    # Final fit on full training set
    model.fit(X, y, sample_weight=sample_weights)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION  (identical to notebook cell 14 + cell 17 threshold logic)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Returns (y_probs, y_pred, classification_report_str).
    predict_proba called ONCE (optimisation: original called it twice).
    Threshold logic identical to notebook cell 17.
    """
    y_probs: np.ndarray = model.predict_proba(X_test)[:, 1]
    y_pred              = (y_probs >= FINAL_THRESHOLD).astype(int)

    report = classification_report(y_test, y_pred)
    logger.info("\n%s", report)
    logger.info("Accuracy: %.4f", accuracy_score(y_test, y_pred))
    logger.info("F1: %.4f",       f1_score(y_test, y_pred, average="weighted"))

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    logger.info("RMSE: %.4f  MAE: %.4f  R2: %.4f", rmse, mae, r2)

    return y_probs, y_pred, report


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTING  (identical to notebook cell 15)
# ─────────────────────────────────────────────────────────────────────────────

def backtest(
    df: pd.DataFrame,
    y_pred: np.ndarray,
) -> tuple[pd.DataFrame, EvalMetrics]:
    """
    Align predictions with the last N rows of df and compute strategy returns.
    Logic is IDENTICAL to notebook cell 15.

    Returns the annotated DataFrame and an EvalMetrics dataclass.
    """
    df_bt = df.iloc[-len(y_pred):].copy()
    df_bt["pred"] = y_pred

    df_bt["position"]        = np.where(df_bt["pred"] == 1, 1, -1)
    df_bt["strategy_return"] = df_bt["position"] * df_bt["future_return"]
    df_bt["cumulative_return"] = (1 + df_bt["strategy_return"]).cumprod()

    total_return = float((df_bt["cumulative_return"].iloc[-1] - 1) * 100)
    avg_return   = float(df_bt["strategy_return"].mean() * 100)
    win_rate     = float((df_bt["strategy_return"] > 0).mean() * 100)

    buy_rows  = df_bt[df_bt["pred"] == 1]
    sell_rows = df_bt[df_bt["pred"] == 0]

    buy_precision  = float((buy_rows["future_return"]  > 0).mean() * 100) if len(buy_rows)  else 0.0
    sell_precision = float((sell_rows["future_return"] < 0).mean() * 100) if len(sell_rows) else 0.0

    logger.info("Backtest  total_return=%.2f%%  win_rate=%.2f%%  "
                "buy_precision=%.2f%%  sell_precision=%.2f%%",
                total_return, win_rate, buy_precision, sell_precision)

    # Build EvalMetrics
    y_true = df_bt["pred"].values   # proxy; full report in evaluate()
    metrics = EvalMetrics(
        accuracy       = float(accuracy_score(df_bt["pred"], df_bt["pred"])),   # placeholder
        f1             = 0.0,   # filled by caller
        buy_precision  = buy_precision,
        sell_precision = sell_precision,
        total_return   = total_return,
        avg_return     = avg_return,
        win_rate       = win_rate,
        total_trades   = len(df_bt),
        report         = "",
    )

    return df_bt, metrics


# ─────────────────────────────────────────────────────────────────────────────
# FULL TRAINING PIPELINE  (convenience wrapper)
# ─────────────────────────────────────────────────────────────────────────────

def full_train_pipeline(
    df: pd.DataFrame,
) -> tuple[XGBClassifier, EvalMetrics]:
    """
    Runs the complete training pipeline:
      prepare → balance → tune → train → evaluate → backtest → save

    Returns the trained model and consolidated EvalMetrics.
    """
    X_train, X_test, y_train, y_test = prepare_data(df)
    X_train_bal, y_train_bal         = balance_data(X_train, y_train)

    best_params = tune_xgb(X_train_bal, y_train_bal)
    model       = train_model(X_train_bal, y_train_bal, best_params)

    y_probs, y_pred, report = evaluate(model, X_test, y_test)

    acc = float(accuracy_score(y_test, y_pred))
    f1  = float(f1_score(y_test, y_pred, average="weighted"))

    df_full = df[df["signal"] != -1]
    _, metrics = backtest(df_full, y_pred)

    metrics.accuracy = acc
    metrics.f1       = f1
    metrics.report   = report

    save_model(model, best_params)
    return model, metrics


# ─────────────────────────────────────────────────────────────────────────────
# MODEL PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def save_model(
    model: XGBClassifier,
    params: dict[str, Any],
) -> None:
    """Persist model and params to disk using joblib."""
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(params, PARAMS_PATH)
    logger.info("Model saved to %s", MODEL_PATH)


def load_model() -> tuple[XGBClassifier, dict[str, Any]]:
    """
    Load a previously saved model + params.
    Raises FileNotFoundError if no saved model exists.
    """
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"No trained model found at '{MODEL_PATH}'. "
            "Run the training pipeline first."
        )
    model  = joblib.load(MODEL_PATH)
    params = joblib.load(PARAMS_PATH) if Path(PARAMS_PATH).exists() else {}
    logger.info("Model loaded from %s", MODEL_PATH)
    return model, params
