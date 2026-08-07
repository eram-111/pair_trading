"""Metrics for finished ledgers: summary stats, bootstrap CIs,
classification quality, calibration. Reads written files and prices;
never reruns run_backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config
from src.contracts import read_parquet


def trade_metrics(trades: pd.DataFrame, cost_bps: int = config.HEADLINE_COST_BPS) -> dict:
    """One ledger -> {"n_trades", "mean_gross", "mean_net", "hit_rate",
    "total_net"} at the given cost. hit_rate = share of trades with
    positive net return."""
    raise NotImplementedError


def bootstrap_ci(values, stat_fn, n_boot: int = 1000, seed: int = config.SEED) -> tuple[float, float, float]:
    """(point, low, high) for any statistic: point = stat_fn(values);
    then n_boot seeded resamples of values with replacement, stat_fn on
    each, CI = the 2.5 and 97.5 percentiles. One function for every CI
    in the project (mean return, AUC, Sharpe, AUC differences).

    Independence caveat: trades overlap in time and cluster by pair and
    regime, so i.i.d. resampling understates true uncertainty; a block
    bootstrap would be more faithful but is out of scope — CIs are
    therefore optimistic lower bounds on width.
    """
    raise NotImplementedError


def sharpe_from_daily(daily_returns: pd.Series) -> float:
    """Annualized Sharpe: mean / std * sqrt(252) of a daily return
    series (engine.daily_strategy_returns output) that the caller has
    ALREADY SLICED to the split's date window — the engine returns the
    full price calendar, and structural zeros outside the split would
    wrongly shrink Sharpe. Inside the window, no-trade days are 0.0 and
    are included."""
    raise NotImplementedError


def max_drawdown(daily_returns: pd.Series) -> float:
    """Largest peak-to-trough drop of the cumulative return path of the
    (already split-sliced) daily series. Returned as a negative number."""
    raise NotImplementedError


def classification_metrics(y_true, p_hat, tau: float) -> dict:
    """{"auc", "precision_at_tau", "recall_at_tau", "brier", "n"} for
    one model's probabilities against the labels; precision/recall use
    enter = (p_hat >= tau). Only for models with real p_hat — e0's NaN
    p_hat has no classification metrics."""
    raise NotImplementedError


def calibration_table(decisions: pd.DataFrame, triggers: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table: join on trigger_id, split rows into n_bins
    quantile bins of p_hat; one row per bin with mean p_hat, observed
    label rate, and count. (The calibration figure just plots this.)"""
    raise NotImplementedError


def strategy_report(track: str, model_name: str, split: str) -> dict:
    """All of the above for one grid cell, with CIs via bootstrap_ci.

    Reads the cell's written decisions/trades files, triggers_{track}
    (labels), taus.json (the cell's tau, for classification_metrics),
    and data/raw/prices.parquet — needed to rebuild the daily return
    series through engine.daily_strategy_returns for Sharpe and max
    drawdown (sliced to the split's window before use).
    """
    raise NotImplementedError
