"""Metrics for finished ledgers: summary stats, bootstrap CIs,
classification quality, calibration. Reads written files and prices;
never reruns run_backtest.

Run: python -m src.metrics --split test   prints and writes the
results-table spine (n_trades, mean net + CI, AUC per cell).
"""
from __future__ import annotations

import argparse
from pathlib import Path

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
    values = np.asarray(values)
    point = float(stat_fn(values))

    rng = np.random.default_rng(seed)
    stats = []
    for draw in range(n_boot):
        sample = values[rng.integers(0, len(values), size=len(values))]
        stats.append(stat_fn(sample))

    low = float(np.percentile(stats, 2.5))
    high = float(np.percentile(stats, 97.5))
    return point, low, high


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


def report_numbers(tracks: list, models: list, split: str) -> pd.DataFrame:
    """The results-table spine: one row per (track, model) cell that
    has written files.

    Per cell: n_trades, mean net at the headline cost with its 95%
    bootstrap CI, and AUC of p_hat against the labels (NaN for e0 —
    all-NaN p_hat has no AUC). AUC uses every decided trigger in the
    split, entered or not: it scores the classifier, not the trades.
    """
    net_col = f"net_ret_{config.HEADLINE_COST_BPS}bps"

    rows = []
    for track in tracks:
        triggers_file = Path(f"data/datasets/triggers_{track}.parquet")
        if not triggers_file.exists():
            continue
        triggers = read_parquet(triggers_file)
        label_by_trigger_id = triggers.set_index("trigger_id")["label"]

        for model_name in models:
            trades_file = Path(f"results/trades_{track}_{model_name}_{split}.parquet")
            decisions_file = Path(f"results/decisions_{track}_{model_name}_{split}.parquet")
            if not trades_file.exists():
                continue
            trades = read_parquet(trades_file)
            decisions = read_parquet(decisions_file)

            mean_net, ci_low, ci_high = bootstrap_ci(trades[net_col], np.mean)

            auc = float("nan")
            has_predictions = decisions["p_hat"].notna().any()
            if has_predictions:
                y_true = label_by_trigger_id.loc[decisions["trigger_id"]].values
                y_predicted =decisions["p_hat"].values
                auc = roc_auc_score(y_true, y_predicted)

            rows.append({"track": track, "model": model_name, "split": split, "n_trades": len(trades), "mean_net": mean_net, "ci_low": ci_low, "ci_high": ci_high, "auc": auc})
    return pd.DataFrame(rows)


def main() -> None:
    """Write + print the report table for one split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", default="a,b")
    parser.add_argument("--models", default="e0,e1,e3")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    table = report_numbers(args.tracks.split(","), args.models.split(","), args.split)

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    table.to_csv(f"results/tables/report_numbers_{args.split}.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
