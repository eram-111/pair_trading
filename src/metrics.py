"""Metrics for finished ledgers; reads written files, never reruns run_backtest.
Run: python -m src.metrics --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw figures to files; must be set before pyplot is imported
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config
from src.contracts import read_parquet, validate_artifact
from src.engine import daily_strategy_returns


# ------------------------- basic statistics -------------------------------

def trade_metrics(trades: pd.DataFrame, cost_bps: int = config.HEADLINE_COST_BPS) -> dict:
    """One ledger -> {"n_trades", "mean_gross", "mean_net", "hit_rate",
    "total_net"} at the given cost."""
    net = trades[f"net_ret_{cost_bps}bps"]
    return {"n_trades": len(trades),
            "mean_gross": float(trades["gross_ret"].mean()),
            "mean_net": float(net.mean()),
            "hit_rate": float((net > 0).mean()),
            "total_net": float(net.sum())}


def bootstrap_mean_ci(values, n_boot: int = 1000, seed: int = config.SEED) -> tuple[float, float, float]:
    """Returns (mean, low, high): the mean of values with a seeded 95%
    bootstrap CI. Trades are not independent, so CIs are optimistic."""
    values = np.asarray(values)
    mean_value = float(values.mean())

    rng = np.random.default_rng(seed)
    sample_means = []
    for draw in range(n_boot):
        # one resample: len(values) rows drawn with replacement
        sample = values[rng.integers(0, len(values), size=len(values))]
        sample_means.append(sample.mean())

    ci_low = float(np.percentile(sample_means, 2.5))
    ci_high = float(np.percentile(sample_means, 97.5))
    return mean_value, ci_low, ci_high


def bootstrap_auc_ci(y_true, p_hat, n_boot: int = 1000, seed: int = config.SEED) -> tuple[float, float, float]:
    """Returns (auc, low, high): the AUC of p_hat against y_true with a
    seeded 95% bootstrap CI. Each resample keeps label/prediction PAIRS
    together by drawing row positions."""
    y_true = np.asarray(y_true)
    p_hat = np.asarray(p_hat)
    auc_value = float(roc_auc_score(y_true, p_hat))

    rng = np.random.default_rng(seed)
    sample_aucs = []
    for draw in range(n_boot):
        sample = rng.integers(0, len(y_true), size=len(y_true))
        sample_aucs.append(roc_auc_score(y_true[sample], p_hat[sample]))

    ci_low = float(np.percentile(sample_aucs, 2.5))
    ci_high = float(np.percentile(sample_aucs, 97.5))
    return auc_value, ci_low, ci_high


def sharpe_from_daily(daily_returns: pd.Series) -> float:
    """Annualized Sharpe (mean/std * sqrt(252)); caller must slice the
    series to the split's date window first."""
    standard_deviation = daily_returns.std()
    if standard_deviation == 0:
        return float("nan")
    return float(daily_returns.mean() / standard_deviation * (252 ** 0.5))


def max_drawdown(daily_returns: pd.Series) -> float:
    """Largest peak-to-trough drop of the (already split-sliced) daily
    series; returned as a negative number."""
    cum_returns = daily_returns.cumsum()
    running_peak = cum_returns.cummax()
    drawdowns = cum_returns - running_peak
    return float(drawdowns.min())


def classification_metrics(y_true, p_hat, tau: float) -> dict:
    """Returns {"auc", "precision_at_tau", "recall_at_tau", "brier", "n"};
    enter = (p_hat >= tau). Not for e0's all-NaN p_hat."""
    y_true = np.asarray(y_true)
    p_hat = np.asarray(p_hat)

    entered_mask = p_hat >= tau

    # precision: of the triggers the model entered, how many reverted
    precision = float("nan")
    if entered_mask.sum() > 0:
        precision = float(y_true[entered_mask].mean())

    # recall: of all reverting triggers, how many the model entered
    recall = float("nan")
    if y_true.sum() > 0:
        recall = float(y_true[entered_mask].sum() / y_true.sum())

    brier = float(((p_hat - y_true) ** 2).mean())
    auc = float(roc_auc_score(y_true, p_hat))
    return {"auc": auc, "precision_at_tau": precision, "recall_at_tau": recall,
            "brier": brier, "n": len(y_true)}


# ------------------------- calibration ------------------------------------

def calibration_table(decisions: pd.DataFrame, triggers: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table: one row per p_hat quantile bin with mean p_hat,
    observed label rate, and count."""
    merged = decisions.merge(triggers[["trigger_id", "label"]], on="trigger_id")
    merged = merged[merged["p_hat"].notna()]

    # quantile bins: equal COUNTS per bin; duplicates="drop" merges bins
    # when many p_hat values tie (e2's clustered probabilities)
    merged["bin"] = pd.qcut(merged["p_hat"], n_bins, duplicates="drop")

    rows = []
    for bin_interval, bin_rows in merged.groupby("bin", observed=True):
        rows.append({"mean_p_hat": float(bin_rows["p_hat"].mean()),
                     "observed_rate": float(bin_rows["label"].mean()),
                     "n": len(bin_rows)})
    return pd.DataFrame(rows)


def calibration_figure(track: str, models: list, split: str) -> None:
    """Reliability diagram; saves results/figures/calibration_{track}_{split}.png."""
    triggers = read_parquet(f"data/datasets/triggers_{track}.parquet")

    fig, calibration_axis = plt.subplots(figsize=(6, 6))
    calibration_axis.plot([0, 1], [0, 1], color="#b0b0b0", linewidth=1, linestyle="--", label="perfect calibration")

    for model_name in models:
        decisions_file = Path(f"results/decisions_{track}_{model_name}_{split}.parquet")
        if not decisions_file.exists():
            continue
        decisions = read_parquet(decisions_file)
        if not decisions["p_hat"].notna().any():
            continue  # e0 predicts nothing

        table = calibration_table(decisions, triggers)
        calibration_axis.plot(table["mean_p_hat"], table["observed_rate"], color=config.MODEL_COLORS[model_name],
                  linewidth=2, marker="o", markersize=5, label=model_name)

    calibration_axis.set_xlabel("mean predicted probability (per bin)")
    calibration_axis.set_ylabel("observed reversion rate (per bin)")
    calibration_axis.set_title(f"Calibration, track {track} ({split} split)")
    calibration_axis.set_xlim(0.0, 1.0)
    calibration_axis.set_ylim(0.0, 1.0)
    calibration_axis.legend(fontsize=9)
    calibration_axis.grid(True, color="#eeeeee", linewidth=0.8)
    calibration_axis.spines["top"].set_visible(False)
    calibration_axis.spines["right"].set_visible(False)

    Path("results/figures").mkdir(parents=True, exist_ok=True)
    out_path = f"results/figures/calibration_{track}_{split}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out_path}")


# ------------------------- turnover-matched control ------------------------

def matched_control_decisions(model_decisions: pd.DataFrame, triggers: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Random strategy matching the model's per-quarter entry counts.
    Returns a decisions table (p_hat=NaN). rng seeded with config.SEED + seed.
    """
    rng = np.random.default_rng(config.SEED + seed)

    merged = model_decisions.merge(triggers[["trigger_id", "trigger_date"]],on="trigger_id")
    merged["quarter"] = merged["trigger_date"].dt.to_period("Q")  # e.g. 2020-03-15 -> 2020Q1

    picked = []

    for i, row in merged.groupby("quarter"):
        entered_trades = row["enter"]
        n_entered = entered_trades.sum()
        if n_entered == 0:
            continue
        picks = rng.choice(row["trigger_id"], size=n_entered, replace=False)
        picked.extend(picks)

    decisions = model_decisions[["trigger_id"]].copy()
    decisions["enter"] = decisions["trigger_id"].isin(picked)
    decisions["p_hat"] = float("nan")
    validate_artifact(decisions, "decisions")
    return decisions


def run_control(track: str, model_name: str, split: str, n_seeds: int = 1000) -> dict:
    """Turnover-matched control for one cell, scored via the e0 ledger (no engine reruns).
    Writes and returns results/control_{track}_{model}_{split}.json.
    """
    net_col = f"net_ret_{config.HEADLINE_COST_BPS}bps"

    triggers = read_parquet(f"data/datasets/triggers_{track}.parquet")
    model_decisions_table = read_parquet(f"results/decisions_{track}_{model_name}_{split}.parquet")
    e0_trades = read_parquet(f"results/trades_{track}_e0_{split}.parquet")

    net_by_trigger_id = e0_trades.set_index("trigger_id")[net_col]

    all_ids_known = model_decisions_table["trigger_id"].isin(net_by_trigger_id.index).all()

    assert all_ids_known, "e0 ledger is missing triggers: run the e0 cell for this split first"

    entered = model_decisions_table[model_decisions_table["enter"]]
    if len(entered) == 0:
        print(f"{track} {model_name}: entered no trades (degenerate cell) — control skipped")
        return {}

    model_returns = net_by_trigger_id.loc[entered["trigger_id"]]
    model_mean_net = float(model_returns.mean())

    control_means = []

    for seed in range(n_seeds):
        control = matched_control_decisions(model_decisions_table, triggers, seed)
        picks = control[control["enter"]]
        control_returns = net_by_trigger_id.loc[picks["trigger_id"]]
        control_mean = float(control_returns.mean())
        control_means.append(control_mean)

    control_means = np.array(control_means)

    beaten = int((control_means < model_mean_net).sum())
    percentile = 100.0 * beaten / n_seeds

    result = {"model_mean_net": model_mean_net, "control_mean": float(control_means.mean()),"control_std": float(control_means.std()),"percentile": percentile, "n_seeds": n_seeds}

    out_file = Path(f"results/control_{track}_{model_name}_{split}.json")
    out_file.write_text(json.dumps(result, indent=2))
    print(f"{track} {model_name}: model={model_mean_net:.5f}, control={result['control_mean']:.5f}, percentile={percentile:.0f}"
    )
    return result


# ------------------------- cost sweep --------------------------------------

def breakeven_bps(trades: pd.DataFrame) -> float:
    """Cost (bps per leg) where mean net return is zero: mean gross / 0.0004, exact."""
    mean = trades["gross_ret"].mean()
    return float(mean / 0.0004)


def make_breakevens_table(tracks: list, models: list, split: str) -> pd.DataFrame:
    """One row per cell that traded: n_trades, mean gross, breakeven
    cost. Written to results/tables/breakevens_{split}.csv."""
    rows = []
    for track in tracks:
        for model_name in models:
            ledger_file = Path(f"results/trades_{track}_{model_name}_{split}.parquet")
            if not ledger_file.exists():
                continue
            trades = read_parquet(ledger_file)
            if len(trades) == 0:
                continue
            rows.append({"track": track, "model": model_name, "split": split,
                         "n_trades": len(trades),
                         "mean_gross": float(trades["gross_ret"].mean()),
                         "breakeven_bps": breakeven_bps(trades)})

    table = pd.DataFrame(rows)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    table.to_csv(f"results/tables/breakevens_{split}.csv", index=False)
    return table


def cost_sweep_figure(tracks: list, models: list, split: str) -> None:
    """Mean net return per trade vs cost, one line per strategy, from precomputed
    net_ret_{c}bps columns. Saves results/figures/cost_sweep_{split}.png.
    """
    fig, sweep_axis = plt.subplots(figsize=(8, 5))

    for track in tracks:
        for model_name in models:
            ledger_file = Path(f"results/trades_{track}_{model_name}_{split}.parquet")
            if not ledger_file.exists():
                continue
            trades = read_parquet(ledger_file)
            if len(trades) == 0:
                continue

            mean_net_per_cost = []
            for c in config.COST_GRID_BPS:
                mean_net_per_cost.append(trades[f"net_ret_{c}bps"].mean())

            label = f"{model_name} (breakeven {breakeven_bps(trades):.1f} bps)"
            sweep_axis.plot(config.COST_GRID_BPS, mean_net_per_cost, color=config.MODEL_COLORS[model_name],
                      linewidth=2, marker="o", markersize=4, label=label)

    sweep_axis.axhline(0.0, color="#b0b0b0", linewidth=1)
    sweep_axis.axvline(config.HEADLINE_COST_BPS, color="#b0b0b0", linewidth=1, linestyle="--")

    sweep_axis.set_xlabel("cost (bps per leg per transaction)")
    sweep_axis.set_ylabel("mean net return per trade")
    sweep_axis.set_title(f"Net return vs trading cost ({split} split)")
    sweep_axis.legend(fontsize=9)
    sweep_axis.grid(True, color="#eeeeee", linewidth=0.8)
    sweep_axis.spines["top"].set_visible(False)
    sweep_axis.spines["right"].set_visible(False)

    Path("results/figures").mkdir(parents=True, exist_ok=True)
    out_path = f"results/figures/cost_sweep_{split}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out_path}")


# ------------------------- the report battery ------------------------------

def decision_accuracy(decisions: pd.DataFrame, label_by_trigger_id: pd.Series) -> float:
    """Share of triggers the strategy called right: it traded and the gap
    closed, or it skipped and the gap did not. Works for every model,
    including e0 (which always trades)."""
    labels = label_by_trigger_id.loc[decisions["trigger_id"]].values
    gap_closed = labels == 1
    correct_mask = decisions["enter"].values == gap_closed
    return float(correct_mask.mean())


def _money_columns(trades: pd.DataFrame) -> dict:
    """The trading half of one report row: n_trades, hit_rate, mean gross
    (the no-cost return) and mean net + CI. No trades gets NaN money."""
    if len(trades) == 0:
        return {"n_trades": 0, "hit_rate": float("nan"), "mean_gross": float("nan"),
                "mean_net": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    net = trades[f"net_ret_{config.HEADLINE_COST_BPS}bps"]
    mean_net, ci_low, ci_high = bootstrap_mean_ci(net)
    return {"n_trades": len(trades), "hit_rate": float((net > 0).mean()),
            "mean_gross": float(trades["gross_ret"].mean()),
            "mean_net": mean_net, "ci_low": ci_low, "ci_high": ci_high}


def _classification_columns(decisions: pd.DataFrame, label_by_trigger_id: pd.Series, tau) -> dict:
    """The prediction half of one report row: AUC + CI and, when the cell
    has a tau, precision/recall/Brier. All NaN for e0 (no predictions)."""
    result = {"auc": float("nan"), "auc_low": float("nan"), "auc_high": float("nan"),
               "precision_at_tau": float("nan"), "recall_at_tau": float("nan"), "brier": float("nan")}

    if not decisions["p_hat"].notna().any():
        return result

    y_true = label_by_trigger_id.loc[decisions["trigger_id"]].values
    y_predicted = decisions["p_hat"].values

    result["auc"], result["auc_low"], result["auc_high"] = bootstrap_auc_ci(y_true, y_predicted)

    if tau is not None:
        cls = classification_metrics(y_true, y_predicted, float(tau))
        result["precision_at_tau"] = cls["precision_at_tau"]
        result["recall_at_tau"] = cls["recall_at_tau"]
        result["brier"] = cls["brier"]
    return result


def report_numbers(tracks: list, models: list, split: str) -> pd.DataFrame:
    """Results-table spine: one row per written (track, model) cell —
    the money columns plus the classification columns."""
    taus = {}
    taus_file = Path("results/frozen/taus.json")
    if taus_file.exists():
        taus = json.loads(taus_file.read_text())

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

            row = {"track": track, "model": model_name, "split": split,
                   "accuracy": decision_accuracy(decisions, label_by_trigger_id)}
            row.update(_money_columns(trades))
            row.update(_classification_columns(decisions, label_by_trigger_id,
                                               taus.get(f"{model_name}_{track}")))
            rows.append(row)
    return pd.DataFrame(rows)


def equity_curve_figure(tracks: list, models: list, split: str) -> None:
    """Cumulative net return per strategy at the headline cost, on the
    split's common date window; legend carries Sharpe and max drawdown.
    Saves results/figures/equity_curve_{split}.png."""
    prices = read_parquet("data/raw/prices.parquet")

    # one full-calendar daily series per cell that traded
    series_by_label = {}
    colors_by_label = {}
    window_start = None
    window_end = None
    for track in tracks:
        triggers_file = Path(f"data/datasets/triggers_{track}.parquet")
        if not triggers_file.exists():
            continue
        triggers = read_parquet(triggers_file)

        for model_name in models:
            trades_file = Path(f"results/trades_{track}_{model_name}_{split}.parquet")
            if not trades_file.exists():
                continue
            trades = read_parquet(trades_file)
            if len(trades) == 0:
                continue

            daily_returns = daily_strategy_returns(trades, prices, triggers)
            label = model_name
            if len(tracks) > 1:
                label = f"{track}-{model_name}"
            series_by_label[label] = daily_returns
            colors_by_label[label] = config.MODEL_COLORS[model_name]

            # the common window: first entry to last exit over all cells
            if window_start is None or trades["entry_date"].min() < window_start:
                window_start = trades["entry_date"].min()
            if window_end is None or trades["exit_date"].max() > window_end:
                window_end = trades["exit_date"].max()

    fig, curve_axis = plt.subplots(figsize=(9, 5))
    for label in series_by_label:
        daily_returns = series_by_label[label].loc[window_start:window_end]
        sharpe = sharpe_from_daily(daily_returns)
        drawdown = max_drawdown(daily_returns)
        cum_returns = daily_returns.cumsum()
        curve_axis.plot(cum_returns.index, cum_returns.values, color=colors_by_label[label], linewidth=2,
                  label=f"{label} (Sharpe {sharpe:.2f}, maxDD {drawdown:.3f})")

    curve_axis.axhline(0.0, color="#b0b0b0", linewidth=1)
    curve_axis.set_xlabel("date")
    curve_axis.set_ylabel(f"cumulative net return @ {config.HEADLINE_COST_BPS} bps")
    curve_axis.set_title(f"Equity curves ({split} split)")
    curve_axis.legend(fontsize=9)
    curve_axis.grid(True, color="#eeeeee", linewidth=0.8)
    curve_axis.spines["top"].set_visible(False)
    curve_axis.spines["right"].set_visible(False)

    Path("results/figures").mkdir(parents=True, exist_ok=True)
    out_path = f"results/figures/equity_curve_{split}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out_path}")


def main() -> None:
    """Every analysis for one split: report table, controls, breakevens,
    cost sweep, calibration figures. Reads written results only."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", default="a,b")
    parser.add_argument("--models", default="e0,e1,e2,e3")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    tracks = args.tracks.split(",")
    models = args.models.split(",")

    table = report_numbers(tracks, models, args.split)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    table.to_csv(f"results/tables/report_numbers_{args.split}.csv", index=False)
    print(table.to_string(index=False))

    # turnover controls for every learned model with written files
    for track in tracks:
        for model_name in models:
            if model_name == "e0":
                continue
            decisions_file = Path(f"results/decisions_{track}_{model_name}_{args.split}.parquet")
            e0_ledger_file = Path(f"results/trades_{track}_e0_{args.split}.parquet")
            if decisions_file.exists() and e0_ledger_file.exists():
                run_control(track, model_name, args.split)

    breakevens = make_breakevens_table(tracks, models, args.split)
    print(breakevens.to_string(index=False))

    cost_sweep_figure(tracks, models, args.split)
    equity_curve_figure(tracks, models, args.split)
    for track in tracks:
        if Path(f"data/datasets/triggers_{track}.parquet").exists():
            calibration_figure(track, models, args.split)


if __name__ == "__main__":
    main()
