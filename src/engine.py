"""Backtest engine: turns entry decisions into a ledger of completed trades.
Enter at t+1. Costs are c bps per leg per transaction = 4c per round trip."""
from __future__ import annotations

import pandas as pd

from src import config
from src.contracts import validate_artifact

def _simple_ret(prices: pd.DataFrame)-> pd.DataFrame:
    """Table of simple returns."""
    prices_shifted_forward = prices.shift(1)
    return (prices - prices_shifted_forward ) / prices_shifted_forward

def _split_stocks(pair_id: str, z_trigger: float) -> tuple[str, str]:
    """Return (long_stock, short_stock). z > 0 shorts stock_a, longs stock_b."""
    stock_a, stock_b = pair_id.split("__")
    if z_trigger > 0:
        return stock_b, stock_a
    return stock_a, stock_b

def run_backtest(zscores: pd.DataFrame, prices: pd.DataFrame, triggers: pd.DataFrame, decisions: pd.DataFrame,  cost_grid_bps: tuple = config.COST_GRID_BPS) -> pd.DataFrame:
    """Replay every accepted trigger. return the trades ledger ("trades" schema),
    with one net_ret_{c}bps = gross_ret - 4*c*0.0001 column per cost-grid value."""
    returns = _simple_ret(prices)

    decision_dict = {}
    for index, row in decisions.iterrows():
        decision_dict[row["trigger_id"]] = row["enter"]

    last_day = len(zscores.index) - 1
    ledger_rows = []
    dropped = 0

    for index, trig in triggers.iterrows():
        trigger_id = trig["trigger_id"]
        # dont enter trade
        if trigger_id not in decision_dict:
            continue
        if not decision_dict[trigger_id]:
            continue

        pair_id = trig["pair_id"]
        z_trigger = trig["z_trigger"]
        trigger_date = trig["trigger_date"]

        long_stock, short_stock = _split_stocks(pair_id, z_trigger)

        z_pair = zscores[pair_id]

        # Entry: one day after the trigger day (t+1).
        trigger_day = zscores.index.get_loc(trigger_date)
        entry_day = trigger_day + 1
        if entry_day > last_day:
            dropped = dropped + 1
            continue

        exit_day = entry_day + config.MAX_HOLD_DAYS
        if exit_day > last_day:
            exit_day = last_day
        exit_reason = "timeout"

        gross_ret = 0.0
        curr_day = entry_day + 1
        while curr_day <= exit_day:
            date = zscores.index[curr_day]
            ret = returns.loc[date, long_stock] - returns.loc[date, short_stock]
            gross_ret += ret

            if abs(z_pair[date]) < config.EXIT_Z:
                exit_day = curr_day
                exit_reason = "reverted"
                break
            curr_day = curr_day + 1

        ledger_rows.append({
            "trigger_id": trigger_id,
            "pair_id": pair_id,
            "entry_date": zscores.index[entry_day],
            "exit_date": zscores.index[exit_day],
            "exit_reason": exit_reason,
            "days_held": exit_day - entry_day,
            "gross_ret": gross_ret,
        })

    if dropped > 0:
        print(f"run_backtest: dropped {dropped} trigger(s) with no next day to enter on")

    if len(ledger_rows):
        trades = pd.DataFrame(ledger_rows)
    else:
        # Empty trade
        trades = pd.DataFrame({
            "trigger_id": pd.Series([], dtype="str"),
            "pair_id": pd.Series([], dtype="str"),
            "entry_date": pd.Series([], dtype="datetime64[us]"),
            "exit_date": pd.Series([], dtype="datetime64[us]"),
            "exit_reason": pd.Series([], dtype="str"),
            "days_held": pd.Series([], dtype="int64"),
            "gross_ret": pd.Series([], dtype="float64"),
        })

    # Cost adjusted returns columns
    for c in cost_grid_bps:
        trades[f"net_ret_{c}bps"] = trades["gross_ret"] - 4 * c * 0.0001

    validate_artifact(trades, "trades")
    return trades


def daily_strategy_returns(trades: pd.DataFrame, prices: pd.DataFrame, triggers: pd.DataFrame, cost_bps: int = config.HEADLINE_COST_BPS) -> pd.Series:
    """One return per day: equal-weight mean of open trades' daily P&L, 0.0 when
    no trades open. costs hit as -2c bps on each trade's entry day and exit day."""
    returns = _simple_ret(prices)
    cost_per_transaction = 2 * cost_bps * 0.0001

    z_trigger_dict = {}
    for index, row in triggers.iterrows():
        z_trigger_dict[row["trigger_id"]] = row["z_trigger"]

    # each day's list of open-trade P&L values
    pnl_lists_dict = {}
    for date in prices.index:
        pnl_lists_dict[date] = []

    for index, trade in trades.iterrows():
        pair_id = trade["pair_id"]
        trigger_id = trade["trigger_id"]
        long_stock, short_stock = _split_stocks(pair_id, z_trigger_dict[trigger_id])
        entry_day = prices.index.get_loc(trade["entry_date"])
        exit_day = prices.index.get_loc(trade["exit_date"])

        # entry transaction cost
        entry_date = prices.index[entry_day]
        pnl_lists_dict[entry_date].append(-cost_per_transaction)
        if exit_day == entry_day:
            pnl_lists_dict[entry_date].append(-cost_per_transaction)
            continue

        # Every held day P&L
        curr_day = entry_day + 1
        while curr_day <= exit_day:
            date = prices.index[curr_day]
            pnl = returns.loc[date, long_stock] - returns.loc[date, short_stock]
            if curr_day == exit_day:
                # exit transaction cost
                pnl -= cost_per_transaction
            pnl_lists_dict[date].append(pnl)
            curr_day = curr_day + 1

    # Average P&L per day
    daily = []
    for date in prices.index:
        values = pnl_lists_dict[date]
        daily_total = 0.0
        if len(values):
            daily_total = sum(values) / len(values)

        daily.append(daily_total)

    return pd.Series(daily, index=prices.index)
